#include <cuda_runtime.h>

namespace
{
    // Tuned for large streaming inputs on modern data-center GPUs (A100/H100 class).
    // - 256 threads/block -> 8 warps/block, a good balance for histogramming.
    // - The logical histogram size is at most 256 bins, so the final flush can map one thread
    //   to one logical bin.
    // - 16 items/thread gives each block a 4 KiB tile, which amortizes control overhead well
    //   without pushing register pressure too high.
    constexpr int blockThreads   = 256;
    constexpr int itemsPerThread = 16;
    constexpr int warpsPerBlock  = blockThreads / 32;

    constexpr unsigned int fullWarpMask = 0xFFFFFFFFu;
    constexpr unsigned int invalidBin   = 0xFFFFFFFFu;

    static_assert(blockThreads % 32 == 0, "blockThreads must be a multiple of warp size");
    static_assert(itemsPerThread > 0, "itemsPerThread must be positive");

    // Shared memory has 32 banks. Padding one slot after every completed 32-bin group
    // changes the bank mapping so bins k and k+32 do not alias the same bank.
    // Example: 256 logical bins -> 263 physical counters.
    constexpr unsigned int padded_range_bins(unsigned int rangeBins)
    {
        return rangeBins ? (rangeBins + ((rangeBins - 1u) >> 5)) : 0u;
    }

    __device__ __forceinline__ unsigned int padded_bin_index(unsigned int bin)
    {
        return bin + (bin >> 5);
    }

    // Canonicalized input values use:
    //   - [0, rangeBins) for valid bins
    //   - invalidBin for both out-of-range characters and tail elements past inputSize
    //
    // __match_any_sync groups equal bins inside the warp; only the leader performs the update.
    // Because each warp owns a private histogram slice in shared memory, this is a plain
    // shared-memory increment instead of a shared-memory atomic.
    //
    // This implementation intentionally targets Volta+ (__match_any_sync), which matches the
    // requested hardware class (A100/H100).
    __device__ __forceinline__ void accumulate_bin(unsigned int bin,
                                                   unsigned int* warpHist,
                                                   unsigned int lane)
    {
        const unsigned int peers = __match_any_sync(fullWarpMask, bin);

        if (bin != invalidBin)
        {
            const unsigned int leader = static_cast<unsigned int>(__ffs(peers) - 1);
            if (lane == leader)
            {
                warpHist[padded_bin_index(bin)] += static_cast<unsigned int>(__popc(peers));
            }
        }
    }

    __global__ __launch_bounds__(blockThreads)
    void histogram_range_kernel(const unsigned char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                unsigned int from,
                                unsigned int rangeBins,
                                unsigned int paddedRangeBins)
    {
        // Layout: one privatized histogram per warp.
        extern __shared__ unsigned int sharedWarpHist[];

        const unsigned int tid    = static_cast<unsigned int>(threadIdx.x);
        const unsigned int lane   = tid & 31u;
        const unsigned int warpId = tid >> 5;

        unsigned int* const warpHist = sharedWarpHist + warpId * paddedRangeBins;

        // Zero the privatized histograms.
        const unsigned int totalSharedCounters = static_cast<unsigned int>(warpsPerBlock) * paddedRangeBins;
        for (unsigned int i = tid; i < totalSharedCounters; i += blockThreads)
        {
            sharedWarpHist[i] = 0u;
        }
        __syncthreads();

        // Persistent/grid-stride processing over block-sized tiles.
        const size_t n              = static_cast<size_t>(inputSize);
        const size_t blockWorkItems = static_cast<size_t>(blockThreads) * static_cast<size_t>(itemsPerThread);
        const size_t gridStride     = static_cast<size_t>(gridDim.x) * blockWorkItems;
        const size_t fullTilesEnd   = (n / blockWorkItems) * blockWorkItems;

        for (size_t tileBase = static_cast<size_t>(blockIdx.x) * blockWorkItems; tileBase < n; tileBase += gridStride)
        {
            const size_t threadBase = tileBase + static_cast<size_t>(tid);

            if (tileBase < fullTilesEnd)
            {
                // Fast path: the whole block tile is in-bounds, so no per-item bounds checks.
                #pragma unroll
                for (int i = 0; i < itemsPerThread; ++i)
                {
                    const size_t idx = threadBase + static_cast<size_t>(i) * blockThreads;
                    const unsigned int offset =
                        static_cast<unsigned int>(input[idx]) - from;
                    const unsigned int bin = (offset < rangeBins) ? offset : invalidBin;
                    accumulate_bin(bin, warpHist, lane);
                }
            }
            else
            {
                // Tail path: keep all warp threads participating in every __match_any_sync call.
                #pragma unroll
                for (int i = 0; i < itemsPerThread; ++i)
                {
                    const size_t idx = threadBase + static_cast<size_t>(i) * blockThreads;

                    unsigned int bin = invalidBin;
                    if (idx < n)
                    {
                        const unsigned int offset =
                            static_cast<unsigned int>(input[idx]) - from;
                        bin = (offset < rangeBins) ? offset : invalidBin;
                    }

                    accumulate_bin(bin, warpHist, lane);
                }
            }
        }

        __syncthreads();

        // Flush warp-private histograms into the final output.
        // For a single-block launch we can store directly; for multi-block launches we use
        // global atomics because different blocks contribute to the same output bins.
        if (tid < rangeBins)
        {
            const unsigned int pbin = padded_bin_index(tid);

            unsigned int sum = 0u;
            #pragma unroll
            for (int w = 0; w < warpsPerBlock; ++w)
            {
                sum += sharedWarpHist[static_cast<unsigned int>(w) * paddedRangeBins + pbin];
            }

            if (gridDim.x == 1)
            {
                histogram[tid] = sum;
            }
            else if (sum != 0u)
            {
                atomicAdd(histogram + tid, sum);
            }
        }
    }
}

// Enqueues the histogram computation on the default stream.
// The caller is responsible for any host/device synchronization.
// For multi-block launches the output buffer must be zeroed first because blocks accumulate
// via global atomics. For a single-block launch the kernel writes the final histogram directly,
// so the memset can be skipped.
void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The problem statement guarantees a valid range; this guard simply avoids undefined behavior
    // if the function is called incorrectly.
    if (from < 0 || to > 255 || from > to)
    {
        return;
    }

    const unsigned int fromU           = static_cast<unsigned int>(from);
    const unsigned int rangeBins       = static_cast<unsigned int>(to - from + 1);
    const unsigned int paddedRangeBins = padded_range_bins(rangeBins);
    const size_t histogramBytes        = static_cast<size_t>(rangeBins) * sizeof(unsigned int);
    const size_t sharedBytes           = static_cast<size_t>(warpsPerBlock) *
                                         static_cast<size_t>(paddedRangeBins) *
                                         sizeof(unsigned int);

    // No input: the correct result is an all-zero histogram.
    if (inputSize == 0u)
    {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
        return;
    }

    const unsigned long long blockWorkItems =
        static_cast<unsigned long long>(blockThreads) * static_cast<unsigned long long>(itemsPerThread);

    const unsigned int workBlocks = static_cast<unsigned int>(
        (static_cast<unsigned long long>(inputSize) + blockWorkItems - 1ull) / blockWorkItems);

    // Launch only a resident wave of blocks. Because the kernel uses a grid-stride loop, this
    // keeps the machine full while minimizing per-block shared-histogram init/flush overhead.
    unsigned int gridBlocks = workBlocks;

    if (gridBlocks > 1u)
    {
        int device = 0;
        int smCount = 1;
        int maxBlocksPerSM = 1;

        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            histogram_range_kernel,
            blockThreads,
            sharedBytes);

        if (smCount < 1)       smCount = 1;
        if (maxBlocksPerSM < 1) maxBlocksPerSM = 1;

        const unsigned int residentBlocks =
            static_cast<unsigned int>(smCount) * static_cast<unsigned int>(maxBlocksPerSM);

        if (gridBlocks > residentBlocks)
        {
            gridBlocks = residentBlocks;
        }
    }

    if (gridBlocks > 1u)
    {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
    }

    histogram_range_kernel<<<gridBlocks, blockThreads, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        fromU,
        rangeBins,
        paddedRangeBins);
}