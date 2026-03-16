#include <cuda_runtime.h>

namespace histogram_detail {

// Tuned defaults for large inputs on recent datacenter GPUs (A100/H100 class):
// - 256 threads/block is a strong general-purpose sweet spot for this workload.
// - 16 chars/thread gives enough work to amortize shared-histogram setup/flush
//   while keeping register pressure and occupancy healthy.
constexpr int itemsPerThread = 16;
constexpr int blockSize = 256;
constexpr int warpSizeConst = 32;
constexpr int warpsPerBlock = blockSize / warpSizeConst;
constexpr int maxBins = 256;

// The range width is at most 256, so a fixed 256-bin private histogram per warp
// costs only 8 KiB/block with 8 warps/block. On modern datacenter GPUs this is
// small enough that occupancy is still thread-limited, while fixed-size storage
// keeps addressing simple and compile-time constant.
constexpr int sharedHistSize = warpsPerBlock * maxBins;
constexpr int sharedInitIterations = sharedHistSize / blockSize;

constexpr unsigned int blockItems =
    static_cast<unsigned int>(blockSize) * static_cast<unsigned int>(itemsPerThread);
constexpr unsigned int fullWarpMask = 0xFFFFFFFFu;
constexpr unsigned int invalidBinKey = 0xFFFFFFFFu;

static_assert(blockSize % warpSizeConst == 0, "blockSize must be a multiple of warp size");
static_assert(sharedHistSize % blockSize == 0, "shared histogram must divide evenly across threads");

inline unsigned int div_up_u32(unsigned int n, unsigned int d) {
    return (n / d) + static_cast<unsigned int>((n % d) != 0u);
}

// Warp-aggregated shared-memory update.
// Every lane participates in __match_any_sync using either its real bin or a sentinel
// for "invalid / out-of-range". For each distinct valid bin in the warp, exactly one
// lane performs the shared-memory atomicAdd with the number of matching lanes.
//
// This reduces atomic traffic substantially on text, where repeated characters such as
// spaces, vowels, punctuation, or common letters frequently appear within a warp.
__device__ __forceinline__ void warp_aggregate_add(
    const unsigned int bin,
    const bool valid,
    const unsigned int lane,
    unsigned int* const warpHist)
{
    const unsigned int key = valid ? bin : invalidBinKey;
    const unsigned int peers = __match_any_sync(fullWarpMask, key);

    if (valid && lane == static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1)) {
        atomicAdd(warpHist + bin, static_cast<unsigned int>(__popc(peers)));
    }
}

__global__ __launch_bounds__(blockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from_u,
                            unsigned int bins)
{
    // One private histogram per warp in shared memory.
    __shared__ unsigned int sharedHist[sharedHistSize];

    const unsigned int thread = static_cast<unsigned int>(threadIdx.x);
    const unsigned int warpId = thread >> 5;
    const unsigned int lane = thread & 31u;
    unsigned int* const warpHist = sharedHist + warpId * maxBins;

    // Zero the entire privatized histogram. With the fixed layout this is exactly
    // 8 stores/thread and the compiler can fully unroll it.
    #pragma unroll
    for (int i = 0; i < sharedInitIterations; ++i) {
        sharedHist[thread + static_cast<unsigned int>(i) * static_cast<unsigned int>(blockSize)] = 0u;
    }
    __syncthreads();

    // Grid-stride over tiles of blockSize * itemsPerThread bytes.
    // Inside each tile, threads use a block-strided index pattern:
    //   idx = blockBase + thread + item * blockSize
    // so each item iteration produces perfectly coalesced byte loads.
    const size_t inputSize64 = static_cast<size_t>(inputSize);
    const size_t blockSpan = static_cast<size_t>(blockItems);
    const size_t gridSpan = static_cast<size_t>(gridDim.x) * blockSpan;

    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * blockSpan;
         blockBase < inputSize64;
         blockBase += gridSpan)
    {
        size_t idx = blockBase + thread;

        // Split the common full-tile path from the final partial-tile path so almost
        // all iterations avoid per-item bounds checks when the input is large.
        const bool fullTile = (inputSize64 - blockBase) >= blockSpan;

        if (fullTile) {
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                // Interpret bytes as unsigned so values 128..255 are handled correctly
                // regardless of whether plain `char` is signed by the host compiler.
                const unsigned int c = static_cast<unsigned char>(input[idx]);
                const unsigned int bin = c - from_u;
                warp_aggregate_add(bin, bin < bins, lane, warpHist);
                idx += static_cast<size_t>(blockSize);
            }
        } else {
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                unsigned int bin = 0u;
                const bool inBounds = idx < inputSize64;

                if (inBounds) {
                    const unsigned int c = static_cast<unsigned char>(input[idx]);
                    bin = c - from_u;
                }

                warp_aggregate_add(bin, inBounds && (bin < bins), lane, warpHist);
                idx += static_cast<size_t>(blockSize);
            }
        }
    }

    __syncthreads();

    // Reduce the warp-private histograms into the global histogram.
    // Since bins <= 256 and blockSize == 256, one thread can own one output bin.
    if (thread < bins) {
        unsigned int sum = sharedHist[thread];

        #pragma unroll
        for (int w = 1; w < warpsPerBlock; ++w) {
            sum += sharedHist[static_cast<unsigned int>(w) * static_cast<unsigned int>(maxBins) + thread];
        }

        if (sum != 0u) {
            atomicAdd(histogram + thread, sum);
        }
    }
}

}  // namespace histogram_detail

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Lightweight asynchronous entry point:
    // - pointers are assumed to reference device memory allocated by cudaMalloc,
    // - caller is responsible for any required synchronization and error checking.
    if (histogram == nullptr) {
        return;
    }

    // The problem statement guarantees valid inputs, but keeping a minimal guard here
    // avoids undefined behavior if the function is called incorrectly.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int bins = static_cast<unsigned int>(to - from + 1);

    // The kernel accumulates with atomics, so start from a zeroed output histogram.
    (void)cudaMemsetAsync(
        histogram,
        0,
        static_cast<size_t>(bins) * sizeof(unsigned int),
        0);

    if (input == nullptr || inputSize == 0u) {
        return;
    }

    // Launch about one resident wave of blocks. Because the kernel uses a grid-stride
    // loop, this is enough to saturate the GPU while minimizing the number of final
    // global atomic reductions (one per bin per block).
    int device = 0;
    int smCount = 1;
    int residentBlocksPerSM = 1;

    if (cudaSuccess == cudaGetDevice(&device)) {
        (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    }

    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &residentBlocksPerSM,
        histogram_detail::histogram_range_kernel,
        histogram_detail::blockSize,
        0);

    if (smCount < 1) {
        smCount = 1;
    }
    if (residentBlocksPerSM < 1) {
        residentBlocksPerSM = 1;
    }

    const unsigned int neededBlocks =
        histogram_detail::div_up_u32(inputSize, histogram_detail::blockItems);

    unsigned int gridBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(residentBlocksPerSM);

    if (gridBlocks > neededBlocks) {
        gridBlocks = neededBlocks;
    }
    if (gridBlocks == 0u) {
        gridBlocks = 1u;
    }

    histogram_detail::histogram_range_kernel<<<gridBlocks, histogram_detail::blockSize>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        bins);
}