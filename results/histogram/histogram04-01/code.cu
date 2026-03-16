#include <cuda_runtime.h>
#include <cstddef>

namespace {

// A 256-thread block is a good fit for modern data-center GPUs and for this
// problem specifically: the histogram range is at most 256 bins, so the final
// flush can map one thread to one bin.
constexpr int blockThreads   = 256;
constexpr int warpThreads    = 32;
constexpr int warpsPerBlock  = blockThreads / warpThreads;

// Tuned default for A100/H100-class GPUs:
// - 16 bytes per thread gives one naturally aligned 128-bit load (uint4) per
//   thread in the fast path.
// - This amortizes loop/index overhead well without materially hurting occupancy.
// Keep this a multiple of 16 to preserve the aligned vectorized load path.
constexpr int itemsPerThread = 16;

constexpr unsigned int invalidBinKey = 0xFFFFFFFFu;

static_assert((blockThreads % warpThreads) == 0, "blockThreads must be a multiple of 32.");
static_assert(blockThreads >= 256, "blockThreads must be at least 256 to cover the maximum 256-bin range.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert((itemsPerThread % 16) == 0, "itemsPerThread must be a multiple of 16.");

// Warp-aggregated increment into a warp-private shared histogram.
// Each block owns warpsPerBlock histograms in shared memory, one per warp.
// __match_any_sync groups lanes that saw the same bin; only the leader for each
// distinct bin performs the increment, adding the population count of the group.
// Because the histogram is private to the warp, no shared-memory atomic is needed
// on the hot path.
__device__ __forceinline__
void add_byte_to_warp_hist(unsigned int* warpHist,
                           unsigned int rangeBins,
                           unsigned int fromValue,
                           unsigned int byteValue,
                           bool valid,
                           unsigned int activeMask,
                           unsigned int lane)
{
    const unsigned int bin = byteValue - fromValue;
    const bool counted = valid && (bin < rangeBins);
    const unsigned int key = counted ? bin : invalidBinKey;

    const unsigned int peers = __match_any_sync(activeMask, key);

    if (counted) {
        const unsigned int leader = static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1);
        if (lane == leader) {
            warpHist[bin] += __popc(peers);
        }
    }
}

// Process four bytes packed in one 32-bit word. The histogram is order-insensitive,
// so the byte order of updates is irrelevant as long as every byte is visited once.
__device__ __forceinline__
void add_packed_word_to_warp_hist(unsigned int packed,
                                  unsigned int* warpHist,
                                  unsigned int rangeBins,
                                  unsigned int fromValue,
                                  unsigned int activeMask,
                                  unsigned int lane)
{
    add_byte_to_warp_hist(warpHist, rangeBins, fromValue,  packed        & 0xFFu, true, activeMask, lane);
    add_byte_to_warp_hist(warpHist, rangeBins, fromValue, (packed >>  8) & 0xFFu, true, activeMask, lane);
    add_byte_to_warp_hist(warpHist, rangeBins, fromValue, (packed >> 16) & 0xFFu, true, activeMask, lane);
    add_byte_to_warp_hist(warpHist, rangeBins, fromValue, (packed >> 24) & 0xFFu, true, activeMask, lane);
}

__global__ __launch_bounds__(blockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int fromValue,
                            unsigned int rangeBins)
{
    constexpr int vectorLoadsPerThread = itemsPerThread / 16;

    // Shared memory layout:
    //   [warp0 histogram][warp1 histogram]...[warpN histogram]
    // Range width is <= 256, so even 8 warp-private histograms only require
    // 8 * 256 * 4 bytes = 8 KiB at the largest range.
    extern __shared__ unsigned int sharedHist[];

    const unsigned int tid    = static_cast<unsigned int>(threadIdx.x);
    const unsigned int lane   = tid & (warpThreads - 1u);
    const unsigned int warpId = tid >> 5;

    const unsigned int totalSharedBins = static_cast<unsigned int>(warpsPerBlock) * rangeBins;
    for (unsigned int i = tid; i < totalSharedBins; i += blockThreads) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Interpret input as bytes 0..255 regardless of whether plain char is signed.
    const unsigned char* const data = reinterpret_cast<const unsigned char*>(input);
    unsigned int* const warpHist = sharedHist + warpId * rangeBins;

    const size_t inputSizeBytes = static_cast<size_t>(inputSize);
    const size_t tileItems      = static_cast<size_t>(blockThreads) * itemsPerThread;
    const size_t gridStride     = static_cast<size_t>(gridDim.x) * tileItems;

    for (size_t tileBase = static_cast<size_t>(blockIdx.x) * tileItems;
         tileBase < inputSizeBytes;
         tileBase += gridStride) {
        const size_t threadBase = tileBase + static_cast<size_t>(tid) * itemsPerThread;

        // Fast path: this lane owns a full, aligned chunk, so use vectorized loads.
        // __activemask() makes the warp intrinsics safe even in the final mixed warp,
        // where some lanes may take the fast path while others fall back to scalar.
        if ((threadBase + itemsPerThread) <= inputSizeBytes) {
            const unsigned int branchMask = __activemask();
            const uint4* const vecPtr = reinterpret_cast<const uint4*>(data + threadBase);

#pragma unroll
            for (int v = 0; v < vectorLoadsPerThread; ++v) {
                const uint4 packed16 = vecPtr[v];
                add_packed_word_to_warp_hist(packed16.x, warpHist, rangeBins, fromValue, branchMask, lane);
                add_packed_word_to_warp_hist(packed16.y, warpHist, rangeBins, fromValue, branchMask, lane);
                add_packed_word_to_warp_hist(packed16.z, warpHist, rangeBins, fromValue, branchMask, lane);
                add_packed_word_to_warp_hist(packed16.w, warpHist, rangeBins, fromValue, branchMask, lane);
            }
        } else {
            // Tail path for the final partial chunk.
            const unsigned int branchMask = __activemask();

#pragma unroll
            for (int i = 0; i < itemsPerThread; ++i) {
                const size_t idx = threadBase + static_cast<size_t>(i);
                const bool valid = idx < inputSizeBytes;
                const unsigned int activeMask = __ballot_sync(branchMask, valid);

                if (activeMask != 0u) {
                    unsigned int byteValue = 0u;
                    if (valid) {
                        byteValue = static_cast<unsigned int>(data[idx]);
                    }
                    add_byte_to_warp_hist(warpHist, rangeBins, fromValue, byteValue, valid, activeMask, lane);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the warp-private histograms into the caller-provided global histogram.
    // blockThreads == 256 and rangeBins <= 256, so each thread handles at most one bin.
    if (tid < rangeBins) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += sharedHist[static_cast<unsigned int>(w) * rangeBins + tid];
        }
        if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

} // anonymous namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int fromValue = static_cast<unsigned int>(from);
    const unsigned int rangeBins = static_cast<unsigned int>(to - from) + 1u;
    const size_t histogramBytes  = static_cast<size_t>(rangeBins) * sizeof(unsigned int);

    // The result buffer must start from zero. This stays asynchronous with respect
    // to the host; the caller owns any host/device synchronization.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedMemBytes =
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(rangeBins) * sizeof(unsigned int);

    // Launch roughly one resident wave of blocks. Because each block grid-strides
    // over the full input and pays fixed costs to clear and flush its private
    // histogram, more blocks than the GPU can keep resident usually only increase
    // overhead for this workload.
    int device = 0;
    int smCount = 1;
    int blocksPerSM = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        histogram_range_kernel,
        blockThreads,
        sharedMemBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (blocksPerSM < 1) {
        blocksPerSM = 1;
    }

    const size_t tileItems = static_cast<size_t>(blockThreads) * itemsPerThread;
    const unsigned int numTiles =
        static_cast<unsigned int>((static_cast<size_t>(inputSize) + tileItems - 1u) / tileItems);
    const unsigned int residentGrid = static_cast<unsigned int>(smCount * blocksPerSM);

    unsigned int launchGrid = (numTiles < residentGrid) ? numTiles : residentGrid;
    if (launchGrid == 0u) {
        launchGrid = 1u;
    }

    histogram_range_kernel<<<launchGrid, blockThreads, sharedMemBytes>>>(
        input,
        histogram,
        inputSize,
        fromValue,
        rangeBins);
}