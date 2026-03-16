#include <cuda_runtime.h>
#include <cstddef>

/*
 * Range-restricted byte histogram optimized for large inputs on modern data-center GPUs
 * (e.g. A100/H100).
 *
 * Design choices:
 *   - blockThreads == 256:
 *       * exactly one thread per possible output bin (the range length is at most 256),
 *       * 8 warps/block is a good latency-hiding point for shared-memory atomics.
 *   - itemsPerThread == 16:
 *       * one aligned 128-bit uint4 load per thread on the fast path,
 *       * enough work per thread to amortize index arithmetic without driving register pressure up.
 *   - Shared-memory privatization:
 *       * each block accumulates into block-private histograms in shared memory,
 *       * only the final per-bin merge touches global memory.
 *   - Multiple shared-memory replicas per warp:
 *       * neighboring lanes are striped across several copies of the histogram,
 *         which reduces shared-memory atomic contention.
 *   - Odd replica stride:
 *       * successive replicas are separated by an odd number of 32-bit words,
 *         so the same logical bin in different replicas rotates across the 32 shared-memory banks,
 *         avoiding systematic bank conflicts between replicas.
 *
 * The public run_histogram() API zeros the output histogram and enqueues work on the current
 * (default) stream. It performs no synchronization because the caller explicitly handles that.
 */

constexpr int itemsPerThread   = 16;
constexpr int blockThreads     = 256;
constexpr int warpThreads      = 32;
constexpr int maxHistogramBins = 256;
constexpr int warpsPerBlock    = blockThreads / warpThreads;
constexpr int vectorBytes      = 16;  // sizeof(uint4)
constexpr int vectorsPerThread = itemsPerThread / vectorBytes;

static_assert(blockThreads == maxHistogramBins,
              "This implementation is tuned for exactly 256 threads per block.");
static_assert((blockThreads % warpThreads) == 0,
              "blockThreads must be a whole number of warps.");
static_assert(itemsPerThread > 0 && (itemsPerThread % vectorBytes) == 0,
              "itemsPerThread must be a positive multiple of 16 for the uint4 fast path.");

static __host__ __device__ __forceinline__ unsigned int private_hist_stride(const unsigned int numBins)
{
    // The next odd integer strictly larger than numBins.
    // Using an odd stride means the same logical bin in neighboring replicas is offset by
    // an odd number of 32-bit words, which is coprime with 32 banks and therefore rotates
    // that bin across banks instead of pinning it to the same bank in every replica.
    return numBins + 1u + (numBins & 1u);
}

static __device__ __forceinline__ void add_byte_to_private_hist(
    const unsigned int byteValue,
    const unsigned int fromValue,
    const unsigned int numBins,
    unsigned int* const myHist)
{
    // The histogram is relative to fromValue: bin 0 counts byte value 'fromValue'.
    // Unsigned subtraction naturally folds both range checks into one compare:
    // if byteValue < fromValue the subtraction underflows and the compare fails.
    const unsigned int bin = byteValue - fromValue;
    if (bin < numBins) {
        atomicAdd(myHist + bin, 1u);
    }
}

static __device__ __forceinline__ void add_word_to_private_hist(
    const unsigned int word,
    const unsigned int fromValue,
    const unsigned int numBins,
    unsigned int* const myHist)
{
    // One 32-bit word holds four bytes. CUDA devices are little-endian, so extracting bytes
    // by shifts yields the original character ordinals.
    add_byte_to_private_hist((word >>  0) & 0xFFu, fromValue, numBins, myHist);
    add_byte_to_private_hist((word >>  8) & 0xFFu, fromValue, numBins, myHist);
    add_byte_to_private_hist((word >> 16) & 0xFFu, fromValue, numBins, myHist);
    add_byte_to_private_hist((word >> 24) & 0xFFu, fromValue, numBins, myHist);
}

static __device__ __forceinline__ void add_uint4_to_private_hist(
    const uint4 packed,
    const unsigned int fromValue,
    const unsigned int numBins,
    unsigned int* const myHist)
{
    // A uint4 is a single 128-bit load containing 16 input bytes.
    add_word_to_private_hist(packed.x, fromValue, numBins, myHist);
    add_word_to_private_hist(packed.y, fromValue, numBins, myHist);
    add_word_to_private_hist(packed.z, fromValue, numBins, myHist);
    add_word_to_private_hist(packed.w, fromValue, numBins, myHist);
}

template <int COPIES_PER_WARP>
__launch_bounds__(blockThreads, 4)
__global__ void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int fromValue,
    unsigned int numBins)
{
    static_assert(COPIES_PER_WARP >= 1 && COPIES_PER_WARP <= warpThreads,
                  "Invalid number of histogram copies per warp.");
    static_assert((COPIES_PER_WARP & (COPIES_PER_WARP - 1)) == 0,
                  "COPIES_PER_WARP must be a power of two.");

    constexpr int copiesPerBlock = warpsPerBlock * COPIES_PER_WARP;

    // Dynamic shared memory layout:
    //   [replica 0 bins ... pad][replica 1 bins ... pad]...[replica N bins ... pad]
    extern __shared__ unsigned int privateHist[];

    const unsigned int stride = private_hist_stride(numBins);
    const unsigned int totalPrivateBins = stride * static_cast<unsigned int>(copiesPerBlock);

    // Initialize all private histograms. The total private storage is intentionally kept modest
    // (roughly a few tens of KiB at the top of each bucket) so this setup cost is amortized well
    // for large inputs.
    for (unsigned int i = threadIdx.x; i < totalPrivateBins; i += blockThreads) {
        privateHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & (warpThreads - 1u);
    const unsigned int warp = threadIdx.x >> 5;

    // Round-robin striping of lanes across replicas:
    //   COPIES_PER_WARP == 32 -> 1 lane/replica
    //   COPIES_PER_WARP == 16 -> 2 lanes/replica
    //   COPIES_PER_WARP ==  8 -> 4 lanes/replica
    //   COPIES_PER_WARP ==  4 -> 8 lanes/replica
    // This spreads neighboring lanes over different replicas instead of clustering them.
    const unsigned int copyInWarp = lane & static_cast<unsigned int>(COPIES_PER_WARP - 1);
    const unsigned int replica    = warp * static_cast<unsigned int>(COPIES_PER_WARP) + copyInWarp;
    unsigned int* const myHist    = privateHist + replica * stride;

    const std::size_t n = static_cast<std::size_t>(inputSize);

    // The input buffer is specified to be allocated with cudaMalloc, so the base pointer is
    // naturally aligned. Keeping itemsPerThread a multiple of 16 preserves alignment for this path.
    const uint4* const inputVec4 = reinterpret_cast<const uint4*>(input);

    const std::size_t firstIndex =
        (static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockThreads) +
         static_cast<std::size_t>(threadIdx.x)) *
        static_cast<std::size_t>(itemsPerThread);

    const std::size_t gridStride =
        static_cast<std::size_t>(gridDim.x) *
        static_cast<std::size_t>(blockThreads) *
        static_cast<std::size_t>(itemsPerThread);

    std::size_t idx = firstIndex;

    // Fast path: each thread processes itemsPerThread bytes at a time via one or more uint4 loads.
    for (; idx + static_cast<std::size_t>(itemsPerThread) <= n; idx += gridStride) {
        const std::size_t vecBase = idx / static_cast<std::size_t>(vectorBytes);

        #pragma unroll
        for (int v = 0; v < vectorsPerThread; ++v) {
            const uint4 packed = inputVec4[vecBase + static_cast<std::size_t>(v)];
            add_uint4_to_private_hist(packed, fromValue, numBins, myHist);
        }
    }

    // Tail path: handle the last partial chunk, if any.
    if (idx < n) {
        const std::size_t tailEnd =
            ((idx + static_cast<std::size_t>(itemsPerThread)) < n)
                ? (idx + static_cast<std::size_t>(itemsPerThread))
                : n;

        for (std::size_t j = idx; j < tailEnd; ++j) {
            add_byte_to_private_hist(static_cast<unsigned int>(input[j]), fromValue, numBins, myHist);
        }
    }

    __syncthreads();

    // numBins <= 256, so one thread per logical output bin is enough for the block-local reduction.
    if (threadIdx.x < numBins) {
        const unsigned int bin = threadIdx.x;

        // Sum this logical bin across all replicas. Pointer bumping avoids repeated multiplies.
        const unsigned int* copyPtr = privateHist + bin;
        unsigned int sum = 0u;

        #pragma unroll 8
        for (int copy = 0; copy < copiesPerBlock; ++copy, copyPtr += stride) {
            sum += *copyPtr;
        }

        if (sum != 0u) {
            // For a single-block launch the output bins are disjoint, so a plain store is enough.
            // Otherwise merge block-private totals with one global atomic per bin.
            if (gridDim.x == 1) {
                histogram[bin] = sum;
            } else {
                atomicAdd(histogram + bin, sum);
            }
        }
    }
}

template <int COPIES_PER_WARP>
static inline void launch_histogram_kernel(
    const unsigned char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    unsigned int fromValue,
    unsigned int numBins)
{
    constexpr int copiesPerBlock = warpsPerBlock * COPIES_PER_WARP;

    const std::size_t sharedBytes =
        static_cast<std::size_t>(copiesPerBlock) *
        static_cast<std::size_t>(private_hist_stride(numBins)) *
        sizeof(unsigned int);

    // The kernel is shared-memory/atomic heavy and streams the input once, so a maximal
    // shared-memory carveout is preferable on Ampere/Hopper-class parts.
    (void)cudaFuncSetAttribute(
        histogram_range_kernel<COPIES_PER_WARP>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    int activeBlocksPerSM = 1;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<COPIES_PER_WARP>,
        blockThreads,
        sharedBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    int device = 0;
    (void)cudaGetDevice(&device);

    int smCount = 1;
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    // With grid-stride looping, launching only enough blocks to saturate the machine is ideal:
    // more blocks would only increase the final global merge traffic.
    constexpr unsigned int tileSize = static_cast<unsigned int>(blockThreads * itemsPerThread);
    const unsigned int requiredBlocks = 1u + (inputSize - 1u) / tileSize;

    unsigned int gridBlocks =
        static_cast<unsigned int>(activeBlocksPerSM) * static_cast<unsigned int>(smCount);
    if (gridBlocks < 1u) {
        gridBlocks = 1u;
    }
    if (gridBlocks > requiredBlocks) {
        gridBlocks = requiredBlocks;
    }

    histogram_range_kernel<COPIES_PER_WARP>
        <<<gridBlocks, blockThreads, sharedBytes>>>(input, histogram, inputSize, fromValue, numBins);
}

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // Assumes the problem-statement preconditions:
    //   0 <= from < to <= 255
    const unsigned int fromValue = static_cast<unsigned int>(from);
    const unsigned int numBins   = static_cast<unsigned int>(to - from + 1);

    // The API computes a fresh histogram, not an accumulation, so zero the destination here.
    (void)cudaMemsetAsync(
        histogram,
        0,
        static_cast<std::size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    // Reinterpret as unsigned bytes so ordinals 128..255 are handled correctly even if host-side
    // char is signed.
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // Adaptive replica count:
    //   <=  32 bins -> 32 copies/warp (1 lane per copy)
    //   <=  64 bins -> 16 copies/warp (2 lanes per copy)
    //   <= 128 bins ->  8 copies/warp (4 lanes per copy)
    //   else        ->  4 copies/warp (8 lanes per copy)
    //
    // These buckets are tuned for 256-thread blocks and keep the shared-memory footprint near
    // a few tens of KiB at the top of each bucket while giving smaller ranges more replicas.
    if (numBins <= 32u) {
        launch_histogram_kernel<32>(inputBytes, histogram, inputSize, fromValue, numBins);
    } else if (numBins <= 64u) {
        launch_histogram_kernel<16>(inputBytes, histogram, inputSize, fromValue, numBins);
    } else if (numBins <= 128u) {
        launch_histogram_kernel<8>(inputBytes, histogram, inputSize, fromValue, numBins);
    } else {
        launch_histogram_kernel<4>(inputBytes, histogram, inputSize, fromValue, numBins);
    }
}