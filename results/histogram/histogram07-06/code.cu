#include <cuda_runtime.h>

namespace {

// 32 privatized histograms per block, one per warp lane / shared-memory bank.
constexpr int histogramCopies = 32;

// 256 threads is a good balance on A100/H100-class GPUs: enough occupancy for
// the 32 KB worst-case shared histogram while not creating excessive shared-atomic
// contention inside a block.
constexpr int blockSize = 256;

// Large-input default tuned for modern GPUs. 16 bytes/thread amortizes the
// per-block shared-memory init/final merge well, keeps register pressure modest,
// and lines up naturally with four 32-bit vectorized loads per thread.
constexpr int itemsPerThread = 16;

// The hot loop uses 32-bit vectorized loads from the cudaMalloc-provided input.
static_assert(sizeof(unsigned int) == 4, "This code assumes 32-bit unsigned int.");
constexpr int bytesPerVectorLoad = 4;
constexpr int vectorLoadsPerThread = itemsPerThread / bytesPerVectorLoad;
constexpr int warpsPerBlock = blockSize / histogramCopies;

static_assert(histogramCopies == 32, "This kernel is written for 32 shared histogram copies.");
static_assert(blockSize % histogramCopies == 0, "blockSize must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(itemsPerThread % bytesPerVectorLoad == 0,
              "itemsPerThread must stay a multiple of 4 to preserve vectorized loads.");

template <bool FullRange>
__device__ __forceinline__ void accumulate_byte(unsigned int value,
                                                unsigned int fromValue,
                                                unsigned int range,
                                                unsigned int lane,
                                                unsigned int *sharedHistogram);

template <>
__device__ __forceinline__ void accumulate_byte<true>(unsigned int value,
                                                      unsigned int /*fromValue*/,
                                                      unsigned int /*range*/,
                                                      unsigned int lane,
                                                      unsigned int *sharedHistogram) {
    atomicAdd(sharedHistogram + value * histogramCopies + lane, 1u);
}

template <>
__device__ __forceinline__ void accumulate_byte<false>(unsigned int value,
                                                       unsigned int fromValue,
                                                       unsigned int range,
                                                       unsigned int lane,
                                                       unsigned int *sharedHistogram) {
    // Unsigned subtraction turns "value < fromValue" into a large number, so the
    // single comparison below rejects both out-of-range cases.
    const unsigned int bin = value - fromValue;
    if (bin < range) {
        atomicAdd(sharedHistogram + bin * histogramCopies + lane, 1u);
    }
}

template <bool FullRange>
__device__ __forceinline__ void accumulate_word(unsigned int packed,
                                                unsigned int fromValue,
                                                unsigned int range,
                                                unsigned int lane,
                                                unsigned int *sharedHistogram) {
    accumulate_byte<FullRange>( packed        & 0xFFu, fromValue, range, lane, sharedHistogram);
    accumulate_byte<FullRange>((packed >>  8) & 0xFFu, fromValue, range, lane, sharedHistogram);
    accumulate_byte<FullRange>((packed >> 16) & 0xFFu, fromValue, range, lane, sharedHistogram);
    accumulate_byte<FullRange>( packed >> 24,          fromValue, range, lane, sharedHistogram);
}

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
#pragma unroll
    for (int offset = histogramCopies / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

// Histogram kernel overview:
// 1. Each block allocates 32 privatized histograms in shared memory.
// 2. Copy c of bin i lives at sharedHistogram[i * 32 + c].
//    Because bank index is (address_in_words % 32), lane c always lands in bank c,
//    so a warp updating the same bin is free of shared-memory bank conflicts.
// 3. Threads update their lane's copy with shared-memory atomics.
// 4. At the end, one warp reduces the 32 copies of each bin via warp shuffles and
//    merges the per-block total into the global histogram.
template <bool FullRange>
__global__ __launch_bounds__(blockSize)
void histogram_kernel(const char * __restrict__ input,
                      unsigned int * __restrict__ histogram,
                      unsigned int inputSize,
                      unsigned int fromValue,
                      unsigned int range) {
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;

    const unsigned int sharedEntryCount = range * histogramCopies;
    for (unsigned int i = threadIdx.x; i < sharedEntryCount; i += blockSize) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    // The problem statement says the input buffer is cudaMalloc'ed, so this fast path
    // uses aligned 32-bit loads. If callers ever pass sub-slices, keeping them 4-byte
    // aligned preserves this path's intended behavior/performance.
    const unsigned char * __restrict__ bytes =
        reinterpret_cast<const unsigned char *>(input);
    const unsigned int * __restrict__ words =
        reinterpret_cast<const unsigned int *>(input);

    // Each block iteration processes blockSize * itemsPerThread bytes.
    // For each vector-load slot i, a warp reads 32 consecutive 32-bit words
    // (128 bytes), which is perfectly coalesced.
    const size_t wordCount = static_cast<size_t>(inputSize) / bytesPerVectorLoad;
    const size_t wordTileStride = static_cast<size_t>(blockSize) * vectorLoadsPerThread;
    const size_t gridWordStride = static_cast<size_t>(gridDim.x) * wordTileStride;
    const size_t fullWordTileLimit =
        (wordCount > static_cast<size_t>(blockSize) * (vectorLoadsPerThread - 1))
            ? (wordCount - static_cast<size_t>(blockSize) * (vectorLoadsPerThread - 1))
            : size_t{0};

    size_t wordBase = static_cast<size_t>(blockIdx.x) * wordTileStride + threadIdx.x;

    // Main path: all vector loads are in bounds, so the inner loop is branch-free.
    for (; wordBase < fullWordTileLimit; wordBase += gridWordStride) {
#pragma unroll
        for (int i = 0; i < vectorLoadsPerThread; ++i) {
            const unsigned int packed = words[wordBase + static_cast<size_t>(i) * blockSize];
            accumulate_word<FullRange>(packed, fromValue, range, lane, sharedHistogram);
        }
    }

    // Tail path for the final partial tile of 32-bit words.
    for (; wordBase < wordCount; wordBase += gridWordStride) {
#pragma unroll
        for (int i = 0; i < vectorLoadsPerThread; ++i) {
            const size_t wordIndex = wordBase + static_cast<size_t>(i) * blockSize;
            if (wordIndex < wordCount) {
                const unsigned int packed = words[wordIndex];
                accumulate_word<FullRange>(packed, fromValue, range, lane, sharedHistogram);
            }
        }
    }

    // Handle the final 0..3 bytes that do not form a full 32-bit word.
    // Only block 0 touches this suffix so the bytes are counted exactly once.
    const size_t tailByteBase = wordCount * bytesPerVectorLoad;
    if (blockIdx.x == 0u) {
        const size_t tailByteCount = static_cast<size_t>(inputSize) - tailByteBase;
        if (static_cast<size_t>(threadIdx.x) < tailByteCount) {
            accumulate_byte<FullRange>(
                static_cast<unsigned int>(bytes[tailByteBase + threadIdx.x]),
                fromValue,
                range,
                lane,
                sharedHistogram);
        }
    }

    __syncthreads();

    // Final merge:
    // Each warp reduces one bin at a time by reading the 32 privatized copies of
    // that bin. Lane c reads copy c, the warp sums via shuffles, and lane 0 merges
    // the per-block total into the global histogram.
    for (unsigned int bin = warp; bin < range; bin += warpsPerBlock) {
        unsigned int blockBinCount = sharedHistogram[bin * histogramCopies + lane];
        blockBinCount = warp_reduce_sum(blockBinCount);

        if (lane == 0u && blockBinCount != 0u) {
            if (gridDim.x == 1u) {
                // Fast path for tiny inputs that need only one block.
                histogram[bin] = blockBinCount;
            } else {
                atomicAdd(histogram + bin, blockBinCount);
            }
        }
    }
}

template <bool FullRange>
inline void launch_histogram_kernel(const char *input,
                                    unsigned int *histogram,
                                    unsigned int inputSize,
                                    unsigned int fromValue,
                                    unsigned int range,
                                    size_t sharedMemBytes) {
    // This kernel benefits more from shared-memory capacity/occupancy than from L1.
    // Request the shared-heavy carveout to maximize resident blocks, especially for
    // the worst-case 32 KB/block shared histogram.
    cudaFuncSetAttribute(histogram_kernel<FullRange>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 1;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogram_kernel<FullRange>,
        blockSize,
        sharedMemBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    // Launch only enough blocks to fill the machine. The grid-stride loop covers
    // the whole input, and limiting the grid avoids unnecessary per-block merges
    // into the global histogram.
    const size_t tileBytes = static_cast<size_t>(blockSize) * itemsPerThread;
    size_t launchBlocks = (static_cast<size_t>(inputSize) + tileBytes - 1u) / tileBytes;

    const size_t maxUsefulBlocks =
        static_cast<size_t>(smCount) * static_cast<size_t>(activeBlocksPerSm);

    if (launchBlocks > maxUsefulBlocks) {
        launchBlocks = maxUsefulBlocks;
    }
    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }

    histogram_kernel<FullRange>
        <<<static_cast<unsigned int>(launchBlocks), blockSize, sharedMemBytes>>>(
            input,
            histogram,
            inputSize,
            fromValue,
            range);
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The output is defined as absolute counts for the requested range, so clear
    // it first. The caller explicitly handles synchronization, so this function
    // only enqueues work on the default stream.
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    cudaMemsetAsync(
        histogram,
        0,
        static_cast<size_t>(range) * sizeof(unsigned int),
        0);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory holds 32 privatized copies of the requested histogram range.
    const size_t sharedMemBytes =
        static_cast<size_t>(range) * histogramCopies * sizeof(unsigned int);

    // range == 256 implies the full [0, 255] histogram; that specialization removes
    // the range test from the hot loop entirely.
    if (range == 256u) {
        launch_histogram_kernel<true>(
            input,
            histogram,
            inputSize,
            0u,
            range,
            sharedMemBytes);
    } else {
        launch_histogram_kernel<false>(
            input,
            histogram,
            inputSize,
            static_cast<unsigned int>(from),
            range,
            sharedMemBytes);
    }
}