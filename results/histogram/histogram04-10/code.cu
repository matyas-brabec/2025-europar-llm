#include <cuda_runtime.h>
#include <cstddef>

/*
 * Range-restricted byte histogram for a device-resident text buffer.
 *
 * Key optimization choices:
 * - Shared-memory privatization: each warp owns its own private histogram in shared memory.
 *   This reduces contention versus a single block-private histogram, which matters for text
 *   where a few characters (space, newline, vowels, etc.) can be very hot.
 * - itemsPerThread = 16 by default: on A100/H100 this is a strong default for large inputs
 *   because it gives each thread one 16-byte aligned chunk per loop iteration, enabling
 *   vectorized uint4 loads, enough ILP to hide latency, and still low register pressure.
 * - Final merge: a block emits only one global atomic add per bin, instead of one global
 *   atomic per matching character.
 *
 * The input is interpreted as unsigned bytes, so the code is correct regardless of whether
 * the host compiler treats `char` as signed or unsigned.
 */
constexpr int threadsPerBlock = 256;
constexpr int itemsPerThread = 16;

static_assert(sizeof(unsigned int) == 4, "This implementation assumes 32-bit unsigned int.");
static_assert(threadsPerBlock % 32 == 0, "threadsPerBlock must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

namespace histogram_detail {

constexpr int histogramWarpSize = 32;
constexpr int targetBlocksPerSM = 8; // 256 threads/block * 8 blocks/SM = 2048 threads/SM on A100/H100.

__host__ __device__ constexpr unsigned int div_up_u32(unsigned int n, unsigned int d) {
    return n / d + static_cast<unsigned int>((n % d) != 0u);
}

__host__ __device__ constexpr unsigned int round_up_to_warp(unsigned int value) {
    return (value + static_cast<unsigned int>(histogramWarpSize - 1)) &
           ~static_cast<unsigned int>(histogramWarpSize - 1);
}

__device__ __forceinline__ void add_byte_to_private_hist(
    unsigned int byteValue,
    unsigned int from,
    unsigned int rangeSize,
    unsigned int* privateHist)
{
    // Unsigned subtraction folds the two range checks
    //    byteValue >= from  &&  byteValue < from + rangeSize
    // into a single comparison.
    const unsigned int bin = byteValue - from;
    if (bin < rangeSize) {
        atomicAdd(privateHist + bin, 1u);
    }
}

__device__ __forceinline__ void add_word_to_private_hist(
    unsigned int word,
    unsigned int from,
    unsigned int rangeSize,
    unsigned int* privateHist)
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        add_byte_to_private_hist(word & 0xFFu, from, rangeSize, privateHist);
        word >>= 8;
    }
}

// Compile-time load specialization:
// - multiples of 16 bytes use uint4 vector loads
// - other multiples of 4 bytes use 32-bit word loads
// - everything else falls back to scalar byte loads
template <int ITEMS_PER_THREAD, bool USE_VEC16 = (ITEMS_PER_THREAD % 16 == 0), bool USE_VEC4 = (ITEMS_PER_THREAD % 4 == 0)>
struct FullChunkProcessor;

template <int ITEMS_PER_THREAD>
struct FullChunkProcessor<ITEMS_PER_THREAD, true, true> {
    __device__ __forceinline__ static void run(
        const unsigned char* ptr,
        unsigned int from,
        unsigned int rangeSize,
        unsigned int* privateHist)
    {
        // The fast path assumes the device buffer comes directly from cudaMalloc, as specified,
        // so the base pointer is sufficiently aligned. With itemsPerThread being a multiple of 16,
        // each thread's chunk start is also 16-byte aligned.
        const uint4* const vecPtr = reinterpret_cast<const uint4*>(ptr);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 16; ++i) {
            const uint4 packed = vecPtr[i];
            add_word_to_private_hist(packed.x, from, rangeSize, privateHist);
            add_word_to_private_hist(packed.y, from, rangeSize, privateHist);
            add_word_to_private_hist(packed.z, from, rangeSize, privateHist);
            add_word_to_private_hist(packed.w, from, rangeSize, privateHist);
        }
    }
};

template <int ITEMS_PER_THREAD>
struct FullChunkProcessor<ITEMS_PER_THREAD, false, true> {
    __device__ __forceinline__ static void run(
        const unsigned char* ptr,
        unsigned int from,
        unsigned int rangeSize,
        unsigned int* privateHist)
    {
        const unsigned int* const wordPtr = reinterpret_cast<const unsigned int*>(ptr);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD / 4; ++i) {
            add_word_to_private_hist(wordPtr[i], from, rangeSize, privateHist);
        }
    }
};

template <int ITEMS_PER_THREAD>
struct FullChunkProcessor<ITEMS_PER_THREAD, false, false> {
    __device__ __forceinline__ static void run(
        const unsigned char* ptr,
        unsigned int from,
        unsigned int rangeSize,
        unsigned int* privateHist)
    {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            add_byte_to_private_hist(ptr[i], from, rangeSize, privateHist);
        }
    }
};

template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    const unsigned int inputSize,
    const unsigned int from,
    const unsigned int rangeSize)
{
    static_assert(THREADS_PER_BLOCK % histogramWarpSize == 0, "THREADS_PER_BLOCK must be a multiple of 32.");
    static_assert(ITEMS_PER_THREAD > 0, "ITEMS_PER_THREAD must be positive.");

    extern __shared__ unsigned int shared[];

    constexpr int warpsPerBlock = THREADS_PER_BLOCK / histogramWarpSize;
    const unsigned int paddedRange = round_up_to_warp(rangeSize);

    // Layout in shared memory:
    //   [warp0 histogram | warp1 histogram | ...]
    // Each warp gets its own privatized histogram, padded to a multiple of 32 bins so every
    // sub-histogram starts on a warp-sized boundary.
    const unsigned int warpId = threadIdx.x / histogramWarpSize;
    unsigned int* const privateHist = shared + warpId * paddedRange;
    const unsigned int totalPrivateBins = static_cast<unsigned int>(warpsPerBlock) * paddedRange;

    // Zero the entire shared-memory histogram slab.
    for (unsigned int i = threadIdx.x; i < totalPrivateBins; i += static_cast<unsigned int>(THREADS_PER_BLOCK)) {
        shared[i] = 0u;
    }
    __syncthreads();

    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // Use 64-bit only for outer address arithmetic so the kernel remains safe for near-4 GiB
    // buffers, while the hot histogram/bin arithmetic stays 32-bit.
    const size_t globalThread =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(THREADS_PER_BLOCK) +
        static_cast<size_t>(threadIdx.x);
    const size_t totalThreads = static_cast<size_t>(gridDim.x) * static_cast<size_t>(THREADS_PER_BLOCK);
    const size_t gridStrideBytes = totalThreads * static_cast<size_t>(ITEMS_PER_THREAD);
    const size_t inputSizeBytes = static_cast<size_t>(inputSize);

    size_t idx = globalThread * static_cast<size_t>(ITEMS_PER_THREAD);

    while (idx < inputSizeBytes) {
        const unsigned int idx32 = static_cast<unsigned int>(idx);
        const unsigned int remaining = inputSize - idx32;

        if (remaining >= static_cast<unsigned int>(ITEMS_PER_THREAD)) {
            FullChunkProcessor<ITEMS_PER_THREAD>::run(inputBytes + idx, from, rangeSize, privateHist);
        } else {
            // Only the very last chunk for some threads takes this path.
            #pragma unroll
            for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
                if (static_cast<unsigned int>(j) < remaining) {
                    add_byte_to_private_hist(inputBytes[idx + static_cast<size_t>(j)], from, rangeSize, privateHist);
                }
            }
        }

        idx += gridStrideBytes;
    }

    __syncthreads();

    // Merge the warp-private histograms. The global histogram is compact:
    // histogram[bin] corresponds to byte value (from + bin).
    for (unsigned int bin = threadIdx.x; bin < rangeSize; bin += static_cast<unsigned int>(THREADS_PER_BLOCK)) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += shared[static_cast<unsigned int>(w) * paddedRange + bin];
        }

        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

} // namespace histogram_detail

void run_histogram(const char* input, unsigned int* histogram, unsigned int inputSize, int from, int to)
{
    // The interface has no stream parameter, so work is enqueued onto the default stream.
    // The caller is responsible for any synchronization.
    const cudaStream_t stream = nullptr;

    // Inclusive range [from, to].
    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int rangeSize = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(rangeSize) * sizeof(unsigned int);

    // Compute a fresh histogram, not an accumulation of previous contents.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    int smCount = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    // With the default 256-thread block size, 8 blocks/SM is the natural saturation target
    // for A100/H100 (2048 resident threads/SM). Grid-stride looping covers the rest of the input.
    const unsigned int workPerBlock =
        static_cast<unsigned int>(threadsPerBlock) * static_cast<unsigned int>(itemsPerThread);
    const unsigned int blocksForInput = histogram_detail::div_up_u32(inputSize, workPerBlock);
    const unsigned int saturationBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(histogram_detail::targetBlocksPerSM);
    const unsigned int gridBlocks = (blocksForInput < saturationBlocks) ? blocksForInput : saturationBlocks;

    // One padded shared histogram per warp.
    const unsigned int paddedRange = histogram_detail::round_up_to_warp(rangeSize);
    const size_t sharedBytes =
        static_cast<size_t>(threadsPerBlock / histogram_detail::histogramWarpSize) *
        static_cast<size_t>(paddedRange) *
        sizeof(unsigned int);

    histogram_detail::histogram_range_kernel<threadsPerBlock, itemsPerThread>
        <<<gridBlocks, threadsPerBlock, sharedBytes, stream>>>(
            input,
            histogram,
            inputSize,
            fromU,
            rangeSize);
}