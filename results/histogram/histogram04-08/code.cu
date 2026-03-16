#include <cuda_runtime.h>

// Range-restricted byte histogram over device-resident text input.
// output[i] counts occurrences of the byte value (from + i); bytes outside
// the inclusive [from, to] interval are ignored.

namespace {

// The requested tuning knob.
// 16 bytes/thread is a strong default on A100/H100-class GPUs for large inputs:
// it maps the steady-state load to one aligned 128-bit load per thread and
// amortizes shared-histogram setup/reduction well. If changed, the code still
// works; the vectorized fast path is used whenever it remains a multiple of 16.
static constexpr int itemsPerThread = 16;

// 256 threads/block lets the block-wide merge map one thread to each possible
// compacted output bin because the byte domain is at most 256 values.
static constexpr int blockSize     = 256;
static constexpr int maxBins       = 256;
static constexpr int warpSizeConst = 32;
static constexpr int vectorBytes   = sizeof(uint4);

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(blockSize % warpSizeConst == 0, "blockSize must be a whole number of warps.");
static_assert(blockSize >= maxBins, "blockSize must be at least 256.");
static_assert(vectorBytes == 16, "uint4 is expected to be 16 bytes.");

__device__ __forceinline__
void accumulate_byte(unsigned int byteValue,
                     unsigned int from,
                     unsigned int numBins,
                     unsigned int* __restrict__ warpHist)
{
    // Unsigned subtraction collapses the two-sided range test into one compare:
    // values below `from` underflow and therefore fail `bin < numBins`.
    const unsigned int bin = byteValue - from;
    if (bin < numBins) {
        atomicAdd(warpHist + bin, 1u);
    }
}

__device__ __forceinline__
void accumulate_packed_word(unsigned int packed,
                            unsigned int from,
                            unsigned int numBins,
                            unsigned int* __restrict__ warpHist)
{
    accumulate_byte( packed        & 0xFFu, from, numBins, warpHist);
    accumulate_byte((packed >>  8) & 0xFFu, from, numBins, warpHist);
    accumulate_byte((packed >> 16) & 0xFFu, from, numBins, warpHist);
    accumulate_byte((packed >> 24) & 0xFFu, from, numBins, warpHist);
}

template <int ITEMS_PER_THREAD, bool VECTORIZED>
struct ThreadChunkProcessor;

template <int ITEMS_PER_THREAD>
struct ThreadChunkProcessor<ITEMS_PER_THREAD, true> {
    __device__ __forceinline__
    static void process_full(const unsigned char* __restrict__ input,
                             size_t base,
                             unsigned int from,
                             unsigned int numBins,
                             unsigned int* __restrict__ warpHist)
    {
        // Fast path: each thread owns a contiguous, 16-byte-aligned slice so a
        // multiple-of-16 itemsPerThread can be read as aligned uint4 vectors.
        #pragma unroll
        for (int v = 0; v < ITEMS_PER_THREAD / vectorBytes; ++v) {
            const size_t idx = base + static_cast<size_t>(v) * static_cast<size_t>(vectorBytes);
            const uint4 packed = *reinterpret_cast<const uint4*>(input + idx);
            accumulate_packed_word(packed.x, from, numBins, warpHist);
            accumulate_packed_word(packed.y, from, numBins, warpHist);
            accumulate_packed_word(packed.z, from, numBins, warpHist);
            accumulate_packed_word(packed.w, from, numBins, warpHist);
        }
    }
};

template <int ITEMS_PER_THREAD>
struct ThreadChunkProcessor<ITEMS_PER_THREAD, false> {
    __device__ __forceinline__
    static void process_full(const unsigned char* __restrict__ input,
                             size_t base,
                             unsigned int from,
                             unsigned int numBins,
                             unsigned int* __restrict__ warpHist)
    {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            accumulate_byte(static_cast<unsigned int>(input[base + static_cast<size_t>(i)]),
                            from,
                            numBins,
                            warpHist);
        }
    }
};

template <int ITEMS_PER_THREAD>
__device__ __forceinline__
void process_tail_chunk(const unsigned char* __restrict__ input,
                        size_t base,
                        size_t inputSize,
                        unsigned int from,
                        unsigned int numBins,
                        unsigned int* __restrict__ warpHist)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const size_t idx = base + static_cast<size_t>(i);
        if (idx < inputSize) {
            accumulate_byte(static_cast<unsigned int>(input[idx]), from, numBins, warpHist);
        }
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int numBins)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / warpSizeConst;
    typedef ThreadChunkProcessor<ITEMS_PER_THREAD, (ITEMS_PER_THREAD % vectorBytes) == 0> Processor;

    // One private histogram per warp in shared memory.
    // This lowers hot-bin contention compared with a single block-private histogram
    // while still reducing global atomics to one add per block/bin in the final merge.
    __shared__ unsigned int s_hist[WARPS_PER_BLOCK][maxBins];

    const unsigned int tid    = threadIdx.x;
    const unsigned int warpId = tid / warpSizeConst;
    unsigned int* const warpHist = s_hist[warpId];

    // Only compacted bins [0, numBins) are live; the rest of each 256-entry row is unused.
    if (tid < numBins) {
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            s_hist[w][tid] = 0u;
        }
    }
    __syncthreads();

    const size_t itemsPerThread64 = static_cast<size_t>(ITEMS_PER_THREAD);
    const size_t blockSpan        = static_cast<size_t>(BLOCK_SIZE) * itemsPerThread64;
    const size_t gridStride       = static_cast<size_t>(gridDim.x) * blockSpan;
    const size_t inputSize64      = static_cast<size_t>(inputSize);

    // Each thread owns a contiguous ITEMS_PER_THREAD-byte slice inside the block's chunk.
    size_t base =
        (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(BLOCK_SIZE) +
         static_cast<size_t>(tid)) * itemsPerThread64;

    // Steady-state loop: the entire per-thread slice is in-bounds, so no per-byte bounds checks.
    for (; base + itemsPerThread64 <= inputSize64; base += gridStride) {
        Processor::process_full(input, base, from, numBins, warpHist);
    }

    // At most one partial slice can remain for this thread because the next candidate
    // slice would start a whole-grid stride later.
    if (base < inputSize64) {
        process_tail_chunk<ITEMS_PER_THREAD>(input, base, inputSize64, from, numBins, warpHist);
    }

    __syncthreads();

    // One thread per compacted output bin reduces the warp-private copies.
    if (tid < numBins) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += s_hist[w][tid];
        }
        if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Per the prompt, `input` and `histogram` are device pointers allocated with cudaMalloc.
    // This function only enqueues work; synchronization is intentionally left to the caller.
    if (histogram == nullptr) {
        return;
    }

    const unsigned int fromU        = static_cast<unsigned int>(from);
    const unsigned int numBins      = static_cast<unsigned int>(to - from) + 1u;
    const size_t histogramBytes     = static_cast<size_t>(numBins) * sizeof(unsigned int);
    cudaStream_t stream             = 0;  // Required interface has no stream parameter.

    // Both operations go to the same stream, so the zero-fill completes before the
    // kernel starts without any host-side synchronization here.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (input == nullptr || inputSize == 0u) {
        return;
    }

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<blockSize, itemsPerThread>,
        blockSize,
        0);

    if (smCount <= 0 || activeBlocksPerSM <= 0) {
        return;
    }

    const size_t itemsPerBlock = static_cast<size_t>(blockSize) * static_cast<size_t>(itemsPerThread);
    const size_t totalChunks   = (static_cast<size_t>(inputSize) + itemsPerBlock - 1u) / itemsPerBlock;
    const size_t residentWaveBlocks =
        static_cast<size_t>(smCount) * static_cast<size_t>(activeBlocksPerSM);

    // Because the kernel is grid-stride, one occupancy-saturating resident wave of blocks
    // is enough to cover arbitrarily large inputs while keeping the final global merge cheap.
    const unsigned int gridBlocks = static_cast<unsigned int>(
        totalChunks < residentWaveBlocks ? totalChunks : residentWaveBlocks);

    if (gridBlocks == 0u) {
        return;
    }

    // Reinterpret as raw unsigned bytes so ordinals 128..255 are preserved correctly
    // regardless of whether plain `char` is signed or unsigned on the host toolchain.
    histogram_range_kernel<blockSize, itemsPerThread>
        <<<gridBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            fromU,
            numBins);
}