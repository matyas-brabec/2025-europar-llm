#include <cuda_runtime.h>

/*
 * Optimized CUDA histogram kernel for text data.
 *
 * - Computes the histogram for a given character range [from, to] (inclusive).
 * - Uses shared memory histogram privatization with 32 copies (one per warp lane).
 * - Each copy is laid out with an odd stride (257) so that accesses to the same
 *   bin from different lanes map to different shared memory banks, avoiding
 *   intra-warp bank conflicts.
 * - Each thread processes ITEMS_PER_THREAD input characters.
 * - Suitable for large inputs on modern GPUs (e.g., A100/H100).
 */

constexpr int ITEMS_PER_THREAD      = 8;   // Tunable: number of items processed by each thread
constexpr int THREADS_PER_BLOCK     = 256; // Tunable: threads per block (must be a multiple of 32)
constexpr int HISTOGRAM_COPIES      = 32;  // One copy per warp lane
constexpr int HISTOGRAM_STRIDE      = 257; // Odd stride >= 256 to avoid bank conflicts
constexpr int HISTOGRAM_SMEM_SIZE   = HISTOGRAM_COPIES * HISTOGRAM_STRIDE;

__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ globalHistogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Shared memory layout:
    // s_hist[copy * HISTOGRAM_STRIDE + bin]
    //   copy in [0, 31]    -> warp lane index
    //   bin  in [0, numBins-1]
    // Stride is odd, so accesses to the same bin from all 32 lanes go to
    // different banks and are conflict-free.
    __shared__ unsigned int s_hist[HISTOGRAM_SMEM_SIZE];

    const int numBins = to - from + 1;
    if (numBins <= 0 || inputSize == 0) {
        return;
    }

    // Zero the entire shared-memory histogram region.
    // This is slightly more than strictly necessary (we zero all 32*257 words),
    // but the overhead is small and simplifies indexing logic.
    for (int i = threadIdx.x; i < HISTOGRAM_SMEM_SIZE; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Per-thread local histogram pointer (one copy per warp lane).
    const int lane = threadIdx.x & 31; // Warp lane index [0, 31]
    unsigned int *localHist = s_hist + lane * HISTOGRAM_STRIDE;

    // Global thread index (64-bit to safely handle very large inputs).
    const unsigned long long globalThreadId =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long totalThreads =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const unsigned long long size64 = static_cast<unsigned long long>(inputSize);

    // Strided loop: each thread processes ITEMS_PER_THREAD elements, with a stride
    // of totalThreads. For iteration i, all threads in a warp access consecutive
    // locations, which yields coalesced global memory loads.
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned long long idx = globalThreadId + static_cast<unsigned long long>(i) * totalThreads;
        if (idx >= size64) {
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= static_cast<unsigned char>(from) && c <= static_cast<unsigned char>(to)) {
            int bin = static_cast<int>(c) - from; // bin in [0, numBins-1]

            // Multiple warps share the same 32 copies, so we use atomicAdd
            // in shared memory to avoid data races. Intra-warp accesses are
            // conflict-free due to the bank-aware layout.
            atomicAdd(&localHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane copies into the global histogram.
    // Each block produces a partial histogram; we accumulate into the global
    // histogram using atomicAdd per bin.
    for (int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
        unsigned int sum = 0;

#pragma unroll
        for (int copy = 0; copy < HISTOGRAM_COPIES; ++copy) {
            sum += s_hist[copy * HISTOGRAM_STRIDE + binIdx];
        }

        if (sum != 0) {
            atomicAdd(&globalHistogram[binIdx], sum);
        }
    }
}

/*
 * Host function that configures and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input chars (cudaMalloc'd)
 *   histogram  - device pointer to output histogram (cudaMalloc'd)
 *   inputSize  - number of chars in the input buffer
 *   from, to   - character range [from, to] (0 <= from < to <= 255)
 *
 * The histogram array must have length (to - from + 1) elements.
 * This function:
 *   - zeros the device histogram (asynchronously in stream 0),
 *   - launches the kernel with an appropriate grid configuration.
 * Synchronization after the kernel is left to the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        // Nothing to do; but make sure histogram is zeroed if range is valid.
        if (from <= to) {
            const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
            cudaMemsetAsync(histogram, 0, numBins * sizeof(unsigned int), 0);
        }
        return;
    }

    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram on the device (asynchronous, stream 0).
    cudaMemsetAsync(histogram, 0, numBins * sizeof(unsigned int), 0);

    // Configure grid and block dimensions.
    // Each thread processes ITEMS_PER_THREAD elements.
    const unsigned long long workPerBlock =
        static_cast<unsigned long long>(THREADS_PER_BLOCK) * ITEMS_PER_THREAD;
    unsigned int numBlocks = static_cast<unsigned int>(
        (static_cast<unsigned long long>(inputSize) + workPerBlock - 1) / workPerBlock);

    if (numBlocks == 0) {
        numBlocks = 1;
    }

    histogram_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(input, histogram, inputSize, from, to);
}