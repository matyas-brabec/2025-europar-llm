#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - Computes a histogram for characters in the inclusive range [from, to].
 * - Input:  device pointer to chars (`input`), length `inputSize`.
 * - Output: device pointer to unsigned int histogram (`histogram`) of size (to - from + 1).
 *
 * Performance design:
 * - Per-block histogram privatization in shared memory.
 *   * To reduce contention further, each warp in a block gets its own private histogram.
 *   * At the end of the kernel, per-warp histograms are reduced to global memory.
 * - Global memory reads use a classic grid-stride loop, which results in coalesced accesses.
 * - The `itemsPerThread` constant controls how many input elements each thread
 *   is expected to process on average via the grid configuration in `run_histogram`.
 *
 * Assumptions:
 * - `0 <= from <= to <= 255` and the histogram size is (to - from + 1) <= 256.
 * - `input` and `histogram` are valid device pointers allocated with `cudaMalloc`.
 * - The caller handles any host-device synchronization after `run_histogram`.
 */

static constexpr int WARP_SIZE        = 32;
static constexpr int THREADS_PER_BLOCK = 256;

/*
 * itemsPerThread:
 *   Controls how many input characters each thread **on average** processes.
 *   This is enforced by selecting the grid size in `run_histogram` such that:
 *
 *     totalThreads ≈ inputSize / itemsPerThread
 *
 *   For large inputs on modern GPUs, a value of 8 provides a good balance between
 *   parallelism, memory latency hiding, and kernel launch overhead.
 */
static constexpr int itemsPerThread   = 8;

static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0,
              "THREADS_PER_BLOCK must be a multiple of WARP_SIZE");

__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int s_hist[];

    const int range = to - from + 1;             // Number of bins in the histogram.
    const int tid   = threadIdx.x;
    const int warpId = tid / WARP_SIZE;          // Warp index within the block.
    const int warpsPerBlock = blockDim.x / WARP_SIZE;

    // -------------------------------------------------------------------------
    // Initialize per-warp histograms in shared memory.
    // Layout: s_hist[w * range + bin], w in [0, warpsPerBlock), bin in [0, range).
    // -------------------------------------------------------------------------
    for (int i = tid; i < range * warpsPerBlock; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Each thread processes multiple input characters using a grid-stride loop:
    //   globalId = blockIdx.x * blockDim.x + threadIdx.x
    //   for (idx = globalId; idx < inputSize; idx += totalThreads) { ... }
    //
    // The grid size in `run_histogram` is chosen so that each thread processes
    // approximately `itemsPerThread` characters on average.
    // -------------------------------------------------------------------------
    const unsigned int globalId     = blockIdx.x * blockDim.x + tid;
    const unsigned int totalThreads = gridDim.x  * blockDim.x;

    for (unsigned int idx = globalId; idx < inputSize; idx += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int cVal = static_cast<int>(c);

        if (cVal >= from && cVal <= to) {
            int bin = cVal - from;
            // Update this warp's private histogram.
            atomicAdd(&s_hist[warpId * range + bin], 1u);
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // Reduce per-warp histograms into the global histogram.
    // Each block contributes its bins via atomicAdd to global memory.
    // -------------------------------------------------------------------------
    for (int bin = tid; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;

        // Accumulate contributions from all warps in this block.
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * range + bin];
        }

        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text buffer (chars).
 *   histogram  - device pointer to histogram buffer (unsigned int),
 *                must have at least (to - from + 1) elements.
 *   inputSize  - number of chars in the input buffer.
 *   from, to   - inclusive range of character codes [from, to] to be counted.
 *
 * Notes:
 *   - The function zeroes the output histogram on the device.
 *   - No host synchronization is performed; the caller must synchronize if needed.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int range = to - from + 1;

    // Initialize the histogram to zero on the device.
    // This ensures the kernel can safely accumulate with atomicAdd.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // If there's no input, we're done after zeroing the histogram.
    if (inputSize == 0) {
        return;
    }

    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int warpsPerBlock   = threadsPerBlock / WARP_SIZE;

    // Determine how many threads and blocks to launch.
    //
    // We want approximately `itemsPerThread` input elements per thread:
    //   totalThreadsNeeded ≈ inputSize / itemsPerThread
    //   blocks = ceil(totalThreadsNeeded / threadsPerBlock)
    //
    // The block count is clamped to the maximum grid dimension in X (65535).
    unsigned long long totalThreadsNeeded =
        (static_cast<unsigned long long>(inputSize) + itemsPerThread - 1ULL) /
        static_cast<unsigned long long>(itemsPerThread);

    unsigned long long blocksULL =
        (totalThreadsNeeded + static_cast<unsigned long long>(threadsPerBlock) - 1ULL) /
        static_cast<unsigned long long>(threadsPerBlock);

    if (blocksULL == 0ULL) {
        blocksULL = 1ULL;
    }
    if (blocksULL > 65535ULL) {
        blocksULL = 65535ULL;
    }
    const int blocks = static_cast<int>(blocksULL);

    // Shared memory size: one histogram per warp in the block.
    const size_t sharedMemSize =
        static_cast<size_t>(range) *
        static_cast<size_t>(warpsPerBlock) *
        sizeof(unsigned int);

    // Launch the kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}