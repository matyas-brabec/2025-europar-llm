#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram for characters in the inclusive range [from, to].
 *
 * - input: pointer to device memory containing 'inputSize' chars of text
 * - histogram: pointer to device memory of size (to - from + 1) unsigned ints
 * - inputSize: number of characters in 'input'
 * - from, to: inclusive range of character codes to histogram; 0 <= from <= to <= 255
 *
 * Optimization strategy:
 *   - Each block has a private histogram in shared memory for the given range.
 *   - Threads atomically update the shared histogram (fast shared memory atomics).
 *   - At the end, the shared histogram is reduced into the global histogram using
 *     a small number of global atomicAdd operations (one per bin per block at most).
 *   - The kernel uses a grid-stride loop to allow arbitrary input sizes while
 *     keeping the number of blocks reasonably small.
 */

__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamic shared memory: per-block histogram for the [from, to] range.
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;

    // Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Total number of threads in the grid.
    const unsigned int totalThreads = static_cast<unsigned int>(gridDim.x) *
                                      static_cast<unsigned int>(blockDim.x);
    const unsigned int globalThreadId =
        static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(blockDim.x) +
        static_cast<unsigned int>(threadIdx.x);

    // Grid-stride loop over the input.
    for (unsigned int idx = globalThreadId; idx < inputSize; idx += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Compute bin index relative to 'from'. Use unsigned comparison to check range:
        // bin is in [0, numBins) iff (unsigned)bin < (unsigned)numBins.
        int bin = static_cast<int>(c) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate per-block shared histogram into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val > 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function that launches the histogram_kernel with appropriate parameters.
 *
 * Parameters:
 *   - input: device pointer to input characters (allocated with cudaMalloc)
 *   - histogram: device pointer to histogram array (size: to - from + 1 unsigned ints)
 *   - inputSize: number of characters in 'input'
 *   - from, to: inclusive character range for the histogram; 0 <= from < to <= 255
 *
 * Assumptions:
 *   - 'input' and 'histogram' are valid device pointers.
 *   - Caller is responsible for any required host-device synchronization.
 *   - This function uses the default stream (stream 0) for cudaMemsetAsync and kernel launch.
 */

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of histogram bins.
    const int numBins = to - from + 1;

    // Ensure the output histogram is initialized to zero.
    // Asynchronous memset on the default stream.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // If there is no input, nothing else to do.
    if (inputSize == 0u) {
        return;
    }

    // Choose a reasonable block size for modern GPUs.
    const int threadsPerBlock = 256;

    // Use enough blocks to cover the input, but cap to avoid excessive launch overhead.
    int blocks = static_cast<int>((inputSize + threadsPerBlock - 1u) / threadsPerBlock);
    const int maxBlocks = 65535; // Safe upper bound for 1D grids on all CUDA-capable GPUs.
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    // Shared memory size per block: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}