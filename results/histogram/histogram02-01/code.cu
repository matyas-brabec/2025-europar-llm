#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram of characters in the specified range [from, to].
 *
 * - input: pointer to the input character array (device memory).
 * - histogram: pointer to the output histogram array (device memory),
 *              of size (to - from + 1) unsigned ints.
 * - inputSize: number of characters in the input buffer.
 * - from, to: inclusive character range [from, to] to be histogrammed.
 *
 * The kernel uses per-block shared memory to accumulate a partial histogram and
 * then atomically merges it into the global histogram.
 */
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int blockThreads = blockDim.x;

    // Initialize the shared-memory histogram for this block.
    for (int i = tid; i < numBins; i += blockThreads) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input data.
    unsigned int globalThreadId = blockIdx.x * blockThreads + tid;
    unsigned int stride = blockThreads * gridDim.x;

    for (unsigned int idx = globalThreadId; idx < inputSize; idx += stride) {
        unsigned char c = static_cast<unsigned char>(input[idx]);  // Ensure 0..255 range
        int bin = static_cast<int>(c) - from;                      // Compute bin index

        // Only count characters that fall within the requested range [from, to].
        if (bin >= 0 && bin < numBins) {
            // Shared-memory atomics are fast on modern GPUs (A100/H100).
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge the per-block shared histogram into the global histogram.
    for (int i = tid; i < numBins; i += blockThreads) {
        unsigned int val = s_hist[i];
        if (val > 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function that sets up and launches the histogram kernel.
 *
 * Parameters:
 * - input: device pointer to input character buffer (allocated with cudaMalloc).
 * - histogram: device pointer to output histogram buffer (allocated with cudaMalloc),
 *              size must be (to - from + 1) * sizeof(unsigned int).
 * - inputSize: number of characters in the input buffer.
 * - from, to: inclusive character range [from, to] to be histogrammed.
 *
 * Notes:
 * - This function initializes the histogram buffer to zero using cudaMemset.
 * - Kernel launch and cudaMemset are asynchronous; caller is responsible for
 *   any required synchronization (e.g., cudaDeviceSynchronize or stream sync).
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins for the requested range.
    const int numBins = to - from + 1;

    // Initialize the output histogram to zero on the device.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there's no input data, nothing more to do.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size for modern GPUs.
    const unsigned int blockSize = 256;

    // Compute grid size to cover all input elements with a grid-stride loop.
    unsigned int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Dynamic shared memory size: one unsigned int per bin per block.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel. Error checking and synchronization are left to the caller.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                             histogram,
                                                             inputSize,
                                                             from,
                                                             to);
}