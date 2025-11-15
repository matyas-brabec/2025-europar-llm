#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a partial histogram of a character buffer.
 *
 * - input: device pointer to an array of chars containing the text.
 * - g_hist: device pointer to an array of unsigned int with size (to - from + 1).
 *           Each element i (0-based) will hold the count of character (i + from).
 * - inputSize: number of characters in the input buffer.
 * - from, to: inclusive range of character codes [from, to] that we care about.
 *
 * The kernel uses per-block shared memory to accumulate a local histogram,
 * which is then merged atomically into the global histogram.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ g_hist,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamic shared-memory array for per-block histogram.
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;

    // Initialize shared histogram to zero.
    // Multiple threads cooperate to zero all bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input data for good load balancing
    // and to handle arbitrarily large inputs with a fixed grid size.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gridStride     = blockDim.x * gridDim.x;

    for (unsigned int idx = globalThreadId; idx < inputSize; idx += gridStride) {
        // Cast to unsigned char to get consistent values in [0, 255],
        // regardless of whether 'char' is signed or unsigned.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Count only characters within the requested [from, to] range.
        if (bin >= 0 && bin < numBins) {
            // Atomic add in shared memory is relatively fast on modern GPUs.
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Merge the per-block histogram into the global histogram.
    // Each thread handles a subset of bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int count = s_hist[i];
        if (count > 0) {
            atomicAdd(&g_hist[i], count);
        }
    }
}

/*
 * Host helper function to run the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text buffer (allocated with cudaMalloc).
 *   histogram  - device pointer to output histogram buffer of size (to - from + 1).
 *   inputSize  - number of characters in 'input'.
 *   from, to   - inclusive range of character codes [from, to] for which to compute the histogram.
 *
 * Notes:
 *   - This function initializes the device 'histogram' buffer to zero.
 *   - Kernel launch is asynchronous; any synchronization is the caller's responsibility.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Ensure the output histogram starts from zero.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there is no input data, nothing more to do.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable launch configuration for modern data-center GPUs.
    const int threadsPerBlock = 256;

    // Use enough blocks to cover the data at least once; the kernel itself
    // uses a grid-stride loop to cover larger arrays.
    unsigned int blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) {
        blocksPerGrid = 1;
    }

    // Shared memory size per block: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}