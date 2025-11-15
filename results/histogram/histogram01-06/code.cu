#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram over a specified inclusive character range [from, to].
 *
 * Parameters:
 *   input      - device pointer to input characters
 *   histogram  - device pointer to global histogram array of size (to - from + 1)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character range [from, to] to be histogrammed
 *
 * Strategy:
 *   - Each block maintains a private histogram in shared memory to reduce global atomic contention.
 *   - Threads cooperatively initialize the shared histogram.
 *   - Threads process input using a grid-stride loop and update shared histogram with atomics.
 *   - After processing, threads cooperatively accumulate the shared histogram into the global histogram.
 */
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;
    const bool fullRange = (from == 0) && (to == 255);

    // Initialize shared histogram to zero
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute global thread index and stride using 64-bit to avoid overflow
    size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // Grid-stride loop over input
    for (size_t i = tid; i < static_cast<size_t>(inputSize); i += stride) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        int code = static_cast<int>(c);

        if (fullRange) {
            // Fast path for full 0-255 range: no range check needed
            atomicAdd(&s_hist[code], 1U);
        } else {
            // General case: only count characters in [from, to]
            if (code >= from && code <= to) {
                int idx = code - from;  // bin index in [0, numBins-1]
                atomicAdd(&s_hist[idx], 1U);
            }
        }
    }

    __syncthreads();

    // Accumulate per-block shared histogram into global histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val > 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'd)
 *   histogram  - device pointer to histogram array (cudaMalloc'd),
 *                must have space for (to - from + 1) unsigned ints
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character range [from, to] to be histogrammed
 *
 * Notes:
 *   - This function does not perform any device synchronization; the caller is responsible
 *     for synchronization and error checking if desired.
 *   - The histogram array is zeroed before the kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of histogram bins (inclusive range)
    const int numBins = to - from + 1;

    // Clear the global histogram to zero
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there is no input, nothing more to do
    if (inputSize == 0) {
        return;
    }

    // Configure CUDA kernel launch parameters
    const int blockSize = 256;  // typical choice for modern GPUs
    int gridSize = (static_cast<int>(inputSize) + blockSize - 1) / blockSize;

    // Shared memory size per block: one unsigned int per bin
    size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the histogram kernel
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                             histogram,
                                                             inputSize,
                                                             from,
                                                             to);
}