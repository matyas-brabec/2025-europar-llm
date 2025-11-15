#include <cuda_runtime.h>

/*
 * CUDA kernel for computing a histogram over a restricted character range.
 *
 * - input:      device pointer to input text (array of chars)
 * - histogram:  device pointer to output histogram (array of unsigned ints)
 * - inputSize:  number of characters in input
 * - from, to:   inclusive character range [from, to] to be counted
 *
 * The histogram array must have (to - from + 1) elements. Bin i counts
 * occurrences of character with ASCII code (from + i).
 *
 * Implementation notes:
 * - Each block keeps a private histogram in shared memory to reduce contention
 *   on global memory atomics.
 * - Threads within a block atomically update the block-local histogram.
 * - At the end, each block atomically accumulates its local histogram into
 *   the global histogram.
 * - Input is processed with a grid-stride loop to handle arbitrary input sizes.
 */
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int sharedHist[];  // block-local histogram

    const int numBins = to - from + 1;

    // Initialize shared histogram to zero.
    // Each thread zeroes multiple bins if necessary.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }

    __syncthreads();

    // Grid-stride loop over the input.
    unsigned int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    while (idx < inputSize) {
        // Treat characters as unsigned to avoid negative values.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {
            // Update block-local histogram; shared-memory atomics are fast.
            atomicAdd(&sharedHist[c - from], 1u);
        }

        idx += stride;
    }

    __syncthreads();

    // Accumulate block-local histogram into global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = sharedHist[i];
        if (val != 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function to launch the histogram kernel.
 *
 * Parameters:
 *  - input:      device pointer to input char buffer (allocated with cudaMalloc)
 *  - histogram:  device pointer to output histogram (allocated with cudaMalloc)
 *  - inputSize:  number of characters in the input buffer
 *  - from, to:   inclusive character range [from, to] over which to compute the histogram
 *
 * Requirements:
 *  - 0 <= from <= to <= 255
 *  - histogram must have (to - from + 1) unsigned int elements in device memory
 *
 * Notes:
 *  - This function does NOT perform any explicit device synchronization;
 *    the caller is responsible for calling cudaDeviceSynchronize() or
 *    equivalent if needed.
 *  - The histogram is zero-initialized using cudaMemsetAsync before the kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic argument validation; silently return on invalid input.
    if (input == nullptr || histogram == nullptr) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int    numBins       = to - from + 1;
    const size_t histogramSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Zero the output histogram on device (asynchronously on default stream).
    cudaMemsetAsync(histogram, 0, histogramSize, 0);

    // Choose a reasonable block size; 256 threads is a good default on modern GPUs.
    const int blockSize = 256;

    // Compute grid size to cover all input elements; ensure at least one block.
    unsigned int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }

    // Shared memory per block: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel on the default stream.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                            histogram,
                                                            inputSize,
                                                            from,
                                                            to);
}