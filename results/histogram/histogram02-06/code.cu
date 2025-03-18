#include <cuda_runtime.h>
#include <cstdio>

// This CUDA kernel computes a histogram for characters in the range [from, to].
// Each block uses a shared memory histogram to accumulate counts locally,
// then the block-level histogram is merged into the global histogram using atomicAdd.
// The input array is expected to contain plain text (chars) and is located in device memory.
// The global histogram array (also in device memory) has length (to - from + 1),
// where histogram[i] will count the occurrences of the character with ordinal (i + from).
//
__global__ void histogramKernel(const char *input, unsigned int *histogram,
                                unsigned int inputSize, int from, int to) {
    // Compute number of histogram bins.
    int numBins = (to - from) + 1;

    // Dynamically allocated shared memory for the block-level histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to zero.
    // Each thread initializes multiple bins if necessary.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index and the total stride across the grid.
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process the input array in a grid-stride loop.
    while (globalIdx < inputSize) {
        // Read the current character.
        unsigned char c = input[globalIdx];
        // Check if the character falls in the desired range.
        if (c >= from && c <= to) {
            // Atomically increment the corresponding bin in shared memory.
            atomicAdd(&s_hist[c - from], 1);
        }
        globalIdx += stride;
    }
    __syncthreads();

    // Merge the block's shared histogram into the global histogram.
    // Each thread is responsible for updating a subset of bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// This function sets up and launches the CUDA histogram kernel.
// It assumes that 'input' and 'histogram' are device pointers allocated via cudaMalloc,
// and that the global histogram array is of size (to - from + 1).
// The function initializes the global histogram to zero using cudaMemset, computes
// the grid and block dimensions, and launches the kernel with sufficient shared memory.
// Host-device synchronization (if needed) must be handled by the caller.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to) {
    // Calculate the number of bins based on the provided range.
    int numBins = (to - from) + 1;

    // Zero the global histogram. This ensures that the counts start at zero.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Define the block size (number of threads per block).
    const int blockSize = 256;
    // Determine the grid size so that all input characters will be processed.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Launch the kernel.
    // The third kernel parameter specifies dynamically allocated shared memory size.
    // Each block will have (numBins * sizeof(unsigned int)) bytes of shared memory.
    histogramKernel<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(input,
                                                                             histogram,
                                                                             inputSize,
                                                                             from,
                                                                             to);
    // Note: The caller is responsible for synchronizing (e.g., via cudaDeviceSynchronize) if needed.
}