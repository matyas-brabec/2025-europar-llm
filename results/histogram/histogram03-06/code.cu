#include <cuda_runtime.h>

// CUDA kernel that computes a partial histogram over a specified char range.
// The kernel uses shared memory to privately accumulate counts per block to reduce
// global atomic operations. After processing, each block atomically adds its shared memory
// results to the global histogram array.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to)
{
    // Calculate number of bins in the histogram (inclusive range).
    int numBins = to - from + 1;

    // Declare dynamic shared memory for the block-level histogram.
    // The shared memory size is specified at kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared memory histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Use grid-stride loop to traverse the input array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < inputSize) {
        // Convert the char to an unsigned char to get the ordinal value (avoid negative values).
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the specified range.
        if (c >= from && c <= to) {
            int bin = c - from;
            // Atomically increment the count in shared memory.
            atomicAdd(&s_hist[bin], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Each thread writes portions of the block's shared histogram back to the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&global_hist[i], s_hist[i]);
    }
}

// Host function that launches the histogram kernel.
// Parameters:
//   input: pointer to device memory containing the input text (array of chars).
//   histogram: pointer to device memory allocated for the histogram (unsigned int array).
//   inputSize: number of characters in the input buffer.
//   from, to: specify the inclusive range of char ordinal values for which the histogram is computed.
// Note: It is assumed that cudaMalloc has been used to allocate memory for both input and histogram arrays.
//       Any necessary host-device synchronization will be performed by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins required.
    int numBins = to - from + 1;

    // Define the number of threads per block.
    int blockSize = 256;

    // Compute the grid size so that all input characters are processed.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Calculate shared memory size required for each block.
    size_t sharedMemBytes = numBins * sizeof(unsigned int);

    // Launch the kernel with the chosen configuration.
    histogram_kernel<<<gridSize, blockSize, sharedMemBytes>>>(input, histogram, inputSize, from, to);
}