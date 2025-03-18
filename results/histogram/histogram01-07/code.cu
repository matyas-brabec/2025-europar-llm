#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel to compute histogram for a given range [from, to].
// The kernel uses block-level shared memory to build a partial histogram,
// and then atomically accumulates into the global histogram.
// Input parameters:
//   input        : pointer to an array of chars (on device) containing the text.
//   inputSize    : total number of characters in the input array.
//   globalHist   : pointer to global histogram (on device) with size = (to - from + 1).
//   from, to     : define the inclusive range of character values to count.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *globalHist, int from, int to)
{
    // Compute the number of histogram bins. For a continuous range [from, to],
    // the number of bins is (to - from + 1).
    const int range = to - from + 1;

    // Declare shared memory for the block-local histogram.
    // The shared memory must be dynamically allocated during kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram to 0.
    // Each thread initializes a subset of the bins.
    for (int i = threadIdx.x; i < range; i += blockDim.x)
    {
        s_hist[i] = 0;
    }

    // Ensure that all threads have zeroed out the shared memory.
    __syncthreads();

    // Process input string using grid-stride loop.
    // Each thread processes multiple elements separated by total number of threads.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize)
    {
        // Cast to unsigned char to properly handle characters with values > 127.
        unsigned int c = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the target range.
        if (c >= static_cast<unsigned int>(from) && c <= static_cast<unsigned int>(to))
        {
            // Increment bin corresponding to current character.
            // Use atomic operation on shared memory.
            atomicAdd(&s_hist[c - from], 1);
        }

        idx += stride;
    }

    // Ensure all threads have finished processing input before accumulating results.
    __syncthreads();

    // Accumulate the block-level histogram (in shared memory) into the global histogram.
    // Each thread updates a subset of bins from the shared histogram.
    for (int i = threadIdx.x; i < range; i += blockDim.x)
    {
        // Even if s_hist[i] is 0, the atomicAdd is correct.
        atomicAdd(&globalHist[i], s_hist[i]);
    }
}

// Host function that sets up and launches the CUDA kernel.
// Parameters:
//   input      : device pointer to the input text array (allocated using cudaMalloc).
//   histogram  : device pointer to the histogram array (allocated using cudaMalloc).
//                It should have room for (to - from + 1) unsigned integers.
//   inputSize  : the number of characters in the input array.
//   from, to   : define the inclusive range of character values for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int range = to - from + 1;

    // Choose a block size. Here we use 256 threads per block as a typical configuration.
    int blockSize = 256;

    // Compute grid size based on input size. Each thread processes multiple elements via grid-stride looping.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Initialize the global histogram to 0 on the device.
    // It's assumed that histogram pointer is allocated on device.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Launch the kernel.
    // The dynamic shared memory size is set to (range * sizeof(unsigned int)).
    histogram_kernel<<<gridSize, blockSize, range * sizeof(unsigned int)>>>(input, inputSize, histogram, from, to);
    
    // Note: Any host-device synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
}