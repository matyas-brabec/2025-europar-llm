#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define block size for kernel execution.
#define BLOCK_SIZE 256

// CUDA kernel to compute the histogram over a specified character range.
// Each block computes a local histogram in shared memory and then atomically
// adds its results to the global histogram array.
// Parameters:
//   input: Pointer to the input text (device memory).
//   histogram: Pointer to the output histogram array (device memory).
//   inputSize: Number of characters in the input text.
//   from: Lower bound (inclusive) of the character range.
//   to: Upper bound (inclusive) of the character range.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Determine the size of the histogram range.
    int range_size = to - from + 1;

    // Allocate shared memory for block-local histogram.
    // The size of shared memory is specified dynamically during kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize each shared memory histogram bin to 0.
    for (int i = threadIdx.x; i < range_size; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array using a grid-stride loop.
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (globalIndex < inputSize)
    {
        // Read the character and cast it to unsigned char to avoid issues with signedness.
        unsigned char value = static_cast<unsigned char>(input[globalIndex]);
        // Update histogram only if the value is within the specified range.
        if (value >= from && value <= to)
        {
            int bin = value - from;
            // Atomically increment the bin in block-local shared memory.
            atomicAdd(&s_hist[bin], 1);
        }
        globalIndex += stride;
    }
    __syncthreads();

    // Each thread writes a portion of the block-local histogram to the global histogram.
    // The atomicAdd ensures that updates from different blocks are combined correctly.
    for (int i = threadIdx.x; i < range_size; i += blockDim.x)
    {
        unsigned int count = s_hist[i];
        if (count > 0)
        {
            atomicAdd(&histogram[i], count);
        }
    }
}

// Host function that sets up and launches the histogram kernel.
// Assumes that 'input' and 'histogram' have been allocated in device memory.
// The 'inputSize' indicates the number of characters in the input buffer.
// The histogram is computed only for characters in the range [from, to] (inclusive).
// The output histogram array has (to - from + 1) elements, where each element at index i
// corresponds to the count for character (i + from).
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Determine the number of histogram bins.
    int range_size = to - from + 1;

    // Define the block size and compute the required number of blocks.
    int blockSize = BLOCK_SIZE;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;

    // Calculate the amount of shared memory needed per block.
    size_t sharedMemSize = range_size * sizeof(unsigned int);

    // Launch the histogram kernel.
    // Host-device synchronization is assumed to be handled by the caller.
    histogram_kernel<<<numBlocks, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}