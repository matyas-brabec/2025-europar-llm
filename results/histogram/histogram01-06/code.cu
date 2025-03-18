#include <cuda_runtime.h>
#include <cstdio>

// This CUDA kernel computes the histogram of characters in the input text array 
// restricted to a continuous range [from, to].
// Each block maintains a private (shared memory) histogram to reduce atomic congestion
// when updating the global histogram array.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int rangeSize, unsigned int *histogram)
{
    // Allocate shared memory for the block-local histogram.
    // The size of the shared histogram is "rangeSize" (i.e. to - from + 1).
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to zero.
    // Each thread initializes one or more bins as needed.
    for (int i = threadIdx.x; i < rangeSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Use grid-stride loop to cover all characters in the input array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < inputSize; i += stride)
    {
        // Cast to unsigned char to correctly interpret the character's ordinal value.
        unsigned char ch = static_cast<unsigned char>(input[i]);
        int offset = (int)ch - from; // Calculate offset relative to low bound 'from'.
        // Only process characters that fall within [from, to].
        if (offset >= 0 && offset < rangeSize)
        {
            // Atomically increment the count in the block-local shared histogram.
            atomicAdd(&s_hist[offset], 1);
        }
    }
    __syncthreads();

    // One thread per bin (or several threads in a loop if needed) will combine the block histograms
    // into the global histogram using atomic operations.
    for (int i = threadIdx.x; i < rangeSize; i += blockDim.x)
    {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to run the histogram kernel.
// The function assumes that 'input' and 'histogram' are device pointers allocated with cudaMalloc.
// The 'histogram' device array should be of size (to - from + 1) and is zero-initialized by this function.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the size of the histogram range.
    int rangeSize = to - from + 1;

    // Choose a block size (threads per block). A common choice is 256 threads per block.
    int threadsPerBlock = 256;
    // Compute the number of blocks needed to cover the input.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    // If the input is empty, ensure that at least one block is launched.
    if (blocks == 0)
    {
        blocks = 1;
    }

    // Initialize the output histogram on the device to zero.
    cudaMemset(histogram, 0, rangeSize * sizeof(unsigned int));

    // Launch the histogram kernel with dynamically allocated shared memory equal to (rangeSize * sizeof(unsigned int)).
    histogram_kernel<<<blocks, threadsPerBlock, rangeSize * sizeof(unsigned int)>>>(input, inputSize, from, rangeSize, histogram);

    // Host-device synchronization is assumed to be handled by the caller.
}