#include <cuda_runtime.h>
#include <cassert>

// CUDA kernel to compute a histogram over a given character range.
// Each block uses shared memory (allocated as dynamic shared memory) to accumulate
// histogram counts for the range [from, to] (inclusive). After the block-level reduction,
// the results are atomically added to the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *globalHist)
{
    // Calculate number of bins of the histogram.
    int numBins = to - from + 1;

    // Dynamically allocated shared memory array for histogram bins.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared memory histogram.
    // Each thread initializes one or more bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process input with a grid-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < inputSize)
    {
        // Convert char to unsigned to avoid negative values.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the histogram range.
        if (c >= from && c <= to)
        {
            int bin = c - from;
            // Atomically increment the appropriate bin in shared memory.
            atomicAdd(&s_hist[bin], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Have each thread add one or more entries from shared memory into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        // Accumulate block's result into global histogram using atomic operations.
        atomicAdd(&globalHist[i], s_hist[i]);
    }
}

// Host function to launch the CUDA kernel for histogram computation.
// This function assumes that 'input' and 'histogram' are allocated on the device (via cudaMalloc)
// and that the caller handles any host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate the input parameter range.
    assert(from >= 0 && from < to && to <= 255);

    // Determine the number of bins in the histogram.
    int numBins = to - from + 1;

    // Zero out the global histogram memory.
    // It is assumed that 'histogram' has been allocated with at least numBins * sizeof(unsigned int).
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Define block and grid sizes.
    int blockSize = 256; // Typical block size.
    // Compute the minimum number of blocks needed to process 'inputSize' elements.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    // Ensure at least one block is launched.
    if (gridSize == 0)
        gridSize = 1;

    // Launch the histogram kernel.
    // The third parameter specifies the size of dynamic shared memory in bytes,
    // which is numBins * sizeof(unsigned int).
    histogram_kernel<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(input, inputSize, from, to, histogram);

    // Note: Any synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
}