#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute a histogram over a given range [from, to] of character values.
// Each block creates a private histogram in shared memory and then atomically merges it
// into the global histogram.
__global__ void histogramKernel(const char *input, unsigned int inputSize, unsigned int *globalHist, int from, int to)
{
    // Calculate the number of histogram bins required.
    int nbins = to - from + 1;

    // Declare dynamic shared memory for the per-block histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes a subset of the histogram bins in shared memory.
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index and the overall stride for the grid-stride loop.
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process the input text in a grid-stride loop.
    // For each character, if it falls within the specified range,
    // atomically increment the corresponding bin in shared memory.
    for (int i = globalId; i < inputSize; i += stride)
    {
        unsigned char ch = input[i];  // Use unsigned char to correctly interpret the value.
        if (ch >= from && ch <= to)
        {
            atomicAdd(&s_hist[ch - from], 1);
        }
    }
    __syncthreads();

    // Merge the shared histogram into the global histogram.
    // Each thread handles a subset of the bins.
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    {
        atomicAdd(&globalHist[i], s_hist[i]);
    }
}

// Host function that prepares and launches the histogram kernel.
// Assumptions:
// - 'input' and 'histogram' are device pointers (allocated with cudaMalloc).
// - 'inputSize' is the number of characters in the input buffer.
// - 'from' and 'to' define the inclusive range of character values to be histogrammed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins for the histogram.
    int nbins = to - from + 1;

    // Initialize the global histogram to zero.
    // This is important since the kernel uses atomicAdd to accumulate counts.
    cudaMemset(histogram, 0, nbins * sizeof(unsigned int));

    // Define the execution configuration.
    // Use 256 threads per block, which is a common choice for modern GPUs.
    int threadsPerBlock = 256;
    // Determine the number of blocks needed to cover the input.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    // Note: For very small inputs, blocks will be 1. For larger inputs, this schedule ensures coverage.

    // Launch the kernel.
    // The third kernel parameter allocates dynamic shared memory: one unsigned int per histogram bin.
    histogramKernel<<<blocks, threadsPerBlock, nbins * sizeof(unsigned int)>>>(input, inputSize, histogram, from, to);
}