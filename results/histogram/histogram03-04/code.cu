#include <cuda_runtime.h>

// CUDA kernel to compute the histogram of an input text array restricted to a specified character range.
// The kernel uses shared memory for privatized histograms within each block to reduce the number of
// atomic updates to the global memory.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int histSize, unsigned int *globalHist)
{
    // Declare dynamically allocated shared memory for the block-local histogram.
    // The shared memory array has "histSize" entries (one per histogram bin).
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared memory histogram to 0.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array in a grid-stride loop.
    // Each thread handles multiple elements separated by gridDim.x * blockDim.x.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize)
    {
        // Read the character and treat it as unsigned to properly compare with `from`.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the requested range [from, from + histSize - 1] i.e., [from, to].
        if (c >= from && c < (from + histSize))
        {
            // Atomically increment the corresponding histogram bin in shared memory.
            atomicAdd(&s_hist[c - from], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // After processing, accumulate the block-local histogram into the global histogram.
    // Each thread processes multiple bins in the shared memory histogram.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        // Use atomic addition to merge the count from shared memory into global memory.
        atomicAdd(&globalHist[i], s_hist[i]);
    }
}

// Host function that invokes the CUDA kernel to compute the histogram of a text input.
// The input array "input" and output array "histogram" are allocated on device memory.
// "inputSize" is the total number of characters, and the histogram covers the range [from, to].
// The histogram array must have a size of (to - from + 1) unsigned integers.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int histSize = to - from + 1;

    // Optionally, clear the global histogram array on the device.
    // Note: This cudaMemset is asynchronous on the default stream.
    cudaMemset(histogram, 0, histSize * sizeof(unsigned int));

    // Define the execution configuration.
    // Using 256 threads per block is a common choice for modern GPUs.
    int blockSize = 256;
    // Compute the number of blocks to cover the inputSize; grid-stride loop ensures full coverage.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Calculate the required dynamic shared memory size (in bytes).
    size_t sharedMemSize = histSize * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, from, histSize, histogram);

    // Note: Any error checking and device synchronization is assumed to be handled by the caller.
}