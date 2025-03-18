#include <cuda_runtime.h>

// Define constant itemsPerThread which specifies how many input chars each thread will process.
// A default value of 128 is chosen, providing a good balance between load distribution and efficiency on modern GPUs.
static const int itemsPerThread = 128;

// CUDA kernel to compute a histogram for a specific character range [from, to].
// The input is a device array of chars and the histogram output is a device array of unsigned ints,
// of size (to - from + 1) where each index i corresponds to the count of character with ordinal (i + from).
// The kernel uses shared memory for block-level privatization of the histogram.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Determine the size of the histogram range.
    int histSize = to - from + 1;
    
    // Allocate a dynamically sized shared memory array for the histogram.
    // The size is provided at kernel launch.
    extern __shared__ unsigned int sharedHist[];
    
    // Initialize the shared histogram to 0.
    // Each thread initializes one or more bins to cover the entire range.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread ID.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes itemsPerThread characters.
    int startIndex = globalThreadId * itemsPerThread;

    // Process the assigned segment.
    for (int i = 0; i < itemsPerThread; i++)
    {
        int idx = startIndex + i;
        if (idx < inputSize)
        {
            // Cast to unsigned char to get a value within 0..255.
            unsigned char ch = static_cast<unsigned char>(input[idx]);
            // Only process characters within the target range.
            if (ch >= from && ch <= to)
            {
                // Use atomicAdd to update the bin in shared memory.
                atomicAdd(&sharedHist[ch - from], 1);
            }
        }
    }
    __syncthreads();

    // One thread (or a group of threads) merges the block's shared histogram into the global histogram.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        atomicAdd(&histogram[i], sharedHist[i]);
    }
}

// Host function that configures and launches the CUDA kernel.
// It assumes that 'input' and 'histogram' are pointers to device memory allocated via cudaMalloc.
// The 'inputSize' parameter is the number of characters in the 'input' array.
// The 'from' and 'to' parameters specify the inclusive range of characters to be histogrammed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute histogram range size.
    int histSize = to - from + 1;

    // Define threads per block.
    int threadsPerBlock = 256; // Chosen to be efficient on modern GPUs.

    // Calculate total number of threads needed.
    // Each thread processes 'itemsPerThread' characters.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Determine grid size (number of blocks).
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel.
    // The third parameter defines the amount of dynamic shared memory needed: one unsigned int per histogram bin.
    histogram_kernel<<<numBlocks, threadsPerBlock, histSize * sizeof(unsigned int)>>>(input, histogram, inputSize, from, to);
}