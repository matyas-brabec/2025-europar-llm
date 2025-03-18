#include <cuda_runtime.h>

// Define the number of input items each thread will process.
// This value is chosen to amortize kernel overhead over many input elements,
// and 128 is a good default for modern NVIDIA GPUs when the input size is large.
#define ITEMS_PER_THREAD 128

// CUDA kernel to compute a histogram on a restricted character range.
// The kernel uses shared memory with 32 privatized copies of the histogram per block to avoid bank conflicts.
// Each thread processes ITEMS_PER_THREAD characters from the input array.
__global__ void histogramKernel(const char *input, unsigned int *globalHistogram, 
                                unsigned int inputSize, int from, int to)
{
    // Compute histogram range size.
    int histSize = to - from + 1;

    // Declare shared memory histogram array.
    // There will be 32 copies of the histogram, each copy holding 'histSize' bins.
    // They are interleaved so that bin i of copy c is stored at s_hist[i * 32 + c].
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes its portion of the shared histogram to zero.
    int tid = threadIdx.x;
    int blockThreads = blockDim.x;
    int totalSharedElements = histSize * 32;
    for (int i = tid; i < totalSharedElements; i += blockThreads)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting global index for the block.
    // Each block processes blockDim.x * ITEMS_PER_THREAD items.
    int blockOffset = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;

    // Each thread is assigned a specific copy (0 <= copyId < 32) to update.
    int copyId = tid % 32;

    // Process ITEMS_PER_THREAD input characters per thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int index = blockOffset + tid + i * blockDim.x;
        if (index < inputSize)
        {
            // Read the character. Cast to unsigned char to ensure proper range value.
            unsigned char ch = input[index];
            // If the character is within the specified range, update the local shared histogram.
            if (ch >= from && ch <= to)
            {
                // Compute index into shared histogram:
                // For a character value 'ch', its corresponding bin index is (ch - from).
                // Each bin has 32 entries; we use the entry at offset (copyId) for this thread.
                int pos = (ch - from) * 32 + copyId;
                atomicAdd(&s_hist[pos], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 shared memory histogram copies into the global histogram.
    // Each thread processes some histogram bins.
    for (int bin = tid; bin < histSize; bin += blockThreads)
    {
        unsigned int sum = 0;
        // Sum over the 32 copies for current bin.
        for (int c = 0; c < 32; c++)
        {
            sum += s_hist[bin * 32 + c];
        }
        // Atomically add the block's bin sum to the global histogram.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function to launch the histogram kernel.
// Assumes that 'input' and 'histogram' arrays are already allocated on the device (via cudaMalloc).
// The 'inputSize' parameter specifies the number of characters in 'input'.
// The histogram will cover the continuous range of characters from 'from' to 'to' (inclusive).
// Caller is responsible for host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, 
                   unsigned int inputSize, int from, int to)
{
    // Compute histogram range size.
    int histSize = to - from + 1;

    // Initialize the global histogram to zero.
    cudaMemset(histogram, 0, histSize * sizeof(unsigned int));

    // Define kernel configuration.
    const int itemsPerThread = ITEMS_PER_THREAD; // Number of input chars processed per thread.
    int blockSize = 256;  // Number of threads per block.
    
    // Compute total number of threads needed so that each processes 'itemsPerThread' characters.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    
    // Calculate required shared memory size per block.
    // There are 'histSize' histogram bins and 32 copies per block.
    size_t sharedMemSize = histSize * 32 * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}