#include <cuda_runtime.h>

// Constant definitions optimized for modern NVIDIA GPUs such as A100/H100.
// ITEMS_PER_THREAD controls how many input characters each thread processes.
constexpr int ITEMS_PER_THREAD = 8;    // Each thread processes 8 characters.
constexpr int NUM_HISTO_COPIES = 8;      // Multiple histogram copies per block to reduce shared memory bank conflicts.

// CUDA kernel that computes a restricted-range histogram from the input text.
// The histogram is computed only for characters in the range [from, to].
// The output globalHistogram is an array of (to - from + 1) unsigned ints where each position i
// corresponds to character (i + from).
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *globalHistogram)
{
    // Compute the number of bins corresponding to the range [from, to].
    int numBins = to - from + 1;
    
    // Allocate shared memory for private histograms.
    // We create NUM_HISTO_COPIES copies of the histogram in shared memory to avoid bank conflicts.
    // The total shared memory size is NUM_HISTO_COPIES * numBins unsigned integers.
    extern __shared__ unsigned int s_hist[];
    
    // Total number of elements in the shared histogram copies.
    int totalHistElements = NUM_HISTO_COPIES * numBins;
    
    // Initialize the shared memory histograms to zero in parallel.
    for (int i = threadIdx.x; i < totalHistElements; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Calculate the global thread index and its corresponding starting index in the input array.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = globalThreadId * ITEMS_PER_THREAD;
    
    // Select one of the multiple shared histogram copies based on threadIdx.x.
    int myCopy = threadIdx.x % NUM_HISTO_COPIES;
    
    // Each thread processes ITEMS_PER_THREAD characters from the input array.
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        int idx = startIdx + i;
        if (idx < inputSize)
        {
            // Read the character from global memory.
            unsigned char ch = static_cast<unsigned char>(input[idx]);
            // Check if the character is within the designated range.
            if (ch >= from && ch <= to)
            {
                int bin = ch - from;
                // Atomically update the private shared histogram.
                atomicAdd(&s_hist[myCopy * numBins + bin], 1);
            }
        }
    }
    __syncthreads();
    
    // Reduce the multiple shared histogram copies into a single result.
    // Each thread iterates over bins in a strided way.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int sum = 0;
        for (int copy = 0; copy < NUM_HISTO_COPIES; ++copy)
        {
            sum += s_hist[copy * numBins + bin];
        }
        // Atomically accumulate the block's partial histogram into the global histogram.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function to configure and launch the CUDA histogram kernel.
// Parameters:
//   input         : Device pointer to the input text (array of chars).
//   histogram     : Device pointer to the output histogram (array of unsigned ints).
//                   The caller must allocate (to - from + 1) unsigned ints.
//   inputSize     : Number of characters in the input buffer.
//   from, to      : The inclusive range of character ordinal values to include in the histogram.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins.
    int numBins = to - from + 1;
    
    // Zero out the global histogram to ensure correct atomic accumulation.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));
    
    // Choose a block size that is optimized for modern GPUs.
    int threadsPerBlock = 256;  // Selected block size.
    
    // Each thread processes ITEMS_PER_THREAD characters.
    // Compute the number of blocks needed to cover the input.
    int blocks = (inputSize + threadsPerBlock * ITEMS_PER_THREAD - 1) / (threadsPerBlock * ITEMS_PER_THREAD);
    
    // Calculate the required dynamic shared memory size.
    size_t sharedMemSize = NUM_HISTO_COPIES * numBins * sizeof(unsigned int);
    
    // Launch the histogram CUDA kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);
}