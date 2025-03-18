#include <cuda_runtime.h>

// The number of input items (chars) processed per thread.
// This value is selected to optimize throughput on modern GPUs.
constexpr int itemsPerThread = 128;

// Number of copies in shared memory (for privatization) to avoid bank conflicts.
// Each thread uses one copy determined by its threadIdx.x modulo NUM_COPIES.
constexpr int NUM_COPIES = 8;

// CUDA kernel to compute histogram over a restricted character range.
// "input"      : pointer to device memory containing plain text (array of chars).
// "inputSize"  : number of characters in the input array.
// "global_histogram" : device memory array where histogram is accumulated.
// "from" and "to" specify the inclusive range from which characters are counted.
__global__
void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *global_histogram, int from, int to)
{
    // Compute the size of the histogram range.
    const int range = to - from + 1;

    // Allocate shared memory for histogram privatization.
    // We use NUM_COPIES copies, each with "range" bins, to reduce bank conflicts.
    extern __shared__ unsigned int sHistogram[]; // total elements: NUM_COPIES * range

    int tid = threadIdx.x;
    int totalSharedElements = NUM_COPIES * range;

    // Initialize shared memory histogram copies to zero.
    // Each thread initializes multiple indices in steps of blockDim.x.
    for (int i = tid; i < totalSharedElements; i += blockDim.x)
    {
        sHistogram[i] = 0;
    }
    __syncthreads();

    // Compute the unique global thread index.
    int globalThreadId = blockIdx.x * blockDim.x + tid;
    // Each thread processes "itemsPerThread" characters, starting at index:
    int base = globalThreadId * itemsPerThread;

    // Each thread picks one shared memory histogram copy to update,
    // chosen as the remainder of its thread index modulo NUM_COPIES.
    int copy = tid % NUM_COPIES;
    unsigned int* localHist = sHistogram + copy * range;

    // Process the input items assigned to this thread.
    #pragma unroll
    for (int i = 0; i < itemsPerThread; i++)
    {
        int pos = base + i;
        if (pos < inputSize)
        {
            // Read character and convert to unsigned to correctly handle values >= 128.
            unsigned char c = static_cast<unsigned char>(input[pos]);
            // Check if the character is within the specified range.
            if (c >= from && c <= to)
            {
                int bin = c - from;
                // Update the shared memory histogram using an atomic operation.
                atomicAdd(&localHist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduction: Sum up the private histogram copies for each histogram bin.
    // We parallelize this stage by having threads cooperate.
    for (int bin = tid; bin < range; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Accumulate contributions from all copies for the current bin.
        for (int copyIdx = 0; copyIdx < NUM_COPIES; copyIdx++)
        {
            sum += sHistogram[copyIdx * range + bin];
        }
        // Atomically accumulate the block's result into the global histogram.
        atomicAdd(&global_histogram[bin], sum);
    }
}

// Host function that launches the CUDA histogram kernel.
// "input"    : pointer to device memory containing the text (array of chars).
// "histogram": pointer to device memory where the resulting histogram is stored.
//              The histogram array length must equal "to - from + 1".
// "inputSize": total number of characters in the input buffer.
// "from" and "to" specify the inclusive range of character ordinal values to process.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the histogram range.
    int range = to - from + 1;

    // Define the execution configuration.
    constexpr int threadsPerBlock = 256; // typical optimal block size for modern GPUs.
    // Total number of threads needed: one thread processes itemsPerThread elements.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate the required shared memory size in bytes for the kernel.
    size_t sharedMemSize = NUM_COPIES * range * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, histogram, from, to);
}