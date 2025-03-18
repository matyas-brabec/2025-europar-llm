#include <cuda_runtime.h>

// Constants controlling kernel behavior.
// ITEMS_PER_THREAD: number of input chars processed by each thread.
// NUM_HISTO_COPIES: number of private copies of the histogram in shared memory per block,
// used to reduce shared memory bank conflicts.
#define ITEMS_PER_THREAD 16
#define NUM_HISTO_COPIES 8

// CUDA kernel to compute histogram for characters within a specified range [from, to].
// The kernel reads from the input array of chars and accumulates counts into a global histogram,
// but uses shared memory privatization (with multiple copies) to reduce contention.
__global__ void histogramKernel(const char *input, unsigned int inputSize,
                                unsigned int *globalHistogram, int from, int to)
{
    // Compute the number of histogram bins to process.
    const int histSize = to - from + 1;

    // Allocate shared memory for the privatized histograms.
    // The total shared memory used per block is NUM_HISTO_COPIES * histSize * sizeof(unsigned int).
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram copies to zero.
    int tid = threadIdx.x;
    int totalSharedElems = NUM_HISTO_COPIES * histSize;
    for (int i = tid; i < totalSharedElems; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread chooses one of the NUM_HISTO_COPIES private histograms to update.
    // Using modulus distributes threads over the available copies.
    int myPrivateIdx = threadIdx.x % NUM_HISTO_COPIES;

    // Each thread processes ITEMS_PER_THREAD characters from the input.
    // Compute the starting index for this thread's batch.
    int baseIdx = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x * ITEMS_PER_THREAD;

    // Loop over the ITEMS_PER_THREAD consecutive characters processed by this thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int index = baseIdx + i;
        if (index < inputSize)
        {
            // Read the character from global memory.
            char c = input[index];
            // Cast to unsigned char for proper numeric comparison.
            unsigned char uc = static_cast<unsigned char>(c);

            // Check if the character falls within the specified range.
            if (uc >= from && uc <= to)
            {
                // Compute the corresponding histogram bin.
                int bin = uc - from;
                // Update the thread's private histogram in shared memory.
                atomicAdd(&s_hist[myPrivateIdx * histSize + bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduction step: combine the multiple shared memory copies into a block-level histogram,
    // and then update the global histogram.
    // The reduction is parallelized over the histogram bins.
    for (int bin = threadIdx.x; bin < histSize; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum contributions from all private copies.
        for (int copy = 0; copy < NUM_HISTO_COPIES; copy++)
        {
            sum += s_hist[copy * histSize + bin];
        }
        // Atomically accumulate the block's bin count into the global histogram.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function to invoke the histogram kernel.
// 'input' and 'histogram' are device pointers allocated with cudaMalloc.
// 'inputSize' is the number of characters in the input array.
// 'from' and 'to' specify the inclusive range of character ordinals to histogram.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins.
    int histSize = to - from + 1;

    // Set the number of threads per block.
    const int threadsPerBlock = 256;
    // Each block processes a contiguous chunk of the input.
    int itemsPerBlock = threadsPerBlock * ITEMS_PER_THREAD;
    // Compute the number of blocks needed to cover the entire input.
    int numBlocks = (inputSize + itemsPerBlock - 1) / itemsPerBlock;

    // Determine the amount of dynamic shared memory needed per block.
    size_t sharedMemSize = sizeof(unsigned int) * NUM_HISTO_COPIES * histSize;

    // Launch the kernel.
    histogramKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(input, inputSize,
                                                                     histogram, from, to);
}