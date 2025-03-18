#include <cuda_runtime.h>
#include <stdio.h>

// Define the number of items each thread processes.
// An ITEMS_PER_THREAD value of 256 is a good default on modern GPUs for large inputs.
#define ITEMS_PER_THREAD 256

// Define the number of shared histogram copies per block to reduce bank conflicts.
#define NUM_HISTO_COPIES 8

// CUDA kernel to compute a histogram for characters in the continuous range [from, to].
// The kernel uses shared memory with multiple copies to privatize histogram updates.
// Each thread processes ITEMS_PER_THREAD input elements and increments its designated copy.
// After processing, the copies are reduced and atomically merged into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int numBins = to - from + 1;

    // Declare dynamic shared memory.
    // Organized as NUM_HISTO_COPIES copies of histograms, each with 'numBins' bins.
    extern __shared__ unsigned int s_hist[];

    // Total number of shared memory elements allocated for histograms.
    int totalSharedElems = NUM_HISTO_COPIES * numBins;

    // Initialize shared memory histogram bins to zero.
    // Distribute work among threads.
    for (int i = threadIdx.x; i < totalSharedElems; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread chooses one of the histogram copies to update to reduce bank conflicts.
    int histCopyIndex = threadIdx.x % NUM_HISTO_COPIES;

    // Calculate this thread's starting index in the global input array.
    // Each block processes (blockDim.x * ITEMS_PER_THREAD) array elements.
    int baseIndex = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;

    // Loop over ITEMS_PER_THREAD elements per thread, striding by blockDim.x.
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = baseIndex + i * blockDim.x;
        if (idx < inputSize) {
            // Read the input character as an unsigned value.
            unsigned char c = static_cast<unsigned char>(input[idx]);
            // Check if the character is within the desired range.
            if (c >= from && c <= to) {
                int bin = c - from;
                // Atomically increment the appropriate bin in the thread’s shared histogram copy.
                atomicAdd(&s_hist[histCopyIndex * numBins + bin], 1);
            }
        }
    }
    __syncthreads();

    // Each thread now reduces the partial histograms from the multiple copies for a subset of bins.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int copy = 0; copy < NUM_HISTO_COPIES; copy++) {
            sum += s_hist[copy * numBins + bin];
        }
        // Atomically add the block result for this bin to the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function that invokes the CUDA histogram kernel.
// 'input' and 'histogram' are device pointers to properly allocated memory.
// 'inputSize' is the number of characters in the input array.
// The histogram is computed only for characters in the continuous range [from, to].
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int numBins = to - from + 1;

    // Set the block size (number of threads per block).
    int threadsPerBlock = 256;
    // Calculate how many blocks are needed so that each thread processes ITEMS_PER_THREAD input elements.
    int blocks = (inputSize + threadsPerBlock * ITEMS_PER_THREAD - 1) / (threadsPerBlock * ITEMS_PER_THREAD);

    // Compute the shared memory size required per block:
    // NUM_HISTO_COPIES copies times numBins bins per copy times size of each unsigned int.
    size_t sharedMemSize = NUM_HISTO_COPIES * numBins * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}