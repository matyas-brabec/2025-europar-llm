#include <cuda_runtime.h>

// Define the number of input items processed per thread.
// This constant is chosen to balance occupancy and memory throughput on modern NVIDIA GPUs.
#define ITEMS_PER_THREAD 128

// CUDA kernel to compute the histogram of characters within a given range [from, to].
// Each block uses 32 private copies of the histogram in shared memory (to avoid bank conflicts)
// stored in a strided layout so that histogram copy c has its entries at s_hist[bin*32 + c].
// The global input is processed in chunks, with each thread processing ITEMS_PER_THREAD characters.
__global__
void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to)
{
    // Compute number of bins for the histogram.
    const int histSize = to - from + 1;

    // Declare externally allocated shared memory for histogram copies.
    // Each of 32 copies has 'histSize' bins stored with a stride of 32.
    extern __shared__ unsigned int s_hist[];
    // Total shared memory size (in elements): histSize * 32.
    const int total_shared = histSize * 32;

    // Initialize shared memory histogram to zero using a grid-stride loop.
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the base index in the global input array for this block.
    // Each block processes blockDim.x * ITEMS_PER_THREAD items.
    unsigned int blockStart = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;

    // Each thread picks one of the 32 histogram copies based on its thread index to avoid bank conflicts.
    int local_copy = threadIdx.x % 32;

    // Each thread processes ITEMS_PER_THREAD characters.
    // The global index for each processed element is computed as:
    // index = blockStart + threadIdx.x + i * blockDim.x for i in [0, ITEMS_PER_THREAD)
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        unsigned int index = blockStart + threadIdx.x + i * blockDim.x;
        if (index < inputSize)
        {
            // Read the character from global memory.
            unsigned char c = input[index];
            // Verify if the character is within the histogram range.
            if (c >= from && c <= to)
            {
                int bin = c - from;
                // Atomically increment the appropriate bin in the thread's shared histogram copy.
                // The bin is located at offset: (bin * 32 + local_copy).
                atomicAdd(&s_hist[bin * 32 + local_copy], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 partial histograms in shared memory and update global histogram.
    // Here, a single thread (threadIdx.x == 0) aggregates the bins from all 32 copies.
    if (threadIdx.x == 0)
    {
        for (int bin = 0; bin < histSize; bin++)
        {
            unsigned int sum = 0;
            // Sum over all 32 copies.
            for (int copy = 0; copy < 32; copy++)
            {
                sum += s_hist[bin * 32 + copy];
            }
            // Atomically add the block's bin total to the global histogram.
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

// Host function to run the histogram CUDA kernel.
// The 'input' buffer and the output 'histogram' buffer are assumed to be allocated with cudaMalloc.
// The caller is responsible for host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Choose a typical block size optimized for modern NVIDIA GPUs.
    const int threadsPerBlock = 256;
    // Each block processes threadsPerBlock * ITEMS_PER_THREAD characters.
    int blocks = (inputSize + threadsPerBlock * ITEMS_PER_THREAD - 1) / (threadsPerBlock * ITEMS_PER_THREAD);

    // Compute the number of histogram bins.
    int histSize = to - from + 1;
    // Compute the required shared memory size: 32 copies of the histogram, each with 'histSize' bins.
    size_t sharedMemSize = histSize * 32 * sizeof(unsigned int);

    // Launch the CUDA kernel to compute the histogram.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}