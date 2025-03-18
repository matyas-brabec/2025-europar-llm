#include <cuda_runtime.h>

// Constant: Number of private histogram copies per thread block.
// We use 32 copies so that threads within the same warp (32 lanes) access distinct copies in shared memory.
#define NUM_HIST_COPIES 32

// Constant: number of input items per thread.
// Chosen as 16 for high throughput on modern NVIDIA GPUs.
const int itemsPerThreadDefault = 16;

// CUDA kernel that computes a histogram for a text array over a specified char range [from, to].
// The input is processed in chunks of 'itemsPerThread' per thread.
// To reduce contention, each thread uses a private copy of the histogram in shared memory.
// We allocate NUM_HIST_COPIES copies (with padding to avoid bank conflicts), and assign each thread a copy determined by (threadIdx.x % NUM_HIST_COPIES).
// Finally, the copies are reduced and the result is accumulated into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *g_hist, unsigned int inputSize,
                                 int from, int to, int itemsPerThread) {
    // Compute the number of histogram bins.
    int histBins = to - from + 1;
    // Compute padded bin count: round up histBins to a multiple of 32 to avoid bank conflicts in shared memory.
    int paddedBins = ((histBins + 31) / 32) * 32;

    // Allocate shared memory: an array of NUM_HIST_COPIES histograms, each of size paddedBins.
    // The layout is: s_hist[copy_index * paddedBins + bin]
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram to zero.
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int totalSharedEntries = NUM_HIST_COPIES * paddedBins;
    for (int i = tid; i < totalSharedEntries; i += blockSize) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes 'itemsPerThread' input characters starting at:
    int start = global_tid * itemsPerThread;

    // Process a contiguous chunk of 'itemsPerThread' characters.
    for (int i = 0; i < itemsPerThread; i++) {
        int idx = start + i;
        if (idx < inputSize) {
            char c = input[idx];
            int bin = (int)c - from; // Map the character to a bin index relative to 'from'
            if (bin >= 0 && bin < histBins) {
                // Select a histogram copy to update.
                // Using (threadIdx.x % NUM_HIST_COPIES) ensures that threads within the same warp
                // access different shared memory banks, avoiding bank conflicts.
                int copyIndex = threadIdx.x % NUM_HIST_COPIES;
                // Compute the index into the shared memory histogram.
                int offset = copyIndex * paddedBins + bin;
                // Atomically increment the bin count in shared memory.
                atomicAdd(&s_hist[offset], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the NUM_HIST_COPIES private histograms for each bin and accumulate into the global histogram.
    // Each thread in the block cooperatively processes a subset of the histogram bins.
    for (int bin = tid; bin < histBins; bin += blockSize) {
        unsigned int sum = 0;
        // Sum contributions from all copies for this bin.
        for (int copy = 0; copy < NUM_HIST_COPIES; copy++) {
            sum += s_hist[copy * paddedBins + bin];
        }
        // Atomically add the block's result to the global histogram.
        atomicAdd(&g_hist[bin], sum);
    }
    // End of kernel.
}

// Host function that launches the histogram kernel.
// 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// 'inputSize' is the number of characters in 'input'.
// The histogram is computed over the range [from, to], and the resulting histogram has (to - from + 1)
// bins, where each bin count corresponds to the occurrence count for a character (ASCII value = bin index + from).
// Any host-device synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Use the default number of items per thread.
    const int itemsPerThread = itemsPerThreadDefault;

    // Determine the total number of threads needed.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Choose block size. 256 threads per block is typical for high throughput.
    const int blockSize = 256;
    // Calculate grid size (number of blocks).
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // Compute number of histogram bins.
    int histBins = to - from + 1;
    // Compute padded bin count for shared memory allocation.
    int paddedBins = ((histBins + 31) / 32) * 32;
    // Total shared memory required: NUM_HIST_COPIES copies * paddedBins * sizeof(unsigned int)
    size_t sharedMemSize = NUM_HIST_COPIES * paddedBins * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to, itemsPerThread);
}