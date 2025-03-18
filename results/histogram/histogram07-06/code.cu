// This CUDA implementation computes a histogram for a restricted character range [from, to]
// for an input text stored as an array of chars. Each output histogram bin corresponds to a
// character value (bin i corresponds to character value i + from). The kernel uses shared memory
// privatization with 32 copies (one per shared memory bank) to reduce bank conflicts.
//
// Each thread processes ITEMS_PER_THREAD characters from the global input and accumulates counts into
// its dedicated shared memory copy. After processing, the 32 copies are reduced per bin and atomically
// merged into the global histogram.
//
// The host function run_histogram configures the kernel launch parameters based on the input size.
// It assumes that 'input' and 'histogram' have been allocated on the device (using cudaMalloc)
// and that any necessary host-device synchronization is handled by the caller.

#include <cuda_runtime.h>

#define ITEMS_PER_THREAD 8  // Default number of input chars processed per thread; adjust as needed.

// CUDA kernel to compute the histogram over a restricted character range.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to) {
    // Compute the number of histogram bins = (to - from + 1).
    int range = to - from + 1;

    // Declare the dynamic shared memory array.
    // It is organized as 32 copies of the histogram bins.
    // The bin i in copy c is stored at shared_hist[i * 32 + c].
    extern __shared__ unsigned int shared_hist[];

    // Each thread selects one of the 32 copies based on its threadIdx.x modulo 32.
    int copyIndex = threadIdx.x % 32;

    // Initialize all shared memory bins for this block.
    int totalSharedBins = range * 32;
    for (int i = threadIdx.x; i < totalSharedBins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting global index for this thread's portion of work.
    // Each thread processes ITEMS_PER_THREAD consecutive chars.
    int globalStart = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index = globalStart + i;
        if (index < inputSize) {
            unsigned char val = input[index];
            // Only process characters within the [from, to] range.
            if (val >= from && val <= to) {
                int bin = val - from;
                // Update the corresponding bin in the thread's private shared memory copy.
                atomicAdd(&shared_hist[bin * 32 + copyIndex], 1);
            }
        }
    }
    __syncthreads();

    // Reduction: Sum the values from all 32 copies for each bin and update the global histogram.
    // Each thread handles a subset of the bins.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int j = 0; j < 32; j++) {
            sum += shared_hist[bin * 32 + j];
        }
        // Atomically add the partial sum from this block to the global histogram.
        atomicAdd(&global_hist[bin], sum);
    }
}

// Host function to run the histogram kernel.
// Assumes the 'input' and 'histogram' pointers refer to device memory allocated via cudaMalloc.
// The 'inputSize' parameter is the total number of characters in the input array.
// The 'from' and 'to' parameters define the inclusive range for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Determine the number of histogram bins.
    int range = to - from + 1;

    // Configure CUDA kernel launch parameters.
    // We choose a block size of 256 threads.
    int blockSize = 256;
    // Each thread processes ITEMS_PER_THREAD characters.
    // Total number of threads required is computed by rounding up inputSize/ITEMS_PER_THREAD.
    int totalThreads = (inputSize + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
    // Calculate the grid size.
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // Initialize the global histogram to zero.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Compute the dynamic shared memory size needed per block.
    // There are 32 copies of the histogram, each with 'range' integers.
    size_t sharedMemorySize = range * 32 * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemorySize>>>(input, histogram, inputSize, from, to);
}