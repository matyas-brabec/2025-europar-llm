#include <cuda_runtime.h>

// Constant that controls how many items each thread processes.
// Chosen as 128 to balance memory bandwidth and thread occupancy on modern GPUs.
#define ITEMS_PER_THREAD 128

// CUDA kernel that computes the histogram for a specified character range [from, to].
// Each block uses shared memory to privately accumulate counts and later updates the global histogram.
__global__ void histogramKernel(const char *input, unsigned int *histogram,
                                unsigned int inputSize, int from, int to) {
    // Calculate the number of histogram bins for the range [from, to].
    int numBins = to - from + 1;

    // Declare shared memory for histogram privatization.
    // The shared histogram size is determined dynamically at kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to zero.
    // Each thread strides across the shared array to initialize multiple bins if needed.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting index for this thread.
    // Each block processes blockDim.x * ITEMS_PER_THREAD elements.
    unsigned int base = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;

    // Each thread processes ITEMS_PER_THREAD elements in a strided manner.
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        unsigned int index = base + i * blockDim.x;
        if (index < inputSize) {
            // Read the character.
            unsigned char c = input[index];
            // Check if the character is within the specified range.
            if (c >= from && c <= to) {
                // Compute the corresponding bin index.
                int bin = c - from;
                // Use atomic addition in shared memory to avoid race conditions.
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // After accumulation in shared memory is complete,
    // each thread copies part of the shared histogram into the global histogram array.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        // Use atomic addition to update the global histogram from the block-level result.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function that launches the CUDA kernel to compute the histogram.
// Parameters:
//   input      - device pointer to the input text array
//   histogram  - device pointer to the output histogram array (preallocated)
//   inputSize  - number of characters in the input array
//   from, to   - specify the inclusive character range [from, to] for the histogram.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Define block size (number of threads per block).
    const int blockSize = 256;

    // Calculate grid size: each block processes blockSize * ITEMS_PER_THREAD characters.
    int gridSize = (inputSize + blockSize * ITEMS_PER_THREAD - 1) / (blockSize * ITEMS_PER_THREAD);

    // Determine the number of bins for the requested character range.
    int numBins = to - from + 1;
    // Compute the size of dynamically allocated shared memory in bytes.
    size_t sharedMemSize = numBins * sizeof(unsigned int);

    // Launch the histogram kernel.
    // The histogram kernel uses shared memory to privatize partial histograms per block.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);

    // Note: Any necessary device-to-host synchronization or error checking is expected to be handled by the caller.
}