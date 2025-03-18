#include <cuda_runtime.h>

// CUDA kernel to compute a histogram for characters in the range [from, to].
// The kernel uses shared memory for a partial histogram per block, then atomically
// accumulates the block histograms into the global histogram.
//
// Parameters:
//   input      - device pointer to the input text (array of chars)
//   histogram  - device pointer to the histogram output array; its size is (to - from + 1)
//   inputSize  - total number of characters in the input array
//   from, to   - define the inclusive range [from, to] of character values to include
//
// The kernel uses dynamic shared memory to hold a temporary histogram (one bin per character in the range).
// Each thread processes multiple input elements in a strided loop, updating the shared histogram using atomic operations.
// After processing, the shared histograms are merged into the global histogram using atomicAdd.
__global__ void histogram_kernel(const char* input, unsigned int* histogram, unsigned int inputSize, int from, int to) {
    // Compute the number of histogram bins (range = to - from + 1)
    int range = to - from + 1;

    // Declare shared memory histogram. The size of this array is provided at kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared memory histogram bins to zero; work is divided among threads.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Calculate the global index and stride length for this thread.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Process the input in a strided loop.
    while (idx < inputSize) {
        // Load a character from global memory.
        unsigned char ch = input[idx];

        // Check if the character is within the specified range.
        if (ch >= from && ch <= to) {
            // Compute the bin index relative to the 'from' value.
            // Atomically update the corresponding bin in shared memory.
            atomicAdd(&s_hist[ch - from], 1U);
        }
        idx += stride;
    }
    __syncthreads();

    // Accumulate each block's shared histogram into the global histogram.
    // Each thread processes bins in a strided fashion.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        // Only perform an atomic update if there were counts in this bin.
        if (s_hist[i] > 0)
            atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to invoke the histogram kernel on the GPU.
//
// Assumptions:
//   - 'input' is a device pointer to the text data (allocated via cudaMalloc).
//   - 'histogram' is a device pointer to an array of (to - from + 1) unsigned integers (allocated via cudaMalloc).
//   - 'inputSize' is the number of characters in the input array.
//   - Host-device synchronization, error checking, and data transfers (if necessary) are handled by the caller.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
        // Define the number of threads per block.
    int blockSize = 256;
    // Compute the number of blocks required to cover the entire input.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Compute the number of histogram bins.
    int range = to - from + 1;
    // Determine the shared memory size (in bytes) required per block.
    size_t sharedMemSize = range * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);

    // Note: Synchronization (e.g., cudaDeviceSynchronize) is assumed to be handled by the caller.
}
