#include <cuda_runtime.h>

// CUDA kernel to compute the histogram for characters in the range [from, to].
// Each block computes a local histogram in shared memory and then accumulates it
// into the global histogram using atomic operations.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 unsigned int *global_hist, int from, int to) {
    // Compute the number of histogram bins.
    int range = to - from + 1;

    // Declare dynamically allocated shared memory for the block's histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram to 0.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input buffer using a grid-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize) {
        // Cast the char to unsigned char to get a correct ordinal value.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if the character is within the [from, to] range.
        if (c >= from && c <= to) {
            // Use atomic add to safely update the shared histogram bin.
            atomicAdd(&s_hist[c - from], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Each thread updates the global histogram with its block's partial results.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&global_hist[i], s_hist[i]);
    }
}

// Host function to launch the CUDA kernel for histogram computation.
// 'input' and 'histogram' are device pointers allocated using cudaMalloc.
// 'inputSize' is the number of characters in the input buffer.
// 'from' and 'to' define the inclusive range of character ordinal values to histogram.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to) {
    // Compute the number of histogram bins.
    int histRange = to - from + 1;

    // Zero the global histogram array on the device.
    cudaMemset(histogram, 0, histRange * sizeof(unsigned int));

    // Define kernel launch configuration.
    // Use 256 threads per block as an optimized configuration.
    int blockSize = 256;
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Calculate the size of shared memory needed (one unsigned int per histogram bin).
    size_t sharedMemSize = histRange * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);
}