// This code implements a CUDA kernel that computes the histogram of a text buffer for a
// user-specified continuous character range [from, to] (inclusive). Each thread block
// uses shared memory to create a private (per-block) histogram so that global atomic updates
// are greatly reduced. After all threads in the block have updated the shared memory,
// the block cooperatively adds its results to the global histogram.
// The run_histogram function launches the kernel with appropriate grid and block dimensions.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel to compute a histogram of input text restricted to a particular character range.
// Parameters:
//   input      - Pointer to an array of chars on device memory representing the text file.
//   global_hist- Pointer to an array of unsigned ints on device memory where the computed histogram will be stored.
//   inputSize  - Total number of characters in the input array.
//   from       - Lower bound (inclusive) of the char range.
//   to         - Upper bound (inclusive) of the char range.
// The histogram is computed for the bins corresponding to the values from "from" to "to" (both inclusive).
__global__ void histogram_kernel(const char *input, unsigned int *global_hist,
                                   unsigned int inputSize, int from, int to) {
    // Compute the number of histogram bins.
    int histSize = to - from + 1;

    // Declare shared memory for per-block histogram privatization.
    // The shared memory size is set during kernel launch.
    extern __shared__ unsigned int local_hist[];

    // Each thread initializes parts of the shared histogram to zero.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Use a grid-stride loop for full coverage of the input array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < inputSize) {
        // Read the character; treat it as unsigned to get a range of 0-255.
        unsigned char c = input[idx];
        // Only count characters within the desired [from, to] range.
        if (c >= from && c <= to) {
            // Increment the corresponding bin in shared memory with an atomic update.
            atomicAdd(&local_hist[c - from], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge the shared histogram into the global histogram.
    // Each thread processes a subset of histogram bins.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        atomicAdd(&global_hist[i], local_hist[i]);
    }
}

// Host function that launches the CUDA kernel to compute the histogram.
// Parameters:
//   input      - Pointer to the device memory holding the text data (allocated via cudaMalloc).
//   histogram  - Pointer to the device memory where the histogram will be stored (allocated via cudaMalloc).
//                The array should have at least (to - from + 1) unsigned int elements.
//   inputSize  - Number of characters in the input array.
//   from       - Lower bound (inclusive) of the histogram range.
//   to         - Upper bound (inclusive) of the histogram range.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Compute the number of bins needed.
    int histSize = to - from + 1;

    // Define CUDA kernel launch parameters.
    // Here we use 256 threads per block. Depending on the input size and GPU, this can be tuned.
    int blockSize = 256;
    // Compute the number of blocks needed to cover all input characters.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    // Set the shared memory size per block to hold the private histogram.
    size_t sharedMemSize = histSize * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}