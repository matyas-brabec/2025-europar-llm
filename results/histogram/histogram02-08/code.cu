// This CUDA code computes a histogram for a specified continuous character range [from, to]
// in a text input stored in device memory. Each block builds a partial histogram in shared
// memory to reduce contention on global memory, and then updates the global histogram via atomic operations.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/// @FIXED
/// extern "C" {

// The CUDA kernel that computes the histogram for characters in the range [from, to].
// Each block uses dynamically allocated shared memory to build a per-block histogram.
__global__ void histogramKernel(const char *input, unsigned int inputSize, unsigned int *histogram, int from, int to)
{
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Allocate dynamically sized shared memory for the block-level histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to zero.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute a global index and a stride that covers the full input.
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Iterate over the input in a strided manner.
    for (int i = globalIdx; i < inputSize; i += stride) {
        // Load the character from global memory.
        unsigned char c = input[i]; // use unsigned to correctly interpret the character value
        // If the character is within the specified range, update the block-level histogram.
        if (c >= from && c <= to) {
            int bin = c - from;
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // Each thread reduces part of the shared histogram to the global histogram.
    // Atomic operations are required because multiple blocks update the same global bins.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function that configures and invokes the CUDA kernel for computing the histogram.
// Parameters:
// - input: Device pointer to the text input array.
// - histogram: Device pointer to the output histogram array.
// - inputSize: Number of characters in the input array.
// - from: Lower bound (inclusive) of the character range to histogram.
// - to: Upper bound (inclusive) of the character range to histogram.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins.
    int range = to - from + 1;

    // If there is no input, ensure the histogram is cleared and return.
    if (inputSize == 0) {
        cudaMemset(histogram, 0, range * sizeof(unsigned int));
        return;
    }

    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks required to cover the input data.
    int numBlocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Zero out the global histogram bins before kernel execution.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Calculate the dynamic shared memory size required for the kernel execution.
    size_t sharedMemorySize = range * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogramKernel<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(input, inputSize, histogram, from, to);
}

/// @FIXED
/// } // extern "C"
