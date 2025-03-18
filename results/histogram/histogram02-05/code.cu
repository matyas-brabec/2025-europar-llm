#include <cuda_runtime.h>

// CUDA kernel that computes a histogram over a specified char range using shared memory.
// Each block accumulates a partial histogram in shared memory, then atomically adds the results to global memory.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to) {
    // Number of bins in the histogram (for characters in range from to to inclusive).
    int num_bins = to - from + 1;

    // Allocate dynamic shared memory for the block's private histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared histogram.
    int tid = threadIdx.x;
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array using a grid-stride loop.
    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = global_tid; i < inputSize; i += stride) {
        // Load the current character and treat it as an unsigned value.
        unsigned char ch = static_cast<unsigned char>(input[i]);
        // Update the shared histogram only if within the specified range.
        if (ch >= from && ch <= to) {
            int bin = ch - from;
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // Each thread accumulates its portion of the shared histogram into global memory.
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&global_hist[i], s_hist[i]);
    }
}

// Host function that configures and launches the CUDA kernel to compute the histogram.
// The 'input' and 'histogram' arrays are device memory already allocated with cudaMalloc.
// The 'histogram' array must have (to - from + 1) elements.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Calculate the number of bins for the specified character range.
    int num_bins = to - from + 1;

    // Zero out the global histogram memory on device to ensure proper accumulation.
    cudaMemset(histogram, 0, num_bins * sizeof(unsigned int));

    // Define a suitable block size for modern GPUs (e.g., 256 threads per block).
    const int threadsPerBlock = 256;
    // Determine the grid size to fully cover the input using a grid-stride loop.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) blocks = 1;

    // Launch the kernel.
    // The third parameter specifies the dynamic shared memory size: one unsigned int per bin.
    histogram_kernel<<<blocks, threadsPerBlock, num_bins * sizeof(unsigned int)>>>(input, histogram, inputSize, from, to);
}