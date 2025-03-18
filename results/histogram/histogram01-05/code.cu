#include <cuda_runtime.h>

// Kernel to compute a histogram for characters within the continuous range [from, to].
// Each block creates a private histogram in shared memory that is later reduced into global memory.
// The dynamic shared memory allocated per block is sized to hold (to - from + 1) unsigned integers.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *globalHist, int from, int to) {
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Declare dynamic shared memory to hold the per-block histogram.
    extern __shared__ unsigned int shHist[];

    // Each thread initializes a subset of the shared memory histogram to 0.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        shHist[i] = 0;
    }
    __syncthreads();

    // Process the input array using a grid-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize) {
        // Cast to unsigned char to correctly handle the range [0,255].
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if character is within the specified range.
        if (c >= from && c <= to) {
            // Compute the bin index relative to the specified 'from'.
            int bin = c - from;
            // Atomically increment the corresponding bin in shared memory.
            atomicAdd(&shHist[bin], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Each thread contributes to accumulating the block's histogram into global histogram.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        // Atomic add the per-block count to the global histogram.
        atomicAdd(&globalHist[i], shHist[i]);
    }
}

// Host function that launches the CUDA kernel for histogram computation.
// Parameters:
//   input      - Device pointer to the input text array.
//   histogram  - Device pointer to the output histogram array (size: to - from + 1).
//   inputSize  - Total number of characters in the input array.
//   from, to   - Define the continuous character range [from, to] to histogram.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Determine the number of bins in the histogram.
    int range = to - from + 1;

    // Zero out the global histogram array before accumulation.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Define block and grid sizes.
    int blockSize = 256;
    // Calculate grid size to cover all input characters.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Launch the kernel with dynamically allocated shared memory.
    // Each block receives shared memory of size 'range * sizeof(unsigned int)'.
    histogram_kernel<<<gridSize, blockSize, range * sizeof(unsigned int)>>>(input, inputSize, histogram, from, to);
}