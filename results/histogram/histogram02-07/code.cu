#include <cuda_runtime.h>

// CUDA kernel to compute a histogram over a continuous range [from, to] using shared memory.
// Each block computes its own sub-histogram in shared memory and then atomically merges it 
// into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *histogram,
                                   unsigned int inputSize, int from, int to) {
    // Compute the number of bins in the histogram.
    int numBins = to - from + 1;

    // Declare dynamically-allocated shared memory buffer.
    // The size of this array is specified during kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram bins to 0.
    // Use a grid-stride loop on the threads within the block.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread processes multiple characters in a grid-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < inputSize) {
        // Cast input character to unsigned to avoid issues with signed char.
        unsigned char ch = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the desired range.
        if (ch >= from && ch <= to) {
            // Atomically increment the corresponding bin in shared memory.
            atomicAdd(&s_hist[ch - from], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge the shared memory histogram into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function that launches the CUDA kernel to compute the histogram.
// 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// 'inputSize' is the number of characters in the input array.
// The histogram is computed for the continuous character range [from, to].
// The output histogram array has (to - from + 1) elements where each element i
// stores the count of occurrences of character with ordinal value (i + from).
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to) {
    // Calculate the number of bins required.
    int numBins = to - from + 1;

    // Initialize the output global histogram to 0.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose block size; this is a typical size for modern NVIDIA GPUs.
    int blockSize = 256;
    // Calculate grid size to cover the entire input array.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    // Ensure at least one block is launched.
    if (gridSize == 0) {
        gridSize = 1;
    }

    // Launch the kernel.
    // The third argument in the execution configuration sets the size of dynamic shared memory.
    // Here we allocate enough shared memory to hold 'numBins' unsigned integers.
    histogram_kernel<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(
        input, histogram, inputSize, from, to);
}