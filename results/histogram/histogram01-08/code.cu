#include <cuda_runtime.h>

// CUDA kernel to compute histogram on a char array in a specific range [from, to].
// Each block allocates a shared memory histogram array sized (to - from + 1).
// Threads in the block collaboratively accumulate counts into shared memory via atomic updates,
// and then merge the block’s partial histogram into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int numBins = to - from + 1;

    // Declare dynamic shared memory for the block's histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared memory histogram bins to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Use a grid-stride loop to process the input array.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    while (idx < inputSize)
    {
        // Read the character and convert to unsigned to ensure correct comparison.
        unsigned char val = input[idx];
        // If the character falls within the specified range, update the shared histogram.
        if (val >= from && val <= to) {
            int bin = val - from;
            atomicAdd(&s_hist[bin], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge the block-level shared histogram into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        // Use atomic addition to ensure correctness when multiple blocks update the histogram.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to launch the histogram kernel.
// Assumes that 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// The 'histogram' array is expected to have (to - from + 1) elements.
// Caller is responsible for host-device synchronization after calling this function.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Determine the number of bins.
    int numBins = to - from + 1;

    // Initialize the global histogram to zero prior to kernel launch.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Define CUDA kernel launch configuration.
    // Use 256 threads per block for a good balance between occupancy and performance.
    int blockSize = 256;
    // Compute the grid size so that all input elements are processed.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    // Optionally cap the grid size (e.g., at 65535) to avoid launching an excessive number of blocks.
    if (gridSize > 65535) {
        gridSize = 65535;
    }

    // Compute the amount of dynamic shared memory required per block.
    size_t sharedMemSize = numBins * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}