#include <cuda_runtime.h>

// CUDA kernel to compute a histogram for a specified character range using shared memory.
// Each block computes a partial histogram in shared memory and then atomically adds it to the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *histogram, int from, int to)
{
    // Calculate the number of histogram bins for the given inclusive range [from, to].
    int histSize = to - from + 1;
    
    // Dynamically allocated shared memory for block-level histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes parts of the shared histogram array.
    int tid = threadIdx.x;
    for (int i = tid; i < histSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Calculate global thread index and stride across the grid.
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes a portion of the input array.
    for (int i = globalId; i < inputSize; i += stride)
    {
        // Convert char to unsigned char to correctly interpret values in the range 0..255.
        unsigned char c = static_cast<unsigned char>(input[i]);
        // Only count characters within the specified range.
        if (c >= from && c <= to)
        {
            // Atomically increment the corresponding bin in shared memory.
            atomicAdd(&s_hist[c - from], 1);
        }
    }
    __syncthreads();

    // Each thread now writes one or more histogram bins from shared memory to global memory.
    for (int i = tid; i < histSize; i += blockDim.x)
    {
        // Atomically add the block-level count to the global histogram.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to launch the histogram kernel.
// The input and histogram arrays are assumed to be allocated on the device.
// Host-device synchronization and error checking are handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Define a block size that is appropriate for the target hardware (e.g., 256 threads per block).
    int blockSize = 256;
    // Determine the grid size based on the input size.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Calculate the number of bins needed for the histogram.
    int histSize = to - from + 1;
    // Determine the size of shared memory required for each block.
    size_t sharedMemSize = histSize * sizeof(unsigned int);

    // Launch the CUDA kernel with the calculated grid and block dimensions.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);

    // Note: Synchronization (e.g., cudaDeviceSynchronize) and error checking should be performed by the caller.
}