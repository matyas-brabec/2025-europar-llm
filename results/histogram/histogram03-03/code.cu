#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to compute histogram of a text buffer for characters in the range [from, to].
// Each block uses shared memory to privatize the histogram and reduce contention on global memory atomics.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *globalHistogram, int from, int to)
{
    // Compute the number of histogram bins in the provided range.
    int binCount = to - from + 1;

    // Declare dynamically allocated shared memory for the private histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram bins to zero in a parallel manner.
    for (int i = threadIdx.x; i < binCount; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Use grid-stride loop to process the input array.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = globalThreadId; i < inputSize; i += stride)
    {
        // Read the character from the input buffer.
        unsigned char c = static_cast<unsigned char>(input[i]);

        // Check if the character falls within the specified range.
        if (c >= from && c <= to)
        {
            // Calculate the bin index corresponding to the character.
            int binIdx = c - from;
            // Atomically update the private histogram in shared memory.
            atomicAdd(&s_hist[binIdx], 1);
        }
    }
    __syncthreads();

    // Each thread updates the global histogram from the shared memory private histogram.
    for (int i = threadIdx.x; i < binCount; i += blockDim.x)
    {
        // Atomic add to ensure correctness when multiple blocks update the same bin.
        atomicAdd(&globalHistogram[i], s_hist[i]);
    }
}

// Host function to launch the CUDA histogram kernel.
// The input array 'input' and output array 'histogram' are assumed to be allocated on the device via cudaMalloc.
// 'inputSize' is the number of characters in the input buffer.
// The histogram is computed for characters in the range [from, to], and the resulting histogram has (to-from+1) entries.
// Device synchronization is assumed to be handled by the caller.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Compute the number of blocks needed to cover the input data.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0)
    {
        blocks = 1;
    }

    // Calculate number of bins and the shared memory size needed per block.
    int binCount = to - from + 1;
    size_t sharedMemorySize = binCount * sizeof(unsigned int);

    // Launch the CUDA kernel with the computed grid dimensions and shared memory allocation.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemorySize>>>(input, inputSize, histogram, from, to);
}
