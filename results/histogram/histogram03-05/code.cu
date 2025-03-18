#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to compute histogram within a given character range [from, to].
// The input is an array of chars stored in device memory, and the histogram is an array of
// unsigned ints (size: to - from + 1) also in device memory.
// Each block maintains a private shared memory histogram to reduce the number of atomic updates
// to the global memory.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *global_histogram)
{
    // Compute the bin count for the specified range [from, to].
    int binCount = to - from + 1;

    // Dynamically allocated shared memory for the block's private histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared histogram array.
    // This loop ensures that all bins are set to 0.
    for (int i = threadIdx.x; i < binCount; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array with a grid-stride loop.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (index < inputSize)
    {
        // Read the input character.
        unsigned char c = static_cast<unsigned char>(input[index]);
        // Check if the character is within the desired range.
        if (c >= from && c <= to)
        {
            // Update the corresponding bin in the shared histogram.
            atomicAdd(&s_hist[c - from], 1);
        }
        index += stride;
    }
    __syncthreads();

    // Merge the block's shared histogram into the global histogram.
    // Each thread handles part of the bins.
    for (int i = threadIdx.x; i < binCount; i += blockDim.x)
    {
        atomicAdd(&global_histogram[i], s_hist[i]);
    }
}

// Host function that sets up and launches the CUDA kernel for histogram computation.
// The 'input' and 'histogram' pointers refer to device memory (allocated via cudaMalloc).
// 'inputSize' is the number of characters in the input buffer.
// The range [from, to] specifies the interval of character ordinals for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins in the histogram.
    int binCount = to - from + 1;

    // Choose a reasonable number of threads per block.
    int threadsPerBlock = 256;

    // Calculate the number of blocks necessary to cover the entire input array.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the histogram kernel.
    // The third kernel launch parameter specifies the dynamic shared memory size in bytes,
    // which is equal to the number of bins times the size of an unsigned int.
    histogram_kernel<<<blocks, threadsPerBlock, binCount * sizeof(unsigned int)>>>(input, inputSize, from, to, histogram);
}