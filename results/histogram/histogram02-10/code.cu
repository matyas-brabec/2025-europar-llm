#include <cuda_runtime.h>
#include <cassert>

// CUDA kernel to compute histogram for characters in the range [rangeFrom, rangeTo].
// Each block maintains a private, shared-memory histogram that is later atomically merged into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 unsigned int *globalHistogram, int rangeFrom, int rangeTo)
{
    // Calculate the histogram size for the given range.
    const int histSize = rangeTo - rangeFrom + 1;

    // Declare shared memory for the block's private histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared histogram to zero.
    for (int bin = threadIdx.x; bin < histSize; bin += blockDim.x)
    {
        s_hist[bin] = 0;
    }
    __syncthreads();

    // Compute a grid-stride loop over the input text array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize)
    {
        // Fetch the character.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // If the character is within the computed range, update the shared histogram.
        if (c >= rangeFrom && c <= rangeTo)
        {
            // Atomic update in shared memory to prevent race conditions.
            atomicAdd(&s_hist[c - rangeFrom], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // After processing, each block atomically adds its shared histogram into the global histogram.
    for (int bin = threadIdx.x; bin < histSize; bin += blockDim.x)
    {
        atomicAdd(&globalHistogram[bin], s_hist[bin]);
    }
    // No need for __syncthreads() after global atomic updates since kernel ends here.
}

// Host function that invokes the CUDA histogram kernel.
// Assumes that 'input' and 'histogram' arrays are allocated on the device (via cudaMalloc).
// The caller is responsible for host-device synchronization.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram,
void run_histogram(const char *input, unsigned int *histogram,
                                unsigned int inputSize, int from, int to)
{
    // Ensure that the input range is valid.
    assert(from >= 0 && from < to && to <= 255);
    const int histSize = to - from + 1;

    // Setup execution configuration.
    // Choosing 256 threads per block (adjustable based on GPU capabilities).
    int threadsPerBlock = 256;
    // Compute the number of blocks such that we cover all input elements.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Optionally, cap the number of blocks for very large inputs when needed.
    // Modern GPUs can support many blocks, so typically we use the calculated value.

    // Launch the kernel with dynamic shared memory size equal to histSize * sizeof(unsigned int).
    histogram_kernel<<<blocks, threadsPerBlock, histSize * sizeof(unsigned int)>>>(
        input, inputSize, histogram, from, to
    );

    // Note: Error checking and host-device synchronization is assumed to be handled by the caller.
}
