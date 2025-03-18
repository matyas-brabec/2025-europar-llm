#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// This CUDA kernel computes a histogram for characters within a given range [from, to].
// The input is an array of chars of length inputSize stored in global memory.
// The output is an array of unsigned integers of length (to - from + 1) stored in global memory.
// Each block uses shared memory to accumulate a local histogram which is then atomically merged
// into the global histogram. The kernel uses a grid-stride loop to process the input.
__global__ void histogram_kernel(const char *input, unsigned int *output, unsigned int inputSize, int from, int to)
{
    // Calculate number of histogram bins.
    int bins = to - from + 1;
    
    // Declare shared memory histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread in the block initializes part of the shared histogram.
    // Loop with stride equal to blockDim.x in order to cover all bins.
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();  // Ensure shared histogram is zeroed before processing.

    // Process the input array in a grid-stride loop.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < inputSize; i += stride)
    {
        // Read the character from global memory.
        unsigned char c = input[i];
        // Check if the character is within the desired range.
        if (c >= from && c <= to)
        {
            int bin = c - from;
            // Use atomicAdd to update the shared histogram.
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();  // Ensure all threads have finished updating shared histogram.

    // Merge the shared histogram into the global histogram.
    // Each thread processes a subset of bins.
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        // Atomic update because multiple blocks may update the same global histogram bin.
        atomicAdd(&output[i], s_hist[i]);
    }
}

// This host function launches the histogram_kernel on the GPU.
// Parameters:
//   input      - pointer to device memory holding a text file as an array of chars
//   histogram  - pointer to device memory where the resulting histogram will be stored
//   inputSize  - number of characters in the input array
//   from, to   - define the inclusive range [from, to] of character ordinal values for which the histogram is calculated
// The input and histogram arrays must have been allocated on the device already.
// Any required host-device synchronization is handled externally.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Define block and grid sizes.
    // Using 256 threads per block is a common choice for many NVIDIA GPUs.
    const int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover the inputSize.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate dynamic shared memory size in bytes.
    // The shared memory is used to store 'to - from + 1' unsigned integers.
    size_t sharedMemSize = (to - from + 1) * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}