#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constant controlling how many input chars are processed by each thread.
// The default value is chosen to be 8, as it provides a good balance between
// global memory throughput and occupancy on modern NVIDIA GPUs.
#define ITEMS_PER_THREAD 8

// CUDA kernel that computes a histogram for a specified character range.
// Input: 'input' array of chars, 'inputSize' is its length.
// 'from' and 'to' specify the inclusive range of ASCII codes to accumulate.
// Output: 'globalHistogram' is an array of (to - from + 1) bins where each bin i
// holds the count of occurrences of (i + from) in the input.
// Optimization: Each block maintains a privatized histogram in shared memory.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 unsigned int *globalHistogram, int from, int to)
{
    // Calculate the number of bins for the histogram.
    int bins = to - from + 1;

    // Allocate shared memory for the block-local histogram.
    // The shared memory size is set at kernel launch to be bins * sizeof(unsigned int).
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to 0 using a thread-stride loop.
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global starting index for this thread.
    // Each block processes blockDim.x * ITEMS_PER_THREAD characters.
    unsigned int baseIndex = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;

    // Loop over ITEMS_PER_THREAD characters per thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        unsigned int idx = baseIndex + i * blockDim.x;
        if (idx < inputSize)
        {
            // Load character and cast to unsigned char to obtain a value in 0-255.
            unsigned char ch = static_cast<unsigned char>(input[idx]);

            // Check if the character is within the specified range.
            if (ch >= from && ch <= to)
            {
                // Map the character to the appropriate bin.
                int bin = ch - from;
                // Update the block-local histogram using an atomic update in shared memory.
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Merge the block's shared histogram into the global histogram.
    // Each thread processes a subset of bins in a thread-stride loop.
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
    {
        // Use an atomic add on global memory since shared histograms from multiple blocks
        // must be safely accumulated.
        atomicAdd(&globalHistogram[i], s_hist[i]);
    }
}

// Host function that invokes the CUDA kernel to compute the histogram.
// Assumptions:
//   - 'input' and 'histogram' are pointers to device memory allocated via cudaMalloc.
//   - 'inputSize' is the number of characters in the input array.
//   - Host-device synchronization (cudaMemcpy, cudaDeviceSynchronize, etc.) is managed by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins.
    int bins = to - from + 1;

    // Set up execution configuration.
    // Choose a block size of 256 threads, which is well-suited for modern GPUs.
    int blockSize = 256;
    // Each block processes blockSize * ITEMS_PER_THREAD input elements.
    int elementsPerBlock = blockSize * ITEMS_PER_THREAD;
    // Compute the number of blocks needed to cover the entire input.
    int gridSize = (inputSize + elementsPerBlock - 1) / elementsPerBlock;

    // Launch the kernel.
    // Allocate dynamic shared memory to hold 'bins' unsigned integers.
    histogram_kernel<<<gridSize, blockSize, bins * sizeof(unsigned int)>>>(input, inputSize, histogram, from, to);
}