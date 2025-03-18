// This code implements a CUDA kernel that computes a histogram for characters in a given range [from, to].
// The input is a plain text array (device pointer), and the output is a histogram array (also on device).
// The kernel is optimized by processing multiple items per thread (controlled by itemsPerThread)
// and by privatizing the histogram in shared memory with multiple copies (one per warp) to avoid bank conflicts.
// The host function run_histogram() sets up and launches the kernel with appropriate parameters.

// Include CUDA runtime header.
#include <cuda_runtime.h>

// Define the number of items (chars) processed per thread.
// This constant is tuned for modern NVIDIA GPUs when processing large inputs.
constexpr int itemsPerThread = 8;

// CUDA kernel to compute the histogram for a specified character range.
__global__ void histogram_kernel(const char *input, unsigned int *histogram,
                                 unsigned int inputSize, int from, int to)
{
    // Compute the size of the histogram (number of bins).
    const int rangeSize = to - from + 1;

    // Determine number of sub-histogram copies in shared memory.
    // We assign one copy per warp in the block.
    const int warpSize = 32;
    const int numCopies = (blockDim.x + warpSize - 1) / warpSize;

    // Allocate shared memory for the privatized histograms.
    // Each copy has 'rangeSize' bins.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram arrays to zero.
    // The total number of shared histogram bins is numCopies * rangeSize.
    for (int i = threadIdx.x; i < numCopies * rangeSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread determines its warp (sub-histogram) index.
    int warpId = threadIdx.x / warpSize;
    // Offset in the shared array for this warp's histogram.
    int histBase = warpId * rangeSize;

    // Compute the starting global index for this thread.
    // Each block processes blockDim.x * itemsPerThread input elements.
    int globalStart = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x;

    // Loop over 'itemsPerThread' items per thread.
    for (int i = 0; i < itemsPerThread; i++) {
        int index = globalStart + i * blockDim.x;
        if (index < inputSize) {
            // Load the character.
            unsigned char ch = static_cast<unsigned char>(input[index]);
            // Check if the character lies in the desired range.
            if (ch >= static_cast<unsigned char>(from) && ch <= static_cast<unsigned char>(to)) {
                // Compute the appropriate histogram bin.
                int bin = ch - from;
                // Atomically increment the bin in the thread's sub-histogram in shared memory.
                atomicAdd(&s_hist[histBase + bin], 1);
            }
        }
    }
    __syncthreads();

    // Now, reduce the multiple shared memory histogram copies into the global histogram.
    // Each thread in the block will accumulate bins in a strided loop.
    for (int bin = threadIdx.x; bin < rangeSize; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum contributions from all the sub-histogram copies.
        for (int copy = 0; copy < numCopies; copy++) {
            sum += s_hist[copy * rangeSize + bin];
        }
        // Atomically add the reduced result to the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function to launch the histogram kernel.
// 'input' and 'histogram' are assumed to be device pointers allocated via cudaMalloc.
// 'inputSize' is the number of characters in the input array.
// 'from' and 'to' specify the inclusive character range for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Compute histogram histogram size.
    const int rangeSize = to - from + 1;

    // Define the threads per block (a typical tuning parameter for modern GPUs).
    const int threadsPerBlock = 256;

    // Calculate the number of blocks needed such that every thread processes 'itemsPerThread' items.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Determine the number of sub-histogram copies per block (one per warp).
    const int warpSize = 32;
    int numCopies = (threadsPerBlock + warpSize - 1) / warpSize;

    // Calculate the size of shared memory required per block.
    size_t sharedMemSize = numCopies * rangeSize * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}