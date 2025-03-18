#include <cuda_runtime.h>

// Define a constant that controls how many input characters each thread processes.
// For modern NVIDIA GPUs (such as the A100/H100) and large input sizes, a value of 32
// typically provides a good balance between memory throughput and occupancy.
constexpr int itemsPerThread = 32;

// CUDA kernel to compute a histogram over a restricted range from 'from' to 'to'.
// Each block uses shared memory to privatize the histogram, which is later
// accumulated atomically into the global histogram.
__global__ void histogramKernel(const char *input, unsigned int *global_hist,
                                unsigned int inputSize, int from, int to) {
    // Determine the size of the histogram range.
    int histSize = to - from + 1;

    // Declare dynamic shared memory array for per-block partial histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram to zero.
    // Each thread initializes multiple bins as required.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting index in the input array for this thread.
    // Each block processes blockDim.x * itemsPerThread items.
    int baseIndex = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x;

    // Process a contiguous chunk of input: each thread processes 'itemsPerThread' characters,
    // striding by blockDim.x to maintain coalesced memory access.
    for (int i = 0; i < itemsPerThread; ++i) {
        int index = baseIndex + i * blockDim.x;
        if (index < inputSize) {
            // Convert input character to unsigned to correctly handle values 0-255.
            unsigned char ch = static_cast<unsigned char>(input[index]);
            // Check if the character falls within the specified range.
            if (ch >= from && ch <= to) {
                // Atomically increment the corresponding bin in shared memory.
                atomicAdd(&s_hist[ch - from], 1);
            }
        }
    }
    __syncthreads();

    // Accumulate the shared histogram into the global histogram.
    // Each thread processes multiple bins if necessary.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        // Atomic addition is used to correctly combine results from different blocks.
        atomicAdd(&global_hist[i], s_hist[i]);
    }
}

// Host function that configures and launches the histogram kernel.
// The 'input' and 'histogram' pointers refer to device memory (allocated with cudaMalloc).
// 'inputSize' is the number of characters in the input buffer.
// 'from' and 'to' specify the inclusive range of character ordinal values to count.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Calculate the number of histogram bins.
    int histSize = to - from + 1;

    // Choose a block size appropriate for modern GPUs.
    const int blockSize = 256;
    // Compute the number of blocks needed such that each thread will process 'itemsPerThread' elements.
    int numBlocks = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    // Launch the kernel with dynamic shared memory allocated to hold 'histSize' unsigned ints.
    histogramKernel<<<numBlocks, blockSize, histSize * sizeof(unsigned int)>>>(
        input, histogram, inputSize, from, to);

    // Note: Synchronization (e.g., cudaDeviceSynchronize) is assumed to be handled by the caller.
}