#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Process 16 items per thread by default.
// This constant is tuned for modern NVIDIA GPUs and large input sizes.
constexpr int itemsPerThread = 16;

// CUDA kernel to compute histogram for characters in the specified range [from, to].
// Each block uses shared memory to accumulate a private histogram which is later atomically added to global memory.
__global__ void histogramKernel(const char *d_input, unsigned int *d_histogram, unsigned int inputSize, int from, int to, int range)
{
    // Declare dynamic shared memory for the block's private histogram.
    // The size (in bytes) of shared memory is provided at kernel launch: range * sizeof(unsigned int)
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to 0.
    // Each thread initializes multiple bins in a strided manner.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute global thread index.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the starting index in the input array for this thread.
    int start = globalThreadId * itemsPerThread;

    // Each thread processes 'itemsPerThread' consecutive characters.
    for (int i = 0; i < itemsPerThread; ++i) {
        int idx = start + i;
        if (idx < inputSize) {
            // Cast to unsigned char to safely handle the char data.
            unsigned char c = static_cast<unsigned char>(d_input[idx]);
            // Check if the character falls within the specified range [from, to].
            if (c >= from && c <= to) {
                // Compute the bin index relative to the histogram range.
                int bin = c - from;
                // Atomically increment the corresponding bin in shared memory.
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Each thread now aggregates parts of the shared histogram into the global histogram.
    // A strided loop is used to allow all shared bins to be processed.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        // Atomically add the block's count to the global histogram.
        atomicAdd(&d_histogram[i], s_hist[i]);
    }
}

// Host function that configures and launches the histogram kernel.
// 'input' and 'histogram' are pointers to device memory allocated via cudaMalloc.
// 'inputSize' is the number of characters in the input array.
// 'from' and 'to' define the inclusive range of characters for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins in the histogram.
    int range = to - from + 1;

    // Select the number of threads per block.
    // 256 threads per block is a commonly efficient configuration on modern GPUs.
    int threadsPerBlock = 256;

    // Compute the total number of threads required: each thread processes 'itemsPerThread' elements.
    // Then, determine the number of blocks needed to cover the entire input.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);
    if (blocks < 1) {
        blocks = 1;
    }

    // Launch the kernel.
    // The dynamic shared memory allocated per block is 'range * sizeof(unsigned int)' bytes.
    histogramKernel<<<blocks, threadsPerBlock, range * sizeof(unsigned int)>>>(input, histogram, inputSize, from, to, range);
}