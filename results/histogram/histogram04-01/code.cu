#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constant that controls how many input characters are processed by each thread.
// This value is chosen as a good balance for modern NVIDIA GPUs when the input size is large.
constexpr int ITEMS_PER_THREAD = 16;

// CUDA kernel to compute the histogram for a text input restricted to the range [from, to].
// Each thread processes ITEMS_PER_THREAD input characters and uses shared memory for block‐level privatization.
// After accumulation in shared memory, one thread per block atomically updates the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *global_histogram,
                                 unsigned int inputSize, int from, int to)
{
    // Compute the histogram range size.
    int histoSize = to - from + 1;

    // Allocate dynamic shared memory for the block-level histogram.
    // The shared memory size is passed at kernel launch and equals histoSize * sizeof(unsigned int).
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes a portion of the shared histogram to 0.
    for (int i = threadIdx.x; i < histoSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread ID.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes ITEMS_PER_THREAD consecutive elements.
    int startIdx = tid * ITEMS_PER_THREAD;
    int endIdx = startIdx + ITEMS_PER_THREAD;

    // Iterate over the assigned items.
    for (int i = startIdx; i < endIdx && i < inputSize; i++)
    {
        // Load the character from the input.
        char c = input[i];
        // Cast to unsigned to avoid negative values if char is signed.
        int val = static_cast<unsigned char>(c);
        // Check if the character falls within the specified [from, to] range.
        if (val >= from && val <= to)
        {
            // Compute the index within the histogram.
            int bin = val - from;
            // Atomically increment the corresponding bin in shared memory.
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // One thread per block accumulates the block's partial histogram into the global histogram.
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < histoSize; i++)
        {
            // Atomically add the block's bin count to the global histogram.
            atomicAdd(&global_histogram[i], s_hist[i]);
        }
    }
}

// Host function that configures the CUDA kernel launch parameters and invokes the histogram kernel.
// It assumes that both 'input' and 'histogram' arrays have been allocated on the device (using cudaMalloc).
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Calculate the total number of threads required.
    // Each thread processes ITEMS_PER_THREAD input characters.
    unsigned int totalThreads = (inputSize + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;

    // Define a typical block size. 256 threads per block is a common choice for modern GPUs.
    int blockSize = 256;
    // Calculate the number of blocks needed.
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // Compute the size of the dynamic shared memory required per block,
    // which depends on the size of the histogram range.
    int histoSize = to - from + 1;
    size_t sharedMemSize = histoSize * sizeof(unsigned int);

    // Launch the histogram kernel on the device.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}