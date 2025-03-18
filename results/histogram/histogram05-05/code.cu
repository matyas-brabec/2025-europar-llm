#include <cuda_runtime.h>

// Constant controlling how many input characters are processed by each thread.
// A default value of 64 is chosen for high throughput on modern NVIDIA GPUs.
constexpr int itemsPerThread = 64;

// Number of private histogram copies per block to reduce shared memory bank conflicts.
// We choose 32 copies, which corresponds to the typical warp size.
constexpr int NUM_COPIES = 32;

// CUDA kernel to compute the histogram of characters within the range [from, to].
// Input:
//   input      - the array of chars (device pointer)
//   global_hist- the output histogram array (device pointer) of size (to - from + 1)
//   inputSize  - total number of characters in the input buffer
//   from, to   - specify the continuous range [from, to] of char ordinal values to be counted
//
// The kernel uses shared memory privatization with multiple copies to reduce bank conflicts,
// processes multiple items per thread, and reduces the per-block histograms into the global one.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins (the histogram covers values from 'from' to 'to' inclusive).
    int binCount = to - from + 1;

    // Declare shared memory for privatized histograms.
    // Shared memory layout: NUM_COPIES rows, each of length 'binCount'.
    // This helps reduce bank conflicts when multiple threads update the same histogram bins.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram memory.
    int sharedSize = NUM_COPIES * binCount;
    for (int i = threadIdx.x; i < sharedSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index and the total number of threads.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads   = gridDim.x * blockDim.x;

    // Process the input in a grid-stride loop.
    // Each thread processes 'itemsPerThread' consecutive characters per iteration.
    for (unsigned int base = globalThreadId * itemsPerThread; base < inputSize; base += totalThreads * itemsPerThread)
    {
        // Process a chunk of 'itemsPerThread' characters.
        for (int j = 0; j < itemsPerThread; j++)
        {
            unsigned int idx = base + j;
            if (idx >= inputSize)
            {
                break;
            }

            // Read the character and cast to unsigned to avoid sign-extension issues.
            unsigned char ch = input[idx];
            
            // Check if the character falls within the specified range.
            if (ch >= from && ch <= to)
            {
                int bin = ch - from;
                // Each thread updates one of the NUM_COPIES histogram copies.
                // The choice of copy is based on its thread index to mitigate bank conflicts.
                int copyId = threadIdx.x % NUM_COPIES;
                // Update the shared histogram using atomic operation to ensure correctness.
                atomicAdd(&s_hist[copyId * binCount + bin], 1);
            }
        }
    }
    __syncthreads();

    // Each thread reduces the privatized histograms for a subset of bins and
    // atomically updates the global histogram.
    for (int bin = threadIdx.x; bin < binCount; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum over all private copies for the current bin.
        for (int copy = 0; copy < NUM_COPIES; copy++)
        {
            sum += s_hist[copy * binCount + bin];
        }
        // Atomically update the global histogram to combine contributions from all blocks.
        atomicAdd(&global_hist[bin], sum);
    }
}

// Host function to launch the histogram_kernel.
// Assumes that 'input' and 'histogram' are device pointers allocated via cudaMalloc,
// and that any necessary host-device synchronization is performed by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Determine the histogram bin count.
    int binCount = to - from + 1;

    // Define the block size (number of threads per block).
    // A block size of 256 is typically efficient on modern GPUs.
    const int blockSize = 256;
    
    // Compute the grid size so that all input characters are processed.
    // Each thread processes 'itemsPerThread' characters.
    int gridSize = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    // Calculate the amount of shared memory required per block.
    // Each block allocates NUM_COPIES * binCount unsigned integers.
    size_t sharedMemSize = NUM_COPIES * binCount * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}