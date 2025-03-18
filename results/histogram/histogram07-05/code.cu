#include <cuda_runtime.h>
#include <cstdio>

// Constant controlling how many input chars are processed by each thread.
// A value of 128 is chosen as a reasonable balance for modern NVIDIA GPUs with large inputs.
constexpr int itemsPerThread = 128;

// CUDA kernel that computes a histogram for a text input restricted to a given char range.
// Each thread processes 'itemsPerThread' characters (if available).
// The shared-memory histogram is privatized into 32 copies to reduce shared memory bank conflicts.
// For a histogram with 'range' bins (range = to - from + 1):
//   - Shared memory is allocated with 'range * 32' unsigned ints.
//   - For a given bin index i and copy c (0 <= c < 32), the element is stored at offset i*32 + c.
// Each thread uses the copy with index = (threadIdx.x % 32).
// After processing, threads cooperatively reduce the 32 copies for each bin and atomically
// add the result into the global histogram.
__global__ void histogramKernel(const char *input, unsigned int *globalHistogram, 
                                unsigned int inputSize, int from, int range)
{
    // Declare shared memory for privatized histogram copies.
    // Total shared memory size is (range * 32) unsigned ints.
    extern __shared__ unsigned int smem[];

    // Initialize shared memory histogram bins to zero.
    int smemSize = range * 32;
    for (int i = threadIdx.x; i < smemSize; i += blockDim.x)
    {
        smem[i] = 0;
    }
    __syncthreads();

    // Compute a unique global thread index.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes a contiguous segment of the input.
    // 'start' is the starting index in the input array for this thread.
    int start = globalThreadId * itemsPerThread;

    // Choose the shared memory copy this thread will update.
    int threadCopy = threadIdx.x % 32;

    // Process 'itemsPerThread' chars assigned to this thread.
    for (int i = 0; i < itemsPerThread; i++)
    {
        int idx = start + i;
        if (idx < inputSize)
        {
            // Load the character.
            unsigned char c = static_cast<unsigned char>(input[idx]);
            // Check if the character is within the specified range.
            if (c >= from && c < from + range)
            {
                // Compute histogram bin for character.
                int bin = c - from;
                // Compute index in shared memory: bin * 32 + threadCopy.
                int histIndex = bin * 32 + threadCopy;
                // Atomically increment the privatized bin in shared memory.
                atomicAdd(&smem[histIndex], 1U);
            }
        }
    }
    __syncthreads();

    // Each thread reduces a subset of histogram bins from shared memory and
    // atomically accumulates the results into the global histogram.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum across the 32 copies for the current bin.
        for (int copy = 0; copy < 32; copy++)
        {
            sum += smem[bin * 32 + copy];
        }
        // Atomically add the block's contribution for this bin to the global histogram.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function that sets up and invokes the CUDA kernel.
// Parameters:
// - input: device pointer to the input text data (an array of 'char').
// - histogram: device pointer to the histogram array for output (unsigned int array).
// - inputSize: total number of characters in the input array.
// - from, to: defines the inclusive range [from, to] of character ordinal values to consider.
// The output histogram has 'to - from + 1' bins, where bin i holds the count for character (i + from).
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Define block size; 256 threads per block is a common choice for modern GPUs.
    int blockSize = 256;

    // Compute total number of threads required such that each thread processes 'itemsPerThread' characters.
    // Ensure that the total threads cover the entire input.
    unsigned int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Compute the number of blocks needed.
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // Calculate the size in bytes needed for shared memory allocation per block.
    // We need space for (range * 32) unsigned ints.
    size_t sharedMemBytes = range * 32 * sizeof(unsigned int);

    // Launch the kernel.
    histogramKernel<<<gridSize, blockSize, sharedMemBytes>>>(input, histogram, inputSize, from, range);
}