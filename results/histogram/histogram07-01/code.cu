#include <cuda_runtime.h>

// Default number of input items processed by each thread.
// This value is chosen to balance occupancy and memory throughput on modern GPUs.
constexpr int itemsPerThread = 16;

// CUDA kernel to compute the histogram for a restricted char range.
// The kernel uses shared memory for privatized histograms, with 32 copies per block
// to avoid shared memory bank conflicts. Each copy is stored such that its bin i is located at offset i*32 + c,
// where c in [0, 31] is the copy index.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins in the histogram.
    int histRange = to - from + 1;

    // Allocate shared memory for 32 copies of the histogram.
    // Layout: s_hist[i*32 + c] corresponds to bin i of copy c.
    extern __shared__ unsigned int s_hist[];

    // Total number of shared histogram entries.
    unsigned int totalSharedElems = histRange * 32;

    // Initialize shared memory histogram to zero using a grid-stride loop.
    for (unsigned int i = threadIdx.x; i < totalSharedElems; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Determine which bank copy this thread will use to update its histogram.
    // Using 32 copies (equal to the warp size) helps avoid shared memory bank conflicts.
    int copy = threadIdx.x % 32;

    // Compute the global starting index for this thread.
    unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int startIndex = globalThreadId * itemsPerThread;

    // Compute the stride over input array for grid-stride processing.
    unsigned int stride = blockDim.x * gridDim.x * itemsPerThread;

    // Process input elements in a grid-stride loop.
    for (unsigned int i = startIndex; i < inputSize; i += stride)
    {
        // Unroll the loop for better performance.
        #pragma unroll
        for (int j = 0; j < itemsPerThread; j++)
        {
            unsigned int pos = i + j;
            if (pos < inputSize)
            {
                // Read the character from input.
                unsigned char ch = input[pos];
                // Check if the character falls within the desired range.
                if (ch >= from && ch <= to)
                {
                    int bin = ch - from;
                    // Update the corresponding bin in the thread's designated histogram copy.
                    // Use atomicAdd to avoid race conditions with other threads writing to the same copy.
                    atomicAdd(&s_hist[bin * 32 + copy], 1);
                }
            }
        }
    }
    __syncthreads();

    // Each thread with threadIdx.x < histRange reduces the 32 bank copies for one bin.
    if (threadIdx.x < (unsigned int)histRange)
    {
        int bin = threadIdx.x;
        unsigned int sum = 0;
        #pragma unroll
        for (int bank = 0; bank < 32; bank++) {
            sum += s_hist[bin * 32 + bank];
        }
        // Atomically accumulate this block's bin count into the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
    // No further synchronization is needed.
}

// Host function to invoke the histogram CUDA kernel.
// The 'input' and 'histogram' parameters are assumed to be device pointers allocated via cudaMalloc.
// The 'inputSize' parameter specifies the number of characters in the input array.
// The 'from' and 'to' parameters specify the inclusive range for which the histogram should be computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins in the histogram.
    int histRange = to - from + 1;

    // Set the number of threads per block.
    int blockSize = 256;

    // Each thread processes 'itemsPerThread' characters;
    // therefore, each block processes blockSize * itemsPerThread characters.
    int blockItems = blockSize * itemsPerThread;

    // Determine the number of blocks required to cover the input data.
    int gridSize = (inputSize + blockItems - 1) / blockItems;
    if (gridSize < 1) gridSize = 1;

    // Calculate the dynamic shared memory size required:
    // We need 32 copies of the histogram, each having 'histRange' bins.
    size_t sharedMemSize = histRange * 32 * sizeof(unsigned int);

    // Launch the kernel with the computed configuration.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}