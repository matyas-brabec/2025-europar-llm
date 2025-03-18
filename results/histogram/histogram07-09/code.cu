#include <cuda_runtime.h>

// Constant controlling how many input chars are processed by each thread.
// This value can be tuned for optimal performance on modern GPUs.
constexpr int itemsPerThread = 16;

// CUDA kernel that computes a histogram over a specified char range.
// Parameters:
//   input       - Pointer to the input text (device memory).
//   inputSize   - Number of characters in the input array.
//   from, to    - Inclusive range of character ordinal values to histogram.
//   global_hist - Pointer to the output histogram array (device memory)
//                 of length (to - from + 1).
//
// The kernel uses shared memory privatization: each block allocates 32 copies
// of the histogram (to avoid shared memory bank conflicts), where each copy is
// stored in a strided fashion so that copy c's bin i is at offset (i * 32 + c).
// Each thread works on one copy (selected by threadIdx.x % 32) and accumulates
// local counts. After processing input data, the block reduces these copies into
// the global histogram using atomicAdd.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 int from, int to, unsigned int *global_hist)
{
    // Compute the number of histogram bins (i.e. the range length).
    int histRange = to - from + 1;

    // Dynamically allocated shared memory for the privatized histograms.
    // There are 32 copies; each copy has 'histRange' bins stored in strided fashion.
    extern __shared__ unsigned int sharedHist[];

    // Total number of shared histogram elements.
    int totalSharedBins = histRange * 32;
    // Initialize all shared histogram elements to zero.
    for (int i = threadIdx.x; i < totalSharedBins; i += blockDim.x)
    {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Each thread uses its designated shared histogram copy determined by threadIdx.x % 32.
    int myCopy = threadIdx.x % 32;

    // Compute global thread index and total number of threads in the grid.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Each thread processes 'itemsPerThread' consecutive input characters at a time.
    // The grid-stride loop allows processing of arbitrarily large input arrays.
    for (unsigned int base = globalThreadId * itemsPerThread; base < inputSize;
         base += totalThreads * itemsPerThread)
    {
        // Unroll loop for processing multiple items per thread.
#pragma unroll
        for (int j = 0; j < itemsPerThread; j++)
        {
            unsigned int index = base + j;
            if (index < inputSize)
            {
                // Read the character and check if it falls within the designated range.
                unsigned char ch = input[index];
                if (ch >= from && ch <= to)
                {
                    int bin = ch - from;
                    // Update the privatized histogram copy using atomic addition to prevent races.
                    atomicAdd(&sharedHist[bin * 32 + myCopy], 1);
                }
            }
        }
    }
    __syncthreads();

    // After processing, reduce the 32 privatized histogram copies into the global histogram.
    // Each thread processes multiple histogram bins in parallel.
    for (int binIdx = threadIdx.x; binIdx < histRange; binIdx += blockDim.x)
    {
        unsigned int sum = 0;
#pragma unroll
        for (int c = 0; c < 32; c++)
        {
            sum += sharedHist[binIdx * 32 + c];
        }
        // Atomically add the block's result for this bin to the global histogram.
        atomicAdd(&global_hist[binIdx], sum);
    }
}

// Host function that configures and launches the histogram_kernel.
// It expects that the 'input' and 'histogram' arrays have been allocated on the device.
// Parameters:
//   input      - Device pointer to the input text buffer.
//   histogram  - Device pointer to the output histogram array.
//   inputSize  - Number of characters in the input buffer.
//   from, to   - Specify the continuous range of character ordinal values to histogram.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int histRange = to - from + 1;

    // Set up execution parameters.
    // Using 256 threads per block is a good default for modern NVIDIA GPUs.
    int blockSize = 256;

    // Each thread processes 'itemsPerThread' characters, so determine how many threads are needed.
    unsigned int totalThreadsNeeded = (inputSize + itemsPerThread - 1) / itemsPerThread;
    int gridSize = (totalThreadsNeeded + blockSize - 1) / blockSize;

    // Calculate the required shared memory size:
    // 32 copies of the histogram, each with 'histRange' unsigned ints.
    size_t sharedMemSize = histRange * 32 * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, from, to, histogram);
}