#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constant to control how many input characters each thread processes.
// This value is chosen to balance work per thread and latency hiding on modern GPUs.
#define ITEMS_PER_THREAD 32

// CUDA kernel to compute a histogram for characters in the range [from, to].
// Each thread processes ITEMS_PER_THREAD characters from the input text and updates a privatized
// block-level histogram stored in shared memory. To avoid shared memory bank conflicts, each block 
// allocates 32 copies of the histogram (one copy per bank) stored in a strided layout: for each bin i,
// the value for copy c is located at offset (i * 32 + c). The copy a thread uses is determined by
// (threadIdx.x % 32). Finally, each block reduces its private histograms into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to, int itemsPerThread)
{
    // Calculate the number of bins needed.
    int histRange = to - from + 1;

    // Dynamically allocated shared memory for privatized histograms.
    // Layout: For each bin i (0 <= i < histRange), there are 32 copies stored at s_hist[i*32 + c] for c=0..31.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram: each thread initializes multiple entries.
    int totalSharedElems = histRange * 32;
    for (int i = threadIdx.x; i < totalSharedElems; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the starting index in the input array for this thread.
    int baseIndex = globalThreadId * itemsPerThread;

    // Each thread uses a specific histogram copy in shared memory.
    // The copy index is determined to be threadIdx.x modulo 32.
    int copyIndex = threadIdx.x % 32;

    // Process ITEMS_PER_THREAD characters assigned to this thread.
    for (int i = 0; i < itemsPerThread; i++)
    {
        int idx = baseIndex + i;
        if (idx < inputSize)
        {
            // Read the character as an unsigned value to get correctly its ordinal value.
            unsigned char ch = input[idx];

            // Check if the character falls within the specified range.
            if (ch >= from && ch <= to)
            {
                int bin = ch - from;
                // Atomically update the corresponding bin in the thread’s private shared histogram copy.
                atomicAdd(&s_hist[bin * 32 + copyIndex], 1);
            }
        }
    }
    __syncthreads();

    // Each thread now participates in reducing the 32 copies of each histogram bin to a single value.
    // Threads iterate over bins in a strided manner.
    for (int bin = threadIdx.x; bin < histRange; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum contributions from all 32 copies for this bin.
        for (int c = 0; c < 32; c++)
        {
            sum += s_hist[bin * 32 + c];
        }
        // Atomically accumulate this block's partial result into the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
    // No need for further synchronization.
}

// Host function that launches the histogram CUDA kernel.
//
// Parameters:
//   input       - Device pointer to an array of input text (chars).
//   histogram   - Device pointer to output histogram array. The array must be allocated with size (to-from+1).
//   inputSize   - Number of characters in the input array.
//   from, to    - Specify the inclusive range of character ordinal values to be histogrammed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Use constant ITEMS_PER_THREAD for per-thread work.
    int itemsPerThread = ITEMS_PER_THREAD;

    // Calculate total number of threads needed so that each thread processes ‘itemsPerThread’ characters.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Define a typical block size for modern NVIDIA GPUs.
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // Calculate the number of bins for the histogram.
    int histRange = to - from + 1;

    // Compute the size of shared memory required per block:
    // 32 copies for each bin, each copy is an unsigned int.
    size_t sharedMemSize = histRange * 32 * sizeof(unsigned int);

    // Launch the kernel. The caller is responsible for any host-device synchronization.
    histogram_kernel<<<numBlocks, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to, itemsPerThread);
}