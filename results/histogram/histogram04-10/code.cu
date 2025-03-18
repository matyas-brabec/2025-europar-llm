#include <cuda_runtime.h>

// Define the number of input characters processed per thread.
// A value of 32 is chosen as a good trade‐off between parallelism and work per thread on modern GPUs.
constexpr int ITEMS_PER_THREAD = 32;

// CUDA kernel to compute a histogram for a subset of characters defined by the continuous range [from, to].
// The input array "input" holds plain text data, and the output "histogram" array (of size (to - from + 1))
// stores the frequency of each character (i.e. histogram[i] holds the count for character with ordinal (i + from)).
// The kernel leverages shared memory for privatizing histograms per block, and each thread further accumulates
// its own counts in local registers (to reduce the number of atomic operations) before merging to shared memory.
__global__ void histogram_kernel(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    int numBins = to - from + 1;

    // Allocate shared memory histogram (static allocation for a maximum of 256 bins).
    // Even if numBins < 256, only the first numBins entries are used.
    __shared__ unsigned int s_hist[256];

    // Each thread initializes a portion of the shared histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread creates a private local histogram in registers.
    // Maximum possible bins is 256, so we allocate an array of that size.
    // Only the first numBins entries will be used.
    unsigned int local_hist[256];
    for (int i = 0; i < numBins; i++)
    {
        local_hist[i] = 0;
    }

    // Calculate the starting global index for this thread.
    // Each thread processes ITEMS_PER_THREAD consecutive characters.
    unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int startIndex = globalThreadId * ITEMS_PER_THREAD;

    // Process ITEMS_PER_THREAD characters per thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        unsigned int index = startIndex + i;
        if (index < inputSize)
        {
            // Use unsigned char conversion to ensure a value in the range [0,255].
            unsigned char ch = input[index];
            int bin = static_cast<int>(ch) - from;
            // Only count characters within the specified range.
            if (bin >= 0 && bin < numBins)
            {
                local_hist[bin]++;
            }
        }
    }

    // Merge the thread's private histogram into the block's shared histogram.
    for (int bin = 0; bin < numBins; bin++)
    {
        unsigned int count = local_hist[bin];
        if (count > 0)
        {
            atomicAdd(&s_hist[bin], count);
        }
    }
    __syncthreads();

    // One thread per block updates the global histogram with the block's accumulated results.
    if (threadIdx.x == 0)
    {
        for (int bin = 0; bin < numBins; bin++)
        {
            unsigned int count = s_hist[bin];
            if (count > 0)
            {
                atomicAdd(&histogram[bin], count);
            }
        }
    }
    // No need for further synchronization as the global update is complete.
}

// Host function to invoke the CUDA histogram kernel.
// The function assumes that 'input' and 'histogram' have been allocated with cudaMalloc,
// and that any required host-device synchronization is performed by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Define the number of threads per block.
    int blockSize = 256;
    // Compute the total number of threads required so that each thread processes ITEMS_PER_THREAD characters.
    int totalThreads = (inputSize + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
    // Calculate the number of blocks (grid size) needed.
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // Determine the number of histogram bins.
    int numBins = to - from + 1;
    // Initialize the global histogram array on the device to 0.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Launch the kernel.
    // The shared memory array is declared statically in the kernel, so no dynamic shared memory size is needed.
    histogram_kernel<<<gridSize, blockSize>>>(input, histogram, inputSize, from, to);

    // Note: Host-device synchronization (e.g. cudaDeviceSynchronize) is assumed to be handled by the caller.
}