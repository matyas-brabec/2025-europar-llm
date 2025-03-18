// This code implements an optimized CUDA kernel to compute a histogram over a specified range of characters.
// The kernel uses shared memory for privatizing the histogram accumulation per block and then reduces the results
// using multiple copies of the histogram stored in shared memory. In particular, we create 32 copies of the histogram,
// with a stride of 32 so that value “i” of histogram copy “c” is stored at offset i*32 + c. Each thread uses
// a copy identified by (threadIdx.x % 32) to avoid shared memory bank conflicts.
//
// Each thread processes "itemsPerThread" consecutive characters from the input array to amortize global memory latency.
// The default value chosen for itemsPerThread is 16, which is typically efficient for modern NVIDIA GPUs on large inputs.
//
// The host function 'run_histogram' sets up the kernel launch parameters and calculates the required amount of shared memory.
// The 'input' and 'histogram' pointers are assumed to be allocated on the device (using cudaMalloc). All device
// synchronization (if any) is handled by the caller of run_histogram.

#include <cuda_runtime.h>

// Default number of input characters processed by each thread.
static constexpr int itemsPerThread = 16;

// CUDA kernel to compute a partial histogram for a subset of input characters.
// Parameters:
//   input          - Pointer to the text data (device memory).
//   inputSize      - Number of characters in the input array.
//   globalHistogram- Pointer to the output histogram (device memory). Each index corresponds to a value in the range [from, to].
//   from           - Lower bound of the character ordinal range (inclusive).
//   to             - Upper bound of the character ordinal range (inclusive).
__global__ void histogram_kernel(const char *input,
                                 unsigned int inputSize,
                                 unsigned int *globalHistogram,
                                 int from,
                                 int to)
{
    // Determine the range size: number of histogram bins.
    int range = to - from + 1;

    // Declare dynamically allocated shared memory.
    // We allocate 32 copies of the histogram (each of length 'range') to avoid bank conflicts.
    // The layout is: For each histogram bin index 'i', the copies are stored at offsets: i*32 + c, c in [0,31].
    extern __shared__ unsigned int s_hist[];

    // Total number of shared memory elements used for the 32 histogram copies.
    int totalSharedElements = range * 32;

    // Initialize shared memory; each thread initializes a portion of the shared memory.
    for (int idx = threadIdx.x; idx < totalSharedElements; idx += blockDim.x)
    {
        s_hist[idx] = 0;
    }
    __syncthreads();

    // Calculate the global thread id so that each thread processes itemsPerThread sequential characters.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = globalThreadId * itemsPerThread;

    // Each thread uses one of the 32 copies identified by threadIdx.x % 32.
    int localCopy = threadIdx.x % 32;

    // Process up to 'itemsPerThread' characters from global memory.
    for (int i = 0; i < itemsPerThread; i++)
    {
        int pos = start + i;
        if (pos < inputSize)
        {
            unsigned char ch = input[pos];
            // If character is within the [from, to] range, update the corresponding histogram bin.
            if (ch >= from && ch <= to)
            {
                int bin = ch - from;
                // Calculate index in shared memory array for this bin and this copy.
                int sharedIndex = bin * 32 + localCopy;
                // Update shared memory histogram using an atomic add to avoid race conditions
                // among threads that share the same copy.
                atomicAdd(&s_hist[sharedIndex], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies of the histogram: each thread with index < range adds up the values from all 32 copies for one or more bins.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum counts across the 32 copies for this histogram bin.
        for (int c = 0; c < 32; c++)
        {
            sum += s_hist[bin * 32 + c];
        }
        // Atomically add the block's partial histogram result to the global histogram.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function to launch the histogram kernel.
// Parameters:
//   input          - Pointer to the input text data (device memory).
//   histogram      - Pointer to the output histogram (device memory). Must have space for (to - from + 1) unsigned ints.
//   inputSize      - Number of characters in the input array.
//   from           - Lower bound of the character range (inclusive).
//   to             - Upper bound of the character range (inclusive).
//
// This function computes the grid dimensions based on the 'inputSize' and 'itemsPerThread', calculates the required
// shared memory size (32 copies of the histogram), and launches the kernel.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Select a thread block size. Here we choose 256 threads per block.
    int blockSize = 256;

    // Compute the number of blocks (grid size) needed so that each thread processes 'itemsPerThread' characters.
    int gridSize = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    // Determine histogram range size.
    int range = to - from + 1;
    // Calculate required shared memory size: 32 copies of the histogram, each of size 'range' unsigned ints.
    size_t sharedMemSize = range * 32 * sizeof(unsigned int);

    // Launch the kernel. Note: No host-device synchronization is performed here; the caller is responsible for it.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);
}