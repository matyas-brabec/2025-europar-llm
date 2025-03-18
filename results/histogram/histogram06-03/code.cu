#include <cuda_runtime.h>

// CUDA kernel to compute histogram of characters in a given range [from, to].
// The kernel utilizes shared memory privatization with 32 copies (banks) to avoid
// shared memory bank conflicts. Each thread processes a fixed number of input
// characters defined by the constant itemsPerThread.
__global__ void histogramKernel(const char *input, unsigned int inputSize,
                                int from, int to, unsigned int *globalHist)
{
    // Constant defining how many characters each thread will process.
    // Chosen as 128 to provide a good balance between occupancy and compute intensity
    // on modern NVIDIA GPUs (e.g., A100 or H100).
    const int itemsPerThread = 128;

    // Number of copies of the histogram stored in shared memory.
    // 32 copies will be used to map each possible warp lane to its own bank.
    const int NUM_COPIES = 32;

    // Compute the number of bins for the histogram.
    int numBins = to - from + 1;

    // Declare shared memory for the privatized histogram.
    // The layout is organized in column-major order:
    //     s_hist[bin * NUM_COPIES + copy]
    // where 'copy' ranges from 0 to NUM_COPIES-1.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram to zero.
    // Each thread initializes a portion of the shared memory.
    for (int i = threadIdx.x; i < numBins * NUM_COPIES; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread ID.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the starting index in the input array for this thread.
    // Every thread processes 'itemsPerThread' consecutive characters.
    int baseIndex = globalThreadId * itemsPerThread;

    // Each thread uses its warp lane (lower 5 bits of threadIdx.x) to select
    // one copy of the histogram. This helps ensuring that threads within
    // the same warp access distinct addresses in shared memory.
    int lane = threadIdx.x & 31;

    // Process 'itemsPerThread' input characters.
    // Use loop unrolling hint for potential performance improvement.
#pragma unroll
    for (int i = 0; i < itemsPerThread; i++)
    {
        int index = baseIndex + i;
        if (index < inputSize)
        {
            char c = input[index];
            // Check if the character is within the specified range.
            if (c >= from && c <= to)
            {
                int bin = c - from;
                // Update the privatized histogram in shared memory.
                // Each update is performed using atomicAdd to avoid race conditions.
                // The histogram is organized so that each bin has NUM_COPIES entries,
                // and the 'lane' index selects the specific copy.
                atomicAdd(&s_hist[bin * NUM_COPIES + lane], 1);
            }
        }
    }
    __syncthreads();

    // Reduction: Sum the contributions from the NUM_COPIES copies of
    // each histogram bin and add the result to the global histogram.
    // The reduction is distributed among the threads in the block using a strided loop.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int sum = 0;
#pragma unroll
        for (int j = 0; j < NUM_COPIES; j++)
        {
            sum += s_hist[bin * NUM_COPIES + j];
        }
        // Update the global histogram using atomicAdd since multiple blocks
        // may update the same bin concurrently.
        atomicAdd(&globalHist[bin], sum);
    }
}

// C++ function that invokes the CUDA kernel with appropriate parameters.
// The input and histogram arrays are assumed to be allocated on the device using cudaMalloc.
// Host-device synchronization is handled by the caller.
/// @FIXED
/// extern "C"
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Constant defining the number of input characters each thread will process.
    const int itemsPerThread = 128;

    // Set a typical number of threads per block for modern GPUs.
    const int threadsPerBlock = 256;

    // Compute the total number of threads required so that every thread
    // processes 'itemsPerThread' inputs.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Compute the number of thread blocks needed.
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Compute the number of bins for the histogram.
    int numBins = to - from + 1;

    // Number of copies in shared memory used to avoid bank conflicts.
    const int NUM_COPIES = 32;

    // Calculate the size (in bytes) of the shared memory needed per block.
    size_t sharedMemSize = numBins * NUM_COPIES * sizeof(unsigned int);

    // Launch the kernel. Note that any error checking and synchronization is assumed
    // to be done by the caller.
    histogramKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(input, inputSize,
                                                                     from, to, histogram);
}
