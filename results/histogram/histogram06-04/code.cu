#include <cuda_runtime.h>

// Constant: number of input characters processed per thread.
// 32 is chosen as a good default value for modern NVIDIA GPUs with large input sizes.
constexpr int itemsPerThread = 32;

// CUDA kernel to compute the histogram for characters in the range [from, to].
// The input text is stored in the device memory array "input" of length "inputSize".
// The output histogram is stored in the device memory array "output_hist", where index i
// corresponds to the count for character with value (i + from).
//
// Optimization details:
// - Each thread processes itemsPerThread characters in a grid-stride loop.
// - A privatized histogram is maintained in shared memory with 32 copies per bin.
//   Each bin has 32 entries (placed with a stride of 32) so that threads from the same warp
//   (with unique lane IDs) access different banks, avoiding shared memory bank conflicts.
// - After processing, the 32 copies for each bin are reduced and atomically added to the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *output_hist)
{
    // Determine number of histogram bins for the specified range.
    int numBins = to - from + 1;

    // Declare shared memory for the privatized histogram.
    // There are 32 copies for each bin: total shared memory size = numBins * 32 unsigned integers.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared memory histogram: each thread initializes one or more slots.
    int tid = threadIdx.x;
    int totalSlots = numBins * 32;
    for (int i = tid; i < totalSlots; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Determine the thread's lane (0-31) within the warp.
    int lane = threadIdx.x & 31;

    // Compute global thread index.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid.
    int totalThreads = blockDim.x * gridDim.x;

    // Each thread processes "itemsPerThread" characters per iteration.
    // Using a grid-stride loop to cover the whole input array.
    for (int i = globalThreadId * itemsPerThread; i < inputSize; i += totalThreads * itemsPerThread)
    {
        // Unroll the loop for itemsPerThread iterations.
        #pragma unroll
        for (int j = 0; j < itemsPerThread; j++)
        {
            int pos = i + j;
            if (pos < inputSize)
            {
                // Read the character.
                unsigned char c = static_cast<unsigned char>(input[pos]);
                // Verify if the character is within the desired range.
                if (c >= from && c <= to)
                {
                    int bin = c - from;
                    // Update the privatized histogram.
                    // Each bin is stored in 32 copies; use the thread's lane to pick one copy.
                    atomicAdd(&s_hist[bin * 32 + lane], 1);
                }
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies for each bin in shared memory and update the global histogram.
    // Each thread iterates over bins in a strided loop.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int binSum = 0;
        // Accumulate counts from all 32 copies for this histogram bin.
        for (int copy = 0; copy < 32; copy++)
        {
            binSum += s_hist[bin * 32 + copy];
        }
        // Atomically add the block's contribution for this bin to the global histogram.
        atomicAdd(&output_hist[bin], binSum);
    }
}

// Host function that configures and launches the CUDA kernel.
// The "input" and "histogram" arrays are already allocated in device memory via cudaMalloc.
// The "inputSize" parameter specifies the number of characters in the input file.
// Host-device synchronization is handled externally to this function.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins for the range [from, to].
    int numBins = to - from + 1;

    // Define number of threads per block.
    // 256 threads per block is a good default for modern NVIDIA GPUs.
    int threadsPerBlock = 256;

    // Compute the total number of threads required across the grid.
    // Each thread processes "itemsPerThread" characters.
    int totalThreadsNeeded = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Calculate the number of blocks required.
    int blocks = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1)
    {
        blocks = 1;
    }

    // Compute the size of shared memory needed per block.
    // There are 32 copies for each of the "numBins" histogram bins.
    size_t sharedMemSize = numBins * 32 * sizeof(unsigned int);

    // Launch the CUDA kernel to compute the histogram.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);
}
