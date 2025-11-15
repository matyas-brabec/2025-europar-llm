#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram over a restricted character range [from, to].
 *
 * - input:     Pointer to device memory containing the input text (array of chars).
 * - histogram: Pointer to device memory for the output histogram (size: to-from+1 bins).
 * - inputSize: Number of characters in the input buffer.
 * - from, to:  Inclusive character range [from, to] to be counted (0 <= from <= to <= 255).
 *
 * Implementation details:
 * - Each block maintains a private histogram in shared memory.
 * - Threads within a block cooperatively initialize the shared histogram to zero.
 * - Threads process the input using a grid-stride loop and update the shared histogram
 *   using atomic operations in shared memory (fast on modern GPUs).
 * - When finished, threads cooperatively add the shared histogram into the global
 *   histogram using atomicAdd on global memory, greatly reducing contention.
 */
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sharedHist[];  // Size determined at kernel launch.
    const int numBins = to - from + 1;

    // Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        sharedHist[i] = 0;
    }

    __syncthreads();

    // Compute global thread index and stride for grid-stride loop.
    const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    // Process input in a grid-stride manner.
    for (unsigned int idx = tid; idx < inputSize; idx += stride)
    {
        // Treat input as unsigned chars to avoid sign-extension issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Only count characters within [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to))
        {
            int bin = static_cast<int>(c) - from;
            // Atomic add in shared memory (fast on recent architectures).
            atomicAdd(&sharedHist[bin], 1U);
        }
    }

    __syncthreads();

    // Accumulate per-block shared histogram into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        unsigned int count = sharedHist[i];
        if (count != 0)
        {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 * - input:      Device pointer (cudaMalloc'ed) to the input text (chars).
 * - histogram:  Device pointer (cudaMalloc'ed) to the histogram array
 *               of size (to - from + 1) unsigned ints.
 * - inputSize:  Number of characters in the input buffer.
 * - from, to:   Inclusive character range [from, to] to be counted.
 *
 * Notes:
 * - This function assumes valid parameters (0 <= from <= to <= 255).
 * - The histogram buffer is zeroed before the kernel launch.
 * - Synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (input == nullptr || histogram == nullptr)
    {
        return;
    }

    if (inputSize == 0)
    {
        return;
    }

    if (from > to)
    {
        return;
    }

    const int numBins = to - from + 1;

    // Zero the device histogram buffer.
    // Using cudaMemsetAsync enqueues the operation in the default stream and
    // ensures correct ordering with the subsequent kernel launch.
    (void)cudaMemsetAsync(histogram, 0, numBins * sizeof(unsigned int), 0);

    // Choose launch configuration.
    //  - 256 threads per block is a good general choice on modern GPUs.
    //  - Limit the number of blocks to avoid excessively large grids; grid-stride
    //    loop ensures all data are processed even with fewer blocks than needed
    //    for a one-pass coverage.
    const int threadsPerBlock = 256;

    int blocksPerGrid = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    if (blocksPerGrid <= 0)
    {
        blocksPerGrid = 1;
    }

    // Cap the number of blocks to a reasonable value to maintain good occupancy
    // without oversubscribing the scheduler excessively. 1024 blocks is a
    // conservative, performant cap for data center GPUs like A100/H100.
    const int maxBlocks = 1024;
    if (blocksPerGrid > maxBlocks)
    {
        blocksPerGrid = maxBlocks;
    }

    // Shared memory size: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel.
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to);
}