#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram of characters in a specified range [from, to].
 *
 * Parameters:
 *   input           - device pointer to the input text buffer (array of chars)
 *   globalHistogram - device pointer to the output histogram (array of unsigned ints)
 *   inputSize       - number of characters in the input buffer
 *   from, to        - inclusive character range [from, to] (0 <= from <= to <= 255)
 *
 * Implementation details:
 *   - Each block builds a partial histogram in shared memory to reduce the number of
 *     global memory atomic operations.
 *   - After processing its portion of the input, each block atomically adds its
 *     shared-memory histogram to the global histogram.
 *   - The global histogram is assumed to be zero-initialized before this kernel is launched.
 */
__global__ void histogram_range_kernel(const char *input,
                                       unsigned int *globalHistogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Dynamic shared memory: one histogram per block.
    // Size (in bytes) is provided at kernel launch as: (to - from + 1) * sizeof(unsigned int).
    extern __shared__ unsigned int sharedHistogram[];

    // Number of bins in the requested character range.
    const int numBins = to - from + 1;

    // Initialize the shared histogram to zero.
    // All threads in the block participate in the initialization.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        sharedHistogram[bin] = 0;
    }

    __syncthreads();

    // Compute global thread index and stride for strided access pattern.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride         = blockDim.x * gridDim.x;

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int to_u   = static_cast<unsigned int>(to);

    // Each thread processes multiple characters with a grid-stride loop.
    for (unsigned int i = globalThreadId; i < inputSize; i += stride)
    {
        // Convert to unsigned to avoid issues with signed chars.
        unsigned char c  = static_cast<unsigned char>(input[i]);
        unsigned int cu  = static_cast<unsigned int>(c);

        // If the character is in the specified range, update the shared histogram.
        if (cu >= from_u && cu <= to_u)
        {
            unsigned int bin = cu - from_u;  // Bin index relative to 'from'
            atomicAdd(&sharedHistogram[bin], 1u);
        }
    }

    __syncthreads();

    // Merge the per-block shared histogram into the global histogram.
    // Again, all threads in the block participate in this step.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int count = sharedHistogram[bin];
        if (count != 0u)
        {
            // Global histogram must be zero-initialized before the kernel launch.
            atomicAdd(&globalHistogram[bin], count);
        }
    }
}

/*
 * Host function that prepares and launches the CUDA kernel to compute a histogram
 * over a specified character range.
 *
 * Parameters:
 *   input      - device pointer to the input text buffer (allocated by cudaMalloc)
 *   histogram  - device pointer to the output histogram array (allocated by cudaMalloc)
 *   inputSize  - number of characters in the input buffer
 *   from, to   - inclusive character range [from, to] to compute the histogram for
 *
 * Notes:
 *   - The size of 'histogram' must be at least (to - from + 1) * sizeof(unsigned int).
 *   - This function performs no device synchronization; the caller is responsible for
 *     synchronization (e.g., cudaDeviceSynchronize) and error checking as needed.
 *   - The histogram buffer is explicitly zeroed via cudaMemset before the kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute the number of bins for the requested range.
    const int numBins = to - from + 1;

    // Zero-initialize the histogram on the device.
    // This allows the kernel to safely use atomicAdd to accumulate counts.
    if (numBins > 0)
    {
        cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));
    }

    // If there is no input data, nothing more to do.
    if (inputSize == 0 || numBins <= 0)
    {
        return;
    }

    // Choose a reasonable block size for modern NVIDIA GPUs.
    const unsigned int threadsPerBlock = 256;

    // Compute the number of blocks. Limit to a safe maximum (e.g., 65535) to stay
    // within legacy grid size constraints while still providing sufficient parallelism.
    const unsigned int maxBlocks = 65535u;
    unsigned int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0)
    {
        blocks = 1;
    }
    if (blocks > maxBlocks)
    {
        blocks = maxBlocks;
    }

    // Dynamic shared memory size: one histogram of 'numBins' bins per block.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_range_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );

    // No cudaDeviceSynchronize() here; caller is responsible for synchronization and
    // error checking (e.g., cudaGetLastError) as needed.
}