#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram for characters in a specified range [from, from + numBins - 1].
 * 
 * - Each block builds a private histogram in shared memory to minimize global memory atomics.
 * - After processing its portion of the input, each block merges its shared histogram into the global histogram.
 */
__global__ void histogram_range_kernel(const char * __restrict__ input,
                                       unsigned int * __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int numBins)
{
    // Dynamic shared memory allocated at kernel launch: numBins * sizeof(unsigned int)
    extern __shared__ unsigned int shist[];

    const int tid       = threadIdx.x;
    const int blockSize = blockDim.x;
    const int gridSize  = gridDim.x;
    const unsigned int globalThreadId = blockIdx.x * blockSize + tid;
    const unsigned int totalThreads   = blockSize * gridSize;

    // Initialize the shared histogram to zero in a strided loop
    for (int i = tid; i < numBins; i += blockSize) {
        shist[i] = 0u;
    }

    __syncthreads();

    // Grid-stride loop over input data
    for (unsigned int idx = globalThreadId; idx < inputSize; idx += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int val = static_cast<int>(c);
        int bin = val - from;

        // Only count characters that fall into the [from, from + numBins - 1] range
        if (bin >= 0 && bin < numBins) {
            // Atomic add in shared memory is much faster than directly in global memory
            atomicAdd(&shist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge the per-block shared histogram into the global histogram
    for (int i = tid; i < numBins; i += blockSize) {
        unsigned int count = shist[i];
        if (count > 0u) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function that launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to the input character buffer
 *   histogram  - device pointer to the output histogram buffer (size: (to - from + 1) * sizeof(unsigned int))
 *   inputSize  - number of characters in the input buffer
 *   from, to   - inclusive range of character codes [from, to] to be histogrammed (0 <= from < to <= 255)
 *
 * Requirements/Assumptions:
 *   - 'input' and 'histogram' are allocated on the device via cudaMalloc.
 *   - Caller handles any required synchronization (e.g., cudaDeviceSynchronize).
 */
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate range; if invalid or empty input, do nothing.
    if (inputSize == 0 || from > to) {
        return;
    }

    // Number of bins in the histogram (inclusive range)
    unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Initialize global histogram to zero
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose a reasonable block size for modern NVIDIA GPUs
    const int blockSize = 256;

    // Compute initial grid size; grid-stride loop in the kernel will cover all elements.
    unsigned int gridSize = (inputSize + blockSize - 1u) / blockSize;

    // Clamp grid size to a safe upper bound (needed only for very large inputs)
    const unsigned int maxBlocks = 65535u;
    if (gridSize == 0u) {
        gridSize = 1u;
    } else if (gridSize > maxBlocks) {
        gridSize = maxBlocks;
    }

    // Shared memory size: one unsigned int per bin per block
    size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel; caller is responsible for error checking and synchronization if desired
    histogram_range_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        static_cast<int>(numBins)
    );
}