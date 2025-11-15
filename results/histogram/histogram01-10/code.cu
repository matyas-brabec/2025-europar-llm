#include <cuda_runtime.h>
#include <cstddef>

/*
 * CUDA kernel to compute a histogram for a specified contiguous character range.
 *
 * Parameters:
 *   input      - device pointer to input text (array of chars)
 *   histogram  - device pointer to output histogram (array of unsigned int)
 *   inputSize  - number of characters in the input
 *   from       - first character code (inclusive) of the range
 *   rangeSize  - number of bins in the range (to - from + 1)
 *
 * The histogram array has 'rangeSize' elements. Bin i counts occurrences of
 * character code (from + i).
 *
 * Strategy:
 *   - Each block builds a private histogram in shared memory using atomic
 *     operations (much faster than atomics to global memory).
 *   - After processing the input with a grid-stride loop, each block atomically
 *     accumulates its shared histogram into the global histogram.
 *   - A special-case fast path is used when the range is the full 0..255.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int rangeSize)
{
    extern __shared__ unsigned int s_hist[];

    // Initialize the per-block shared histogram to zero.
    for (int i = threadIdx.x; i < rangeSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride         = blockDim.x * gridDim.x;

    // Fast path when the range covers all 256 possible byte values.
    if (from == 0 && rangeSize == 256) {
        for (unsigned int idx = globalThreadId; idx < inputSize; idx += stride) {
            unsigned char c = static_cast<unsigned char>(input[idx]);
            // No bounds check needed: c is in [0,255].
            atomicAdd(&s_hist[c], 1);
        }
    } else {
        // General path: only count characters within [from, from + rangeSize - 1].
        for (unsigned int idx = globalThreadId; idx < inputSize; idx += stride) {
            unsigned char c = static_cast<unsigned char>(input[idx]);
            int bin = static_cast<int>(c) - from;
            if (bin >= 0 && bin < rangeSize) {
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }

    __syncthreads();

    // Accumulate the per-block histogram into the global histogram.
    // The "if (val != 0)" reduces unnecessary atomic operations.
    for (int i = threadIdx.x; i < rangeSize; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function to configure and launch the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text (array of chars allocated with cudaMalloc)
 *   histogram  - device pointer to output histogram (array of unsigned int allocated with cudaMalloc)
 *   inputSize  - number of characters in the input buffer
 *   from       - first character code (inclusive) of the histogram range (0 <= from <= 255)
 *   to         - last character code (inclusive) of the histogram range (from <= to <= 255)
 *
 * The histogram array must have space for (to - from + 1) unsigned int entries.
 * Synchronization and error checking (if desired) should be handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute the number of bins in the target range.
    int rangeSize = to - from + 1;
    if (rangeSize <= 0) {
        // Invalid range; nothing to do.
        return;
    }

    // Initialize the histogram on the device to zero.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeSize) * sizeof(unsigned int));

    if (inputSize == 0) {
        // No data to process.
        return;
    }

    // Use a reasonable block size for A100/H100 GPUs.
    const int blockSize = 256;

    // Grid size chosen to cover the input using a grid-stride loop.
    int gridSize = static_cast<int>((inputSize + blockSize - 1) / blockSize);
    if (gridSize <= 0) {
        gridSize = 1;
    }

    // Allocate one unsigned int per bin in shared memory for each block.
    const size_t sharedMemSize = static_cast<size_t>(rangeSize) * sizeof(unsigned int);

    // Launch the kernel. All operations are enqueued on the default stream.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        rangeSize
    );

    // Caller is responsible for synchronization and error checking.
}