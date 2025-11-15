#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - Computes a histogram for a contiguous character range [from, to].
 * - Input:  device pointer 'input' to an array of chars (size = inputSize).
 * - Output: device pointer 'histogram' with (to - from + 1) bins.
 *
 * Optimization strategy:
 * - Use per-block shared-memory histograms (size <= 256 bins) to privatize
 *   updates and drastically reduce global memory atomic contention.
 * - Each block:
 *     1. Initializes its shared histogram for the target range.
 *     2. Processes a contiguous chunk of the input.
 *     3. Accumulates its shared histogram into the global histogram.
 *
 * itemsPerThread:
 * - Each thread processes 'itemsPerThread' characters.
 * - This increases the work per thread and amortizes kernel launch and
 *   synchronization overhead while maintaining good memory coalescing.
 * - For large inputs on A100/H100, a value of 16 is a good default trade-off
 *   between occupancy and memory throughput.
 */

static constexpr int BLOCK_SIZE      = 256;
static constexpr int itemsPerThread  = 16;

/*
 * Device kernel: compute histogram of characters in the range [from, from + range - 1].
 *
 * Parameters:
 *  - input:     device pointer to input characters.
 *  - histogram: device pointer to global histogram (size == range).
 *  - inputSize: number of characters in the input.
 *  - from:      lowest character code to account for (0 <= from <= 255).
 *  - range:     number of bins (range == to - from + 1, 1 <= range <= 256).
 */
__global__ void histogram_range_kernel(const char *__restrict__ input,
                                       unsigned int *__restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int range)
{
    // Shared-memory histogram privatized per block.
    // Maximum possible range is 256 (characters 0..255).
    __shared__ unsigned int sHist[256];

    // Initialize only the part of the shared histogram that we actually use.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        sHist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    // Layout:
    //   blockBase = blockIdx.x * (blockDim.x * itemsPerThread)
    //   threadBase = blockBase + threadIdx.x
    // Each thread processes 'itemsPerThread' characters at
    // indices: threadBase + k * blockDim.x, k = 0..itemsPerThread-1.
    // This ensures coalesced memory accesses across threads in a warp.
    unsigned int blockBase  = blockIdx.x * blockDim.x * itemsPerThread;
    unsigned int threadBase = blockBase + threadIdx.x;

    #pragma unroll
    for (int k = 0; k < itemsPerThread; ++k) {
        unsigned int idx = threadBase + k * blockDim.x;
        if (idx >= inputSize) {
            break;  // No more input for this thread.
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Only count characters within the requested range.
        if (bin >= 0 && bin < range) {
            // Shared memory atomics are much faster than global atomics.
            atomicAdd(&sHist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate the shared histogram into the global histogram.
    // Each block contributes at most 'range' global atomic operations.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int val = sHist[i];
        if (val != 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function: launch the histogram kernel.
 *
 * Parameters:
 *  - input:     device pointer to input characters.
 *  - histogram: device pointer to output histogram (size = to - from + 1).
 *  - inputSize: number of characters in the input.
 *  - from:      lowest character code in the range (0 <= from < to <= 255).
 *  - to:        highest character code in the range.
 *
 * Assumptions:
 *  - 'input' and 'histogram' have been allocated with cudaMalloc.
 *  - The caller is responsible for synchronization (e.g., cudaDeviceSynchronize),
 *    so this function does not synchronize the device.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0) {
        // Nothing to do for empty input.
        return;
    }

    // Validate range (as per problem statement these should be valid).
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    int range = to - from + 1;  // Number of bins in the histogram.

    // Initialize the global histogram to zero.
    // This is asynchronous; any required synchronization is up to the caller.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Configure kernel launch:
    // Each block processes BLOCK_SIZE * itemsPerThread characters.
    unsigned long long itemsPerBlock =
        static_cast<unsigned long long>(BLOCK_SIZE) * static_cast<unsigned long long>(itemsPerThread);

    unsigned int gridSize =
        static_cast<unsigned int>((static_cast<unsigned long long>(inputSize) + itemsPerBlock - 1ULL) / itemsPerBlock);

    if (gridSize == 0) {
        // Fallback in case of extremely small inputSize (though unlikely here).
        gridSize = 1;
    }

    // Launch kernel: no dynamic shared memory needed (we use static sHist[256]).
    histogram_range_kernel<<<gridSize, BLOCK_SIZE>>>(input, histogram, inputSize, from, range);
}