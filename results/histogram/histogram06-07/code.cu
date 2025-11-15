#include <cuda_runtime.h>

/*
 * CUDA kernel for computing a character histogram over a restricted ASCII range.
 *
 * - Input:  plain text buffer `input` of length `inputSize` bytes (device memory).
 * - Range:  characters with ordinal values in [from, to], inclusive (0 <= from <= to <= 255).
 * - Output: `histogram` (device memory) of length (to - from + 1) unsigned ints.
 *           histogram[i] holds the count of character (from + i).
 *
 * Optimization details:
 * - Each thread processes `itemsPerThread` characters (or fewer near the end of the input).
 * - Each thread block privatizes the histogram in shared memory to reduce global atomics.
 * - To avoid shared memory bank conflicts:
 *     * We create 32 copies (one per bank) of the histogram in shared memory.
 *     * Layout: sharedHist[bin * 32 + lane], where lane = threadIdx.x % 32.
 *     * For a given bin, lanes 0..31 in a warp access indices bin*32 + 0..31, which map
 *       to different banks (4-byte words, 32 banks), eliminating intra-warp conflicts.
 * - Each block accumulates into its shared histogram copies, then reduces the 32 copies
 *   per bin into a single value and atomically adds it to the global histogram.
 */

constexpr int itemsPerThread = 16;   // Default: each thread processes up to 16 input chars.
constexpr int WARP_SIZE      = 32;   // NVIDIA GPUs have 32 threads per warp and 32 shared mem banks.

__global__ void histogram_range_kernel(const char * __restrict__ input,
                                       unsigned int * __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    const int numBins = to - from + 1;
    const int lane    = threadIdx.x & (WARP_SIZE - 1);  // warp lane ID: 0..31

    // Shared memory: 32 copies of the histogram, one per warp lane.
    // Indexing: sharedHist[bin * WARP_SIZE + lane]
    extern __shared__ unsigned int sharedHist[];

    // Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < numBins * WARP_SIZE; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    const unsigned int elementsPerBlock = blockDim.x * itemsPerThread;
    const unsigned int blockStart       = blockIdx.x * elementsPerBlock;

    // Process up to `itemsPerThread` items per thread, with coalesced global loads.
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = blockStart + threadIdx.x + i * blockDim.x;
        if (idx >= inputSize) {
            break;
        }

        unsigned char ch = static_cast<unsigned char>(input[idx]);
        int chVal        = static_cast<int>(ch);

        // Restrict to the [from, to] range.
        if (chVal >= from && chVal <= to) {
            int bin = chVal - from;  // bin in [0, numBins-1]
            // Each warp lane updates its own copy to avoid intra-warp bank conflicts.
            atomicAdd(&sharedHist[bin * WARP_SIZE + lane], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 copies per bin into a single count and update the global histogram.
    // We use the first `numBins` threads in the block for this reduction.
    if (threadIdx.x < numBins) {
        int bin  = threadIdx.x;
        int base = bin * WARP_SIZE;

        unsigned int sum = 0;
        for (int l = 0; l < WARP_SIZE; ++l) {
            sum += sharedHist[base + l];
        }

        if (sum > 0) {
            // One atomicAdd per bin per block to global memory.
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host helper function to launch the histogram kernel.
 *
 * Parameters:
 *   - input:      device pointer to input chars (plain text).
 *   - histogram:  device pointer to output histogram array of length (to - from + 1).
 *   - inputSize:  number of chars in the input buffer.
 *   - from, to:   inclusive character range [from, to] to be counted.
 *
 * Assumptions:
 *   - `input` and `histogram` are allocated via cudaMalloc.
 *   - Caller handles all host-device synchronization (e.g., cudaDeviceSynchronize()).
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Initialize the device histogram to zero.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose execution configuration.
    //  - blockDim.x must be a multiple of 32 for proper lane-based banking.
    //  - 256 threads per block is a good default for modern GPUs (A100/H100).
    const int blockSize = 256;
    const unsigned int elementsPerBlock = blockSize * itemsPerThread;

    unsigned int gridSize = (inputSize + elementsPerBlock - 1) / elementsPerBlock;
    if (gridSize == 0) {
        gridSize = 1;
    }

    // Shared memory: numBins * 32 (copies per bin) * sizeof(unsigned int).
    const size_t sharedMemSize = static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    histogram_range_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, histogram, inputSize, from, to
    );
}