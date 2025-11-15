#include <cuda_runtime.h>

// Number of input characters processed by each thread.
// A value of 16 works well on recent NVIDIA data-center GPUs (A100/H100) for large inputs.
static constexpr int itemsPerThread = 16;

// Thread block size chosen as a multiple of the warp size for simplicity.
static constexpr int threadsPerBlock = 256;

/*
 * CUDA kernel to compute a partial histogram of a text buffer for characters
 * in the inclusive range [from, to].
 *
 * - input:       pointer to device memory containing 'inputSize' chars
 * - globalHist:  pointer to device memory for histogram, length = (to - from + 1)
 * - inputSize:   number of chars in 'input'
 * - from, to:    inclusive range [from, to] of character ordinal values (0..255)
 *
 * The kernel uses:
 *  - Per-warp privatized histograms in shared memory to reduce contention
 *    and to mitigate shared-memory bank conflicts.
 *  - A small padding per histogram to further reduce bank conflicts.
 *  - Shared-memory atomics for intra-block accumulation (fast on modern GPUs).
 *  - Global atomics only for block-level reduction per bin.
 */
__global__
void histogram_kernel(const char *input,
                      unsigned int *globalHist,
                      unsigned int inputSize,
                      int from,
                      int to)
{
    extern __shared__ unsigned int sharedHist[];

    const int warpSz        = 32;
    const int tid           = threadIdx.x;
    const int warpsPerBlock = blockDim.x / warpSz;

    // Number of bins for this histogram range.
    const int histSize      = to - from + 1;
    // Padding per warp histogram to reduce shared-memory bank conflicts.
    const int histSizePadded = histSize + 1;

    // Initialize all per-warp histograms in shared memory to zero.
    // We zero all bins including padding.
    const int totalBins = histSizePadded * warpsPerBlock;
    for (int i = tid; i < totalBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Each warp gets its own private histogram.
    const int warpId = tid / warpSz;
    unsigned int *warpHist = sharedHist + warpId * histSizePadded;

    // Global index for this thread's first element.
    unsigned int baseIndex =
        (blockIdx.x * blockDim.x + static_cast<unsigned int>(tid)) * itemsPerThread;

    // Process up to 'itemsPerThread' characters per thread.
    for (int item = 0; item < itemsPerThread; ++item) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(item);
        if (idx >= inputSize) {
            break;
        }

        // Treat characters as unsigned to consistently get values 0..255.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Only count characters within [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {

            int bin = static_cast<int>(c) - from;  // 0 .. (histSize-1)

            // Shared-memory atomic increments are fast and avoid global contention.
            atomicAdd(&warpHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // Each thread handles a subset of bins to parallelize the reduction.
    for (int bin = tid; bin < histSize; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate contribution from all warps in this block.
        for (int warp = 0; warp < warpsPerBlock; ++warp) {
            sum += sharedHist[warp * histSizePadded + bin];
        }
        if (sum > 0) {
            // One global atomic per bin per block.
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

/*
 * Host-side function to run the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to char buffer of length 'inputSize'
 *   histogram  - device pointer to unsigned int buffer of length (to - from + 1)
 *   inputSize  - number of characters in 'input'
 *   from, to   - inclusive character range [from, to], with 0 <= from < to <= 255
 *
 * Notes:
 *   - This function assumes that 'input' and 'histogram' have been allocated with cudaMalloc.
 *   - The histogram buffer is zeroed before the kernel launch.
 *   - Any required synchronization after the kernel launch is the caller's responsibility.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation.
    if (input == nullptr || histogram == nullptr || inputSize == 0) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int histSize       = to - from + 1;
    const int histSizePadded = histSize + 1;

    // Initialize the histogram on device to zero.
    cudaMemset(histogram, 0, static_cast<size_t>(histSize) * sizeof(unsigned int));

    // Kernel configuration:
    const unsigned int blockSize        = threadsPerBlock;
    const unsigned int elementsPerBlock = blockSize * itemsPerThread;

    // Compute the grid size so that all input characters are covered.
    unsigned int gridSize =
        (inputSize + elementsPerBlock - 1) / elementsPerBlock;
    if (gridSize == 0) {
        return;
    }

    // Shared memory size: one padded histogram per warp in each block.
    const int warpsPerBlock = blockSize / 32;
    const size_t sharedMemSize =
        static_cast<size_t>(histSizePadded) * warpsPerBlock * sizeof(unsigned int);

    // Launch the kernel. Synchronization (if needed) is handled by the caller.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}