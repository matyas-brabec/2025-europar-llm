#include <cuda_runtime.h>

/*
 * CUDA histogram kernel with shared-memory privatization optimized
 * for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - The input is an array of chars in device memory.
 * - The histogram is computed only for characters in the range [from, to].
 * - The output is an array of (to - from + 1) unsigned ints in device memory.
 *
 * Optimization details:
 * - Each thread processes ITEMS_PER_THREAD characters from global memory
 *   to amortize scheduling and indexing overhead.
 * - Histogram privatization is performed in shared memory using 32 copies
 *   (one per warp lane) of each histogram bin.
 * - Shared histogram layout: shHist[bin * 32 + laneId]
 *     * For a fixed bin, all 32 lanes write to different shared-memory banks.
 *     * This eliminates intra-warp bank conflicts.
 * - To handle multiple warps per block safely, updates into shared memory use
 *   atomicAdd (required for correctness when different warps update the same
 *   bin copy).
 * - After processing input, block-local histograms are reduced (summing over
 *   the 32 lane-specific copies) and the result is accumulated into the
 *   global histogram with atomicAdd (per bin per block).
 */

static constexpr int WARP_SIZE       = 32;
// Tunable parameter: number of characters processed per thread.
// 16 is a good default for large inputs on recent data-center GPUs.
static constexpr int ITEMS_PER_THREAD = 16;

// CUDA kernel: compute partial histogram for a subset of input.
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int shHist[];  // Size: (range * WARP_SIZE) uints

    const int range = to - from + 1;
    const unsigned int tid     = threadIdx.x;
    const unsigned int blockId = blockIdx.x;
    const unsigned int blockDimX = blockDim.x;

    // Compute lane ID within warp: 0..31
    const unsigned int laneId = tid & (WARP_SIZE - 1);

    // Initialize shared-memory histogram to zero:
    // Each thread zeroes multiple elements if needed.
    const int shSize = range * WARP_SIZE;
    for (int i = tid; i < shSize; i += blockDimX) {
        shHist[i] = 0;
    }
    __syncthreads();

    // Global thread index
    const unsigned int globalThreadId = blockId * blockDimX + tid;
    // First input index processed by this thread
    unsigned int baseIndex = globalThreadId * ITEMS_PER_THREAD;

    // Process ITEMS_PER_THREAD characters per thread
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Only count characters within [from, to]
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {

            int bin = static_cast<int>(c) - from;  // 0 .. (range-1)

            // Each warp lane updates its own copy of the histogram bin.
            // Layout: shHist[bin * WARP_SIZE + laneId]
            // This ensures that, for a given bin, each lane in a warp
            // accesses a different shared-memory bank (no intra-warp conflicts).
            unsigned int shIndex = static_cast<unsigned int>(bin) * WARP_SIZE + laneId;

            // Multiple warps in the same block can still update the same
            // (bin, lane) pair; use atomicAdd to preserve correctness.
            atomicAdd(&shHist[shIndex], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 lane-specific copies per bin into a single value and
    // accumulate into global memory. Use at most `range` threads per block
    // for this step (others are idle).
    for (int bin = tid; bin < range; bin += blockDimX) {
        unsigned int sum = 0;
        int base = bin * WARP_SIZE;

        // Sum the contributions from all 32 lane copies for this bin
        #pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) {
            sum += shHist[base + l];
        }

        // Only perform global atomicAdd when there is something to add
        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host helper: launch the histogram kernel.
//
// Parameters:
//   input      - device pointer to input char buffer
//   histogram  - device pointer to output histogram (size: to - from + 1)
//   inputSize  - number of characters in `input`
//   from, to   - inclusive character range [from, to] (0 <= from < to <= 255)
//
// Notes:
// - This function assumes `input` and `histogram` are allocated with cudaMalloc.
// - The caller is responsible for any synchronization (e.g., cudaDeviceSynchronize)
//   and error checking after this function returns.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation (defensive, in case of misuse).
    if (input == nullptr || histogram == nullptr) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int range = to - from + 1;

    // Zero the output histogram on the device.
    // Using cudaMemsetAsync with the default stream allows the caller to
    // synchronize later if desired.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), 0);

    // Choose a typical block size for modern GPUs.
    const int blockSize = 256;

    // Compute the number of threads needed to cover the input such that
    // each thread processes ITEMS_PER_THREAD elements (or fewer at the tail).
    unsigned long long totalThreadsNeeded =
        (static_cast<unsigned long long>(inputSize) + ITEMS_PER_THREAD - 1ULL) / ITEMS_PER_THREAD;

    // Compute grid size (number of blocks).
    unsigned int gridSize = static_cast<unsigned int>(
        (totalThreadsNeeded + static_cast<unsigned long long>(blockSize) - 1ULL) /
        static_cast<unsigned long long>(blockSize));

    if (gridSize == 0) {
        gridSize = 1;  // Handle inputSize == 0 gracefully.
    }

    // Shared memory size per block: range bins * 32 lane copies * sizeof(unsigned int)
    size_t sharedMemSize = static_cast<size_t>(range) * WARP_SIZE * sizeof(unsigned int);

    // Launch the kernel in the default stream (stream 0).
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}