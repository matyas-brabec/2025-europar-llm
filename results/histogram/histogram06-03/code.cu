#include <cuda_runtime.h>

/*
 * CUDA-optimized histogram kernel and host launcher.
 *
 * The kernel computes a histogram over a specified character range [from, to]
 * for a large input buffer of plain text. The input is a device pointer to
 * chars, and the output is a device pointer to an array of unsigned ints
 * (length = to - from + 1) holding counts per character.
 *
 * Optimization highlights:
 *  - Each thread processes `itemsPerThread` input characters to increase ILP.
 *  - Per-block histogram privatization in shared memory reduces global atomics.
 *  - 32 copies of the histogram are stored in shared memory with a stride of 32,
 *    so that for a given bin, each warp lane updates its own copy located in a
 *    distinct shared memory bank:
 *
 *        shared index = bin * WARP_SIZE + lane
 *
 *    Because the bank number is (index % 32), for bin*32 + lane we get
 *    bank = lane, which is conflict-free within a warp.
 *  - The reduction of the 32 privatized histograms to a single global histogram
 *    uses warp shuffles and maintains the same conflict-free access pattern.
 *
 * The host function `run_histogram`:
 *  - Assumes `input` and `histogram` are device pointers allocated by cudaMalloc.
 *  - Zeros the output histogram on the device.
 *  - Launches the kernel with a configuration tuned for modern NVIDIA data
 *    center GPUs (e.g., A100/H100) and large inputs.
 *  - Performs no synchronization; the caller is responsible for it.
 */

constexpr int WARP_SIZE       = 32;
constexpr int MAX_BINS        = 256;
/* Number of items processed per thread.
 * A value of 16 is a good balance of ILP and occupancy on modern GPUs
 * for large, bandwidth-bound workloads.
 */
constexpr int itemsPerThread  = 16;


/*
 * CUDA kernel that computes a histogram over [from, to] for the input buffer.
 *
 * Parameters:
 *   input      - device pointer to input characters
 *   globalHist - device pointer to output histogram of length (to - from + 1)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character range [from, to], 0 <= from <= to <= 255
 */
__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ globalHist,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    /* 32-way privatized histogram in shared memory:
     * Layout is [bin][lane] with a stride of WARP_SIZE.
     * Index into shared memory is: bin * WARP_SIZE + lane.
     *
     * MAX_BINS is 256, so this uses 256 * 32 * 4 bytes = 32KB of shared memory.
     */
    __shared__ unsigned int s_hist[WARP_SIZE * MAX_BINS];

    const int numBins = to - from + 1;
    const int lane    = threadIdx.x & (WARP_SIZE - 1);

    /* Each block processes blockDim.x * itemsPerThread input elements. */
    const unsigned int blockItems   = blockDim.x * itemsPerThread;
    const unsigned int blockOffset  = blockIdx.x * blockItems;
    const unsigned int firstIndex   = blockOffset + threadIdx.x;

    /* Initialize only the portion of the shared histogram needed for [from, to]. */
    const int totalSharedEntries = numBins * WARP_SIZE;
    for (int i = threadIdx.x; i < totalSharedEntries; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    /* Process input characters, privatizing updates into shared memory.
     *
     * Each thread processes itemsPerThread characters, with indices:
     *    firstIndex + i * blockDim.x   (for i = 0 .. itemsPerThread-1)
     *
     * For a character c in [from, to], we compute:
     *    bin = c - from
     * and update:
     *    s_hist[bin * WARP_SIZE + lane]
     *
     * This access pattern is bank-conflict-free: threads in a warp use the
     * same bin but different lane IDs, so addresses map to distinct banks.
     */
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = firstIndex + static_cast<unsigned int>(i) * blockDim.x;
        if (idx >= inputSize) {
            break;  // no more data for this thread
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Single range check using unsigned comparison: 0 <= bin < numBins
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            unsigned int offset = static_cast<unsigned int>(bin * WARP_SIZE + lane);
            atomicAdd(&s_hist[offset], 1u);
        }
    }

    __syncthreads();

    /* Reduce the 32 privatized histograms per bin and update global histogram.
     *
     * Only the first warp in the block participates in the reduction. For each
     * bin, each lane reads its own copy (s_hist[bin * WARP_SIZE + lane]) and
     * we perform a warp-wide reduction via shuffles. Lane 0 then atomically
     * accumulates the per-bin sum into the global histogram.
     *
     * Access during reduction keeps the same conflict-free pattern as updates.
     */
    if (threadIdx.x < WARP_SIZE) {
        for (int bin = 0; bin < numBins; ++bin) {
            unsigned int val = s_hist[bin * WARP_SIZE + lane];

            // Warp-wide reduction over 32 lanes
            #pragma unroll
            for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFFu, val, delta);
            }

            if (lane == 0 && val > 0u) {
                atomicAdd(&globalHist[bin], val);
            }
        }
    }
}


/*
 * Host function that sets up and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'ed)
 *   histogram  - device pointer to histogram buffer of length (to - from + 1)
 *                (cudaMalloc'ed, but contents may be uninitialized)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character range [from, to], 0 <= from <= to <= 255
 *
 * Notes:
 *   - This function does not perform any device synchronization; the caller
 *     is responsible for synchronization (e.g., cudaDeviceSynchronize or
 *     events).
 *   - All operations are issued on the default stream (0).
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Zero the output histogram on the device (async in default stream).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // Nothing to process for empty inputs.
    if (inputSize == 0) {
        return;
    }

    // Kernel launch configuration tuned for modern GPUs.
    const int blockSize = 256;  // 8 warps per block
    const unsigned int blockItems = static_cast<unsigned int>(blockSize * itemsPerThread);
    const unsigned int numBlocks  =
        (inputSize + blockItems - 1u) / blockItems;

    histogramKernel<<<numBlocks, blockSize>>>(input, histogram, inputSize, from, to);
}