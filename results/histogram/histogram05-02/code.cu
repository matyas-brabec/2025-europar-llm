#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * Key features:
 *  - Each block maintains multiple privatized histograms in shared memory
 *    (one per warp) to reduce contention.
 *  - The shared histograms are padded (stride = MAX_BINS + 1) to mitigate
 *    shared memory bank conflicts.
 *  - Each thread processes `itemsPerThread` input characters to amortize
 *    kernel launch and indexing overhead for large inputs.
 *  - Shared-memory atomics are used for per-block accumulation, followed
 *    by a block-wide reduction and a single set of global atomics.
 */

static constexpr int WARP_SIZE       = 32;
static constexpr int BLOCK_SIZE      = 256;   // 8 warps per block is a good fit for A100/H100
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
static constexpr int MAX_BINS        = 256;   // char range [0,255]
static constexpr int HIST_STRIDE     = MAX_BINS + 1; // +1 padding to reduce bank conflicts

// Number of input characters processed by each thread.
// 8 is a good default for large inputs on data center GPUs: enough work per thread
// without excessively increasing register pressure.
static constexpr int itemsPerThread  = 8;

/*
 * CUDA kernel: compute histogram over a subrange [from, to] (inclusive).
 *
 * Parameters:
 *  - input:       device pointer to the input character array
 *  - globalHist:  device pointer to the histogram array of size (to - from + 1)
 *  - inputSize:   number of characters in the input
 *  - from, to:    character range [from, to] (0 <= from <= to <= 255)
 */
__global__
void histogramKernel(const char * __restrict__ input,
                     unsigned int * __restrict__ globalHist,
                     unsigned int inputSize,
                     int from,
                     int to)
{
    // Shared memory layout:
    // We have WARPS_PER_BLOCK independent histograms.
    // Each histogram has HIST_STRIDE bins, where only the first (rangeSize)
    // bins are actually used; the extra padding helps avoid bank conflicts.
    __shared__ unsigned int sHist[WARPS_PER_BLOCK * HIST_STRIDE];

    const int tid    = threadIdx.x;
    const int bid    = blockIdx.x;
    const int gtid   = bid * blockDim.x + tid;
    const int rangeSize = to - from + 1;

    if (rangeSize <= 0) {
        return;
    }

    // Zero the shared histograms.
    // We clear the full allocated shared array; this is small (≈8 KB for
    // WARPS_PER_BLOCK=8, HIST_STRIDE=257) and simplifies the code.
    for (int i = tid; i < WARPS_PER_BLOCK * HIST_STRIDE; i += blockDim.x) {
        sHist[i] = 0;
    }
    __syncthreads();

    // Each warp has its own privatized histogram.
    const int warpId  = tid / WARP_SIZE;
    const int laneId  = tid % WARP_SIZE;
    const int histBase = warpId * HIST_STRIDE;

    // Each thread processes itemsPerThread consecutive characters.
    const unsigned int baseIndex = gtid * itemsPerThread;

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + i;
        if (idx >= inputSize) {
            break;  // input may not be a multiple of itemsPerThread * #threads
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {

            int bin = static_cast<int>(c) - from;  // bin in [0, rangeSize-1]

            // Update the warp-private histogram in shared memory.
            // Shared-memory atomics are fast on modern GPUs.
            atomicAdd(&sHist[histBase + bin], 1u);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // Each thread accumulates a subset of bins across all warps and performs
    // a single global atomicAdd per bin.
    for (int bin = tid; bin < rangeSize; bin += blockDim.x) {
        unsigned int sum = 0;

        // Sum this bin across all warp-private histograms.
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += sHist[w * HIST_STRIDE + bin];
        }

        if (sum > 0) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

/*
 * Host-side wrapper: configure and launch the histogram kernel.
 *
 * Parameters:
 *  - input:     device pointer to input characters (cudaMalloc'ed)
 *  - histogram: device pointer to histogram buffer of size (to - from + 1)
 *  - inputSize: number of characters in input
 *  - from, to:  character range [from, to]
 *
 * Assumptions:
 *  - `input` and `histogram` are valid device pointers.
 *  - Any necessary host-device synchronization (e.g., cudaDeviceSynchronize)
 *    is handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0) {
        return;
    }

    // Clamp the range to valid char values, in case the caller passes
    // something slightly out of bounds.
    if (from < 0)   from = 0;
    if (to > 255)   to   = 255;
    if (to < from)  return;

    const int rangeSize = to - from + 1;

    // Clear the global histogram on the device.
    // This is asynchronous with respect to the host.
    cudaMemset(histogram, 0, rangeSize * sizeof(unsigned int));

    // Determine grid size.
    const unsigned int charsPerBlock = BLOCK_SIZE * itemsPerThread;
    unsigned int gridSize = (inputSize + charsPerBlock - 1) / charsPerBlock;

    if (gridSize == 0) {
        return;
    }

    // Launch the kernel.
    dim3 block(BLOCK_SIZE);
    dim3 grid(gridSize);

    histogramKernel<<<grid, block>>>(input, histogram, inputSize, from, to);
}