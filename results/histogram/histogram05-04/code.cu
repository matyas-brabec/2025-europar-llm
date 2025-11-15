#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA data center GPUs (e.g., A100/H100).
 *
 * Features:
 *  - Each block uses multiple private histograms in shared memory (one per warp).
 *    This reduces contention and shared-memory bank conflicts compared to a single
 *    shared histogram per block.
 *  - Each thread processes a fixed number of input characters: itemsPerThread.
 *    This increases arithmetic intensity and improves memory throughput for large inputs.
 *  - Per-warp shared histograms are reduced to a single global histogram via global atomics.
 *
 * The histogram is restricted to a continuous character range [from, to], where
 * 0 <= from < to <= 255. The output histogram has (to - from + 1) bins, and
 * histogram[i] counts occurrences of the character with value (from + i).
 */

// Number of input characters processed by each thread.
// This value is chosen to balance memory throughput and occupancy on modern GPUs.
static constexpr int itemsPerThread = 16;

// Threads per block. Must be a multiple of 32 (warp size).
static constexpr int BLOCK_SIZE = 256;

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                int from, int to)
{
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size (32).");

    extern __shared__ unsigned int sharedHist[];

    const int tid  = threadIdx.x;
    const int warpId  = tid >> 5;       // threadIdx.x / 32
    const int laneId  = tid & 31;       // threadIdx.x % 32
    const int warpsPerBlock = BLOCK_SIZE / 32;

    // Convert range bounds to unsigned char for efficient comparison.
    const unsigned char u_from = static_cast<unsigned char>(from);
    const unsigned char u_to   = static_cast<unsigned char>(to);
    const unsigned int  rangeSize = static_cast<unsigned int>(u_to - u_from + 1);

    // Layout of sharedHist: [warp0_hist | warp1_hist | ... | warp(warpsPerBlock-1)_hist]
    // Each warp's private histogram has 'rangeSize' bins.
    unsigned int* warpHist = sharedHist + warpId * rangeSize;

    // Initialize all per-warp histograms in shared memory to zero.
    for (unsigned int i = tid; i < static_cast<unsigned int>(warpsPerBlock) * rangeSize; i += BLOCK_SIZE) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int n = inputSize;
    const unsigned int globalThreadId = blockIdx.x * BLOCK_SIZE + tid;
    const unsigned int baseIndex = globalThreadId * ITEMS_PER_THREAD;

    // Each thread processes up to ITEMS_PER_THREAD consecutive characters.
    // This produces coalesced loads when BLOCK_SIZE is a multiple of 32.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= n) {
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= u_from && c <= u_to) {
            unsigned int bin = static_cast<unsigned int>(c - u_from);
            // Use shared-memory atomicAdd to accumulate into this warp's private histogram.
            atomicAdd(&warpHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // Only the first warp in each block participates in the reduction.
    if (warpId == 0) {
        for (unsigned int bin = static_cast<unsigned int>(laneId); bin < rangeSize; bin += 32u) {
            unsigned int sum = 0;
            #pragma unroll
            for (int w = 0; w < warpsPerBlock; ++w) {
                sum += sharedHist[w * rangeSize + bin];
            }

            if (sum > 0u) {
                // Global atomic add to accumulate per-block results.
                atomicAdd(&histogram[bin], sum);
            }
        }
    }
}

/*
 * Host-side wrapper that configures and launches the CUDA histogram kernel.
 *
 * Parameters:
 *  - input:      Device pointer to input text buffer (array of chars).
 *  - histogram:  Device pointer to output histogram buffer (array of unsigned ints).
 *                Must have space for (to - from + 1) elements, allocated with cudaMalloc.
 *  - inputSize:  Number of characters in the input buffer.
 *  - from, to:   Inclusive character range [from, to] (0 <= from < to <= 255).
 *
 * Notes:
 *  - This function initializes the device-side histogram buffer to zero.
 *  - All operations (cudaMemset and kernel launch) are asynchronous with respect
 *    to the host. The caller is responsible for performing any required
 *    synchronization (e.g., cudaDeviceSynchronize or stream synchronization) and
 *    error checking on CUDA API calls if desired.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute the number of bins in the requested range.
    const unsigned char u_from = static_cast<unsigned char>(from);
    const unsigned char u_to   = static_cast<unsigned char>(to);
    const unsigned int  rangeSize = static_cast<unsigned int>(u_to - u_from + 1);

    // Initialize the device histogram to zero.
    // The caller is responsible for synchronization.
    cudaMemset(histogram, 0, rangeSize * sizeof(unsigned int));

    if (inputSize == 0) {
        // Nothing to do; histogram remains zero.
        return;
    }

    const unsigned int threadsPerBlock = static_cast<unsigned int>(BLOCK_SIZE);

    // Total number of threads required if each thread processes itemsPerThread characters.
    const unsigned int totalThreadsNeeded =
        (inputSize + static_cast<unsigned int>(itemsPerThread) - 1u) /
        static_cast<unsigned int>(itemsPerThread);

    // Grid size such that gridSize * BLOCK_SIZE >= totalThreadsNeeded.
    const unsigned int gridSize =
        (totalThreadsNeeded + threadsPerBlock - 1u) / threadsPerBlock;

    if (gridSize == 0) {
        // Input size > 0 but configuration computed to zero (should not happen with above math),
        // still guard against launching an empty grid.
        return;
    }

    // One private histogram of 'rangeSize' bins per warp in the block.
    const unsigned int warpsPerBlock = threadsPerBlock / 32u;
    const size_t sharedMemSize =
        static_cast<size_t>(rangeSize) *
        static_cast<size_t>(warpsPerBlock) *
        sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<BLOCK_SIZE, itemsPerThread>
        <<<gridSize, threadsPerBlock, sharedMemSize>>>(input,
                                                       histogram,
                                                       inputSize,
                                                       from,
                                                       to);
}