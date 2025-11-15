#include <cuda_runtime.h>

// Number of input characters processed by each thread.
// For large inputs on modern GPUs (A100/H100), a value of 8 is a good balance
// between memory throughput and occupancy.
static constexpr int ITEMS_PER_THREAD = 8;

// Warp size on all current NVIDIA GPUs.
static constexpr int WARP_SIZE = 32;

/*
 * CUDA kernel: compute histogram over a specified character range [from, to].
 *
 * - input:      device pointer to a buffer of 'inputSize' chars.
 * - histogram:  device pointer to an array of (to - from + 1) unsigned ints.
 *               histogram[i] counts occurrences of character (i + from).
 * - inputSize:  number of characters in 'input'.
 * - from, to:   inclusive range of characters to count (0 <= from < to <= 255).
 *
 * Optimization strategy:
 * - Each block builds its histogram in shared memory, then merges into global.
 * - To reduce atomic contention and shared-memory bank conflicts, each warp
 *   maintains its own private copy of the histogram in shared memory.
 *   Finally, the per-warp histograms are reduced inside the block.
 * - Shared memory layout:
 *     [ warp 0 hist | warp 1 hist | ... | warp (warpCount-1) hist ]
 *   where each "hist" section has 'binsPadded' entries, padded to a multiple
 *   of WARP_SIZE to align with shared-memory banks.
 * - Each thread processes ITEMS_PER_THREAD consecutive characters. Grid size
 *   is chosen so that all input bytes are covered exactly once.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    const int range        = to - from + 1;     // number of histogram bins
    const int blockThreads = blockDim.x;
    const int threadId     = threadIdx.x;
    const int globalThread = blockIdx.x * blockThreads + threadId;

    // Number of warps in this block.
    const int warpCount = (blockThreads + WARP_SIZE - 1) / WARP_SIZE;

    // Stride (in bins) for each per-warp histogram, padded to avoid
    // cross-warp bank conflicts by aligning each copy to a full bank set.
    const int binsPadded = ((range + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Shared memory backing for all per-warp histograms.
    extern __shared__ unsigned int shHist[];

    // Pointer to this warp's private histogram copy.
    const int warpId = threadId / WARP_SIZE;
    unsigned int* const myHist = shHist + warpId * binsPadded;

    // ------------------------------------------------------------------------
    // 1) Initialize shared histograms to zero.
    //    All threads in the block cooperate to zero all warp copies.
    // ------------------------------------------------------------------------
    for (int i = threadId; i < warpCount * binsPadded; i += blockThreads) {
        shHist[i] = 0;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // 2) Each thread processes ITEMS_PER_THREAD characters.
    //
    //    Thread t processes input indices:
    //      base = t * ITEMS_PER_THREAD
    //      [base, base + ITEMS_PER_THREAD)
    //    (clamped by inputSize).
    //
    //    Hits in [from, to] are accumulated into the warp-private histogram.
    // ------------------------------------------------------------------------
    const unsigned int baseIndex = static_cast<unsigned int>(globalThread) * ITEMS_PER_THREAD;

    if (baseIndex < inputSize) {
        const unsigned char* __restrict__ data =
            reinterpret_cast<const unsigned char*>(input);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const unsigned int idx = baseIndex + i;
            if (idx >= inputSize) {
                break;
            }

            const unsigned char c = data[idx];
            if (c >= static_cast<unsigned char>(from) &&
                c <= static_cast<unsigned char>(to)) {

                const int bin = static_cast<int>(c) - from;
                // Shared-memory atomic add: very fast on modern GPUs.
                atomicAdd(&myHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // 3) Reduce per-warp histograms into global histogram.
    //
    //    For each bin 'b', sum shHist[w * binsPadded + b] over all warps 'w'
    //    in the block, then atomically add the result into the global histogram.
    //    The reduction is parallelized by distributing bins across threads.
    // ------------------------------------------------------------------------
    for (int bin = threadId; bin < range; bin += blockThreads) {
        unsigned int sum = 0;
        for (int w = 0; w < warpCount; ++w) {
            sum += shHist[w * binsPadded + bin];
        }
        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host helper function to launch the histogram kernel.
 *
 * Parameters:
 * - input:      device pointer to input character buffer (cudaMalloc'ed).
 * - histogram:  device pointer to histogram array (cudaMalloc'ed).
 * - inputSize:  number of characters in 'input'.
 * - from, to:   inclusive character range to histogram (0 <= from < to <= 255).
 *
 * Notes:
 * - This function assumes that any required host-device synchronization
 *   (e.g., cudaDeviceSynchronize) is handled by the caller.
 * - The histogram buffer is zeroed before the kernel launch.
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

    const int range = to - from + 1;

    // Initialize global histogram to zero.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Choose a block size that gives good occupancy and enough warps
    // to amortize shared-memory histogram initialization.
    const int threadsPerBlock = 256;

    // Number of input elements handled by a single block.
    const unsigned int elementsPerBlock =
        static_cast<unsigned int>(threadsPerBlock) * ITEMS_PER_THREAD;

    // Grid size chosen so that all input characters are covered.
    const unsigned int gridSize =
        (inputSize + elementsPerBlock - 1) / elementsPerBlock;

    if (gridSize == 0) {
        return;
    }

    // Shared memory size: per-warp histogram copies, padded to avoid
    // shared-memory bank conflicts between warps.
    const int warpCount  = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
    const int binsPadded = ((range + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    const size_t shMemSize =
        static_cast<size_t>(warpCount) * binsPadded * sizeof(unsigned int);

    histogram_kernel<<<gridSize, threadsPerBlock, shMemSize>>>(
        input, histogram, inputSize, from, to);
}