#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - Computes a histogram over a contiguous character range [from, to].
 * - Input:  plain-text buffer `input` of length `inputSize` bytes (on device).
 * - Output: `histogram` array of unsigned int counts for each character in [from, to].
 *
 * Design:
 * - Per-block histogram privatization in shared memory:
 *     * Each block accumulates counts into a shared-memory histogram for the
 *       requested range.
 *     * After processing its portion of the input, each block atomically
 *       adds its shared histogram into the global histogram.
 *
 * - itemsPerThread (ITEMS_PER_THREAD) controls how many input characters each
 *   thread processes. The chosen default is 16, which works well for large
 *   inputs on modern data-center GPUs, providing a good balance between
 *   memory-level parallelism and scheduling flexibility.
 *
 * - Grid-stride loop with ITEMS_PER_THREAD-way unrolling ensures that:
 *     * The entire input is processed regardless of grid size.
 *     * Global memory loads are fully coalesced across threads in a warp.
 *
 * - Input characters are treated as unsigned bytes [0, 255].
 * - Only characters whose ordinal value lies within [from, to] are counted.
 */

static constexpr int ITEMS_PER_THREAD   = 16;   // Tunable processing factor per thread
static constexpr int THREADS_PER_BLOCK  = 256;  // Tunable thread block size (must be a multiple of 32)

/**
 * CUDA kernel: compute histogram of characters in a given range [from, to].
 *
 * @param input      Device pointer to input characters.
 * @param histogram  Device pointer to global histogram (size: to - from + 1).
 * @param inputSize  Number of characters in the input buffer.
 * @param from       Start of character range (inclusive), 0 <= from <= 255.
 * @param to         End of character range (inclusive), from <= to <= 255.
 *
 * Dynamic shared memory layout:
 *   sharedHist[0 .. (range_len - 1)] : per-block histogram for range [from, to]
 */
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sharedHist[];

    const int range_start = from;
    const int range_len   = to - from + 1;

    // Initialize shared histogram to zero.
    // Each thread zeros multiple bins in a strided fashion.
    for (int i = threadIdx.x; i < range_len; i += blockDim.x) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int threadId     = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = gridDim.x * blockDim.x;

    // Stride for the outer loop: every iteration each thread will process
    // ITEMS_PER_THREAD items, spaced by 'totalThreads' between them.
    const unsigned int outerStride = totalThreads * ITEMS_PER_THREAD;

    // Grid-stride outer loop:
    //   - idx is the base index for this iteration for the current thread.
    //   - For each outer loop iteration, we process up to ITEMS_PER_THREAD
    //     positions: idx + k * totalThreads, k = 0..ITEMS_PER_THREAD-1.
    for (unsigned int idx = threadId; idx < inputSize; idx += outerStride) {

#pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            unsigned int i = idx + static_cast<unsigned int>(k) * totalThreads;
            if (i >= inputSize) {
                break;
            }

            // Load input character, treating it as unsigned [0, 255].
            unsigned char c = static_cast<unsigned char>(input[i]);

            // Compute bin index relative to 'from'.
            // Use unsigned comparison for a single bounds check:
            //   0 <= bin < range_len   <=>   (unsigned)bin < (unsigned)range_len
            int bin = static_cast<int>(c) - range_start;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range_len)) {
                // Atomic add into per-block shared histogram.
                atomicAdd(&sharedHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Merge per-block histogram into the global histogram.
    // Each thread handles multiple bins, using atomicAdd to handle inter-block
    // contention on the same global bins.
    for (int i = threadIdx.x; i < range_len; i += blockDim.x) {
        unsigned int val = sharedHist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/**
 * Host function: launch the CUDA histogram kernel with appropriate parameters.
 *
 * This function assumes that:
 *   - `input` and `histogram` are device pointers allocated with cudaMalloc.
 *   - `histogram` has size at least (to - from + 1) * sizeof(unsigned int).
 *   - Host-device synchronization (e.g., cudaDeviceSynchronize) is performed
 *     by the caller if needed.
 *
 * The function:
 *   - Zeros the output histogram on the device.
 *   - Chooses a reasonable grid size for the given input size.
 *   - Launches the CUDA kernel with shared memory sized to the range length.
 *
 * @param input       Device pointer to input chars.
 * @param histogram   Device pointer to histogram (device memory).
 * @param inputSize   Number of chars in the input buffer.
 * @param from        Start of the character range (inclusive).
 * @param to          End of the character range (inclusive).
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int rangeLen = to - from + 1;

    // Zero the histogram output buffer on the device.
    // This is asynchronous with respect to the host.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // If input is empty, nothing more to do.
    if (inputSize == 0) {
        return;
    }

    // Determine grid size.
    // Each thread processes ITEMS_PER_THREAD items in each outer loop "tile".
    const unsigned int threadsPerBlock = THREADS_PER_BLOCK;

    // Aim for roughly one block per (threadsPerBlock * ITEMS_PER_THREAD) elements.
    unsigned int blocks = (inputSize + (threadsPerBlock * ITEMS_PER_THREAD - 1))
                          / (threadsPerBlock * ITEMS_PER_THREAD);

    if (blocks == 0) {
        blocks = 1;
    }

    // Clamp blocks to a safe upper bound for grid.x (compatible with all devices).
    const unsigned int maxBlocks = 65535u;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Shared memory size: one unsigned int per histogram bin in the requested range.
    const size_t sharedMemSize = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch the kernel. Execution order relative to cudaMemset is preserved in
    // the default stream without explicit synchronization.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}