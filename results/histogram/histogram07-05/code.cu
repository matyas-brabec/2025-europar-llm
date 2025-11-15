#include <cuda_runtime.h>

/*
 * Optimized histogram kernel for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - Computes histogram of input characters restricted to [from, to] (inclusive).
 * - Uses shared-memory privatization with 32 copies of the histogram per block
 *   to reduce contention and avoid shared-memory bank conflicts.
 * - Layout of shared histogram:
 *       sharedHist[ bin * 32 + copy ]
 *   where:
 *       bin  in [0, numBins)
 *       copy in [0, 31] (selected as threadIdx.x % 32)
 * - Each thread processes `itemsPerThread` input characters.
 */

static constexpr int WARP_SIZE      = 32;
static constexpr int NUM_COPIES     = 32;   // Required to match the 32 shared-memory banks
static constexpr int itemsPerThread = 16;   // Tuned for large inputs on recent data-center GPUs

// Compile-time sanity check
static_assert(NUM_COPIES == WARP_SIZE, "NUM_COPIES must equal WARP_SIZE (32) for bank-aware layout.");

__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ globalHist,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    // Dynamic shared memory: 32 copies of [from..to] histogram
    extern __shared__ unsigned int sharedHist[];

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int blockThreads = blockDim.x;

    // Total number of entries in shared histogram (32 copies)
    const int sharedSize = numBins * NUM_COPIES;

    // ------------------------------------------------------------------------
    // 1. Initialize shared histogram to zero
    // ------------------------------------------------------------------------
    for (int i = tid; i < sharedSize; i += blockThreads) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // 2. Each thread processes up to `itemsPerThread` characters
    //    Characters are assigned in contiguous chunks:
    //      baseIndex = globalThreadId * itemsPerThread
    // ------------------------------------------------------------------------
    const unsigned int globalThreadId =
        static_cast<unsigned int>(blockIdx.x) * blockThreads + static_cast<unsigned int>(tid);

    const unsigned int baseIndex = globalThreadId * itemsPerThread;

    const int copyIndex = tid % NUM_COPIES;  // Shared histogram copy chosen for this thread

    for (int item = 0; item < itemsPerThread; ++item) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(item);
        if (idx >= inputSize)
            break;

        // Convert char to unsigned and map into bin index relative to `from`
        int val = static_cast<unsigned char>(input[idx]) - from;

        // Use unsigned comparison to check 0 <= val < numBins
        if (static_cast<unsigned int>(val) < static_cast<unsigned int>(numBins)) {
            int bin = val;
            int offset = bin * NUM_COPIES + copyIndex;

            // Shared-memory atomic is required because multiple warps in the block
            // may update the same (bin, copy) concurrently.
            atomicAdd(&sharedHist[offset], 1u);
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // 3. Reduce the 32 copies per bin into a single value and update global histogram
    // ------------------------------------------------------------------------
    for (int bin = tid; bin < numBins; bin += blockThreads) {
        unsigned int sum = 0;
        int base = bin * NUM_COPIES;

        #pragma unroll
        for (int c = 0; c < NUM_COPIES; ++c) {
            sum += sharedHist[base + c];
        }

        if (sum > 0) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 *   - input      : device pointer to input chars (cudaMalloc'd), size `inputSize`
 *   - histogram  : device pointer to histogram array (cudaMalloc'd),
 *                  size `(to - from + 1)` unsigned ints
 *   - inputSize  : number of characters in `input`
 *   - from, to   : inclusive character range [from, to], with 0 <= from <= to <= 255
 *
 * Notes:
 *   - This function assumes default stream (0). All operations are asynchronous
 *     with respect to the host; the caller is responsible for synchronization.
 *   - The histogram buffer is zeroed before the kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic validation; if nothing to do, just return.
    if (input == nullptr || histogram == nullptr)
        return;
    if (inputSize == 0)
        return;
    if (from > to)
        return;

    const int numBins = to - from + 1;

    // Zero the output histogram on device (asynchronously on default stream)
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // Kernel launch configuration
    const int threadsPerBlock = 256;

    // Each thread processes `itemsPerThread` elements.
    const unsigned int workPerBlock = static_cast<unsigned int>(threadsPerBlock * itemsPerThread);

    // Number of blocks needed to cover all input characters
    unsigned int blocks =
        (inputSize + workPerBlock - 1u) / workPerBlock;
    if (blocks == 0)
        blocks = 1;

    // Shared memory size in bytes: numBins * NUM_COPIES * sizeof(unsigned int)
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * NUM_COPIES * sizeof(unsigned int);

    // Launch the kernel on the default stream
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to
    );

    // No synchronization here; caller is responsible for it.
}