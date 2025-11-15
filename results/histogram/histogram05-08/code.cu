#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA data-center GPUs (e.g., A100/H100).
 *
 * Features:
 *  - Shared-memory privatization of the histogram per thread block.
 *  - Multiple copies of the histogram per block to reduce contention and bank conflicts.
 *  - Configurable number of input items processed per thread via `itemsPerThread`.
 *  - Coalesced global memory reads.
 *
 * Histogram semantics:
 *  - Input:  `input`  - device pointer to an array of chars (plain text).
 *  - Range:  [from, to], with 0 <= from < to <= 255.
 *  - Output: `histogram` - device pointer to an array of (to - from + 1) unsigned ints.
 *             histogram[i] accumulates the count of character (unsigned char)(from + i).
 *
 * The caller must provide device memory for `input` and `histogram` using cudaMalloc.
 * The caller is also responsible for any required synchronization after run_histogram().
 */

namespace {

/* Number of characters each thread processes.
 * A value of 8 is a good default for large inputs on A100/H100-class GPUs:
 * it keeps global reads coalesced, gives enough work per thread to hide latency,
 * and still allows high occupancy with modest register pressure.
 */
static constexpr int itemsPerThread   = 8;

/* Threads per block: 256 is a commonly optimal choice on recent GPUs,
 * balancing occupancy and per-block resource usage.
 */
static constexpr int threadsPerBlock  = 256;

/* Number of privatized histogram copies per block.
 * More copies reduce atomic contention and shared-memory bank conflicts,
 * at the cost of a small per-block reduction overhead.
 * 8 copies * 256 bins * 4 bytes ≈ 8 KB shared memory (plus padding), which is small.
 */
static constexpr int histCopies       = 8;

/*
 * CUDA kernel to compute a partial histogram over a given character range.
 */
__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    // Range size (number of bins).
    const int range = to - from + 1;

    // Thread identifiers.
    const int tid      = threadIdx.x;
    const int nThreads = blockDim.x;

    // Each block processes a contiguous chunk of the input.
    const unsigned int blockItems = static_cast<unsigned int>(nThreads) *
                                    static_cast<unsigned int>(itemsPerThread);
    const unsigned int blockBase  = blockIdx.x * blockItems;

    // Dynamic shared memory layout:
    // sharedHist[copy][bin], with padding to reduce bank conflicts.
    extern __shared__ unsigned int sharedHist[];

    // Padding: make each histogram row (per copy) have a non-multiple-of-32 stride
    // to reduce shared-memory bank conflicts between copies.
    const int rowStride   = range + 1;              // one extra element per copy
    const int totalShared = histCopies * rowStride; // total shared memory slots

    // Initialize shared histograms to zero (cooperatively).
    for (int i = tid; i < totalShared; i += nThreads) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Assign each thread to one of the privatized histogram copies to reduce contention.
    const int copyIdx = tid % histCopies;
    unsigned int *localHist = sharedHist + copyIdx * rowStride;

    // Process itemsPerThread characters per thread.
    // Mapping: each block covers [blockBase, blockBase + blockItems),
    //          and each thread strides by blockDim.x within that.
    const unsigned int threadBaseIndex = blockBase + tid;

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = threadBaseIndex + static_cast<unsigned int>(i) *
                                             static_cast<unsigned int>(nThreads);
        if (idx >= inputSize) {
            break; // Out of bounds
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Only count characters in the requested [from, to] range.
        if (bin >= 0 && bin < range) {
            // Shared-memory atomic add: fast on modern GPUs.
            atomicAdd(&localHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce privatized histograms into the global histogram.
    // Each thread is responsible for a strided subset of bins.
    for (int bin = tid; bin < range; bin += nThreads) {
        unsigned int sum = 0;
        // Sum across all copies.
        for (int copy = 0; copy < histCopies; ++copy) {
            sum += sharedHist[copy * rowStride + bin];
        }
        // Atomically add the block's contribution to the global histogram.
        if (sum > 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

} // anonymous namespace

/*
 * Host function that configures and launches the CUDA histogram kernel.
 *
 * Parameters:
 *  - input      : device pointer to an array of chars (input text).
 *  - histogram  : device pointer to an array of unsigned ints of size (to - from + 1).
 *  - inputSize  : number of chars in the input buffer.
 *  - from, to   : inclusive range of character codes to accumulate (0 <= from < to <= 255).
 *
 * The function:
 *  1. Clears the histogram buffer on the device.
 *  2. Chooses a launch configuration based on itemsPerThread and threadsPerBlock.
 *  3. Allocates the appropriate amount of dynamic shared memory for privatized histograms.
 *  4. Launches the kernel.
 *
 * Synchronization and error checking (if desired) should be handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0u || from > to) {
        // Nothing to do.
        return;
    }

    const int range = to - from + 1;

    // Clear the output histogram on the device so we start from zero.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Compute launch configuration.
    const int blockSize      = threadsPerBlock;
    const unsigned int itemsPerBlock =
        static_cast<unsigned int>(blockSize) * static_cast<unsigned int>(itemsPerThread);

    unsigned int numBlocks = (inputSize + itemsPerBlock - 1u) / itemsPerBlock;
    if (numBlocks == 0u) {
        numBlocks = 1u; // Fallback, though inputSize > 0 implies numBlocks >= 1
    }

    // Dynamic shared memory size (in bytes):
    // histCopies * (range + 1 padding) * sizeof(unsigned int)
    const size_t sharedMemSize =
        static_cast<size_t>(histCopies) *
        static_cast<size_t>(range + 1) *
        sizeof(unsigned int);

    // Launch the kernel on the default stream.
    histogramKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );

    // No synchronization here; caller is responsible for it if needed.
}