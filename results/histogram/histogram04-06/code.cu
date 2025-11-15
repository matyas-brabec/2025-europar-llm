#include <cuda_runtime.h>

/*
 * CUDA histogram kernel with per-block privatization in shared memory.
 *
 *  - Computes histogram for characters in the inclusive range [from, to].
 *  - Input:  'input' is a device pointer to 'inputSize' chars.
 *  - Output: 'histogram' is a device pointer to (to - from + 1) unsigned ints.
 *
 * Optimization notes:
 *  - Each block maintains a private histogram in shared memory, which greatly
 *    reduces contention on global memory atomics.
 *  - Each thread processes 'itemsPerThread' characters to increase instruction-
 *    level parallelism and reduce scheduling overhead.
 *  - Mapping from threads to input indices is chosen to keep accesses within
 *    a warp contiguous and thus fully coalesced.
 *  - The caller (run_histogram) zeros the global histogram via cudaMemset
 *    before launching the kernel.
 */

// Number of input characters processed by each thread.
// A value of 8 is a good default for modern NVIDIA data-center GPUs (A100/H100),
// providing a balance between ILP and occupancy on large inputs.
static constexpr int itemsPerThread  = 8;

// Threads per block (tunable). 256 is a common sweet spot for histogram-like
// workloads on modern NVIDIA GPUs.
static constexpr int threadsPerBlock = 256;

__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    // Dynamic shared-memory array storing the per-block histogram.
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;
    const unsigned long long n = static_cast<unsigned long long>(inputSize);

    // Initialize shared histogram to zero. Threads cooperate in a strided loop.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        s_hist[bin] = 0;
    }
    __syncthreads();

    // Coalesced access pattern:
    // Each block processes a consecutive chunk of size blockDim.x * itemsPerThread.
    // Within that chunk, at each loop iteration 'i', all threads in the block
    // read consecutive elements.
    const unsigned long long blockOffset =
        static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(blockDim.x) *
        static_cast<unsigned long long>(itemsPerThread);

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned long long idx =
            blockOffset +
            static_cast<unsigned long long>(i) * static_cast<unsigned long long>(blockDim.x) +
            static_cast<unsigned long long>(threadIdx.x);

        if (idx >= n) {
            // This thread has no more elements to process.
            break;
        }

        // Load character and convert to unsigned to avoid sign-extension issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Only count characters within the requested [from, to] range.
        if (bin >= 0 && bin < numBins) {
            // Shared-memory atomic add: fast on modern GPUs and confined to the block.
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Flush per-block histogram to global memory.
    // Each thread handles a strided subset of bins.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int val = s_hist[bin];
        if (val != 0) {
            atomicAdd(&histogram[bin], val);
        }
    }
}

/*
 * Host helper that configures and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input chars (cudaMalloc'd).
 *   histogram  - device pointer to histogram array (cudaMalloc'd).
 *   inputSize  - number of chars in the input buffer.
 *   from, to   - inclusive character range [from, to], 0 <= from < to <= 255.
 *
 * Notes:
 *   - This function zeroes the device histogram memory before launching
 *     the kernel.
 *   - Any host-device synchronization (e.g., cudaDeviceSynchronize) is left
 *     to the caller, as requested.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Handle degenerate cases: ensure histogram is defined to be all zeros.
    if (numBins <= 0) {
        // Invalid range; nothing to do.
        return;
    }

    // Zero the global histogram buffer. This is cheap (<= 256 bins).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0) {
        // No data to process; histogram remains zeroed.
        return;
    }

    // Compute grid configuration.
    const unsigned int blockSize = threadsPerBlock;
    const unsigned long long elemsPerBlock =
        static_cast<unsigned long long>(blockSize) *
        static_cast<unsigned long long>(itemsPerThread);

    const unsigned long long totalElems = static_cast<unsigned long long>(inputSize);

    unsigned int gridSize = static_cast<unsigned int>(
        (totalElems + elemsPerBlock - 1ull) / elemsPerBlock);

    if (gridSize == 0) {
        gridSize = 1;
    }

    // Shared memory size per block: one unsigned int per bin.
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel. Synchronization (if needed) is the caller's responsibility.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}