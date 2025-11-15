#include <cuda_runtime.h>

/*
 * CUDA histogram of a text buffer over a restricted character range [from, to].
 *
 * Design:
 *   - Each block builds a private histogram in shared memory to reduce global
 *     memory contention.
 *   - To avoid shared-memory bank conflicts, we keep 32 copies of the histogram
 *     per block, one per warp lane. Copy c (0 <= c < 32) stores bin i at
 *     offset i * 32 + c. Each thread uses copy (threadIdx.x % 32).
 *   - Updates to shared memory histograms use atomicAdd, but only across
 *     warps (within a warp threads write to distinct banks/addresses).
 *   - At the end of the kernel, each block reduces its 32 copies into a single
 *     per-block histogram and atomically accumulates into the global histogram.
 *
 * Launch configuration:
 *   - blockSize: 256 threads per block
 *   - itemsPerThread: each thread processes 8 input characters
 *     => 256 * 8 = 2048 characters per block
 *   - These settings are chosen to provide good occupancy and memory-throughput
 *     balance on modern GPUs such as A100/H100 for large inputs.
 */

static constexpr int kBlockSize          = 256;
static constexpr int kHistogramCopies    = 32;  // One per warp lane
static constexpr int itemsPerThread      = 8;   // Characters processed per thread

// CUDA kernel implementing the histogram calculation.
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int shist[];  // Size: numBins * kHistogramCopies

    const int numBins = to - from + 1;
    const int lane    = threadIdx.x & (kHistogramCopies - 1);  // threadIdx.x % 32

    // 1. Initialize shared-memory histogram copies to zero.
    const int totalSharedElems = numBins * kHistogramCopies;
    for (int i = threadIdx.x; i < totalSharedElems; i += blockDim.x) {
        shist[i] = 0;
    }
    __syncthreads();

    // 2. Each thread processes itemsPerThread input characters.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int startIndex     = globalThreadId * itemsPerThread;

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = startIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            break;
        }

        // Promote to unsigned to avoid sign-extension issues with char.
        unsigned char ch = static_cast<unsigned char>(input[idx]);

        // Only count characters in the [from, to] range.
        if (ch >= static_cast<unsigned char>(from) &&
            ch <= static_cast<unsigned char>(to)) {

            int bin = static_cast<int>(ch) - from;  // 0 <= bin < numBins

            // Strided layout: bin i, copy c -> offset = i * kHistogramCopies + c
            unsigned int offset = static_cast<unsigned int>(bin) * kHistogramCopies
                                  + static_cast<unsigned int>(lane);

            // Atomic add in shared memory to handle cross-warp collisions.
            atomicAdd(&shist[offset], 1u);
        }
    }

    __syncthreads();

    // 3. Reduce the 32 copies per bin into a single value and accumulate
    //    into the global histogram.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * kHistogramCopies;

        #pragma unroll
        for (int c = 0; c < kHistogramCopies; ++c) {
            sum += shist[base + c];
        }

        if (sum > 0) {
            // Global histogram is of size numBins, aligned with [from, to].
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host wrapper function.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'ed)
 *   histogram  - device pointer to output histogram (cudaMalloc'ed)
 *                of size (to - from + 1) * sizeof(unsigned int)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character range [from, to] (0 <= from < to <= 255)
 *
 * Notes:
 *   - This function does not perform host-device synchronization; the caller
 *     is responsible for that (e.g., via cudaDeviceSynchronize or events).
 *   - The histogram buffer is zeroed before kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Ensure the global histogram starts from zero.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Compute launch configuration.
    const unsigned int charsPerBlock = static_cast<unsigned int>(kBlockSize) *
                                       static_cast<unsigned int>(itemsPerThread);

    unsigned int numBlocks = (inputSize + charsPerBlock - 1u) / charsPerBlock;
    if (numBlocks == 0u) {
        // For empty input, we still launch at least one block so that the
        // kernel is well-defined, although it will not update any bins.
        numBlocks = 1u;
    }

    // Shared memory size: numBins copies, each with kHistogramCopies entries.
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) *
        static_cast<size_t>(kHistogramCopies) *
        sizeof(unsigned int);

    histogram_kernel<<<numBlocks, kBlockSize, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}