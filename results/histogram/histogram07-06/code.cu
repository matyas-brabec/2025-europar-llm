#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

/*
 * CUDA histogram kernel with shared-memory privatization.
 *
 * - Computes a histogram of the input characters restricted to the inclusive range [from, to].
 * - 'from' and 'to' are integer character codes in [0,255], with from <= to.
 * - The output histogram has (to - from + 1) bins; bin i corresponds to character (from + i).
 *
 * Optimization details:
 *
 * - Shared memory privatization:
 *   Each thread block keeps a private histogram in shared memory to reduce global memory
 *   contention. At the end of the kernel, per-block histograms are merged into the global
 *   histogram using atomic operations.
 *
 * - 32 histogram copies per block (one per warp lane):
 *   To avoid shared memory bank conflicts when multiple threads in a warp update the same
 *   histogram bin, the histogram is replicated 32 times. Copy 'c' (0 <= c < 32) is used
 *   by all threads whose threadIdx.x % 32 == c.
 *
 *   Layout in shared memory:
 *     For bin index i (0 <= i < numBins) and copy c (0 <= c < 32),
 *       the element is stored at s_hist[i * 32 + c].
 *
 *   This layout ensures that when all 32 lanes in a warp increment the same bin index i,
 *   they access addresses in different shared memory banks (no bank conflicts).
 *
 * - itemsPerThread:
 *   Each thread processes a fixed number of input characters, controlled by the constant
 *   'itemsPerThread'. For large inputs on modern data-center GPUs such as A100 / H100,
 *   a value of 16 typically provides a good balance between memory-level parallelism and
 *   occupancy.
 *
 * - Grid-stride loop with bounded iterations:
 *   The kernel uses a grid-stride loop where each thread processes up to 'itemsPerThread'
 *   characters at positions:
 *       idx = globalThreadId + k * (gridDim.x * blockDim.x),  for k = 0..itemsPerThread-1,
 *   stopping early if idx >= inputSize.
 */

static constexpr int itemsPerThread = 16;       // Tunable: chars processed per thread
static constexpr int NUM_HISTO_COPIES = 32;     // One copy per warp lane

__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int s_hist[];    // Size: numBins * NUM_HISTO_COPIES

    const int numBins = to - from + 1;
    const int totalSharedBins = numBins * NUM_HISTO_COPIES;

    // Zero out shared histogram (all copies) cooperatively.
    for (int idx = threadIdx.x; idx < totalSharedBins; idx += blockDim.x) {
        s_hist[idx] = 0;
    }
    __syncthreads();

    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads   = gridDim.x * blockDim.x;
    const int lane           = threadIdx.x & (NUM_HISTO_COPIES - 1); // threadIdx.x % 32

    // Each thread processes up to itemsPerThread characters using a grid-stride pattern.
    for (int k = 0; k < itemsPerThread; ++k) {
        unsigned int idx = static_cast<unsigned int>(globalThreadId) +
                           static_cast<unsigned int>(k) * static_cast<unsigned int>(totalThreads);
        if (idx >= inputSize) {
            break;
        }

        unsigned char ch = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(ch) - from;
        if (bin >= 0 && bin < numBins) {
            int sharedIdx = bin * NUM_HISTO_COPIES + lane;
            // Shared memory atomics are fast on modern architectures.
            atomicAdd(&s_hist[sharedIdx], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane copies into a single value per bin and add to global histogram.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * NUM_HISTO_COPIES;

        #pragma unroll
        for (int c = 0; c < NUM_HISTO_COPIES; ++c) {
            sum += s_hist[base + c];
        }

        if (sum > 0) {
            // Multiple thread blocks contribute to the same global bins, hence atomicAdd.
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host wrapper to launch the histogram kernel.
 *
 * Parameters:
 *   input      - Device pointer to input text buffer (chars) allocated via cudaMalloc.
 *   histogram  - Device pointer to output histogram (unsigned ints) allocated via cudaMalloc.
 *                Must have at least (to - from + 1) elements.
 *   inputSize  - Number of characters in the input buffer.
 *   from, to   - Inclusive character code range [from, to] to histogram (0 <= from <= to <= 255).
 *
 * Behavior:
 *   - The function zeros out the output histogram on the device.
 *   - It configures and launches the CUDA kernel with dynamically allocated shared memory
 *     sufficient for 32 copies of the sub-range histogram.
 *   - Any stream synchronization or error checking is assumed to be handled by the caller.
 */

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Validate range; assume caller provides sane values per problem statement.
    const int numBins = to - from + 1;
    if (numBins <= 0) {
        return;
    }

    // Clear the output histogram on device (asynchronously, in the current stream).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a reasonable block size for modern GPUs.
    constexpr int blockSize = 256;

    // Number of blocks so that total threads * itemsPerThread >= inputSize.
    unsigned int blocks =
        (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
    if (blocks == 0) {
        blocks = 1;
    }

    // Shared memory size: 32 copies of the histogram, each with numBins entries.
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * NUM_HISTO_COPIES * sizeof(unsigned int);

    // Launch kernel. Synchronization and error checking are handled by the caller.
    histogramKernel<<<blocks, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}