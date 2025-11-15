#include <cuda_runtime.h>

/*
 * CUDA histogram over a restricted character range.
 *
 *  - The input is a device pointer to a buffer of chars (bytes).
 *  - The output is a device pointer to an array of unsigned int counters.
 *  - The histogram covers the continuous range [from, to] (inclusive).
 *  - The histogram array must have size (to - from + 1) unsigned ints.
 *
 *  The implementation uses:
 *    - Per-block histograms in shared memory to reduce global atomic contention.
 *    - Atomic adds in shared memory during accumulation.
 *    - A final reduction from per-block shared histograms to the global histogram.
 *
 *  itemsPerThread controls how many characters each thread processes.
 *  A value of 16 is a good balance for modern NVIDIA data center GPUs (A100/H100):
 *    - Enough work per thread to amortize control overhead and hide memory latency.
 *    - Not so much that register pressure or divergence becomes problematic.
 */

static constexpr int itemsPerThread = 16;

/*
 * Kernel: build a histogram for characters in [from, from + range - 1].
 *
 * Parameters:
 *   input      - device pointer to input chars
 *   histogram  - device pointer to global histogram (size = range)
 *   inputSize  - number of chars in input
 *   from       - first character code in the range
 *   range      - number of histogram bins (to - from + 1)
 *
 * Notes:
 *   - input and histogram are assumed to be properly allocated on the device.
 *   - histogram is assumed to be zero-initialized before the kernel launch.
 *   - Uses dynamic shared memory: range * sizeof(unsigned int).
 */
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int range)
{
    // Shared memory for per-block histogram.
    extern __shared__ unsigned int sHist[];

    // Initialize shared histogram to zero. Multiple threads cooperate.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        sHist[i] = 0u;
    }

    __syncthreads();

    // Global linear thread index.
    const unsigned int globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread is responsible for up to itemsPerThread characters,
    // starting from baseIndex, laid out contiguously.
    const unsigned int baseIndex = globalThreadId * itemsPerThread;

    // Process up to itemsPerThread characters per thread.
#pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            // For the last few threads in the grid, we may run out of work.
            break;
        }

        // Load character as unsigned to avoid sign-extension issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Compute bin index relative to 'from'.
        // localBin is in [-from, 255-from], but casting to unsigned and
        // comparing with 'range' handles the negative case efficiently.
        int localBin = static_cast<int>(c) - from;
        if (static_cast<unsigned int>(localBin) < static_cast<unsigned int>(range)) {
            // Atomic add to shared histogram bin.
            atomicAdd(&sHist[localBin], 1u);
        }
    }

    __syncthreads();

    // Reduce per-block shared histogram into the global histogram.
    // Each thread handles multiple bins to cover the full range.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int val = sHist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host wrapper: run_histogram
 *
 * Parameters:
 *   input      - device pointer to input chars (cudaMalloc'ed)
 *   histogram  - device pointer to histogram array (cudaMalloc'ed)
 *   inputSize  - number of chars in input
 *   from       - first character code in the range (0 <= from <= 255)
 *   to         - last character code in the range (from <= to <= 255)
 *
 * Behavior:
 *   - Zeros the histogram buffer for the range [from, to].
 *   - Launches the CUDA kernel configured to process inputSize characters.
 *   - Assumes the caller handles any required synchronization and error checking.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic validation: nothing to do if input is empty or range is invalid.
    if (inputSize == 0 || from > to) {
        return;
    }

    // Compute number of bins in the requested range.
    int range = to - from + 1;

    // Zero the histogram for the specified range on the device.
    // The rest of the buffer (if any) is left untouched.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Choose a reasonable block size for modern GPUs.
    // 256 threads per block is a good default for a variety of workloads.
    const int threadsPerBlock = 256;

    // Each thread processes up to itemsPerThread characters.
    const unsigned int workPerBlock = threadsPerBlock * itemsPerThread;

    // Compute number of blocks needed to cover all input characters.
    unsigned int blocks =
        (inputSize + workPerBlock - 1) / workPerBlock;

    if (blocks == 0) {
        blocks = 1;
    }

    // Dynamic shared memory size required for the per-block histogram.
    const size_t sharedMemSize =
        static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch the kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        range
    );

    // Note: Synchronization and error checking (e.g., cudaGetLastError)
    // are deliberately left to the caller, as specified.
}