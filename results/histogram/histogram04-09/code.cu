#include <cuda_runtime.h>

// Number of input characters processed by each thread.
// A value of 16 provides a good balance for modern NVIDIA data-center GPUs
// (e.g., A100/H100) when processing large inputs.
constexpr int ITEMS_PER_THREAD = 16;

/**
 * CUDA kernel to compute a histogram of characters within a specified range.
 *
 * - input:     device pointer to input characters
 * - histogram: device pointer to output histogram (size = (to - from + 1))
 * - inputSize: number of characters in input
 * - from, to:  inclusive range of character codes to count (0 <= from <= to <= 255)
 *
 * The kernel uses per-block shared-memory histograms to reduce contention on
 * global memory. Each block:
 *   1. Initializes its shared histogram (size = numBins).
 *   2. Processes ITEMS_PER_THREAD input items per thread (strided access for coalescing).
 *   3. Accumulates its shared histogram into the global histogram.
 */
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int shHist[];

    const int numBins = to - from + 1;

    // Initialize the shared histogram to zero.
    // Use a strided loop so all bins are covered even if blockDim.x < numBins.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        shHist[bin] = 0;
    }
    __syncthreads();

    const unsigned int totalThreads = gridDim.x * blockDim.x;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes ITEMS_PER_THREAD characters in a strided fashion:
    // index = tid + k * totalThreads
    // This ensures coalesced loads across warps and full coverage of the input
    // when ITEMS_PER_THREAD * totalThreads >= inputSize.
    for (unsigned int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const unsigned int idx = tid + k * totalThreads;
        if (idx >= inputSize) {
            break;
        }

        const unsigned char c = static_cast<unsigned char>(input[idx]);
        const int binIdx = static_cast<int>(c) - from;

        // Only count characters within [from, to].
        if (binIdx >= 0 && binIdx < numBins) {
            // Shared-memory atomics are fast on modern GPUs.
            atomicAdd(&shHist[binIdx], 1u);
        }
    }
    __syncthreads();

    // Accumulate the per-block shared histogram into the global histogram.
    // Each thread handles a subset of bins to minimize global atomics.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        const unsigned int val = shHist[bin];
        if (val != 0) {
            atomicAdd(&histogram[bin], val);
        }
    }
}

/**
 * Host-side helper to launch the histogram kernel.
 *
 * Parameters:
 *   - input:      device pointer to input character buffer (cudaMalloc'ed)
 *   - histogram:  device pointer to output histogram (cudaMalloc'ed)
 *   - inputSize:  number of characters in input
 *   - from, to:   inclusive range of character codes to count
 *
 * Requirements:
 *   - histogram must have space for (to - from + 1) unsigned ints.
 *   - Any necessary device synchronization after the kernel launch is the
 *     responsibility of the caller; this function does not synchronize.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        // Nothing to do.
        return;
    }

    const int numBins = to - from + 1;

    // Initialize the output histogram on device to zero.
    // cudaMemset is asynchronous with respect to the host.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a reasonable block size for modern GPUs.
    const int threadsPerBlock = 256;
    const unsigned int itemsPerThread = ITEMS_PER_THREAD;

    // Total number of threads needed so that ITEMS_PER_THREAD * totalThreads >= inputSize.
    const unsigned int totalThreads =
        (inputSize + itemsPerThread - 1) / itemsPerThread;

    if (totalThreads == 0) {
        return;
    }

    const unsigned int blocks =
        (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Dynamic shared memory size per block: one bin per histogram entry.
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel. The caller is responsible for any required
    // cudaDeviceSynchronize or stream synchronization.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}