#include <cuda_runtime.h>

/*
 * Optimized text histogram on [from, to] character range using CUDA.
 *
 * Key design points:
 *  - Each thread processes a fixed number of characters (itemsPerThread).
 *  - Per-block histogram privatization in shared memory to reduce global atomics.
 *  - 32 duplicated copies of the histogram in shared memory to:
 *      * Reduce contention on shared-memory atomics.
 *      * Avoid bank conflicts by striding copies.
 *  - Layout of shared histogram:
 *        sharedHist[ bin * 32 + copy ]
 *    where:
 *        bin  \in [0, numBins)
 *        copy \in [0, 32)
 *    and copy is chosen as threadIdx.x % 32.
 *
 *  - After processing, per-block shared histograms are reduced to a single
 *    histogram per block, and merged into the global histogram with atomicAdd.
 *
 * Assumptions:
 *  - Device is a modern NVIDIA GPU (e.g., A100/H100).
 *  - input and histogram pointers passed to run_histogram are device pointers
 *    allocated via cudaMalloc.
 *  - Caller handles synchronization (e.g., cudaDeviceSynchronize) after calling
 *    run_histogram.
 */

static constexpr int itemsPerThread = 8;  // Tuned for large inputs on modern GPUs (A100/H100)


/*
 * CUDA kernel: builds a histogram over characters in [from, to].
 *
 * Parameters:
 *  input       - device pointer to input text (chars)
 *  globalHist  - device pointer to histogram array of size (to - from + 1)
 *  inputSize   - number of characters in the input buffer
 *  from, to    - inclusive character range [from, to] (0 <= from < to <= 255)
 *
 * Shared memory:
 *  extern __shared__ unsigned int sharedHist[];
 *  Size: (numBins * 32) * sizeof(unsigned int),
 *  where numBins = to - from + 1.
 */
__global__ void histogramKernel(const char *__restrict__ input,
                                unsigned int *__restrict__ globalHist,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int sharedHist[];  // 32 * numBins entries

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;   // threadIdx.x % 32

    // 1. Initialize the shared-memory histograms to zero.
    const int totalSharedEntries = numBins * 32;
    for (int idx = tid; idx < totalSharedEntries; idx += blockDim.x) {
        sharedHist[idx] = 0;
    }
    __syncthreads();

    // 2. Build per-thread contributions into the shared-memory histograms.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + tid;
    const unsigned int baseIndex      = globalThreadId * itemsPerThread;

    // Process up to itemsPerThread characters per thread.
    // Using a fixed small number improves instruction-level parallelism
    // and amortizes thread launch overhead.
#pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            break;
        }

        // Load character and convert to unsigned to get 0..255 range.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Only count characters within [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to))
        {
            int bin = static_cast<int>(c) - from;  // 0..(numBins-1)
            // Layout: histCopy[bin][copy] at index = bin * 32 + copy.
            int offset = bin * 32 + lane;

            // Shared-memory atomicAdd handles concurrent updates from multiple
            // warps that share the same "copy".
            atomicAdd(&sharedHist[offset], 1u);
        }
    }

    __syncthreads();

    // 3. Reduce the 32 histogram copies per block into a single histogram,
    //    and atomically accumulate into the global histogram.
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * 32;

#pragma unroll
        for (int copy = 0; copy < 32; ++copy) {
            sum += sharedHist[base + copy];
        }

        if (sum > 0) {
            // One atomicAdd per bin per block, significantly fewer than
            // per-character global atomics.
            atomicAdd(&globalHist[bin], sum);
        }
    }
}


/*
 * Host function: configures and launches histogramKernel.
 *
 * Parameters:
 *  input      - device pointer to input text (chars), cudaMalloc'ed
 *  histogram  - device pointer to output histogram array (unsigned int),
 *               of size (to - from + 1), cudaMalloc'ed
 *  inputSize  - number of characters in the input buffer
 *  from, to   - inclusive character range [from, to] (0 <= from < to <= 255)
 *
 * Behavior:
 *  - Clears the output histogram on device.
 *  - Launches the CUDA kernel with appropriate grid/block/shared-mem settings.
 *  - Does not perform device-wide synchronization; caller is responsible.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins; handle trivial/empty cases.
    const int numBins = (to >= from) ? (to - from + 1) : 0;
    if (numBins <= 0) {
        return;
    }

    // Ensure output histogram is zeroed before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0) {
        // Nothing to process; histogram already cleared.
        return;
    }

    // Block size chosen as a multiple of warpSize for high occupancy.
    // 256 threads/block is a good default for modern NVIDIA GPUs.
    const int blockSize = 256;

    // Total number of threads required so that each processes itemsPerThread elements.
    const unsigned int threadsNeeded =
        (inputSize + static_cast<unsigned int>(itemsPerThread) - 1) /
        static_cast<unsigned int>(itemsPerThread);

    const unsigned int blocks =
        (threadsNeeded + static_cast<unsigned int>(blockSize) - 1) /
        static_cast<unsigned int>(blockSize);

    // Shared memory size: 32 copies of the histogram, each with numBins entries.
    const size_t sharedMemBytes =
        static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Launch kernel. All operations are enqueued on the default stream.
    histogramKernel<<<blocks, blockSize, sharedMemBytes>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );

    // No synchronization here; caller is responsible for cudaDeviceSynchronize()
    // or appropriate stream synchronization and error checking.
}