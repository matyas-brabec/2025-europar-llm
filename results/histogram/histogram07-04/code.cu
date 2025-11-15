#include <cuda_runtime.h>

/*
 * CUDA kernel and host launcher for computing a histogram over a specified
 * character range [from, to] for a large text buffer.
 *
 * Key optimization ideas:
 *  - The input is processed in parallel by many threads.
 *  - Each block maintains a private histogram in shared memory to reduce
 *    expensive global-memory atomics.
 *  - To avoid shared-memory bank conflicts when threads in a warp update
 *    the same bin, we keep 32 copies of the histogram in shared memory,
 *    one per bank. Each thread uses copy (threadIdx.x % 32).
 *  - The 32 copies are stored with a stride of 32 so that for any bin i,
 *    the 32 copies occupy consecutive banks:
 *        shared_index = i * 32 + copy_id
 *    where copy_id is in [0, 31].
 *  - At the end of the kernel, the 32 copies per bin are reduced to a
 *    single value and accumulated into the global histogram.
 *
 *  itemsPerThread controls how many input items each thread processes.
 *  It is a compile-time constant tuned for modern NVIDIA data-center GPUs
 *  like A100/H100 and large inputs.
 */

static constexpr int itemsPerThread = 16;   // Tunable: per-thread workload (per kernel invocation)

/*
 * CUDA kernel: compute a partial histogram for a range [from, from+numBins-1].
 *
 * Parameters:
 *   d_input    - device pointer to input characters
 *   inputSize  - number of characters in d_input
 *   from       - first character (inclusive) in the histogram range
 *   numBins    - number of bins (to - from + 1)
 *   d_hist     - device pointer to global histogram array of length numBins
 *
 * For each block:
 *   - Allocate numBins * 32 unsigned ints in shared memory.
 *   - Zero the shared histogram copies.
 *   - Each thread processes up to itemsPerThread characters and updates
 *     its chosen copy of the histogram using shared-memory atomics.
 *   - Finally, the copies are reduced into a single histogram per block
 *     and accumulated into d_hist with global atomics.
 */
__global__ void histogram_kernel(const char* __restrict__ d_input,
                                 unsigned int inputSize,
                                 int from,
                                 int numBins,
                                 unsigned int* __restrict__ d_hist)
{
    // Dynamic shared memory: 32 copies of the histogram.
    // Layout: for bin i and copy c (0 <= c < 32),
    //         shared index = i * 32 + c
    // This ensures that when all lanes in a warp access the same bin index
    // (but different copies), they hit different banks and avoid conflicts.
    extern __shared__ unsigned int s_hist[];

    const int tid  = threadIdx.x;
    const int lane = threadIdx.x & 31;      // copy ID in [0,31], matches bank ID
    const int copiesPerBlock = 32;

    const unsigned int globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int totalSharedBins =
        static_cast<unsigned int>(numBins) * copiesPerBlock;

    // 1. Initialize shared histogram copies to zero.
    //    Each thread zeros multiple entries with a stride of blockDim.x.
    for (unsigned int idx = tid; idx < totalSharedBins; idx += blockDim.x) {
        s_hist[idx] = 0;
    }

    __syncthreads();

    // 2. Main accumulation loop:
    //    Each thread processes up to itemsPerThread characters starting at
    //    baseIndex = globalThreadId * itemsPerThread.
    const unsigned int baseIndex = globalThreadId * static_cast<unsigned int>(itemsPerThread);

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            break;  // No more input for this thread
        }

        unsigned char ch = static_cast<unsigned char>(d_input[idx]);
        int bin = static_cast<int>(ch) - from;

        // Only count characters within [from, from + numBins - 1]
        if (bin >= 0 && bin < numBins) {
            unsigned int shIdx =
                static_cast<unsigned int>(bin) * copiesPerBlock +
                static_cast<unsigned int>(lane);

            // Shared-memory atomic add is fast and ensures correctness
            // when multiple warps (with the same lane index) update
            // the same histogram copy and bin.
            atomicAdd(&s_hist[shIdx], 1u);
        }
    }

    __syncthreads();

    // 3. Reduce the 32 copies of each bin and accumulate into global memory.
    //    Each thread handles multiple bins with a stride of blockDim.x.
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        unsigned int offset =
            static_cast<unsigned int>(bin) * copiesPerBlock;

        #pragma unroll
        for (int c = 0; c < copiesPerBlock; ++c) {
            sum += s_hist[offset + static_cast<unsigned int>(c)];
        }

        if (sum > 0) {
            atomicAdd(&d_hist[bin], sum);
        }
    }
}

/*
 * Host function: run_histogram
 *
 * Computes a histogram of the input character buffer restricted to the
 * character range [from, to]. The caller is responsible for allocating
 * device memory for input and histogram via cudaMalloc, and for any
 * required synchronization (e.g., cudaDeviceSynchronize) after this call.
 *
 * Parameters:
 *   input      - device pointer to an array of chars (cudaMalloc'ed)
 *   histogram  - device pointer to an array of unsigned ints (cudaMalloc'ed)
 *                of length (to - from + 1)
 *   inputSize  - number of characters in input
 *   from       - lower bound (inclusive), 0 <= from <= 255
 *   to         - upper bound (inclusive), from <= to <= 255
 *
 * Behavior:
 *   - The function zeros the histogram buffer.
 *   - Launches a CUDA kernel using a reasonable configuration for
 *     modern GPUs (e.g., A100/H100).
 *   - No synchronization is performed; the kernel runs asynchronously.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins in the requested range.
    // Assumes 0 <= from <= to <= 255.
    const int numBins = to - from + 1;

    // If there is no data, just zero the histogram and return.
    if (inputSize == 0) {
        if (numBins > 0) {
            cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));
        }
        return;
    }

    // Initialize the global histogram to zero, because the kernel uses
    // atomicAdd to accumulate per-block results.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Kernel launch configuration.
    // - 256 threads per block is a good default for modern GPUs, balancing
    //   occupancy and per-thread resource usage.
    // - Each thread processes itemsPerThread elements, so each block
    //   processes blockDim.x * itemsPerThread input characters.
    const int threadsPerBlock = 256;
    const unsigned int itemsPerBlock =
        static_cast<unsigned int>(threadsPerBlock) *
        static_cast<unsigned int>(itemsPerThread);

    // Number of blocks needed to cover all input data with the chosen
    // itemsPerThread. Each thread will process up to itemsPerThread items.
    unsigned int gridSize =
        (inputSize + itemsPerBlock - 1u) / itemsPerBlock;

    if (gridSize == 0u) {
        gridSize = 1u;
    }

    // Dynamic shared memory size:
    // 32 copies of the histogram, each with numBins bins.
    // Layout: [bin0_copy0, bin0_copy1, ..., bin0_copy31,
    //          bin1_copy0, bin1_copy1, ..., bin1_copy31,
    //          ...
    //          bin(numBins-1)_copy31]
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Launch the kernel. All work is done asynchronously on the default stream.
    histogram_kernel<<<gridSize, threadsPerBlock, sharedMemSize>>>(
        input,
        inputSize,
        from,
        numBins,
        histogram
    );
}