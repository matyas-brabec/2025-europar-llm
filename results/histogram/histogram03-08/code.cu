#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram over a restricted character range [from, to].
 *
 * Optimization strategy:
 *  - Use shared memory for histogram privatization to drastically reduce the number
 *    of atomic operations on global memory.
 *  - Each warp in a block gets its own private histogram in shared memory.
 *    Threads in the same warp atomically update that warp-private histogram.
 *  - After processing, per-warp histograms are reduced within the block and the
 *    block-level results are added to the global histogram using a small number
 *    of global atomic operations (one per bin per block).
 *
 * Parameters:
 *  input         - device pointer to input text buffer (chars)
 *  histogram     - device pointer to output histogram (size = to - from + 1)
 *  numElements   - number of characters in input
 *  from, to      - inclusive range of character codes to track (0 <= from < to <= 255)
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int numElements,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[]; // shared memory: per-warp histograms

    const int tid            = threadIdx.x;
    const int blockThreads   = blockDim.x;
    const int globalThreadId = blockIdx.x * blockThreads + tid;
    const int gridStride     = blockThreads * gridDim.x;
    const int rangeLen       = to - from + 1;

    const int warp_id        = tid / warpSize;
    const int warpsPerBlock  = (blockThreads + warpSize - 1) / warpSize;

    // Pointer to this warp's private histogram segment in shared memory
    unsigned int* warp_hist = s_hist + warp_id * rangeLen;

    // Initialize the entire shared-memory histogram (all warps) to zero.
    // All threads in the block cooperate to clear warpsPerBlock * rangeLen bins.
    for (int i = tid; i < warpsPerBlock * rangeLen; i += blockThreads) {
        s_hist[i] = 0;
    }

    __syncthreads();

    // Grid-stride loop over input data for good load balancing and coalesced loads.
    for (unsigned int idx = globalThreadId; idx < numElements; idx += gridStride) {
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Compute bin index relative to 'from'.
        // Using unsigned comparison to compact range check:
        //   if 0 <= (c - from) < rangeLen => c in [from, to]
        int bin = static_cast<int>(c) - from;
        unsigned int uBin = static_cast<unsigned int>(bin);
        if (uBin < static_cast<unsigned int>(rangeLen)) {
            // Atomic update in shared memory (fast compared to global atomics).
            atomicAdd(&warp_hist[uBin], 1);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into global histogram.
    // Each thread accumulates several bins if rangeLen > blockDim.x.
    for (int bin = tid; bin < rangeLen; bin += blockThreads) {
        unsigned int sum = 0;

        // Sum contributions from all warps in this block for this bin.
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }

        // Only perform global atomicAdd if there is a contribution.
        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host function to launch the CUDA histogram kernel.
 *
 * Assumptions:
 *  - 'input' points to device memory containing 'inputSize' chars.
 *  - 'histogram' points to device memory large enough to hold (to - from + 1)
 *    unsigned int counters.
 *  - All required host-device synchronization (e.g., cudaDeviceSynchronize)
 *    will be performed by the caller; this function only launches the kernel.
 *
 * Steps:
 *  1. Zero the histogram output buffer on the device.
 *  2. Configure a grid/block layout suitable for modern data-center GPUs.
 *  3. Compute required shared memory size for per-warp histograms.
 *  4. Launch the histogram kernel.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Trivial cases: nothing to process.
    if (inputSize == 0 || from > to) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Initialize the device histogram to zero.
    cudaMemset(histogram, 0, rangeLen * sizeof(unsigned int));

    // Use a reasonable number of threads per block for histogram-like workloads.
    const int THREADS_PER_BLOCK = 256;

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // Number of blocks needed to cover all elements (grid-stride loop will handle rest).
    unsigned int blocksNeeded = (inputSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Cap the number of blocks to avoid extreme oversubscription.
    // Using up to 32 blocks per SM is typically sufficient for full occupancy.
    unsigned int maxBlocks = props.multiProcessorCount * 32;
    unsigned int blocks = (blocksNeeded > maxBlocks) ? maxBlocks : blocksNeeded;
    if (blocks == 0) {
        blocks = 1;
    }

    // Compute warps per block and shared memory size for per-warp histograms.
    const int warpsPerBlock = (THREADS_PER_BLOCK + props.warpSize - 1) / props.warpSize;
    const size_t sharedMemSize =
        static_cast<size_t>(rangeLen) *
        static_cast<size_t>(warpsPerBlock) *
        sizeof(unsigned int);

    // Launch the kernel. Caller is responsible for checking errors and synchronizing.
    histogram_kernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}