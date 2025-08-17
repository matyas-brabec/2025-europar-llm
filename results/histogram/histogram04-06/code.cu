#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA histogram for a restricted character range with shared-memory privatization.

  Design highlights:
  - Each block builds a privatized histogram in shared memory to reduce global atomic contention.
  - Further privatization: each warp owns its own sub-histogram within shared memory (warp-private histograms).
    This reduces intra-block contention significantly on skewed data.
  - After processing, the per-warp histograms are reduced and added to the global histogram with atomics.
  - Threads process several items each (itemsPerThread) to improve memory throughput and ILP.
  - Memory access pattern is coalesced: each iteration reads a contiguous segment.
  - Assumes 'input' and 'histogram' point to device memory allocated by cudaMalloc.
  - The caller is responsible for synchronization; this function only enqueues operations on the default stream.

  Notes on tuning constants:
  - itemsPerThread is set to 16 by default. On modern GPUs (A100/H100), this yields good memory throughput
    for large inputs by balancing ILP and occupancy for a 256-thread block.
  - threadsPerBlock is set to 256. This yields 8 warps per block, which provides enough parallelism while
    keeping shared memory usage low (<= 8 KB for a 256-bin range).
*/

static constexpr int itemsPerThread = 16;  // Tunable: number of input bytes processed by each thread
static constexpr int THREADS_PER_BLOCK = 256; // Tunable: must be a multiple of WARP_SIZE for best results
static constexpr int WARP_SIZE = 32;

template <int ITEMS_PER_THREAD>
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ globalHist,
                                       unsigned int inputSize,
                                       int from, int to)
{
    // Basic invariants (assumed valid per problem statement)
    const int numBins = to - from + 1;
    const int t = threadIdx.x;
    const int blockThreads = blockDim.x;

    // Warp/Shared-memory layout
    const int warpId = t / WARP_SIZE;
    const int numWarps = (blockThreads + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ unsigned int smem[];
    // Layout: [numWarps][numBins], flattened as contiguous memory.
    unsigned int* warpHists = smem;
    unsigned int* myWarpHist = warpHists + warpId * numBins;

    // Zero initialize the per-warp histograms in shared memory
    for (int i = t; i < numWarps * numBins; i += blockThreads) {
        warpHists[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over "segments" of size blockDim.x * ITEMS_PER_THREAD per block
    const size_t blockWork = static_cast<size_t>(blockThreads) * ITEMS_PER_THREAD;
    const size_t gridWork = static_cast<size_t>(gridDim.x) * blockWork;

    for (size_t blockStart = static_cast<size_t>(blockIdx.x) * blockWork;
         blockStart < static_cast<size_t>(inputSize);
         blockStart += gridWork)
    {
        // Base for this thread within the block segment
        size_t threadBase = blockStart + static_cast<size_t>(t);

        // Process ITEMS_PER_THREAD elements, striding by blockThreads to keep accesses coalesced
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            size_t idx = threadBase + static_cast<size_t>(i) * blockThreads;
            if (idx < static_cast<size_t>(inputSize)) {
                // Load byte and map to integer 0..255
                unsigned int v = static_cast<unsigned int>(input[idx]);

                // Restrict to requested range and increment the appropriate bin
                if (v >= static_cast<unsigned int>(from) && v <= static_cast<unsigned int>(to)) {
                    const int bin = static_cast<int>(v) - from;
                    // Atomic in shared memory: fast on modern GPUs; using warp-private hist reduces contention
                    atomicAdd(&myWarpHist[bin], 1U);
                }
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms and add to global histogram
    for (int bin = t; bin < numBins; bin += blockThreads) {
        unsigned int sum = 0;
        // Sum contributions of all warps for this bin
        for (int w = 0; w < numWarps; ++w) {
            sum += warpHists[w * numBins + bin];
        }
        if (sum) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

/*
  Host API:
  - input: device pointer to input bytes (cudaMalloc'd)
  - histogram: device pointer to output histogram array (cudaMalloc'd), length = (to - from + 1)
  - inputSize: number of bytes in input
  - from, to: inclusive character range [from, to], with 0 <= from <= to <= 255

  This function:
  - Zeros the output histogram on device.
  - Launches the histogram kernel with a configuration tuned for modern data center GPUs.
  - Does not synchronize; caller is responsible for synchronization.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate range and handle trivial cases. We still zero the histogram if range is valid.
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const int numBins = to - from + 1;

    // Zero the output histogram on device
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0) return;

    // Launch configuration
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const unsigned int blockWork = threadsPerBlock * itemsPerThread;
    unsigned int numBlocks = (inputSize + blockWork - 1) / blockWork;
    if (numBlocks == 0) numBlocks = 1;

    // Dynamic shared memory size: one histogram per warp
    const int warpsPerBlock = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
    const size_t smemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel on the default stream
    histogram_range_kernel<itemsPerThread><<<numBlocks, threadsPerBlock, smemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}