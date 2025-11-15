#include <cuda_runtime.h>
#include <cstdint>

// Number of items (characters) processed by each thread.
//
// 8 is a good default for modern data-center GPUs (A100/H100) and large inputs:
// it increases arithmetic intensity and hides memory latency while keeping
// register usage and launch granularity reasonable.
static constexpr int itemsPerThread = 8;

__global__
void histogram_kernel(const char * __restrict__ input,
                      unsigned int * __restrict__ histogram,
                      unsigned int inputSize,
                      int from,
                      int numBins)
{
    // Thread identifiers within the block.
    const int tid           = threadIdx.x;
    const int lane          = tid & 31;     // warp lane index 0..31
    const int blockThreads  = blockDim.x;

    // We maintain 32 copies of the histogram in shared memory, one logically
    // associated with each warp lane ID. This allows each lane in a warp to
    // update its own copy, eliminating intra-warp bank conflicts.
    const int numCopies = 32;

    // Stride between consecutive histogram copies in shared memory.
    // Making the stride odd (coprime with 32) ensures that when all 32 lanes
    // access the same bin index in their respective copy, they hit distinct
    // banks, avoiding bank conflicts.
    const int stride = (numBins & 1) ? numBins : (numBins + 1);

    // Shared-memory layout:
    // s_hist[ copy * stride + bin ] for copy in [0, 31], bin in [0, numBins-1].
    extern __shared__ unsigned int s_hist[];
    unsigned int *localHistBase = s_hist;

    // Zero-initialize the shared histograms.
    const int totalSharedElems = numCopies * stride;
    for (int i = tid; i < totalSharedElems; i += blockThreads) {
        localHistBase[i] = 0;
    }

    __syncthreads();

    // Pointer to the sub-histogram this thread will update. All threads with the
    // same lane ID (across warps within the block) share this sub-histogram.
    unsigned int *myHist = localHistBase + lane * stride;

    // Global thread index tailored for itemsPerThread processing pattern.
    const unsigned int globalThreadBase =
        blockIdx.x * blockThreads * itemsPerThread + tid;

    // Each thread processes itemsPerThread characters spaced blockThreads apart.
    // This preserves coalesced access for each iteration of the loop.
    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        const unsigned int idx = globalThreadBase + i * blockThreads;
        if (idx >= inputSize) break;

        const unsigned char c = static_cast<unsigned char>(input[idx]);
        const int value = static_cast<int>(c);
        const int bin   = value - from;

        // Use an unsigned comparison to fold the lower-bound and upper-bound
        // checks into a single test.
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            // Multiple warps share the same per-lane histogram copy, so
            // shared-memory atomics are necessary for correctness.
            atomicAdd(&myHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane sub-histograms into the output histogram.
    // The output 'histogram' array has numBins entries where index i corresponds
    // to character (from + i).
    for (int bin = tid; bin < numBins; bin += blockThreads) {
        unsigned int sum = 0;

        // Accumulate this bin across all 32 copies.
        #pragma unroll
        for (int copy = 0; copy < numCopies; ++copy) {
            sum += localHistBase[copy * stride + bin];
        }

        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host-side launcher. The caller is responsible for:
//  - allocating 'input' and 'histogram' in device memory (cudaMalloc)
//  - ensuring 'histogram' is zero-initialized before calling this function
//    (e.g., via cudaMemset)
//  - performing any required host-device synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Block size chosen to balance occupancy and shared-memory contention.
    // 256 threads per block is a solid choice on modern NVIDIA GPUs.
    const int blockSize = 256;

    // Each block processes blockSize * itemsPerThread characters.
    const unsigned int itemsPerBlock = blockSize * itemsPerThread;
    const unsigned int gridSize =
        (inputSize + itemsPerBlock - 1) / itemsPerBlock;

    // Shared memory requirements: 32 copies * stride * sizeof(unsigned int).
    const int stride = (numBins & 1) ? numBins : (numBins + 1);
    const size_t sharedMemSize = 32u * static_cast<size_t>(stride) * sizeof(unsigned int);

    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input, histogram, inputSize, from, numBins);
}