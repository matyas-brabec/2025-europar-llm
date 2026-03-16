#include <cuda_runtime.h>

/*
    Range-restricted byte histogram for a device-resident char buffer.

    High-level algorithm:
      1) Each thread block accumulates into a privatized histogram in shared memory.
      2) The shared histogram is replicated 32 times per block, one logical copy per warp lane.
      3) Threads grid-stride over the input; each thread processes `itemsPerThread` chars per iteration.
      4) After the block finishes its local accumulation, the 32 replicas are reduced and atomically
         added to the final global histogram.

    Important layout detail requested by the prompt:
      sharedHistogram[bin * 32 + replica]

    with `replica == laneId`.

    Because the counters are 32-bit words and modern NVIDIA GPUs have 32 shared-memory banks,
    the bank selected by that address is:

        bank = (bin * 32 + laneId) % 32 = laneId

    So every lane always hits its own bank, independent of the bin number. This eliminates
    intra-warp bank conflicts for the histogram updates.

    The prompt asks for 32 copies per *thread block* (not per warp). Therefore multiple warps
    in the same block share each per-lane replica, and correctness requires shared-memory
    atomicAdd() for the updates.
*/

// 256 threads/block is a strong default on A100/H100 for this atomic-heavy pattern:
// it gives enough warps for good latency hiding without creating as much inter-warp
// contention on the shared replicas as a 512-thread block would.
constexpr int blockThreads = 256;

// Tuning knob requested by the prompt.
// For this kernel, 16 chars/thread is a good default on modern data-center GPUs:
// enough ILP to amortize loop/index overhead, while keeping register pressure and
// unrolled code size moderate.
constexpr int itemsPerThread = 16;

// Exactly 32 replicas are required by the prompt; this matches warp width and the
// number of shared-memory banks on modern NVIDIA GPUs.
constexpr int replicaCount = 32;

constexpr int warpsPerBlock = blockThreads / replicaCount;
constexpr size_t tileItems = static_cast<size_t>(blockThreads) * itemsPerThread;

static_assert(replicaCount == 32, "This implementation assumes 32 shared-memory banks / 32 warp lanes.");
static_assert(blockThreads % replicaCount == 0, "blockThreads must be a multiple of 32.");

// Warp-wide sum. Only lane 0 receives the full sum; that is sufficient for the final write-back.
__device__ __forceinline__ unsigned int warpReduceSum(unsigned int value) {
    #pragma unroll
    for (int offset = replicaCount / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

// Consume one byte and update the block-private shared histogram if the byte falls in range.
// The single-comparison range test works because unsigned subtraction turns
//     from <= byteValue <= to
// into
//     0 <= (byteValue - from) < numBins.
__device__ __forceinline__ void accumulateByte(
    unsigned int byteValue,
    unsigned int* sharedHistogram,
    unsigned int from,
    unsigned int numBins,
    unsigned int laneId)
{
    const unsigned int bin = byteValue - from;
    if (bin < numBins) {
        atomicAdd(&sharedHistogram[bin * replicaCount + laneId], 1u);
    }
}

__global__ __launch_bounds__(blockThreads)
void histogramKernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from,
    unsigned int numBins)
{
    // Dynamic shared memory size is numBins * 32 counters.
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int laneId = threadIdx.x & (replicaCount - 1);
    const unsigned int warpId = threadIdx.x >> 5;

    // Zero the block-private histogram.
    // Each warp owns a subset of bins, and each lane writes its own replica.
    // This access pattern is also bank-conflict free because replica == laneId.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        sharedHistogram[bin * replicaCount + laneId] = 0u;
    }
    __syncthreads();

    const size_t inputSize64 = static_cast<size_t>(inputSize);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * tileItems;

    // Process all fully covered tiles without bounds checks in the inner loop.
    // In each unrolled round, threads in a warp read consecutive bytes, so global
    // loads remain naturally coalesced.
    size_t tileBase = static_cast<size_t>(blockIdx.x) * tileItems;
    for (; tileBase + tileItems <= inputSize64; tileBase += gridStride) {
        size_t idx = tileBase + static_cast<size_t>(threadIdx.x);

        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item, idx += blockThreads) {
            accumulateByte(static_cast<unsigned int>(input[idx]), sharedHistogram, from, numBins, laneId);
        }
    }

    // Tail path for the final partial tile(s) assigned to this block.
    for (; tileBase < inputSize64; tileBase += gridStride) {
        size_t idx = tileBase + static_cast<size_t>(threadIdx.x);

        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item, idx += blockThreads) {
            if (idx < inputSize64) {
                accumulateByte(static_cast<unsigned int>(input[idx]), sharedHistogram, from, numBins, laneId);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 replicas for each bin.
    // Each warp owns a subset of bins; lane l reads replica l of that bin.
    // Since the replicas are interleaved as [bin][lane], these reads are also bank-conflict free.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        const unsigned int local = sharedHistogram[bin * replicaCount + laneId];
        const unsigned int sum = warpReduceSum(local);

        // Multiple blocks contribute to the same final histogram, so the global write-back is atomic.
        if (laneId == 0 && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Defensive validation. The prompt states the range is valid; if it is not,
    // there is no well-defined histogram size to operate on, so just return.
    if (from < 0 || from > 255 || to < 0 || to > 255 || to < from) {
        return;
    }

    // Inclusive range [from, to].
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // The function is expected to produce a fresh histogram, so clear the output first.
    // No host synchronization is performed here; caller requested to handle synchronization externally.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0) {
        return;
    }

    // Shared-memory footprint:
    //   numBins * 32 replicas * 4 bytes
    // Worst case is 256 * 32 * 4 = 32 KiB, well within the normal per-block dynamic
    // shared-memory limit on modern NVIDIA GPUs, so no opt-in max-dynamic-shmem attribute is needed.
    const size_t sharedBytes = static_cast<size_t>(numBins) * replicaCount * sizeof(unsigned int);

    // Launch exactly one SM-resident wave of blocks.
    // Because each block grid-strides over the full input while reusing a single privatized histogram,
    // extra block waves would mostly increase block init/final-reduction overhead and the number of
    // global atomic write-backs.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogramKernel,
        blockThreads,
        sharedBytes);
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    int gridBlocks = smCount * activeBlocksPerSm;
    const size_t numTiles = (static_cast<size_t>(inputSize) + tileItems - 1) / tileItems;
    if (numTiles < static_cast<size_t>(gridBlocks)) {
        gridBlocks = static_cast<int>(numTiles);
    }
    if (gridBlocks < 1) {
        gridBlocks = 1;
    }

    histogramKernel<<<gridBlocks, blockThreads, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        numBins);
}