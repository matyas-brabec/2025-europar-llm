#include <cuda_runtime.h>

/*
  Range-restricted byte histogram for modern NVIDIA GPUs.

  Algorithm:
  1) Each thread block builds a private histogram in shared memory.
  2) To avoid shared-memory bank conflicts, every logical bin is replicated 32 times
     (one replica per warp lane) and the shared layout is transposed as [bin][lane].
     For 32-bit counters on the 32-bank shared memory used by A100/H100-class GPUs,
     bank = lane, independent of the bin value, so a warp update is bank-conflict free.
  3) After the block has processed all of its grid-stride tiles, the 32 replicas of each
     bin are reduced by a warp and atomically accumulated into the final global histogram.

  Tuning:
  - itemsPerThread controls how many input bytes each thread processes per tile.
  - Default = 16 because it maps naturally to one aligned 16-byte uint4 load per thread
    from cudaMalloc-aligned input, which is a strong default on recent data-center GPUs.
*/

namespace
{
constexpr int blockSize = 256;
constexpr int itemsPerThread = 16;
constexpr int histogramReplicaShift = 5;
constexpr int histogramReplicas = 1 << histogramReplicaShift;  // 32 replicas, one per warp lane.
constexpr int warpsPerBlock = blockSize / histogramReplicas;
constexpr int vectorBytes = sizeof(uint4);

static_assert(histogramReplicas == 32, "This kernel assumes 32-bank shared memory.");
static_assert(blockSize % histogramReplicas == 0, "blockSize must be a multiple of warp size.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert((itemsPerThread % vectorBytes) == 0,
              "itemsPerThread must be a multiple of 16 to preserve aligned uint4 loads.");

__device__ __forceinline__ void addByteToSharedHistogram(
    const unsigned int value,
    const unsigned int from,
    const unsigned int numBins,
    unsigned int* const laneHistogram)
{
    // Treat the input byte as an unsigned ordinal in [0, 255].
    // Unsigned subtraction followed by a single range check is cheaper than
    // two separate comparisons: values below 'from' underflow and fail the test.
    const unsigned int bin = value - from;
    if (bin < numBins)
    {
        // Shared-memory layout is [bin][lane], so for a fixed bin all 32 lanes in a warp
        // touch consecutive 32-bit words and therefore distinct banks.
        atomicAdd(laneHistogram + (bin << histogramReplicaShift), 1u);
    }
}

__device__ __forceinline__ void accumulatePackedWord(
    const unsigned int packed,
    const unsigned int from,
    const unsigned int numBins,
    unsigned int* const laneHistogram)
{
    addByteToSharedHistogram((packed >>  0) & 0xFFu, from, numBins, laneHistogram);
    addByteToSharedHistogram((packed >>  8) & 0xFFu, from, numBins, laneHistogram);
    addByteToSharedHistogram((packed >> 16) & 0xFFu, from, numBins, laneHistogram);
    addByteToSharedHistogram((packed >> 24) & 0xFFu, from, numBins, laneHistogram);
}

__device__ __forceinline__ void accumulateVector16(
    const uint4 packed16,
    const unsigned int from,
    const unsigned int numBins,
    unsigned int* const laneHistogram)
{
    accumulatePackedWord(packed16.x, from, numBins, laneHistogram);
    accumulatePackedWord(packed16.y, from, numBins, laneHistogram);
    accumulatePackedWord(packed16.z, from, numBins, laneHistogram);
    accumulatePackedWord(packed16.w, from, numBins, laneHistogram);
}

__global__ __launch_bounds__(blockSize, 4)
void histogramKernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    const unsigned int inputSize,
    const int from,
    const int to)
{
    // Dynamic shared memory holds numBins * 32 counters, laid out as [bin][lane].
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int lane = threadIdx.x & (histogramReplicas - 1);
    const unsigned int warpId = threadIdx.x >> histogramReplicaShift;

    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const unsigned int numSharedCounters = numBins << histogramReplicaShift;

    // Each thread writes the counters for its lane at indices:
    // laneHistogram[bin << 5] == sharedHistogram[(bin << 5) + lane]
    unsigned int* const laneHistogram = sharedHistogram + lane;

    // Zero the block-private histogram.
    for (unsigned int i = threadIdx.x; i < numSharedCounters; i += blockSize)
    {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    // Input is guaranteed to come from cudaMalloc according to the problem statement,
    // so the base pointer is highly aligned. Because itemsPerThread is a multiple of 16
    // and each thread processes a contiguous chunk, all vector loads stay 16-byte aligned.
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);
    const uint4* const inputVec4 = reinterpret_cast<const uint4*>(input);

    const size_t inputSizeZ = static_cast<size_t>(inputSize);
    const size_t blockWork = static_cast<size_t>(blockSize) * static_cast<size_t>(itemsPerThread);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockWork;

    // Thread t in a block starts at a contiguous chunk of itemsPerThread bytes.
    size_t base =
        static_cast<size_t>(blockIdx.x) * blockWork +
        static_cast<size_t>(threadIdx.x) * static_cast<size_t>(itemsPerThread);

    // Fast path for full chunks: vectorized 16-byte loads.
    if (inputSizeZ >= static_cast<size_t>(itemsPerThread))
    {
        const size_t lastFullBase = inputSizeZ - static_cast<size_t>(itemsPerThread);
        for (; base <= lastFullBase; base += gridStride)
        {
#pragma unroll
            for (int offset = 0; offset < itemsPerThread; offset += vectorBytes)
            {
                const uint4 packed16 = inputVec4[(base + static_cast<size_t>(offset)) >> 4];
                accumulateVector16(packed16, fromU, numBins, laneHistogram);
            }
        }
    }

    // Tail path: at most one partial chunk per thread.
    if (base < inputSizeZ)
    {
#pragma unroll
        for (int i = 0; i < itemsPerThread; ++i)
        {
            const size_t idx = base + static_cast<size_t>(i);
            if (idx < inputSizeZ)
            {
                addByteToSharedHistogram(
                    static_cast<unsigned int>(inputBytes[idx]),
                    fromU,
                    numBins,
                    laneHistogram);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane replicas of each bin with one warp per logical bin.
    // The read pattern is also bank-conflict free because sharedHistogram[(bin << 5) + lane]
    // maps lane k to bank k.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock)
    {
        const unsigned int localCount = sharedHistogram[(bin << histogramReplicaShift) + lane];
        const unsigned int blockCount = __reduce_add_sync(0xFFFFFFFFu, localCount);

        if (lane == 0 && blockCount != 0u)
        {
            // When only one block is launched, the global atomic is unnecessary.
            if (gridDim.x == 1)
            {
                histogram[bin] = blockCount;
            }
            else
            {
                atomicAdd(histogram + bin, blockCount);
            }
        }
    }
}
}  // namespace

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // The problem statement guarantees a valid range, but a cheap guard avoids
    // accidental misuse from causing invalid shared-memory sizing.
    if (from < 0 || to > 255 || from > to)
    {
        return;
    }

    const int numBins = to - from + 1;

    // The output buffer is defined as the final histogram, not an accumulator,
    // so clear it asynchronously before the kernel launch.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Nothing else to do for empty input; the asynchronous memset already produced
    // the correct all-zero histogram.
    if (inputSize == 0)
    {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(numBins) * static_cast<size_t>(histogramReplicas) * sizeof(unsigned int);

    // This kernel is shared-memory heavy and streams the input, so preferring shared
    // memory over L1 is the right default on recent architectures.
    cudaFuncSetAttribute(histogramKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        histogramKernel,
        blockSize,
        sharedBytes);

    if (blocksPerSM < 1)
    {
        blocksPerSM = 1;
    }

    const size_t blockWork = static_cast<size_t>(blockSize) * static_cast<size_t>(itemsPerThread);
    const size_t neededBlocks =
        (static_cast<size_t>(inputSize) + blockWork - 1) / blockWork;

    // The kernel uses a grid-stride loop, so launching only the number of blocks that
    // can reside concurrently is typically best: it keeps all SMs busy while minimizing
    // block-finalization overhead and the number of global atomics in the final merge.
    int gridSize = smCount * blocksPerSM;
    if (gridSize > static_cast<int>(neededBlocks))
    {
        gridSize = static_cast<int>(neededBlocks);
    }
    if (gridSize < 1)
    {
        gridSize = 1;
    }

    histogramKernel<<<gridSize, blockSize, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        from,
        to);
}