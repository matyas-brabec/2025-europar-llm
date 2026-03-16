#include <cuda_runtime.h>
#include <cstdint>

// Range-restricted byte histogram for a device-resident text buffer.
//
// histogram[bin] counts the number of input bytes whose unsigned ordinal value is
// (from + bin). Bytes outside [from, to] are ignored.
//
// Optimization strategy:
// 1) Each block accumulates into shared memory instead of hammering global memory.
// 2) Shared-memory histogram privatization uses 32 copies per block, indexed by warp lane.
//    This spreads hot-bin updates across the 32 shared-memory banks.
// 3) The distance between two copies is forced to an odd number of 32-bit words.
//    For same-bin updates from a warp, that makes the bank mapping a permutation mod 32,
//    removing the worst intra-warp bank-conflict pattern.
// 4) The grid is launched in a persistent-style configuration and walks the input with a
//    grid-stride loop, amortizing the final global reduction.
// 5) Full tiles use coalesced uchar4 vector loads; the rare partial tail tile falls back
//    to scalar loads.

constexpr int blockSize = 256;

// Requested tuning knob.
// 16 is a strong default on A100/H100-class GPUs for large inputs: enough ILP to
// cover latency, but still a modest register footprint for this atomic-heavy kernel.
constexpr int itemsPerThread = 16;

constexpr int warpLanes = 32;
constexpr int sharedHistCopies = warpLanes;

// Use 4-byte vector loads in the fast path.
using LoadType = uchar4;
constexpr int vectorWidth = sizeof(LoadType);

constexpr size_t tileCharsPerBlock =
    static_cast<size_t>(blockSize) * static_cast<size_t>(itemsPerThread);

static_assert(blockSize % warpLanes == 0, "blockSize must be a multiple of the warp size.");
static_assert(blockSize % vectorWidth == 0,
              "blockSize must preserve vector-load alignment across tiles.");

__host__ __device__ __forceinline__ int privatized_hist_stride(const int numBins)
{
    // Force an odd stride between histogram copies. With the per-lane copy layout
    // below, an odd stride makes copy->bank mapping a permutation modulo 32.
    return numBins | 1;
}

__device__ __forceinline__ void accumulate_byte(const unsigned int byteValue,
                                                const unsigned int lower,
                                                const unsigned int numBins,
                                                unsigned int *threadHist)
{
    // Unsigned subtraction + a single comparison handles both bounds at once.
    const unsigned int bin = byteValue - lower;
    if (bin < numBins) {
        atomicAdd(&threadHist[bin], 1u);
    }
}

__device__ __forceinline__ void accumulate_load(const LoadType packed,
                                                const unsigned int lower,
                                                const unsigned int numBins,
                                                unsigned int *threadHist)
{
    accumulate_byte(static_cast<unsigned int>(packed.x), lower, numBins, threadHist);
    accumulate_byte(static_cast<unsigned int>(packed.y), lower, numBins, threadHist);
    accumulate_byte(static_cast<unsigned int>(packed.z), lower, numBins, threadHist);
    accumulate_byte(static_cast<unsigned int>(packed.w), lower, numBins, threadHist);
}

__global__ __launch_bounds__(blockSize)
void histogram_range_kernel(const unsigned char *__restrict__ input,
                            unsigned int *__restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int lower,
                            const unsigned int numBins)
{
    // Shared-memory layout:
    //   [copy 0][copy 1]...[copy 31]
    // Each thread updates the copy selected by its lane ID.
    // Different warps reuse the same lane-indexed copy set, which keeps shared memory
    // usage modest while removing the dominant intra-warp bank-conflict pattern.
    extern __shared__ unsigned int sHist[];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (warpLanes - 1);
    const unsigned int tidU = static_cast<unsigned int>(tid);
    const size_t tid64 = static_cast<size_t>(tid);

    const int stride = privatized_hist_stride(static_cast<int>(numBins));
    const int sharedWords = sharedHistCopies * stride;
    unsigned int *const threadHist = sHist + lane * stride;

    // Zero the block-private histograms.
    for (int i = tid; i < sharedWords; i += blockSize) {
        sHist[i] = 0u;
    }
    __syncthreads();

    constexpr int vecLoadsPerThread = itemsPerThread / vectorWidth;
    constexpr int scalarRemainder = itemsPerThread % vectorWidth;

    const size_t inputSize64 = static_cast<size_t>(inputSize);
    const size_t firstTileBase = static_cast<size_t>(blockIdx.x) * tileCharsPerBlock;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * tileCharsPerBlock;

    // Full tiles can use vectorized loads when the input pointer itself is aligned.
    // A raw cudaMalloc buffer is aligned, but this extra check keeps the kernel correct
    // even if a caller passes an offset pointer into that buffer.
    const bool canVectorize =
        (reinterpret_cast<std::uintptr_t>(input) &
         static_cast<std::uintptr_t>(vectorWidth - 1)) == 0;

    const LoadType *const inputVec = reinterpret_cast<const LoadType *>(input);

    for (size_t tileBase = firstTileBase; tileBase < inputSize64; tileBase += gridStride) {
        if (canVectorize && tileBase + tileCharsPerBlock <= inputSize64) {
            // Fast path for a full tile: vector loads with no per-element bounds checks.
            const size_t tileBaseVec = tileBase / static_cast<size_t>(vectorWidth);

            #pragma unroll
            for (int vec = 0; vec < vecLoadsPerThread; ++vec) {
                const LoadType packed =
                    inputVec[tileBaseVec + static_cast<size_t>(vec) * blockSize + tid64];
                accumulate_load(packed, lower, numBins, threadHist);
            }

            const size_t scalarBase =
                tileBase +
                static_cast<size_t>(vecLoadsPerThread) * blockSize * static_cast<size_t>(vectorWidth);

            #pragma unroll
            for (int rem = 0; rem < scalarRemainder; ++rem) {
                const size_t idx = scalarBase + static_cast<size_t>(rem) * blockSize + tid64;
                accumulate_byte(static_cast<unsigned int>(input[idx]), lower, numBins, threadHist);
            }
        } else {
            // Tail path (or misaligned input pointer): scalar loads keep the fast path clean.
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                const size_t idx = tileBase + static_cast<size_t>(item) * blockSize + tid64;
                if (idx < inputSize64) {
                    accumulate_byte(static_cast<unsigned int>(input[idx]), lower, numBins, threadHist);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 privatized copies into the final global histogram.
    for (unsigned int bin = tidU; bin < numBins; bin += blockSize) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int copy = 0; copy < sharedHistCopies; ++copy) {
            sum += sHist[copy * stride + static_cast<int>(bin)];
        }

        if (sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

static inline int ceil_div_size_t(const size_t x, const size_t y)
{
    return static_cast<int>(x / y + ((x % y) != 0));
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int lower = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // This function computes a fresh histogram, so clear the output first.
    // The caller requested that synchronization be handled externally; this function
    // only enqueues work on the default stream.
    cudaMemsetAsync(
        histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(sharedHistCopies) *
        static_cast<size_t>(privatized_hist_stride(static_cast<int>(numBins))) *
        sizeof(unsigned int);

    // Persistent-style launch:
    // use roughly the number of concurrently resident blocks, then let each block
    // process many tiles through a grid-stride loop. This minimizes the number of
    // block-final global atomic reductions compared to launching one block per tile.
    int activeBlocksPerSM = 1;
    int queriedActiveBlocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &queriedActiveBlocks, histogram_range_kernel, blockSize, sharedBytes) == cudaSuccess &&
        queriedActiveBlocks > 0) {
        activeBlocksPerSM = queriedActiveBlocks;
    }

    int smCount = 1;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int queriedSMs = 0;
        if (cudaDeviceGetAttribute(
                &queriedSMs, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            queriedSMs > 0) {
            smCount = queriedSMs;
        }
    }

    const int blocksNeeded =
        ceil_div_size_t(static_cast<size_t>(inputSize), tileCharsPerBlock);

    int gridSize = smCount * activeBlocksPerSM;
    if (gridSize < 1) {
        gridSize = 1;
    }
    if (gridSize > blocksNeeded) {
        gridSize = blocksNeeded;
    }

    histogram_range_kernel<<<gridSize, blockSize, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char *>(input),  // preserve byte ordinals 0..255
        histogram,
        inputSize,
        lower,
        numBins);
}