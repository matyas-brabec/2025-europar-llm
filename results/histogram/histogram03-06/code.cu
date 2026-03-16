#include <cuda_runtime.h>
#include <stddef.h>

namespace
{

// The histogram range is restricted to byte values, so the maximum number of bins is 256.
// Choosing 256 threads/block lets the final shared->global flush map one thread to one bin.
constexpr int kWarpSize      = 32;
constexpr int kMaxBins       = 256;
constexpr int kBlockSize     = kMaxBins;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr int kVectorBytes   = static_cast<int>(sizeof(uint4));

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a whole number of warps.");
static_assert(kBlockSize == kMaxBins, "This kernel relies on one thread per possible output bin.");
static_assert(sizeof(uint4) == 16, "The vectorized load path assumes 16-byte uint4 loads.");

// Update one warp-private shared-memory histogram with a single byte.
// activeMask names the currently participating lanes in the warp for this loop iteration.
__device__ __forceinline__ void accumulate_byte_warp_private(
    unsigned int c,
    int from,
    unsigned int numBins,
    unsigned int activeMask,
    int lane,
    unsigned int* warpHist)
{
    const int  bin     = static_cast<int>(c) - from;
    const bool inRange = static_cast<unsigned int>(bin) < numBins;

    // Group equal bins inside the warp so only one lane performs the shared-memory atomic.
    // Invalid bytes use -1 as a sentinel key and are ignored after the match.
    const unsigned int peers = __match_any_sync(activeMask, inRange ? bin : -1);

    if (inRange && lane == (__ffs(peers) - 1)) {
        atomicAdd(&warpHist[bin], static_cast<unsigned int>(__popc(peers)));
    }
}

// Process four bytes packed in a 32-bit word.  Byte order does not matter for a histogram.
__device__ __forceinline__ void accumulate_packed_word(
    unsigned int packed,
    int from,
    unsigned int numBins,
    unsigned int activeMask,
    int lane,
    unsigned int* warpHist)
{
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        accumulate_byte_warp_private(packed & 0xFFu, from, numBins, activeMask, lane, warpHist);
        packed >>= 8;
    }
}

/*
 * Histogram kernel for the contiguous byte range [from, from + numBins - 1].
 *
 * Optimization strategy:
 *   1) Each block uses shared memory for histogram privatization.
 *   2) Privatization is further split per warp to reduce intra-block contention.
 *   3) __match_any_sync collapses same-bin updates within a warp before shared atomics.
 *   4) The common fast path reads input as uint4 (16 bytes/load) when the pointer is aligned.
 *   5) Each block flushes only one partial sum per bin to global memory.
 *
 * The output histogram is indexed from zero: output bin i counts byte value (from + i).
 */
__global__ __launch_bounds__(kBlockSize)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    unsigned int numBins)
{
    extern __shared__ unsigned int sharedHist[];

    const unsigned int tid         = threadIdx.x;
    const unsigned int warpId      = tid / kWarpSize;
    const int          lane        = static_cast<int>(tid % kWarpSize);
    const size_t       globalTid   = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + tid;
    const size_t       totalThreads = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

    // Shared layout: [warp0 bins][warp1 bins]...[warpN bins].
    const unsigned int sharedEntries = static_cast<unsigned int>(kWarpsPerBlock) * numBins;
    for (unsigned int i = tid; i < sharedEntries; i += blockDim.x) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    unsigned int* const warpHist = sharedHist + warpId * numBins;
    const unsigned char* __restrict__ bytes = reinterpret_cast<const unsigned char*>(input);
    const size_t inputBytes = static_cast<size_t>(inputSize);

    // Fast path: cudaMalloc allocations are naturally aligned, so this is the expected case.
    if ((reinterpret_cast<size_t>(input) & static_cast<size_t>(kVectorBytes - 1)) == 0u) {
        const size_t numVec = inputBytes / static_cast<size_t>(kVectorBytes);
        const uint4* __restrict__ vecInput = reinterpret_cast<const uint4*>(input);

        for (size_t i = globalTid; i < numVec; i += totalThreads) {
            const unsigned int activeMask = __activemask();
            const uint4 v = vecInput[i];

            accumulate_packed_word(v.x, from, numBins, activeMask, lane, warpHist);
            accumulate_packed_word(v.y, from, numBins, activeMask, lane, warpHist);
            accumulate_packed_word(v.z, from, numBins, activeMask, lane, warpHist);
            accumulate_packed_word(v.w, from, numBins, activeMask, lane, warpHist);
        }

        const size_t tailStart = numVec * static_cast<size_t>(kVectorBytes);
        for (size_t pos = tailStart + globalTid; pos < inputBytes; pos += totalThreads) {
            const unsigned int activeMask = __activemask();
            accumulate_byte_warp_private(
                static_cast<unsigned int>(bytes[pos]),
                from,
                numBins,
                activeMask,
                lane,
                warpHist);
        }
    } else {
        // Rare correctness fallback for misaligned sub-pointers.
        for (size_t pos = globalTid; pos < inputBytes; pos += totalThreads) {
            const unsigned int activeMask = __activemask();
            accumulate_byte_warp_private(
                static_cast<unsigned int>(bytes[pos]),
                from,
                numBins,
                activeMask,
                lane,
                warpHist);
        }
    }

    __syncthreads();

    // Because kBlockSize == 256 and numBins <= 256, one thread can own one output bin.
    if (tid < numBins) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHist[static_cast<unsigned int>(w) * numBins + tid];
        }
        if (sum != 0u) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

} // namespace

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // Preconditions are those stated in the problem:
    //   - input and histogram are device pointers allocated with cudaMalloc
    //   - 0 <= from < to <= 255
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // This API computes a fresh histogram each time; keep the memset asynchronous.
    (void)cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedMemBytes =
        static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Cap the grid at the occupancy-saturating resident block count so the final shared->global
    // merge does not create more block flushes than needed.
    int device = 0;
    int smCount = 1;
    int activeBlocksPerSM = 1;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel,
        kBlockSize,
        static_cast<int>(sharedMemBytes));

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSM);

    // Use one 16-byte vector per thread as the first-pass work estimate; grid-stride loops
    // handle larger inputs without needing more launch-time blocks.
    const size_t bytesPerBlockFirstPass =
        static_cast<size_t>(kBlockSize) * static_cast<size_t>(kVectorBytes);

    unsigned int gridSize = static_cast<unsigned int>(
        (static_cast<size_t>(inputSize) + bytesPerBlockFirstPass - 1u) / bytesPerBlockFirstPass);

    if (gridSize == 0u) {
        gridSize = 1u;
    }
    if (gridSize > maxResidentBlocks) {
        gridSize = maxResidentBlocks;
    }

    histogram_range_kernel<<<gridSize, kBlockSize, sharedMemBytes, 0>>>(
        input,
        histogram,
        inputSize,
        from,
        numBins);
}