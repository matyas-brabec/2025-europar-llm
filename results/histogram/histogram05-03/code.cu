#include <cuda_runtime.h>
#include <cstddef>

namespace histogram_detail {

// User-requested tuning knob.
// 16 is a strong default on A100/H100-class GPUs for large inputs: it amortizes
// per-block shared-histogram setup/reduction while keeping register pressure modest.
constexpr int itemsPerThread = 16;

// 256 threads is intentional:
// - it is a common throughput sweet spot on modern NVIDIA GPUs,
// - it gives 8 warps/block,
// - and it matches the maximum possible histogram size (256 bins), so the final
//   block-local reduction can assign one thread to one output bin.
constexpr int kBlockSize = 256;
constexpr int kWarpSize  = 32;
constexpr int kMaxBins   = 256;

// Two launch variants are used:
// - <= 128 bins: 4 copies/warp remain thread-limited (full occupancy on A100/H100)
//   and further reduce residual shared-memory bank pressure.
// - > 128 bins: 2 copies/warp keep the worst-case 256-bin histogram at maximum occupancy.
constexpr int          kCopiesPerWarpWide   = 2;
constexpr int          kCopiesPerWarpNarrow = 4;
constexpr unsigned int kFourCopyCutoffBins  = 128u;

static_assert(kBlockSize % kWarpSize == 0, "kBlockSize must be a whole number of warps.");
static_assert(kBlockSize >= kMaxBins, "kBlockSize must cover the full 256-bin maximum.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

static_assert(kCopiesPerWarpWide > 0 && kCopiesPerWarpWide <= kWarpSize, "Invalid wide-range copy count.");
static_assert((kCopiesPerWarpWide & (kCopiesPerWarpWide - 1)) == 0, "Wide-range copy count must be a power of two.");

static_assert(kCopiesPerWarpNarrow > 0 && kCopiesPerWarpNarrow <= kWarpSize, "Invalid narrow-range copy count.");
static_assert((kCopiesPerWarpNarrow & (kCopiesPerWarpNarrow - 1)) == 0, "Narrow-range copy count must be a power of two.");

// Shared-memory layout is copy-major:
//   [copy0 bins...padding][copy1 bins...padding]...
//
// The stride is rounded up to a multiple of the 32 shared-memory banks and then skewed by +1.
// This makes adjacent copies start on successive banks, so the same logical bin in different
// copies does not alias onto the same bank.
__host__ __device__ constexpr unsigned int padded_bins(const unsigned int numBins) {
    return ((numBins + (kWarpSize - 1u)) & ~(kWarpSize - 1u)) + 1u;
}

template <int BLOCK_SIZE, int COPIES_PER_WARP>
__host__ __device__ constexpr std::size_t shared_bytes_for_bins(const unsigned int numBins) {
    constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;
    return static_cast<std::size_t>(kWarpsPerBlock * COPIES_PER_WARP) *
           static_cast<std::size_t>(padded_bins(numBins)) *
           sizeof(unsigned int);
}

// One byte update with warp-aggregated shared-memory atomics.
//
// The key encodes the exact target counter inside the warp as (bin, copyId).
// __match_any_sync() groups lanes that would hit the same shared counter, and only the
// leader lane performs the atomicAdd with the population count. For skewed text data
// (spaces, vowels, newlines, punctuation), this removes a large fraction of exact-address
// atomic collisions at very low overhead on Volta+ / Ampere / Hopper.
template <int COPIES_PER_WARP>
__device__ __forceinline__ void accumulate_byte_warp_aggregated(
    const unsigned char value,
    unsigned int* const localHist,
    const unsigned int from,
    const unsigned int numBins,
    const int copyId,
    const int laneId)
{
    const unsigned int bin = static_cast<unsigned int>(value) - from;
    if (bin < numBins) {
        const unsigned int key =
            bin * static_cast<unsigned int>(COPIES_PER_WARP) + static_cast<unsigned int>(copyId);

        const unsigned int peers      = __match_any_sync(__activemask(), key);
        const int          leaderLane = __ffs(peers) - 1;

        if (laneId == leaderLane) {
            atomicAdd(localHist + bin, static_cast<unsigned int>(__popc(peers)));
        }
    }
}

// Kernel overview:
// 1. Each block allocates several private histograms in shared memory:
//      warpsPerBlock * copiesPerWarp
//    This privatizes updates and spreads them across bank-skewed copies.
// 2. Threads walk the input in a grid-stride loop. In each iteration, a thread processes
//    itemsPerThread bytes spaced by blockDim.x, so every warp load is naturally coalesced.
// 3. At the end, the block reduces all of its shared sub-histograms and accumulates the
//    result into the final global histogram.
//
// Input bytes are read as unsigned char, not plain char, so values >= 128 are handled
// correctly regardless of the host compiler's signed-char default.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int COPIES_PER_WARP>
__launch_bounds__(BLOCK_SIZE)
__global__ void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    const unsigned int inputSize,
    const unsigned int from,
    const unsigned int numBins)
{
    constexpr int         kWarpsPerBlock = BLOCK_SIZE / kWarpSize;
    constexpr int         kTotalCopies   = kWarpsPerBlock * COPIES_PER_WARP;
    constexpr std::size_t kTileSize      = static_cast<std::size_t>(BLOCK_SIZE) * ITEMS_PER_THREAD;

    extern __shared__ unsigned int sHist[];

    const int tid    = threadIdx.x;
    const int laneId = tid & (kWarpSize - 1);
    const int warpId = tid / kWarpSize;

    // Low-bit lane-to-copy mapping interleaves neighboring lanes across copies and is why
    // COPIES_PER_WARP is kept a power of two.
    const int copyId = laneId & (COPIES_PER_WARP - 1);

    const unsigned int paddedBins         = padded_bins(numBins);
    const unsigned int totalSharedCounters = static_cast<unsigned int>(kTotalCopies) * paddedBins;

    // Zero the shared sub-histograms.
    for (unsigned int i = static_cast<unsigned int>(tid); i < totalSharedCounters; i += BLOCK_SIZE) {
        sHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int localHistOffset =
        static_cast<unsigned int>(warpId * COPIES_PER_WARP + copyId) * paddedBins;
    unsigned int* const localHist = sHist + localHistOffset;

    const std::size_t inputSize64 = static_cast<std::size_t>(inputSize);

    // 64-bit grid-stride index math avoids wrap-around for inputs close to the 4 GiB
    // limit implied by the unsigned-int inputSize parameter.
    const std::size_t gridStride = static_cast<std::size_t>(gridDim.x) * kTileSize;

    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * kTileSize + static_cast<std::size_t>(tid);
         index < inputSize64;
         index += gridStride)
    {
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const std::size_t idx = index + static_cast<std::size_t>(item) * BLOCK_SIZE;

            if (idx < inputSize64) {
                accumulate_byte_warp_aggregated<COPIES_PER_WARP>(
                    input[idx], localHist, from, numBins, copyId, laneId);
            }
        }
    }

    __syncthreads();

    // Final block-local reduction: because BLOCK_SIZE >= 256, one thread can own one bin.
    if (static_cast<unsigned int>(tid) < numBins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int copy = 0; copy < kTotalCopies; ++copy) {
            sum += sHist[static_cast<unsigned int>(copy) * paddedBins + static_cast<unsigned int>(tid)];
        }

        // Tiny-input fast path: when only one block is launched, a plain store is sufficient.
        if (gridDim.x == 1) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

// Host-side launcher for one kernel variant.
// Only one resident wave of blocks is launched: the kernel itself grid-strides over the input,
// so extra waves would just repeat shared-histogram init/reduction more times.
template <int COPIES_PER_WARP>
inline void launch_histogram_variant(
    const char* input,
    unsigned int* histogram,
    const unsigned int inputSize,
    const unsigned int from,
    const unsigned int numBins)
{
    const std::size_t sharedBytes = shared_bytes_for_bins<kBlockSize, COPIES_PER_WARP>(numBins);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        histogram_range_kernel<kBlockSize, itemsPerThread, COPIES_PER_WARP>,
        kBlockSize,
        sharedBytes);

    if (blocksPerSM < 1) {
        blocksPerSM = 1;
    }

    constexpr std::size_t kTileSize =
        static_cast<std::size_t>(kBlockSize) * static_cast<std::size_t>(itemsPerThread);

    const unsigned int neededBlocks = static_cast<unsigned int>(
        (static_cast<std::size_t>(inputSize) + kTileSize - 1u) / kTileSize);

    unsigned int launchBlocks = static_cast<unsigned int>(smCount * blocksPerSM);
    if (launchBlocks > neededBlocks) {
        launchBlocks = neededBlocks;
    }
    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }

    const dim3 grid(launchBlocks);
    const dim3 block(kBlockSize);

    histogram_range_kernel<kBlockSize, itemsPerThread, COPIES_PER_WARP>
        <<<grid, block, sharedBytes>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            from,
            numBins);
}

} // namespace histogram_detail

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    const unsigned int fromU   = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The output histogram must start from zero because the kernel only accumulates
    // block-local totals. This memset is asynchronous, matching the requirement that
    // synchronization is owned by the caller.
    cudaMemsetAsync(
        histogram,
        0,
        static_cast<std::size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    // Range-dependent copy count:
    // - <= 128 bins: 4 copies/warp are effectively "free" on A100/H100 (still full occupancy).
    // -  > 128 bins: 2 copies/warp keep the full 256-bin case occupancy-maximal.
    if (numBins <= histogram_detail::kFourCopyCutoffBins) {
        histogram_detail::launch_histogram_variant<histogram_detail::kCopiesPerWarpNarrow>(
            input, histogram, inputSize, fromU, numBins);
    } else {
        histogram_detail::launch_histogram_variant<histogram_detail::kCopiesPerWarpWide>(
            input, histogram, inputSize, fromU, numBins);
    }

    // No host-side synchronization here by design; caller handles ordering / sync / error checks.
}