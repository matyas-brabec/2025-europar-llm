#include <cuda_runtime.h>

namespace {

// Fixed launch configuration tuned for modern data-center GPUs (A100/H100 class).
//
// 256 threads/block is deliberate:
// - the maximum possible histogram width is 256 bins (byte values 0..255),
// - so a block can map one thread to one logical output bin during shared-memory
//   initialization and the final flush step.
//
// 8 resident 256-thread blocks/SM = 2048 threads/SM = 64 warps/SM, which matches
// Ampere/Hopper's thread/warp occupancy limit. With the shared-memory footprint used
// below (8 KiB/block), this still fits comfortably.
constexpr int kBlockSize          = 256;
constexpr int kItemsPerThread     = 8;    // Unroll factor: good ILP without excessive code size.
constexpr int kWarpSize           = 32;
constexpr int kWarpsPerBlock      = kBlockSize / kWarpSize;
constexpr int kLocalHistogramBins = 256;  // Full byte alphabet.
constexpr int kTargetBlocksPerSM  = 8;

constexpr unsigned int kFullMask   = 0xFFFFFFFFu;
constexpr unsigned int kInvalidBin = 0xFFFFFFFFu;

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a whole number of warps.");

// Warp-local duplicate aggregation.
// For the current byte position, lanes with the same bin key are grouped with __match_any_sync().
// Only the leader of each equal-key group performs the shared-memory atomicAdd, with the increment
// equal to the group size. This reduces the number of shared-memory atomics substantially for
// text-like inputs with hot characters (space, letters, newline, punctuation, ...).
//
// Even though each warp owns a private sub-histogram, atomicAdd is still used here instead of
// plain "+=" because modern GPUs support independent thread scheduling; non-atomic updates are
// not guaranteed safe across divergent paths / loop iterations. With warp aggregation, these
// atomics are low-contention and therefore cheap.
__device__ __forceinline__ void warpAggregateAndAdd(unsigned int key,
                                                    unsigned int lane,
                                                    unsigned int* __restrict__ warpHist) {
    const unsigned int peers = __match_any_sync(kFullMask, key);
    if (key != kInvalidBin && lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        atomicAdd(warpHist + key, static_cast<unsigned int>(__popc(peers)));
    }
}

// Keep launch_bounds in sync with kBlockSize / kTargetBlocksPerSM above.
__launch_bounds__(256, 8)
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       unsigned int from,
                                       unsigned int range) {
    // One private 256-bin sub-histogram per warp.
    // Shared-memory footprint = 8 warps * 256 bins * 4 bytes = 8 KiB/block.
    //
    // Using full 256-bin warp-private histograms keeps addressing simple and removes inter-warp
    // contention entirely. Only the requested output range is initialized and flushed, so the
    // extra unused bins add negligible cost while keeping the kernel compact and fast.
    __shared__ unsigned int sharedHist[kWarpsPerBlock * kLocalHistogramBins];

    const unsigned int warpId = threadIdx.x >> 5;
    const unsigned int lane   = threadIdx.x & (kWarpSize - 1u);
    unsigned int* const warpHist = sharedHist + warpId * kLocalHistogramBins;

    // range <= 256 by problem definition, so threads 0..range-1 map 1:1 to logical bins.
    // Local bin b corresponds to byte value (from + b).
    if (threadIdx.x < range) {
        const unsigned int bin = threadIdx.x;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sharedHist[w * kLocalHistogramBins + bin] = 0u;
        }
    }
    __syncthreads();

    // Read through unsigned char so bytes are interpreted as 0..255 regardless of whether
    // plain char is signed or unsigned on the compilation platform.
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // Unrolled grid-stride processing.
    //
    // blockBase is block-uniform on purpose: every lane executes the same number of warp-level
    // match operations. Lanes that fall past the end of the input simply contribute kInvalidBin.
    //
    // For each unrolled item, a warp reads a contiguous 32-byte segment. Across all 8 warps,
    // the block reads a contiguous 256-byte stripe, so the byte loads are fully coalesced.
    const size_t n          = static_cast<size_t>(inputSize);
    const size_t blockChunk = static_cast<size_t>(kBlockSize) * kItemsPerThread;
    const size_t gridChunk  = static_cast<size_t>(gridDim.x) * blockChunk;

    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * blockChunk;
         blockBase < n;
         blockBase += gridChunk) {
        #pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const size_t i = blockBase + static_cast<size_t>(item) * kBlockSize + threadIdx.x;

            unsigned int key = kInvalidBin;
            if (i < n) {
                // Convert the input byte into a local bin index in [0, range) if it belongs to
                // the requested interval [from, from + range - 1]. Unsigned arithmetic makes the
                // lower-bound check branchless.
                key = static_cast<unsigned int>(inputBytes[i]) - from;
                if (key >= range) {
                    key = kInvalidBin;
                }
            }

            warpAggregateAndAdd(key, lane, warpHist);
        }
    }

    __syncthreads();

    // Reduce the warp-private sub-histograms inside the block and flush once per logical bin.
    // This reduces global-memory atomics from "one per matching byte" to "one per bin per block".
    if (threadIdx.x < range) {
        const unsigned int bin = threadIdx.x;
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHist[w * kLocalHistogramBins + bin];
        }

        // histogram[bin] counts occurrences of byte value (from + bin).
        //
        // If a single block processes the entire input, it can write the final answer directly.
        // Otherwise multiple blocks contribute to the same output bin, so global atomicAdd is
        // required when flushing block-private counts.
        if (gridDim.x == 1) {
            histogram[bin] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The problem statement guarantees valid device allocations and lets the caller manage
    // synchronization. This function therefore only enqueues work into the default stream
    // and returns immediately.
    //
    // Small defensive guards are kept to avoid illegal launches if the contract is violated.
    if (histogram == nullptr) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(range) * sizeof(unsigned int);

    // Empty input: just zero the requested output range.
    if (input == nullptr || inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
        return;
    }

    // One block covers kBlockSize * kItemsPerThread bytes per outer loop iteration.
    const size_t blockChunk = static_cast<size_t>(kBlockSize) * kItemsPerThread;
    unsigned int blocksForInput = static_cast<unsigned int>(
        (static_cast<size_t>(inputSize) + blockChunk - 1u) / blockChunk);

    unsigned int grid = blocksForInput;

    // If one block is enough, skip the memset: the kernel writes every output bin directly.
    if (grid > 1u) {
        // Cap the grid at 8 blocks/SM. On Ampere/Hopper, that already fills the 64-warp / 2048-thread
        // SM limit. Launching more blocks would only increase per-block flush traffic to global memory
        // and the amount of shared-histogram initialization work.
        int device = 0;
        int smCount = 1;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

        const unsigned int maxBlocks = static_cast<unsigned int>(
            (smCount > 0 ? smCount : 1) * kTargetBlocksPerSM);

        if (grid > maxBlocks) {
            grid = maxBlocks;
        }

        // Multi-block accumulation requires the output histogram to start at zero.
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
    }

    histogram_range_kernel<<<grid, kBlockSize>>>(input, histogram, inputSize, fromU, range);
}