#include <cuda_runtime.h>

namespace histogram_detail {

// High-level design:
// - The requested histogram range is contiguous and has at most 256 bins.
// - We therefore use a fixed 256-thread block so the final reduction can map one
//   thread to one possible logical bin.
// - Each block owns a shared-memory histogram, but it is privatized per warp to
//   reduce contention further.
// - Input is consumed with 16-byte vector loads (uint4) for good memory throughput.
// - On modern NVIDIA GPUs (A100/H100 and other Volta+ parts), __match_any_sync()
//   lets a warp collapse duplicate byte values so only one lane performs the
//   shared-memory update for each distinct byte seen by that warp step.
// - After the block finishes scanning its grid-stride slice, the per-warp shared
//   histograms are reduced and the block emits one global atomicAdd per nonzero bin.

constexpr int          kHistogramBlockThreads = 256;
constexpr int          kTargetBlocksPerSM     = 4;
constexpr int          kMaxLogicalBins        = 256;
constexpr int          kSharedBankCount       = 32;
constexpr int          kWarpHistogramStride   = kMaxLogicalBins + (kMaxLogicalBins / kSharedBankCount); // 264
constexpr unsigned int kVectorLoadBytes       = 16u;    // one uint4
constexpr unsigned int kOutOfRangeKey         = ~0u;    // sentinel for lanes whose byte is outside [from, to]

static_assert(sizeof(uint4) == kVectorLoadBytes, "uint4 must remain a 16-byte vector load.");
static_assert(kHistogramBlockThreads % 32 == 0, "Block size must be a whole number of warps.");
static_assert(kHistogramBlockThreads >= kMaxLogicalBins,
              "Block size must cover the maximum possible histogram bin count.");

// Skew the shared-memory layout by one padding slot per 32 logical bins.
// This spreads accesses that would otherwise land on the same bank when bins differ by 32.
__device__ __forceinline__ unsigned int shared_bin_index(const unsigned int logical_bin) {
    return logical_bin + (logical_bin >> 5);
}

__device__ __forceinline__ void accumulate_byte_to_warp_hist(
    const unsigned int byte_value,
    const unsigned int from,
    const unsigned int bins,
    const unsigned int lane_id,
    const unsigned int active_mask,
    unsigned int* const warp_hist)
{
    const unsigned int logical_bin = byte_value - from;
    const bool         in_range    = logical_bin < bins;
    const unsigned int key         = in_range ? logical_bin : kOutOfRangeKey;

    // Group lanes that saw the same logical bin. Typical text has a skewed byte
    // distribution (spaces, vowels, punctuation, etc.), so this often reduces
    // many per-byte updates into a single shared-memory atomicAdd.
    const unsigned int peer_mask   = __match_any_sync(active_mask, key);

    if (in_range) {
        const unsigned int leader_lane =
            static_cast<unsigned int>(__ffs(static_cast<int>(peer_mask)) - 1);

        if (leader_lane == lane_id) {
            // Even though the histogram is warp-private, we still use atomics here.
            // On Volta+ GPUs, independent thread scheduling means some lanes can
            // reach different loop iterations at different times; atomics keep the
            // update correct without forcing a __syncwarp() after every byte.
            atomicAdd(&warp_hist[shared_bin_index(logical_bin)],
                      static_cast<unsigned int>(__popc(peer_mask)));
        }
    }
}

__device__ __forceinline__ void accumulate_packed_word_to_warp_hist(
    const unsigned int packed_word,
    const unsigned int from,
    const unsigned int bins,
    const unsigned int lane_id,
    const unsigned int active_mask,
    unsigned int* const warp_hist)
{
    // packed_word contains 4 adjacent bytes loaded from the input stream.
    accumulate_byte_to_warp_hist((packed_word >>  0) & 0xFFu, from, bins, lane_id, active_mask, warp_hist);
    accumulate_byte_to_warp_hist((packed_word >>  8) & 0xFFu, from, bins, lane_id, active_mask, warp_hist);
    accumulate_byte_to_warp_hist((packed_word >> 16) & 0xFFu, from, bins, lane_id, active_mask, warp_hist);
    accumulate_byte_to_warp_hist((packed_word >> 24) & 0xFFu, from, bins, lane_id, active_mask, warp_hist);
}

template <int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS, kTargetBlocksPerSM)
__global__ void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    const unsigned int inputSize,
    const unsigned int from,
    const unsigned int bins)
{
    static_assert(BLOCK_THREADS % 32 == 0, "Block size must be a whole number of warps.");
    static_assert(BLOCK_THREADS >= kMaxLogicalBins,
                  "Block size must cover the maximum possible histogram bin count.");

    constexpr unsigned int kBlockThreadsU = static_cast<unsigned int>(BLOCK_THREADS);
    constexpr int          kWarpsPerBlock = BLOCK_THREADS / 32;

    // One histogram per warp. Keeping warps private avoids inter-warp contention.
    __shared__ unsigned int warp_histograms[kWarpsPerBlock][kWarpHistogramStride];

    // Flattened pointer makes the initialization loop simple and fast.
    unsigned int* const shared_storage = &warp_histograms[0][0];

    for (unsigned int i = threadIdx.x;
         i < static_cast<unsigned int>(kWarpsPerBlock * kWarpHistogramStride);
         i += kBlockThreadsU)
    {
        shared_storage[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane_id       = threadIdx.x & 31u;
    const unsigned int warp_id       = threadIdx.x >> 5;
    unsigned int* const warp_hist    = warp_histograms[warp_id];
    const unsigned int global_thread = blockIdx.x * kBlockThreadsU + threadIdx.x;
    const unsigned int global_stride = gridDim.x * kBlockThreadsU;

    // Vectorized main pass over 16-byte chunks.
    const unsigned int vector_count  = inputSize / kVectorLoadBytes;
    const uint4* const input_vectors = reinterpret_cast<const uint4*>(input);

    for (unsigned int i = global_thread; i < vector_count; i += global_stride) {
        const uint4 packed = input_vectors[i];

        // All lanes currently executing this loop iteration use the same mask.
        const unsigned int active_mask = __activemask();

        accumulate_packed_word_to_warp_hist(packed.x, from, bins, lane_id, active_mask, warp_hist);
        accumulate_packed_word_to_warp_hist(packed.y, from, bins, lane_id, active_mask, warp_hist);
        accumulate_packed_word_to_warp_hist(packed.z, from, bins, lane_id, active_mask, warp_hist);
        accumulate_packed_word_to_warp_hist(packed.w, from, bins, lane_id, active_mask, warp_hist);
    }

    // Scalar cleanup for the final 0..15 leftover bytes.
    const unsigned int tail_start = vector_count * kVectorLoadBytes;
    for (unsigned int i = tail_start + global_thread; i < inputSize; i += global_stride) {
        const unsigned int active_mask = __activemask();
        accumulate_byte_to_warp_hist(static_cast<unsigned int>(input[i]),
                                     from, bins, lane_id, active_mask, warp_hist);
    }

    __syncthreads();

    // Final reduction: one thread per logical bin sums the per-warp partials and
    // emits exactly one global atomicAdd for this block/bin pair.
    if (threadIdx.x < bins) {
        const unsigned int logical_bin = threadIdx.x;
        const unsigned int shared_bin  = shared_bin_index(logical_bin);

        unsigned int block_sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            block_sum += warp_histograms[w][shared_bin];
        }

        if (block_sum != 0u) {
            atomicAdd(histogram + logical_bin, block_sum);
        }
    }
}

} // namespace histogram_detail

// Assumptions from the problem statement:
// - input and histogram are device pointers allocated with cudaMalloc.
// - histogram has room for (to - from + 1) unsigned ints.
// - The caller is responsible for any needed synchronization.
// This function therefore queues work asynchronously on default-stream semantics
// and returns immediately.
void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The prompt guarantees a valid inclusive range, but guard anyway to avoid UB.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int bins = static_cast<unsigned int>(to - from) + 1u;
    const size_t       histogram_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // Zero the output asynchronously before accumulation.
    cudaMemsetAsync(histogram, 0, histogram_bytes, 0);

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    // Each work item is one 16-byte vector load, plus a possible final tail item.
    const unsigned int work_items =
        (inputSize / histogram_detail::kVectorLoadBytes) +
        ((inputSize % histogram_detail::kVectorLoadBytes) != 0u ? 1u : 0u);

    unsigned int blocks_needed =
        (work_items + histogram_detail::kHistogramBlockThreads - 1u) /
        histogram_detail::kHistogramBlockThreads;

    if (blocks_needed == 0u) {
        blocks_needed = 1u;
    }

    // Cap the grid to a modest number of blocks per SM. That is enough to keep
    // A100/H100-class GPUs busy while limiting the final global-atomic pressure
    // (one global atomic per block/bin).
    unsigned int target_grid =
        static_cast<unsigned int>(sm_count) *
        static_cast<unsigned int>(histogram_detail::kTargetBlocksPerSM);

    if (target_grid == 0u) {
        target_grid = 1u;
    }

    const unsigned int grid = (blocks_needed < target_grid) ? blocks_needed : target_grid;

    histogram_detail::histogram_range_kernel<histogram_detail::kHistogramBlockThreads>
        <<<grid, histogram_detail::kHistogramBlockThreads, 0, 0>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            static_cast<unsigned int>(from),
            bins);
}