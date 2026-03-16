#include <cuda_runtime.h>
#include <cstddef>

/*
 * Histogram implementation tuned for modern NVIDIA data-center GPUs (A100/H100 class).
 *
 * Design choices:
 * - The histogram range is at most 256 bins, so a block can cheaply keep private partial histograms
 *   in shared memory.
 * - Each warp owns its own shared-memory sub-histogram. This removes inter-warp update contention
 *   while keeping the shared-memory footprint tiny: 8 warps * 256 bins * 4 bytes = 8 KB/block.
 * - Input is scanned with grid-stride loops and 16-byte vector loads (uint4). The prompt guarantees
 *   cudaMalloc-allocated buffers, so using the original allocation pointer yields naturally aligned
 *   vector loads on the intended fast path.
 * - For ASCII-sized ranges and smaller, duplicate bin hits inside a warp are common for text. In that
 *   case we use warp-aggregated updates (__ballot_sync + __match_any_sync) so that each unique bin
 *   touched by a warp step performs only one shared-memory atomicAdd.
 * - For wider ranges, duplicate probability drops and plain shared-memory atomics to warp-private
 *   histograms are usually cheaper.
 * - After the local accumulation phase, each block reduces its warp-private histograms once into the
 *   final global histogram.
 *
 * The host launcher intentionally does not synchronize; the caller owns all host/device synchronization,
 * exactly as requested.
 */

namespace histogram_detail {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kWarpAggregateThresholdBins = 128;  // Text-heavy / ASCII-sized ranges benefit most.

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");
static_assert(kVectorBytes == 16, "This kernel assumes uint4 is 16 bytes.");

inline std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

/*
 * Update policy specialization:
 *   WarpAggregate = true  -> use warp-aggregated shared-memory atomics.
 *   WarpAggregate = false -> direct shared-memory atomic per byte.
 */
template <bool WarpAggregate>
struct HistogramUpdater;

template <>
struct HistogramUpdater<true> {
    __device__ __forceinline__ static void add_byte(unsigned int byte_value,
                                                    unsigned int* warp_hist,
                                                    unsigned int from,
                                                    unsigned int range,
                                                    unsigned int lane,
                                                    unsigned int active_mask) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        const unsigned int rel = byte_value - from;
        const bool in_range = (rel < range);

        // Only lanes currently active in this divergent region participate in the ballot.
        const unsigned int valid_mask = __ballot_sync(active_mask, in_range);

        // Only lanes that actually hit the requested range participate in __match_any_sync.
        // Each unique bin in the warp elects one leader lane that performs a single atomicAdd
        // with the number of peers hitting the same bin.
        if (in_range) {
            const unsigned int peers = __match_any_sync(valid_mask, rel);
            if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
                atomicAdd(warp_hist + rel, static_cast<unsigned int>(__popc(peers)));
            }
        }
#else
        // Fallback for pre-Volta architectures. The stated target is A100/H100, so this path is
        // not performance-critical; it exists only to keep the code broadly compilable.
        (void)lane;
        (void)active_mask;
        const unsigned int rel = byte_value - from;
        if (rel < range) {
            atomicAdd(warp_hist + rel, 1u);
        }
#endif
    }
};

template <>
struct HistogramUpdater<false> {
    __device__ __forceinline__ static void add_byte(unsigned int byte_value,
                                                    unsigned int* warp_hist,
                                                    unsigned int from,
                                                    unsigned int range,
                                                    unsigned int /*lane*/,
                                                    unsigned int /*active_mask*/) {
        const unsigned int rel = byte_value - from;
        if (rel < range) {
            atomicAdd(warp_hist + rel, 1u);
        }
    }
};

template <bool WarpAggregate>
__device__ __forceinline__ void add_word(unsigned int word,
                                         unsigned int* warp_hist,
                                         unsigned int from,
                                         unsigned int range,
                                         unsigned int lane,
                                         unsigned int active_mask) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        HistogramUpdater<WarpAggregate>::add_byte(word & 0xFFu, warp_hist, from, range, lane, active_mask);
        word >>= 8;
    }
}

template <bool WarpAggregate>
__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input_chars,
                            unsigned int* __restrict__ histogram,
                            std::size_t input_size,
                            unsigned int from,
                            unsigned int range) {
    extern __shared__ unsigned int shared_hists[];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & (kWarpSize - 1u);

    // One private histogram per warp.
    unsigned int* const warp_hist = shared_hists + warp_id * range;

    // Zero all warp-private histograms cooperatively.
    const unsigned int total_shared_bins = static_cast<unsigned int>(kWarpsPerBlock) * range;
    for (unsigned int i = tid; i < total_shared_bins; i += kBlockThreads) {
        shared_hists[i] = 0u;
    }
    __syncthreads();

    const unsigned char* const input = reinterpret_cast<const unsigned char*>(input_chars);
    const uint4* const vec_input = reinterpret_cast<const uint4*>(input);

    const std::size_t vec_count = input_size / kVectorBytes;
    const std::size_t global_thread = static_cast<std::size_t>(blockIdx.x) * kBlockThreads + tid;
    const std::size_t grid_stride = static_cast<std::size_t>(gridDim.x) * kBlockThreads;

    // Main vectorized path: each thread consumes 16 bytes per iteration.
    for (std::size_t vec_idx = global_thread; vec_idx < vec_count; vec_idx += grid_stride) {
        const uint4 v = vec_input[vec_idx];
        const unsigned int active_mask = __activemask();

        add_word<WarpAggregate>(v.x, warp_hist, from, range, lane, active_mask);
        add_word<WarpAggregate>(v.y, warp_hist, from, range, lane, active_mask);
        add_word<WarpAggregate>(v.z, warp_hist, from, range, lane, active_mask);
        add_word<WarpAggregate>(v.w, warp_hist, from, range, lane, active_mask);
    }

    // Tail path for the last [0, 15] bytes. Reusing the same update machinery keeps behavior
    // consistent even under partial-warp activity.
    const std::size_t tail_begin = vec_count * kVectorBytes;
    for (std::size_t idx = tail_begin + global_thread; idx < input_size; idx += grid_stride) {
        const unsigned int active_mask = __activemask();
        HistogramUpdater<WarpAggregate>::add_byte(static_cast<unsigned int>(input[idx]),
                                                  warp_hist,
                                                  from,
                                                  range,
                                                  lane,
                                                  active_mask);
    }

    __syncthreads();

    // Reduce warp-private histograms to the final global histogram.
    for (unsigned int bin = tid; bin < range; bin += kBlockThreads) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_hists[w * range + bin];
        }
        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

}  // namespace histogram_detail

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    const unsigned int from_u = static_cast<unsigned int>(from);
    const std::size_t histogram_bytes = static_cast<std::size_t>(range) * sizeof(unsigned int);

    // The output must contain exactly the requested histogram, not an accumulation onto old data.
    // Use the default stream because the interface does not provide a stream parameter.
    cudaMemsetAsync(histogram, 0, histogram_bytes, 0);

    // No input bytes => a zeroed histogram is already the correct answer.
    if (inputSize == 0) {
        return;
    }

    int device = 0;
    int sm_count = 1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sm_count < 1) {
            sm_count = 1;
        }
    }

    const bool use_warp_aggregate = (range <= histogram_detail::kWarpAggregateThresholdBins);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(histogram_detail::kWarpsPerBlock) * range * sizeof(unsigned int);

    // Use occupancy to size the grid to one full device wave of resident blocks.
    int active_blocks_per_sm = 1;
    cudaError_t occ_status = cudaSuccess;
    if (use_warp_aggregate) {
        occ_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            histogram_detail::histogram_range_kernel<true>,
            histogram_detail::kBlockThreads,
            shared_bytes);
    } else {
        occ_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            histogram_detail::histogram_range_kernel<false>,
            histogram_detail::kBlockThreads,
            shared_bytes);
    }
    if (occ_status != cudaSuccess || active_blocks_per_sm < 1) {
        active_blocks_per_sm = 1;
    }

    const int max_blocks = sm_count * active_blocks_per_sm;

    // The main loop consumes 16 bytes per work item. A single occupancy-saturating wave is enough,
    // because the kernel uses grid-stride loops to cover arbitrarily large inputs.
    const std::size_t work_items =
        histogram_detail::ceil_div(static_cast<std::size_t>(inputSize),
                                   static_cast<std::size_t>(histogram_detail::kVectorBytes));
    const std::size_t blocks_needed =
        histogram_detail::ceil_div(work_items,
                                   static_cast<std::size_t>(histogram_detail::kBlockThreads));

    int blocks = max_blocks;
    if (blocks_needed < static_cast<std::size_t>(max_blocks)) {
        blocks = static_cast<int>(blocks_needed);
    }
    if (blocks < 1) {
        blocks = 1;
    }

    if (use_warp_aggregate) {
        histogram_detail::histogram_range_kernel<true>
            <<<blocks, histogram_detail::kBlockThreads, shared_bytes>>>(
                input,
                histogram,
                static_cast<std::size_t>(inputSize),
                from_u,
                range);
    } else {
        histogram_detail::histogram_range_kernel<false>
            <<<blocks, histogram_detail::kBlockThreads, shared_bytes>>>(
                input,
                histogram,
                static_cast<std::size_t>(inputSize),
                from_u,
                range);
    }

    // Intentionally no synchronization here. The caller owns synchronization and error handling.
}