#include <cuda_runtime.h>

/*
 * Range-restricted byte histogram for a device-resident text buffer.
 *
 * Optimization strategy tuned for modern NVIDIA data-center GPUs:
 * - 256-thread CTAs: this gives 8 warps/CTA and exactly one thread per possible output bin
 *   during the final block reduction (the requested range is at most 256 bytes wide).
 * - One 256-bin shared histogram per warp: 8 warps * 256 bins * 4 bytes = 8 KiB/CTA, which is
 *   cheap on A100/H100 and removes inter-warp contention in the hot path.
 * - The shared histograms are indexed by the raw byte value 0..255, not by (value - from).
 *   That keeps the inner loop simple and avoids variable-sized shared-memory indexing.
 * - The input buffer is guaranteed by the problem statement to come from cudaMalloc, so its base
 *   pointer is naturally aligned for 16-byte uint4 loads.
 * - __match_any_sync groups equal byte values within a warp. Only the leader of each group issues
 *   the shared-memory atomic add, which reduces atomic traffic for skewed text and for narrow
 *   [from, to] ranges where most bytes are discarded.
 */

namespace {
constexpr int           kBlockThreads         = 256;
constexpr unsigned int  kBlockThreadsU        = 256u;
constexpr int           kWarpSize             = 32;
constexpr int           kWarpsPerBlock        = kBlockThreads / kWarpSize;  // 8
constexpr int           kSharedBins           = 256;
constexpr unsigned int  kVectorBytes          = 16u;                        // sizeof(uint4)
constexpr unsigned int  kBytesPerBlock        = kBlockThreadsU * kVectorBytes;
constexpr unsigned int  kMaxBlocks            = 1024u;                      // ~8 CTAs/SM on H100-class GPUs
constexpr unsigned int  kSingleBlockThreshold = 64u * 1024u;               // small-input fast path
constexpr unsigned int  kInvalidKey           = 0xFFFFFFFFu;

static_assert(kBlockThreads == kSharedBins,
              "The final reduction relies on having one thread available per possible output bin.");

/*
 * Aggregate one byte into the warp-private shared histogram.
 *
 * Unsigned arithmetic folds the lower-bound and upper-bound tests into one compare:
 *   (byte_value - from_u) < bins
 *
 * All active lanes participate in __match_any_sync. Lanes whose byte is out of range use a sentinel
 * key so they still participate in the warp primitive but do not update the histogram.
 *
 * Only the leader lane of each equal-key group issues the atomic add, using the population count of
 * the group as the increment.
 */
__device__ __forceinline__
void aggregate_byte_to_warp_hist(unsigned int byte_value,
                                 unsigned int from_u,
                                 unsigned int bins,
                                 unsigned int* warp_hist,
                                 unsigned int active_mask,
                                 unsigned int lane_mask)
{
    const bool in_range = ((byte_value - from_u) < bins);
    const unsigned int key = in_range ? byte_value : kInvalidKey;

    const unsigned int peer_mask   = __match_any_sync(active_mask, key);
    const unsigned int leader_mask = peer_mask & (0u - peer_mask);  // isolate least-significant set bit

    if (in_range && leader_mask == lane_mask) {
        atomicAdd(warp_hist + key, static_cast<unsigned int>(__popc(peer_mask)));
    }
}

/*
 * Unpack four bytes from one 32-bit word and feed them to the warp-aggregation helper.
 * The function is force-inlined so the compiler can keep the whole hot path in registers.
 */
__device__ __forceinline__
void process_packed_word(unsigned int packed,
                         unsigned int from_u,
                         unsigned int bins,
                         unsigned int* warp_hist,
                         unsigned int active_mask,
                         unsigned int lane_mask)
{
    aggregate_byte_to_warp_hist( packed        & 0xFFu, from_u, bins, warp_hist, active_mask, lane_mask);
    aggregate_byte_to_warp_hist((packed >>  8) & 0xFFu, from_u, bins, warp_hist, active_mask, lane_mask);
    aggregate_byte_to_warp_hist((packed >> 16) & 0xFFu, from_u, bins, warp_hist, active_mask, lane_mask);
    aggregate_byte_to_warp_hist((packed >> 24) & 0xFFu, from_u, bins, warp_hist, active_mask, lane_mask);
}

__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from_u,
                            unsigned int bins)
{
    __shared__ unsigned int warp_histograms[kWarpsPerBlock][kSharedBins];

    const unsigned int tid          = threadIdx.x;
    const unsigned int global_tid   = blockIdx.x * kBlockThreadsU + tid;
    const unsigned int total_threads = gridDim.x * kBlockThreadsU;
    const unsigned int warp_id      = tid >> 5;
    const unsigned int lane_mask    = 1u << (tid & 31u);

    unsigned int* const warp_hist         = warp_histograms[warp_id];
    const unsigned char* const byte_input = reinterpret_cast<const unsigned char*>(input);
    const uint4* const vec_input          = reinterpret_cast<const uint4*>(input);

    // Zero the warp-private shared histograms. This is only 8 KiB/CTA.
    for (unsigned int i = tid; i < static_cast<unsigned int>(kWarpsPerBlock * kSharedBins); i += kBlockThreadsU) {
        reinterpret_cast<unsigned int*>(warp_histograms)[i] = 0u;
    }
    __syncthreads();

    // Main vectorized pass: each thread reads 16 bytes at a time via a naturally aligned uint4 load.
    const unsigned int vecCount = inputSize >> 4;  // inputSize / 16
    for (unsigned int i = global_tid; i < vecCount; i += total_threads) {
        const uint4 v = vec_input[i];
        const unsigned int active_mask = __activemask();

        process_packed_word(v.x, from_u, bins, warp_hist, active_mask, lane_mask);
        process_packed_word(v.y, from_u, bins, warp_hist, active_mask, lane_mask);
        process_packed_word(v.z, from_u, bins, warp_hist, active_mask, lane_mask);
        process_packed_word(v.w, from_u, bins, warp_hist, active_mask, lane_mask);
    }

    // Handle the tail (< 16 bytes total) with a scalar path. The overhead is negligible.
    const unsigned int tail_begin = vecCount * kVectorBytes;
    for (unsigned int i = tail_begin + global_tid; i < inputSize; i += total_threads) {
        const unsigned int key = static_cast<unsigned int>(byte_input[i]);
        if ((key - from_u) < bins) {
            atomicAdd(warp_hist + key, 1u);
        }
    }

    __syncthreads();

    /*
     * Reduce the warp-private histograms to the requested output range.
     * Because the output width is at most 256, one thread can own one output bin.
     *
     * - Single-CTA path: overwrite the output directly, so no prior memset is needed.
     * - Multi-CTA path: each CTA atomically accumulates its partial sums into the global histogram.
     */
    if (tid < bins) {
        const unsigned int out_bin = tid;
        const unsigned int key     = from_u + out_bin;

        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += warp_histograms[w][key];
        }

        if (gridDim.x == 1) {
            histogram[out_bin] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + out_bin, sum);
        }
    }
}
}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The contract guarantees: 0 <= from < to <= 255.
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int bins   = static_cast<unsigned int>(to - from) + 1u;

    /*
     * Small-input fast path:
     * A single CTA usually wins for tiny inputs because it avoids:
     *   1) an asynchronous memset of the output buffer, and
     *   2) the final global atomic accumulation.
     *
     * Larger inputs use enough CTAs to saturate the GPU, but the count is capped so the
     * per-block global flush cost stays bounded.
     *
     * The caller is responsible for synchronization and error handling; this function only
     * enqueues work on the default stream.
     */
    if (inputSize == 0u) {
        (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));
        return;
    }

    unsigned int blocks;
    if (inputSize <= kSingleBlockThreshold) {
        blocks = 1u;
    } else {
        blocks = 1u + ((inputSize - 1u) / kBytesPerBlock);  // overflow-safe ceil(inputSize / kBytesPerBlock)
        if (blocks > kMaxBlocks) {
            blocks = kMaxBlocks;
        }
    }

    if (blocks > 1u) {
        (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));
    }

    histogram_range_kernel<<<blocks, kBlockThreads>>>(input, histogram, inputSize, from_u, bins);
}