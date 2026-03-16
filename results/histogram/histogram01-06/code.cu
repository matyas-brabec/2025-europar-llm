#include <cuda_runtime.h>
#include <cstdint>

namespace detail {

/*
 * Range-restricted byte histogram tuned for modern data-center GPUs (A100/H100 class).
 *
 * Design choices:
 *  - One 256-counter shared histogram per warp (warp-private slices). This removes
 *    inter-warp contention in the hot accumulation path.
 *  - The aligned bulk of the input is consumed via 16-byte vector loads (uint4).
 *  - Within a warp, lanes targeting the same bin are coalesced with __match_any_sync(),
 *    so the shared histogram sees one update per distinct value seen by the participating
 *    lanes instead of one update per byte.
 *  - One final merge per block/bin into the global output.
 *
 * Bytes are always interpreted as unsigned values in [0, 255], independent of the
 * implementation-defined signedness of plain 'char'.
 */
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr unsigned int kAlphabetBins = 256u;
constexpr unsigned int kVectorBytes = 16u;
constexpr unsigned int kVectorAlignMask = kVectorBytes - 1u;

/*
 * A fixed cap of 1024 blocks is close to ~8 resident-like blocks/SM on A100/H100-class
 * parts while avoiding too much per-block merge overhead once the device is saturated.
 */
constexpr unsigned int kMaxGridBlocks = 1024u;

/*
 * Launch heuristic: wider requested ranges require more per-block initialization/merge work,
 * so give each block more input to amortize that overhead.
 */
constexpr unsigned int kMinTargetBytesPerBlock = 8u * 1024u;
constexpr unsigned int kTargetBytesPerRequestedBin = 64u;

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a multiple of warp size.");

template <bool kFullRange>
struct ByteAccumulator;

template <>
struct ByteAccumulator<true> {
    __device__ __forceinline__
    static void add(const unsigned int byte_value,
                    unsigned int* const warp_hist,
                    const unsigned int /*from*/,
                    const unsigned int /*bins*/,
                    const unsigned int lane,
                    const unsigned int active_mask)
    {
        const unsigned int peers = __match_any_sync(active_mask, byte_value);
        const unsigned int leader_lane = static_cast<unsigned int>(__ffs(peers) - 1);
        if (lane == leader_lane) {
            const unsigned int count = static_cast<unsigned int>(__popc(peers));
            /*
             * Even though each warp owns its own shared-memory slice, keeping the update as
             * an atomicAdd makes the code correct independent of compiler predication choices
             * and independent thread scheduling. On Ampere/Hopper, shared-memory atomics are
             * fast, and __match_any_sync() has already reduced the update rate sharply.
             */
            atomicAdd(warp_hist + byte_value, count);
        }
    }
};

template <>
struct ByteAccumulator<false> {
    __device__ __forceinline__
    static void add(const unsigned int byte_value,
                    unsigned int* const warp_hist,
                    const unsigned int from,
                    const unsigned int bins,
                    const unsigned int lane,
                    const unsigned int active_mask)
    {
        /*
         * Unsigned subtraction lets one compare implement both bounds:
         * bytes below 'from' underflow to a large unsigned value and fail idx < bins.
         */
        const unsigned int idx = byte_value - from;
        const bool valid = idx < bins;
        const unsigned int valid_mask = __ballot_sync(active_mask, valid);

        if (valid) {
            const unsigned int peers = __match_any_sync(valid_mask, idx);
            const unsigned int leader_lane = static_cast<unsigned int>(__ffs(peers) - 1);
            if (lane == leader_lane) {
                const unsigned int count = static_cast<unsigned int>(__popc(peers));
                atomicAdd(warp_hist + idx, count);
            }
        }
    }
};

template <bool kFullRange>
__device__ __forceinline__
void process_word(const unsigned int word,
                  unsigned int* const warp_hist,
                  const unsigned int from,
                  const unsigned int bins,
                  const unsigned int lane,
                  const unsigned int active_mask)
{
    ByteAccumulator<kFullRange>::add( word        & 0xFFu, warp_hist, from, bins, lane, active_mask);
    ByteAccumulator<kFullRange>::add((word >>  8) & 0xFFu, warp_hist, from, bins, lane, active_mask);
    ByteAccumulator<kFullRange>::add((word >> 16) & 0xFFu, warp_hist, from, bins, lane, active_mask);
    ByteAccumulator<kFullRange>::add((word >> 24) & 0xFFu, warp_hist, from, bins, lane, active_mask);
}

template <bool kFullRange>
__global__ __launch_bounds__(kBlockSize)
void histogram_kernel(const unsigned char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      const unsigned int inputSize,
                      const unsigned int from,
                      const unsigned int bins)
{
    __shared__ unsigned int s_hist[kWarpsPerBlock][kAlphabetBins];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kWarpSize - 1u);
    const unsigned int warp_id = tid >> 5;
    unsigned int* const warp_hist = s_hist[warp_id];

    /* Only initialize the requested bin subset; the rest of the 256-entry slice is unused. */
    for (unsigned int bin = lane; bin < bins; bin += kWarpSize) {
        warp_hist[bin] = 0u;
    }
    __syncthreads();

    const unsigned int global_thread = blockIdx.x * blockDim.x + tid;
    const unsigned int grid_threads = gridDim.x * blockDim.x;

    /* Align the main bulk to 16 bytes so the vectorized loop can use uint4 loads safely. */
    const unsigned int misalignment =
        static_cast<unsigned int>(reinterpret_cast<std::uintptr_t>(input) & kVectorAlignMask);
    unsigned int prefix = (kVectorBytes - misalignment) & kVectorAlignMask;
    if (prefix > inputSize) {
        prefix = inputSize;
    }

    /*
     * The prefix is shorter than 16 bytes, so assigning it to the first few global threads
     * is enough and avoids an extra scalar loop.
     */
    if (global_thread < prefix) {
        const unsigned int active_mask = __activemask();
        ByteAccumulator<kFullRange>::add(
            static_cast<unsigned int>(input[global_thread]),
            warp_hist,
            from,
            bins,
            lane,
            active_mask);
    }

    const unsigned char* const aligned_input = input + prefix;
    const unsigned int vec_bytes = (inputSize - prefix) & ~kVectorAlignMask;
    const unsigned int num_vec = vec_bytes / kVectorBytes;
    const uint4* const input4 = reinterpret_cast<const uint4*>(aligned_input);

    for (unsigned int i = global_thread; i < num_vec; i += grid_threads) {
        const unsigned int active_mask = __activemask();
        const uint4 v = input4[i];
        process_word<kFullRange>(v.x, warp_hist, from, bins, lane, active_mask);
        process_word<kFullRange>(v.y, warp_hist, from, bins, lane, active_mask);
        process_word<kFullRange>(v.z, warp_hist, from, bins, lane, active_mask);
        process_word<kFullRange>(v.w, warp_hist, from, bins, lane, active_mask);
    }

    const unsigned int tail_start = prefix + vec_bytes;
    const unsigned int tail_bytes = inputSize - tail_start;

    /* Like the prefix, the tail is also shorter than 16 bytes. */
    if (global_thread < tail_bytes) {
        const unsigned int active_mask = __activemask();
        ByteAccumulator<kFullRange>::add(
            static_cast<unsigned int>(input[tail_start + global_thread]),
            warp_hist,
            from,
            bins,
            lane,
            active_mask);
    }

    __syncthreads();

    /*
     * Merge warp-private shared histograms into the output.
     * - Single-block launch: direct stores avoid both a memset and global atomics.
     * - Multi-block launch: one global atomic per non-zero bin/block.
     */
    for (unsigned int bin = tid; bin < bins; bin += blockDim.x) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += s_hist[w][bin];
        }

        if (gridDim.x == 1) {
            histogram[bin] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

}  // namespace detail

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    /*
     * Preconditions from the prompt:
     *   - 0 <= from < to <= 255
     *   - input and histogram are device pointers allocated by cudaMalloc
     *   - histogram has length (to - from + 1)
     *
     * By design, this function only enqueues work on the default stream and does not
     * synchronize; ordering/error handling beyond launch submission is left to the caller.
     */
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const size_t histogram_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogram_bytes, 0);
        return;
    }

    unsigned int target_bytes_per_block = bins * detail::kTargetBytesPerRequestedBin;
    if (target_bytes_per_block < detail::kMinTargetBytesPerBlock) {
        target_bytes_per_block = detail::kMinTargetBytesPerBlock;
    }

    unsigned int blocks = 1u + (inputSize - 1u) / target_bytes_per_block;
    if (blocks > detail::kMaxGridBlocks) {
        blocks = detail::kMaxGridBlocks;
    }

    /*
     * When only one block is used, the kernel writes the complete result directly,
     * so a separate memset is unnecessary.
     */
    if (blocks != 1u) {
        cudaMemsetAsync(histogram, 0, histogram_bytes, 0);
    }

    const dim3 grid(blocks);
    const dim3 block(detail::kBlockSize);
    const unsigned char* const input_u8 = reinterpret_cast<const unsigned char*>(input);
    const unsigned int from_u = static_cast<unsigned int>(from);

    /* Fast path for the full [0,255] alphabet: no range check in the hot loop. */
    if (from == 0 && bins == detail::kAlphabetBins) {
        detail::histogram_kernel<true><<<grid, block>>>(
            input_u8,
            histogram,
            inputSize,
            0u,
            detail::kAlphabetBins);
    } else {
        detail::histogram_kernel<false><<<grid, block>>>(
            input_u8,
            histogram,
            inputSize,
            from_u,
            bins);
    }
}