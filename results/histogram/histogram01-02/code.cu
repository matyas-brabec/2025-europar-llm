#include <cuda_runtime.h>
#include <cstddef>

namespace
{
/*
 * Range-restricted byte histogram specialized for modern NVIDIA GPUs.
 *
 * The public API uses `char*`, but histogramming is defined on byte ordinals 0..255,
 * so the kernel interprets the input as unsigned bytes.
 *
 * Hot-loop design:
 *   1) Each warp owns a private shared-memory sub-histogram.
 *   2) Input is read as 16-byte vectors (`uint4`) for efficient global loads.
 *   3) `__match_any_sync` groups lanes that saw the same bin in the current byte
 *      position; only the leader lane updates the warp-private histogram with the
 *      whole peer count. This removes atomics from the hot path.
 *   4) At the end of the block, per-warp histograms are reduced and each block emits
 *      only one global update per logical output bin.
 */
constexpr int HISTOGRAM_BLOCK_SIZE = 256;
constexpr int HISTOGRAM_WARP_SIZE = 32;
constexpr int HISTOGRAM_WARPS_PER_BLOCK = HISTOGRAM_BLOCK_SIZE / HISTOGRAM_WARP_SIZE;
constexpr int HISTOGRAM_MAX_BINS = 256;

/*
 * One pad slot per 32 logical bins. This cheap padding breaks the worst-case shared
 * memory bank aliasing for bins separated by multiples of 32.
 */
constexpr int HISTOGRAM_SHARED_PAD = HISTOGRAM_MAX_BINS / HISTOGRAM_WARP_SIZE;      // 8
constexpr int HISTOGRAM_SHARED_STRIDE = HISTOGRAM_MAX_BINS + HISTOGRAM_SHARED_PAD;  // 264
constexpr int HISTOGRAM_SHARED_WORDS = HISTOGRAM_WARPS_PER_BLOCK * HISTOGRAM_SHARED_STRIDE;

/*
 * 256 threads/block => 8 warps/block. Capping the grid to 4 blocks/SM gives 32 resident
 * warps/SM, which is enough to saturate this kernel on A100/H100-class parts while keeping
 * the final global flush relatively small.
 */
constexpr int HISTOGRAM_TARGET_BLOCKS_PER_SM = 4;

constexpr unsigned int HISTOGRAM_VECTOR_BYTES = 16u;
constexpr unsigned int HISTOGRAM_BLOCK_PASS_BYTES =
    static_cast<unsigned int>(HISTOGRAM_BLOCK_SIZE) * HISTOGRAM_VECTOR_BYTES;
constexpr unsigned int HISTOGRAM_INVALID_KEY = 0xFFFFFFFFu;

static_assert(HISTOGRAM_BLOCK_SIZE % HISTOGRAM_WARP_SIZE == 0, "Block size must be warp-aligned.");
static_assert(sizeof(uint4) == HISTOGRAM_VECTOR_BYTES, "Kernel assumes 16-byte uint4 loads.");

__device__ __forceinline__ unsigned int smem_bin_index(const unsigned int bin)
{
    return bin + (bin >> 5);
}

__device__ __forceinline__ void accumulate_byte_warp_private(const unsigned int byte_value,
                                                             unsigned int* const warp_hist,
                                                             const unsigned int from_u,
                                                             const unsigned int range_u,
                                                             const unsigned int lane_id)
{
    /*
     * Unsigned subtraction + one range comparison filters both sides of the inclusive
     * [from, to] interval: valid iff 0 <= byte_value - from_u < range_u.
     */
    const unsigned int bin = byte_value - from_u;
    const bool valid = (bin < range_u);

    /*
     * All currently active lanes must participate in __match_any_sync with the same mask.
     * Out-of-range bytes therefore use a sentinel key rather than skipping the intrinsic.
     */
    const unsigned int active = __activemask();
    const unsigned int key = valid ? bin : HISTOGRAM_INVALID_KEY;
    const unsigned int peers = __match_any_sync(active, key);

    /*
     * Only one lane per equal-key subgroup updates shared memory, adding the number of
     * peer lanes in one shot. Because each warp owns a private histogram slice, this is
     * a plain shared-memory add, not an atomic.
     */
    const unsigned int leader_lane = static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1);
    if (valid && lane_id == leader_lane)
    {
        warp_hist[smem_bin_index(bin)] += __popc(peers);
    }
}

__device__ __forceinline__ void accumulate_word_warp_private(const unsigned int word,
                                                             unsigned int* const warp_hist,
                                                             const unsigned int from_u,
                                                             const unsigned int range_u,
                                                             const unsigned int lane_id)
{
    accumulate_byte_warp_private((word >> 0)  & 0xFFu, warp_hist, from_u, range_u, lane_id);
    accumulate_byte_warp_private((word >> 8)  & 0xFFu, warp_hist, from_u, range_u, lane_id);
    accumulate_byte_warp_private((word >> 16) & 0xFFu, warp_hist, from_u, range_u, lane_id);
    accumulate_byte_warp_private((word >> 24) & 0xFFu, warp_hist, from_u, range_u, lane_id);
}

__global__ __launch_bounds__(HISTOGRAM_BLOCK_SIZE, HISTOGRAM_TARGET_BLOCKS_PER_SM)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int from_u,
                            const unsigned int range_u)
{
    /*
     * Fixed 8.25 KiB/block shared footprint:
     *   8 warp-private histograms * 264 padded counters/warp.
     */
    __shared__ unsigned int shared_hist[HISTOGRAM_SHARED_WORDS];

    /* Zero all warp-private histograms cooperatively. */
#pragma unroll
    for (unsigned int i = threadIdx.x; i < static_cast<unsigned int>(HISTOGRAM_SHARED_WORDS); i += HISTOGRAM_BLOCK_SIZE)
    {
        shared_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane_id = threadIdx.x & (HISTOGRAM_WARP_SIZE - 1);
    const unsigned int warp_id = threadIdx.x >> 5;
    unsigned int* const warp_hist = &shared_hist[warp_id * HISTOGRAM_SHARED_STRIDE];

    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    /*
     * The problem statement says `input` comes from cudaMalloc, so the base address is
     * naturally aligned for 16-byte vector loads.
     */
    const unsigned int vectorized_count = inputSize / HISTOGRAM_VECTOR_BYTES;
    const uint4* const vector_input = reinterpret_cast<const uint4*>(input);

    for (unsigned int vec_idx = global_thread; vec_idx < vectorized_count; vec_idx += total_threads)
    {
        const uint4 packed = vector_input[vec_idx];
        accumulate_word_warp_private(packed.x, warp_hist, from_u, range_u, lane_id);
        accumulate_word_warp_private(packed.y, warp_hist, from_u, range_u, lane_id);
        accumulate_word_warp_private(packed.z, warp_hist, from_u, range_u, lane_id);
        accumulate_word_warp_private(packed.w, warp_hist, from_u, range_u, lane_id);
    }

    /* Handle the final 0..15 bytes that do not fit in a uint4. */
    const unsigned int tail_start = vectorized_count * HISTOGRAM_VECTOR_BYTES;
    for (unsigned int i = tail_start + global_thread; i < inputSize; i += total_threads)
    {
        accumulate_byte_warp_private(static_cast<unsigned int>(input[i]), warp_hist, from_u, range_u, lane_id);
    }

    __syncthreads();

    /* Reduce per-warp histograms into the compact output range [0, range_u). */
    const bool single_block = (gridDim.x == 1u);
    if (threadIdx.x < range_u)
    {
        const unsigned int bin = threadIdx.x;
        const unsigned int padded_bin = smem_bin_index(bin);

        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < HISTOGRAM_WARPS_PER_BLOCK; ++w)
        {
            sum += shared_hist[w * HISTOGRAM_SHARED_STRIDE + padded_bin];
        }

        /*
         * For a one-block launch there is no inter-block contention, so bypass the
         * global atomic entirely. Otherwise each block contributes once per bin.
         */
        if (single_block)
        {
            histogram[bin] = sum;
        }
        else if (sum != 0u)
        {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

inline unsigned int div_up_u32(const unsigned int n, const unsigned int d)
{
    return (n + d - 1u) / d;
}

inline int histogram_sm_count()
{
    /*
     * CUDA's current device is per host thread, so cache the SM count per host thread too.
     * This avoids a device-attribute query on every invocation while remaining correct if
     * different host threads use different current devices.
     */
    static thread_local int cached_device = -1;
    static thread_local int cached_sm_count = 1;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return cached_sm_count;
    }

    if (device != cached_device)
    {
        int sm_count = 0;
        if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            sm_count > 0)
        {
            cached_sm_count = sm_count;
        }
        else
        {
            cached_sm_count = 1;
        }
        cached_device = device;
    }

    return cached_sm_count;
}
} // namespace

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    /* Defensive validation; the prompt guarantees valid bounds, so this is not on the hot path. */
    if (from < 0 || to > 255 || from > to)
    {
        return;
    }

    /* Compact output: histogram[i] counts byte value (from + i). */
    const unsigned int range_u = static_cast<unsigned int>(to - from + 1);
    if (range_u == 0u || range_u > static_cast<unsigned int>(HISTOGRAM_MAX_BINS))
    {
        return;
    }

    /* The output buffer is defined as a fresh histogram, not an accumulator. */
    const size_t histogram_bytes = static_cast<size_t>(range_u) * sizeof(unsigned int);

    /*
     * The required API has no stream parameter, so this function enqueues work on the
     * default stream and returns immediately. Synchronization is intentionally left
     * to the caller, exactly as requested.
     */
    cudaMemsetAsync(histogram, 0, histogram_bytes);

    if (inputSize == 0u)
    {
        return;
    }

    /*
     * Grid-stride loops make large grids unnecessary. Launch a small multiple of SM count
     * so each block processes many bytes before the final global flush.
     */
    unsigned int grid = div_up_u32(inputSize, HISTOGRAM_BLOCK_PASS_BYTES);
    const unsigned int max_grid =
        static_cast<unsigned int>(histogram_sm_count()) * static_cast<unsigned int>(HISTOGRAM_TARGET_BLOCKS_PER_SM);
    if (grid > max_grid)
    {
        grid = max_grid;
    }

    histogram_range_kernel<<<grid, HISTOGRAM_BLOCK_SIZE>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        range_u);
}