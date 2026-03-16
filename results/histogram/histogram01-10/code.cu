#include <cuda_runtime.h>
#include <cstdint>

// Range-restricted histogram for a device-resident text buffer.
//
// Optimization strategy for modern NVIDIA data-center GPUs (A100/H100 class):
// - 256-thread blocks, so the final block-local reduction has one thread available for every
//   possible output bin (the requested range is at most 256 values wide).
// - One private 256-bin shared-memory histogram per warp. Full 256-bin tables are used even
//   for smaller requested ranges because 8 KiB/block is still tiny, while raw-byte indexing
//   becomes trivial and address arithmetic stays fast.
// - Warp-aggregated updates via __match_any_sync(): identical characters observed by multiple
//   lanes in the same warp collapse to a single shared-memory atomicAdd with the combined count.
// - 16-byte vectorized loads for 16-byte-aligned input pointers (the normal cudaMalloc case).
//   A scalar fallback kernel handles uncommon misaligned device pointers correctly.
// - The host wrapper zeroes the output with cudaMemsetAsync() and launches asynchronously on
//   the default stream. No host/device synchronization is performed here, per the requirement.
namespace {

constexpr int kAlphabetBins            = 256;
constexpr int kWarpSize                = 32;
constexpr int kBlockThreads            = 256;
constexpr int kWarpsPerBlock           = kBlockThreads / kWarpSize;
constexpr int kGridOversubscription    = 2;
constexpr int kVectorBytes             = static_cast<int>(sizeof(uint4));
constexpr unsigned int kInvalidKey     = 0xFFFFFFFFu;
constexpr int kSharedCountersPerBlock  = kWarpsPerBlock * kAlphabetBins;
constexpr int kSharedClearsPerThread   = kSharedCountersPerBlock / kBlockThreads;

static_assert(kBlockThreads == 256, "This implementation is tuned for 256-thread blocks.");
static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a whole number of warps.");
static_assert(kAlphabetBins == 256, "This implementation assumes byte values in the range 0..255.");
static_assert(kSharedCountersPerBlock % kBlockThreads == 0,
              "Shared histogram clearing must divide evenly across threads.");

struct LaunchConfigCache {
    int device                 = -1;
    int sm_count               = 1;
    int blocks_per_sm_aligned  = 1;
    int blocks_per_sm_scalar   = 1;
};

// Every active lane must participate in the same warp primitive. To keep the control flow
// uniform, out-of-range bytes are mapped to an invalid sentinel key; they participate in
// __match_any_sync(), but never issue an atomicAdd. Valid bytes use their raw ordinal value
// (0..255) directly as the shared-memory histogram index.
__device__ __forceinline__
void accumulate_raw_char_warp_aggregated(unsigned int raw_char,
                                         unsigned int from_u,
                                         unsigned int range_u,
                                         unsigned int* warp_hist,
                                         unsigned int lane_id)
{
    const unsigned int key =
        (((raw_char - from_u) <= range_u) ? raw_char : kInvalidKey);

    const unsigned int peers = __match_any_sync(__activemask(), key);

    if (key != kInvalidKey &&
        lane_id == static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1))
    {
        atomicAdd(warp_hist + key, static_cast<unsigned int>(__popc(peers)));
    }
}

// A 32-bit packed word contains 4 characters. Extract them and feed each byte to the
// warp-aggregated update path.
__device__ __forceinline__
void process_packed_word(unsigned int packed_word,
                         unsigned int from_u,
                         unsigned int range_u,
                         unsigned int* warp_hist,
                         unsigned int lane_id)
{
    accumulate_raw_char_warp_aggregated((packed_word >>  0) & 0xFFu, from_u, range_u, warp_hist, lane_id);
    accumulate_raw_char_warp_aggregated((packed_word >>  8) & 0xFFu, from_u, range_u, warp_hist, lane_id);
    accumulate_raw_char_warp_aggregated((packed_word >> 16) & 0xFFu, from_u, range_u, warp_hist, lane_id);
    accumulate_raw_char_warp_aggregated((packed_word >> 24) & 0xFFu, from_u, range_u, warp_hist, lane_id);
}

// The template parameter selects between:
// - kAligned16 == true : vectorized uint4 loads (16 bytes/thread/iteration)
// - kAligned16 == false: generic scalar loads for uncommon misaligned device pointers
template <bool kAligned16>
__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from_u,
                            unsigned int to_u)
{
    // One full 256-bin histogram per warp in shared memory.
    __shared__ unsigned int warp_histograms[kWarpsPerBlock][kAlphabetBins];

    // Clear the shared histograms cooperatively. With 256 threads and 8 warps/block,
    // each thread clears exactly 8 counters.
    #pragma unroll
    for (int i = 0; i < kSharedClearsPerThread; ++i) {
        const unsigned int linear = threadIdx.x + i * kBlockThreads;
        warp_histograms[linear >> 8][linear & 0xFFu] = 0u;
    }
    __syncthreads();

    const unsigned int lane_id = threadIdx.x & (kWarpSize - 1);
    const unsigned int warp_id = threadIdx.x >> 5;
    unsigned int* const warp_hist = warp_histograms[warp_id];

    const unsigned int range_u   = to_u - from_u;
    const unsigned int bin_count = range_u + 1u;

    const unsigned char* __restrict__ input_bytes =
        reinterpret_cast<const unsigned char*>(input);

    const size_t global_thread = static_cast<size_t>(blockIdx.x) * kBlockThreads + threadIdx.x;
    const size_t grid_stride   = static_cast<size_t>(gridDim.x) * kBlockThreads;

    if (kAligned16) {
        // cudaMalloc() base pointers are heavily aligned, so this is the common fast path.
        const uint4* __restrict__ input_vec =
            reinterpret_cast<const uint4*>(input_bytes);

        const size_t chunk_count = static_cast<size_t>(inputSize) / kVectorBytes;

        for (size_t chunk = global_thread; chunk < chunk_count; chunk += grid_stride) {
            const uint4 v = input_vec[chunk];
            process_packed_word(v.x, from_u, range_u, warp_hist, lane_id);
            process_packed_word(v.y, from_u, range_u, warp_hist, lane_id);
            process_packed_word(v.z, from_u, range_u, warp_hist, lane_id);
            process_packed_word(v.w, from_u, range_u, warp_hist, lane_id);
        }

        // Handle the final 0..15 bytes not covered by the vectorized loop.
        const size_t tail_start = chunk_count * static_cast<size_t>(kVectorBytes);
        for (size_t i = tail_start + global_thread; i < static_cast<size_t>(inputSize); i += grid_stride) {
            accumulate_raw_char_warp_aggregated(static_cast<unsigned int>(input_bytes[i]),
                                                from_u, range_u, warp_hist, lane_id);
        }
    } else {
        // Generic path for device pointers that are not 16-byte aligned.
        for (size_t i = global_thread; i < static_cast<size_t>(inputSize); i += grid_stride) {
            accumulate_raw_char_warp_aggregated(static_cast<unsigned int>(input_bytes[i]),
                                                from_u, range_u, warp_hist, lane_id);
        }
    }

    __syncthreads();

    // Reduce the per-warp shared histograms into the final output range.
    // Because kBlockThreads == 256 and bin_count <= 256, each thread handles at most one bin.
    if (threadIdx.x < bin_count) {
        const unsigned int raw_char = from_u + threadIdx.x;

        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += warp_histograms[w][raw_char];
        }

        if (sum != 0u) {
            // For a single-block launch, each output bin is produced by exactly one thread,
            // so a plain store is sufficient and cheaper than a global atomic.
            if (gridDim.x == 1) {
                histogram[threadIdx.x] = sum;
            } else {
                atomicAdd(histogram + threadIdx.x, sum);
            }
        }
    }
}

inline bool is_16_byte_aligned(const void* ptr)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) &
            static_cast<std::uintptr_t>(kVectorBytes - 1)) == 0u;
}

// The kernel uses grid-stride loops, so only enough blocks to fully occupy the GPU
// (plus a small oversubscription factor) are needed. "logical_work_items" means:
// - aligned kernel: number of 16-byte chunks
// - scalar kernel : number of bytes
inline int choose_grid_size(size_t logical_work_items, int sm_count, int blocks_per_sm)
{
    if (sm_count < 1) {
        sm_count = 1;
    }
    if (blocks_per_sm < 1) {
        blocks_per_sm = 1;
    }

    int grid = static_cast<int>((logical_work_items + static_cast<size_t>(kBlockThreads) - 1) /
                                static_cast<size_t>(kBlockThreads));
    if (grid < 1) {
        grid = 1;
    }

    int max_grid = sm_count * blocks_per_sm * kGridOversubscription;
    if (max_grid < 1) {
        max_grid = 1;
    }

    return (grid < max_grid) ? grid : max_grid;
}

// Cache per host thread and current device so repeated calls do not keep re-querying
// occupancy and SM count. This is purely a launch-configuration cache; the kernel
// itself remains fully asynchronous.
inline const LaunchConfigCache& get_launch_config_for_current_device()
{
    static thread_local LaunchConfigCache cache;

    int current_device = 0;
    cudaGetDevice(&current_device);

    if (cache.device != current_device) {
        cache.device = current_device;
        cache.sm_count = 1;
        cache.blocks_per_sm_aligned = 1;
        cache.blocks_per_sm_scalar  = 1;

        cudaDeviceGetAttribute(&cache.sm_count, cudaDevAttrMultiProcessorCount, current_device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cache.blocks_per_sm_aligned,
            histogram_range_kernel<true>,
            kBlockThreads,
            0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cache.blocks_per_sm_scalar,
            histogram_range_kernel<false>,
            kBlockThreads,
            0);

        if (cache.sm_count < 1) {
            cache.sm_count = 1;
        }
        if (cache.blocks_per_sm_aligned < 1) {
            cache.blocks_per_sm_aligned = 1;
        }
        if (cache.blocks_per_sm_scalar < 1) {
            cache.blocks_per_sm_scalar = 1;
        }
    }

    return cache;
}

} // anonymous namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The function is intentionally fully asynchronous. Because the signature does not provide
    // a stream, operations are issued to the default stream. The caller is responsible for any
    // later synchronization and/or CUDA error checking.
    cudaStream_t stream = 0;

    const unsigned int from_u   = static_cast<unsigned int>(from);
    const unsigned int to_u     = static_cast<unsigned int>(to);
    const unsigned int bin_count = (to_u - from_u) + 1u;

    // The output is defined as the histogram for this call, not an accumulation across calls.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bin_count) * sizeof(unsigned int), stream);

    if (inputSize == 0u) {
        return;
    }

    const bool aligned16 = is_16_byte_aligned(input);
    const LaunchConfigCache& cfg = get_launch_config_for_current_device();

    if (aligned16) {
        const size_t chunk_count = static_cast<size_t>(inputSize) / static_cast<size_t>(kVectorBytes);
        const int grid = choose_grid_size(chunk_count, cfg.sm_count, cfg.blocks_per_sm_aligned);

        histogram_range_kernel<true><<<grid, kBlockThreads, 0, stream>>>(
            input, histogram, inputSize, from_u, to_u);
    } else {
        const int grid = choose_grid_size(static_cast<size_t>(inputSize),
                                          cfg.sm_count,
                                          cfg.blocks_per_sm_scalar);

        histogram_range_kernel<false><<<grid, kBlockThreads, 0, stream>>>(
            input, histogram, inputSize, from_u, to_u);
    }
}