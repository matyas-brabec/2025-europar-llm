#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// This implementation targets modern NVIDIA GPUs (e.g. A100/H100) and uses:
//   * per-warp shared-memory histograms to reduce contention,
//   * warp-aggregated updates via __match_any_sync so equal byte values within a warp
//     collapse to one shared-memory atomic update,
//   * 16-byte vectorized loads on the aligned bulk of the input.
//
// A 256-thread block is intentional: the requested histogram range can never exceed 256 bins,
// so one thread can own one output bin during the shared-memory init and final flush steps.
constexpr int kBlockThreads  = 256;
constexpr int kWarpSize      = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kByteValues    = 256;

// One padding slot per 32 bins reduces shared-memory bank-aliasing for byte values that differ
// by multiples of 32. For 256 bins this yields 264 slots, with a few intentional holes.
constexpr int kPaddedBins = kByteValues + (kByteValues / kWarpSize);

// The hot path reads 16 bytes at a time.
constexpr unsigned int kVectorBytes = sizeof(uint4);

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size");

__device__ __forceinline__ int padded_bin_index(int value) {
    // Insert one hole every 32 bins.
    return value + (value >> 5);
}

__device__ __forceinline__ void accumulate_filtered_byte(
    unsigned int* __restrict__ warp_hist,
    unsigned char byte_value,
    int from,
    unsigned int range,
    unsigned int active_mask,
    int lane)
{
    const int value = static_cast<int>(byte_value);

    // Unsigned interval test: true iff from <= value < from + range.
    const bool in_range = static_cast<unsigned int>(value - from) < range;

    // Only the currently active lanes participate. This is important because the final iteration
    // of the grid-stride loop can leave only a subset of warp lanes active.
    const unsigned int participating = __ballot_sync(active_mask, in_range);

    if (in_range) {
        // Group all in-range lanes that saw the same byte value.
        const unsigned int group  = __match_any_sync(participating, value);
        const int leader_lane     = __ffs(group) - 1;
        const unsigned int count  = __popc(group);

        // Only one lane per equal-value group performs the shared-memory atomic update.
        if (lane == leader_lane) {
            atomicAdd(&warp_hist[padded_bin_index(value)], count);
        }
    }
}

__device__ __forceinline__ void process_packed_u32(
    unsigned int* __restrict__ warp_hist,
    unsigned int packed,
    int from,
    unsigned int range,
    unsigned int active_mask,
    int lane)
{
    // CUDA GPUs are little-endian, so the four source bytes are extracted with shifts.
    accumulate_filtered_byte(
        warp_hist,
        static_cast<unsigned char>( packed        & 0xFFu),
        from,
        range,
        active_mask,
        lane);

    accumulate_filtered_byte(
        warp_hist,
        static_cast<unsigned char>((packed >>  8) & 0xFFu),
        from,
        range,
        active_mask,
        lane);

    accumulate_filtered_byte(
        warp_hist,
        static_cast<unsigned char>((packed >> 16) & 0xFFu),
        from,
        range,
        active_mask,
        lane);

    accumulate_filtered_byte(
        warp_hist,
        static_cast<unsigned char>((packed >> 24) & 0xFFu),
        from,
        range,
        active_mask,
        lane);
}

__global__ __launch_bounds__(kBlockThreads, 4)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // One full 256-bin histogram per warp, indexed by absolute byte value [0,255].
    // This keeps the update path simple and fast after the range test passes.
    __shared__ unsigned int warp_hist[kWarpsPerBlock][kPaddedBins];

    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    const int lane           = static_cast<int>(threadIdx.x & (kWarpSize - 1));
    const int warp_id        = static_cast<int>(threadIdx.x >> 5);

    unsigned int* const warp_bins = warp_hist[warp_id];

    // Only bins in the requested range are ever touched or read, so only those shared-memory
    // slots need to be initialized even though the per-warp histogram covers all byte values.
    if (threadIdx.x < range) {
        const int value  = from + static_cast<int>(threadIdx.x);
        const int padded = padded_bin_index(value);

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            warp_hist[w][padded] = 0u;
        }
    }

    __syncthreads();

    // Interpret the input bytes as unsigned ordinals [0,255] regardless of host-side char signedness.
    const unsigned char* const input_u8 = reinterpret_cast<const unsigned char*>(input);

    // Align the bulk path to 16-byte vector loads. The unaligned prefix is shorter than one vector
    // load, so block 0 handles it cheaply.
    const unsigned int misalignment =
        static_cast<unsigned int>(reinterpret_cast<uintptr_t>(input_u8) & (kVectorBytes - 1u));

    unsigned int prefix = (kVectorBytes - misalignment) & (kVectorBytes - 1u);
    if (prefix > inputSize) {
        prefix = inputSize;
    }

    if (blockIdx.x == 0 && threadIdx.x < prefix) {
        const unsigned int active_mask = __activemask();
        accumulate_filtered_byte(
            warp_bins,
            input_u8[threadIdx.x],
            from,
            range,
            active_mask,
            lane);
    }

    const unsigned int remaining   = inputSize - prefix;
    const unsigned int vec16_count = remaining / kVectorBytes;
    const uint4* const input16     = reinterpret_cast<const uint4*>(input_u8 + prefix);

    const unsigned int global_tid  = blockIdx.x * kBlockThreads + threadIdx.x;
    const unsigned int grid_stride = kBlockThreads * gridDim.x;

    // Main streaming loop over aligned 16-byte chunks.
    for (unsigned int idx = global_tid; idx < vec16_count; idx += grid_stride) {
        const unsigned int active_mask = __activemask();
        const uint4 v = input16[idx];

        process_packed_u32(warp_bins, v.x, from, range, active_mask, lane);
        process_packed_u32(warp_bins, v.y, from, range, active_mask, lane);
        process_packed_u32(warp_bins, v.z, from, range, active_mask, lane);
        process_packed_u32(warp_bins, v.w, from, range, active_mask, lane);
    }

    // Tail shorter than one vector load: again, block 0 handles it.
    const unsigned int tail_base  = prefix + vec16_count * kVectorBytes;
    const unsigned int tail_count = inputSize - tail_base;

    if (blockIdx.x == 0 && threadIdx.x < tail_count) {
        const unsigned int active_mask = __activemask();
        accumulate_filtered_byte(
            warp_bins,
            input_u8[tail_base + threadIdx.x],
            from,
            range,
            active_mask,
            lane);
    }

    __syncthreads();

    // Reduce the per-warp shared histograms into the caller-provided output.
    // Each block emits at most one global atomic add per requested bin.
    if (threadIdx.x < range) {
        const int value  = from + static_cast<int>(threadIdx.x);
        const int padded = padded_bin_index(value);

        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += warp_hist[w][padded];
        }

        if (sum != 0u) {
            atomicAdd(histogram + threadIdx.x, sum);
        }
    }
}

} // namespace

void run_histogram(
    const char *input,
    unsigned int *histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // The input and output pointers are device pointers allocated by cudaMalloc.
    // The function performs no host-device copies and no synchronization.
    // Because the signature has no stream parameter, it uses the default stream.
    const unsigned int range = static_cast<unsigned int>(to - from + 1);

    if (cudaMemsetAsync(
            histogram,
            0,
            static_cast<std::size_t>(range) * sizeof(unsigned int),
            0) != cudaSuccess) {
        return;
    }

    if (inputSize == 0u) {
        return;
    }

    // Launch a persistent-style grid sized to the current device's occupancy for this kernel.
    // Extra work is handled by the kernel's grid-stride loop, which avoids launching excess blocks
    // that would only add another histogram-flush pass.
    unsigned int max_blocks = 1u;
    int device = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;

    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            histogram_range_kernel,
            kBlockThreads,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        max_blocks = static_cast<unsigned int>(sm_count) *
                     static_cast<unsigned int>(active_blocks_per_sm);
    }

    if (max_blocks == 0u) {
        max_blocks = 1u;
    }

    // Approximate work in units of 16 input bytes. The scalar prefix/tail are at most 15 bytes each
    // and do not materially affect launch sizing.
    const unsigned int work_items = (inputSize + kVectorBytes - 1u) / kVectorBytes;

    unsigned int blocks = (work_items + kBlockThreads - 1u) / kBlockThreads;
    if (blocks == 0u) {
        blocks = 1u;
    }
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    histogram_range_kernel<<<blocks, kBlockThreads, 0, 0>>>(
        input,
        histogram,
        inputSize,
        from,
        to);
}