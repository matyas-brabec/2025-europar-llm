#include <cuda_runtime.h>

// Restricted-range character histogram for device-resident text.
//
// Design choices aimed at modern data-center GPUs (A100/H100 class):
//   * 256-thread blocks so the final flush can map one thread to one possible output bin.
//   * One 256-bin shared histogram per warp (8 warp-private histograms per block).  This is only
//     8 KiB of shared memory per block, which is small on A100/H100, yet it removes cross-warp
//     shared-memory contention while reducing global-memory atomics to one add per bin per block.
//   * 16-byte vectorized input loads (uint4) from the cudaMalloc-allocated input buffer.
//   * Warp-aggregated shared-memory atomics via __match_any_sync() to collapse identical bin
//     updates seen by multiple lanes in the same warp before touching shared memory.
//
// Notes:
//   * Input bytes are interpreted as unsigned values, so characters 128..255 are handled
//     correctly even when `char` is signed on the host/device compiler.
//   * The host launcher zeros the output histogram asynchronously because the kernel only
//     accumulates counts with atomicAdd().
//   * No host/device synchronization is performed here; the caller owns synchronization.

namespace {
constexpr int kBlockThreads   = 256;
constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr int kMaxBins        = 256;
constexpr int kSharedBins     = kWarpsPerBlock * kMaxBins;
constexpr unsigned int kVecBytes = static_cast<unsigned int>(sizeof(uint4));

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of the warp size.");
static_assert(kBlockThreads >= kMaxBins,
              "This kernel relies on one thread being available for each possible output bin.");

// Overflow-safe ceil(x / y) for 32-bit unsigned integers.
__host__ __device__ constexpr unsigned int ceil_div_u32(unsigned int x, unsigned int y) {
    return x == 0u ? 0u : 1u + (x - 1u) / y;
}

// Warp-aggregated increment into a warp-private shared histogram.
// Only the leader of each group of lanes targeting the same bin issues the shared-memory atomic,
// with the increment equal to the number of matching lanes.
__device__ __forceinline__ void warp_aggregated_shared_add(
    unsigned int* hist,
    unsigned int rel_bin,
    bool valid,
    unsigned int lane,
    unsigned int active_mask)
{
    const unsigned int valid_mask = __ballot_sync(active_mask, valid);
    if (valid) {
        const unsigned int peers = __match_any_sync(valid_mask, rel_bin);
        if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
            atomicAdd(hist + rel_bin, __popc(peers));
        }
    }
}

// Consume 4 packed bytes from a 32-bit word loaded from the input stream.
__device__ __forceinline__ void process_packed_u32(
    unsigned int word,
    unsigned int from_u,
    unsigned int range,
    unsigned int* hist,
    unsigned int lane,
    unsigned int active_mask)
{
    unsigned int rel = ( word        & 0xFFu) - from_u;
    warp_aggregated_shared_add(hist, rel, rel < range, lane, active_mask);

    rel = ((word >>  8) & 0xFFu) - from_u;
    warp_aggregated_shared_add(hist, rel, rel < range, lane, active_mask);

    rel = ((word >> 16) & 0xFFu) - from_u;
    warp_aggregated_shared_add(hist, rel, rel < range, lane, active_mask);

    rel = ((word >> 24) & 0xFFu) - from_u;
    warp_aggregated_shared_add(hist, rel, rel < range, lane, active_mask);
}
} // namespace

__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from_u,
    unsigned int range)
{
    // Each warp owns a full 256-bin shared histogram.  The range is at most 256, so this fixed
    // layout gives constant-stride addressing while staying very small (8 KiB/block total).
    __shared__ unsigned int shared_hists[kSharedBins];

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;

    unsigned int* const my_hist = shared_hists + warp * kMaxBins;

    // Only the bins that can be touched by this launch (0 .. range-1 in the relative histogram)
    // need to be cleared.  __syncwarp() is required here because different lanes zero different
    // bins, and independent thread scheduling could otherwise let some lanes start updating bins
    // before other lanes finished clearing them.
    for (unsigned int bin = lane; bin < range; bin += kWarpSize) {
        my_hist[bin] = 0u;
    }
    __syncwarp();

    // Interpret input as unsigned bytes so ordinals 128..255 are handled correctly regardless of
    // `char` signedness.  The prompt states that the input buffer is cudaMalloc-allocated, which
    // makes 16-byte vector loads a good fit on modern GPUs.
    const unsigned char* const input_u8  = reinterpret_cast<const unsigned char*>(input);
    const uint4* const input_vec4        = reinterpret_cast<const uint4*>(input_u8);

    const unsigned int num_vec     = inputSize / kVecBytes;
    const unsigned int tail_base   = num_vec * kVecBytes;
    const unsigned int global_tid  = blockIdx.x * kBlockThreads + tid;
    const unsigned int grid_stride = gridDim.x * kBlockThreads;

    // Main vectorized grid-stride loop over 16-byte chunks.
    for (unsigned int vec_idx = global_tid; vec_idx < num_vec; vec_idx += grid_stride) {
        // The loop can be partially active in the last iteration, so use the current active mask
        // for warp-synchronous aggregation.
        const unsigned int active_mask = __activemask();
        const uint4 v = input_vec4[vec_idx];

        process_packed_u32(v.x, from_u, range, my_hist, lane, active_mask);
        process_packed_u32(v.y, from_u, range, my_hist, lane, active_mask);
        process_packed_u32(v.z, from_u, range, my_hist, lane, active_mask);
        process_packed_u32(v.w, from_u, range, my_hist, lane, active_mask);
    }

    // Handle the <= 15-byte tail once, using block 0 only, so the hot loop stays fully vectorized.
    if (blockIdx.x == 0u) {
        const unsigned int tail_count = inputSize - tail_base;
        if (tid < tail_count) {
            const unsigned int rel =
                static_cast<unsigned int>(input_u8[tail_base + tid]) - from_u;
            if (rel < range) {
                atomicAdd(my_hist + rel, 1u);
            }
        }
    }

    // Block-wide barrier before reading other warps' privatized histograms.
    __syncthreads();

    // Final flush: because range <= 256 and blockDim.x == 256, thread tid can own output bin tid.
    // Each block contributes at most one global atomic add per output bin.
    if (tid < range) {
        unsigned int block_total = 0u;
        const unsigned int* hist_ptr = shared_hists + tid;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            block_total += *hist_ptr;
            hist_ptr += kMaxBins;
        }

        if (block_total != 0u) {
            atomicAdd(histogram + tid, block_total);
        }
    }
}

void run_histogram(
    const char *input,
    unsigned int *histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // The problem statement guarantees valid inputs, but keep minimal defensive checks.
    if (histogram == nullptr || from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range  = static_cast<unsigned int>(to - from + 1);

    // The kernel accumulates with atomicAdd(), so start by clearing the output histogram.
    // This is intentionally asynchronous; the caller owns synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), 0);

    if (input == nullptr || inputSize == 0u) {
        return;
    }

    // The kernel uses a grid-stride loop, so an occupancy-sized grid is enough to fully cover
    // arbitrarily large inputs.  Launching many more blocks would mainly increase the number of
    // final global atomic flushes (one per bin per block) without increasing steady-state SM use.
    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    int active_blocks_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        histogram_range_kernel,
        kBlockThreads,
        0);
    if (active_blocks_per_sm < 1) {
        active_blocks_per_sm = 1;
    }

    const unsigned int work_items         = ceil_div_u32(inputSize, kVecBytes); // 16-byte chunks
    const unsigned int blocks_needed      = ceil_div_u32(work_items, static_cast<unsigned int>(kBlockThreads));
    const unsigned int max_occupancy_grid =
        static_cast<unsigned int>(sm_count) * static_cast<unsigned int>(active_blocks_per_sm);

    unsigned int grid = blocks_needed;
    if (grid == 0u) {
        grid = 1u;
    }
    if (grid > max_occupancy_grid) {
        grid = max_occupancy_grid;
    }
    if (grid == 0u) {
        grid = 1u;
    }

    histogram_range_kernel<<<grid, kBlockThreads>>>(
        input,
        histogram,
        inputSize,
        from_u,
        range);
}