#include <cuda_runtime.h>

namespace {

// The kernel is specialized for a fixed 256-thread block size.
// On A100/H100-class GPUs this gives 8 warps per block, which is a good
// balance between occupancy, shared-memory footprint, and reduction cost.
constexpr int BLOCK_THREADS    = 256;
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = BLOCK_THREADS / WARP_SIZE;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int FULL_BYTE_RANGE  = 256;

static_assert(BLOCK_THREADS % WARP_SIZE == 0, "BLOCK_THREADS must be a multiple of warp size.");
static_assert(ITEMS_PER_THREAD == 4, "The vectorized load path assumes four bytes per thread.");
static_assert(FULL_BYTE_RANGE == 256, "This kernel is specialized for byte histograms.");

// Each warp owns a private histogram in shared memory.
// The caller passes a fixed participant mask captured at a converged point.
// That fixed mask is essential on Volta+ independent thread scheduling:
// it lets us safely use plain shared-memory increments (not shared atomics)
// because __syncwarp(participant_mask) closes each update step before the next
// byte is processed by any participating lane.
__device__ __forceinline__
void accumulate_rel_to_warp_histogram(unsigned int rel,
                                      unsigned int bins,
                                      unsigned int lane,
                                      unsigned int participant_mask,
                                      unsigned int* __restrict__ warp_hist)
{
    // Group equal values inside the participating subset of the warp.
    const unsigned int peers = __match_any_sync(participant_mask, rel);

    // Only the first lane in each equal-value group performs the update.
    // `rel < bins` is a branchless range test because `rel` was computed as
    // (byte_value - range_begin) in unsigned arithmetic.
    if (rel < bins && lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        warp_hist[rel] += static_cast<unsigned int>(__popc(peers));
    }

    // Ensure all participating lanes finish this byte's update before any of
    // them start the next one; this is what makes the plain shared-memory
    // increment above safe on modern GPUs with independent thread scheduling.
    __syncwarp(participant_mask);
}

// Shared-memory optimized histogram kernel.
// - One private 256-bin histogram per warp lives in shared memory.
// - Input is consumed in 4-byte vector loads for better global-memory efficiency.
// - __match_any_sync collapses duplicate byte values within a warp so each
//   distinct value updates the shared histogram once per sub-step.
// - At the end, the block reduces its warp-private histograms and accumulates
//   the result into the output histogram.
__global__ __launch_bounds__(BLOCK_THREADS)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int range_begin,
                            unsigned int bins)
{
    // Fixed-size shared storage is cheap here:
    // 8 warps * 256 bins * 4 bytes = 8 KiB per block.
    __shared__ unsigned int warp_histograms[WARPS_PER_BLOCK][FULL_BYTE_RANGE];

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane    = tid & (WARP_SIZE - 1u);

    unsigned int* const warp_hist = warp_histograms[warp_id];

    // Zero only the bins actually needed for this invocation.
    for (unsigned int bin = lane; bin < bins; bin += WARP_SIZE) {
        warp_hist[bin] = 0u;
    }
    __syncwarp();

    const unsigned char* const input_bytes = reinterpret_cast<const unsigned char*>(input);

    // The problem states the buffers are cudaMalloc-allocated, so the base
    // pointer is naturally aligned for efficient 4-byte vector loads.
    const uchar4* const input_vec4 = reinterpret_cast<const uchar4*>(input_bytes);

    const size_t n      = static_cast<size_t>(inputSize);
    size_t index        = (static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + tid) * ITEMS_PER_THREAD;
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK_THREADS * ITEMS_PER_THREAD;

    // Fast path: process four bytes per thread per iteration.
    for (; index + ITEMS_PER_THREAD <= n; index += stride) {
        // All lanes active in this loop body have a full 4-byte chunk, so a
        // single fixed mask is valid for all four sub-steps below.
        const unsigned int chunk_mask = __activemask();
        const uchar4 bytes = input_vec4[index >> 2];

        accumulate_rel_to_warp_histogram(static_cast<unsigned int>(bytes.x) - range_begin,
                                         bins, lane, chunk_mask, warp_hist);
        accumulate_rel_to_warp_histogram(static_cast<unsigned int>(bytes.y) - range_begin,
                                         bins, lane, chunk_mask, warp_hist);
        accumulate_rel_to_warp_histogram(static_cast<unsigned int>(bytes.z) - range_begin,
                                         bins, lane, chunk_mask, warp_hist);
        accumulate_rel_to_warp_histogram(static_cast<unsigned int>(bytes.w) - range_begin,
                                         bins, lane, chunk_mask, warp_hist);
    }

    // Tail path: at most three leftover bytes per thread.
    // A fresh mask is captured for each tail byte because the active subset can shrink.
    if (index < n) {
        const unsigned int tail_mask0 = __activemask();
        accumulate_rel_to_warp_histogram(static_cast<unsigned int>(input_bytes[index + 0]) - range_begin,
                                         bins, lane, tail_mask0, warp_hist);

        if (index + 1 < n) {
            const unsigned int tail_mask1 = __activemask();
            accumulate_rel_to_warp_histogram(static_cast<unsigned int>(input_bytes[index + 1]) - range_begin,
                                             bins, lane, tail_mask1, warp_hist);
        }

        if (index + 2 < n) {
            const unsigned int tail_mask2 = __activemask();
            accumulate_rel_to_warp_histogram(static_cast<unsigned int>(input_bytes[index + 2]) - range_begin,
                                             bins, lane, tail_mask2, warp_hist);
        }

        if (index + 3 < n) {
            const unsigned int tail_mask3 = __activemask();
            accumulate_rel_to_warp_histogram(static_cast<unsigned int>(input_bytes[index + 3]) - range_begin,
                                             bins, lane, tail_mask3, warp_hist);
        }
    }

    // Make all warp-private histograms visible to the whole block before reduction.
    __syncthreads();

    // One thread per output bin reduces the per-warp partials.
    for (unsigned int bin = tid; bin < bins; bin += BLOCK_THREADS) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int warp = 0; warp < WARPS_PER_BLOCK; ++warp) {
            sum += warp_histograms[warp][bin];
        }

        if (sum != 0u) {
            // If the grid has only one block, no global atomic is needed.
            if (gridDim.x == 1) {
                histogram[bin] = sum;
            } else {
                atomicAdd(histogram + bin, sum);
            }
        }
    }
}

} // namespace

// Host launcher.
// Assumptions from the problem statement:
// - `input` and `histogram` are device pointers allocated with cudaMalloc.
// - `histogram` has exactly (to - from + 1) elements.
// - The caller handles any needed synchronization/error checking after launch.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int range_begin = static_cast<unsigned int>(from);
    const unsigned int bins        = static_cast<unsigned int>(to - from + 1);

    // Always clear the destination because the contract is to produce the full histogram.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count <= 0) {
        sm_count = 1;
    }

    // Grid-stride loops let us cap the launch to an occupancy-saturating grid.
    int active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        histogram_range_kernel,
        BLOCK_THREADS,
        0);
    if (active_blocks_per_sm <= 0) {
        active_blocks_per_sm = 1;
    }

    const unsigned int bytes_per_block =
        static_cast<unsigned int>(BLOCK_THREADS * ITEMS_PER_THREAD);

    const unsigned int blocks_needed =
        (inputSize + bytes_per_block - 1u) / bytes_per_block;

    const unsigned int max_grid =
        static_cast<unsigned int>(sm_count) *
        static_cast<unsigned int>(active_blocks_per_sm);

    const unsigned int grid =
        (blocks_needed < max_grid) ? blocks_needed : max_grid;

    histogram_range_kernel<<<grid, BLOCK_THREADS>>>(
        input,
        histogram,
        inputSize,
        range_begin,
        bins);
}