#include <cuda_runtime.h>

namespace {

// Optimized for modern data-center GPUs (A100/H100 class).
// Requirements/assumptions from the prompt:
//   * input and histogram are already device pointers allocated with cudaMalloc.
//   * histogram has exactly (to - from + 1) bins.
//   * caller handles any required synchronization.
// Design:
//   * The requested range is at most 256 bins, so a 256-thread block can map
//     one thread to one final output bin during the block-local reduction.
//   * Each warp owns a private histogram in shared memory. With 8 warps/block
//     and at most 256 bins, this costs only 8 * 256 * 4 B = 8 KiB/block.
//   * The hot path uses __match_any_sync to aggregate identical bins inside a
//     warp so that duplicate characters generate a single shared-memory update.
//   * Final block partials are reduced and accumulated into the global output.
//   * This uses Volta+ warp-match intrinsics, which is appropriate for A100/H100.
constexpr int kHistogramBlockSize = 256;
constexpr int kWarpSize           = 32;
constexpr int kWarpsPerBlock      = kHistogramBlockSize / kWarpSize;
constexpr int kItemsPerThread     = 4;

static_assert(kHistogramBlockSize % kWarpSize == 0,
              "Block size must be a whole number of warps.");

// Branch-free select: pred ? when_true : when_false.
// These helpers intentionally avoid control-flow branches before warp-synchronous
// intrinsics so that all currently active lanes reach them together.
__device__ __forceinline__ unsigned int select_u32(unsigned int when_true,
                                                   unsigned int when_false,
                                                   bool pred) {
    const unsigned int mask = 0u - static_cast<unsigned int>(pred);
    return when_false ^ ((when_false ^ when_true) & mask);
}

__device__ __forceinline__ size_t select_size_t(size_t when_true,
                                                size_t when_false,
                                                bool pred) {
    const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(pred);
    return when_false ^ ((when_false ^ when_true) & mask);
}

// Update one byte into a warp-private shared histogram.
// All currently active lanes participate in __match_any_sync. Misses are given a
// unique high-bit key so they cannot collide with any real histogram bin.
__device__ __forceinline__ void accumulate_byte_to_warp_hist(
    unsigned int* __restrict__ warp_hist,
    unsigned int range_begin,
    unsigned int num_bins,
    unsigned int byte_value,
    bool valid,
    int lane,
    unsigned int exec_mask) {

    const unsigned int bin = byte_value - range_begin;
    const bool hit = valid && (bin < num_bins);

    const unsigned int invalid_key = 0x80000000u | static_cast<unsigned int>(lane);
    const unsigned int key = select_u32(bin, invalid_key, hit);

    const unsigned int peers = __match_any_sync(exec_mask, key);
    const int leader_lane = __ffs(peers) - 1;

    if (hit && lane == leader_lane) {
        warp_hist[bin] += static_cast<unsigned int>(__popc(peers));
    }

    // The leader-only update can diverge; reconverge the active lanes before
    // processing the next byte.
    __syncwarp(exec_mask);
}

__global__ __launch_bounds__(kHistogramBlockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int input_size,
                            unsigned int range_begin,
                            unsigned int num_bins) {
    extern __shared__ unsigned int shared_hist[];

    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;

    const size_t warp_stride = static_cast<size_t>(num_bins);
    unsigned int* const warp_hist =
        shared_hist + static_cast<size_t>(warp_id) * warp_stride;

    // Zero the per-warp shared histograms.
    const int total_shared_bins = kWarpsPerBlock * static_cast<int>(num_bins);
    for (int i = tid; i < total_shared_bins; i += kHistogramBlockSize) {
        shared_hist[i] = 0u;
    }
    __syncthreads();

    // Treat the input as raw bytes in [0,255]. This avoids any ambiguity from
    // whether plain 'char' is signed or unsigned on the compilation target.
    const unsigned char* const data = reinterpret_cast<const unsigned char*>(input);

    const size_t n           = static_cast<size_t>(input_size);
    const size_t last        = n - 1u;
    const size_t thread_base = static_cast<size_t>(blockIdx.x) * kHistogramBlockSize +
                               static_cast<size_t>(tid);
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * kHistogramBlockSize;
    const size_t loop_stride = grid_stride * kItemsPerThread;

    // Grid-stride loop. Each iteration processes 4 coalesced bytes per thread.
    for (size_t idx0 = thread_base; idx0 < n; idx0 += loop_stride) {
        // The set of active lanes is fixed for the four bytes handled in this
        // loop iteration, so capture it once and reuse it.
        const unsigned int exec_mask = __activemask();

        const size_t idx1 = idx0 + grid_stride;
        const size_t idx2 = idx1 + grid_stride;
        const size_t idx3 = idx2 + grid_stride;

        const bool v1 = idx1 < n;
        const bool v2 = idx2 < n;
        const bool v3 = idx3 < n;

        // Branch-free tail handling: out-of-range lanes read the last valid byte,
        // and the valid flag suppresses any contribution from those lanes.
        const unsigned int c0 = static_cast<unsigned int>(data[idx0]);
        const unsigned int c1 = static_cast<unsigned int>(
            data[select_size_t(idx1, last, v1)]);
        const unsigned int c2 = static_cast<unsigned int>(
            data[select_size_t(idx2, last, v2)]);
        const unsigned int c3 = static_cast<unsigned int>(
            data[select_size_t(idx3, last, v3)]);

        accumulate_byte_to_warp_hist(warp_hist, range_begin, num_bins, c0, true, lane, exec_mask);
        accumulate_byte_to_warp_hist(warp_hist, range_begin, num_bins, c1, v1,  lane, exec_mask);
        accumulate_byte_to_warp_hist(warp_hist, range_begin, num_bins, c2, v2,  lane, exec_mask);
        accumulate_byte_to_warp_hist(warp_hist, range_begin, num_bins, c3, v3,  lane, exec_mask);
    }

    __syncthreads();

    // num_bins <= 256, so one thread can own one final output bin.
    if (static_cast<unsigned int>(tid) < num_bins) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_hist[static_cast<size_t>(w) * warp_stride +
                               static_cast<size_t>(tid)];
        }

        // Single-block launch: write the final result directly.
        // Multi-block launch: atomically accumulate this block's partial result.
        if (gridDim.x == 1) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

} // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const int numBinsSigned = to - from + 1;
    if (numBinsSigned <= 0) {
        return;
    }

    const unsigned int rangeBegin = static_cast<unsigned int>(from);
    const unsigned int numBins    = static_cast<unsigned int>(numBinsSigned);
    const size_t histogramBytes   = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // No input: just zero the output histogram asynchronously and return.
    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogramBytes);
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(numBins) * kWarpsPerBlock * sizeof(unsigned int);

    // One block processes kHistogramBlockSize * kItemsPerThread bytes per
    // grid-stride "round". For very small inputs, one block is sufficient and
    // the kernel writes the final histogram directly, so no memset is needed.
    constexpr unsigned int kBytesPerBlockIteration =
        static_cast<unsigned int>(kHistogramBlockSize * kItemsPerThread);

    const unsigned int neededBlocks =
        1u + (inputSize - 1u) / kBytesPerBlockIteration;

    unsigned int gridSize = 1u;

    if (neededBlocks > 1u) {
        // Use an occupancy-sized resident grid. On large inputs, each resident
        // block keeps pulling work via the grid-stride loop until the whole
        // input is consumed.
        int device = 0;
        int smCount = 1;
        int activeBlocksPerSM = 1;

        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
                                                      histogram_range_kernel,
                                                      kHistogramBlockSize,
                                                      sharedBytes);

        if (smCount < 1) {
            smCount = 1;
        }
        if (activeBlocksPerSM < 1) {
            activeBlocksPerSM = 1;
        }

        const unsigned int residentBlocks =
            static_cast<unsigned int>(smCount) *
            static_cast<unsigned int>(activeBlocksPerSM);

        gridSize = (neededBlocks < residentBlocks) ? neededBlocks : residentBlocks;
        if (gridSize == 0u) {
            gridSize = 1u;
        }
    }

    // Multi-block launches accumulate partials with atomicAdd, so the output
    // must start at zero. Single-block launches write every bin directly.
    if (gridSize > 1u) {
        cudaMemsetAsync(histogram, 0, histogramBytes);
    }

    // input and histogram are device pointers per the API contract; this host
    // wrapper only schedules asynchronous device work and does not synchronize.
    histogram_range_kernel<<<gridSize, kHistogramBlockSize, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        rangeBegin,
        numBins);
}