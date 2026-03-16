#include <cuda_runtime.h>

namespace histogram_detail {

// Range-restricted byte histogram optimized for modern NVIDIA GPUs.
// Hot-path design:
//   * 256-thread blocks (8 warps per block).
//   * One 256-bin shared-memory histogram slice per warp.
//   * __match_any_sync() aggregates identical byte values inside a warp so only
//     one lane issues the shared-memory atomicAdd for each equal-value group.
//   * uchar4 vectorized loads reduce loop/control overhead and preserve coalescing.
// After processing its grid-stride share of the input, each block emits at most
// one global update per requested output bin.

constexpr int kHistogramBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kHistogramBlockSize / kWarpSize;
constexpr int kByteValueCount = 256;
constexpr int kSharedHistogramSize = kWarpsPerBlock * kByteValueCount;

constexpr unsigned int kVectorBytes = 4u;  // uchar4
constexpr unsigned int kGridHeuristicBytesPerBlock =
    static_cast<unsigned int>(kHistogramBlockSize) * kVectorBytes;

static_assert(kHistogramBlockSize % kWarpSize == 0,
              "Histogram block size must be a multiple of warp size.");

inline unsigned int div_up_u32(unsigned int n, unsigned int d) {
    return n / d + static_cast<unsigned int>(n % d != 0u);
}

__device__ __forceinline__
void accumulate_byte(unsigned int* __restrict__ warp_hist,
                     unsigned char value,
                     unsigned int from,
                     unsigned int range,
                     unsigned int active_mask,
                     unsigned int lane) {
    // Convert the byte value to the output histogram index relative to 'from'.
    // Unsigned arithmetic makes values below 'from' wrap above 'range', so a
    // single "bin < range" check filters both sides of the interval.
    const unsigned int bin = static_cast<unsigned int>(value) - from;

    // Group lanes that saw the same bin value. Only the first lane in each
    // equal-value group updates shared memory, using the group population as
    // the increment. This substantially reduces shared-memory atomic pressure
    // on skewed text distributions.
    const unsigned int peers = __match_any_sync(active_mask, bin);
    const unsigned int leader = static_cast<unsigned int>(__ffs(peers) - 1);

    if (bin < range && lane == leader) {
        atomicAdd(warp_hist + bin, static_cast<unsigned int>(__popc(peers)));
    }
}

__global__ __launch_bounds__(kHistogramBlockSize)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int range) {
    // Fixed 256-bin stride per warp keeps shared-memory addressing simple and fast.
    // Only bins [0, range) are touched and later reduced into the output.
    __shared__ unsigned int s_warp_hist[kSharedHistogramSize];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & (kWarpSize - 1u);

    // Zero the full 8 KiB shared footprint. This is cheap and avoids any range-
    // dependent addressing in the hot path.
    #pragma unroll
    for (unsigned int i = tid; i < static_cast<unsigned int>(kSharedHistogramSize); i += kHistogramBlockSize) {
        s_warp_hist[i] = 0u;
    }
    __syncthreads();

    unsigned int* const my_hist = s_warp_hist + warp_id * kByteValueCount;
    const unsigned int global_thread = blockIdx.x * kHistogramBlockSize + tid;
    const unsigned int global_stride = gridDim.x * kHistogramBlockSize;

    // The problem states that the input array itself is allocated by cudaMalloc,
    // so the base pointer is naturally aligned for uchar4 vector loads.
    const unsigned int vec_count = inputSize / kVectorBytes;
    const uchar4* const input4 = reinterpret_cast<const uchar4*>(input);

    for (unsigned int i = global_thread; i < vec_count; i += global_stride) {
        const uchar4 v = input4[i];
        const unsigned int mask = __activemask();

        accumulate_byte(my_hist, v.x, from, range, mask, lane);
        accumulate_byte(my_hist, v.y, from, range, mask, lane);
        accumulate_byte(my_hist, v.z, from, range, mask, lane);
        accumulate_byte(my_hist, v.w, from, range, mask, lane);
    }

    // At most 3 bytes remain after the uchar4 sweep.
    const unsigned int tail_base = vec_count * kVectorBytes;
    const unsigned int tail_count = inputSize - tail_base;
    if (global_thread < tail_count) {
        const unsigned int mask = __activemask();
        accumulate_byte(my_hist, input[tail_base + global_thread], from, range, mask, lane);
    }

    __syncthreads();

    // Reduce the warp-private histograms to the final global histogram.
    // Output position i stores the count for byte value (from + i).
    for (unsigned int bin = tid; bin < range; bin += kHistogramBlockSize) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += s_warp_hist[w * kByteValueCount + bin];
        }

        if (sum != 0u) {
            if (gridDim.x == 1) {
                histogram[bin] = sum;
            } else {
                atomicAdd(histogram + bin, sum);
            }
        }
    }
}

} // namespace histogram_detail

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // histogram[i] counts occurrences of byte value (from + i).
    // The API provides no stream parameter, so work is enqueued on the default stream.
    // Synchronization and post-launch error handling are intentionally left to the caller.
    cudaStream_t stream = 0;

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range = static_cast<unsigned int>(to - from) + 1u;

    // The kernel accumulates into global memory, so clear only the requested output range.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), stream);

    if (inputSize == 0u) {
        return;
    }

    int grid = static_cast<int>(
        histogram_detail::div_up_u32(inputSize, histogram_detail::kGridHeuristicBytesPerBlock));

    // The kernel uses a grid-stride loop, so launching more blocks than can be
    // resident at once only increases the number of final global flushes.
    if (grid > 1) {
        int device = 0;
        int sm_count = 1;
        int blocks_per_sm = 1;

        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            histogram_detail::histogram_range_kernel,
            histogram_detail::kHistogramBlockSize,
            0);

        const int max_grid = sm_count * (blocks_per_sm > 0 ? blocks_per_sm : 1);
        if (grid > max_grid) {
            grid = max_grid;
        }
    }

    histogram_detail::histogram_range_kernel<<<grid, histogram_detail::kHistogramBlockSize, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from_u,
        range);
}