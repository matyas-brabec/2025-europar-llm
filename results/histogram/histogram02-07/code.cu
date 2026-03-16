#include <cuda_runtime.h>
#include <cstddef>

/*
    Restricted-range byte histogram for a device-resident text buffer.

    histogram[i] counts occurrences of byte value (from + i), for i in [0, to - from].

    Optimization strategy:
      - Treat the input as unsigned bytes so ordinals 128..255 are counted correctly even if
        the C++ 'char' type is signed.
      - Each block owns one private histogram per warp in shared memory. This removes cross-warp
        contention inside the block.
      - Each warp-private histogram is padded every 32 logical bins, which breaks the shared-memory
        bank-aliasing pattern for bins that differ by 32 (a common case for text, e.g. 'A' vs 'a').
      - Within each warp, duplicate updates to the same bin are merged with __match_any_sync():
        only one lane performs the shared-memory atomicAdd(), adding the population count of the
        whole peer group.
      - Blocks traverse the input with a grid-stride loop and flush only one partial histogram
        per block to global memory.

    The host launcher is intentionally asynchronous and uses the current default stream because
    the requested API does not take an explicit stream parameter.
*/

namespace {

constexpr int kWarpSize             = 32;
constexpr int kBlockSize            = 256;
constexpr int kItemsPerThread       = 8;
constexpr int kWarpsPerBlock        = kBlockSize / kWarpSize;
constexpr int kTargetBlocksPerSM    = 8;   // 256 threads/block * 8 blocks/SM = 64 warps/SM.
constexpr int kBankGroupLog2        = 5;   // 32 shared-memory banks.

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a multiple of 32.");

__host__ __device__ __forceinline__
unsigned int warp_histogram_stride(unsigned int bins) {
    // One padding element after each complete group of 32 logical bins.
    // Equivalent to: bins + floor((bins - 1) / 32), with bins >= 1.
    return bins + ((bins - 1u) >> kBankGroupLog2);
}

__device__ __forceinline__
unsigned int padded_bin_index(unsigned int local_bin) {
    return local_bin + (local_bin >> kBankGroupLog2);
}

__device__ __forceinline__
void accumulate_byte(unsigned int* warp_hist,
                     unsigned char byte,
                     unsigned int from_u,
                     unsigned int bins,
                     int lane) {
    const unsigned int local_bin = static_cast<unsigned int>(byte) - from_u;

#if __CUDA_ARCH__ >= 700
    const bool valid = (local_bin < bins);

    // Use the current execution mask because this helper is also called from the tail path,
    // where only a subset of lanes may still be in bounds.
    const unsigned int valid_mask = __ballot_sync(__activemask(), valid);
    if (valid_mask == 0u) {
        return;
    }

    // Collapse identical updates inside the warp. Only the leader of each peer group performs
    // the shared-memory atomic, adding the number of lanes in that group.
    if (valid) {
        const unsigned int peers = __match_any_sync(valid_mask, local_bin);
        if (lane == (__ffs(peers) - 1)) {
            atomicAdd(&warp_hist[padded_bin_index(local_bin)],
                      static_cast<unsigned int>(__popc(peers)));
        }
    }
#else
    // Fallback for older architectures. The target hardware in this task is modern (A100/H100
    // class), so the fast path above is the intended one.
    if (local_bin < bins) {
        atomicAdd(&warp_hist[padded_bin_index(local_bin)], 1u);
    }
#endif
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE, kTargetBlocksPerSM)
void histogram_kernel(const unsigned char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      unsigned int input_size,
                      unsigned int from_u,
                      unsigned int bins) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / kWarpSize;

    extern __shared__ unsigned int shared_hist[];

    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;

    // Shared-memory layout: WARPS_PER_BLOCK private histograms, each padded to reduce bank conflicts.
    const unsigned int warp_stride           = warp_histogram_stride(bins);
    const unsigned int total_shared_counters = static_cast<unsigned int>(WARPS_PER_BLOCK) * warp_stride;

    for (unsigned int i = static_cast<unsigned int>(tid); i < total_shared_counters; i += BLOCK_SIZE) {
        shared_hist[i] = 0u;
    }
    __syncthreads();

    unsigned int* const warp_hist = shared_hist + static_cast<unsigned int>(warp_id) * warp_stride;

    const unsigned long long tile_span =
        static_cast<unsigned long long>(BLOCK_SIZE) * static_cast<unsigned long long>(ITEMS_PER_THREAD);
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * tile_span;
    const unsigned long long input_size_64 =
        static_cast<unsigned long long>(input_size);

    // Fast path: whole block-sized tiles are in bounds, so no per-load bounds checks are needed.
    unsigned long long block_base = static_cast<unsigned long long>(blockIdx.x) * tile_span;
    for (; block_base + tile_span <= input_size_64; block_base += stride) {
        const unsigned long long thread_base = block_base + static_cast<unsigned long long>(tid);

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned long long pos =
                thread_base + static_cast<unsigned long long>(item * BLOCK_SIZE);
            accumulate_byte(warp_hist, input[pos], from_u, bins, lane);
        }
    }

    // Tail path for the final partial tile, if any.
    for (; block_base < input_size_64; block_base += stride) {
        const unsigned long long thread_base = block_base + static_cast<unsigned long long>(tid);

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned long long pos =
                thread_base + static_cast<unsigned long long>(item * BLOCK_SIZE);
            if (pos < input_size_64) {
                accumulate_byte(warp_hist, input[pos], from_u, bins, lane);
            }
        }
    }

    __syncthreads();

    const bool single_block = (gridDim.x == 1);

    // Reduce the warp-private histograms into one block histogram and flush it.
    // Because bins <= 256 and BLOCK_SIZE == 256, each thread handles at most one logical bin.
    for (unsigned int bin = static_cast<unsigned int>(tid); bin < bins; bin += BLOCK_SIZE) {
        const unsigned int idx = padded_bin_index(bin);
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += shared_hist[static_cast<unsigned int>(w) * warp_stride + idx];
        }

        if (single_block) {
            // Single-block launch: write final results directly, so no prior memset is needed.
            histogram[bin] = sum;
        } else if (sum != 0u) {
            // Multi-block launch: accumulate block partials into the global histogram.
            atomicAdd(&histogram[bin], sum);
        }
    }
}

inline int current_sm_count() {
    // Per-thread cache keeps the launcher cheap when called repeatedly.
    thread_local int cached_device   = -1;
    thread_local int cached_sm_count = 0;

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (device != cached_device || cached_sm_count <= 0) {
            int sm_count = 1;
            if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
                sm_count < 1) {
                sm_count = 1;
            }
            cached_device   = device;
            cached_sm_count = sm_count;
        }
    }

    return (cached_sm_count > 0) ? cached_sm_count : 1;
}

} // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The contract states that:
    //   - input and histogram are device pointers allocated with cudaMalloc
    //   - 0 <= from < to <= 255
    //   - histogram has exactly (to - from + 1) entries
    //
    // The operation is inclusive on both ends of the range.
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const size_t histogram_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // Stream 0 means "current default stream" and therefore follows the application's chosen
    // default-stream semantics (legacy default stream or per-thread default stream).
    const cudaStream_t stream = 0;

    if (inputSize == 0u) {
        // Empty input: the correct histogram is all zeros.
        cudaMemsetAsync(histogram, 0, histogram_bytes, stream);
        return;
    }

    constexpr unsigned int kBlockItems =
        static_cast<unsigned int>(kBlockSize * kItemsPerThread);

    const int max_blocks_by_input = static_cast<int>(
        (static_cast<unsigned long long>(inputSize) +
         static_cast<unsigned long long>(kBlockItems) - 1ull) /
        static_cast<unsigned long long>(kBlockItems));

    int blocks = max_blocks_by_input;

    // Cap the grid to a GPU-sized working set. On A100/H100-class GPUs, 8 resident 256-thread
    // blocks per SM reaches the 64-warp occupancy limit while keeping the number of block partials low.
    if (blocks > kTargetBlocksPerSM) {
        const int max_blocks_by_gpu = current_sm_count() * kTargetBlocksPerSM;
        if (blocks > max_blocks_by_gpu) {
            blocks = max_blocks_by_gpu;
        }
    }

    // For multi-block launches the kernel atomically accumulates block partials, so the global
    // histogram must start from zero. For the single-block case the kernel writes final values
    // directly and we can skip the memset.
    if (blocks > 1) {
        cudaMemsetAsync(histogram, 0, histogram_bytes, stream);
    }

    const size_t shared_mem_bytes =
        static_cast<size_t>(kWarpsPerBlock) *
        static_cast<size_t>(warp_histogram_stride(bins)) *
        sizeof(unsigned int);

    histogram_kernel<kBlockSize, kItemsPerThread>
        <<<blocks, kBlockSize, shared_mem_bytes, stream>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            static_cast<unsigned int>(from),
            bins);
}