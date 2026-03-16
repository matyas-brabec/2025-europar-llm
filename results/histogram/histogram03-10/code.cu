#include <cuda_runtime.h>
#include <cstdint>

// Optimized range-restricted byte histogram for modern NVIDIA GPUs.
// Design choices:
//  - 32-bit vectorized input loads. The prompt guarantees cudaMalloc-allocated buffers,
//    so the input base pointer is naturally aligned for 32-bit accesses.
//  - Per-warp privatized histograms in shared memory. This dramatically reduces the
//    number of global atomics: each block emits at most one atomicAdd per non-zero bin.
//  - Warp-aggregated updates using __match_any_sync. When multiple lanes in a warp see
//    the same byte/bin, a single leader lane performs one shared-memory atomicAdd with
//    the population count of that group. This is especially effective on skewed text.
//  - Shared-memory padding: one extra slot is inserted every 32 logical bins so bins
//    separated by multiples of 32 do not systematically alias the same bank.
//  - A 256-thread block is a good fit here because the maximum histogram width is 256;
//    in the merge phase, a full-range histogram maps naturally to one thread per bin.

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
constexpr unsigned int kInvalidTag = 0xFFFFFFFFu;  // Cannot collide with any valid bin (< 256).
static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");

// Map a logical bin index to a padded physical shared-memory index.
// Physical layout inserts one padding element after every 32 logical bins:
//   phys = bin + floor(bin / 32)
// Example for bins 0..63:
//   logical 0..31  -> phys 0..31
//   logical 32..63 -> phys 33..64  (phys 32 is padding)
__device__ __forceinline__ unsigned int padded_bin_index(unsigned int bin) {
    return bin + (bin >> 5);
}

// Warp-aggregated shared-memory histogram update.
// Every currently active lane participates in __match_any_sync using the same active mask.
// Invalid lanes contribute a sentinel tag and are ignored afterwards.
// For each distinct valid bin value present in the warp, only one leader lane performs
// the atomicAdd, with the increment equal to the number of matching lanes.
__device__ __forceinline__ void update_warp_histogram(
    unsigned int active_mask,
    bool valid,
    unsigned int bin,
    unsigned int* __restrict__ warp_hist,
    unsigned int lane_id) {
    const unsigned int tag = valid ? bin : kInvalidTag;
    const unsigned int group = __match_any_sync(active_mask, tag);
    const unsigned int leader = static_cast<unsigned int>(__ffs(group) - 1);

    if (valid && lane_id == leader) {
        atomicAdd(&warp_hist[padded_bin_index(bin)], __popc(group));
    }
}

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from,
    unsigned int bins,
    unsigned int paddedBins) {
    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpSize;

    extern __shared__ unsigned int shared_histograms[];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & (kWarpSize - 1);
    const unsigned int warp_id = tid >> 5;
    unsigned int* const warp_hist = shared_histograms + warp_id * paddedBins;

    // Zero all warp-private shared histograms cooperatively.
    const unsigned int shared_elems =
        static_cast<unsigned int>(kWarpsPerBlock) * paddedBins;
    for (unsigned int i = tid; i < shared_elems; i += BLOCK_THREADS) {
        shared_histograms[i] = 0u;
    }
    __syncthreads();

    const unsigned int global_tid = blockIdx.x * BLOCK_THREADS + tid;
    const unsigned int total_threads = gridDim.x * BLOCK_THREADS;

    const unsigned char* const input_u8 =
        reinterpret_cast<const unsigned char*>(input);
    const uint32_t* const input_u32 =
        reinterpret_cast<const uint32_t*>(input_u8);

    const unsigned int wordCount = inputSize >> 2;   // Number of full 4-byte words.
    const unsigned int tailStart = wordCount << 2;   // First leftover byte.
    const unsigned int tailCount = inputSize - tailStart;  // 0..3

    // Main path: process 4 bytes per iteration with a grid-stride loop.
    // Byte order inside the 32-bit word is irrelevant because histogram accumulation
    // is commutative; we simply unpack all four bytes and count them.
    for (unsigned int wordIdx = global_tid; wordIdx < wordCount; wordIdx += total_threads) {
        const uint32_t packed = input_u32[wordIdx];
        const unsigned int active_mask = __activemask();

        uint32_t v = packed;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int byte = v & 0xFFu;
            const unsigned int bin = byte - from;  // Underflow is fine; the range check is bin < bins.
            update_warp_histogram(active_mask, bin < bins, bin, warp_hist, lane_id);
            v >>= 8;
        }
    }

    // Tail path: at most 3 bytes remain. Let only the first warp of the first block
    // process them; this avoids extra kernel launches and avoids overflow-prone
    // tailStart + global_tid arithmetic when inputSize is close to UINT_MAX.
    if (blockIdx.x == 0u && warp_id == 0u && tailCount != 0u) {
        const unsigned int active_mask = __activemask();
        const bool valid_lane = lane_id < tailCount;
        const unsigned int byte = valid_lane ? static_cast<unsigned int>(input_u8[tailStart + lane_id]) : 0u;
        const unsigned int bin = byte - from;
        update_warp_histogram(active_mask, valid_lane && (bin < bins), bin, warp_hist, lane_id);
    }

    __syncthreads();

    // Merge all warp-private shared histograms in the block and emit one global atomicAdd
    // per non-zero bin for this block.
    for (unsigned int bin = tid; bin < bins; bin += BLOCK_THREADS) {
        const unsigned int phys = padded_bin_index(bin);
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_histograms[w * paddedBins + phys];
        }

        if (sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

void run_histogram(
    const char *input,
    unsigned int *histogram,
    unsigned int inputSize,
    int from,
    int to) {
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const unsigned int paddedBins = bins + ((bins - 1u) >> 5);
    const size_t histogramBytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // The caller explicitly handles synchronization, so keep the API fully async.
    cudaMemsetAsync(histogram, 0, histogramBytes);

    if (inputSize == 0u) {
        return;
    }

    constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
    const size_t sharedBytes =
        static_cast<size_t>(kWarpsPerBlock) *
        static_cast<size_t>(paddedBins) *
        sizeof(unsigned int);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<kBlockThreads>,
        kBlockThreads,
        sharedBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    // Safe ceil(inputSize / 4) without overflow.
    const unsigned int workItems =
        (inputSize >> 2) + ((inputSize & 3u) != 0u ? 1u : 0u);

    unsigned int grid =
        (workItems + static_cast<unsigned int>(kBlockThreads) - 1u) /
        static_cast<unsigned int>(kBlockThreads);
    if (grid == 0u) {
        grid = 1u;
    }

    // Saturate the GPU but avoid launching excess blocks, since each block contributes
    // a final block-to-global histogram merge.
    const unsigned int maxResidentGrid =
        static_cast<unsigned int>(smCount) *
        static_cast<unsigned int>(activeBlocksPerSM);
    if (grid > maxResidentGrid) {
        grid = maxResidentGrid;
    }

    histogram_range_kernel<kBlockThreads><<<grid, kBlockThreads, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        bins,
        paddedBins);
}