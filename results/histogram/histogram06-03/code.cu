#include <cuda_runtime.h>

// Range-restricted byte histogram optimized for modern NVIDIA data-center GPUs
// (A100/H100 class: 32-lane warps, 32-bank shared memory).
//
// Performance strategy:
// - Each thread block builds a private histogram in shared memory, then merges once to global
//   memory. This drastically reduces global atomic pressure versus directly atomically updating
//   the output histogram for every input byte.
// - To avoid shared-memory bank conflicts, the block-private histogram is replicated 32 times:
//   one logical copy per warp lane. The layout is transposed as [bin][lane], i.e.
//       shared[(bin << 5) + lane]
//   rather than [lane][bin]. For 4-byte counters on 32-bank shared memory, bank = index % 32,
//   so lane l always maps to bank l. This makes same-instruction accesses from a warp
//   bank-conflict-free, independent of the histogram range size.
// - Threads from different warps but with the same lane ID share the same copy. Correctness
//   across warps is provided by shared-memory atomicAdd; the 32-copy scheme specifically removes
//   intra-warp bank conflicts as requested.
// - Only the requested [from, to] sub-histogram is privatized in shared memory, not a full
//   256-bin histogram. Narrower ranges therefore use less shared memory and can run at higher
//   occupancy.
// - The final merge uses warp-level reduction: each lane loads one of the 32 copies for a bin,
//   the warp sums the 32 values with shuffle-down, and lane 0 performs one global atomicAdd.
//
// Tuning notes:
// - itemsPerThread = 16 is a good default for large text buffers on recent GPUs. It enables
//   the aligned 128-bit load fast path (thanks to cudaMalloc alignment) while keeping register
//   pressure modest.
// - kBlockSize = 256 (8 warps/block) is a good compromise between latency hiding and contention
//   on the 32 lane-owned shared histogram copies.
// - The host launcher caps the effective launch at 4 resident blocks/SM. That is usually enough
//   to saturate A100/H100-class GPUs for this kernel while avoiding unnecessary block-level
//   partial histograms that would increase the final global merge traffic.

constexpr int itemsPerThread      = 16;
constexpr int kBlockSize          = 256;
constexpr int kHistogramCopies    = 32;
constexpr int kHistogramCopiesLog2 = 5;   // 32 == 1 << 5
constexpr int kTargetBlocksPerSM  = 4;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(kHistogramCopies == 32, "This implementation intentionally uses exactly 32 shared histogram copies.");
static_assert((1 << kHistogramCopiesLog2) == kHistogramCopies, "kHistogramCopiesLog2 must match kHistogramCopies.");
static_assert(kBlockSize % kHistogramCopies == 0, "kBlockSize must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum_u32(unsigned int value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

// Update one byte into the lane-owned shared histogram copy.
// The range test uses unsigned arithmetic so that
//     (c >= from && c < from + rangeSize)
// becomes one subtraction plus one compare.
__device__ __forceinline__ void update_shared_hist(unsigned int c,
                                                   unsigned int from,
                                                   unsigned int rangeSize,
                                                   unsigned int lane,
                                                   unsigned int* __restrict__ sharedHist)
{
    const unsigned int bin = c - from;
    if (bin < rangeSize) {
        atomicAdd(&sharedHist[(bin << kHistogramCopiesLog2) + lane], 1u);
    }
}

// Count four bytes loaded as one packed 32-bit word. Histogramming is order-independent,
// so we only need the four byte values, not their original order in memory.
__device__ __forceinline__ void process_packed4(unsigned int packed,
                                                unsigned int from,
                                                unsigned int rangeSize,
                                                unsigned int lane,
                                                unsigned int* __restrict__ sharedHist)
{
    update_shared_hist( packed        & 0xFFu, from, rangeSize, lane, sharedHist);
    update_shared_hist((packed >>  8) & 0xFFu, from, rangeSize, lane, sharedHist);
    update_shared_hist((packed >> 16) & 0xFFu, from, rangeSize, lane, sharedHist);
    update_shared_hist((packed >> 24) & 0xFFu, from, rangeSize, lane, sharedHist);
}

// Fast path for a full thread chunk.
// - If ItemsPerThread is a multiple of 16 (the default), use aligned 128-bit loads.
// - Otherwise, if it is a multiple of 4, use aligned 32-bit loads.
// - Otherwise fall back to scalar loads.
// The cudaMalloc alignment guarantee provided by the problem statement makes the vectorized
// paths valid for the default configuration.
template <int ItemsPerThread>
__device__ __forceinline__ void process_full_chunk(const unsigned char* __restrict__ ptr,
                                                   unsigned int from,
                                                   unsigned int rangeSize,
                                                   unsigned int lane,
                                                   unsigned int* __restrict__ sharedHist)
{
    if (ItemsPerThread % 16 == 0) {
        const uint4* vec = reinterpret_cast<const uint4*>(ptr);
        #pragma unroll
        for (int i = 0; i < ItemsPerThread / 16; ++i) {
            const uint4 v = vec[i];
            process_packed4(v.x, from, rangeSize, lane, sharedHist);
            process_packed4(v.y, from, rangeSize, lane, sharedHist);
            process_packed4(v.z, from, rangeSize, lane, sharedHist);
            process_packed4(v.w, from, rangeSize, lane, sharedHist);
        }
    } else if (ItemsPerThread % 4 == 0) {
        const unsigned int* words = reinterpret_cast<const unsigned int*>(ptr);
        #pragma unroll
        for (int i = 0; i < ItemsPerThread / 4; ++i) {
            process_packed4(words[i], from, rangeSize, lane, sharedHist);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            update_shared_hist(ptr[i], from, rangeSize, lane, sharedHist);
        }
    }
}

// Tail path for the final partial chunk of the input.
template <int ItemsPerThread>
__device__ __forceinline__ void process_tail_chunk(const unsigned char* __restrict__ ptr,
                                                   unsigned int remaining,
                                                   unsigned int from,
                                                   unsigned int rangeSize,
                                                   unsigned int lane,
                                                   unsigned int* __restrict__ sharedHist)
{
    #pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        if (static_cast<unsigned int>(i) < remaining) {
            update_shared_hist(ptr[i], from, rangeSize, lane, sharedHist);
        }
    }
}

// histogram[bin] counts byte value (from + bin).
// We pass (from, rangeSize) instead of (from, to) because it makes the range filter cheaper.
template <int ItemsPerThread>
__global__ __launch_bounds__(kBlockSize, kTargetBlocksPerSM)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int rangeSize)
{
    extern __shared__ unsigned int sharedHist[];

    // Treat the input as unsigned bytes so values in [128, 255] are handled correctly even
    // if host-side `char` is signed.
    const unsigned char* __restrict__ inputBytes =
        reinterpret_cast<const unsigned char*>(input);

    // Shared storage holds exactly rangeSize * 32 counters.
    const unsigned int sharedEntries = rangeSize << kHistogramCopiesLog2;
    for (unsigned int i = threadIdx.x; i < sharedEntries; i += blockDim.x) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & (kHistogramCopies - 1);

    // Use 64-bit indices for the grid-stride loop to avoid wraparound on very large inputs.
    const size_t n = static_cast<size_t>(inputSize);
    const size_t blockTileSize =
        static_cast<size_t>(blockDim.x) * static_cast<size_t>(ItemsPerThread);
    const size_t gridTileStride =
        static_cast<size_t>(gridDim.x) * blockTileSize;

    // Each block processes a grid-stride sequence of contiguous tiles.
    // Within a tile, each thread processes a contiguous chunk of ItemsPerThread bytes.
    for (size_t tileBase = static_cast<size_t>(blockIdx.x) * blockTileSize;
         tileBase < n;
         tileBase += gridTileStride)
    {
        const size_t threadBase =
            tileBase + static_cast<size_t>(threadIdx.x) * static_cast<size_t>(ItemsPerThread);

        if (threadBase >= n) {
            continue;
        }

        const unsigned int remaining = static_cast<unsigned int>(n - threadBase);
        const unsigned char* __restrict__ threadPtr = inputBytes + threadBase;

        if (remaining >= static_cast<unsigned int>(ItemsPerThread)) {
            process_full_chunk<ItemsPerThread>(threadPtr, from, rangeSize, lane, sharedHist);
        } else {
            process_tail_chunk<ItemsPerThread>(threadPtr, remaining, from, rangeSize, lane, sharedHist);
        }
    }

    __syncthreads();

    // Final merge:
    // one warp reduces the 32 lane-owned copies for each bin. This is bank-conflict-free too,
    // because lane l reads shared[(bin << 5) + l], i.e. always its own bank.
    const unsigned int warpId        = threadIdx.x >> kHistogramCopiesLog2;
    const unsigned int warpsPerBlock = blockDim.x >> kHistogramCopiesLog2;

    for (unsigned int bin = warpId; bin < rangeSize; bin += warpsPerBlock) {
        unsigned int sum = sharedHist[(bin << kHistogramCopiesLog2) + lane];
        sum = warp_reduce_sum_u32(sum);
        if (lane == 0u && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

static inline unsigned int ceil_div_u32(unsigned int num, unsigned int den)
{
    return num / den + ((num % den) != 0u);
}

// Enqueues the histogram computation on the default stream.
// No host-device synchronization is performed here; the caller requested to manage that.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int from_u    = static_cast<unsigned int>(from);
    const unsigned int rangeSize = static_cast<unsigned int>(to - from + 1);

    const size_t histogramBytes =
        static_cast<size_t>(rangeSize) * sizeof(unsigned int);

    // 32 shared-memory copies of the requested sub-histogram.
    const size_t sharedBytes =
        (static_cast<size_t>(rangeSize) << kHistogramCopiesLog2) * sizeof(unsigned int);

    // The API does not provide a stream parameter, so use the default stream.
    const cudaStream_t stream = 0;

    // Blocks merge with global atomicAdd, so the output histogram must start from zero.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    // Empty input: the asynchronous memset above already produced the correct all-zero result.
    if (inputSize == 0u) {
        return;
    }

    int device  = 0;
    int smCount = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSM = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<itemsPerThread>,
        kBlockSize,
        sharedBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    // Histogram kernels pay a final global merge cost per block. Launch enough blocks to keep
    // the machine busy, but not so many that we create unnecessary extra partial histograms.
    int usefulBlocksPerSM = activeBlocksPerSM;
    if (usefulBlocksPerSM > kTargetBlocksPerSM) {
        usefulBlocksPerSM = kTargetBlocksPerSM;
    }

    const unsigned int tileSize     = static_cast<unsigned int>(kBlockSize * itemsPerThread);
    const unsigned int neededBlocks = ceil_div_u32(inputSize, tileSize);
    const int maxUsefulBlocks       = smCount * usefulBlocksPerSM;

    int gridSize = static_cast<int>(neededBlocks);
    if (gridSize > maxUsefulBlocks) {
        gridSize = maxUsefulBlocks;
    }
    if (gridSize < 1) {
        gridSize = 1;
    }

    histogram_range_kernel<itemsPerThread>
        <<<gridSize, kBlockSize, sharedBytes, stream>>>(
            input,
            histogram,
            inputSize,
            from_u,
            rangeSize);

    // Intentionally no synchronization here.
}