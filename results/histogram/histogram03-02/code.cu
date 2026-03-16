#include <cuda_runtime.h>

namespace {

// 256 threads is a natural fit for a byte histogram:
// - 8 warps per block gives good occupancy on A100/H100-class GPUs.
// - the reduction phase can map one thread to one possible output bin because the
//   requested range size is at most 256.
constexpr int kBlockThreads      = 256;
constexpr int kWarpSize          = 32;
constexpr int kWarpsPerBlock     = kBlockThreads / kWarpSize;
constexpr int kMaxBins           = 256;          // full unsigned-char domain
constexpr int kSharedHistEntries = kWarpsPerBlock * kMaxBins;
constexpr int kSharedInitIters   = kSharedHistEntries / kBlockThreads;
constexpr int kUnroll            = 4;
constexpr unsigned int kInvalidKey          = 0xFFFFFFFFu;
constexpr unsigned int kTargetBytesPerBlock = 16u * 1024u;  // amortize per-block init/flush cost

static_assert(kBlockThreads % kWarpSize == 0,
              "Block size must be a whole number of warps.");
static_assert(kBlockThreads == kMaxBins,
              "This implementation relies on one reducer thread per possible byte value.");
static_assert(kSharedHistEntries % kBlockThreads == 0,
              "Shared-histogram init expects an integer number of stores per thread.");

// Warp-aggregated update into a warp-private shared-memory histogram.
//
// The key optimization is:
//   1) each warp owns its own 256-bin shared-memory histogram slice;
//   2) __match_any_sync() groups lanes with the same target bin;
//   3) only one leader lane per unique bin performs the increment.
//
// Because a warp's histogram slice is private to that warp, once we reduce a bin to a
// single leader lane there is no same-address inter-thread race left, so the increment
// is a plain shared-memory add instead of a shared-memory atomic.
__device__ __forceinline__
void add_byte_to_warp_hist(const unsigned int value,
                           const unsigned int from,
                           const unsigned int numBins,
                           const unsigned int lane,
                           unsigned int* __restrict__ warpHist)
{
    const unsigned int active = __activemask();

    // Unsigned subtraction gives a cheap range test:
    // - in range  => bin in [0, numBins)
    // - out of range below 'from' underflows to a very large unsigned value
    // - out of range above 'to' also yields bin >= numBins
    const unsigned int bin   = value - from;
    const bool valid         = (bin < numBins);

    // All active lanes must participate in __match_any_sync(). Lanes that do not map
    // to the requested range use a sentinel key outside the real bin domain [0, 255].
    const unsigned int key     = valid ? bin : kInvalidKey;
    const unsigned int matches = __match_any_sync(active, key);

    if (valid) {
        const unsigned int leader = static_cast<unsigned int>(__ffs(matches) - 1);
        if (lane == leader) {
            warpHist[bin] += __popc(matches);
        }
    }
}

__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int numBins)
{
    // One private 256-bin histogram per warp.
    __shared__ unsigned int s_warpHist[kSharedHistEntries];

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & (kWarpSize - 1);
    const unsigned int warpId = tid >> 5;

    unsigned int* const warpHist = s_warpHist + warpId * kMaxBins;

    // Fixed-size shared footprint (8 warps * 256 bins = 8 KiB). Zeroing the full shared
    // histogram keeps the kernel simple and avoids dynamic-shared-memory bookkeeping.
    #pragma unroll
    for (int i = 0; i < kSharedInitIters; ++i) {
        s_warpHist[tid + i * kBlockThreads] = 0u;
    }
    __syncthreads();

    const unsigned long long inputSize64 = static_cast<unsigned long long>(inputSize);
    const unsigned long long globalTid   =
        static_cast<unsigned long long>(blockIdx.x) * kBlockThreads + tid;
    const unsigned long long gridStride  =
        static_cast<unsigned long long>(gridDim.x) * kBlockThreads;
    const unsigned long long loopStride  = gridStride * kUnroll;

    // Grid-stride traversal with modest unrolling to raise ILP without bloating register use.
    for (unsigned long long base = globalTid; base < inputSize64; base += loopStride) {
        add_byte_to_warp_hist(static_cast<unsigned int>(input[base]), from, numBins, lane, warpHist);

        const unsigned long long idx1 = base + gridStride;
        if (idx1 < inputSize64) {
            add_byte_to_warp_hist(static_cast<unsigned int>(input[idx1]), from, numBins, lane, warpHist);
        }

        const unsigned long long idx2 = idx1 + gridStride;
        if (idx2 < inputSize64) {
            add_byte_to_warp_hist(static_cast<unsigned int>(input[idx2]), from, numBins, lane, warpHist);
        }

        const unsigned long long idx3 = idx2 + gridStride;
        if (idx3 < inputSize64) {
            add_byte_to_warp_hist(static_cast<unsigned int>(input[idx3]), from, numBins, lane, warpHist);
        }
    }

    __syncthreads();

    // Because kBlockThreads == 256 and numBins <= 256, each output bin is owned by at most
    // one thread in the block during the final reduction.
    if (tid < numBins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += s_warpHist[w * kMaxBins + tid];
        }

        if (gridDim.x == 1) {
            // Single-block launch: write every output bin directly, including zeros.
            histogram[tid] = sum;
        } else if (sum != 0u) {
            // Multi-block launch: one global atomic per non-zero bin contribution from this block.
            atomicAdd(histogram + tid, sum);
        }
    }
}

// Cache the maximum resident grid size for the current device.
// The kernel shape is fixed, so the occupancy result is stable per device.
inline int max_resident_blocks_for_kernel()
{
    thread_local int cachedDevice    = -1;
    thread_local int cachedGridLimit = 0;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 1;
    }

    if (device != cachedDevice || cachedGridLimit <= 0) {
        int smCount = 1;
        int activeBlocksPerSm = 1;

        if (cudaDeviceGetAttribute(&smCount,
                                   cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess || smCount <= 0) {
            smCount = 1;
        }

        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSm,
                                                          histogram_range_kernel,
                                                          kBlockThreads,
                                                          0) != cudaSuccess ||
            activeBlocksPerSm <= 0) {
            activeBlocksPerSm = 1;
        }

        cachedGridLimit = smCount * activeBlocksPerSm;
        if (cachedGridLimit <= 0) {
            cachedGridLimit = 1;
        }

        cachedDevice = device;
    }

    return cachedGridLimit;
}

} // namespace

// Enqueues the histogram work on stream 0. The caller is responsible for any later
// synchronization and error handling, per the requested API contract.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int fromU   = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
        return;
    }

    // Launch enough blocks to fill the GPU on large inputs, but avoid launching an excessive
    // number of blocks for small/medium buffers because each block has a fixed histogram
    // init/reduction cost.
    const unsigned int blocksForInput =
        static_cast<unsigned int>(
            (static_cast<unsigned long long>(inputSize) +
             static_cast<unsigned long long>(kTargetBytesPerBlock) - 1ull) /
            static_cast<unsigned long long>(kTargetBytesPerBlock));

    const unsigned int maxBlocks =
        static_cast<unsigned int>(max_resident_blocks_for_kernel());

    unsigned int launchBlocks =
        (blocksForInput < maxBlocks) ? blocksForInput : maxBlocks;

    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }

    // Multi-block launches accumulate into global memory, so the destination must be zeroed
    // first. Single-block launches skip the memset because the kernel writes every output bin.
    if (launchBlocks > 1u) {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
    }

    // Reinterpret the input as unsigned bytes so values 128..255 are handled by ordinal value,
    // independent of whether host-side 'char' is signed or unsigned.
    const unsigned char* const inputBytes =
        reinterpret_cast<const unsigned char*>(input);

    histogram_range_kernel<<<launchBlocks, kBlockThreads, 0, 0>>>(
        inputBytes,
        histogram,
        inputSize,
        fromU,
        numBins);
}