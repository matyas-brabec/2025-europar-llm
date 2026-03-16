#include <cuda_runtime.h>

namespace {

// 256 threads is deliberate: the histogram has at most 256 bins, so the final
// block->global flush can map one thread to one bin in the worst case.
constexpr int blockThreads = 256;

// Compile-time tuning knob requested by the problem.
// 16 is a strong default on A100/H100-class GPUs for large inputs: it gives
// enough ILP to hide byte-load/shared-atomic latency and keeps the unrolled
// inner loop compact enough to avoid unnecessary register/code-size growth.
constexpr int itemsPerThread = 16;

constexpr int warpsPerBlock = blockThreads / 32;
constexpr unsigned int invalidBin = ~0u;

static_assert(blockThreads % 32 == 0, "blockThreads must be a multiple of 32");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive");

// Update a warp-private shared-memory histogram.
// On Volta+ we use warp-aggregated atomics via __match_any_sync so that all
// lanes in a warp targeting the same bin collapse to a single shared atomic.
// This is especially effective for real text where spaces/newlines/common
// letters repeat frequently. The invalidBin sentinel lets every lane execute
// the warp intrinsic uniformly, even for out-of-range bytes or tail lanes.
__device__ __forceinline__ void warp_aggregated_shared_add(
    unsigned int* warpHistogram,
    unsigned int bin,
    unsigned int lane)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const unsigned int activeMask = __activemask();
    const unsigned int peers = __match_any_sync(activeMask, bin);
    const int leader = __ffs(static_cast<int>(peers)) - 1;

    if (bin != invalidBin && static_cast<int>(lane) == leader) {
        atomicAdd(warpHistogram + bin, static_cast<unsigned int>(__popc(peers)));
    }
#else
    (void)lane;
    if (bin != invalidBin) {
        atomicAdd(warpHistogram + bin, 1u);
    }
#endif
}

// Single-pass histogram kernel.
// - Input is treated as raw unsigned bytes [0, 255].
// - Only bytes in the inclusive range [fromOrdinal, fromOrdinal + numBins - 1]
//   are counted.
// - Shared-memory privatization is per warp, not just per block, which reduces
//   intra-block contention. Each block accumulates across all of its grid-stride
//   tiles and flushes once at the end, so global atomic traffic is only
//   O(blocks * bins), not O(input bytes).
template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(blockThreads)
void histogram_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int fromOrdinal,
    unsigned int numBins)
{
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid = threadIdx.x;
    const unsigned int warpId = tid >> 5;
    const unsigned int lane = tid & 31u;

    // Shared layout: [warp0 bins][warp1 bins]...[warpN bins].
    // With 256 threads => 8 warps, and with at most 256 bins, the maximum shared
    // memory footprint is 8 * 256 * 4 B = 8 KiB.
    const unsigned int sharedBinCount = static_cast<unsigned int>(warpsPerBlock) * numBins;
    for (unsigned int i = tid; i < sharedBinCount; i += static_cast<unsigned int>(blockThreads)) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    unsigned int* const warpHistogram = sharedHistogram + warpId * numBins;

    const size_t n = static_cast<size_t>(inputSize);
    const size_t blockWork = static_cast<size_t>(blockThreads) * static_cast<size_t>(ITEMS_PER_THREAD);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockWork;

    // Strip-mined access pattern:
    // item k of a warp reads 32 consecutive bytes, so loads are naturally
    // coalesced. With the default tuning, one block touches
    // 256 threads * 16 items/thread = 4096 bytes per outer-loop iteration.
    for (size_t blockOffset = static_cast<size_t>(blockIdx.x) * blockWork;
         blockOffset < n;
         blockOffset += gridStride) {
        const size_t threadBase = blockOffset + static_cast<size_t>(tid);

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const size_t idx =
                threadBase + static_cast<size_t>(item) * static_cast<size_t>(blockThreads);

            unsigned int bin = invalidBin;
            if (idx < n) {
                const unsigned int value = static_cast<unsigned int>(input[idx]);
                const unsigned int candidate = value - fromOrdinal;
                if (candidate < numBins) {
                    bin = candidate;
                }
            }

            warp_aggregated_shared_add(warpHistogram, bin, lane);
        }
    }

    __syncthreads();

    // End-of-block merge over the warp-private histograms. Because numBins <= 256
    // and blockThreads == 256, even the worst case is at most one bin per thread.
    for (unsigned int bin = tid; bin < numBins; bin += static_cast<unsigned int>(blockThreads)) {
        unsigned int sum = 0u;

#pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += sharedHistogram[static_cast<unsigned int>(w) * numBins + bin];
        }

        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

} // namespace

void run_histogram(const char* input, unsigned int* histogram, unsigned int inputSize, int from, int to)
{
    // The problem statement guarantees valid device pointers and a valid range,
    // so we keep the launcher lean and avoid extra host-side checks.

    // Inclusive range [from, to] => numBins = to - from + 1.
    // histogram[0] counts byte value `from`, histogram[numBins - 1] counts `to`.
    const unsigned int fromOrdinal = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // The kernel only accumulates, so zero the output slice first.
    // No synchronization is performed here; the caller owns stream/device sync.
    // We use stream 0 because the requested API does not carry a stream handle.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedMemBytes =
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // The kernel grid-strides over the input, so correctness does not depend on
    // the exact grid size. We cap the launch to about two occupancy waves:
    // enough work to saturate the GPU while avoiding gratuitous extra blocks that
    // would only increase the final global merge traffic.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSm,
        histogram_kernel<itemsPerThread>,
        blockThreads,
        sharedMemBytes);

    const size_t workPerBlock =
        static_cast<size_t>(blockThreads) * static_cast<size_t>(itemsPerThread);
    const unsigned int totalBlocksNeeded = static_cast<unsigned int>(
        (static_cast<size_t>(inputSize) + workPerBlock - 1) / workPerBlock);

    unsigned int launchBlocks = totalBlocksNeeded;
    if (smCount > 0 && maxActiveBlocksPerSm > 0) {
        const unsigned int launchCap =
            static_cast<unsigned int>(smCount) *
            static_cast<unsigned int>(maxActiveBlocksPerSm) * 2u;
        if (launchBlocks > launchCap) {
            launchBlocks = launchCap;
        }
    }

    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }

    // Reinterpret the device buffer as unsigned bytes so values 128..255 are
    // handled correctly even on platforms where plain char is signed.
    histogram_kernel<itemsPerThread><<<launchBlocks, blockThreads, sharedMemBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        fromOrdinal,
        numBins);
}