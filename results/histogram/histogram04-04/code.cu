#include <cuda_runtime.h>

namespace {

// Tuned for modern data-center GPUs (A100/H100):
// - 256 threads/block gives 8 full warps and lets threadIdx.x map directly to the 256 possible byte values.
// - itemsPerThread = 16 makes each CTA process 4 KiB per grid-stride tile, which is a good default for
//   large byte streams: it amortizes shared-histogram setup/flush costs without hurting occupancy.
constexpr int threadsPerBlock = 256;
constexpr int itemsPerThread = 16;
constexpr int warpSizeConst = 32;
constexpr int warpsPerBlock = threadsPerBlock / warpSizeConst;
constexpr int histogramBinsMax = 256;
constexpr unsigned int invalidBin = histogramBinsMax;  // sentinel outside the legal [0,255] bin range
constexpr unsigned int fullWarpMask = 0xffffffffu;

static_assert(threadsPerBlock == histogramBinsMax,
              "The flush phase assumes one thread per possible byte value.");
static_assert(threadsPerBlock % warpSizeConst == 0,
              "threadsPerBlock must be a multiple of warp size.");

__device__ __forceinline__
void add_to_warp_private_histogram(unsigned int* myWarpHist,
                                   unsigned int key,
                                   unsigned int lane)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    // Warp-aggregated atomic update:
    // lanes that saw the same byte value form an equivalence class, and only the leader
    // performs the shared-memory atomic with the class population count.
    const unsigned int sameKeyMask = __match_any_sync(fullWarpMask, key);
    if (key != invalidBin && lane == static_cast<unsigned int>(__ffs(sameKeyMask) - 1)) {
        atomicAdd(myWarpHist + key, __popc(sameKeyMask));
    }
#else
    // Fallback for pre-Volta builds.
    if (key != invalidBin) {
        atomicAdd(myWarpHist + key, 1u);
    }
#endif
}

__global__ __launch_bounds__(threadsPerBlock)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int range)
{
    // Each warp owns a private 256-bin histogram in shared memory.
    // Using the full byte space keeps indexing trivial; the cost is only 8 KiB/CTA,
    // which is not occupancy-limiting on A100/H100 with 256-thread blocks.
    __shared__ unsigned int warpHist[warpsPerBlock * histogramBinsMax];

    // Zero the privatized histograms. Because 8 warps * 256 bins = 2048 counters and we use
    // 256 threads, every thread clears exactly 8 counters.
    #pragma unroll
    for (int w = 0; w < warpsPerBlock; ++w) {
        warpHist[w * threadsPerBlock + threadIdx.x] = 0u;
    }
    __syncthreads();

    const unsigned char* __restrict__ bytes =
        reinterpret_cast<const unsigned char*>(input);

    const unsigned int lane = threadIdx.x & (warpSizeConst - 1u);
    const unsigned int warp = threadIdx.x >> 5;
    unsigned int* const myWarpHist = warpHist + warp * histogramBinsMax;

    const size_t n = static_cast<size_t>(inputSize);
    const size_t blockTileSize = static_cast<size_t>(threadsPerBlock) * itemsPerThread;
    const size_t gridTileStride = blockTileSize * gridDim.x;

    // Persistent/grid-stride traversal:
    // each block accumulates many tiles into its shared histogram and flushes to global memory once,
    // which greatly reduces global atomic traffic.
    for (size_t tileBase = static_cast<size_t>(blockIdx.x) * blockTileSize;
         tileBase < n;
         tileBase += gridTileStride)
    {
        const size_t threadBase = tileBase + threadIdx.x;

        // Per-thread work is laid out as threadIdx.x + item*blockDim.x so each unrolled step
        // issues coalesced byte loads across the warp.
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            const size_t idx = threadBase + static_cast<size_t>(item) * threadsPerBlock;

            // Sentinel 256 means "ignore this byte" (either out of bounds or outside [from, to]).
            unsigned int key = invalidBin;

            if (idx < n) {
                const unsigned int value = bytes[idx];
                // Unsigned wraparound turns the lower-bound check into part of the same compare:
                // value is in range iff (value - from) <= (to - from).
                const unsigned int delta = value - from;
                if (delta <= range) {
                    key = delta;  // output histogram is indexed from zero: value -> value - from
                }
            }

            add_to_warp_private_histogram(myWarpHist, key, lane);
        }
    }

    __syncthreads();

    // Final reduction: one thread per possible byte value sums the 8 warp-private counters
    // and atomically adds the block total to the global histogram.
    if (threadIdx.x <= range) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += warpHist[w * histogramBinsMax + threadIdx.x];
        }
        if (sum != 0u) {
            atomicAdd(histogram + threadIdx.x, sum);
        }
    }
}

} // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int ufrom = static_cast<unsigned int>(from);
    const unsigned int range = static_cast<unsigned int>(to - from);
    const size_t histogramBytes =
        static_cast<size_t>(range + 1u) * sizeof(unsigned int);

    // The kernel accumulates into the destination histogram, so clear it first.
    // Per the requested interface, this function only enqueues work; synchronization
    // and asynchronous error handling are left to the caller.
    cudaMemsetAsync(histogram, 0, histogramBytes);

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    int smCount = 1;
    int activeBlocksPerSM = 1;

    // Query a resident-grid launch size. Because the kernel uses a grid-stride loop,
    // launching roughly one resident wave of CTAs is usually best: it keeps the GPU full
    // while avoiding unnecessary per-CTA init/final-flush overhead.
    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM, histogram_range_kernel, threadsPerBlock, 0);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const unsigned int blockTileSize = threadsPerBlock * itemsPerThread;
    const int totalTiles = static_cast<int>(
        (static_cast<unsigned long long>(inputSize) + blockTileSize - 1ull) /
        blockTileSize);
    const int residentGridSize = smCount * activeBlocksPerSM;
    const int gridSize = (totalTiles < residentGridSize) ? totalTiles : residentGridSize;

    histogram_range_kernel<<<gridSize, threadsPerBlock>>>(
        input, histogram, inputSize, ufrom, range);
}