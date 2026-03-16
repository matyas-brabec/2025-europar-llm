#include <cuda_runtime.h>

// Tuning constants for modern data-center NVIDIA GPUs (A100/H100 class).
// - 256 threads/block gives 8 warps per block: enough concurrency while keeping
//   inter-warp contention on the shared histogram copies under control.
// - 8 items/thread is a good large-input default for byte histograms: it
//   amortizes loop/index overhead without inflating register pressure.
constexpr int itemsPerThread  = 8;
constexpr int blockThreads    = 256;
constexpr int histogramCopies = 32;
constexpr unsigned int fullWarpMask = 0xffffffffu;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(histogramCopies == 32, "This layout assumes 32 shared-memory banks.");
static_assert(blockThreads % histogramCopies == 0,
              "blockThreads must be a multiple of 32 so blocks are composed of full warps.");

// Warp-wide reduction used when summing the 32 privatized copies of one histogram bin.
// Lane c reads copy c, so the shared-memory read pattern is also bank-conflict free.
__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
#pragma unroll
    for (int offset = histogramCopies / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(fullWarpMask, value, offset);
    }
    return value;
}

// Range-restricted byte histogram.
//
// Shared-memory privatization layout requested by the prompt:
//   sharedHistogram[bin * 32 + copy]
//
// where copy == threadIdx.x % 32.
//
// This layout makes copy c of every bin live in bank c:
// - Hot update path: lane c updates copy c -> each lane hits a distinct bank.
// - Final reduction path: one warp sums one bin at a time, and lane c reads copy c
//   -> again one bank per lane, so no shared-memory bank conflicts.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS)
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       unsigned int lower,
                                       unsigned int bins) {
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;  // Also the histogram copy index: threadIdx.x % 32.
    const unsigned int warp = tid >> 5;   // threadIdx.x / 32

    constexpr unsigned int warpsPerBlock =
        static_cast<unsigned int>(BLOCK_THREADS / histogramCopies);
    constexpr size_t blockTileItems =
        static_cast<size_t>(BLOCK_THREADS) * static_cast<size_t>(ITEMS_PER_THREAD);
    constexpr size_t maxItemOffset =
        static_cast<size_t>(ITEMS_PER_THREAD - 1) * static_cast<size_t>(BLOCK_THREADS);

    const size_t n = static_cast<size_t>(inputSize);
    const size_t gridTileItems = static_cast<size_t>(gridDim.x) * blockTileItems;
    size_t base = static_cast<size_t>(blockIdx.x) * blockTileItems + static_cast<size_t>(tid);

    const unsigned char* __restrict__ bytes =
        reinterpret_cast<const unsigned char*>(input);

    // Zero all shared histogram copies. The layout is dense, so a simple thread-stride loop is efficient.
    const unsigned int privatizedBins = bins * static_cast<unsigned int>(histogramCopies);
    for (unsigned int i = tid; i < privatizedBins; i += BLOCK_THREADS) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    // Fast path for complete tiles: no per-item bounds checks in the hot loop.
    // Only the single possible partial tile owned by this block needs the checked path below.
    const size_t fullTileEnd = (n > maxItemOffset) ? (n - maxItemOffset) : 0;

    for (; base < fullTileEnd; base += gridTileItems) {
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const unsigned int ch =
                static_cast<unsigned int>(bytes[base + static_cast<size_t>(item) * BLOCK_THREADS]);

            // Inclusive range test using unsigned arithmetic:
            // - ch < lower  => underflow, bin becomes very large
            // - ch > upper  => bin >= bins
            const unsigned int bin = ch - lower;
            if (bin < bins) {
                atomicAdd(&sharedHistogram[bin * histogramCopies + lane], 1u);
            }
        }
    }

    // Tail path for the final partial tile.
    for (; base < n; base += gridTileItems) {
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const size_t idx = base + static_cast<size_t>(item) * BLOCK_THREADS;
            if (idx < n) {
                const unsigned int ch = static_cast<unsigned int>(bytes[idx]);
                const unsigned int bin = ch - lower;
                if (bin < bins) {
                    atomicAdd(&sharedHistogram[bin * histogramCopies + lane], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 privatized copies of each bin back into the global histogram.
    // Each warp handles one bin at a time:
    //   lane c reads copy c      -> bank-conflict-free
    //   warp reduction sums them -> one total per bin
    //   lane 0 does one global atomic add per bin and per block
    for (unsigned int bin = warp; bin < bins; bin += warpsPerBlock) {
        unsigned int sum = sharedHistogram[bin * histogramCopies + lane];
        sum = warp_reduce_sum(sum);
        if (lane == 0u && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host launcher.
// The API contract says input/histogram are already cudaMalloc-allocated device buffers,
// and the caller is responsible for any synchronization. Because the function signature
// has no stream parameter, work is issued on the default stream.
void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The problem statement guarantees 0 <= from < to <= 255.
    const unsigned int lower = static_cast<unsigned int>(from);
    const unsigned int bins  = static_cast<unsigned int>(to - from + 1);
    const cudaStream_t stream = 0;

    // Zero exactly the requested output range.
    cudaMemsetAsync(histogram,
                    0,
                    static_cast<size_t>(bins) * sizeof(unsigned int),
                    stream);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory must hold 32 copies of the requested histogram range.
    const size_t sharedBytes =
        static_cast<size_t>(bins) * static_cast<size_t>(histogramCopies) * sizeof(unsigned int);

    // Size the grid from actual occupancy for this dynamic shared-memory footprint.
    // The kernel is grid-stride and large-input oriented, so one occupancy-saturating
    // wave is sufficient; each block consumes more work through the grid-stride loop.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogram_range_kernel<blockThreads, itemsPerThread>,
        blockThreads,
        sharedBytes);
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    const size_t tileItems =
        static_cast<size_t>(blockThreads) * static_cast<size_t>(itemsPerThread);
    const size_t numTiles =
        (static_cast<size_t>(inputSize) + tileItems - 1) / tileItems;

    const int maxResidentGrid = smCount * activeBlocksPerSm;
    int grid = maxResidentGrid;
    if (numTiles < static_cast<size_t>(grid)) {
        grid = static_cast<int>(numTiles);
    }
    if (grid < 1) {
        grid = 1;
    }

    histogram_range_kernel<blockThreads, itemsPerThread>
        <<<grid, blockThreads, sharedBytes, stream>>>(input, histogram, inputSize, lower, bins);
}