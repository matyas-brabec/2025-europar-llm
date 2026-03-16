#include <cuda_runtime.h>

// Tuned for modern datacenter GPUs (A100/H100 class).
// - 32 copies: one privatized histogram copy per warp lane / shared-memory bank.
// - 256 threads: 8 warps per block, a good balance between occupancy and per-copy contention.
// - 16 items/thread: 4096 input bytes per full block tile; this is a strong default for
//   large inputs because it amortizes loop overhead without needlessly increasing code size
//   or register pressure.
constexpr int histogramCopies = 32;
constexpr int blockThreads = 256;
constexpr int itemsPerThread = 16;

static_assert(histogramCopies == 32, "The bank-striped layout assumes 32 shared-memory banks.");
static_assert(blockThreads % histogramCopies == 0, "blockThreads must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

/*
  Range-restricted byte histogram.

  Design:
  - Input is treated as unsigned bytes so ordinal values 0..255 are handled correctly even if
    the host compiler's plain `char` is signed.
  - Each block owns 32 privatized histograms in shared memory.
  - Shared-memory layout is bank-striped:
        counter(bin, copy) -> sharedHistogram[bin * 32 + copy]
    For a thread with lane/copy `c`, every update goes to addresses:
        c, 32 + c, 64 + c, ...
    so that thread always touches the same bank, and a warp updating arbitrary bins remains
    bank-conflict free because lane `l` maps to bank `l` (modulo the base-bank offset).
  - The shared histogram lives for the whole grid-stride traversal and is flushed only once,
    which strongly amortizes initialization and global-atomic overhead on large inputs.
  - Final reduction uses one warp per bin: lane `l` reads copy `l`, then a warp shuffle sum
    produces the block contribution with no shared-memory bank conflicts.
*/
template <int ItemsPerThread>
__global__ __launch_bounds__(blockThreads)
void histogram_range_kernel(const char * __restrict__ input,
                            unsigned int * __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int numBins) {
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (histogramCopies - 1); // Equivalent to threadIdx.x % 32.
    const unsigned int warp = tid >> 5;                    // threadIdx.x / 32.
    const unsigned int warpsPerBlock = blockDim.x >> 5;

    const size_t totalSharedCounters = static_cast<size_t>(numBins) * histogramCopies;

    // Zero all 32 privatized histogram copies.
    for (size_t i = tid; i < totalSharedCounters; i += blockDim.x) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    const unsigned char *const inputBytes = reinterpret_cast<const unsigned char *>(input);

    // Each thread updates its own lane-selected histogram copy.
    unsigned int *const myHistogramCopy = sharedHistogram + lane;

    const size_t n = static_cast<size_t>(inputSize);
    const size_t threadStride = static_cast<size_t>(blockDim.x);
    const size_t blockItems = threadStride * ItemsPerThread;
    const size_t gridStrideItems = blockItems * static_cast<size_t>(gridDim.x);

    // Block-stride traversal over contiguous tiles. Within a tile, item `k` for thread `t`
    // reads tileBase + k * blockDim.x + t, so each warp performs coalesced byte loads.
    size_t tileBase = static_cast<size_t>(blockIdx.x) * blockItems;

    // Fast path for full tiles: no per-item bounds checks.
    for (; tileBase + blockItems <= n; tileBase += gridStrideItems) {
        size_t idx = tileBase + tid;

        #pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item, idx += threadStride) {
            const unsigned int ch = inputBytes[idx];

            // Unsigned subtraction turns the inclusive [from, to] test into one range compare.
            const unsigned int bin = ch - from;
            if (bin < numBins) {
                atomicAdd(myHistogramCopy + static_cast<size_t>(bin) * histogramCopies, 1u);
            }
        }
    }

    // Tail tile.
    if (tileBase < n) {
        size_t idx = tileBase + tid;

        #pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item, idx += threadStride) {
            if (idx < n) {
                const unsigned int ch = inputBytes[idx];
                const unsigned int bin = ch - from;
                if (bin < numBins) {
                    atomicAdd(myHistogramCopy + static_cast<size_t>(bin) * histogramCopies, 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 copies back to the output histogram.
    // One warp owns one bin at a time, so lane `l` reads copy `l`; those 32 loads are
    // naturally bank conflict free with the requested bank-striped layout.
    for (unsigned int bin = warp; bin < numBins; bin += warpsPerBlock) {
        const unsigned int *const binCopies =
            sharedHistogram + static_cast<size_t>(bin) * histogramCopies;

        unsigned int blockCount = binCopies[lane];
        blockCount = warp_reduce_sum(blockCount);

        if (lane == 0) {
            if (gridDim.x == 1) {
                // Single-block launch: no global atomic needed.
                histogram[bin] = blockCount;
            } else if (blockCount != 0u) {
                atomicAdd(histogram + bin, blockCount);
            }
        }
    }
}

static inline unsigned int ceil_div_u32(unsigned int a, unsigned int b) {
    // Overflow-safe ceil(a / b) for 32-bit unsigned inputs.
    return a / b + ((a % b) != 0u);
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Dynamic shared memory stores 32 privatized copies of the requested histogram range.
    // Full-range worst case: 256 bins * 32 copies * 4 bytes = 32 KiB per block.
    const size_t sharedBytes =
        static_cast<size_t>(numBins) * histogramCopies * sizeof(unsigned int);

    // This function computes a fresh histogram, so clear the destination first.
    // No host-device synchronization is performed here; the caller handles it.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    // Launch only a resident wave of blocks. Because each block keeps its privatized shared
    // histogram for the entire grid-stride traversal, this persistent-CTA style launch avoids
    // paying shared-memory setup/flush overhead more often than necessary.
    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<itemsPerThread>,
        blockThreads,
        sharedBytes);

    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    if (smCount < 1) {
        smCount = 1;
    }

    const unsigned int blockItems = blockThreads * itemsPerThread;
    const unsigned int totalTiles = ceil_div_u32(inputSize, blockItems);
    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(activeBlocksPerSM) * static_cast<unsigned int>(smCount);
    const unsigned int gridBlocks =
        (totalTiles < maxResidentBlocks) ? totalTiles : maxResidentBlocks;

    histogram_range_kernel<itemsPerThread><<<gridBlocks, blockThreads, sharedBytes, 0>>>(
        input,
        histogram,
        inputSize,
        fromU,
        numBins);
}