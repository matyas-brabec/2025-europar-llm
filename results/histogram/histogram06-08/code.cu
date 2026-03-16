#include <cuda_runtime.h>

// Range-restricted byte histogram optimized for modern NVIDIA data-center GPUs.
//
// Design summary:
// - Each thread block builds a privatized histogram in shared memory.
// - Every bin is replicated 32 times, one logical copy per warp lane / shared-memory bank.
// - A thread updates only the copy indexed by its lane, so warp-wide updates are bank-conflict free.
// - After the block finishes its grid-stride chunk of the input, it reduces the 32 copies per bin
//   with warp shuffles and atomically adds the result to the output histogram in global memory.

namespace {

// 256 threads (8 warps) is a strong default on A100/H100 for this shared-atomic histogram:
// enough warps to hide latency, while keeping inter-warp contention on each per-lane copy moderate.
constexpr int kWarpSize          = 32;
constexpr int kHistogramCopies   = 32;
constexpr int kHistogramStride   = 32;  // stride in 32-bit words between consecutive bins of one copy
constexpr int kBlockThreads      = 256;
constexpr int kWarpsPerBlock     = kBlockThreads / kWarpSize;

// Large-input default: 16 bytes/thread => 4 KiB of input per CTA tile.
// This amortizes shared-histogram init/flush overhead well on Ampere/Hopper
// without inflating code size or register pressure too much.
constexpr int    itemsPerThread  = 16;
constexpr size_t kBlockWork      = static_cast<size_t>(kBlockThreads) * static_cast<size_t>(itemsPerThread);

constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(kHistogramCopies == kWarpSize, "Need one histogram copy per warp lane.");
static_assert((kBlockThreads % kWarpSize) == 0, "Block size must be a whole number of warps.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

}  // namespace

__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            size_t inputSize,
                            unsigned int rangeBegin,
                            unsigned int rangeSize) {
    // Dynamic shared-memory layout:
    //   s_hist[bin * 32 + lane]
    //
    // This is a bin-major, copy-minor layout with 32 logical copies of every bin.
    // Copy 'lane' occupies indices:
    //   lane, lane + 32, lane + 64, ...
    //
    // Because shared memory has 32 banks and counters are 32-bit wide, lane l always maps to
    // bank (base_bank + l) % 32. Therefore, when a warp updates any set of bins, each lane touches
    // a distinct bank and the update is bank-conflict free within that warp.
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid   = threadIdx.x;
    const unsigned int lane  = tid & (kWarpSize - 1);
    const unsigned int warp  = tid >> 5;
    const unsigned int words = rangeSize * kHistogramStride;

    // Zero only the portion of shared memory needed for the requested [from, to] range.
    for (unsigned int i = tid; i < words; i += kBlockThreads) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // This thread always updates the copy that corresponds to its warp lane.
    // Different warps share the same 32 copies, so shared-memory atomics are still required
    // to resolve inter-warp races on the same per-lane copy.
    unsigned int* const laneHistogram = s_hist + lane;

    const size_t gridWork = static_cast<size_t>(gridDim.x) * kBlockWork;

    // Persistent grid-stride processing. Each CTA consumes kBlockWork bytes per tile.
    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * kBlockWork;
         blockBase < inputSize;
         blockBase += gridWork) {
        const size_t threadBase = blockBase + tid;

        if (blockBase + kBlockWork <= inputSize) {
            // Fast path for full 4 KiB CTA tiles: no per-item bounds checks.
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                const size_t idx = threadBase + static_cast<size_t>(item) * kBlockThreads;
                const unsigned int value = static_cast<unsigned int>(input[idx]);

                // Branchless range test:
                //   if value < rangeBegin, unsigned underflow makes bin very large,
                //   so the compare fails automatically.
                const unsigned int bin = value - rangeBegin;
                if (bin < rangeSize) {
                    atomicAdd(laneHistogram + bin * kHistogramStride, 1u);
                }
            }
        } else {
            // Tail tile.
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                const size_t idx = threadBase + static_cast<size_t>(item) * kBlockThreads;
                if (idx < inputSize) {
                    const unsigned int value = static_cast<unsigned int>(input[idx]);
                    const unsigned int bin = value - rangeBegin;
                    if (bin < rangeSize) {
                        atomicAdd(laneHistogram + bin * kHistogramStride, 1u);
                    }
                }
            }
        }
    }

    __syncthreads();

    // Flush the block-private histogram to global memory.
    // Each warp reduces one bin at a time: lane l reads copy l, so the flush path is also
    // bank-conflict free.
    for (unsigned int bin = warp; bin < rangeSize; bin += kWarpsPerBlock) {
        const unsigned int binBase = bin * kHistogramStride;
        unsigned int sum = s_hist[binBase + lane];
        sum = warp_reduce_sum(sum);

        // histogram[bin] corresponds to byte value (rangeBegin + bin).
        if (lane == 0 && sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int rangeBegin = static_cast<unsigned int>(from);
    const unsigned int rangeSize  = static_cast<unsigned int>(to - from + 1);

    // histogram[0] corresponds to byte value 'from'.
    // The result must be produced from scratch, so zero the output first.
    // Use stream 0 and leave synchronization to the caller, as requested.
    cudaMemsetAsync(
        histogram,
        0,
        static_cast<size_t>(rangeSize) * sizeof(unsigned int),
        0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(rangeSize) * kHistogramStride * sizeof(unsigned int);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel,
        kBlockThreads,
        sharedBytes);

    // Launch no more CTAs than can be resident concurrently.
    // The kernel is grid-stride, so oversubscribing the grid would only increase
    // per-CTA initialization / flush overhead and global atomic traffic.
    const size_t blocksForInput =
        (static_cast<size_t>(inputSize) + (kBlockWork - 1)) / kBlockWork;

    unsigned int gridSize = static_cast<unsigned int>(smCount * activeBlocksPerSM);
    if (gridSize == 0u) {
        gridSize = 1u;
    }
    if (static_cast<size_t>(gridSize) > blocksForInput) {
        gridSize = static_cast<unsigned int>(blocksForInput);
    }
    if (gridSize == 0u) {
        gridSize = 1u;
    }

    histogram_range_kernel<<<gridSize, kBlockThreads, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        static_cast<size_t>(inputSize),
        rangeBegin,
        rangeSize);
}