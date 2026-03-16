#include <cuda_runtime.h>

// Device-resident range-limited byte histogram.
//
// Design tuned for large inputs on modern data-center GPUs (A100/H100 class):
//   * Shared-memory privatization keeps almost all atomics on-chip.
//   * Each CTA owns 32 interleaved copies of the histogram (one per shared-memory
//     bank / warp lane), which removes intra-warp bank conflicts during updates.
//   * A persistent grid-stride traversal sized to resident CTAs minimizes the
//     number of per-block global histogram flushes.
//   * The 32 copies are reduced with warp shuffles; lane j reads copy j, so the
//     readback is bank-conflict free as well.

namespace {

// User-tunable throughput knob. For large streaming inputs on recent NVIDIA GPUs,
// 16 items/thread is a strong default: it amortizes index math and loop overhead
// while keeping register pressure and code size modest.
constexpr int itemsPerThread = 16;

// 256 threads/block is a good balance here: enough warps to hide latency, but not
// so many that shared-memory atomic contention is unnecessarily amplified.
constexpr int threadsPerBlock = 256;

constexpr int kWarpSize = 32;

// Shared memory has 32 banks on modern NVIDIA GPUs. Using 32 interleaved copies
// makes the bank index equal to the lane id for every update. Max shared-memory
// footprint is therefore 256 bins * 32 copies * 4 B = 32 KiB.
constexpr int sharedHistogramCopies = 32;

constexpr int warpsPerBlock = threadsPerBlock / kWarpSize;
constexpr unsigned int fullWarpMask = 0xFFFFFFFFu;

constexpr size_t itemsPerBlock =
    static_cast<size_t>(threadsPerBlock) * static_cast<size_t>(itemsPerThread);
constexpr size_t lastItemOffset =
    static_cast<size_t>(itemsPerThread - 1) * static_cast<size_t>(threadsPerBlock);

static_assert((threadsPerBlock % kWarpSize) == 0,
              "threadsPerBlock must be a multiple of warp size.");
static_assert(sharedHistogramCopies == kWarpSize,
              "This implementation relies on 32-lane warps and 32 shared-memory banks.");
static_assert((sharedHistogramCopies & (sharedHistogramCopies - 1)) == 0,
              "sharedHistogramCopies must be a power of two.");
static_assert(itemsPerBlock > lastItemOffset,
              "Tail handling assumes at most one partial grid-stride iteration.");

// Update one bank-striped shared-memory sub-histogram entry.
//
// The range check is intentionally done with unsigned subtraction:
//   bin = value - rangeStart
// and a single comparison:
//   bin < numBins
// Values below rangeStart underflow and therefore fail the comparison.
__device__ __forceinline__
void accumulate_byte(unsigned char value,
                     unsigned int rangeStart,
                     unsigned int numBins,
                     unsigned int copy,
                     unsigned int* __restrict__ s_hist) {
    const unsigned int bin = static_cast<unsigned int>(value) - rangeStart;
    if (bin < numBins) {
        atomicAdd(&s_hist[bin * sharedHistogramCopies + copy], 1u);
    }
}

__global__ __launch_bounds__(threadsPerBlock)
void histogram_kernel(const char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      unsigned int inputSize,
                      unsigned int rangeStart,
                      unsigned int numBins) {
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid    = static_cast<unsigned int>(threadIdx.x);
    const unsigned int lane   = tid & static_cast<unsigned int>(kWarpSize - 1);
    const unsigned int warpId = tid >> 5;  // kWarpSize == 32

    const unsigned int totalCounters = numBins * sharedHistogramCopies;

    // Zero the CTA-private shared histogram.
    for (unsigned int i = tid; i < totalCounters; i += threadsPerBlock) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned char* __restrict__ input_u8 =
        reinterpret_cast<const unsigned char*>(input);
    const size_t inputSizeZ = static_cast<size_t>(inputSize);

    // Persistent grid-stride traversal.
    size_t base = static_cast<size_t>(blockIdx.x) * itemsPerBlock + static_cast<size_t>(tid);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * itemsPerBlock;

    // Fast path: all items in the unrolled bundle are in-bounds, so no per-item checks.
    for (; base + lastItemOffset < inputSizeZ; base += gridStride) {
        const unsigned char* p = input_u8 + base;
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            accumulate_byte(
                p[static_cast<size_t>(item) * static_cast<size_t>(threadsPerBlock)],
                rangeStart,
                numBins,
                lane,
                s_hist);
        }
    }

    // Tail: because gridStride >= itemsPerBlock > lastItemOffset, there can be at most
    // one partially full iteration left for each thread.
    if (base < inputSizeZ) {
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            const size_t idx =
                base + static_cast<size_t>(item) * static_cast<size_t>(threadsPerBlock);
            if (idx < inputSizeZ) {
                accumulate_byte(input_u8[idx], rangeStart, numBins, lane, s_hist);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 bank-striped copies back to one count per bin.
    // Each warp handles one bin at a time; lane j reads copy j, which is bank-conflict free.
    for (unsigned int bin = warpId; bin < numBins; bin += static_cast<unsigned int>(warpsPerBlock)) {
        unsigned int sum = s_hist[bin * sharedHistogramCopies + lane];

        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(fullWarpMask, sum, offset);
        }

        if (lane == 0u && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int rangeStart = static_cast<unsigned int>(from);
    const unsigned int numBins    = static_cast<unsigned int>(to - from + 1);

    // The contract is to compute a histogram, not accumulate into an existing one.
    // Use async memset because the caller explicitly owns synchronization.
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory footprint = numBins * 32 copies * 4 B.
    const size_t sharedBytes =
        static_cast<size_t>(numBins) *
        static_cast<size_t>(sharedHistogramCopies) *
        sizeof(unsigned int);

    // Use a persistent-style grid sized to the number of resident CTAs. This keeps all
    // SMs busy on large inputs while minimizing the number of final global histogram flushes.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_kernel,
        threadsPerBlock,
        sharedBytes);

    unsigned int maxResidentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSM);
    if (maxResidentBlocks == 0u) {
        maxResidentBlocks = 1u;
    }

    const unsigned int neededBlocks = static_cast<unsigned int>(
        (static_cast<size_t>(inputSize) + itemsPerBlock - 1) / itemsPerBlock);

    const unsigned int gridSize =
        (neededBlocks < maxResidentBlocks) ? neededBlocks : maxResidentBlocks;

    // Default stream launch; the function signature does not expose a stream parameter,
    // and the caller is responsible for any synchronization/error handling policy.
    histogram_kernel<<<gridSize, threadsPerBlock, sharedBytes, 0>>>(
        input, histogram, inputSize, rangeStart, numBins);
}