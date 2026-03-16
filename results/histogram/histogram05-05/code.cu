#include <cuda_runtime.h>
#include <cstdint>

/*
 * Range-restricted byte histogram for device-resident input.
 *
 * Output layout:
 *   histogram[i] == count of byte value (from + i)
 *
 * Important correctness detail:
 *   The input is interpreted as unsigned bytes, not as possibly signed C++ char.
 *   This makes ranges in [128, 255] work correctly on platforms where plain
 *   char is signed.
 *
 * Performance strategy:
 *   1) Each block builds a private histogram in shared memory.
 *   2) The shared histogram is replicated 32 times per block (one logical copy
 *      per warp lane / shared-memory bank).
 *   3) Lane N always updates copy N. The per-copy stride is forced to be odd
 *      (numBins or numBins + 1), so the same bin across the 32 copies maps to
 *      the 32 banks instead of colliding on a subset of banks.
 *   4) After scanning its portion of the input, the block merges the 32 copies
 *      and performs only one global atomic add per output bin.
 *   5) Full tiles use uchar4 vector loads; unaligned pointers or the final
 *      partial tile fall back to scalar loads.
 */
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
/*
 * Chosen for A100/H100-class GPUs:
 *   - 256 threads = 8 warps/block, a strong balance between occupancy and
 *     intra-block shared-atomic contention.
 *   - 16 bytes/thread gives enough ILP to amortize setup and shared-memory
 *     initialization while keeping register pressure low.
 *   - 16 is also a natural fit for four uchar4 vector loads.
 */
constexpr int itemsPerThread = 16;
constexpr int kSharedHistogramCopies = kWarpSize;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kVectorWidth = 4;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of 32.");
static_assert(kSharedHistogramCopies == kWarpSize,
              "This implementation uses one shared histogram copy per warp lane.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert((itemsPerThread % kVectorWidth) == 0,
              "itemsPerThread must be a multiple of 4 for the vectorized path.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value)
{
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ void accumulate_byte(
    const unsigned int value,
    const unsigned int rangeBegin,
    const unsigned int numBins,
    unsigned int* myHistogram)
{
    /*
     * Unsigned subtraction turns the inclusive range test
     *   rangeBegin <= value <= rangeBegin + numBins - 1
     * into a single comparison:
     *   bin = value - rangeBegin; if (bin < numBins) ...
     *
     * Values below rangeBegin wrap around and therefore fail the comparison.
     */
    const unsigned int bin = value - rangeBegin;
    if (bin < numBins) {
        atomicAdd(myHistogram + bin, 1u);
    }
}

__device__ __forceinline__ void accumulate_uchar4(
    const uchar4 v,
    const unsigned int rangeBegin,
    const unsigned int numBins,
    unsigned int* myHistogram)
{
    accumulate_byte(static_cast<unsigned int>(v.x), rangeBegin, numBins, myHistogram);
    accumulate_byte(static_cast<unsigned int>(v.y), rangeBegin, numBins, myHistogram);
    accumulate_byte(static_cast<unsigned int>(v.z), rangeBegin, numBins, myHistogram);
    accumulate_byte(static_cast<unsigned int>(v.w), rangeBegin, numBins, myHistogram);
}

__global__ __launch_bounds__(kBlockThreads)
void histogram_kernel(const char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      const unsigned int inputSize,
                      const unsigned int rangeBegin,
                      const unsigned int numBins)
{
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (kWarpSize - 1u);
    const unsigned int warpId = tid >> 5;

    /*
     * Force an odd stride for each replicated histogram:
     *   even numBins -> numBins + 1
     *   odd  numBins -> numBins
     *
     * With 32 banks, any odd stride is relatively prime to 32. Therefore,
     * "same bin across the 32 copies" becomes a full permutation of the banks,
     * which removes the worst bank-conflict pattern. Padding cost is at most
     * one counter per copy.
     */
    const unsigned int copyStride = numBins | 1u;
    const unsigned int totalSharedWords = kSharedHistogramCopies * copyStride;

    // Zero the block-private shared histograms.
    for (unsigned int i = tid; i < totalSharedWords; i += kBlockThreads) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Interpret input as unsigned bytes so values 128..255 are handled correctly.
    const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(input);

    // Lane N always updates copy N.
    const unsigned int myHistBase = lane * copyStride;
    unsigned int* const myHistogram = s_hist + myHistBase;

    const size_t totalCount = static_cast<size_t>(inputSize);
    const size_t tileSize =
        static_cast<size_t>(kBlockThreads) * static_cast<size_t>(itemsPerThread);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * tileSize;

    size_t tileStart = static_cast<size_t>(blockIdx.x) * tileSize;

    /*
     * Vectorized full-tile path.
     *
     * uchar4 loads require 4-byte alignment for defined behavior. cudaMalloc
     * allocations are naturally aligned, but this runtime check keeps the kernel
     * correct even if the caller passes a shifted device pointer into a larger
     * allocation.
     */
    const bool canVectorize =
        ((reinterpret_cast<std::uintptr_t>(bytes) &
          static_cast<std::uintptr_t>(kVectorWidth - 1)) == 0);

    if (canVectorize && totalCount >= tileSize) {
        constexpr int kVectorLoadsPerThread = itemsPerThread / kVectorWidth;
        const size_t vectorSpan =
            static_cast<size_t>(kBlockThreads) * static_cast<size_t>(kVectorWidth);
        const size_t lastFullTileStart = totalCount - tileSize;

        for (; tileStart <= lastFullTileStart; tileStart += gridStride) {
            #pragma unroll
            for (int v = 0; v < kVectorLoadsPerThread; ++v) {
                const size_t idx =
                    tileStart +
                    static_cast<size_t>(v) * vectorSpan +
                    static_cast<size_t>(tid) * static_cast<size_t>(kVectorWidth);

                const uchar4 data = *reinterpret_cast<const uchar4*>(bytes + idx);
                accumulate_uchar4(data, rangeBegin, numBins, myHistogram);
            }
        }
    }

    // Scalar path for the final partial tile, or for all tiles if vectorization is not possible.
    for (; tileStart < totalCount; tileStart += gridStride) {
        const size_t base = tileStart + static_cast<size_t>(tid);

        #pragma unroll
        for (int i = 0; i < itemsPerThread; ++i) {
            const size_t idx =
                base + static_cast<size_t>(i) * static_cast<size_t>(kBlockThreads);
            if (idx < totalCount) {
                accumulate_byte(static_cast<unsigned int>(bytes[idx]),
                                rangeBegin,
                                numBins,
                                myHistogram);
            }
        }
    }

    __syncthreads();

    /*
     * Warp-parallel merge of the 32 replicated shared histograms:
     *   - lane N reads copy N for the current bin,
     *   - the warp sums those 32 partial counts with shuffles,
     *   - lane 0 performs the final global atomic add.
     *
     * That keeps global atomic traffic down to one atomic per (block, bin) pair
     * instead of one atomic per matching input byte.
     */
    for (unsigned int bin = warpId; bin < numBins; bin += kWarpsPerBlock) {
        const unsigned int partial = s_hist[lane * copyStride + bin];
        const unsigned int sum = warp_reduce_sum(partial);

        if (lane == 0u && sum != 0u) {
            atomicAdd(histogram + bin, sum);
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
    /*
     * Contract:
     *   - input and histogram are device pointers allocated by cudaMalloc.
     *   - histogram has exactly (to - from + 1) counters.
     *   - this function intentionally does not synchronize; the caller owns
     *     stream ordering / completion handling.
     */
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int rangeBegin = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Overwrite only the compact output range requested by the caller.
    (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(kSharedHistogramCopies) *
        static_cast<size_t>(numBins | 1u) *
        sizeof(unsigned int);

    /*
     * The kernel uses a grid-stride loop, so there is no benefit in launching
     * more blocks than the GPU can keep resident. Extra blocks would only add
     * more final global-atomic merge work.
     */
    int device = 0;
    int smCount = 1;
    int activeBlocksPerSm = 1;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSm,
                                                        histogram_kernel,
                                                        kBlockThreads,
                                                        sharedBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    const unsigned int tileSize =
        static_cast<unsigned int>(kBlockThreads) * static_cast<unsigned int>(itemsPerThread);

    // inputSize > 0 here, so this overflow-safe form is valid.
    const unsigned int numTiles = 1u + (inputSize - 1u) / tileSize;
    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSm);
    const unsigned int gridSize =
        (numTiles < maxResidentBlocks) ? numTiles : maxResidentBlocks;

    histogram_kernel<<<gridSize, kBlockThreads, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        rangeBegin,
        numBins);
}