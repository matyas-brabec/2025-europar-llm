#include <cuda_runtime.h>

namespace {

// Range-restricted byte histogram with block-private shared-memory accumulation.
// Input bytes are treated as raw unsigned ordinals [0, 255], independent of the
// signedness of plain 'char' on the host.
//
// Shared-memory privatization / bank-conflict avoidance:
//   * Each block owns 32 logical histogram copies: one copy per warp lane.
//   * Physical layout is transposed as shared[bin][laneCopy], flattened as
//       shared[(bin << histogramCopyShift) + laneCopy].
//   * Shared memory has 32 banks and the counters are 32-bit, so the bank index is
//       ((bin * 32) + laneCopy) % 32 == laneCopy.
//     Therefore every lane in a warp always hits a distinct bank, even when the
//     entire warp updates the same bin. This completely removes intra-warp bank
//     conflicts while preserving 32 independent copies per block as requested.
//
// Kernel tuning choices for modern data-center GPUs (A100/H100):
//   * blockThreads = 256 is a strong balance between occupancy and shared-atomic
//     contention within a block.
//   * itemsPerThread = 16 is a good large-input default: it amortizes loop/index
//     overhead and maps naturally to one aligned 16-byte vector load (uint4) per
//     thread on the hot path. If retuned, keeping it a multiple of 16 preserves
//     the vectorized load path automatically.
constexpr int histogramCopies    = 32;
constexpr int histogramCopyShift = 5;
constexpr int blockThreads       = 256;
constexpr int warpsPerBlock      = blockThreads / histogramCopies;
constexpr int itemsPerThread     = 16;

constexpr int vectorBytes = static_cast<int>(sizeof(uint4));  // 16 bytes
constexpr int vectorLoads = itemsPerThread / vectorBytes;
constexpr bool useVectorLoads = (itemsPerThread % vectorBytes) == 0;

static_assert(histogramCopies == 32, "This implementation assumes one shared histogram copy per warp lane.");
static_assert((1 << histogramCopyShift) == histogramCopies, "histogramCopyShift must match histogramCopies.");
static_assert((blockThreads % histogramCopies) == 0, "blockThreads must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(vectorBytes == 16, "uint4 is expected to be 16 bytes.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ void add_byte(unsigned int* __restrict__ sharedHist,
                                         unsigned int lane,
                                         unsigned int byteValue,
                                         unsigned int from,
                                         unsigned int numBins) {
    const unsigned int bin = byteValue - from;
    if (bin < numBins) {
        atomicAdd(sharedHist + (bin << histogramCopyShift) + lane, 1u);
    }
}

__device__ __forceinline__ void add_word(unsigned int* __restrict__ sharedHist,
                                         unsigned int lane,
                                         unsigned int word,
                                         unsigned int from,
                                         unsigned int numBins) {
    add_byte(sharedHist, lane,  word        & 0xffu, from, numBins);
    add_byte(sharedHist, lane, (word >> 8)  & 0xffu, from, numBins);
    add_byte(sharedHist, lane, (word >> 16) & 0xffu, from, numBins);
    add_byte(sharedHist, lane, (word >> 24) & 0xffu, from, numBins);
}

__device__ __forceinline__ void add_uint4(unsigned int* __restrict__ sharedHist,
                                          unsigned int lane,
                                          uint4 v,
                                          unsigned int from,
                                          unsigned int numBins) {
    add_word(sharedHist, lane, v.x, from, numBins);
    add_word(sharedHist, lane, v.y, from, numBins);
    add_word(sharedHist, lane, v.z, from, numBins);
    add_word(sharedHist, lane, v.w, from, numBins);
}

__global__ __launch_bounds__(blockThreads)
void histogram_kernel(const unsigned char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      unsigned int inputSize,
                      unsigned int from,
                      unsigned int numBins) {
    extern __shared__ unsigned int sharedHist[];

    const unsigned int lane   = threadIdx.x & (histogramCopies - 1);
    const unsigned int warpId = threadIdx.x >> histogramCopyShift;

    const size_t inputSizeZ = static_cast<size_t>(inputSize);
    const size_t sharedCount = static_cast<size_t>(numBins) * histogramCopies;

    // Zero the block-private shared histogram copies.
    for (size_t i = threadIdx.x; i < sharedCount; i += blockThreads) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const size_t tileSize    = static_cast<size_t>(blockThreads) * itemsPerThread;
    const size_t gridStride  = static_cast<size_t>(gridDim.x) * tileSize;
    const size_t threadBase0 = static_cast<size_t>(threadIdx.x) * itemsPerThread;

    // Grid-stride over block-sized tiles. Each thread consumes itemsPerThread bytes
    // from its tile, accumulating into the block-private shared histogram.
    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * tileSize;
         blockBase < inputSizeZ;
         blockBase += gridStride) {
        const size_t threadBase = blockBase + threadBase0;
        if (threadBase >= inputSizeZ) {
            continue;
        }

        const size_t remaining = inputSizeZ - threadBase;
        const unsigned char* threadPtr = input + threadBase;

        if (useVectorLoads) {
            if (remaining >= static_cast<size_t>(itemsPerThread)) {
                // Fast path: full per-thread chunk. With the prompt's cudaMalloc
                // contract and itemsPerThread == 16 by default, this becomes one
                // aligned uint4 load per thread.
                const uint4* vectorPtr = reinterpret_cast<const uint4*>(threadPtr);
                #pragma unroll
                for (int chunk = 0; chunk < vectorLoads; ++chunk) {
                    add_uint4(sharedHist, lane, vectorPtr[chunk], from, numBins);
                }
            } else {
                // Final partial chunk.
                #pragma unroll
                for (int i = 0; i < itemsPerThread; ++i) {
                    if (static_cast<size_t>(i) < remaining) {
                        add_byte(sharedHist, lane, threadPtr[i], from, numBins);
                    }
                }
            }
        } else {
            // Scalar fallback for non-16-byte-multiple retunings of itemsPerThread.
            if (remaining >= static_cast<size_t>(itemsPerThread)) {
                #pragma unroll
                for (int i = 0; i < itemsPerThread; ++i) {
                    add_byte(sharedHist, lane, threadPtr[i], from, numBins);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < itemsPerThread; ++i) {
                    if (static_cast<size_t>(i) < remaining) {
                        add_byte(sharedHist, lane, threadPtr[i], from, numBins);
                    }
                }
            }
        }
    }

    __syncthreads();

    // Fold the 32 lane-private copies back into one per-bin block result.
    // Each warp reduces one bin at a time across its 32 lanes, then atomically
    // accumulates into the final global histogram. This limits expensive global
    // atomics to one per non-zero bin per block.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        const unsigned int partial = sharedHist[(bin << histogramCopyShift) + lane];
        const unsigned int total = warp_reduce_sum(partial);
        if (lane == 0 && total != 0u) {
            atomicAdd(histogram + bin, total);
        }
    }
}

}  // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The output histogram is defined for exactly this invocation, so zero it first.
    // This is asynchronous; the caller explicitly handles synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(numBins) * histogramCopies * sizeof(unsigned int);

    // Launch only enough CTAs to fill all currently resident slots at this shared-
    // memory footprint. The kernel grid-strides over the rest of the input, so extra
    // CTAs would not increase steady-state occupancy; they would only increase the
    // final global-atomic flush traffic.
    int device = 0;
    int smCount = 1;
    int activeBlocksPerSm = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogram_kernel,
        blockThreads,
        sharedBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    const unsigned int tileItems = blockThreads * itemsPerThread;
    const unsigned int blocksNeeded = (inputSize + tileItems - 1u) / tileItems;
    const int maxResidentGrid = smCount * activeBlocksPerSm;

    int grid = static_cast<int>(blocksNeeded);
    if (grid < 1) {
        grid = 1;
    }
    if (grid > maxResidentGrid) {
        grid = maxResidentGrid;
    }

    // Reinterpret the char buffer as raw unsigned bytes so ordinal values are always
    // in the expected 0..255 range regardless of host-side char signedness.
    histogram_kernel<<<grid, blockThreads, sharedBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        numBins);
}