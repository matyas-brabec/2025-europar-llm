#include <cuda_runtime.h>
#include <cstddef>

namespace {

// Tuned for recent data-center GPUs (A100/H100 class):
// - 256 threads/block gives 8 warps, enough latency hiding without creating excessive
//   cross-warp contention on the 32 shared-memory histogram copies.
// - itemsPerThread = 16 means each block processes a 4 KiB tile per iteration
//   (256 * 16 bytes), which is a good balance between amortizing block setup/final merge
//   and keeping register pressure modest.
constexpr int threadsPerBlock = 256;
constexpr int histogramCopies = 32;
constexpr int itemsPerThread = 16;

constexpr unsigned int warpsPerBlock =
    static_cast<unsigned int>(threadsPerBlock / 32);
constexpr unsigned int workPerBlock =
    static_cast<unsigned int>(threadsPerBlock * itemsPerThread);
constexpr std::size_t fullTileLastItemOffset =
    static_cast<std::size_t>(threadsPerBlock) *
    static_cast<std::size_t>(itemsPerThread - 1);
constexpr unsigned int fullWarpMask = 0xFFFFFFFFu;

static_assert(histogramCopies == 32,
              "This bank-conflict-avoidance scheme requires 32 histogram copies.");
static_assert(threadsPerBlock % 32 == 0,
              "threadsPerBlock must be a multiple of the warp size.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
    // Standard tree reduction within one warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(fullWarpMask, value, offset);
    }
    return value;
}

// Shared-memory layout:
//   bin i, copy c  ->  s_hist[i * 32 + c]
//
// This layout is chosen so that lane c always hits bank c:
// - During updates, each thread uses copy = threadIdx.x % 32, so every lane in a warp
//   accesses a distinct bank regardless of which bin it updates.
// - Shared atomics are still needed because different warps in the same block reuse the
//   same 32 copies.
// - During the final reduction, one warp reduces one bin at a time; lane c reads copy c,
//   again producing conflict-free bank access.
__launch_bounds__(threadsPerBlock)
__global__ void histogram_kernel(const unsigned char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 unsigned int from,
                                 unsigned int range) {
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warpId = tid >> 5;

    // Pointer to this thread's private bank-aligned "column" through all bins:
    // myHist[bin * 32] == s_hist[bin * 32 + lane]
    unsigned int* const myHist = s_hist + lane;

    // Zero the shared histogram copies cooperatively.
    const unsigned int sharedEntries = range * histogramCopies;
    for (unsigned int i = tid; i < sharedEntries; i += threadsPerBlock) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride over block-sized tiles. Keeping only about one resident wave of blocks
    // (chosen by the host launcher) lets each block reuse its shared-memory histogram
    // across many tiles and emit only one global merge at the end.
    const std::size_t gridStride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(workPerBlock);
    std::size_t base =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(workPerBlock) +
        static_cast<std::size_t>(tid);

    // Fast path for full tiles: all items are in bounds, so the inner loop has no bounds checks.
    // The bin test uses unsigned subtraction:
    //   bin = value - from;
    //   if (bin < range) ...
    // This folds both lower and upper bound checks into one comparison.
    for (; base + fullTileLastItemOffset < inputSize; base += gridStride) {
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            const unsigned int value =
                static_cast<unsigned int>(input[base + static_cast<std::size_t>(item) * threadsPerBlock]);
            const unsigned int bin = value - from;
            if (bin < range) {
                atomicAdd(&myHist[bin * histogramCopies], 1u);
            }
        }
    }

    // Tail path: only the last partial tile needs bounds checks.
    for (; base < inputSize; base += gridStride) {
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            const std::size_t idx =
                base + static_cast<std::size_t>(item) * threadsPerBlock;
            if (idx < inputSize) {
                const unsigned int value = static_cast<unsigned int>(input[idx]);
                const unsigned int bin = value - from;
                if (bin < range) {
                    atomicAdd(&myHist[bin * histogramCopies], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 shared copies into the final global histogram.
    // One warp reduces one bin at a time:
    //   lane c reads copy c of the bin, then the warp sums across lanes.
    for (unsigned int bin = warpId; bin < range; bin += warpsPerBlock) {
        const unsigned int* const binBase = s_hist + bin * histogramCopies;
        unsigned int sum = binBase[lane];
        sum = warp_reduce_sum(sum);

        if (lane == 0 && sum != 0u) {
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
    // The problem statement guarantees valid arguments, but a cheap guard avoids undefined
    // behavior if the function is misused.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int range = static_cast<unsigned int>(to - from + 1);

    // The kernel only accumulates into the output, so clear it first.
    // The caller requested to handle host/device synchronization externally, so use the
    // default stream asynchronously and do not synchronize here.
    cudaMemsetAsync(histogram,
                    0,
                    static_cast<std::size_t>(range) * sizeof(unsigned int),
                    0);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory for 32 copies of the requested histogram range:
    //   range bins * 32 copies * sizeof(unsigned int)
    const std::size_t sharedBytes =
        static_cast<std::size_t>(range) *
        static_cast<std::size_t>(histogramCopies) *
        sizeof(unsigned int);

    // This kernel is dominated by shared-memory histogram privatization plus streaming reads,
    // so prefer the maximum shared-memory carveout.
    (void)cudaFuncSetAttribute(histogram_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    // Launch about one resident wave of blocks. For this kernel that is preferable to
    // launching one short-lived block per input tile, because each block initializes and
    // later merges its private shared histogram only once.
    int device = 0;
    int numSms = 1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int queriedSms = 0;
        if (cudaDeviceGetAttribute(&queriedSms,
                                   cudaDevAttrMultiProcessorCount,
                                   device) == cudaSuccess &&
            queriedSms > 0) {
            numSms = queriedSms;
        }
    }

    int residentBlocksPerSm = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentBlocksPerSm,
                                                      histogram_kernel,
                                                      threadsPerBlock,
                                                      sharedBytes) != cudaSuccess ||
        residentBlocksPerSm < 1) {
        residentBlocksPerSm = 1;
    }

    // Use an overflow-safe ceil_div because inputSize is 32-bit.
    const unsigned int blocksForInput = 1u + (inputSize - 1u) / workPerBlock;

    unsigned int gridSize =
        static_cast<unsigned int>(residentBlocksPerSm) *
        static_cast<unsigned int>(numSms);
    if (gridSize == 0u) {
        gridSize = 1u;
    }
    if (gridSize > blocksForInput) {
        gridSize = blocksForInput;
    }

    histogram_kernel<<<gridSize, threadsPerBlock, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        range);
}