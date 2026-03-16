#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Tunables chosen for modern data-center GPUs (A100/H100 class):
// - itemsPerThread = 16:
//     one aligned 128-bit load per thread per chunk on cudaMalloc memory;
//     good amortization of block-private histogram setup/final merge without
//     noticeably increasing register pressure.
//   If retuning, prefer multiples of 16 to keep the 128-bit load path enabled.
// - blockSize = 256:
//     8 warps per CTA is a good balance between occupancy and cross-warp
//     contention on the 32 lane-private shared histograms.
// - histogramCopies = 32:
//     one logical histogram copy per warp lane, matching the 32 shared-memory banks.
constexpr int itemsPerThread  = 16;
constexpr int blockSize       = 256;
constexpr int histogramCopies = 32;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(blockSize % 32 == 0, "blockSize must be a multiple of warp size.");
static_assert(histogramCopies == 32, "The bank-avoidance scheme assumes exactly 32 copies.");

__device__ __forceinline__ void update_lane_private_histogram(
    unsigned int value,
    unsigned int from_u,
    unsigned int range,
    unsigned int* __restrict__ lanePrivateHistogram)
{
    // Unsigned subtraction turns the inclusive test value in [from, from + range - 1]
    // into a single compare. If value < from_u, the subtraction underflows and the
    // result is >= range, so the branch is still rejected correctly.
    const unsigned int bin = value - from_u;
    if (bin < range) {
        // lanePrivateHistogram is sHist + lane. The stride between consecutive bins of
        // the same logical copy is 32 words, so the actual shared address is:
        //   sHist[bin * 32 + lane]
        //
        // Shared-memory bank = address % 32 for 4-byte words, therefore bank = lane
        // regardless of bin. A warp can update arbitrary bins without intra-warp bank
        // conflicts. Different warps reuse the same 32 lane-private copies, so only
        // inter-warp collisions remain; those are handled by fast shared-memory atomics.
        atomicAdd(&lanePrivateHistogram[bin * histogramCopies], 1u);
    }
}

__device__ __forceinline__ void consume_packed_u32(
    uint32_t packed,
    unsigned int from_u,
    unsigned int range,
    unsigned int* __restrict__ lanePrivateHistogram)
{
    update_lane_private_histogram( packed        & 0xFFu, from_u, range, lanePrivateHistogram);
    update_lane_private_histogram((packed >>  8) & 0xFFu, from_u, range, lanePrivateHistogram);
    update_lane_private_histogram((packed >> 16) & 0xFFu, from_u, range, lanePrivateHistogram);
    update_lane_private_histogram((packed >> 24) & 0xFFu, from_u, range, lanePrivateHistogram);
}

template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(blockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from_u,
                            unsigned int range)
{
    extern __shared__ unsigned int sHist[];

    // Interleaved shared layout:
    //   [bin0_copy0, bin0_copy1, ... bin0_copy31,
    //    bin1_copy0, bin1_copy1, ... bin1_copy31, ...]
    //
    // Each warp lane uses the logical copy with the same index as its lane id.
    // This layout is stronger than storing 32 contiguous copies with padding:
    // the bank index depends only on lane, not on the bin value.
    const unsigned int lane = threadIdx.x & 31u;
    unsigned int* const lanePrivateHistogram = sHist + lane;

    const unsigned int sharedElementCount = range * histogramCopies;

    // Zero the block-private histogram.
    for (unsigned int i = threadIdx.x; i < sharedElementCount; i += blockDim.x) {
        sHist[i] = 0u;
    }
    __syncthreads();

    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);
    const unsigned int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = gridDim.x * blockDim.x;
    constexpr unsigned int itemsPerThreadU = static_cast<unsigned int>(ITEMS_PER_THREAD);

    // Full chunks are processed with compile-time-unrolled loads. For the default
    // ITEMS_PER_THREAD = 16 this becomes one aligned uint4 (128-bit) load per thread.
    // Because input is allocated with cudaMalloc and base = chunk * ITEMS_PER_THREAD,
    // the vectorized paths are aligned whenever ITEMS_PER_THREAD is a multiple of 16 or 4.
    const unsigned int fullChunks = inputSize / itemsPerThreadU;

    for (unsigned int chunk = globalThread; chunk < fullChunks; chunk += totalThreads) {
        const unsigned int base = chunk * itemsPerThreadU;

        if constexpr ((ITEMS_PER_THREAD % 16) == 0) {
            const uint4* const chunkPtr = reinterpret_cast<const uint4*>(inputBytes + base);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD / 16; ++i) {
                const uint4 v = chunkPtr[i];
                consume_packed_u32(v.x, from_u, range, lanePrivateHistogram);
                consume_packed_u32(v.y, from_u, range, lanePrivateHistogram);
                consume_packed_u32(v.z, from_u, range, lanePrivateHistogram);
                consume_packed_u32(v.w, from_u, range, lanePrivateHistogram);
            }
        } else if constexpr ((ITEMS_PER_THREAD % 4) == 0) {
            const uint32_t* const chunkPtr = reinterpret_cast<const uint32_t*>(inputBytes + base);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD / 4; ++i) {
                consume_packed_u32(chunkPtr[i], from_u, range, lanePrivateHistogram);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                update_lane_private_histogram(static_cast<unsigned int>(inputBytes[base + i]),
                                              from_u,
                                              range,
                                              lanePrivateHistogram);
            }
        }
    }

    // Handle the final partial chunk, if any. This tail is always < ITEMS_PER_THREAD bytes.
    const unsigned int tailStart = fullChunks * itemsPerThreadU;
    for (unsigned int idx = tailStart + globalThread; idx < inputSize; idx += totalThreads) {
        update_lane_private_histogram(static_cast<unsigned int>(inputBytes[idx]),
                                      from_u,
                                      range,
                                      lanePrivateHistogram);
    }

    __syncthreads();

    // Collapse the 32 shared copies into one per-block histogram and merge into global memory.
    // Only one global atomic per non-zero bin and block is emitted.
    for (unsigned int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0u;
        const unsigned int base = bin * histogramCopies;

        #pragma unroll
        for (int copy = 0; copy < histogramCopies; ++copy) {
            sum += sHist[base + copy];
        }

        if (sum != 0u) {
            if (gridDim.x == 1) {
                // Single-block launch: the global histogram was zeroed before launch,
                // so a plain store is sufficient.
                histogram[bin] = sum;
            } else {
                atomicAdd(&histogram[bin], sum);
            }
        }
    }
}

} // anonymous namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int range = static_cast<unsigned int>(to - from + 1);

    // The API contract is "compute histogram", not "accumulate into existing contents",
    // so clear the output first. This is asynchronous; the caller owns synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(range) * static_cast<size_t>(histogramCopies) * sizeof(unsigned int);

    // Grid-stride chunking means we do not need one CTA per chunk. Launch only the
    // number of CTAs that can reside concurrently; that keeps the SMs busy while
    // minimizing the number of per-block merges into the global histogram.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
                                                  histogram_range_kernel<itemsPerThread>,
                                                  blockSize,
                                                  sharedBytes);

    constexpr unsigned int itemsPerThreadU = static_cast<unsigned int>(itemsPerThread);
    const unsigned int fullChunks = inputSize / itemsPerThreadU;
    const unsigned int blocksForOnePass =
        (fullChunks == 0u) ? 1u : (1u + (fullChunks - 1u) / static_cast<unsigned int>(blockSize));

    int gridSize = activeBlocksPerSM * smCount;
    if (gridSize < 1) {
        gridSize = 1;
    }
    if (blocksForOnePass < static_cast<unsigned int>(gridSize)) {
        gridSize = static_cast<int>(blocksForOnePass);
    }

    histogram_range_kernel<itemsPerThread><<<gridSize, blockSize, sharedBytes, 0>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        range);
}