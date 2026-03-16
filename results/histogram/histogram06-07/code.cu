#include <cuda_runtime.h>

/*
Range-restricted histogram for device-resident text.

Design:
- Each CTA builds a private histogram in shared memory.
- Every bin is replicated 32 times: one copy per warp lane / shared-memory bank.
- Shared layout is [bin][copy], i.e. blockHistogram[bin * bankStride + lane].
  With bankStride == 32 words, copy 'lane' always maps to bank 'lane', so the
  hot update path is free of intra-warp bank conflicts regardless of the bin.
- Threads from different warps but the same lane still share the same copy, so
  shared atomicAdd is required; replication reduces contention by 32x versus a
  single CTA-private histogram.
- itemsPerThread = 16 gives one aligned uint4 load per thread, which is a good
  balance of load efficiency, loop overhead, register pressure, and occupancy on
  modern data-center GPUs such as A100/H100.
*/

constexpr int histogramCopies = 32;
constexpr int bankStride = 32;   // One full shared-memory bank cycle in 4-byte words.
constexpr int vectorBytes = 16;  // uint4 load width in bytes.
constexpr int blockSize = 256;
constexpr int itemsPerThread = 16;

static_assert(histogramCopies == 32, "This implementation requires 32 shared histogram copies.");
static_assert(bankStride == 32, "This implementation relies on a 32-word shared-memory stride.");
static_assert(histogramCopies == bankStride, "Each copy must stay in a fixed shared-memory bank.");
static_assert(sizeof(unsigned int) == 4, "Shared-memory bank mapping assumes 4-byte counters.");
static_assert(blockSize > 0 && (blockSize % histogramCopies) == 0,
              "blockSize must be a positive multiple of 32.");
static_assert(itemsPerThread > 0 && (itemsPerThread % vectorBytes) == 0,
              "itemsPerThread must be a positive multiple of 16 for aligned uint4 loads.");

__device__ __forceinline__
void add_byte_to_lane_histogram(const unsigned int byteValue,
                                unsigned int* __restrict__ laneHistogram,
                                const unsigned int from,
                                const unsigned int numBins)
{
    const unsigned int bin = byteValue - from;
    if (bin < numBins) {
        atomicAdd(&laneHistogram[bin * bankStride], 1u);
    }
}

__device__ __forceinline__
void add_packed_word_to_lane_histogram(const unsigned int packedBytes,
                                       unsigned int* __restrict__ laneHistogram,
                                       const unsigned int from,
                                       const unsigned int numBins)
{
    add_byte_to_lane_histogram((packedBytes      ) & 0xFFu, laneHistogram, from, numBins);
    add_byte_to_lane_histogram((packedBytes >>  8) & 0xFFu, laneHistogram, from, numBins);
    add_byte_to_lane_histogram((packedBytes >> 16) & 0xFFu, laneHistogram, from, numBins);
    add_byte_to_lane_histogram((packedBytes >> 24) & 0xFFu, laneHistogram, from, numBins);
}

__device__ __forceinline__
void add_packed16_to_lane_histogram(const uint4 packed16,
                                    unsigned int* __restrict__ laneHistogram,
                                    const unsigned int from,
                                    const unsigned int numBins)
{
    add_packed_word_to_lane_histogram(packed16.x, laneHistogram, from, numBins);
    add_packed_word_to_lane_histogram(packed16.y, laneHistogram, from, numBins);
    add_packed_word_to_lane_histogram(packed16.z, laneHistogram, from, numBins);
    add_packed_word_to_lane_histogram(packed16.w, laneHistogram, from, numBins);
}

__device__ __forceinline__
unsigned int warp_reduce_sum(unsigned int value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE)
void range_histogram_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int numBins)
{
    extern __shared__ unsigned int blockHistogram[];

    const unsigned int lane = threadIdx.x & (histogramCopies - 1);
    const unsigned int warpId = threadIdx.x / histogramCopies;
    constexpr unsigned int warpsPerBlock = BLOCK_SIZE / histogramCopies;

    // laneHistogram is this thread's per-lane shared copy:
    // laneHistogram[bin * bankStride] == blockHistogram[bin * bankStride + lane].
    unsigned int* const laneHistogram = blockHistogram + lane;

    // Clear the CTA-private shared histogram.
    const unsigned int sharedCount = numBins * histogramCopies;
    for (unsigned int i = threadIdx.x; i < sharedCount; i += BLOCK_SIZE) {
        blockHistogram[i] = 0u;
    }
    __syncthreads();

    constexpr int vectorsPerThread = ITEMS_PER_THREAD / vectorBytes;
    const size_t inputSize64 = static_cast<size_t>(inputSize);
    const size_t blockWork = static_cast<size_t>(BLOCK_SIZE) * ITEMS_PER_THREAD;
    const size_t gridWork = static_cast<size_t>(gridDim.x) * blockWork;
    const size_t threadOffset = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD;

    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * blockWork;
         blockBase < inputSize64;
         blockBase += gridWork) {
        const size_t threadBase = blockBase + threadOffset;

        if (threadBase + ITEMS_PER_THREAD <= inputSize64) {
            // input is cudaMalloc-aligned and threadBase is a multiple of 16,
            // so this vectorized path uses aligned uint4 loads.
            const uint4* inputVec = reinterpret_cast<const uint4*>(input + threadBase);

            #pragma unroll
            for (int v = 0; v < vectorsPerThread; ++v) {
                add_packed16_to_lane_histogram(inputVec[v], laneHistogram, from, numBins);
            }
        } else if (threadBase < inputSize64) {
            // Only the very last partial chunk falls back to scalar byte loads.
            const size_t remaining = inputSize64 - threadBase;

            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (static_cast<size_t>(i) < remaining) {
                    add_byte_to_lane_histogram(
                        static_cast<unsigned int>(input[threadBase + static_cast<size_t>(i)]),
                        laneHistogram,
                        from,
                        numBins);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 lane-private copies. Each warp handles one bin at a time:
    // lane l reads copy l, which keeps this reduction bank-conflict free too.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        unsigned int sum = blockHistogram[bin * bankStride + lane];
        sum = warp_reduce_sum(sum);

        if (lane == 0u && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int fromValue = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);
    const size_t sharedBytes = static_cast<size_t>(numBins) * histogramCopies * sizeof(unsigned int);

    // The kernel accumulates per-CTA partial results into the global histogram,
    // so the output must be zeroed first. The API has no stream parameter, so
    // both operations use stream 0; synchronization/error handling is left to
    // the caller as requested.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 1;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int residentBlocksPerSM = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &residentBlocksPerSM,
        range_histogram_kernel<blockSize, itemsPerThread>,
        blockSize,
        sharedBytes);
    if (residentBlocksPerSM < 1) {
        residentBlocksPerSM = 1;
    }

    const size_t blockWork = static_cast<size_t>(blockSize) * itemsPerThread;
    const size_t tiles = (static_cast<size_t>(inputSize) + blockWork - 1) / blockWork;
    const size_t maxResidentGrid = static_cast<size_t>(smCount) * static_cast<size_t>(residentBlocksPerSM);
    const unsigned int gridSize =
        static_cast<unsigned int>(tiles < maxResidentGrid ? tiles : maxResidentGrid);

    if (gridSize == 0u) {
        return;
    }

    // Launch only as many CTAs as can be resident concurrently. The kernel uses
    // a grid-stride loop, so more CTAs would mainly increase the number of final
    // global atomics without improving steady-state occupancy.
    range_histogram_kernel<blockSize, itemsPerThread><<<gridSize, blockSize, sharedBytes, 0>>>(
        reinterpret_cast<const unsigned char*>(input),  // treat chars as 0..255 ordinals
        histogram,
        inputSize,
        fromValue,
        numBins);
}