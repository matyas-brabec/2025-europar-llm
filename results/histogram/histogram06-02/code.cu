#include <cuda_runtime.h>

// Range-restricted byte histogram for GPU-resident text buffers.
//
// Tuning rationale for A100/H100-class GPUs:
// - blockSize = 256: good balance between occupancy and shared-atomic contention.
// - itemsPerThread = 16: enough work per thread to amortize per-CTA setup/finalization
//   on large inputs, while keeping register pressure modest.
// - CTA-private shared histogram: absorbs the high-rate per-character increments.
// - 32 replicas per bin (one per warp lane): for bin b and lane l, the counter lives at
//   sharedHistogram[b * 32 + l]. This is conceptually 32 histograms with a stride of 32
//   counters between consecutive bins. A thread in lane l therefore always hits bank l,
//   eliminating intra-warp bank conflicts on updates.
// - Warp-level final reduction: one warp reduces the 32 replicas of a bin conflict-free
//   and emits a single global atomicAdd.
// - Worst-case shared footprint is 256 bins * 32 replicas * 4 B = 32 KB per CTA.

constexpr int blockSize = 256;
constexpr int minBlocksPerSm = 4;
constexpr int itemsPerThread = 16;

constexpr int histogramReplicas = 32;
constexpr int histogramReplicaShift = 5;
constexpr int warpsPerBlock = blockSize / histogramReplicas;

constexpr int vectorWidthBytes = 4;  // uchar4
constexpr int vectorsPerThread = itemsPerThread / vectorWidthBytes;
constexpr unsigned int fullWarpMask = 0xffffffffu;
constexpr unsigned int tileCharsPerBlock = static_cast<unsigned int>(blockSize * itemsPerThread);

static_assert(blockSize % histogramReplicas == 0, "blockSize must be a whole number of warps.");
static_assert(histogramReplicas == (1 << histogramReplicaShift), "Replica shift must match replica count.");
static_assert(itemsPerThread % vectorWidthBytes == 0, "itemsPerThread must be a multiple of 4 for uchar4 loads.");

static __device__ __forceinline__
void update_lane_private_hist(unsigned int* laneReplica,
                              unsigned int symbol,
                              unsigned int from,
                              unsigned int bins)
{
    const unsigned int bin = symbol - from;
    if (bin < bins) {
        // All warps in the CTA share the same 32 replicas, so shared-memory atomicAdd
        // is still required to resolve inter-warp collisions. Within a warp, however,
        // each lane touches a different bank because replica index == lane index.
        atomicAdd(laneReplica + (bin << histogramReplicaShift), 1u);
    }
}

static __device__ __forceinline__
void update_lane_private_hist4(unsigned int* laneReplica,
                               const uchar4 packed,
                               unsigned int from,
                               unsigned int bins)
{
    update_lane_private_hist(laneReplica, static_cast<unsigned int>(packed.x), from, bins);
    update_lane_private_hist(laneReplica, static_cast<unsigned int>(packed.y), from, bins);
    update_lane_private_hist(laneReplica, static_cast<unsigned int>(packed.z), from, bins);
    update_lane_private_hist(laneReplica, static_cast<unsigned int>(packed.w), from, bins);
}

static __device__ __forceinline__
unsigned int warp_reduce_sum(unsigned int value)
{
    #pragma unroll
    for (int offset = histogramReplicas / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(fullWarpMask, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(blockSize, minBlocksPerSm)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int bins)
{
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int lane = threadIdx.x & static_cast<unsigned int>(histogramReplicas - 1);
    const unsigned int warpId = threadIdx.x >> histogramReplicaShift;

    // Conceptually there are 32 per-CTA histograms. Physically they are interleaved by bin:
    //   [bin0: rep0..rep31][bin1: rep0..rep31]...
    // Replica `lane` is therefore the strided view sharedHistogram[lane + 32 * bin].
    unsigned int* const laneReplica = sharedHistogram + lane;

    // Treat the input as bytes so chars 128..255 are handled correctly even if `char` is signed.
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    const unsigned int sharedCounterCount = bins << histogramReplicaShift;

    // Cooperative zeroing of the CTA-private histogram.
    for (unsigned int i = threadIdx.x; i < sharedCounterCount; i += blockSize) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    const size_t tileSize = static_cast<size_t>(tileCharsPerBlock);
    const size_t gridStride = tileSize * gridDim.x;
    const size_t inputSize64 = static_cast<size_t>(inputSize);

    // Persistent grid-stride loop over 4 KB tiles (with the default tuning).
    for (size_t base = static_cast<size_t>(blockIdx.x) * tileSize; base < inputSize64; base += gridStride) {
        const size_t remaining = inputSize64 - base;
        const unsigned char* const tileInput = inputBytes + base;

        if (remaining >= tileSize) {
            // Full-tile fast path.
            // Because the buffer comes from cudaMalloc and base advances in multiples of tileSize,
            // tileInput is 4-byte aligned here, so uchar4 loads are safe.
            // Each vector iteration makes a warp read 32 consecutive uchar4 values = 128 bytes,
            // which is naturally coalesced.
            const uchar4* const vectorInput = reinterpret_cast<const uchar4*>(tileInput);

            #pragma unroll
            for (int v = 0; v < vectorsPerThread; ++v) {
                const uchar4 packed = vectorInput[v * blockSize + threadIdx.x];
                update_lane_private_hist4(laneReplica, packed, from, bins);
            }
        } else {
            // Tail cleanup for the last partial tile handled by this CTA.
            #pragma unroll
            for (int item = 0; item < itemsPerThread; ++item) {
                const unsigned int localIndex = threadIdx.x + item * blockSize;
                if (localIndex < remaining) {
                    update_lane_private_hist(
                        laneReplica,
                        static_cast<unsigned int>(tileInput[localIndex]),
                        from,
                        bins);
                }
            }
        }
    }

    __syncthreads();

    // Final CTA reduction. One warp reduces one bin at a time:
    // lane L reads replica L of the bin, so the shared-memory reads are again bank-conflict free.
    for (unsigned int bin = warpId; bin < bins; bin += warpsPerBlock) {
        const unsigned int partial = sharedHistogram[(bin << histogramReplicaShift) + lane];
        const unsigned int sum = warp_reduce_sum(partial);

        if (lane == 0u && sum != 0u) {
            // One global atomic add per bin per CTA.
            atomicAdd(histogram + bin, sum);
        }
    }
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int fromValue = static_cast<unsigned int>(from);
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // The output represents counts, not increments, so clear it before launching the kernel.
    // This is asynchronous with respect to the host and ordered before the kernel in stream 0.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(bins) * static_cast<size_t>(histogramReplicas) * sizeof(unsigned int);

    // Persistent-style launch: start roughly as many CTAs as can be resident, then let each CTA
    // grid-stride over the input. This minimizes the number of per-CTA histogram initializations
    // and final global reductions while still filling the GPU.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogram_range_kernel,
        blockSize,
        sharedBytes);

    const size_t numTiles =
        (static_cast<size_t>(inputSize) + static_cast<size_t>(tileCharsPerBlock) - 1u) /
        static_cast<size_t>(tileCharsPerBlock);

    const unsigned int sms = (smCount > 0) ? static_cast<unsigned int>(smCount) : 1u;
    const unsigned int residentBlocks =
        sms * ((activeBlocksPerSm > 0) ? static_cast<unsigned int>(activeBlocksPerSm) : 1u);

    const int gridSize = static_cast<int>(
        (numTiles < static_cast<size_t>(residentBlocks)) ? numTiles : residentBlocks);

    histogram_range_kernel<<<gridSize, blockSize, sharedBytes, 0>>>(
        input,
        histogram,
        inputSize,
        fromValue,
        bins);

    // No synchronization here by design; the caller owns stream/device synchronization.
}