#include <cuda_runtime.h>

/*
 * Range-restricted byte histogram for a device-resident text buffer.
 *
 * Optimization strategy:
 * 1) Shared-memory privatization:
 *    Each thread block accumulates into a block-private histogram in shared memory.
 *    That collapses expensive per-byte global atomics down to one global atomic add
 *    per (block, bin) during the final merge.
 *
 * 2) Bank-conflict-free shared histogram copies:
 *    The shared histogram is replicated 32 times (one replica per warp lane) and laid
 *    out as [bin][replica]. Each thread updates replica = lane_id, so the bank index is:
 *
 *        bank = (bin * 32 + lane_id) % 32 = lane_id
 *
 *    Therefore the update path is bank-conflict-free within a warp for arbitrary bin
 *    values. The replication also reduces atomic contention on hot bins by up to 32x
 *    versus a single shared histogram.
 *
 * 3) Vectorized input traversal:
 *    The input is read as uchar4 values so each thread processes 4 bytes per load.
 *    itemsPerThread defaults to 16 bytes/thread, which is a good Ampere/Hopper default:
 *    enough ILP to amortize loop/range-check overhead without inflating register pressure.
 *
 * 4) Occupancy-limited launch:
 *    run_histogram launches only enough blocks to fill the GPU. Because each block keeps
 *    its private histogram across the entire grid-stride traversal and merges only once,
 *    overlaunching would mainly increase the final global atomic traffic.
 */

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kCharsPerVector = 4;

// Tuning knob requested by the problem statement.
// Keep this a multiple of 4 so the vectorized uchar4 load path remains valid.
constexpr int itemsPerThread = 16;

constexpr unsigned int kWarpsPerBlock =
    static_cast<unsigned int>(kBlockSize / kWarpSize);
constexpr unsigned int kHistogramCopies =
    static_cast<unsigned int>(kWarpSize);
constexpr unsigned int kVectorsPerThread =
    static_cast<unsigned int>(itemsPerThread / kCharsPerVector);
constexpr unsigned int kBlockWork =
    static_cast<unsigned int>(kBlockSize * itemsPerThread);
constexpr unsigned int kVectorStrideBytes =
    static_cast<unsigned int>(kBlockSize * kCharsPerVector);
constexpr unsigned int kLaneMask = kHistogramCopies - 1u;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(kBlockSize % kWarpSize == 0, "kBlockSize must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(itemsPerThread % kCharsPerVector == 0,
              "itemsPerThread must be a multiple of 4 for vectorized uchar4 loads.");
static_assert(256u * kHistogramCopies * sizeof(unsigned int) <= 48u * 1024u,
              "The worst-case shared histogram must fit in the default dynamic shared-memory limit.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ void add_byte_to_private_histogram(
    unsigned char c,
    unsigned int from,
    unsigned int binCount,
    unsigned int* __restrict__ sharedHistogram,
    unsigned int replica)
{
    // Unsigned subtraction implements the two-sided range check in one comparison:
    //   c < from         -> underflow -> very large value -> rejected
    //   c > to           -> bin >= binCount             -> rejected
    //   from <= c <= to  -> 0 <= bin < binCount        -> accepted
    const unsigned int bin = static_cast<unsigned int>(c) - from;
    if (bin < binCount) {
        atomicAdd(&sharedHistogram[bin * kHistogramCopies + replica], 1u);
    }
}

__device__ __forceinline__ void add_vec4_to_private_histogram(
    uchar4 v,
    unsigned int from,
    unsigned int binCount,
    unsigned int* __restrict__ sharedHistogram,
    unsigned int replica)
{
    add_byte_to_private_histogram(v.x, from, binCount, sharedHistogram, replica);
    add_byte_to_private_histogram(v.y, from, binCount, sharedHistogram, replica);
    add_byte_to_private_histogram(v.z, from, binCount, sharedHistogram, replica);
    add_byte_to_private_histogram(v.w, from, binCount, sharedHistogram, replica);
}

__global__ __launch_bounds__(kBlockSize)
void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from,
    unsigned int binCount)
{
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & kLaneMask;
    const unsigned int warp = tid >> 5;

    const size_t inputSize64 = static_cast<size_t>(inputSize);
    const size_t blockWork = static_cast<size_t>(kBlockWork);
    const size_t gridWork = static_cast<size_t>(gridDim.x) * blockWork;
    const size_t vectorStrideBytes = static_cast<size_t>(kVectorStrideBytes);

    // Zero the block-private histogram copies.
    const unsigned int sharedCounters = binCount * kHistogramCopies;
    for (unsigned int i = tid; i < sharedCounters; i += static_cast<unsigned int>(kBlockSize)) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    // Grid-stride traversal. Each block retains its private histogram across all assigned
    // tiles and merges to global memory only once at the end.
    for (size_t blockBase = static_cast<size_t>(blockIdx.x) * blockWork;
         blockBase < inputSize64;
         blockBase += gridWork)
    {
        const size_t tileRemaining = inputSize64 - blockBase;

        if (tileRemaining >= blockWork) {
            // Fast path: a full tile. The pointer is 4-byte aligned because:
            // - cudaMalloc returns aligned storage,
            // - the API is assumed to pass the base allocation pointer,
            // - blockBase is a multiple of 4,
            // - each thread accesses tid * 4.
            const uchar4* tile4 = reinterpret_cast<const uchar4*>(input + blockBase);

            #pragma unroll
            for (unsigned int vec = 0; vec < kVectorsPerThread; ++vec) {
                const uchar4 v = tile4[vec * static_cast<unsigned int>(kBlockSize) + tid];
                add_vec4_to_private_histogram(v, from, binCount, sharedHistogram, lane);
            }
        } else {
            // Tail tile: keep the same blocked access pattern, but guard loads near the end.
            #pragma unroll
            for (unsigned int vec = 0; vec < kVectorsPerThread; ++vec) {
                const size_t idx =
                    blockBase +
                    static_cast<size_t>(vec) * vectorStrideBytes +
                    static_cast<size_t>(tid) * kCharsPerVector;

                if (idx < inputSize64) {
                    const size_t remaining = inputSize64 - idx;

                    if (remaining >= kCharsPerVector) {
                        const uchar4 v = *reinterpret_cast<const uchar4*>(input + idx);
                        add_vec4_to_private_histogram(v, from, binCount, sharedHistogram, lane);
                    } else {
                        add_byte_to_private_histogram(input[idx], from, binCount, sharedHistogram, lane);
                        if (remaining > 1) {
                            add_byte_to_private_histogram(input[idx + 1], from, binCount, sharedHistogram, lane);
                        }
                        if (remaining > 2) {
                            add_byte_to_private_histogram(input[idx + 2], from, binCount, sharedHistogram, lane);
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 replicas for each bin. One warp handles one bin at a time:
    // lane r reads replica r, then the warp sum is reduced with shuffles.
    for (unsigned int bin = warp; bin < binCount; bin += kWarpsPerBlock) {
        unsigned int sum = sharedHistogram[bin * kHistogramCopies + lane];
        sum = warp_reduce_sum(sum);

        if (lane == 0 && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

} // namespace

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // The API contract says input and histogram are device pointers from cudaMalloc.
    // This function only enqueues work on the default stream; the caller performs any
    // synchronization it wants after the launch.
    cudaStream_t stream = nullptr;

    const unsigned int binCount = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(binCount) * sizeof(unsigned int);
    const size_t sharedMemBytes =
        static_cast<size_t>(binCount) * kHistogramCopies * sizeof(unsigned int);
    const size_t blockWork = static_cast<size_t>(kBlockWork);

    // Compute a fresh histogram, not an accumulation.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u) {
        return;
    }

    // Bias the unified L1/shared-memory partition toward shared memory; this kernel is
    // dominated by shared-memory atomics and uses very little reusable L1 data.
    // This is a performance hint, not a correctness requirement.
    cudaFuncSetAttribute(
        histogram_range_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

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
        histogram_range_kernel,
        kBlockSize,
        sharedMemBytes);
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    // Because each block merges its private histogram only once, launch just enough blocks
    // to fill the GPU rather than creating unnecessary extra blocks that would only increase
    // the final global atomic traffic.
    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSm);
    const unsigned int blocksNeeded = static_cast<unsigned int>(
        (static_cast<size_t>(inputSize) + blockWork - 1u) / blockWork);
    const unsigned int gridSize =
        (blocksNeeded < maxResidentBlocks) ? blocksNeeded : maxResidentBlocks;

    histogram_range_kernel<<<static_cast<int>(gridSize), kBlockSize, sharedMemBytes, stream>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        binCount);

    // No synchronization here by design; the caller requested to own synchronization.
}