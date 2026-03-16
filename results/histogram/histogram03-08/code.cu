#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
static_assert((kBlockThreads % kWarpSize) == 0, "kBlockThreads must be a multiple of 32.");

constexpr unsigned int kMaxBins = 256u;

// One extra slot is reserved per warp-private histogram.
// - It acts as padding, so neighboring warp histograms are not separated by an exact
//   multiple of 32 banks.
// - It is also used as the out-of-range sentinel key for warp matching.
constexpr unsigned int kInvalidBin = kMaxBins;         // index 256
constexpr unsigned int kWarpHistStride = kMaxBins + 1u; // 257

constexpr std::size_t kVectorBytes = sizeof(uint4);
constexpr std::size_t kVectorAlignMask = kVectorBytes - 1u;

// Launch heuristic: bias slightly toward more CTAs on mid-size inputs.
// The per-block global merge cost is only one atomic add per output bin, so a
// moderately larger grid is usually worth it to keep the GPU busy.
constexpr unsigned int kLaunchBytesPerThread = 8u;
constexpr unsigned int kTargetBytesPerBlock =
    static_cast<unsigned int>(kBlockThreads) * kLaunchBytesPerThread;

// Collapse equal-bin lanes inside a warp before touching shared memory.
// Because each warp owns a private histogram segment, one elected leader lane can
// update shared memory with a plain increment instead of a shared-memory atomic.
//
// The explicit __syncwarp(activeMask) is required for correctness on Volta+
// independent thread scheduling: all participating lanes must reconverge before the
// next warp-wide primitive that uses the same active mask.
__device__ __forceinline__
void warp_aggregated_private_add(const unsigned int bin,
                                 const unsigned int numBins,
                                 const unsigned int activeMask,
                                 const unsigned int laneBit,
                                 unsigned int* const warpHist)
{
    const unsigned int key = (bin < numBins) ? bin : kInvalidBin;
    const unsigned int peers = __match_any_sync(activeMask, key);

    if (key < numBins && (peers & (laneBit - 1u)) == 0u) {
        warpHist[key] += __popc(peers);
    }

    __syncwarp(activeMask);
}

__device__ __forceinline__
void accumulate_packed_u32(const unsigned int word,
                           const unsigned int from,
                           const unsigned int numBins,
                           const unsigned int activeMask,
                           const unsigned int laneBit,
                           unsigned int* const warpHist)
{
    warp_aggregated_private_add(((word >>  0) & 0xFFu) - from, numBins, activeMask, laneBit, warpHist);
    warp_aggregated_private_add(((word >>  8) & 0xFFu) - from, numBins, activeMask, laneBit, warpHist);
    warp_aggregated_private_add(((word >> 16) & 0xFFu) - from, numBins, activeMask, laneBit, warpHist);
    warp_aggregated_private_add(((word >> 24) & 0xFFu) - from, numBins, activeMask, laneBit, warpHist);
}

// Per-block algorithm:
// 1) Zero warp-private histograms in shared memory.
// 2) Process the input with grid-stride loops. A short scalar prefix is peeled so
//    the main path can use aligned uint4 loads.
// 3) Reduce the warp-private histograms and merge one partial histogram per block
//    into the global output.
__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int from,
                            const unsigned int numBins)
{
    __shared__ unsigned int sharedHist[kWarpsPerBlock * kWarpHistStride];

    #pragma unroll
    for (unsigned int i = threadIdx.x; i < kWarpsPerBlock * kWarpHistStride; i += kBlockThreads) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned int laneBit = 1u << lane;
    const unsigned int warp = threadIdx.x / kWarpSize;
    unsigned int* const warpHist = sharedHist + warp * kWarpHistStride;

    const std::size_t globalThread = static_cast<std::size_t>(blockIdx.x) * kBlockThreads + threadIdx.x;
    const std::size_t totalThreads = static_cast<std::size_t>(gridDim.x) * kBlockThreads;

    const unsigned char* const inputBytes =
        reinterpret_cast<const unsigned char*>(input);
    const std::size_t totalBytes = static_cast<std::size_t>(inputSize);

    // Peel up to 15 bytes so the main uint4 path is always 16-byte aligned even if
    // the incoming char* is not.
    const std::size_t misalignment =
        static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(inputBytes)) & kVectorAlignMask;
    const std::size_t rawHeadBytes = (kVectorBytes - misalignment) & kVectorAlignMask;
    const std::size_t headBytes = (rawHeadBytes < totalBytes) ? rawHeadBytes : totalBytes;

    // Scalar prefix.
    for (std::size_t i = globalThread; i < headBytes; i += totalThreads) {
        const unsigned int activeMask = __activemask();
        const unsigned int bin = static_cast<unsigned int>(inputBytes[i]) - from;
        warp_aggregated_private_add(bin, numBins, activeMask, laneBit, warpHist);
    }

    // Main aligned vectorized body.
    const std::size_t remainingBytes = totalBytes - headBytes;
    const std::size_t vecCount = remainingBytes / kVectorBytes;
    const uint4* const input4 =
        reinterpret_cast<const uint4*>(inputBytes + headBytes);

    for (std::size_t vec = globalThread; vec < vecCount; vec += totalThreads) {
        const unsigned int activeMask = __activemask();
        const uint4 packed = input4[vec];

        accumulate_packed_u32(packed.x, from, numBins, activeMask, laneBit, warpHist);
        accumulate_packed_u32(packed.y, from, numBins, activeMask, laneBit, warpHist);
        accumulate_packed_u32(packed.z, from, numBins, activeMask, laneBit, warpHist);
        accumulate_packed_u32(packed.w, from, numBins, activeMask, laneBit, warpHist);
    }

    // Scalar tail.
    const std::size_t tailStart = headBytes + vecCount * kVectorBytes;
    for (std::size_t i = tailStart + globalThread; i < totalBytes; i += totalThreads) {
        const unsigned int activeMask = __activemask();
        const unsigned int bin = static_cast<unsigned int>(inputBytes[i]) - from;
        warp_aggregated_private_add(bin, numBins, activeMask, laneBit, warpHist);
    }

    __syncthreads();

    // Merge the per-warp private histograms.
    // With a multi-block launch we need global atomics, but only one per block/bin.
    const bool singleBlock = (gridDim.x == 1u);

    for (unsigned int bin = threadIdx.x; bin < numBins; bin += kBlockThreads) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHist[w * kWarpHistStride + bin];
        }

        if (sum != 0u) {
            if (singleBlock) {
                histogram[bin] = sum;
            } else {
                atomicAdd(histogram + bin, sum);
            }
        }
    }
}

static inline unsigned int ceil_div_u32(const unsigned int x, const unsigned int y)
{
    return (x + y - 1u) / y;
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The kernel merges block-private results with global atomic adds, so the
    // requested output range must be cleared first. This is intentionally async;
    // the caller owns host-device synchronization.
    (void)cudaMemsetAsync(histogram, 0, static_cast<std::size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    // The API does not expose a stream parameter, so all work is enqueued on the
    // default stream. Occupancy is queried at runtime so the launch adapts to the
    // actual target GPU and compiler register allocation.
    int device = 0;
    int smCount = 1;
    int activeBlocksPerSM = 1;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM, histogram_range_kernel, kBlockThreads, 0);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSM);

    const unsigned int logicalBlocks = ceil_div_u32(inputSize, kTargetBytesPerBlock);
    unsigned int grid = (logicalBlocks < maxResidentBlocks) ? logicalBlocks : maxResidentBlocks;
    if (grid == 0u) {
        grid = 1u;
    }

    histogram_range_kernel<<<grid, kBlockThreads>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        numBins);
}