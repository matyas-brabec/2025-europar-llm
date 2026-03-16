#include <cuda_runtime.h>
#include <cstddef>

/*
  Range-restricted histogram over a device-resident char buffer.

  Output convention:
    histogram[i] counts occurrences of byte value (from + i), for i in [0, to-from].

  Optimization strategy for modern NVIDIA GPUs:
    - Each warp owns a private shared-memory histogram row to avoid cross-warp
      contention while scanning the input.
    - Within a warp, __match_any_sync() groups lanes that saw the same byte/bin,
      so one lane can issue a single shared-memory atomicAdd weighted by popc(peers).
      This is effective for text, where many lanes often see the same characters.
    - Input is read as uint4 (16 bytes per thread per iteration) to improve
      global-memory throughput.
    - Each block flushes its shared partials to global memory with at most one
      update per output bin, so global atomic pressure scales with
      (#blocks * #requested_bins), not with inputSize.
    - The byte domain is fixed at 256 values, so we reserve a full 256-bin shared
      histogram per warp. On A100/H100-class GPUs the 8 KiB/block cost is tiny
      and enables fixed-size, low-overhead indexing.
*/

namespace {

constexpr unsigned int kByteDomain          = 256u;
constexpr unsigned int kBlockSize           = kByteDomain;  // 1 thread per possible byte value during init/flush.
constexpr unsigned int kWarpSize            = 32u;
constexpr unsigned int kNumWarps            = kBlockSize / kWarpSize;  // 8 warps for 256 threads.
constexpr unsigned int kVectorBytes         = 16u;                     // uint4 load width.
constexpr unsigned int kInvalidBin          = 0xFFFFFFFFu;
constexpr unsigned int kTargetBlocksPerSM   = 8u;                      // 8 * 256 threads = 2048 threads/SM.

static_assert(kBlockSize % kWarpSize == 0u, "Block size must be a multiple of warp size.");
static_assert(kBlockSize == kByteDomain,
              "This implementation relies on one thread per byte value during shared-memory init/flush.");
static_assert(sizeof(uint4) == kVectorBytes, "kVectorBytes must match the vectorized load width.");

/*
  Accumulate one byte into the warp-private shared histogram.

  - rangeBase is `from`.
  - numBins is `(to - from + 1)`.
  - The local/output bin is (byte_value - rangeBase) when the byte is in range.
  - Out-of-range bytes use a sentinel key so every currently active lane can still
    participate in the same warp collective without branching beforehand.
*/
__device__ __forceinline__
void accumulate_byte(const unsigned int byte_value,
                     const unsigned int rangeBase,
                     const unsigned int numBins,
                     const unsigned int lane,
                     const unsigned int activeMask,
                     unsigned int* const warpHist)
{
    const unsigned int bin = byte_value - rangeBase;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const unsigned int key   = (bin < numBins) ? bin : kInvalidBin;
    const unsigned int peers = __match_any_sync(activeMask, key);

    if (key != kInvalidBin) {
        const unsigned int leader = static_cast<unsigned int>(__ffs(peers) - 1);
        if (lane == leader) {
            atomicAdd(warpHist + key, static_cast<unsigned int>(__popc(peers)));
        }
    }
#else
    // Fallback path for older compilation targets: still correct, just without
    // warp-level duplicate collapsing.
    (void)lane;
    (void)activeMask;
    if (bin < numBins) {
        atomicAdd(warpHist + bin, 1u);
    }
#endif
}

__device__ __forceinline__
void accumulate_word(const unsigned int word,
                     const unsigned int rangeBase,
                     const unsigned int numBins,
                     const unsigned int lane,
                     const unsigned int activeMask,
                     unsigned int* const warpHist)
{
    accumulate_byte((word      ) & 0xFFu, rangeBase, numBins, lane, activeMask, warpHist);
    accumulate_byte((word >>  8) & 0xFFu, rangeBase, numBins, lane, activeMask, warpHist);
    accumulate_byte((word >> 16) & 0xFFu, rangeBase, numBins, lane, activeMask, warpHist);
    accumulate_byte((word >> 24) & 0xFFu, rangeBase, numBins, lane, activeMask, warpHist);
}

__global__ __launch_bounds__(kBlockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int rangeBase,
                            const unsigned int numBins)
{
    // One shared histogram row per warp; rows are indexed by output-bin offset [0, 255].
    __shared__ unsigned int s_warpHist[kNumWarps][kByteDomain];

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & (kWarpSize - 1u);
    const unsigned int warpId = tid >> 5;

    // Fixed 256-thread block: every thread zeroes one byte-value column across all warp rows.
#pragma unroll
    for (unsigned int w = 0; w < kNumWarps; ++w) {
        s_warpHist[w][tid] = 0u;
    }
    __syncthreads();

    unsigned int* const myWarpHist = s_warpHist[warpId];

    // Read bytes as unsigned values; plain `char` may be signed, but histogram
    // bins are defined over ordinal values [0, 255].
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);
    const uint4* const inputVec = reinterpret_cast<const uint4*>(inputBytes);

    const unsigned int globalThread = blockIdx.x * kBlockSize + tid;
    const unsigned int totalThreads = gridDim.x * kBlockSize;

    const unsigned int vecCount = inputSize / kVectorBytes;
    for (unsigned int idx = globalThread; idx < vecCount; idx += totalThreads) {
        // All threads that reach this point participate in the same collectives for
        // this loop iteration, so one active mask is valid for all 16 bytes here.
        const unsigned int activeMask = __activemask();
        const uint4 v = inputVec[idx];

        accumulate_word(v.x, rangeBase, numBins, lane, activeMask, myWarpHist);
        accumulate_word(v.y, rangeBase, numBins, lane, activeMask, myWarpHist);
        accumulate_word(v.z, rangeBase, numBins, lane, activeMask, myWarpHist);
        accumulate_word(v.w, rangeBase, numBins, lane, activeMask, myWarpHist);
    }

    // Handle the final 0..15 leftover bytes. Only the first `tailCount` global
    // threads do work here, so each leftover byte is processed exactly once.
    const unsigned int tailBase  = vecCount * kVectorBytes;
    const unsigned int tailCount = inputSize - tailBase;
    if (globalThread < tailCount) {
        const unsigned int activeMask = __activemask();
        accumulate_byte(static_cast<unsigned int>(inputBytes[tailBase + globalThread]),
                        rangeBase, numBins, lane, activeMask, myWarpHist);
    }

    __syncthreads();

    // Reduce the 8 warp-private rows into the final output.
    // Only bins [0, numBins) are meaningful for this launch.
    if (tid < numBins) {
        unsigned int sum = 0u;
#pragma unroll
        for (unsigned int w = 0; w < kNumWarps; ++w) {
            sum += s_warpHist[w][tid];
        }

        // If the whole job fits in one block, this block owns the final answer and
        // can overwrite the output directly. Otherwise, accumulate block partials.
        if (gridDim.x == 1u) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

}  // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Per the API contract:
    //   - `input` and `histogram` are device pointers allocated with cudaMalloc.
    //   - 0 <= from < to <= 255.
    //   - The output buffer has exactly (to - from + 1) elements.
    //
    // This function only enqueues work. It does not synchronize; the caller owns
    // any host/device synchronization. Because the API has no stream parameter,
    // work is issued to the current default stream.

    const unsigned int rangeBase = static_cast<unsigned int>(from);
    const unsigned int numBins   = static_cast<unsigned int>(to - from + 1);

    // Empty input still requires a valid zero histogram.
    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, static_cast<std::size_t>(numBins) * sizeof(unsigned int));
        return;
    }

    // One thread processes one 16-byte vector item per loop iteration.
    const unsigned int vecWorkItems = inputSize / kVectorBytes;
    const unsigned int tailCount    = inputSize - vecWorkItems * kVectorBytes;
    const unsigned int workItems    = vecWorkItems + (tailCount != 0u ? 1u : 0u);
    const unsigned int maxBlocksNeeded = (workItems + kBlockSize - 1u) / kBlockSize;

    unsigned int gridSize = maxBlocksNeeded;

    // If more than 8 blocks are needed, cap the launch to ~8 CTAs/SM. With 256-thread
    // blocks this matches the 2048-thread/SM limit on modern data-center GPUs; extra
    // work is covered by the grid-stride loop.
    //
    // Cache the SM count per host thread/device pair so repeated launches avoid a
    // runtime attribute query in the hot path.
    if (gridSize > kTargetBlocksPerSM) {
        thread_local int cachedDevice  = -1;
        thread_local int cachedSMCount = 0;

        int device = 0;
        cudaGetDevice(&device);
        if (device != cachedDevice) {
            cachedSMCount = 0;
            cudaDeviceGetAttribute(&cachedSMCount, cudaDevAttrMultiProcessorCount, device);
            cachedDevice = device;
        }

        const unsigned int targetGrid =
            static_cast<unsigned int>(cachedSMCount) * kTargetBlocksPerSM;

        if (targetGrid != 0u && gridSize > targetGrid) {
            gridSize = targetGrid;
        }
    }

    if (gridSize == 0u) {
        gridSize = 1u;
    }

    // Multi-block launches accumulate partials with global atomicAdd, so the output
    // histogram must be zeroed first. Single-block launches overwrite every output
    // bin directly, so skipping memset avoids an extra runtime call on small inputs.
    if (gridSize != 1u) {
        cudaMemsetAsync(histogram, 0, static_cast<std::size_t>(numBins) * sizeof(unsigned int));
    }

    histogram_range_kernel<<<gridSize, kBlockSize>>>(input, histogram, inputSize, rangeBase, numBins);
}