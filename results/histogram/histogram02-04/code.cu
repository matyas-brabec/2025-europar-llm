#include <cuda_runtime.h>

namespace {

// Tuned for modern datacenter GPUs (A100/H100 class):
// - 256 threads/block lets the final block-local reduction map one thread to one
//   possible output bin (the range is at most 256 bins).
// - 4-way unrolling reduces loop/control overhead without creating excessive
//   register pressure for this histogram-style workload.
constexpr int kWarpSize        = 32;
constexpr int kBlockThreads    = 256;
constexpr int kItemsPerLoop    = 4;
constexpr int kWarpsPerBlock   = kBlockThreads / kWarpSize;

static_assert((kBlockThreads % kWarpSize) == 0, "Block size must be a multiple of 32.");
static_assert(kBlockThreads >= 256, "Block size must cover the maximum possible bin count.");

inline unsigned int ceil_div_u32(const unsigned int n, const unsigned int d) {
    return (n == 0u) ? 0u : (1u + (n - 1u) / d);
}

// Called collectively by the currently active lanes of a warp.
// Fast path for Volta+ (the intended target: A100/H100) uses:
//   1) a warp-private histogram slice in shared memory,
//   2) __match_any_sync() to collapse identical bytes inside the warp so only
//      one leader lane updates a given bin per step, and
//   3) __syncwarp() as the ordering primitive that makes plain shared-memory
//      increments safe across consecutive update steps.
//
// Important detail:
//   We synchronize the full active mask, not just the lanes that matched the
//   range, because a lane that is "invalid" for the current byte may become the
//   leader for the next byte and must observe all prior shared-memory updates.
__device__ __forceinline__
void accumulate_warp_private_byte(const unsigned int c,
                                  const unsigned int from,
                                  const unsigned int numBins,
                                  const unsigned int activeMask,
                                  unsigned int* const warpHistogram,
                                  const unsigned int lane)
{
    const unsigned int bin   = c - from;
    const bool         valid = (bin < numBins);

    const unsigned int validMask = __ballot_sync(activeMask, valid);

    // validMask is the same in every participating lane because it is produced
    // by the ballot itself, so this branch is warp-uniform for active lanes.
    if (validMask != 0u) {
        const unsigned int peerMask = __match_any_sync(validMask, bin);

        // Only one leader lane per distinct bin updates the warp-private slice.
        // Because each warp owns a disjoint slice of shared memory and because
        // __match_any_sync() guarantees one leader per distinct bin in this
        // update step, a plain increment is sufficient here.
        if (valid) {
            const unsigned int leader = static_cast<unsigned int>(__ffs(peerMask) - 1);
            if (lane == leader) {
                warpHistogram[bin] += static_cast<unsigned int>(__popc(peerMask));
            }
        }

        __syncwarp(activeMask);
    }
}

// Range-restricted histogram kernel.
// Input is treated as unsigned bytes (0..255) regardless of whether host-side
// char is signed or unsigned.
__global__ __launch_bounds__(kBlockThreads)
void histogram_char_range_kernel(const unsigned char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 const unsigned int inputSize,
                                 const unsigned int from,
                                 const unsigned int numBins)
{
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid    = static_cast<unsigned int>(threadIdx.x);
    const unsigned int warpId = tid >> 5;
    const unsigned int lane   = tid & 31u;

    // One private histogram per warp. This reduces contention versus a single
    // block-wide shared histogram, while still keeping the per-block footprint small.
    unsigned int* const warpHistogram = sharedHistogram + warpId * numBins;
    const unsigned int sharedBins = static_cast<unsigned int>(kWarpsPerBlock) * numBins;

    // Cooperative zero-initialization of all warp-private histograms.
    for (unsigned int i = tid; i < sharedBins; i += kBlockThreads) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    const unsigned int globalThread = static_cast<unsigned int>(blockIdx.x) * kBlockThreads + tid;
    const unsigned int totalThreads = static_cast<unsigned int>(gridDim.x) * kBlockThreads;

    // Unrolled grid-stride loop.
    // For a fixed unroll slot, consecutive lanes read consecutive bytes, so
    // accesses remain coalesced.
    unsigned int idx = globalThread;
    const unsigned int unrolledStep  = totalThreads * static_cast<unsigned int>(kItemsPerLoop);
    const unsigned int unrolledLimit =
        (inputSize > static_cast<unsigned int>(kItemsPerLoop - 1) * totalThreads)
            ? (inputSize - static_cast<unsigned int>(kItemsPerLoop - 1) * totalThreads)
            : 0u;

    for (; idx < unrolledLimit; idx += unrolledStep) {
        // The final trip can have a partially active warp, so the helper uses
        // the true active mask rather than assuming all 32 lanes participate.
        const unsigned int activeMask = __activemask();

        accumulate_warp_private_byte(static_cast<unsigned int>(input[idx]),
                                     from, numBins, activeMask, warpHistogram, lane);
        accumulate_warp_private_byte(static_cast<unsigned int>(input[idx + totalThreads]),
                                     from, numBins, activeMask, warpHistogram, lane);
        accumulate_warp_private_byte(static_cast<unsigned int>(input[idx + 2u * totalThreads]),
                                     from, numBins, activeMask, warpHistogram, lane);
        accumulate_warp_private_byte(static_cast<unsigned int>(input[idx + 3u * totalThreads]),
                                     from, numBins, activeMask, warpHistogram, lane);
    }

    // Scalar tail.
    for (; idx < inputSize; idx += totalThreads) {
        const unsigned int activeMask = __activemask();
        accumulate_warp_private_byte(static_cast<unsigned int>(input[idx]),
                                     from, numBins, activeMask, warpHistogram, lane);
    }

    __syncthreads();

    // Finalize this block's private histograms into the global output.
    // The requested range has at most 256 bins, so with 256 threads/block each
    // possible output bin can be handled by one thread.
    if (tid < numBins) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHistogram[static_cast<unsigned int>(w) * numBins + tid];
        }

        if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
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
    // Problem contract guarantees:
    //   0 <= from < to <= 255
    // and both pointers refer to device memory allocated by cudaMalloc.
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The kernel accumulates partial block results into the output histogram,
    // so clear the destination first. This is intentionally asynchronous; the
    // caller requested to handle synchronization externally.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Use occupancy information, but cap the launch to 4 blocks/SM.
    // This kernel has a non-trivial fixed cost per block (shared-memory zeroing
    // and final merge), and on A100/H100-class GPUs 4 resident 256-thread blocks/SM
    // already provide ample occupancy while avoiding unnecessary extra block-level
    // merge traffic to the global histogram.
    int device = 0;
    (void)cudaGetDevice(&device);

    int smCount = 1;
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int maxBlocksPerSM = 1;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        histogram_char_range_kernel,
        kBlockThreads,
        static_cast<int>(sharedBytes));
    if (maxBlocksPerSM < 1) {
        maxBlocksPerSM = 1;
    }

    const int targetBlocksPerSM = (maxBlocksPerSM > 4) ? 4 : maxBlocksPerSM;
    int grid = smCount * targetBlocksPerSM;

    // Do not launch more blocks than can do useful work in the first unrolled pass.
    const unsigned int maxUsefulBlocks =
        ceil_div_u32(inputSize, static_cast<unsigned int>(kBlockThreads * kItemsPerLoop));
    if (grid > static_cast<int>(maxUsefulBlocks)) {
        grid = static_cast<int>(maxUsefulBlocks);
    }
    if (grid < 1) {
        grid = 1;
    }

    // Reinterpret as unsigned bytes so values 128..255 are handled by ordinal
    // value rather than sign extension of plain char.
    histogram_char_range_kernel<<<grid, kBlockThreads, sharedBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        numBins);
}