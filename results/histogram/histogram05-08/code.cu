#include <cuda_runtime.h>

/*
  Optimized CUDA histogram over a restricted byte range [from, to] inclusive.

  Key optimization techniques:
  - Shared-memory histogram privatization: Each thread block uses shared memory to
    accumulate a partial histogram, reducing global memory atomics.
  - Multiple shared-memory histogram copies per warp (sub-histograms) to reduce shared
    memory bank conflicts and intra-warp contention.
  - Each thread processes a fixed number of items (itemsPerThread) with coalesced
    global memory access (via a strided access pattern), providing instruction-level
    parallelism (ILP) and better memory throughput on modern GPUs (A100/H100).

  Notes:
  - The global output histogram must be zeroed before accumulation; run_histogram() does this.
  - The input pointer and histogram pointer are device pointers (allocated with cudaMalloc).
  - No host-device synchronization is performed here; the caller handles synchronization.
*/

// Tunables chosen for modern data center GPUs (A100/H100).
// - itemsPerThread: number of bytes processed by each thread in one pass.
//   Value 16 typically provides a good balance between ILP and occupancy for large inputs.
static constexpr int itemsPerThread = 16;

// - subHistPerWarp: number of duplicated sub-histograms per warp in shared memory.
//   Threads select a sub-histogram based on (laneId % subHistPerWarp) to spread updates
//   and reduce shared memory bank conflicts and contention. Using 4 keeps dynamic shared
//   memory under 48KB for up to 256 bins with 256-thread blocks (portable without special attrs).
static constexpr int subHistPerWarp = 4;

#ifndef __CUDACC_RTC__  // warpSize is defined in device code; provide a fallback for host compilation units.
static constexpr int WARP_SIZE = 32;
#else
#define WARP_SIZE warpSize
#endif

// Utility to get lane and warp IDs within a block
static __device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static __device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

// CUDA kernel: compute partial histograms in shared memory and aggregate to global memory.
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ globalHist,
                                unsigned int N,
                                int from,
                                int to)
{
    // Number of bins in the requested range [from, to]
    const int numBins = to - from + 1;

    // Compute the per-block shared histogram layout:
    // shared memory holds (warpsPerBlock * subHistPerWarp) copies, each of length numBins.
    extern __shared__ unsigned int s_mem[];
    const int warpsPerBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    unsigned int* s_hist = s_mem;  // size: warpsPerBlock * subHistPerWarp * numBins

    // Zero initialize the shared memory histograms cooperatively
    const int totalSubHists = warpsPerBlock * subHistPerWarp;
    const int totalSharedBins = totalSubHists * numBins;
    for (int i = threadIdx.x; i < totalSharedBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Each thread selects its sub-histogram:
    // - warp-specific region selected by warpId
    // - duplicate within that region selected by laneId % subHistPerWarp
    const int wId = warp_id();
    const int lId = lane_id();
    const int dupId = lId & (subHistPerWarp - 1);  // subHistPerWarp is power of two (4)

    unsigned int* mySubHist = s_hist + ((wId * subHistPerWarp + dupId) * numBins);

    // Global indexing: each thread processes itemsPerThread items in a grid-stride fashion
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    // Process itemsPerThread items per thread with coalesced access across threads.
    #pragma unroll
    for (int rep = 0; rep < itemsPerThread; ++rep) {
        unsigned int idx = t + static_cast<unsigned int>(rep) * stride;
        if (idx >= N) break;

        // Load byte and map it into bin index if within [from, to]
        unsigned char byte = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(byte) - from;

        // Fast range check using unsigned compare
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            // Increment the bin in our selected shared-memory sub-histogram.
            // atomicAdd is still required because multiple threads may map to the same
            // address even with duplication (subHistPerWarp < warp size).
            atomicAdd(&mySubHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce all sub-histograms within the block and update the global histogram.
    // Assign bins to threads in a strided fashion so each bin is updated exactly once per block.
    for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
        unsigned int sum = 0u;
        // Sum over all (warpsPerBlock * subHistPerWarp) sub-histograms for bin b
        for (int w = 0; w < warpsPerBlock; ++w) {
            const unsigned int* warpBase = s_hist + (w * subHistPerWarp * numBins);
            #pragma unroll
            for (int d = 0; d < subHistPerWarp; ++d) {
                sum += warpBase[d * numBins + b];
            }
        }
        // Single global atomicAdd per bin per block
        atomicAdd(&globalHist[b], sum);
    }
}

// Host launcher: prepares the grid configuration and calls the CUDA kernel.
void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Zero the output histogram on device (caller handles synchronization semantics).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Nothing to process
    if (inputSize == 0) {
        return;
    }

    // Launch configuration:
    // - 256 threads per block balances occupancy and shared memory footprint.
    // - Grid size ensures totalThreads * itemsPerThread >= inputSize (single-pass).
    const int blockSize = 256;
    const unsigned long long workPerBlock = static_cast<unsigned long long>(blockSize) * static_cast<unsigned long long>(itemsPerThread);
    unsigned int gridSize = static_cast<unsigned int>((static_cast<unsigned long long>(inputSize) + workPerBlock - 1ULL) / workPerBlock);
    if (gridSize == 0) gridSize = 1;

    // Dynamic shared memory size: warpsPerBlock * subHistPerWarp * numBins * sizeof(uint)
    const int warpsPerBlock = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
    const size_t dynamicSmemSize = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(subHistPerWarp) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel (default stream). Caller handles any synchronization needed.
    histogramKernel<<<gridSize, blockSize, dynamicSmemSize>>>(input, histogram, inputSize, from, to);
}