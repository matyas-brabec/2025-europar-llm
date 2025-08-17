#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA histogram for a restricted character range [from..to].

  Design:
  - Each thread processes a fixed number of input items (itemsPerThread) to amortize scheduling overhead and improve memory throughput.
  - Each block maintains a privatized histogram in shared memory, further split per-warp to reduce contention.
  - Shared histograms are padded to mitigate bank conflicts: index mapping uses i + (i >> 5) with 32-bin periodic padding.
  - After counting, per-warp histograms are reduced within the block, and only rangeLen atomic updates per block are issued to the global histogram.
  - The output histogram holds exactly (to - from + 1) bins for ordinals [from..to].

  Notes:
  - The input is an array of chars (signedness of char is implementation-defined), so we cast to unsigned char to interpret as [0..255].
  - The caller is responsible for synchronization; this function launches asynchronous work on the default stream.
  - The histogram buffer is zeroed here before the kernel launch to produce fresh results.
*/

// Tunables for modern NVIDIA data-center GPUs (e.g., A100/H100).
// itemsPerThread is chosen to balance memory throughput and occupancy for large inputs.
static constexpr int BLOCK_SIZE = 256;
static constexpr int itemsPerThread = 16;

// CUDA kernel implementing the range-restricted histogram with shared-memory privatization.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ outHist,
                                       unsigned int inputSize,
                                       int from, int to)
{
    // Validate assumptions in comments; inputs are assumed valid per problem statement:
    // 0 <= from < to <= 255, outHist has (to - from + 1) bins.

    const int rangeLen = to - from + 1;
    // Shared-memory padding to avoid bank conflicts: allocate ceil(rangeLen/32) extra slots
    // and map logical bin i to physical i + (i >> 5).
    const int pad = (rangeLen + 31) >> 5;
    const int paddedLen = rangeLen + pad;

    extern __shared__ unsigned int sHist[]; // Layout: warpsPerBlock consecutive sub-histograms, each of length paddedLen.

    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;
    const int bid   = blockIdx.x;

    const int warpsPerBlock = bdim >> 5;
    const int warpId        = tid >> 5;   // warp index within the block
    const int laneId        = tid & 31;   // lane index within the warp

    // Pointer to this warp's private histogram in shared memory.
    unsigned int* warpHist = sHist + warpId * paddedLen;

    // Zero initialize the entire shared histogram (all warps' sub-histograms).
    for (int i = tid; i < paddedLen * warpsPerBlock; i += bdim) {
        sHist[i] = 0u;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    const size_t blockStart = static_cast<size_t>(bid) * static_cast<size_t>(bdim) * static_cast<size_t>(itemsPerThread);

    // Process itemsPerThread items per thread, spaced by blockDim to ensure fully coalesced loads per iteration.
    #pragma unroll
    for (int j = 0; j < itemsPerThread; ++j) {
        const size_t idx = blockStart + static_cast<size_t>(j) * static_cast<size_t>(bdim) + static_cast<size_t>(tid);
        if (idx < static_cast<size_t>(inputSize)) {
            // Read using read-only cache path on capable architectures for better caching behavior.
            const unsigned char v = static_cast<unsigned char>(__ldg(input + idx));
            // If the character falls within [from..to], update the corresponding bin in this warp's sub-histogram.
            if (v >= from && v <= to) {
                const int bin = static_cast<int>(v) - from;
                const int pbin = bin + (bin >> 5); // padded index to reduce bank conflicts
                atomicAdd(&warpHist[pbin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce all warp-local histograms into a single set of totals and merge into global memory.
    // Each thread handles multiple bins in a strided fashion to parallelize the reduction.
    for (int bin = tid; bin < rangeLen; bin += bdim) {
        unsigned int total = 0;
        const int pbin = bin + (bin >> 5);
        // Accumulate contributions across all warps for this bin.
        for (int w = 0; w < warpsPerBlock; ++w) {
            total += sHist[w * paddedLen + pbin];
        }
        // Only perform atomicAdd if there is a non-zero contribution.
        if (total) {
            atomicAdd(&outHist[bin], total);
        }
    }
    // No need for __syncthreads() here, we are leaving the kernel.
}

// Host-side launcher.
// Assumptions:
// - 'input' and 'histogram' are device pointers allocated with cudaMalloc.
// - 'histogram' has space for exactly (to - from + 1) unsigned ints.
// - The caller performs any required device synchronization after this call.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (input == nullptr || histogram == nullptr) return;
    if (to < from) return;

    const int rangeLen = to - from + 1;

    // Zero the output histogram so we can use atomicAdd in the kernel safely.
    // This is asynchronous with respect to the host; caller should synchronize if needed.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Determine launch configuration.
    const unsigned long long elemsPerBlock = static_cast<unsigned long long>(BLOCK_SIZE) * static_cast<unsigned long long>(itemsPerThread);
    if (inputSize == 0 || elemsPerBlock == 0ull) return;

    // Compute number of blocks to cover the input; round up.
    const unsigned long long grid64 = (static_cast<unsigned long long>(inputSize) + elemsPerBlock - 1ull) / elemsPerBlock;
    const int grid = static_cast<int>(grid64);

    if (grid <= 0) return;

    // Dynamic shared memory size: padded bins per warp times number of warps.
    const int pad = (rangeLen + 31) / 32;
    const int paddedLen = rangeLen + pad;
    const int warpsPerBlock = BLOCK_SIZE / 32;
    const size_t shmemBytes = static_cast<size_t>(paddedLen) * static_cast<size_t>(warpsPerBlock) * sizeof(unsigned int);

    // Launch kernel on the default stream.
    histogram_range_kernel<<<grid, BLOCK_SIZE, shmemBytes>>>(input, histogram, inputSize, from, to);

    // No synchronization here; caller handles it.
}