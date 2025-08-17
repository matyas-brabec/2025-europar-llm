#include <cuda_runtime.h>
#include <stdint.h>

// Number of input items processed by each thread. On A100/H100, values in [8..32] work well.
// 16 is a good balance for large inputs and moderate register pressure.
static constexpr int itemsPerThread = 16;

// Warp-sum utility using shuffles (assumes full warp participation).
static inline __device__ unsigned int warp_reduce_sum(unsigned int v) {
    // Use warp synchronous reduction. Mask is full warp for simplicity.
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

/*
  Kernel computes a histogram over characters in the inclusive range [from, to].
  - input: device pointer to chars
  - histogram: device pointer to output bins (size: numBins = to-from+1). This is assumed zeroed before kernel.
  - n: number of bytes in input
  - from, to: define the [from, to] (inclusive) range to count

  Optimization details:
  - Shared memory privatization: 32 copies of the histogram are kept in shared memory with layout:
      s_hist[bin * 32 + laneId], where laneId is threadIdx.x % 32.
    This places each copy in a distinct shared memory bank. Within a warp, each lane increments a separate
    copy, avoiding intra-warp bank conflicts. Multiple warps share these 32 copies, so we use shared-memory atomics.
  - Each thread processes itemsPerThread items in a block-striped pattern for good global memory coalescing:
      idx = base + j * blockDim.x
  - Final reduction: First warp (threads 0..31) reduces the 32 copies per bin with warp shuffles and performs
    a single atomicAdd per bin to global memory, minimizing global atomics.
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int n,
                                       int from, int to)
{
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    if (numBins == 0) return; // nothing to do

    // Shared-memory layout: numBins rows, 32 columns (one per warp lane).
    extern __shared__ unsigned int s_hist[];
    const unsigned int laneId = threadIdx.x & 31u;

    // Initialize shared histogram copies. Use only the first warp for conflict-free init.
    if (threadIdx.x < 32) {
        for (unsigned int b = 0; b < numBins; ++b) {
            s_hist[b * 32 + laneId] = 0;
        }
    }
    __syncthreads();

    // Each thread processes itemsPerThread items, striding by blockDim.x to keep loads coalesced.
    const unsigned int blockBase = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = blockDim.x;
    const unsigned int range = static_cast<unsigned int>(to - from); // avoids two comparisons via unsigned arithmetic

    #pragma unroll
    for (int j = 0; j < itemsPerThread; ++j) {
        unsigned int idx = blockBase + static_cast<unsigned int>(j) * stride;
        if (idx < n) {
            unsigned int uc = static_cast<unsigned int>(input[idx]); // 0..255
            // Branchless check if uc in [from, to]:
            unsigned int b = uc - static_cast<unsigned int>(from);   // Underflow produces a large value
            if (b <= range) {
                // Update the lane-private copy of bin b in shared memory.
                // Indexing scheme bin*32 + lane places each lane in a separate bank.
                atomicAdd(&s_hist[b * 32u + laneId], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 copies per bin and update global histogram.
    if (threadIdx.x < 32) {
        for (unsigned int b = 0; b < numBins; ++b) {
            unsigned int val = s_hist[b * 32u + laneId];
            unsigned int sum = warp_reduce_sum(val);
            if (laneId == 0u) {
                atomicAdd(&histogram[b], sum);
            }
        }
    }
}

// Host API: Launch the histogram kernel.
// - input: device pointer (cudaMalloc'd) to chars
// - histogram: device pointer (cudaMalloc'd) to unsigned int array of size (to - from + 1)
// - inputSize: number of chars in input
// - from, to: inclusive range [from, to] to be histogrammed
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (inputSize == 0 || from > to) {
        // Ensure output is zeroed for empty/no-op case.
        if (from <= to) {
            const size_t numBins = static_cast<size_t>(to - from + 1);
            cudaMemset(histogram, 0, numBins * sizeof(unsigned int));
        }
        return;
    }

    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram on device to accumulate only current results.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Configure launch. 256 threads per block is a good default for A100/H100 for this pattern.
    const int blockSize = 256;
    const unsigned int workPerBlock = static_cast<unsigned int>(blockSize * itemsPerThread);
    const unsigned int gridSize = (inputSize + workPerBlock - 1) / workPerBlock;

    // Dynamic shared memory size: numBins rows * 32 lane-copies * 4 bytes.
    const size_t shmemSize = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    histogram_range_kernel<<<gridSize, blockSize, shmemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from, to
    );
}