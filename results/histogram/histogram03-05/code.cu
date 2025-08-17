#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel for a character range [from, to] inclusive.

  Key optimizations:
  - Per-warp privatized histograms in shared memory: reduces contention on global memory drastically.
  - Grid-stride loop over the input for full device utilization.
  - Branchless in-range test using unsigned comparison for fast filtering.
  - Only one atomicAdd per bin per block to global memory during the merge phase.

  Assumptions:
  - 0 <= from < to <= 255
  - input points to device memory containing 'inputSize' characters
  - histogram points to device memory with at least (to - from + 1) unsigned int elements
  - histogram is zeroed before the kernel runs (we zero it in run_histogram using cudaMemset)
  - blockDim.x is a multiple of warpSize (enforced in run_histogram)
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from, int to)
{
    const int rangeLen = to - from + 1; // Number of bins to compute
    const int warpSz   = 32;
    const int warpsPerBlock = blockDim.x / warpSz;

    // Shared memory layout: warpsPerBlock stacks of 'rangeLen' counters (one stack per warp)
    extern __shared__ unsigned int s_warpHists[];
    // Zero initialize all shared-memory histograms
    for (int i = threadIdx.x; i < warpsPerBlock * rangeLen; i += blockDim.x) {
        s_warpHists[i] = 0;
    }
    __syncthreads();

    const int laneId = threadIdx.x & (warpSz - 1);
    const int warpId = threadIdx.x >> 5;
    unsigned int* warpHist = s_warpHists + warpId * rangeLen;

    // Grid-stride loop to process all input bytes
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < inputSize;
         idx += gridDim.x * blockDim.x)
    {
        // Load a byte and map it to the [from, to] range if applicable
        unsigned int ch = static_cast<unsigned int>(input[idx]); // 0..255
        int bin = static_cast<int>(ch) - from;

        // Fast in-range test: true if 0 <= bin < rangeLen
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(rangeLen)) {
            // Increment per-warp shared histogram to avoid global atomics
            atomicAdd(&warpHist[bin], 1);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and merge into global memory
    for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_warpHists[w * rangeLen + bin];
        }
        // Single global atomic add per (block, bin)
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
  Host launcher:
  - Computes a reasonable grid configuration for modern data center GPUs (A100/H100).
  - Ensures histogram output is zeroed before computation.
  - Launches the kernel with dynamic shared memory sized to warpsPerBlock * rangeLen counters.

  Notes:
  - No device synchronization is performed here; ordering relies on the default stream.
  - If the caller uses custom streams, they may adapt this to accept a cudaStream_t.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int rangeLen = to - from + 1;
    if (inputSize == 0 || rangeLen <= 0) {
        // Nothing to do, but make sure histogram is zero if rangeLen > 0.
        if (rangeLen > 0) {
            cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));
        }
        return;
    }

    // Zero the output histogram (range is contiguous and tightly packed)
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Choose execution configuration
    // - Threads per block: multiple of warp size (recommend 256 or 512).
    // - Blocks: scale with SM count to saturate GPU, but also bound by input size.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const int threadsPerBlock = 256; // Must be a multiple of 32
    const int warpsPerBlock   = threadsPerBlock / 32;

    // A good heuristic for occupancy: 8 blocks per SM (tune if needed)
    int maxActiveBlocks = prop.multiProcessorCount * 8;

    // Also bound by the amount of work
    int blocksByWork = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    int numBlocks = blocksByWork < maxActiveBlocks ? blocksByWork : maxActiveBlocks;
    if (numBlocks < 1) numBlocks = 1;

    // Dynamic shared memory size: one per-warp histogram of 'rangeLen' counters
    size_t shmemSize = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Reinterpret input as unsigned char* to guarantee 0..255 byte semantics
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);

    // Launch kernel on the default stream; caller handles synchronization if needed
    histogram_range_kernel<<<numBlocks, threadsPerBlock, shmemSize>>>(d_input, histogram, inputSize, from, to);
}