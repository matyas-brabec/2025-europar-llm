#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

/*
  CUDA kernel to compute a histogram over a specified [from, to] character range
  for an input text buffer.

  Optimization strategy:
  - Use shared memory for per-block sub-histograms to reduce global memory atomics.
  - Privatize the shared histogram per warp (one sub-histogram per warp) to reduce
    intra-block contention and shared-memory atomics to at most 32-way contention.
  - Perform a block-level reduction of the per-warp histograms into a final per-block
    histogram, then commit to global memory with at most (to - from + 1) atomic adds
    per block.

  Notes:
  - Characters are treated as unsigned bytes to avoid signed-char pitfalls.
  - Range check is implemented branch-efficiently using unsigned comparison.
  - The global histogram must be zeroed before launching this kernel if a fresh
    histogram is desired. The provided run_histogram() takes care of zeroing.

  Parameters:
  - input: pointer to device memory containing the input text (as unsigned bytes).
  - histogram: pointer to device memory for the output histogram (size = rangeLen).
  - inputSize: number of bytes in the input buffer.
  - from: lower bound (inclusive) of the character range to histogram (0..255).
  - rangeLen: number of histogram bins = to - from + 1 (1..256).

  Dynamic shared memory layout:
  - s_hist is an array of 'warpsPerBlock * rangeLen' unsigned ints.
    The sub-histogram for warp w starts at s_hist[w * rangeLen].
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int rangeLen)
{
    extern __shared__ unsigned int s_hist[];
    const int tid = threadIdx.x;
    const int tpb = blockDim.x;
    const int WARP = 32;
    const int warpsPerBlock = (tpb + WARP - 1) / WARP;
    const int warpId = tid / WARP;

    // Zero the entire per-block shared histogram (all warp-private sub-histograms).
    for (int i = tid; i < rangeLen * warpsPerBlock; i += tpb) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each warp writes into its own sub-histogram to reduce contention.
    unsigned int* s_hist_base = s_hist + warpId * rangeLen;

    // Grid-stride loop for coalesced reads and scalability across grid sizes.
    const unsigned int stride = tpb * gridDim.x;
    for (unsigned int idx = blockIdx.x * tpb + tid; idx < inputSize; idx += stride) {
        unsigned int c = static_cast<unsigned int>(input[idx]); // 0..255
        int bin = static_cast<int>(c) - from;
        // Use unsigned range check to combine lower+upper bounds in one compare:
        // valid iff 0 <= bin < rangeLen
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(rangeLen)) {
            // Atomic add in shared memory (fast on modern GPUs). Contention is at most 32-way.
            atomicAdd(&s_hist_base[bin], 1U);
        }
    }
    __syncthreads();

    // Reduce the per-warp histograms into a single per-block histogram and commit to global memory.
    for (int bin = tid; bin < rangeLen; bin += tpb) {
        unsigned int sum = 0;
        // Accumulate counts from all warp-private sub-histograms for this bin.
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }
        // Multiple blocks contribute to the final global histogram; use atomic add.
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
  Host-side launcher.
  - input: device pointer to input text buffer (chars/bytes).
  - histogram: device pointer to output histogram buffer of size (to - from + 1).
  - inputSize: number of bytes in input.
  - from, to: inclusive character range to compute histogram over.

  This function:
  - Computes the range length.
  - Zeros the output histogram on device (fresh histogram).
  - Chooses a launch configuration suitable for A100/H100-class GPUs.
  - Allocates dynamic shared memory sized as (rangeLen * warpsPerBlock * sizeof(unsigned int)).
  - Launches the kernel; no synchronization is performed here (caller handles it).
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (!input || !histogram || from > to) {
        return;
    }

    const int rangeLen = (to - from) + 1;
    if (rangeLen <= 0) {
        return;
    }

    // Zero the output histogram (device memory). This ensures fresh counts.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    if (inputSize == 0) {
        // Nothing to process.
        return;
    }

    // Select launch configuration.
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    const int threadsPerBlock = 256; // Good balance for A100/H100
    const int warpsPerBlock = (threadsPerBlock + 31) / 32;

    // Aim for a reasonable number of blocks relative to SM count.
    // Use min(blocksByInput, targetBlocks) to avoid launching too many blocks for small inputs.
    const int maxBlocks = 65535;
    const int targetBlocks = prop.multiProcessorCount * 8; // 6-8 per SM is typical
    int blocksByInput = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    if (blocksByInput < 1) blocksByInput = 1;

    int numBlocks = blocksByInput;
    if (numBlocks > targetBlocks) numBlocks = targetBlocks;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;

    // Dynamic shared memory size: one sub-histogram per warp.
    const size_t sharedMemBytes = static_cast<size_t>(rangeLen) * warpsPerBlock * sizeof(unsigned int);

    // Launch the kernel. The input is treated as unsigned bytes.
    histogram_range_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        rangeLen
    );
}