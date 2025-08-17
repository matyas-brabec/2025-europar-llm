#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized range-restricted histogram on NVIDIA A100/H100 GPUs.

  Key features:
  - Range restriction: only characters in [from, to] are counted.
  - Shared-memory privatization: each warp in a block has a private sub-histogram
    to reduce contention and global atomics.
  - itemsPerThread: compile-time constant controlling workload per thread for
    high memory throughput and reduced launch overhead on large inputs.

  Notes:
  - Input is treated as bytes (unsigned char) to avoid issues with signed char.
  - Global histogram must be zeroed before accumulation; we do this with cudaMemsetAsync.
  - No host-device synchronization in this function; the caller handles it.
*/

// Tunable parameter: number of items processed per thread.
// 16 is a strong default on Ampere/Hopper class GPUs for large, memory-bound inputs.
// This yields 4096 items per 256-thread block, balancing occupancy and memory throughput.
static constexpr int itemsPerThread = 16;

// CUDA kernel: Compute histogram for characters in [from, to] inclusive.
// input: pointer to device memory with 'n' bytes of text.
// global_hist: pointer to device memory with (to - from + 1) bins (uint32).
// Shared memory layout (dynamic): warpsPerBlock contiguous histograms of length rangeLen each.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int n,
                                       int from,
                                       int to)
{
    const int tid    = threadIdx.x;
    const int blockThreads = blockDim.x;
    const int warpsPerBlock = blockThreads / warpSize;
    const int warpId = tid / warpSize;

    const int rangeLen = to - from + 1;
    if (rangeLen <= 0) return;  // guard; should not happen if inputs are valid

    extern __shared__ unsigned int s_hist[]; // size: rangeLen * warpsPerBlock

    // Zero the shared histograms (one per warp)
    for (int i = tid; i < rangeLen * warpsPerBlock; i += blockThreads) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Base index of the current thread's chunk of the input
    const unsigned int threadBase = (blockIdx.x * blockThreads + tid) * itemsPerThread;

    // Each warp writes to its own sub-histogram region in shared memory
    unsigned int* __restrict__ warp_hist = s_hist + warpId * rangeLen;

    // Process up to itemsPerThread items per thread
    // Use unsigned arithmetic for comparisons with [from, to]
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int to_u   = static_cast<unsigned int>(to);

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        const unsigned int idx = threadBase + static_cast<unsigned int>(i);
        if (idx < n) {
            const unsigned int c = static_cast<unsigned int>(input[idx]); // treat as unsigned byte
            if (c >= from_u && c <= to_u) {
                // Shared-memory atomic to per-warp histogram bin
                atomicAdd(&warp_hist[c - from_u], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms and accumulate into global histogram
    // Each thread reduces a subset of bins across warps and performs one global atomicAdd per bin.
    for (int bin = tid; bin < rangeLen; bin += blockThreads) {
        unsigned int sum = 0;
        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }
        if (sum > 0) {
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

// Host function to launch the kernel.
// input: device pointer to text buffer (char* allocated with cudaMalloc)
// histogram: device pointer to output bins (uint32* allocated with cudaMalloc), length = (to - from + 1)
// inputSize: number of bytes in input
// from, to: inclusive range [from, to], 0 <= from <= to <= 255
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (!input || !histogram) return;
    if (inputSize == 0) {
        // Nothing to do; ensure histogram is zeroed
        if (to >= from) {
            const size_t rangeLen = static_cast<size_t>(to - from + 1);
            (void)cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int), 0);
        }
        return;
    }
    if (from > to) return;

    const int rangeLen = to - from + 1;

    // Zero the output histogram (asynchronous; default stream 0).
    (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int), 0);

    // Choose a block size optimized for shared memory atomics and good occupancy.
    // 256 threads/block provides 8 warps per block; with warp-private histograms,
    // shared memory usage is: rangeLen * warpsPerBlock * sizeof(uint) <= 256 * 8 * 4 = 8 KB
    // which is friendly to occupancy on A100/H100.
    const int threadsPerBlock = 256;

    // Compute number of threads needed so each thread processes up to itemsPerThread items
    const unsigned int threadsNeeded = (inputSize + itemsPerThread - 1) / itemsPerThread;
    if (threadsNeeded == 0) return;

    const unsigned int blocks = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    // Dynamic shared memory size: one sub-histogram per warp in the block
    /// @FIXED
    int warpSize = 0;
    /// @FIXED
    cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
    const size_t shmemSize = static_cast<size_t>(rangeLen) * static_cast<size_t>(threadsPerBlock / warpSize) * sizeof(unsigned int);

    // Launch kernel on default stream
    const unsigned char* input_u8 = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<blocks, threadsPerBlock, shmemSize>>>(input_u8, histogram, inputSize, from, to);
}
