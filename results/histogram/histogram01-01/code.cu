#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized CUDA kernel to compute a histogram for a specified contiguous range [from, to]
  over an input buffer of chars (bytes). The kernel uses:
    - Per-warp private histograms in shared memory to minimize global memory atomics.
    - Grid-stride loop with loop unrolling to utilize memory bandwidth.
    - Shared-memory atomics (fast on modern GPUs) to accumulate per-warp counts.
    - A single global atomicAdd per bin per block at the end to combine results.

  The output histogram has length (to - from + 1), where histogram[i] corresponds
  to the count of byte value (from + i).
*/

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tuneable launch parameters
#ifndef HIST_BLOCK_DIM
#define HIST_BLOCK_DIM 256   // 256 threads per block is a good default for Ampere/Hopper
#endif

#ifndef HIST_UNROLL
#define HIST_UNROLL 4        // Process 4 items per thread per outer loop iteration
#endif

// CUDA kernel. Pointers are marked __restrict__ to help the compiler optimize memory accesses.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int inputSize,
                                       unsigned int from,
                                       unsigned int to,
                                       unsigned int* __restrict__ histogram)
{
    // Compute the number of bins in the requested range.
    const unsigned int rangeLen = to - from + 1u;

    // Shared memory layout: warpsPerBlock private histograms, each of size rangeLen.
    extern __shared__ unsigned int s_warpHists[];
    const int warpsPerBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int warpOffset = warpId * rangeLen;

    // Zero initialize per-warp histograms in shared memory.
    for (unsigned int i = threadIdx.x; i < (unsigned int)(warpsPerBlock * rangeLen); i += blockDim.x) {
        s_warpHists[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input with unrolling.
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Cache 'from'/'to' as bytes to reduce conversions.
    const unsigned char f = static_cast<unsigned char>(from);
    const unsigned char t = static_cast<unsigned char>(to);

    // Process input bytes in a grid-stride, unrolled loop.
    for (unsigned int base = globalThreadId; base < inputSize; base += totalThreads * HIST_UNROLL) {
        #pragma unroll
        for (int k = 0; k < HIST_UNROLL; ++k) {
            unsigned int idx = base + k * totalThreads;
            if (idx < inputSize) {
                // Load one byte. Using unsigned char prevents sign-extension issues.
                unsigned char c = input[idx];
                // Fast range test and update the per-warp histogram using shared-memory atomics.
                if (c >= f && c <= t) {
                    unsigned int bin = static_cast<unsigned int>(c) - static_cast<unsigned int>(f);
                    atomicAdd(&s_warpHists[warpOffset + bin], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram using one global atomic per bin per block.
    for (unsigned int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate the same bin across all warps in the block.
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_warpHists[w * rangeLen + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host function to launch the histogram kernel.
// - input: device pointer to char buffer (cudaMalloc'd)
// - histogram: device pointer to unsigned int buffer of size (to - from + 1) (cudaMalloc'd)
// - inputSize: number of chars in input
// - from, to: inclusive byte range [from, to] to compute histogram for
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Convert to proper types and compute range length.
    unsigned int uFrom = static_cast<unsigned int>(from);
    unsigned int uTo   = static_cast<unsigned int>(to);
    unsigned int rangeLen = uTo - uFrom + 1u;

    // Initialize the output histogram to zero asynchronously.
    // Using the default stream (0) preserves launch order without forcing host-device sync.
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int), 0);

    // Determine a reasonable grid size based on occupancy and input size.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int blockDim = HIST_BLOCK_DIM;
    const int warpsPerBlock = (blockDim + WARP_SIZE - 1) / WARP_SIZE;
    const size_t smemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    int blocksPerSM = 0;
    // Query occupancy to find the maximum number of active blocks per SM for the given configuration.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
                                                  histogram_range_kernel,
                                                  blockDim,
                                                  smemBytes);

    int gridDim = prop.multiProcessorCount * (blocksPerSM > 0 ? blocksPerSM : 1);

    // Optionally cap the grid to avoid oversubscribing for very small inputs.
    // Rough heuristic: ensure at least ~HIST_UNROLL elements per thread when possible.
    unsigned long long threadsTotal = static_cast<unsigned long long>(gridDim) * static_cast<unsigned long long>(blockDim);
    if (threadsTotal > 0ULL) {
        unsigned long long idealBlocks = (static_cast<unsigned long long>(inputSize) + (unsigned long long)(blockDim * HIST_UNROLL) - 1ULL)
                                         / (unsigned long long)(blockDim * HIST_UNROLL);
        if (idealBlocks == 0ULL) idealBlocks = 1ULL;
        if ((unsigned long long)gridDim > idealBlocks) {
            gridDim = static_cast<int>(idealBlocks);
        }
    } else {
        gridDim = 1;
    }

    // Launch the kernel.
    histogram_range_kernel<<<gridDim, blockDim, smemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        inputSize,
        uFrom,
        uTo,
        histogram
    );
}