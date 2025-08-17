#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram over a restricted ASCII range using shared-memory privatization.

  Key ideas:
  - Each block builds a private histogram in shared memory to drastically reduce global atomics.
  - To further reduce contention within a block, each warp accumulates into its own private sub-histogram.
  - After processing, we aggregate the per-warp histograms to one partial histogram per block and emit a
    single atomicAdd per bin per block to global memory.

  Notes:
  - Input is treated as unsigned bytes to avoid sign-extension problems with char.
  - The output histogram has length (to - from + 1), where bin i counts the occurrences of character (from + i).
  - The host function zeros the output histogram asynchronously prior to the kernel launch.
  - Grid-stride loop ensures all input is covered regardless of grid size.
  - The host picks a reasonable launch configuration using CUDA occupancy APIs.
*/

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// CUDA kernel: builds a histogram for characters in [from, from + rangeLen - 1] inclusive.
// - input: device pointer to input characters
// - global_hist: device pointer to global histogram with length 'rangeLen'
// - inputSize: number of characters in 'input'
// - from: lower bound of the character range (inclusive)
// - rangeLen: number of bins (to - from + 1)
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int inputSize,
                                       unsigned int from,
                                       unsigned int rangeLen)
{
    // Dynamic shared memory layout:
    // [ warp0_hist (rangeLen) | warp1_hist (rangeLen) | ... | warp{numWarps-1}_hist (rangeLen) ]
    extern __shared__ unsigned int s_hist[];
    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;
    const unsigned int warpId = tid >> 5;  // tid / 32
    const unsigned int numWarps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

    // Zero the entire shared memory space cooperatively
    for (unsigned int i = tid; i < numWarps * rangeLen; i += nthreads) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process input with a grid-stride loop
    size_t idx = static_cast<size_t>(blockIdx.x) * nthreads + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * nthreads;

    // Each warp writes into its own histogram slice to reduce inter-warp contention
    unsigned int* warp_hist = s_hist + warpId * rangeLen;

    // Compute the upper bound once to simplify the range check
    const unsigned int upper = from + rangeLen; // exclusive upper bound

    while (idx < inputSize) {
        unsigned int c = static_cast<unsigned int>(input[idx]); // 0..255
        // Range check: c in [from, from + rangeLen)
        if (c >= from && c < upper) {
            // Atomic into shared memory (warp-private slice) to handle intra-warp collisions
            atomicAdd(&warp_hist[c - from], 1u);
        }
        idx += stride;
    }
    __syncthreads();

    // Aggregate per-warp histograms and emit to global memory.
    // One atomicAdd per bin per block.
    for (unsigned int bin = tid; bin < rangeLen; bin += nthreads) {
        unsigned int sum = 0;
        // Sum across warps for this bin
        for (unsigned int w = 0; w < numWarps; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }
        if (sum != 0) {
            atomicAdd(&global_hist[bin], sum);
        }
    }
    // No need for __syncthreads() here; kernel is exiting.
}

// Host function to invoke the kernel.
// - input: device pointer to input characters (cudaMalloc'ed), size 'inputSize'
// - histogram: device pointer to output histogram array (cudaMalloc'ed), size (to - from + 1) uints
// - inputSize: number of characters in 'input'
// - from, to: inclusive range [from, to] over which to compute the histogram
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Basic validation; assumes inputs adhere to the specified contract.
    if (!input || !histogram || inputSize == 0) {
        // Even if there's nothing to do, ensure the output is zeroed for correctness.
        if (histogram && from <= to) {
            unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);
            cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));
        }
        return;
    }

    // Clamp/validate range for safety; contract states 0 <= from < to <= 255
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (to < from) {
        return;
    }

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram asynchronously
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Choose launch configuration
    constexpr int blockSize = 256; // multiple of warp size; good balance for A100/H100
    const int warpsPerBlock = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
    const size_t sharedMemBytes = static_cast<size_t>(warpsPerBlock) * rangeLen * sizeof(unsigned int);

    // Use occupancy API to pick a good grid size
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxActiveBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM,
        histogram_range_kernel,
        blockSize,
        sharedMemBytes);

    int gridSize = maxActiveBlocksPerSM * props.multiProcessorCount;
    if (gridSize <= 0) gridSize = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;

    // Launch kernel
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<gridSize, blockSize, sharedMemBytes>>>(
        d_input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        rangeLen);
}