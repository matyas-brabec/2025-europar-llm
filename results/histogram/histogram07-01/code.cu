#include <cuda_runtime.h>
#include <stdint.h>

/*
Optimized CUDA histogram (range-restricted) for text data.

Key features:
- Range-restricted histogram: only counts characters whose ordinal is in [from, to].
- Shared memory privatization with 32 copies to avoid bank conflicts within a warp.
  The layout is strided so that for each histogram bin i and lane c (0..31):
    shared_index = i * 32 + c
  This makes value i of copy c map to a unique shared memory bank.
- Each thread updates the copy indexed by its lane (threadIdx.x % 32).
- itemsPerThread controls how many input characters each thread processes per grid-stride iteration.
  A value of 16 is a good default for modern data center GPUs (A100/H100) and large inputs.
- Grid-stride loop with unrolling ensures coalesced loads for each unrolled step.

Assumptions:
- 'input' and 'histogram' are device pointers allocated with cudaMalloc.
- 'inputSize' is the number of chars in the input buffer.
- The caller handles synchronization; run_histogram does not call cudaDeviceSynchronize.

Notes on performance:
- Using 256 threads per block keeps shared-memory contention reasonable (8 warps/block),
  and with 32KB of dynamic shared memory for the worst case (256 bins), multiple blocks can be active per SM.
- The final reduction of 32 copies per bin into global memory uses per-bin atomicAdd. Since the number of bins
  is small (<= 256), this phase is negligible relative to scanning the input.
*/

#ifndef __CUDACC_RTC__
#define CUDA_FORCEINLINE __forceinline__
#else
#define CUDA_FORCEINLINE
#endif

// Controls how many input characters each thread processes per grid-stride iteration.
// Chosen for modern NVIDIA GPUs with large inputs to amortize control overhead while keeping register pressure modest.
static constexpr int itemsPerThread = 16;

// CUDA kernel: computes the histogram for characters in [from, from + numBins - 1]
// input     : device pointer to input characters (0..255 values are expected)
// histogram : device pointer to result array of length numBins (initialized to 0 by host)
// n         : number of input characters
// from      : starting character (inclusive)
// numBins   : number of bins (to - from + 1)
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int n,
                                       unsigned int from,
                                       unsigned int numBins)
{
    // Shared memory layout:
    // 32 copies of the histogram, each strided by 32 so that bin i of copy c is at s_hist[i * 32 + c].
    extern __shared__ unsigned int s_hist[];
    const int lane = threadIdx.x & 31; // 0..31

    // Zero out the shared histogram (all 32 copies)
    for (unsigned int i = threadIdx.x; i < numBins * 32u; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop with unrolling by itemsPerThread to improve coalescing:
    // For a fixed k in [0, itemsPerThread), threads within a warp access consecutive elements.
    const unsigned int gtid   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int base = gtid; base < n; base += stride * itemsPerThread) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            unsigned int idx = base + static_cast<unsigned int>(k) * stride;
            if (idx < n) {
                // Load as unsigned to get 0..255 regardless of host 'char' signedness
                unsigned int u = static_cast<unsigned int>(input[idx]);
                // Single-range check using unsigned arithmetic: (u - from) < numBins
                unsigned int bin = u - from;
                if (bin < numBins) {
                    // Update the lane's private copy to avoid intra-warp bank conflicts
                    atomicAdd(&s_hist[bin * 32u + static_cast<unsigned int>(lane)], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 copies to a single histogram and add to global memory.
    // Each thread reduces multiple bins in a strided fashion.
    for (unsigned int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += s_hist[i * 32u + static_cast<unsigned int>(c)];
        }
        if (sum) {
            atomicAdd(&histogram[i], sum);
        }
    }
}

// Host API: launches the histogram kernel for the specified range [from, to].
// input      : device pointer to char array (cudaMalloc'd) containing the text
// histogram  : device pointer to unsigned int array (cudaMalloc'd) of size (to - from + 1)
// inputSize  : number of chars in 'input'
// from, to   : inclusive range of character ordinals to count; 0 <= from < to <= 255
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins for the requested range
    const unsigned int ufrom   = static_cast<unsigned int>(from);
    const unsigned int uto     = static_cast<unsigned int>(to);
    const unsigned int numBins = uto - ufrom + 1u;

    // Zero the output histogram on device (caller handles sync if needed)
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose launch configuration
    const int blockSize = 256; // 8 warps; good balance for shared memory and contention
    unsigned int gridSize = 0;

    // Aim for one pass over the data with itemsPerThread elements per thread (grid-stride handles larger inputs).
    // This keeps the number of blocks reasonable for both small and very large inputs.
    const unsigned long long chunk = static_cast<unsigned long long>(blockSize) * static_cast<unsigned long long>(itemsPerThread);
    gridSize = static_cast<unsigned int>((static_cast<unsigned long long>(inputSize) + chunk - 1ull) / chunk);
    if (gridSize == 0) gridSize = 1;

    // Dynamic shared memory: 32 copies of the histogram, each with numBins entries
    const size_t sharedMemSize = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Cast input to unsigned char* to avoid reinterpreting sign on device
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);

    // Launch the kernel
    histogram_range_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, histogram, inputSize, ufrom, numBins);
}