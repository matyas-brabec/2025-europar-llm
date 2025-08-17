#include <cuda_runtime.h>
#include <stdint.h>

// Histogram for a restricted character range using shared-memory privatization
// optimized for modern NVIDIA data center GPUs (A100/H100).
//
// Key optimizations and design choices:
// - Each thread processes itemsPerThread characters per grid-stride iteration
//   to reduce control overhead and improve L2/L1 cache/L2 sector utilization.
// - Per-warp privatized histograms in shared memory reduce both atomic contention
//   and shared memory bank conflicts compared to a single per-block histogram.
// - Using shared memory atomics (fast on recent GPUs) for warp-private updates.
// - Final accumulation from shared memory to global memory uses atomicAdd to
//   safely merge counts across blocks.
//
// Notes:
// - The histogram covers characters in the inclusive range [from, to].
// - Input is treated as unsigned characters to avoid sign-extension issues.
// - The output histogram must have length (to - from + 1).
// - The run_histogram() function expects device pointers (cudaMalloc-ed) for
//   both input and histogram and assumes the caller handles synchronization.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable constant: number of input characters processed by each thread per loop iteration.
// 8 is a good default for large inputs on modern GPUs (balanced ILP vs. register pressure).
static constexpr int itemsPerThread = 8;

// CUDA kernel: compute histogram for characters in [from, to].
// input: device pointer to input text buffer (bytes)
// gHist: device pointer to output histogram array with length (to - from + 1)
// inputSize: number of bytes in input
// from, to: inclusive character range [from, to]
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ gHist,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int sSubHist[]; // Layout: per-warp privatized histograms back-to-back

    const int rangeLen = to - from + 1;
    if (rangeLen <= 0) return; // guard

    const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warpId    = threadIdx.x >> 5;
    const int warpsPerBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Base pointer to this warp's private histogram in shared memory
    unsigned int* warpHist = sSubHist + warpId * rangeLen;

    // Initialize all per-warp histograms in the block to zero.
    // Use a block-wide loop to avoid redundant synchronizations and keep it simple.
    for (int i = threadIdx.x; i < warpsPerBlock * rangeLen; i += blockDim.x) {
        sSubHist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop where every thread processes 'itemsPerThread' items per stride.
    const unsigned int threadsPerGrid = gridDim.x * blockDim.x;
    const unsigned long long strideBytes = (unsigned long long)threadsPerGrid * itemsPerThread;

    for (unsigned long long base = (unsigned long long)tid * itemsPerThread;
         base < (unsigned long long)inputSize;
         base += strideBytes)
    {
        // Process up to itemsPerThread items starting at 'base'.
        // Guard each access by bounds check for the tail.
#pragma unroll
        for (int i = 0; i < itemsPerThread; ++i) {
            unsigned long long idx = base + (unsigned long long)i;
            if (idx < (unsigned long long)inputSize) {
                unsigned int c = static_cast<unsigned int>(input[idx]);
                // Fast range check for [from, to]
                if (c >= static_cast<unsigned int>(from) && c <= static_cast<unsigned int>(to)) {
                    // Shared-memory atomic add to the per-warp histogram bin
                    atomicAdd(&warpHist[c - static_cast<unsigned int>(from)], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into global histogram.
    // Each thread accumulates over warps for a subset of bins, then atomically adds to global.
    for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
#pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += sSubHist[w * rangeLen + bin];
        }
        if (sum) {
            atomicAdd(&gHist[bin], sum);
        }
    }
}

// Host helper to launch the histogram kernel.
// - input: device pointer to input bytes (cudaMalloc)
// - histogram: device pointer to output histogram array of size (to - from + 1) (cudaMalloc)
// - inputSize: number of bytes in input
// - from, to: inclusive character range [from, to]
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (from > to) return;
    const int rangeLen = to - from + 1;

    // Zero the output histogram; asynchronous memset is fine (caller handles sync)
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int), 0);

    // Choose a performant launch configuration.
    // blockDim: multiple of 32 for warp-based privatization; 256 is a good default.
    const int blockDimX = 256;

    // Determine a grid size that balances coverage and occupancy.
    // 'blocksForFullCoverage' is the number of blocks to cover the input with exactly one pass;
    // we cap it to a multiple of SMs to avoid launching an excessive number of blocks.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    const int smCount = props.multiProcessorCount;

    const unsigned long long workPerBlock = (unsigned long long)blockDimX * itemsPerThread;
    unsigned int blocksForFullCoverage = (inputSize == 0) ? 1u
        : static_cast<unsigned int>((((unsigned long long)inputSize + workPerBlock - 1ULL) / workPerBlock));
    // Cap blocks to a reasonable multiple of SMs (e.g., 32x) to keep scheduling efficient.
    unsigned int gridDimX = blocksForFullCoverage;
    const unsigned int cap = smCount > 0 ? static_cast<unsigned int>(smCount) * 32u : 1024u;
    if (gridDimX > cap) gridDimX = cap;
    if (gridDimX == 0) gridDimX = 1;

    // Shared memory size: one rangeLen-sized histogram per warp in the block.
    const int warpsPerBlock = (blockDimX + WARP_SIZE - 1) / WARP_SIZE;
    const size_t sharedMemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch kernel
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<gridDimX, blockDimX, sharedMemBytes>>>(d_input, histogram, inputSize, from, to);
}