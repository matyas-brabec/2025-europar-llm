#include <cuda_runtime.h>

// Optimized CUDA kernel to compute a histogram over a specified contiguous byte range [lower, upper].
// - Uses per-warp privatized histograms in shared memory to minimize global atomic contention.
// - Each block accumulates its private histogram in shared memory, then reduces across warps and
//   emits a single atomicAdd per bin to the global histogram.
// - The input is treated as raw bytes (unsigned char) so that all 0..255 values are handled correctly.
//
// Parameters:
//   input:     device pointer to input chars (treated as bytes)
//   globalHist: device pointer to output histogram (length = upper - lower + 1), initialized to zero by caller/wrapper
//   n:         number of input characters
//   lower:     inclusive lower bound of range (0..255)
//   upper:     inclusive upper bound of range (lower..255)
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ globalHist,
                                       unsigned int n,
                                       unsigned int lower,
                                       unsigned int upper)
{
    extern __shared__ unsigned int s_hist[]; // Layout: numWarps contiguous sub-histograms, each of size rangeLen

    const unsigned int tid      = threadIdx.x;
    const unsigned int bdim     = blockDim.x;
    const unsigned int gdim     = gridDim.x;
    const unsigned int warpSize = 32;
    const unsigned int warpId   = tid / warpSize;
    const unsigned int numWarps = (bdim + warpSize - 1) / warpSize;

    // Compute range length and prepare for a branchless in-range test:
    // A value v is in [lower, upper] iff (v - lower) <= (upper - lower) for unsigned arithmetic.
    const unsigned int range     = upper - lower;       // 0..255
    const unsigned int rangeLen  = range + 1u;          // 1..256

    // Zero the per-warp privatized shared histogram
    for (unsigned int i = tid; i < numWarps * rangeLen; i += bdim) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over input
    const unsigned int stride = bdim * gdim;
    for (unsigned int idx = blockIdx.x * bdim + tid; idx < n; idx += stride) {
        unsigned char v = static_cast<unsigned char>(input[idx]);
        unsigned int delta = static_cast<unsigned int>(v) - lower;
        if (delta <= range) {
            // Per-warp privatization reduces atomic contention across the block
            unsigned int bin = delta;
            atomicAdd(&s_hist[warpId * rangeLen + bin], 1u);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into a single histogram and commit to global memory.
    // Only the first rangeLen threads participate to avoid unnecessary work.
    if (tid < rangeLen) {
        unsigned int sum = 0u;
        // Accumulate counts for this bin across all warps in the block
        for (unsigned int w = 0; w < numWarps; ++w) {
            sum += s_hist[w * rangeLen + tid];
        }
        if (sum != 0u) {
            atomicAdd(&globalHist[tid], sum);
        }
    }
    // No need for a final __syncthreads() since the kernel ends here.
}

// Host wrapper to set up and launch the histogram kernel.
// - Assumes 'input' and 'histogram' point to device memory allocated via cudaMalloc.
// - 'inputSize' is the number of bytes in 'input'.
// - 'from' and 'to' specify the inclusive range; the output 'histogram' has length (to - from + 1).
// - This function zeroes the output histogram region before launching the kernel.
// - Synchronization (if desired) is left to the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Basic parameter handling (assumes valid, per problem statement)
    const unsigned int lower = static_cast<unsigned int>(from);
    const unsigned int upper = static_cast<unsigned int>(to);
    const unsigned int range = upper - lower;
    const unsigned int rangeLen = range + 1u;

    // Zero the output histogram asynchronously on the default stream
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int), 0);

    // Choose launch configuration
    const int THREADS_PER_BLOCK = 256;  // Good balance on A100/H100; 8 warps per block
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Aim for enough blocks to saturate the GPU without creating excessive global atomics.
    // Target each block to process ~32 KiB of data as a starting point; cap at 16x SMs.
    const size_t targetBytesPerThread = 128; // bytes per thread "target" (heuristic)
    const size_t targetBytesPerBlock  = THREADS_PER_BLOCK * targetBytesPerThread;
    int blocksByWork = static_cast<int>((inputSize + targetBytesPerBlock - 1) / targetBytesPerBlock);
    int maxBlocks    = prop.multiProcessorCount * 16; // 16x SMs typically saturates memory-bound kernels
    if (blocksByWork < 1) blocksByWork = 1;
    int blocks = (blocksByWork < maxBlocks) ? blocksByWork : maxBlocks;

    // Dynamic shared memory size: one privatized histogram per warp
    const int numWarps = (THREADS_PER_BLOCK + 31) / 32;
    const size_t sharedBytes = static_cast<size_t>(numWarps) * static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch
    histogram_range_kernel<<<blocks, THREADS_PER_BLOCK, sharedBytes>>>(input, histogram, inputSize, lower, upper);
}