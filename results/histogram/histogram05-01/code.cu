#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100, H100).
 *
 * Key features:
 *  - Processes multiple input characters per thread (itemsPerThread) to increase ILP.
 *  - Uses shared memory with one privatized histogram per warp to reduce global atomic contention.
 *  - Adds padding to each warp's histogram to reduce shared memory bank conflicts between warps.
 *  - Aggregates per-warp histograms into a single global histogram with one atomicAdd per bin per block.
 *
 * The histogram is built for a contiguous character range [from, to] (inclusive).
 * Input characters are treated as unsigned bytes (0–255).
 */

static constexpr int itemsPerThread   = 8;    // Default number of input chars processed per thread.
static constexpr int threadsPerBlock  = 256;  // Must be a multiple of 32 (warp size).

// Sanity check at compile-time.
static_assert(threadsPerBlock % 32 == 0, "threadsPerBlock must be a multiple of warp size (32).");

/*
 * Device kernel: computes a histogram for characters in the range [from, to].
 *  - input:      device pointer to input chars.
 *  - histogram:  device pointer to output histogram (size = (to - from + 1) uints).
 *  - inputSize:  number of bytes in input.
 *  - from, to:   character range [from, to] (0 <= from <= to <= 255).
 */
__global__
void histogramKernel(const char * __restrict__ input,
                     unsigned int * __restrict__ histogram,
                     unsigned int inputSize,
                     int from,
                     int to)
{
    extern __shared__ unsigned int s_hist[];

    const int numBins        = to - from + 1;          // Number of histogram bins.
    const int warpSizeLocal  = 32;                     // Using 32 explicitly; warpSize is 32 on NVIDIA GPUs.
    const int warpsPerBlock  = blockDim.x / warpSizeLocal;

    // Add 1 element of padding per warp histogram to avoid alignment of same-bin entries across warps
    // to the same bank, which helps to reduce shared memory bank conflicts across warps.
    const int binsPerWarpHist = numBins + 1;

    const int threadId  = threadIdx.x;
    const int warpId    = threadId / warpSizeLocal;

    // Pointer to this warp's private histogram in shared memory.
    unsigned int *warpHist = s_hist + warpId * binsPerWarpHist;

    // Initialize the entire shared memory region (all warp copies).
    for (int i = threadId; i < warpsPerBlock * binsPerWarpHist; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    // Grid-stride loop with itemsPerThread unrolling.
    const size_t n        = static_cast<size_t>(inputSize);
    const size_t stride   = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    size_t globalThreadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadId;

    for (size_t base = globalThreadId; base < n; base += stride * itemsPerThread) {
#pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            size_t idx = base + static_cast<size_t>(k) * stride;
            if (idx >= n)
                break;

            unsigned char uc = static_cast<unsigned char>(input[idx]);
            int cv = static_cast<int>(uc);

            if (cv >= from && cv <= to) {
                int bin = cv - from;
                // Use atomicAdd on shared memory to handle concurrent updates from threads in the same warp.
                atomicAdd(&warpHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    for (int bin = threadId; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;

        // Accumulate this bin across all warp-private histograms in the block.
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * binsPerWarpHist + bin];
        }

        // Perform one atomic add per bin per block to global memory.
        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host function: launches the histogram kernel.
 *
 * Parameters:
 *  - input:      device pointer to input text (chars), allocated with cudaMalloc.
 *  - histogram:  device pointer to output histogram (unsigned int array),
 *                size must be at least (to - from + 1) * sizeof(unsigned int),
 *                allocated with cudaMalloc.
 *  - inputSize:  number of chars in the input buffer.
 *  - from, to:   character range [from, to], with 0 <= from < to <= 255.
 *
 * Notes:
 *  - This function zeroes the histogram buffer on the device before launching the kernel.
 *  - No device synchronization is performed here; the caller is responsible for synchronization.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0) {
        // Nothing to do; caller is responsible for ensuring histogram contents as needed.
        return;
    }

    const int numBins = to - from + 1;

    // Initialize output histogram on the device to zero.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Determine grid size: each block processes (threadsPerBlock * itemsPerThread) items per "round".
    unsigned int blocks =
        (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);
    if (blocks == 0) {
        blocks = 1;
    }

    // Clamp the grid size to the device's maximum grid dimension in X.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (blocks > static_cast<unsigned int>(prop.maxGridSize[0])) {
        blocks = static_cast<unsigned int>(prop.maxGridSize[0]);
    }

    const int warpsPerBlock  = threadsPerBlock / 32;
    const int binsPerWarpHist = numBins + 1;
    const size_t sharedMemSize =
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(binsPerWarpHist) * sizeof(unsigned int);

    // Launch the kernel. No synchronization is performed here.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}