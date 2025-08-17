#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

/*
  High-performance histogram computation on modern NVIDIA GPUs (A100/H100).
  - The kernel computes a histogram for the specified inclusive character range [from, to].
  - Input is a device pointer to a char buffer (text); output is a device pointer to an array
    of unsigned int counts of length (to - from + 1).
  - The implementation uses shared memory privatization per warp to greatly reduce contention.
  - Shared-memory atomics are used to update the warp-private histograms, followed by a
    per-block reduction and atomic adds to the global histogram.

  itemsPerThread:
  - Controls how many input characters each thread processes per grid-stride iteration.
  - Selected default: 32, which provides a good balance between instruction overhead and
    memory throughput on A100/H100 for large inputs.
*/

constexpr int kBlockSize = 256;       // Reasonable default for occupancy and shared-memory usage
constexpr int itemsPerThread = 32;    // Tunable: number of characters processed per thread per iteration

// Kernel implementing a warp-private shared-memory histogram for range [from, from + rangeLen - 1]
template <int kItemsPerThread>
__global__ void histogramRangeKernel(const char* __restrict__ input,
                                     unsigned int* __restrict__ histogram,
                                     unsigned int inputSize,
                                     int from,
                                     int rangeLen)
{
    // Each warp owns a private histogram in shared memory to reduce contention.
    // We pad the histogram size to a multiple of 32 to reduce shared-memory bank conflicts.
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int warpsPerBlock = (bdim + warpSize - 1) / warpSize;
    const int warpId = tid / warpSize;

    const int histStride = (rangeLen + 31) & ~31; // round up to next multiple of 32
    extern __shared__ unsigned int sHist[];       // size: warpsPerBlock * histStride

    // Zero the shared histograms
    for (int i = tid; i < warpsPerBlock * histStride; i += bdim)
        sHist[i] = 0u;
    __syncthreads();

    // Grid-stride loop, each thread processes kItemsPerThread characters per outer iteration
    const size_t gridStride = static_cast<size_t>(bdim) * static_cast<size_t>(gridDim.x);
    size_t idxBase = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(bdim) + static_cast<size_t>(tid);

    const int fr = from;
    const unsigned int len = static_cast<unsigned int>(rangeLen);

    for (size_t idx = idxBase; idx < static_cast<size_t>(inputSize); idx += gridStride * kItemsPerThread)
    {
        #pragma unroll
        for (int k = 0; k < kItemsPerThread; ++k)
        {
            size_t i = idx + static_cast<size_t>(k) * gridStride;
            if (i >= static_cast<size_t>(inputSize)) break;

            // Read character and convert to [0..255]
            unsigned int c = static_cast<unsigned char>(input[i]);

            // Compute bin index relative to 'from'; range check using unsigned compare:
            // If 0 <= bin < len, the character is in the requested range.
            int bin = static_cast<int>(c) - fr;
            if (static_cast<unsigned int>(bin) < len)
            {
                // Update warp-private histogram in shared memory
                atomicAdd(&sHist[warpId * histStride + bin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce warp-private histograms and update global histogram
    for (int bin = tid; bin < rangeLen; bin += bdim)
    {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w)
            sum += sHist[w * histStride + bin];

        if (sum)
            atomicAdd(&histogram[bin], sum);
    }
}

// Host function to launch the histogram kernel.
// input      - device pointer to input char buffer
// histogram  - device pointer to output array of length (to - from + 1)
// inputSize  - number of characters in input
// from, to   - inclusive character range [from, to] (0 <= from < to <= 255)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate trivial cases (also guards against undefined behavior if inputs are invalid)
    if (!input || !histogram) return;
    if (inputSize == 0) return;
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const int rangeLen = to - from + 1;

    // Zero the output histogram on device (required because we use atomicAdd)
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Determine grid size:
    // - For a memory-bound kernel, a few waves of blocks per SM is typically best.
    // - Also consider the data-driven minimal number of blocks.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const unsigned int blocksByData =
        (inputSize + (kBlockSize * itemsPerThread) - 1) / (kBlockSize * itemsPerThread);
    const unsigned int blocksBySM = static_cast<unsigned int>(prop.multiProcessorCount) * 8u; // 8x SMs
    unsigned int gridSize = blocksByData > 0 ? blocksByData : 1u;
    if (gridSize > blocksBySM) gridSize = blocksBySM;
    if (gridSize > 65535u) gridSize = 65535u; // 1D grid limit

    // Compute dynamic shared memory size: warpsPerBlock * paddedBins
    /// @FIXED
    int warpSize = 0;
    /// @FIXED
    cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device);
    const int warpsPerBlock = (kBlockSize + warpSize - 1) / warpSize;
    const int histStride = ((rangeLen + 31) / 32) * 32;
    const size_t shmemSize = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(histStride) * sizeof(unsigned int);

    // Launch kernel
    histogramRangeKernel<itemsPerThread><<<gridSize, kBlockSize, shmemSize>>>(
        input, histogram, inputSize, from, rangeLen
    );
}
