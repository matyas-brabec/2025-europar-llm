#include <cuda_runtime.h>
#include <cstdint>

/*
  High-level overview:
  - This kernel computes a histogram for a specified contiguous character range [from, to].
  - The input is a device pointer to a char buffer of length inputSize.
  - The output is a device pointer to an array of unsigned int counters of length (to - from + 1).
  - The implementation uses a per-warp privatized histogram in shared memory to reduce global atomic contention.
    Each warp accumulates into its own sub-histogram using shared-memory atomicAdd (fast on A100/H100 for 32-bit).
    After processing, these per-warp histograms are reduced within the block and atomically added to global memory.
  - The kernel uses a grid-stride loop for high occupancy and to handle arbitrary input sizes efficiently.

  Notes:
  - Characters are treated as unsigned bytes (0..255), regardless of the host compiler's char signedness.
  - The wrapper function run_histogram zeroes the output histogram and launches the kernel.
  - Host-device synchronization is assumed to be handled by the caller; the wrapper does not synchronize.
*/

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable kernel block size; 512 threads balances occupancy and shared memory usage well on A100/H100.
#ifndef HIST_BLOCK_SIZE
#define HIST_BLOCK_SIZE 512
#endif

// CUDA kernel: compute histogram counts for characters in [from, from + range - 1] (range = to - from + 1)
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       size_t n,
                                       int from,
                                       int range)
{
    extern __shared__ unsigned int s_hist[]; // Size: warpsPerBlock * range

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int warpOffset = warpId * range;

    // Zero the per-warp shared-memory histograms
    for (int i = tid; i < warpsPerBlock * range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input buffer
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid; idx < n; idx += stride) {
        // Treat input bytes as unsigned to obtain ordinal 0..255
        unsigned char uc = static_cast<unsigned char>(input[idx]);
        // Compute bin index and check whether within [from, from + range - 1] using unsigned range check
        int bin = static_cast<int>(uc) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            // Accumulate into this warp's private sub-histogram in shared memory
            atomicAdd(&s_hist[warpOffset + bin], 1U);
        }
    }
    __syncthreads();

    // Reduce the per-warp histograms into the global histogram using one atomicAdd per bin and block
    for (int bin = tid; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate counts across all warps in the block for this bin
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * range + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host-side wrapper to launch the kernel. Assumes input and histogram are device pointers allocated with cudaMalloc.
// The histogram buffer must have length (to - from + 1).
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins and handle trivial cases
    int range = to - from + 1;
    if (range <= 0) {
        return; // Nothing to do for invalid range (defensive; caller promises 0 <= from < to <= 255)
    }

    // Zero the output histogram on device
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Select launch configuration
    const int blockSize = HIST_BLOCK_SIZE;
    int gridSize = static_cast<int>((static_cast<size_t>(inputSize) + blockSize - 1) / blockSize);
    if (gridSize <= 0) gridSize = 1;

    // Dynamic shared memory size: warpsPerBlock * range * sizeof(unsigned int)
    const int warpsPerBlock = blockSize / WARP_SIZE;
    const size_t shmemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch kernel (no explicit synchronization; caller is responsible)
    histogram_range_kernel<<<gridSize, blockSize, shmemBytes>>>(input, histogram, static_cast<size_t>(inputSize), from, range);
}