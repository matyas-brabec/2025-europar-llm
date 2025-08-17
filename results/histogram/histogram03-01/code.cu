#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel that computes the histogram for the character range [from, to].
// This kernel uses per-warp privatized histograms in shared memory to reduce
// contention on global memory atomics. Each warp accumulates into its own
// sub-histogram in shared memory, and after processing input, the per-warp
// histograms are reduced to a single result which is atomically added into
// the global histogram output (of size (to - from + 1)).
//
// Parameters:
//   input      - device pointer to input characters
//   histogram  - device pointer to output histogram (length = to - from + 1)
//   inputSize  - number of chars in input
//   from, to   - inclusive range [from, to], 0 <= from <= to <= 255
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from, int to)
{
    // Compute the number of bins in the requested range.
    const int rangeLen = to - from + 1;
    if (rangeLen <= 0 || inputSize == 0) {
        return;
    }

    // Per-warp histogram privatization in shared memory
    // Layout: s_hist[warpId][bin], flattened as warpId*rangeLen + bin
    extern __shared__ unsigned int s_hist[];
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;        // threadIdx.x / 32
    const int warpCount = blockDim.x >> 5;

    // Zero initialize the entire shared memory histogram (all warps)
    for (int i = tid; i < rangeLen * warpCount; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input to allow arbitrary grid size
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + tid; idx < inputSize; idx += stride) {
        // Treat input as unsigned to get 0..255 ordinal
        const unsigned int c = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
        // Fast in-range test: (c - from) < rangeLen, using unsigned arithmetic
        const unsigned int delta = c - static_cast<unsigned int>(from);
        if (delta < static_cast<unsigned int>(rangeLen)) {
            // Atomic add into this warp's private shared-memory histogram
            // Using shared memory atomics (fast on modern GPUs)
            atomicAdd(&s_hist[warpId * rangeLen + static_cast<int>(delta)], 1u);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a single per-block result and write to global
    // Each thread handles multiple bins with a stride of blockDim.x
    for (int bin = tid; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0u;
        // Accumulate contributions from all warps for this bin
        for (int w = 0; w < warpCount; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }
        // Atomically add to global histogram (one atomic per bin per block)
        if (sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
    // No need to __syncthreads() here since kernel is ending
}

// Host function that configures and launches the CUDA kernel.
//
// The input and histogram buffers are assumed to reside in device memory
// (allocated with cudaMalloc). The histogram buffer must have length
// (to - from + 1) elements. This function zeroes the histogram and launches
// the kernel. Synchronization (e.g., cudaDeviceSynchronize) is managed by
// the caller.
//
// Parameters:
//   input      - device pointer to input characters
//   histogram  - device pointer to output histogram (length = to - from + 1)
//   inputSize  - number of chars in input
//   from, to   - inclusive range [from, to], 0 <= from < to <= 255
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate parameters conservatively; silently return on invalid input.
    if (!input || !histogram || inputSize == 0u) {
        return;
    }
    if (from < 0 || to < 0 || from > to || to > 255) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Zero the output histogram; required since kernel performs atomicAdd into it.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int), 0);

    // Choose a good default block size. 256 threads hits a sweet spot on A100/H100,
    // and keeps shared memory usage modest for up to 256 bins.
    const int blockSize = 256;
    const int warpCount = blockSize / 32;

    // Dynamic shared memory size per block: per-warp histogram
    const size_t smemBytes = static_cast<size_t>(rangeLen) * warpCount * sizeof(unsigned int);

    // Determine a grid size based on occupancy
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, histogram_range_kernel, blockSize, smemBytes);
    if (blocksPerSM <= 0) {
        // Fallback in the unlikely case occupancy calculation fails
        blocksPerSM = 1;
    }

    // Use an occupancy-based grid size with some oversubscription to cover latency
    int gridSize = props.multiProcessorCount * blocksPerSM * 2;
    if (gridSize <= 0) gridSize = 1;

    // Launch kernel on default stream
    histogram_range_kernel<<<gridSize, blockSize, smemBytes>>>(
        input, histogram, inputSize, from, to
    );

    // Error reporting is optional; synchronization is handled by the caller.
    // cudaGetLastError(); // Optionally query launch errors
}