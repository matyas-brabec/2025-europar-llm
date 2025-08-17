#include <cuda_runtime.h>

/*
  Histogram over a restricted [from, to] character range (inclusive), optimized with shared-memory privatization.

  - input: device pointer to a plain-text buffer of size inputSize bytes
  - histogram: device pointer to an array of (to - from + 1) unsigned ints
               This buffer will be zero-initialized by run_histogram before kernel launch.
  - from, to: 0 <= from < to <= 255 define an inclusive character range.
              The histogram bin 'i' corresponds to character code (from + i).

  Optimization notes:
  - Each thread block builds a private histogram in shared memory (size = to - from + 1 bins).
    Per-character updates use shared-memory atomics (fast on modern GPUs), substantially reducing
    contention on global memory atomics.
  - After processing its grid-stride range of the input, each block accumulates its shared histogram
    into the global histogram with a single pass, using global atomics per bin.
  - Dynamic shared memory size is small (<= 256 bins * 4 bytes = 1 KB), so occupancy remains high.

  Synchronization:
  - This function does not call cudaDeviceSynchronize(). The caller is responsible for synchronization.
*/

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int inputSize,
                                       int from,
                                       int to,
                                       unsigned int* __restrict__ global_hist)
{
    // Number of bins in the [from, to] inclusive range
    const int bins = to - from + 1;

    // Dynamic shared memory for per-block privatized histogram
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram to zero (strip-mined over threads in the block)
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over input
    const unsigned char* data = reinterpret_cast<const unsigned char*>(input);
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int idx = tid; idx < inputSize; idx += stride) {
        unsigned char c = data[idx];
        // Compute bin as (c - from). Use unsigned range check to fold two comparisons into one.
        int bin = static_cast<int>(c) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(bins)) {
            // Shared-memory atomic increment (fast on modern architectures)
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate this block's shared histogram into the global histogram.
    // Each thread contributes a strided subset of bins.
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0u) {
            atomicAdd(&global_hist[i], val);
        }
    }
}

// Host function to launch the kernel.
// Assumptions:
// - 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// - 'histogram' has capacity for (to - from + 1) unsigned ints.
// - The caller performs any required synchronization after this call.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute number of bins; assume valid inputs with 0 <= from < to <= 255
    const int bins = to - from + 1;

    // Zero-initialize the output histogram on device to ensure correctness
    // since the kernel uses atomicAdd for the final accumulation.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Choose a reasonable block size; 256 is a good balance for atomics and occupancy.
    const int blockSize = 256;

    // Determine a good grid size using occupancy, taking dynamic shared memory into account.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const size_t dynSmem = static_cast<size_t>(bins) * sizeof(unsigned int);
    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
                                                  histogram_range_kernel,
                                                  blockSize,
                                                  dynSmem);

    if (maxBlocksPerSM <= 0) {
        // Fallback: ensure at least one block per SM
        maxBlocksPerSM = 1;
    }

    int gridSize = maxBlocksPerSM * prop.multiProcessorCount;
    if (gridSize <= 0) {
        gridSize = 1;
    }

    // Launch kernel with dynamic shared memory sized to the number of bins.
    histogram_range_kernel<<<gridSize, blockSize, dynSmem>>>(input, inputSize, from, to, histogram);
}