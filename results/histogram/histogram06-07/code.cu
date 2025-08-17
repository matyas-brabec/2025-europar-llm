#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA kernel optimized for NVIDIA A100/H100 GPUs to compute a histogram over a specified
  character range [from, to] from an input char buffer.

  Optimization highlights:
  - Shared memory histogram privatization with 32 copies (one per warp lane) to avoid
    intra-warp bank conflicts during updates.
  - Copies are placed with a stride such that stride % 32 == 1 so that, for the same bin,
    warp lanes access distinct banks: bank = (lane * stride + bin) % 32 = (lane + bin) % 32.
    This yields a per-bin bank permutation across lanes, preventing conflicts.
  - Each thread processes itemsPerThread input items, with a grid-stride pattern that
    maintains fully coalesced global memory accesses for each iteration.
  - Block-local histograms are reduced across the 32 copies and then atomically added to
    the global histogram.

  Notes:
  - The global histogram is assumed to be zeroed by run_histogram() before kernel launch.
  - The caller is responsible for synchronization (as specified).
*/

// Tunable launch parameters (selected for modern data-center GPUs and large inputs).
// - 256 threads per block is a good balance between occupancy and shared memory usage.
// - itemsPerThread=16 makes each thread process 16 items in a grid-stride pattern, providing
//   high memory throughput while keeping register pressure reasonable.
static constexpr int kThreadsPerBlock  = 256;
static constexpr int kItemsPerThread   = 16;

// Compute a stride so that stride % 32 == 1 (with 32 banks and 4-byte words).
// We round bins up to a multiple of 32 and add +1 so that stride ≡ 1 (mod 32).
__host__ __device__ __forceinline__ int compute_hist_stride(int numBins) {
    int rounded = (numBins + 31) & ~31; // round up to multiple of 32
    return rounded + 1;                 // ensure stride % 32 == 1
}

template<int ITEMS_PER_THREAD>
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ g_hist,
                                       unsigned int N,
                                       int from, int to)
{
    // Number of bins we will compute for [from, to] inclusive
    const int numBins = to - from + 1;

    // Compute stride to distribute 32 copies across banks with stride % 32 == 1
    const int stride = compute_hist_stride(numBins);

    // 32 copies of the histogram, each of length 'stride'
    extern __shared__ unsigned int s_hist[]; // size = 32 * stride

    // Zero shared memory in parallel
    for (int i = threadIdx.x; i < 32 * stride; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Lane id within warp; each lane uses its own copy to avoid intra-warp conflicts
    const int lane = threadIdx.x & 31;
    unsigned int* lane_hist = s_hist + lane * stride;

    // Global thread id and total threads in grid for coalesced grid-stride access
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    // Process ITEMS_PER_THREAD items per thread using a grid-stride pattern:
    // At each iteration i, threads in a warp touch consecutive indices, ensuring coalesced loads.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = static_cast<unsigned int>(gtid) + static_cast<unsigned int>(i) * static_cast<unsigned int>(totalThreads);
        if (idx < N) {
            // Cast to unsigned char to map char [-128,127] to [0,255] if char is signed
            unsigned char c = static_cast<unsigned char>(input[idx]);
            int bin = static_cast<int>(c) - from;

            // Branchless in-range check: (unsigned)bin < (unsigned)numBins iff from <= c <= to
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                // Shared-memory atomicAdd is fast on modern GPUs
                atomicAdd(&lane_hist[bin], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce 32 copies into a single block histogram and add to global memory.
    // Each thread handles multiple bins if needed.
    for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
        unsigned int sum = 0;
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            sum += s_hist[l * stride + b];
        }
        if (sum) {
            atomicAdd(&g_hist[b], sum);
        }
    }
}

/*
  Host-side launcher.

  Parameters:
  - input: device pointer (cudaMalloc'd) to the input chars
  - histogram: device pointer (cudaMalloc'd) to the output histogram of size (to - from + 1)
  - inputSize: number of chars in the input buffer
  - from, to: inclusive character range [from, to], with 0 <= from < to <= 255

  Behavior:
  - Zeroes the output histogram
  - Launches the CUDA kernel with an appropriate grid/block configuration and shared memory size
  - Does not perform any explicit device synchronization (caller is responsible)
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int numBins = to - from + 1;

    // Zero the output histogram so we produce counts from scratch.
    // Caller handles synchronization; default stream enforces order with subsequent kernel.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0 || numBins <= 0) {
        return;
    }

    // Compute grid configuration so that total capacity covers the input.
    // Each thread processes kItemsPerThread items.
    const int threadsPerBlock = kThreadsPerBlock;
    const unsigned int totalThreadsNeeded = (inputSize + kItemsPerThread - 1) / kItemsPerThread;
    const int blocks = (static_cast<int>(totalThreadsNeeded) + threadsPerBlock - 1) / threadsPerBlock;

    if (blocks <= 0) {
        return;
    }

    // Compute dynamic shared memory size: 32 copies * stride
    const int stride = compute_hist_stride(numBins);
    const size_t sharedBytes = static_cast<size_t>(32) * static_cast<size_t>(stride) * sizeof(unsigned int);

    // Launch kernel (default stream). Caller will synchronize if needed.
    histogram_range_kernel<kItemsPerThread><<<blocks, threadsPerBlock, sharedBytes>>>(
        input, histogram, inputSize, from, to);
}