#include <cuda_runtime.h>
#include <stdint.h>

// Tunable parameters chosen for modern NVIDIA data center GPUs (A100/H100):
// - THREADS_PER_BLOCK: 256 threads give good occupancy and balance with shared memory usage.
// - ITEMS_PER_THREAD: each thread processes multiple input characters to amortize overhead.
//   A value of 16 trades well between memory throughput and atomic contention for large inputs.
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_THREAD  = 16;

// CUDA kernel that computes a histogram for a specified character range [from, to].
// The input is treated as unsigned bytes to avoid issues with signed char on some compilers.
// The histogram is computed only for characters in the range and written to global_hist,
// which must have (to - from + 1) bins. The kernel uses per-warp privatized shared memory
// histograms to reduce global atomic contention, and then reduces to global memory.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int s_hist[]; // Layout: [warp0 bins][warp1 bins]...[warpN bins]

    const int numBins = to - from + 1;
    if (numBins <= 0) return; // Guard (shouldn't happen given preconditions)

    const int warp_id        = threadIdx.x >> 5;           // warpSize is 32
    const int warpsPerBlock  = (blockDim.x + 31) >> 5;
    const int warp_hist_base = warp_id * numBins;

    // Initialize the per-warp shared histograms to zero.
    for (int i = threadIdx.x; i < numBins * warpsPerBlock; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process ITEMS_PER_THREAD input items per thread in a block-striped pattern.
    // Base index for this thread and block.
    size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        size_t idx = base + static_cast<size_t>(it) * blockDim.x;
        if (idx >= inputSize) break; // Subsequent idx will also be out of bounds

        unsigned char c = input[idx];
        int bin = static_cast<int>(c) - from;

        // Efficient range check: ensures 0 <= bin < numBins using unsigned comparison
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            // Update this warp's privatized histogram in shared memory to reduce inter-warp contention.
            atomicAdd(&s_hist[warp_hist_base + bin], 1U);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // Use all threads in the block to sum across warps for each bin in a strided fashion.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * numBins + bin];
        }
        if (sum) {
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

// Host function that launches the histogram kernel.
// - input: device pointer to input chars (allocated with cudaMalloc)
// - histogram: device pointer to output histogram array of size (to - from + 1) (allocated with cudaMalloc)
// - inputSize: number of characters in input
// - from, to: inclusive character range [from, to], with 0 <= from < to <= 255
// Any host-device synchronization is assumed to be handled by the caller.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins and early exit if invalid or empty input.
    const int numBins = to - from + 1;
    if (numBins <= 0) {
        return;
    }

    // Ensure the output histogram is zero-initialized before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0) {
        return;
    }

    const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
    const unsigned int blockItems      = threadsPerBlock * ITEMS_PER_THREAD;
    const unsigned int numBlocks       = (inputSize + blockItems - 1) / blockItems;

    if (numBlocks == 0) {
        return;
    }

    // Shared memory size: one privatized histogram per warp.
    const unsigned int warpsPerBlock = (threadsPerBlock + 31) / 32;
    const size_t sharedMemBytes = static_cast<size_t>(numBins) * warpsPerBlock * sizeof(unsigned int);

    // Launch kernel. Cast input to unsigned char to avoid sign-extension issues.
    histogram_range_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}