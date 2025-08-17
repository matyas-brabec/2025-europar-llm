#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel for a specified character range [from, to].
  - Uses shared memory per block to privatize the histogram and reduce global atomics.
  - Each thread processes `itemsPerThread` input characters in a coalesced manner:
    idx = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x + k * blockDim.x
  - The final block-local histogram is reduced to global memory via atomicAdd (range <= 256).
  - The input characters are treated as unsigned bytes (0..255) to avoid sign issues with 'char'.
*/

// Tunable knob: number of input items processed per thread.
// On A100/H100, 16 is a good default for large inputs, balancing occupancy and memory throughput.
static constexpr int itemsPerThread = 16;

template <int ItemsPerThread>
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int rangeLen)
{
    extern __shared__ unsigned int s_hist[];

    // Initialize the per-block shared histogram.
    for (int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Coalesced processing: each iteration k, threads in a warp touch consecutive bytes.
    unsigned int base = blockIdx.x * blockDim.x * ItemsPerThread + threadIdx.x;

    #pragma unroll
    for (int k = 0; k < ItemsPerThread; ++k) {
        unsigned int idx = base + k * blockDim.x;
        if (idx < inputSize) {
            // Read character as unsigned to preserve 0..255 range regardless of 'char' signedness.
            unsigned char uc = static_cast<unsigned char>(input[idx]);

            // Compute bin relative to 'from'. The range check is done using unsigned comparison.
            int bin = static_cast<int>(uc) - from;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(rangeLen)) {
                atomicAdd(&s_hist[bin], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce per-block histogram into the global histogram.
    for (int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        unsigned int v = s_hist[i];
        if (v) {
            atomicAdd(&histogram[i], v);
        }
    }
}

/*
  Host-side launcher.
  - input and histogram are device pointers allocated with cudaMalloc.
  - histogram has (to - from + 1) entries and will be zeroed here before accumulation.
  - No synchronization is performed; the caller handles it if needed.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int rangeLen = to - from + 1;

    // Zero the output histogram on device to ensure correctness.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // If there's no input, we are done after zeroing.
    if (inputSize == 0) {
        return;
    }

    // Choose a good default configuration for modern data center GPUs.
    const int threadsPerBlock = 256;
    unsigned int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);
    if (blocks == 0) blocks = 1;

    // Shared memory size per block: one unsigned int per bin in the requested range.
    const size_t shmemSize = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    histogram_range_kernel<itemsPerThread><<<blocks, threadsPerBlock, shmemSize>>>(
        input, histogram, inputSize, from, rangeLen);
}