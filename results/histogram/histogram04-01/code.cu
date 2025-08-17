#include <cuda_runtime.h>
#include <cstdint>

// High-level overview:
// - This kernel computes a histogram over a specified contiguous character range [from, to].
// - Each block maintains a private histogram in shared memory to minimize global memory atomics.
// - Threads process multiple input characters each (itemsPerThread) in a coalesced access pattern.
// - After processing, each block atomically accumulates its shared histogram into global memory.
// - The host function run_histogram() prepares and launches the kernel and zeroes the output.
//
// Key design choices for modern data center GPUs (A100/H100):
// - THREADS_PER_BLOCK = 256 is a good default for occupancy and shared-memory atomic throughput.
// - itemsPerThread = 16 balances instruction overhead and memory bandwidth utilization for large inputs.
// - The read pattern uses "k * total_threads + thread_id" indexing to maximize coalescing.
//
// Notes:
// - Input is treated as bytes; we cast char to unsigned char to correctly handle signed char.
// - Only characters in [from, to] are counted; others are ignored.
// - The output histogram must have length (to - from + 1) and is zeroed before launching the kernel.
// - No host-device sync is performed here; the caller is responsible for synchronization.

static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int itemsPerThread    = 16;

__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from, int to)
{
    extern __shared__ unsigned int s_hist[];

    const int nbins = to - from + 1; // Assumed valid (1..256)
    // Initialize shared-memory privatized histogram
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    // Coalesced read pattern: each iteration k processes a contiguous "stripe"
    // across threads, ensuring good global memory coalescing.
    #pragma unroll
    for (int k = 0; k < itemsPerThread; ++k) {
        unsigned int idx = t + k * total_threads;
        if (idx < inputSize) {
            unsigned char uc = input[idx];
            // Compute bucket relative to 'from'; use unsigned comparison to avoid branching on two conditions.
            int bucket = static_cast<int>(uc) - from;
            if (static_cast<unsigned int>(bucket) < static_cast<unsigned int>(nbins)) {
                atomicAdd(&s_hist[bucket], 1u);
            }
        }
    }

    __syncthreads();

    // Accumulate per-block histogram into global histogram
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

static inline unsigned int ceil_div(unsigned int a, unsigned int b) {
    return (a + b - 1u) / b;
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Handle trivial cases early
    if (inputSize == 0u) {
        // Zero the output histogram for consistency
        const unsigned int nbins0 = static_cast<unsigned int>(to - from + 1);
        if (nbins0 > 0u) {
            cudaMemset(histogram, 0, nbins0 * sizeof(unsigned int));
        }
        return;
    }

    const unsigned int nbins = static_cast<unsigned int>(to - from + 1);
    // Ensure the output histogram starts from zero
    cudaMemset(histogram, 0, nbins * sizeof(unsigned int));

    // Compute launch configuration:
    // Each thread processes itemsPerThread items, so total coverage per block is THREADS_PER_BLOCK * itemsPerThread.
    unsigned int blocks = ceil_div(inputSize, static_cast<unsigned int>(THREADS_PER_BLOCK * itemsPerThread));
    if (blocks == 0u) blocks = 1u;

    const dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    const dim3 gridDim(blocks, 1, 1);

    const size_t shmemBytes = static_cast<size_t>(nbins) * sizeof(unsigned int);
    const unsigned char* uinput = reinterpret_cast<const unsigned char*>(input);

    histogram_range_kernel<<<gridDim, blockDim, shmemBytes>>>(uinput, histogram, inputSize, from, to);
}