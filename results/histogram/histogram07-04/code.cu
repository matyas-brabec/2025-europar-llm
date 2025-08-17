#include <cuda_runtime.h>

// Number of items each thread processes (tuned for modern NVIDIA datacenter GPUs).
// This controls the amount of instruction-level parallelism per thread and the
// per-thread work granularity without excessively harming memory coalescing,
// thanks to the grid-stride loop pattern used below.
constexpr int itemsPerThread = 16;

// Number of privatized copies of the histogram per block to avoid shared memory
// bank conflicts. This must match the number of banks (32) and is selected so
// each thread uses the copy with index (threadIdx.x % 32).
constexpr int NUM_COPIES = 32;

// CUDA kernel: compute histogram for the character range [from, to] inclusive.
// - input: device pointer to chars
// - inputSize: number of chars
// - histogram: device pointer to output histogram of length (to - from + 1)
// - from/to: character bounds, 0 <= from < to <= 255
//
// Design:
// - Each block maintains a privatized shared-memory histogram with 32 copies to avoid
//   bank conflicts: bin i in copy c is stored at offset i*32 + c.
// - Each thread updates the copy with index (threadIdx.x % 32).
// - Updates are done with shared-memory atomics to handle inter-warp collisions.
// - After processing, per-block partial histograms are reduced across the 32 copies
//   and atomically added to the global histogram.
// - Each thread processes `itemsPerThread` items per outer grid-stride iteration,
//   preserving coalesced global reads when possible.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int inputSize,
                                       unsigned int* __restrict__ histogram,
                                       int from, int to)
{
    extern __shared__ unsigned int s_hist[]; // Size: (range * NUM_COPIES)
    const int range = to - from + 1;
    const int laneCopy = threadIdx.x & (NUM_COPIES - 1); // threadIdx.x % 32

    // Zero-initialize shared histograms
    for (int i = threadIdx.x; i < range * NUM_COPIES; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Total number of threads in the grid
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process input using a two-level grid-stride loop:
    // Outer stride walks chunks of itemsPerThread per thread,
    // Inner fixed-size loop processes up to itemsPerThread items,
    // maintaining coalesced reads for each inner iteration.
    for (unsigned int start = tid; start < inputSize; start += totalThreads * itemsPerThread) {
        unsigned int idx = start;
        #pragma unroll
        for (int it = 0; it < itemsPerThread; ++it) {
            if (idx < inputSize) {
                unsigned char uc = static_cast<unsigned char>(input[idx]);
                int r = static_cast<int>(uc) - from;
                // Range check without branching: cast to unsigned introduces a single bounds check
                if (static_cast<unsigned int>(r) < static_cast<unsigned int>(range)) {
                    // Shared histogram index with bank-conflict-free layout: i*32 + c
                    atomicAdd(&s_hist[r * NUM_COPIES + laneCopy], 1u);
                }
            }
            idx += totalThreads;
        }
    }

    __syncthreads();

    // Reduce 32 privatized copies per bin and add to global histogram
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        #pragma unroll
        for (int c = 0; c < NUM_COPIES; ++c) {
            sum += s_hist[bin * NUM_COPIES + c];
        }
        // Global accumulation across blocks
        atomicAdd(&histogram[bin], sum);
    }
}

// Host-side launcher.
// - input: device pointer to input chars
// - histogram: device pointer to output histogram buffer (length = to - from + 1),
//              allocated with cudaMalloc.
// - inputSize: number of chars in input
// - from, to: inclusive character range [from, to], 0 <= from < to <= 255
//
// Notes:
// - This function zeroes the output histogram on device before launching the kernel.
// - The caller is responsible for any necessary synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int range = to - from + 1;

    // Handle trivial case early
    if (inputSize == 0 || range <= 0) {
        // Ensure histogram is zeroed
        if (range > 0) {
            cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));
        }
        return;
    }

    // Zero output histogram
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Kernel launch configuration
    const int threads = 256; // Multiple of warp size, well-suited for Ampere/Hopper
    const unsigned int itemsPerStep = static_cast<unsigned int>(threads * itemsPerThread);
    unsigned int blocks = (inputSize + itemsPerStep - 1) / itemsPerStep;
    if (blocks == 0) blocks = 1;
    if (blocks > 65535u) blocks = 65535u; // Conservative cap compatible with all archs

    // Shared memory size: range bins, each with NUM_COPIES privatized counters
    const size_t shmemBytes = static_cast<size_t>(range) * NUM_COPIES * sizeof(unsigned int);

    histogram_range_kernel<<<blocks, threads, shmemBytes>>>(input, inputSize, histogram, from, to);
}