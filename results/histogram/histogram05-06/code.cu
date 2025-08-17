#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel for character ranges on modern NVIDIA GPUs (A100/H100).

  Key design points:
  - itemsPerThread: Each thread processes this many input chars (strip-mined and unrolled
    for coalesced global memory loads).
  - Shared-memory histogram privatization: We use a per-block shared histogram to minimize
    global atomics. To avoid shared memory bank conflicts, the shared histogram is
    replicated across 32 lanes (one per SM bank). Each thread updates the replica that
    corresponds to its lane ID, ensuring that, within a warp, updates to the same bin
    go to distinct banks. Cross-warp updates still need atomicAdd to guarantee correctness.
  - Final reduction: Each block reduces its 32 replicated copies into a single per-bin sum
    and atomically adds it to the global histogram.
*/

#ifndef HISTO_ITEMS_PER_THREAD
// Empirically good default for large inputs on A100/H100. Tune as needed.
static constexpr int itemsPerThread = 16;
#else
static constexpr int itemsPerThread = HISTO_ITEMS_PER_THREAD;
#endif

// Number of shared memory banks (and warp lanes) on modern NVIDIA GPUs.
static constexpr int SHM_BANKS = 32;

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ out,
                                       unsigned int N,
                                       int from, int to)
{
    // Range length and sanity
    const int numBins = to - from + 1;
    if (numBins <= 0) return;

    // Shared memory layout:
    // For each bin, we store SHM_BANKS replicas (one per lane/bank).
    // Indexing: sh[bin * SHM_BANKS + lane]
    extern __shared__ unsigned int sh[];

    // Zero the shared histogram replicas in parallel.
    for (unsigned int i = threadIdx.x; i < (unsigned int)(numBins * SHM_BANKS); i += blockDim.x) {
        sh[i] = 0u;
    }
    __syncthreads();

    // Identify this thread's lane (0..31). Using this to select the per-lane replica.
    const int lane = threadIdx.x & (SHM_BANKS - 1);

    // Strip-mined, grid-stride traversal, arranged for coalesced reads:
    // For each "tile" of blockDim.x * itemsPerThread chars assigned per block,
    // we iterate itemsPerThread times, each time threads read consecutive addresses.
    const size_t blockSpan = (size_t)blockDim.x * (size_t)itemsPerThread;
    const size_t gridSpan  = (size_t)gridDim.x  * blockSpan;

    for (size_t base = (size_t)blockIdx.x * blockSpan; base < (size_t)N; base += gridSpan) {
        #pragma unroll
        for (int it = 0; it < itemsPerThread; ++it) {
            size_t idx = base + (size_t)it * (size_t)blockDim.x + (size_t)threadIdx.x;
            if (idx < (size_t)N) {
                // Read char and map to bin if in range.
                unsigned char uc = static_cast<unsigned char>(input[idx]);
                int bin = (int)uc - from;
                if ((unsigned)bin < (unsigned)numBins) {
                    // Per-lane replicated update to avoid bank conflicts within a warp.
                    // Atomic is still required to safely combine updates from different warps.
                    atomicAdd(&sh[bin * SHM_BANKS + lane], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the SHM_BANKS replicas for each bin into a single sum and commit to global memory.
    // Threads cooperate over bins; each thread handles a strided set of bins.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int l = 0; l < SHM_BANKS; ++l) {
            sum += sh[bin * SHM_BANKS + l];
        }
        if (sum) {
            atomicAdd(&out[bin], sum);
        }
    }
}

/*
  Host launcher:
  - input: pointer to device memory containing input chars
  - histogram: pointer to device memory for histogram of size (to - from + 1) uints
  - inputSize: number of chars in input
  - from, to: inclusive range [from, to] for which the histogram is computed

  Notes:
  - histogram is zeroed prior to kernel launch.
  - No synchronization is performed here; caller is responsible.
*/
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Handle trivial/invalid ranges gracefully
    if (inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Zero the output histogram (device memory), since the kernel uses atomicAdd.
    cudaMemset(histogram, 0, (size_t)numBins * sizeof(unsigned int));

    // Choose a sensible block size for A100/H100.
    const int blockSize = 256;  // 8 warps; good balance for occupancy and latency hiding.

    // Compute grid size targeting at least one block per SM and covering the input.
    // With grid-stride loops, overprovisioning is fine; cap at typical 1D grid limit 65535.
    const unsigned long long spanPerBlock = (unsigned long long)blockSize * (unsigned long long)itemsPerThread;
    unsigned int gridSize = (unsigned int)((inputSize + spanPerBlock - 1ull) / spanPerBlock);
    if (gridSize == 0) gridSize = 1;
    if (gridSize > 65535u) gridSize = 65535u;

    // Dynamic shared memory size: numBins bins replicated across SHM_BANKS lanes.
    const size_t sharedBytes = (size_t)numBins * (size_t)SHM_BANKS * sizeof(unsigned int);

    // Launch the kernel.
    histogram_range_kernel<<<gridSize, blockSize, sharedBytes>>>(input, histogram, inputSize, from, to);

    // Error checking is optional and no synchronization is performed here, per the problem statement.
    // cudaGetLastError();  // Uncomment if you wish to capture immediate launch errors.
}