#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel restricted to a specified character range [from, to].
  Key performance features for A100/H100:
    - Per-warp privatized histograms in shared memory (reduces intra-block contention).
    - Warp-aggregated shared-memory atomic updates using __match_any_sync to drastically
      lower the number of atomics when multiple lanes target the same bin.
    - Bank-conflict reduction via padded indexing in shared memory.
    - Grid-stride loop with unrolling for higher ILP.
  Notes:
    - 'input' and 'histogram' are device pointers allocated by cudaMalloc (as per problem statement).
    - The caller handles global synchronization; this function does not call cudaDeviceSynchronize().
    - The host wrapper zeros 'histogram' before launching the kernel.
*/

#ifndef HISTOGRAM_UNROLL
#define HISTOGRAM_UNROLL 4
#endif

// Compute padded index to mitigate shared memory bank conflicts.
// Inserts one padding slot after every 32 bins.
__device__ __forceinline__ int pad_index(int bin) {
    return bin + (bin >> 5); // bin + floor(bin/32)
}

// Kernel implementing the histogram for range [from, to] inclusive.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int n,
                                       int from, int to)
{
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warpId   = tid >> 5;
    const int nWarps   = blockDim.x >> 5;
    const int range    = to - from + 1;

    // Padded length for shared histogram to reduce bank conflicts
    const int paddedLen = range + (range >> 5);

    extern __shared__ unsigned int smem[];
    // Layout: nWarps privatized histograms, each of size paddedLen
    unsigned int* warp_hist = smem + warpId * paddedLen;

    // Initialize all shared histograms to zero
    for (int i = tid; i < paddedLen * nWarps; i += blockDim.x) {
        smem[i] = 0u;
    }
    __syncthreads();

    // Helper lambda: process single byte at index idx
    auto process_byte = [&](unsigned int idx) {
        // Load the input byte; cast to unsigned char to avoid sign-extension.
        unsigned char c = static_cast<unsigned char>(__ldg(input + idx));

        // Compute bin relative to 'from'. Use unsigned compare for in-range test.
        int bin = static_cast<int>(c) - from;
        bool inRange = static_cast<unsigned int>(bin) < static_cast<unsigned int>(range);

        // Warp-aggregated atomic to shared memory:
        // Assign unique sentinel keys for out-of-range lanes to prevent grouping.
        int key = inRange ? bin : (-1 - lane); // unique per lane when out of range

        // Restrict matching to active lanes (handles divergence correctly).
        unsigned int active = __activemask();
        unsigned int grp    = __match_any_sync(active, key);

        // One leader per group performs a single atomicAdd of the group's size.
        int leader = __ffs(grp) - 1;
        if (lane == leader && inRange) {
            int count = __popc(grp);
            int pidx  = pad_index(bin);
            atomicAdd(&warp_hist[pidx], count);
        }
    };

    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Unrolled grid-stride loop (process 4 iterations per round when possible)
    for (; i + (HISTOGRAM_UNROLL - 1) * stride < n; i += HISTOGRAM_UNROLL * stride) {
#pragma unroll
        for (int u = 0; u < HISTOGRAM_UNROLL; ++u) {
            process_byte(i + u * stride);
        }
    }
    // Remainder
    for (; i < n; i += stride) {
        process_byte(i);
    }

    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and flush to global.
    // Each thread handles multiple bins strided by blockDim.x.
    for (int bin = tid; bin < range; bin += blockDim.x) {
        unsigned int sum = 0u;
        int pidx = pad_index(bin);
        // Accumulate across warp-private histograms
        for (int w = 0; w < nWarps; ++w) {
            sum += smem[w * paddedLen + pidx];
        }
        // Atomically add block's total contribution to global histogram
        atomicAdd(&global_hist[bin], sum);
    }
}

/*
  Host wrapper to launch the histogram kernel.

  Parameters:
    - input: device pointer to input chars
    - histogram: device pointer to output histogram (size >= (to - from + 1) * sizeof(unsigned int))
    - inputSize: number of chars in input
    - from, to: inclusive range of character ordinals to count (0 <= from < to <= 255)

  Behavior:
    - Zeros the histogram buffer.
    - Launches the kernel with tuned configuration for A100/H100.
    - Does not synchronize; caller is responsible for synchronization.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int range = to - from + 1;
    if (range <= 0) return;

    // Zero the output histogram (device memory)
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Choose launch configuration:
    // - 512 threads per block (16 warps) offers a good balance of ILP and occupancy.
    // - Grid size is based on input size and capped to a multiple of SM count.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int threads = 512;                            // Must be a multiple of 32
    const int nWarps  = threads / 32;
    const int paddedLen = range + (range >> 5);
    const size_t shmemSize = static_cast<size_t>(paddedLen) * nWarps * sizeof(unsigned int);

    // Heuristic for number of blocks: enough to cover data with unrolling, capped by 20x SMs
    unsigned int blocks = (inputSize + (threads * HISTOGRAM_UNROLL) - 1) / (threads * HISTOGRAM_UNROLL);
    if (blocks == 0) blocks = 1;
    const unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 20u;
    if (blocks > maxBlocks) blocks = maxBlocks;

    // Launch the kernel
    histogram_range_kernel<<<blocks, threads, shmemSize>>>(input, histogram, inputSize, from, to);
}