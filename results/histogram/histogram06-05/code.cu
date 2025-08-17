#include <cuda_runtime.h>
#include <stdint.h>

// This kernel computes a histogram on the substring range [from, to] (inclusive) of the input characters.
// It uses shared-memory privatization with 32 copies of the histogram (one per warp lane) to minimize
// intra-warp shared memory bank conflicts. The layout is transposed as [bin][lane] so that each lane
// accesses a different bank for a given bin. Final per-bin reduction across the 32 copies is done via
// warp-level shuffles and atomically accumulated into the global histogram (one array of length to-from+1).
//
// Key optimization points:
// - itemsPerThread controls the number of input elements processed per thread to improve memory throughput.
// - 32 copies of the shared histogram (one per lane) ensure that during counting, threads in the same warp
//   never contend on the same bank for the same bin (bank index equals lane).
// - Reduction across the 32 copies uses warp shuffles. Each warp reduces disjoint sets of bins and only
//   lane 0 issues an atomicAdd to global memory per bin, reducing the number of atomics.
//
// Assumptions for best performance:
// - blocks with thread counts that are multiples of 32 (warp size).
// - large inputs for good occupancy and throughput on A100/H100 class GPUs.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable: how many input chars each thread processes. Chosen for modern GPUs (A100/H100) and large inputs.
static constexpr int itemsPerThread = 8;

__global__ void histogram_kernel_range_32copies(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ globalHist,
    unsigned int inputSize,
    int from,
    int to)
{
    const int numBins = to - from + 1;
    // Shared memory layout: smem[bin * 32 + lane]
    // For each bin, 32 lane-private counters are placed contiguously (stride == 32).
    // This ensures bank index = (base/4 + bin*32 + lane) % 32 == lane, i.e., no intra-warp conflicts.
    extern __shared__ unsigned int smem[];

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warpId = tid >> 5;                       // warp index within the block
    const int warpsPerBlock = blockDim.x >> 5;

    // Initialize shared histogram copies to zero using a conflict-free pattern:
    // Each warp lane initializes its own lane slot for a subset of bins.
    for (int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        smem[bin * WARP_SIZE + lane] = 0;
    }
    __syncthreads();

    // Process input: each block owns a contiguous chunk; each thread processes itemsPerThread elements with stride blockDim.x
    const unsigned int blockSpan = blockDim.x * itemsPerThread;
    unsigned int baseIndex = blockIdx.x * blockSpan + tid;

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + i * blockDim.x;
        if (idx < inputSize) {
            unsigned int c = static_cast<unsigned int>(input[idx]);
            // Count only characters in [from, to]
            if (c >= static_cast<unsigned int>(from) && c <= static_cast<unsigned int>(to)) {
                const int bin = static_cast<int>(c) - from;
                // Lane-private copy per bin; shared-memory atomic avoids inter-warp races.
                atomicAdd(&smem[bin * WARP_SIZE + lane], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 lane-private copies per bin within each warp and add to global histogram.
    // Each warp handles bins in a strided fashion over [0, numBins).
    for (int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        unsigned int val = smem[bin * WARP_SIZE + lane];

        // Warp-wide reduction (sum over 32 lanes)
        // After this, lane 0 holds the sum across all lane-private copies for this bin within the block.
        unsigned int mask = 0xFFFFFFFFu;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (lane == 0) {
            atomicAdd(&globalHist[bin], val);
        }
    }
    // No need for further synchronization; kernel is done.
}

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (inputSize == 0) return;
    const int numBins = to - from + 1;
    if (numBins <= 0) return;

    // Zero the output histogram (as we atomically accumulate into it).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Configure the kernel launch. Use a warp-multiple block size.
    const int threadsPerBlock = 256; // 8 warps; good balance for A100/H100 while limiting shared atomic contention
    // Each block processes threadsPerBlock * itemsPerThread input elements.
    const unsigned int workPerBlock = threadsPerBlock * itemsPerThread;
    unsigned int numBlocks = (inputSize + workPerBlock - 1) / workPerBlock;
    if (numBlocks == 0) numBlocks = 1;

    // Shared memory size: numBins * 32 lane copies.
    size_t shmemSize = static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    // Launch the kernel. The input is already on the device per problem statement.
    histogram_kernel_range_32copies<<<numBlocks, threadsPerBlock, shmemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}