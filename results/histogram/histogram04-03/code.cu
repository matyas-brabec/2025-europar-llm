#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel: compute histogram of input chars restricted to [from, to].
// Optimized for modern NVIDIA GPUs using per-warp privatized histograms in shared memory.
// Each thread processes 'itemsPerThread' items per grid-stride iteration to increase memory throughput.
namespace {
    // Tunable constants for launch configuration and per-thread workload.
    // itemsPerThread controls how many input chars each thread processes per "tile".
    // 8 is a good default on A100/H100 for large inputs; try 16 for even higher bandwidth if registers allow.
    static constexpr int BLOCK_SIZE = 256;     // Threads per block (must be multiple of 32)
    static constexpr int itemsPerThread = 8;   // Unroll factor / items processed per thread per tile
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size (32).");
    static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

    // Padding to reduce shared memory bank conflicts across adjacent bins.
    // 32 provides a full bank stride separation between adjacent warps.
    static constexpr int SHMEM_PAD = 32;

    __global__ void histogram_range_kernel(const unsigned char* __restrict__ d_input,
                                           unsigned int n,
                                           unsigned int* __restrict__ d_hist,
                                           int from, int to)
    {
        // Compute range length once and share among threads
        const int rangeLen = to - from + 1;

        // Warp info
        const int warpId  = threadIdx.x / warpSize;
        const int laneId  = threadIdx.x & (warpSize - 1);
        const int numWarps = blockDim.x / warpSize;

        // Shared memory layout: per-warp privatized histograms with padding to reduce bank conflicts.
        extern __shared__ unsigned int smem[];
        const int shPitch = rangeLen + SHMEM_PAD; // pitch per warp-hist in shared memory
        unsigned int* warpHist = smem + warpId * shPitch;

        // Zero initialize all shared histograms (all warps) cooperatively
        for (int i = threadIdx.x; i < shPitch * numWarps; i += blockDim.x) {
            smem[i] = 0u;
        }
        __syncthreads();

        // Grid-stride "tiled" loop: each block covers BLOCK_SIZE * itemsPerThread items per iteration
        const size_t blockTile = static_cast<size_t>(blockDim.x) * itemsPerThread;
        size_t base = static_cast<size_t>(blockIdx.x) * blockTile + threadIdx.x;
        const size_t gridTileStride = static_cast<size_t>(gridDim.x) * blockTile;

        // Process items in a coalesced fashion:
        // For each tile, threads read contiguous positions separated by blockDim.x,
        // unrolled by itemsPerThread, then advance by gridTileStride to the next tile.
        while (base < n) {
            #pragma unroll
            for (int j = 0; j < itemsPerThread; ++j) {
                size_t idx = base + static_cast<size_t>(j) * blockDim.x;
                if (idx < n) {
                    unsigned int c = static_cast<unsigned int>(d_input[idx]); // 0..255
                    // Branchless range check: valid if 0 <= (c - from) < rangeLen
                    int bin = static_cast<int>(c) - from;
                    if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(rangeLen)) {
                        // Shared memory atomic add: fast on modern GPUs, reduces global contention
                        atomicAdd(&warpHist[bin], 1u);
                    }
                }
            }
            base += gridTileStride;
        }
        __syncthreads();

        // Reduce per-warp histograms into global histogram.
        // Each thread reduces a subset of bins to increase parallelism.
        for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
            unsigned int sum = 0;
            // Accumulate across all warps in the block for this bin
            for (int w = 0; w < numWarps; ++w) {
                sum += smem[w * shPitch + bin];
            }
            // Global accumulation across blocks requires atomics
            atomicAdd(&d_hist[bin], sum);
        }
        // No __syncthreads() needed here; kernel ends after all threads have written.
    }
}

// Host-side wrapper to run the histogram kernel.
// - input: device pointer to the input text buffer (chars) allocated by cudaMalloc
// - histogram: device pointer to output histogram array of length (to - from + 1), allocated by cudaMalloc
// - inputSize: number of chars in input
// - from, to: inclusive character ordinal range [from, to] (0 <= from < to <= 255)
//
// Note:
// - This function zeroes the output histogram before launching the kernel.
// - It performs no explicit device synchronization; the caller is responsible for synchronization if needed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Early exit: empty input or invalid range length (defensive; caller guarantees valid range)
    const int rangeLen = to - from + 1;
    if (inputSize == 0 || rangeLen <= 0) {
        if (rangeLen > 0) {
            cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));
        }
        return;
    }

    // Zero-initialize the output histogram
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Kernel launch configuration
    const int blockSize = BLOCK_SIZE;
    const unsigned int elemsPerBlock = static_cast<unsigned int>(blockSize * itemsPerThread);
    // Number of tiles (blocks) to cover the input once. If larger than SM capacity, the scheduler will run in waves.
    unsigned int gridSize = (inputSize + elemsPerBlock - 1) / elemsPerBlock;
    if (gridSize == 0) gridSize = 1;

    // Dynamic shared memory size: per-warp histograms with padding
    const int numWarps = blockSize / 32;
    const int shPitch = rangeLen + SHMEM_PAD;
    const size_t shmemBytes = static_cast<size_t>(shPitch) * static_cast<size_t>(numWarps) * sizeof(unsigned int);

    // Launch the kernel on the default stream
    histogram_range_kernel<<<gridSize, blockSize, shmemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        inputSize,
        histogram,
        from,
        to
    );
}