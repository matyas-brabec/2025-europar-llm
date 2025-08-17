#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

/*
  Optimized CUDA histogram for a restricted character range [from, to] with the following features:
  - Shared memory privatization with 32 copies to minimize bank conflicts.
  - Each thread updates the histogram copy selected by (threadIdx.x % 32), so warp lanes map 1:1 to banks.
  - Shared-memory layout is strided so that bin i of copy c is at offset i*32 + c, ensuring each copy occupies one bank.
  - Each thread processes 'itemsPerThread' contiguous bytes, chosen for high throughput on modern GPUs.
  - Grid-stride loop allows flexible grid sizing while keeping per-thread contiguous processing.
  - Per-block reduction over the 32 copies, then atomicAdd to the global histogram (only bins, not per-element).
  - Assumes 'input' and 'histogram' are device allocations (cudaMalloc), and 'histogram' has (to - from + 1) elements.
*/

#ifndef ITEMS_PER_THREAD
// Tuned default for modern NVIDIA data center GPUs (A100/H100) on large inputs.
// 16 contiguous bytes per thread provides good L2 utilization and coalescing with minimal register pressure.
static constexpr int itemsPerThread = 16;
#else
static constexpr int itemsPerThread = ITEMS_PER_THREAD;
#endif

// CUDA kernel to compute histogram over range [from, to].
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       size_t N,
                                       int from,
                                       int to)
{
    // Number of bins in requested range (inclusive).
    const int bins = to - from + 1;

    // Dynamic shared memory layout:
    // We allocate 32 copies of 'bins' counters, strided so that:
    //   copy c (0..31), bin i (0..bins-1) is located at: offset = i*32 + c
    // This places each copy of a given bin in a distinct bank, avoiding bank conflicts
    // when lanes (c) in a warp update their respective copy concurrently.
    extern __shared__ unsigned int shmem_hist[];

    // Lane index (0..31). We use lane as the copy selector.
    const int lane = threadIdx.x & 31;
    const int block_threads = blockDim.x;

    // Zero-initialize the shared histograms.
    // We have bins*32 elements to zero.
    for (int i = threadIdx.x; i < bins * 32; i += block_threads) {
        shmem_hist[i] = 0;
    }
    __syncthreads();

    // Global thread id
    const size_t gtid = static_cast<size_t>(blockIdx.x) * block_threads + threadIdx.x;
    // Grid-stride measured in items (bytes). Each thread processes itemsPerThread items per stride.
    const size_t grid_items_stride = static_cast<size_t>(gridDim.x) * block_threads * itemsPerThread;

    // Start index (in bytes) for this thread
    size_t base = gtid * static_cast<size_t>(itemsPerThread);

    // Process items in grid-stride fashion, 'itemsPerThread' contiguous bytes per thread per stride.
    while (base < N) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            size_t idx = base + static_cast<size_t>(k);
            if (idx >= N) break;
            unsigned char v = static_cast<unsigned char>(input[idx]);

            // Compute bin index relative to 'from' and check range via unsigned comparison:
            // (unsigned)bin < (unsigned)bins is equivalent to (from <= v <= to)
            int bin = static_cast<int>(v) - from;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(bins)) {
                // Update shared histogram copy selected by lane.
                // Multiple warps may share the same 'lane' (copy), so use atomicAdd in shared memory.
                atomicAdd(&shmem_hist[bin * 32 + lane], 1u);
            }
        }
        base += grid_items_stride;
    }

    __syncthreads();

    // Reduce the 32 copies per bin into a single value and add to the global histogram.
    // Each thread handles a subset of bins.
    for (int bin = threadIdx.x; bin < bins; bin += block_threads) {
        unsigned int sum = 0;
        const int base_off = bin * 32;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += shmem_hist[base_off + c];
        }
        // Accumulate into global histogram.
        atomicAdd(&global_hist[bin], sum);
    }
}

// Host-side launcher.
// - input: device pointer to input text (chars)
// - histogram: device pointer to output histogram with size (to - from + 1) unsigned ints
// - inputSize: number of chars in input
// - from, to: inclusive range [from, to], with 0 <= from <= to <= 255
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (inputSize == 0 || from > to) {
        // Nothing to do; ensure output is zeroed for correctness.
        if (from <= to) {
            const size_t bins = static_cast<size_t>(to - from + 1);
            cudaMemset(histogram, 0, bins * sizeof(unsigned int));
        }
        return;
    }

    const int bins = to - from + 1;
    const size_t shmem_bytes = static_cast<size_t>(bins) * 32u * sizeof(unsigned int);

    // Zero the global histogram output buffer before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Choose a reasonable block size for modern GPUs. 256 balances occupancy and register usage well.
    const int blockSize = 256;

    // Compute a grid size that both covers the data (for small inputs) and provides enough parallelism.
    // We use occupancy to get an upper bound on useful blocks, then clamp by required blocks to cover input.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        histogram_range_kernel,
        blockSize,
        shmem_bytes
    );

    const int smCount = prop.multiProcessorCount;
    unsigned int maxUsefulBlocks = (maxBlocksPerSM > 0 ? maxBlocksPerSM : 1) * (smCount > 0 ? smCount : 1);

    // Blocks required to cover the input if each thread processes 'itemsPerThread' items once.
    const size_t threadsNeeded = (static_cast<size_t>(inputSize) + itemsPerThread - 1) / itemsPerThread;
    unsigned int requiredBlocks = static_cast<unsigned int>((threadsNeeded + blockSize - 1) / blockSize);
    if (requiredBlocks == 0) requiredBlocks = 1;

    // Final grid size: don't exceed occupancy-saturating blocks, but cover small inputs.
    const unsigned int gridSize = (requiredBlocks < maxUsefulBlocks) ? requiredBlocks : maxUsefulBlocks;

    // Launch the kernel with dynamic shared memory.
    histogram_range_kernel<<<gridSize, blockSize, shmem_bytes>>>(
        input,
        histogram,
        static_cast<size_t>(inputSize),
        from,
        to
    );
}