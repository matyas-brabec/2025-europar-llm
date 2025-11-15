#include <cuda_runtime.h>

// Number of characters processed by each thread.
// A value of 8 is a good balance for modern NVIDIA data-center GPUs (A100/H100)
// for large, memory-bound workloads like histograms.
static constexpr int itemsPerThread = 8;

// CUDA kernel: computes a partial histogram for a given [from, to] character range.
// The input text buffer is partitioned among blocks and threads. Each block builds
// a privatized histogram in shared memory using 32 bank-separated copies.
// Finally, each block atomically accumulates its partial histogram into the
// global histogram array.
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to) {
    // Dynamic shared memory layout:
    // We allocate (numBins * 32) 32-bit counters where:
    //   - numBins = to - from + 1
    //   - For bin 'b' (0 <= b < numBins) and lane 'l' (0 <= l < 32),
    //     the counter is at s_hist[b * 32 + l].
    //
    // This mapping ensures that for a fixed bin 'b', the 32 per-lane counters
    // are located in different banks (stride of 32), so that when a warp of
    // 32 threads updates the same bin using different lanes, there are no
    // shared-memory bank conflicts.
    extern __shared__ unsigned int s_hist[];

    const int warpSize = 32;
    const int lane     = threadIdx.x & (warpSize - 1);

    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Initialize shared histogram copies to zero.
    // We have numBins * warpSize counters in total.
    for (unsigned int i = threadIdx.x; i < numBins * warpSize; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    // Each block processes a contiguous chunk of the input:
    //   blockChunkSize = blockDim.x * itemsPerThread
    //
    // Within the block, processing is organized so threads in a warp access
    // consecutive locations for coalesced global memory loads:
    //
    //   blockBase = blockIdx.x * blockChunkSize
    //   for item = 0..itemsPerThread-1:
    //     idx = blockBase + item * blockDim.x + threadIdx.x
    //
    // This pattern ensures that for fixed 'item', threads in the block read
    // a contiguous span of the input, preserving coalescing.
    const size_t blockChunkSize = static_cast<size_t>(blockDim.x) * itemsPerThread;
    const size_t blockBase      = static_cast<size_t>(blockIdx.x) * blockChunkSize;

    for (int item = 0; item < itemsPerThread; ++item) {
        const size_t idx = blockBase + static_cast<size_t>(item) * blockDim.x + threadIdx.x;
        if (idx >= inputSize) {
            break;
        }

        // Load one character and interpret as unsigned to get ordinal 0..255,
        // regardless of whether char is signed or unsigned on the host.
        const unsigned char c  = static_cast<unsigned char>(input[idx]);
        const int           ci = static_cast<int>(c);

        // Only count characters within the requested [from, to] range.
        if (ci >= from && ci <= to) {
            const unsigned int bin = static_cast<unsigned int>(ci - from);  // 0 .. numBins-1
            const unsigned int sharedIndex = bin * warpSize + static_cast<unsigned int>(lane);

            // Multiple warps in a block share the same per-lane copies, so we
            // must use atomicAdd on shared memory to avoid write races between
            // warps that have the same lane ID and update the same bin.
            atomicAdd(&s_hist[sharedIndex], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane copies for each bin into a single count and
    // accumulate into the global histogram. Work is distributed across
    // threads in the block via a simple strided loop.
    for (unsigned int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        const unsigned int base = bin * warpSize;

        // Accumulate contributions from all 32 per-lane copies.
        #pragma unroll
        for (int l = 0; l < warpSize; ++l) {
            sum += s_hist[base + static_cast<unsigned int>(l)];
        }

        // Atomically add the block's bin contribution into the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function: configures and launches the histogram kernel.
//   - input: device pointer to input text (chars)
//   - histogram: device pointer to output histogram array of size (to - from + 1)
//   - inputSize: number of characters in the input buffer
//   - from, to: inclusive character ordinal range [from, to] to be counted
//
// The input and histogram arrays are assumed to be allocated with cudaMalloc.
// This function zero-initializes the histogram on the device, launches the kernel,
// and relies on the caller for any required synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    if (from > to) {
        return;  // invalid range; nothing to do
    }

    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // If there are no characters to process, simply zero the histogram and return.
    if (inputSize == 0) {
        cudaMemset(histogram, 0, numBins * sizeof(unsigned int));
        return;
    }

    // Zero the global histogram. The caller is assumed to want a fresh histogram.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Kernel launch configuration.
    // Use a block size that is a multiple of 32 for full warp utilization.
    const int blockSize = 256;

    // Each block processes blockSize * itemsPerThread characters.
    const unsigned int itemsPerBlock = static_cast<unsigned int>(blockSize) * itemsPerThread;

    // Number of blocks needed to cover the entire input.
    const unsigned int gridSize =
        (inputSize + itemsPerBlock - 1u) / itemsPerBlock;

    if (gridSize == 0) {
        // Should not happen when inputSize > 0, but guard for completeness.
        return;
    }

    // Dynamic shared memory size: numBins * 32 copies * sizeof(unsigned int)
    const size_t sharedMemSize = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}