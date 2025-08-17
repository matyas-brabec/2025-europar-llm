#include <cuda_runtime.h>
#include <cstdint>

/*
  Optimized CUDA histogram for a restricted character range [from, to].

  Key optimizations:
  - Each thread processes multiple input items (itemsPerThread) using a grid-stride loop for full coalescing.
  - Shared-memory privatization with one sub-histogram per warp lane (32 copies) to avoid bank conflicts.
  - A padding of +1 between sub-histograms ensures that the same bin across sub-histograms maps to different banks.
  - Per-block shared histograms are merged locally, then atomically added to the global histogram.

  Assumptions:
  - 'input' and 'histogram' are device pointers allocated with cudaMalloc.
  - 'inputSize' is the number of chars in the input buffer.
  - Caller handles host-device synchronization if needed.

  Output:
  - 'histogram' contains (to - from + 1) counters; bin i corresponds to char code (from + i).
*/

// Tunable constants chosen for modern NVIDIA data center GPUs (A100/H100).
static constexpr int THREADS_PER_BLOCK = 256;  // 8 warps per block is a good balance for occupancy and shared mem usage.
static constexpr int ITEMS_PER_THREAD  = 16;   // Each thread processes 16 chars; good throughput for large inputs on modern GPUs.
static constexpr int REPLICATES        = 32;   // One sub-histogram per warp lane to avoid bank conflicts within a warp.
static constexpr int PADDING           = 1;    // +1 padding so stride is not a multiple of 32 banks; avoids bank conflicts.

/*
  CUDA kernel:
  - input: device buffer of bytes (text)
  - inputSize: number of bytes to process
  - from, to: inclusive byte range [from, to] (0..255)
  - globalHist: device output histogram of length (to - from + 1)
*/
__global__ void histogram_kernel(const unsigned char* __restrict__ input,
                                 unsigned int inputSize,
                                 int from,
                                 int to,
                                 unsigned int* __restrict__ globalHist)
{
    // Compute the target range size and shared-memory layout.
    const int range  = to - from + 1;         // Number of bins we care about (1..256)
    const int stride = range + PADDING;       // Padded stride between replicated histograms
    extern __shared__ unsigned int shmem[];   // Layout: REPLICATES rows, each of length 'stride'

    // Zero the entire shared-memory replicated histograms.
    // All threads cooperate to initialize REPLICATES * stride elements.
    for (int i = threadIdx.x; i < REPLICATES * stride; i += blockDim.x) {
        shmem[i] = 0u;
    }
    __syncthreads();

    // Each warp lane writes to its own sub-histogram to avoid bank conflicts and reduce contention.
    const int lane   = threadIdx.x & 31;                // 0..31
    unsigned int* myHist = shmem + lane * stride;       // Pointer to this lane's private sub-histogram

    // Grid-stride loop with ITEMS_PER_THREAD unrolling for fully coalesced loads and fewer loop iterations.
    const size_t globalThreadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t gridStride     = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t base = globalThreadId; base < inputSize; base += gridStride * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
            const size_t pos = base + static_cast<size_t>(it) * gridStride;
            if (pos >= inputSize) break;

            const unsigned char v = input[pos];
            const int bin = static_cast<int>(v) - from;

            // Single-range check using unsigned comparison: valid if 0 <= bin < range
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
                // Atomic add to shared-memory per-lane sub-histogram
                atomicAdd(&myHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Merge the REPLICATES sub-histograms into a single per-block result and add to the global histogram.
    // Each thread reduces multiple bins to leverage parallelism.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0u;
        // Accumulate this bin across all replicated sub-histograms.
        #pragma unroll
        for (int r = 0; r < REPLICATES; ++r) {
            sum += shmem[r * stride + bin];
        }
        // Atomically add block's bin sum to global histogram (multiple blocks may contend).
        atomicAdd(&globalHist[bin], sum);
    }
}

/*
  Host wrapper to configure and launch the histogram kernel.

  Parameters:
  - input: device pointer to input text (char buffer)
  - histogram: device pointer to output histogram array of length (to - from + 1)
  - inputSize: number of chars in 'input'
  - from, to: inclusive range [from, to] in [0, 255], from < to
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute range length and shared memory requirements.
    const int range = to - from + 1;
    // Zero the output histogram buffer.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Choose launch configuration.
    // Use occupancy API to select number of blocks based on shared memory usage.
    const size_t shmemBytes = static_cast<size_t>(REPLICATES) * static_cast<size_t>(range + PADDING) * sizeof(unsigned int);
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
                                                  (const void*)histogram_kernel,
                                                  THREADS_PER_BLOCK,
                                                  shmemBytes);

    // Ensure at least one block; optionally oversubscribe to keep SMs busy with large inputs.
    int gridBlocks = blocksPerSM * smCount;
    if (gridBlocks <= 0) {
        gridBlocks = smCount > 0 ? smCount * 4 : 256; // Reasonable fallback
    }

    // Launch the kernel. We reinterpret the input as unsigned bytes to avoid signed-char issues.
    histogram_kernel<<<gridBlocks, THREADS_PER_BLOCK, shmemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        inputSize,
        from,
        to,
        histogram
    );
}