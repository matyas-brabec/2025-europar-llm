#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram-of-characters CUDA kernel restricted to a [from, to] range.

  Key optimization points:
  - Shared memory privatization: Each thread block builds a local histogram in shared memory.
  - 32 replicated copies (one per warp lane) to avoid shared-memory bank conflicts.
    Layout is [bin][lane] with stride 32 so address = bin*32 + lane.
    With 4-byte banks and 32 banks, lane 'l' always maps to bank 'l':
      bank = (bin*32 + lane) % 32 = lane
    This guarantees conflict-free access within a warp for both updates and reduction.
  - Warp-level reduction to combine the 32 copies per bin with no bank conflicts.
  - Grid-stride loop with itemsPerThread elements per thread for high memory throughput.

  Assumptions:
  - input is a device pointer to a char buffer of length inputSize.
  - histogram is a device pointer to an array of unsigned int of length (to-from+1).
  - 0 <= from < to <= 255.
  - Caller handles stream/synchronization as appropriate.

  Target hardware:
  - Modern NVIDIA data center GPUs (e.g., A100, H100).
*/

// Number of input items processed per thread (tuned for modern GPUs and large inputs).
// 16 is a good default to balance memory throughput and occupancy.
static constexpr int itemsPerThread = 16;

// CUDA kernel
__global__ void histogramRangeKernel(const char* __restrict__ input,
                                     unsigned int* __restrict__ histogram,
                                     unsigned int inputSize,
                                     int from,
                                     int to)
{
    const int numBins = to - from + 1;
    // Dynamically allocated shared memory: numBins * 32 counters (one per warp lane)
    extern __shared__ unsigned int shmem[];
    unsigned int* shist = shmem; // Layout: [bin][lane], index = bin*32 + lane

    const int lane            = threadIdx.x & 31;   // warp lane id (0..31)
    const int warpId          = threadIdx.x >> 5;   // warp id within the block
    const int warpsPerBlock   = blockDim.x >> 5;

    // Zero the shared histogram (all 32 copies)
    for (int i = threadIdx.x; i < numBins * 32; i += blockDim.x) {
        shist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop with itemsPerThread elements per thread
    const unsigned int tpb        = blockDim.x;
    const unsigned int baseThread = blockIdx.x * tpb + threadIdx.x;
    const unsigned int stride     = gridDim.x * tpb * itemsPerThread;

    const unsigned int ufrom = static_cast<unsigned int>(from);
    const unsigned int urange = static_cast<unsigned int>(to - from);

    for (unsigned int idxBase = baseThread; idxBase < inputSize; idxBase += stride) {
        #pragma unroll
        for (int it = 0; it < itemsPerThread; ++it) {
            unsigned int idx = idxBase + it * tpb;
            if (idx < inputSize) {
                // Convert to unsigned char to ensure 0..255 range regardless of signed char default
                unsigned int c = static_cast<unsigned char>(input[idx]);
                unsigned int bin = c - ufrom; // will underflow to large if c < from
                if (bin <= urange) {
                    // Update lane-private copy of the histogram bin in shared memory.
                    // Addressing: bin*32 + lane guarantees bank = lane (no intra-warp bank conflicts).
                    atomicAdd(&shist[bin * 32 + lane], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce 32 per-lane copies for each bin using warp-level reduction.
    // Each warp handles a subset of bins: b = warpId, warpId + warpsPerBlock, ...
    // For a given bin, every lane reads its lane-private counter (conflict-free), then we reduce.
    for (int b = warpId; b < numBins; b += warpsPerBlock) {
        unsigned int val = shist[b * 32 + lane];

        // Warp-wide sum reduction using shuffles (assumes 32-lane warps).
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }

        if (lane == 0) {
            // Accumulate block-local result into the global histogram with one atomic per bin.
            atomicAdd(&histogram[b], val);
        }
    }
}

/*
  Host-side launcher.
  - input: device pointer to input characters
  - histogram: device pointer to output array of length (to - from + 1)
  - inputSize: number of characters in input
  - from, to: inclusive character range [from, to] (0 <= from < to <= 255)
  Notes:
  - This function zeroes the output histogram asynchronously before launching the kernel.
  - Caller is responsible for synchronization if needed.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (inputSize == 0 || to < from) return;

    const int numBins = to - from + 1;

    // Clear output histogram (as kernel accumulates with atomics)
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a reasonable block size; 256 threads (8 warps) per block balances occupancy and contention.
    constexpr int threadsPerBlock = 256;

    // Dynamic shared memory size: numBins * 32 copies
    const size_t sharedMemBytes = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Determine a good grid size using occupancy (with given shared memory usage)
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxActiveBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM,
        histogramRangeKernel,
        threadsPerBlock,
        static_cast<int>(sharedMemBytes));

    // Required blocks to give each thread itemsPerThread work once (grid-stride ensures coverage anyway)
    const unsigned int denom = static_cast<unsigned int>(threadsPerBlock) * static_cast<unsigned int>(itemsPerThread);
    unsigned int requiredBlocks = (inputSize + denom - 1) / denom;

    // Cap by occupancy-derived limit to avoid excessive small blocks when input is very large
    unsigned int occupancyBlocks = (maxActiveBlocksPerSM > 0)
        ? static_cast<unsigned int>(maxActiveBlocksPerSM) * static_cast<unsigned int>(prop.multiProcessorCount)
        : requiredBlocks;

    // Ensure at least one block
    unsigned int gridBlocks = requiredBlocks > 0 ? requiredBlocks : 1u;
    if (occupancyBlocks > 0 && gridBlocks > occupancyBlocks) {
        gridBlocks = occupancyBlocks;
    }
    if (gridBlocks == 0) gridBlocks = 1;

    // Launch kernel
    histogramRangeKernel<<<gridBlocks, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to);
}