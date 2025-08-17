#include <cuda_runtime.h>
#include <stdint.h>

// Histogram kernel optimized for A100/H100 class GPUs.
// Key design points:
// - Block-private histogram in shared memory to reduce global atomics.
// - To avoid shared memory bank conflicts, we store 32 copies of each bin,
//   one per warp lane, laid out with a stride of 32 (bin*32 + lane).
//   Within a warp, each lane accesses its own copy, guaranteeing conflict-free
//   shared memory access for a given bin.
// - We use shared-memory atomics when different warps in the same block update
//   the same (bin, lane) entry concurrently.
// - After counting, we reduce the 32 copies per bin within each block using
//   warp-level reductions and perform a single atomicAdd per bin per block to
//   the global histogram.
// - Each thread processes 'itemsPerThread' input bytes. Threads within a block
//   access a contiguous chunk in a coalesced pattern: idx = blockStart + tid + j*blockDim.x.
//   This keeps global memory accesses coalesced for each iteration 'j'.
//
// The output histogram covers only the [from, to] range (inclusive) and has
// (to - from + 1) entries. Input is treated as unsigned bytes to properly
// handle 0..255 ordinals.
//
// Tuning knobs:
// - itemsPerThread: Set to 16 by default for modern data-center GPUs and large inputs.
// - blockDim.x: 256 by default. Adjust as needed for occupancy and workload characteristics.

static constexpr int itemsPerThread = 16;  // Tunable: good default for A100/H100

// CUDA kernel
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ globalHist,
                                       unsigned int n,
                                       int from, int to)
{
    extern __shared__ unsigned int s_hist[]; // Layout: [numBins][32], index = bin*32 + lane

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;              // Warp lane [0,31]
    const int warpId  = tid >> 5;              // Warp index within block
    const int numWarps = blockDim.x >> 5;

    // Zero the shared histogram (including all 32 copies per bin).
    for (int i = tid; i < numBins * 32; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of input of size blockDim.x * itemsPerThread.
    // Threads within the block process items in a coalesced pattern:
    // idx = blockStart + threadIdx.x + j * blockDim.x for j in [0, itemsPerThread).
    const unsigned int blockChunk = blockDim.x * itemsPerThread;
    const unsigned int blockStart = blockIdx.x * blockChunk;

    // Process up to itemsPerThread items per thread.
    #pragma unroll
    for (int j = 0; j < itemsPerThread; ++j) {
        unsigned int idx = blockStart + tid + j * blockDim.x;
        if (idx >= n) break;

        unsigned char c = input[idx];
        int v = static_cast<int>(c);
        if (v >= from && v <= to) {
            int bin = v - from;                        // 0 .. numBins-1
            // Update the lane-private copy for this bin. Different warps can collide on the same
            // (bin,lane) entry, so we must use an atomic in shared memory to avoid races.
            atomicAdd(&s_hist[bin * 32 + lane], 1);
        }
    }
    __syncthreads();

    // Reduce the 32 lane-copies per bin using warp-level reductions.
    // Assign bins to warps in a strided fashion.
    for (int bin = warpId; bin < numBins; bin += numWarps) {
        unsigned int v = s_hist[bin * 32 + lane];
        // Warp reduction: sum across lanes.
        // Each lane holds its own column value; reduce to lane 0.
        unsigned int mask = 0xFFFFFFFFu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);

        if (lane == 0 && v > 0) {
            atomicAdd(&globalHist[bin], v);
        }
    }
}

// Host launcher.
// - input: device pointer to input bytes (char array allocated with cudaMalloc).
// - histogram: device pointer to output histogram of size (to - from + 1) (allocated with cudaMalloc).
// - inputSize: number of input bytes.
// - from, to: inclusive character range [from, to] (0 <= from < to <= 255).
//
// Any synchronization is handled by the caller; this function issues async operations on the default stream.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int numBins = to - from + 1;
    if (numBins <= 0) return; // Defensive: nothing to do if range invalid.

    // Clear the output histogram.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there's no input, nothing else to do.
    if (inputSize == 0) return;

    // Kernel launch configuration.
    // Use 256 threads per block (8 warps). Each thread processes itemsPerThread items.
    // Each block's shared memory usage is numBins * 32 * sizeof(unsigned int).
    const int blockSize = 256;
    const unsigned int itemsPerBlock = blockSize * itemsPerThread;
    const unsigned int gridSize = (inputSize + itemsPerBlock - 1) / itemsPerBlock;

    if (gridSize == 0) return;

    const size_t smemSize = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);

    histogram_range_kernel<<<gridSize, blockSize, smemSize>>>(d_input, histogram, inputSize, from, to);
}