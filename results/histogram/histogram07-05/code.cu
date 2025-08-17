#include <cuda_runtime.h>
#include <stdint.h>

// Tuneable compile-time constant controlling the number of input chars processed per thread.
// For large inputs on modern data-center GPUs (A100/H100), 16 offers a good balance between
// memory throughput and occupancy.
static constexpr int itemsPerThread = 16;

// CUDA kernel that computes a histogram over a specified inclusive character range [from, to].
// - input: device pointer to input chars (plain text).
// - globalHist: device pointer to output histogram array of size (to - from + 1) unsigned ints.
// - inputSize: number of chars in the input buffer.
// - from, to: inclusive range of character ordinal values to count (0 <= from < to <= 255).
//
// Optimization details:
// - Shared-memory privatization with 32 copies of the histogram to avoid shared mem bank conflicts.
//   Each thread updates copy number (threadIdx.x % 32), and histogram bins are laid out with a
//   stride of 32 so that bin i of copy c resides at sHist[i * 32 + c]. This maps lanes 0..31 to
//   distinct banks for each increment, eliminating intra-warp bank conflicts.
// - Each thread processes `itemsPerThread` items per block "tile", with a grid-stride over tiles.
//   Within a tile, index pattern idx = tileStart + k*blockDim.x + threadIdx.x preserves coalescing.
// - Reduction: the 32 shared copies are summed per bin and atomically added to the global histogram.
//
// Notes:
// - Shared-memory atomicAdd is used for correctness across warps, as multiple warps in the same block
//   can increment the same per-copy counters concurrently.
// - Global histogram is assumed to be zeroed prior to kernel launch (handled in run_histogram).
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ globalHist,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sHist[]; // Size: 32 * numBins
    const int numBins = to - from + 1;
    const int lane = threadIdx.x & 31; // 0..31

    // Zero-initialize all shared histogram copies.
    for (int i = threadIdx.x; i < numBins * 32; i += blockDim.x) {
        sHist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride over "tiles" of size blockDim.x * itemsPerThread per block.
    const unsigned int tileSizePerBlock = blockDim.x * itemsPerThread;
    const unsigned int gridTileStride = gridDim.x * tileSizePerBlock;

    for (unsigned int tileStart = blockIdx.x * tileSizePerBlock;
         tileStart < inputSize;
         tileStart += gridTileStride)
    {
        // Process itemsPerThread elements per thread within the tile.
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            unsigned int idx = tileStart + static_cast<unsigned int>(k) * blockDim.x + threadIdx.x;
            if (idx < inputSize) {
                unsigned int ch = static_cast<unsigned char>(input[idx]);
                if (ch >= static_cast<unsigned int>(from) && ch <= static_cast<unsigned int>(to)) {
                    unsigned int bin = ch - static_cast<unsigned int>(from);
                    // Update this thread's shared-memory histogram copy. Use shared-memory atomic
                    // because different warps can update the same copy concurrently.
                    atomicAdd(&sHist[bin * 32u + static_cast<unsigned int>(lane)], 1u);
                }
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies into one scalar per bin and atomically add to global histogram.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += sHist[bin * 32 + c];
        }
        if (sum) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

// Host-side launcher that prepares and invokes the histogram kernel.
// - input: device pointer to input chars (cudaMalloc'd).
// - histogram: device pointer to output histogram (cudaMalloc'd), size = (to - from + 1) uints.
// - inputSize: number of chars in input.
// - from, to: inclusive range [from, to] of chars to count.
// Assumes synchronization is handled by the caller (e.g., cudaDeviceSynchronize after this call).
extern "C" void run_histogram(const char *input, unsigned int *histogram,
                              unsigned int inputSize, int from, int to)
{
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Ensure output is zeroed before accumulation from blocks.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose a reasonable launch configuration for modern data-center GPUs.
    // - blockDim: 256 threads (good balance of occupancy and resource usage).
    // - gridDim: at least a few multiples of SM count for saturation, but also proportional to work.
    int blockSize = 256;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    int smCount = 0;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        smCount = prop.multiProcessorCount;
    }
    if (smCount <= 0) smCount = 80; // Conservative fallback

    const unsigned int tileSizePerBlock = static_cast<unsigned int>(blockSize) * itemsPerThread;
    // Number of tiles needed to cover input (each tile is processed by one block at a time).
    unsigned int tiles = (inputSize + tileSizePerBlock - 1u) / tileSizePerBlock;

    // Aim for enough blocks to keep the GPU busy; use max of tiles and (smCount * 4).
    unsigned int minGrid = static_cast<unsigned int>(smCount * 4);
    unsigned int gridSize = (tiles > 0u) ? tiles : 1u;
    if (gridSize < minGrid) gridSize = minGrid;

    // Shared memory size: 32 bank-aligned copies of the histogram (each bin is 4 bytes).
    size_t sharedMemBytes = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Launch kernel
    histogram_kernel<<<gridSize, blockSize, sharedMemBytes>>>(input, histogram, inputSize, from, to);
}