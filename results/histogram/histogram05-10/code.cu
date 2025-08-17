#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

// Tunable constant: how many input chars each thread processes per tile.
// For modern NVIDIA data center GPUs (A100/H100) and large inputs,
// a value of 16 provides a good balance between memory throughput and latency hiding.
static constexpr int itemsPerThread = 16;

// Compute padded index to avoid shared memory bank conflicts.
// We add one padding element for each 32 bins: idx + floor(idx/32).
// This prevents bins that are 32 apart from mapping to the same bank.
__device__ __forceinline__ int padded_index(int idx) {
    return idx + (idx >> 5);
}

// CUDA kernel for histogram computation over a restricted character range [from, to].
// - input: device pointer to input characters
// - histogram: device pointer to output histogram with (to - from + 1) bins
// - inputSize: number of characters in input
// - from, to: inclusive character range to count
//
// Optimization strategy:
// - Use shared memory to build per-block histograms.
// - Privatize histogram per warp to reduce atomic contention across warps.
// - Add padding to each histogram to avoid shared memory bank conflicts.
// - Use a grid-stride tiled loop; each block processes tiles of size blockDim.x * itemsPerThread,
//   and each thread processes itemsPerThread characters per tile with coalesced accesses.
__global__ void histogramKernelRange(const char* __restrict__ input,
                                     unsigned int* __restrict__ histogram,
                                     unsigned int inputSize,
                                     int from,
                                     int to)
{
    const int range = to - from + 1;
    if (range <= 0) return;

    // Number of warps per block (blockDim.x must be a multiple of warpSize)
    const int warpsPerBlock = blockDim.x >> 5; // divide by 32
    const int warpId = threadIdx.x >> 5;       // 0..warpsPerBlock-1
    const int laneId = threadIdx.x & 31;

    // Shared memory layout:
    // We allocate warpsPerBlock copies of the histogram, each of size paddedRange.
    // Padding reduces shared memory bank conflicts when threads in a warp update nearby bins.
    const int paddedRange = range + (range >> 5); // add one padding uint per 32 bins
    extern __shared__ unsigned int s_mem[];
    unsigned int* const s_hist = s_mem; // total size: warpsPerBlock * paddedRange

    // Zero shared histograms cooperatively.
    // Each thread initializes multiple elements with a stride of blockDim.x.
    for (int i = threadIdx.x; i < warpsPerBlock * paddedRange; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Pointer to this warp's private histogram in shared memory
    unsigned int* const s_warp_hist = s_hist + warpId * paddedRange;

    // Grid-stride tiled loop.
    // Each tile covers blockDim.x * itemsPerThread input elements per block.
    const size_t tileSize = (size_t)blockDim.x * (size_t)itemsPerThread;
    for (size_t tileStart = (size_t)blockIdx.x * tileSize;
         tileStart < (size_t)inputSize;
         tileStart += (size_t)gridDim.x * tileSize)
    {
        // Each thread processes itemsPerThread elements within the tile, with coalesced access:
        // pos = tileStart + threadIdx.x + k * blockDim.x
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            size_t pos = tileStart + (size_t)threadIdx.x + (size_t)k * (size_t)blockDim.x;
            if (pos >= (size_t)inputSize) break;

            unsigned char c = static_cast<unsigned char>(input[pos]);
            int bin = (int)c - from;
            // Fast unsigned bounds check for 0 <= bin < range
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
                // Update this warp's private shared histogram with bank-conflict-friendly index
                atomicAdd(&s_warp_hist[padded_index(bin)], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // Use all threads in the block to sum across warp copies and update global memory.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int sum = 0u;
        int pi = padded_index(i);
        // Accumulate bin i across all warps' private histograms
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * paddedRange + pi];
        }
        if (sum > 0u) {
            atomicAdd(&histogram[i], sum);
        }
    }
}

// Host function to launch the histogram kernel.
// - input: device pointer to input characters (cudaMalloc'ed)
// - histogram: device pointer to output histogram (cudaMalloc'ed) of size (to - from + 1)
// - inputSize: number of characters in input
// - from, to: inclusive range of ASCII codes to count (0 <= from < to <= 255)
//
// The function zeros the output histogram, configures the launch parameters,
// and invokes the CUDA kernel. Caller handles any host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate parameters (defensive; inputs are assumed valid per problem statement)
    if (!input || !histogram) return;
    if (from > to) return;
    if (from < 0) from = 0;
    if (to > 255) to = 255;

    const int range = to - from + 1;
    if (range <= 0) return;

    // Zero the output histogram on device to ensure fresh counts.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Choose launch configuration.
    // Threads per block: multiple of 32 for warp-based privatization.
    // 512 threads per block is a good default for Ampere/Hopper with modest shared memory use.
    constexpr int threadsPerBlock = 512;
    const int warpsPerBlock = threadsPerBlock / 32;

    // Compute dynamic shared memory size per block:
    // warpsPerBlock copies of the histogram, each of size (range + range/32) uints.
    const int paddedRange = range + (range >> 5);
    const size_t sharedMemBytes = (size_t)warpsPerBlock * (size_t)paddedRange * sizeof(unsigned int);

    // Determine an appropriate grid size.
    // Use a tile-based approach: each block processes blockDim.x * itemsPerThread elements per iteration.
    size_t tileSize = (size_t)threadsPerBlock * (size_t)itemsPerThread;
    size_t numTiles = (inputSize + tileSize - 1) / tileSize;

    // Query device properties to cap grid size appropriately.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // Target a grid with enough blocks to saturate the GPU while avoiding excessive launches.
    // 8x the number of SMs is a common heuristic for memory-bound kernels.
    int suggestedBlocks = props.multiProcessorCount * 8;
    int maxGridX = props.maxGridSize[0];
    int gridBlocks = (int)std::min((size_t)suggestedBlocks, numTiles);
    if (gridBlocks < 1) gridBlocks = 1;
    if (gridBlocks > maxGridX) gridBlocks = maxGridX;

    // Launch kernel
    histogramKernelRange<<<gridBlocks, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to
    );
}