#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel for a char range using shared-memory privatization
  with 32 bank-avoiding copies.

  Key design points:
  - Each thread processes 'itemsPerThread' input characters in a grid-stride loop.
    The addressing scheme ensures coalesced global memory reads:
      For a given iteration and inner item index 'it', all threads in a warp
      access consecutive input positions (idx = base + it * totalThreads).
  - Shared histogram privatization: 32 copies are stored in shared memory, one
    per shared-memory bank. Within a warp, each lane writes to its own "copy"
    (copy index = lane = threadIdx.x % 32), mapping each update to a distinct bank.
  - Histogram layout in shared memory uses a stride of 32 so that bin i for copy c
    resides at s_hist[i * 32 + c]. This avoids bank conflicts inside a warp.
  - After processing, each block reduces the 32 copies per bin to a single sum and
    atomically adds it to the global histogram.
  - The host launcher computes the required dynamic shared memory size
    (nBins * 32 * sizeof(unsigned int)), zeros the output histogram, and
    launches with a block/grid configuration suitable for large inputs on
    modern data center GPUs (e.g., A100/H100).

  Notes:
  - Assumes 0 <= from <= to <= 255 and 'input'/'histogram' are device pointers
    allocated by cudaMalloc. The 'histogram' buffer must be at least
    (to - from + 1) * sizeof(unsigned int) bytes.
  - The caller is responsible for stream synchronization if needed.
*/

// Controls how many input chars each thread processes per grid-stride "tile".
// 8 is a good default for modern NVIDIA data center GPUs with large inputs.
static constexpr int itemsPerThread = 8;

__global__ void histogramKernelRange(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    extern __shared__ unsigned int s_hist[]; // 32 copies, strided by 32: bin i, copy c => s_hist[i*32 + c]

    const int nBins = to - from + 1;
    const int lane  = threadIdx.x & 31; // lane within warp, also our shared-memory copy index [0..31]

    // Zero-initialize all shared-memory copies of the histogram.
    for (int i = threadIdx.x; i < nBins * 32; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop where each thread processes 'itemsPerThread' items per iteration.
    // Use an addressing scheme that maintains coalescence for each 'it':
    // base advances in tiles of totalThreads * itemsPerThread.
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t base = gid; base < inputSize; base += static_cast<size_t>(totalThreads) * itemsPerThread) {
        #pragma unroll
        for (int it = 0; it < itemsPerThread; ++it) {
            size_t idx = base + static_cast<size_t>(it) * totalThreads;
            if (idx < inputSize) {
                // Load as unsigned to avoid sign-extension issues for chars >= 128
                unsigned int uc = static_cast<unsigned char>(input[idx]);
                if (uc >= static_cast<unsigned int>(from) && uc <= static_cast<unsigned int>(to)) {
                    int bin = static_cast<int>(uc) - from;
                    // Atomic add to shared memory. With 32 copies strided by 32, threads in a warp
                    // update different banks (no bank conflicts). Cross-warp conflicts are handled by the atomic.
                    atomicAdd(&s_hist[bin * 32 + lane], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 privatized copies per bin into a single value and accumulate to global histogram.
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
        unsigned int sum = 0u;
        const int baseOff = bin * 32;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += s_hist[baseOff + c];
        }
        // Accumulate to global memory (multiple blocks use atomics).
        atomicAdd(&histogram[bin], sum);
    }
}

/*
  Host function to launch the histogram kernel.

  Parameters:
  - input: device pointer to the input char buffer
  - histogram: device pointer to the output histogram array of length (to - from + 1)
  - inputSize: number of chars in 'input'
  - from, to: inclusive character range [from, to] (0 <= from <= to <= 255)

  Behavior:
  - Zeroes the output histogram on the device.
  - Chooses a block size and grid size suitable for modern GPUs and large inputs.
  - Allocates enough dynamic shared memory for 32 copies of the sub-histogram.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (!input || !histogram) return;
    if (from > to) return;

    const int nBins = to - from + 1;
    if (nBins <= 0) return;

    // Zero the output histogram. The kernel performs atomic additions across blocks.
    cudaMemset(histogram, 0, static_cast<size_t>(nBins) * sizeof(unsigned int));

    // Choose a block size and grid size.
    // 256 threads per block is a good default on A100/H100 for this workload.
    const int blockSize = 256;

    // Determine a grid size that both covers the input and provides enough parallelism.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int smCount = prop.multiProcessorCount;

    // Target ~32 blocks per SM to provide ample parallelism without overcommitting resources.
    // The grid-stride loop ensures full coverage even if this grid is smaller than the input.
    const int targetBlocks = smCount * 32;

    // If the input is small, reduce the grid so that total threads * itemsPerThread approximately cover it.
    // Ensure at least 1 block.
    const size_t blocksToCover = (static_cast<size_t>(inputSize) + (static_cast<size_t>(blockSize) * itemsPerThread) - 1)
                               / (static_cast<size_t>(blockSize) * itemsPerThread);
    int gridSize = static_cast<int>(blocksToCover > 0 ? blocksToCover : 1);
    if (gridSize > targetBlocks) gridSize = targetBlocks;

    // Dynamic shared memory size: 32 copies of the histogram, strided by 32.
    const size_t sharedBytes = static_cast<size_t>(nBins) * 32u * sizeof(unsigned int);

    // Launch kernel (default stream). Caller handles synchronization if needed.
    histogramKernelRange<<<gridSize, blockSize, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}