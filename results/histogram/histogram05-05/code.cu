#include <cuda_runtime.h>

// itemsPerThread controls how many input characters each thread processes in a strided loop.
// On modern NVIDIA data center GPUs (A100/H100), values in the 8–16 range work well.
// We choose 8 as a balanced default for bandwidth-bound workloads and to keep register pressure modest.
static constexpr int itemsPerThread = 8;

/*
  Kernel overview:

  - Computes a histogram over a contiguous character range [from, to], inclusive.
  - Input: 'input' is a device pointer to an array of chars of size 'n'.
  - Output: 'histogram' is a device pointer to an array of unsigned ints of size bins = to - from + 1.
    It must be zero-initialized before the kernel launches (handled in run_histogram).
  - Shared memory privatization:
      We create multiple copies of the histogram per thread block to avoid shared memory bank conflicts.
      Specifically, we create:
        warpsPerBlock copies (one per warp) AND each warp's histogram is replicated 32 times (one per lane).
      Layout:
        sHist[ warpsPerBlock ][ bins ][ 32 lanes ]
      Update scheme:
        Each thread (uniquely identified by its laneId within its warp) updates its own lane's copy
        at index [warpId][bin][laneId]. Within a warp, simultaneously updating the same 'bin' by all lanes
        maps to addresses with different bank indices (bin*32 + laneId), guaranteeing conflict-free updates.
      Reduction:
        1) Sum across lane replicas (32) for each warp, for each bin.
        2) Sum across warps for each bin.
        3) Atomically add the per-block bin totals into the global histogram (one atomic per bin per block).

  - Each thread processes 'itemsPerThread' consecutive input characters in a grid-stride loop.

  Notes:
    - We cast input chars to unsigned char to interpret character codes 0..255 consistently, regardless of 'char' signedness.
    - Range check is done branch-efficiently with unsigned comparison: 0 <= (unsigned)(val - from) < (unsigned)bins.
    - The kernel expects blockDim.x to be a multiple of 32 (enforced by launcher).
    - Dynamic shared memory size must be: warpsPerBlock * bins * 32 * sizeof(unsigned int).
*/
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int n,
                                       int from,
                                       int to)
{
    const int bins = to - from + 1;
    const int warpSizeConst = 32;

    // Identify lane and warp within the block
    const int laneId = threadIdx.x & (warpSizeConst - 1);
    const int warpId = threadIdx.x >> 5; // divide by 32
    const int warpsPerBlock = blockDim.x >> 5;

    // Shared memory layout: [warpsPerBlock][bins][32 lanes]
    extern __shared__ unsigned int sHist[];

    // Zero-initialize the shared memory histograms cooperatively
    const unsigned int totalSharedCounters = (unsigned int)(warpsPerBlock * bins * warpSizeConst);
    for (unsigned int i = threadIdx.x; i < totalSharedCounters; i += blockDim.x) {
        sHist[i] = 0;
    }
    __syncthreads();

    // Pointer to this thread's (lane's) private histogram within its warp's shared memory region
    // sHist index for (warpId, bin, laneId) = warpId*bins*32 + bin*32 + laneId
    unsigned int* warpLaneHistBase = sHist + (warpId * bins * warpSizeConst) + laneId;

    // Compute grid-stride loop over input with 'itemsPerThread' contiguous items per thread per iteration
    const unsigned long long N = n;
    const unsigned long long tpbItems = (unsigned long long)blockDim.x * (unsigned long long)gridDim.x * (unsigned long long)itemsPerThread;
    unsigned long long threadStart = ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x)
                                   * (unsigned long long)itemsPerThread;

    for (unsigned long long base = threadStart; base < N; base += tpbItems) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            unsigned long long idx = base + (unsigned long long)k;
            if (idx < N) {
                unsigned int v = static_cast<unsigned char>(input[idx]); // normalize char to 0..255
                int b = (int)v - from;
                // Use unsigned comparison to handle both bounds in a single check
                if ((unsigned int)b < (unsigned int)bins) {
                    // Increment this thread's lane-specific replica of bin 'b'.
                    // Address = warpLaneHistBase + b*32
                    unsigned int* addr = warpLaneHistBase + (b * warpSizeConst);
                    // No need for atomics: only this thread updates [warpId][b][laneId]
                    *addr += 1U;
                }
            }
        }
    }
    __syncthreads();

    // Reduce across the 32 lane-replicas within each warp for every bin.
    // Use all threads in the block to parallelize this step.
    const int warpBinCount = warpsPerBlock * bins;
    for (int t = threadIdx.x; t < warpBinCount; t += blockDim.x) {
        int w = t / bins;
        int b = t - w * bins; // t % bins
        unsigned int* basePtr = sHist + (w * bins * warpSizeConst) + (b * warpSizeConst); // points to [w][b][0]
        unsigned int sum = 0;
        #pragma unroll
        for (int r = 0; r < warpSizeConst; ++r) {
            sum += basePtr[r];
        }
        // Store the per-warp bin sum in the [lane 0] location for reuse
        basePtr[0] = sum;
    }
    __syncthreads();

    // Reduce across warps and update global histogram (one atomicAdd per bin per block)
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        unsigned int total = 0;
        // Accumulate the per-warp sums stored at lane 0
        for (int w = 0; w < warpsPerBlock; ++w) {
            total += sHist[(w * bins * warpSizeConst) + (b * warpSizeConst) + 0];
        }
        if (total) {
            atomicAdd(&histogram[b], total);
        }
    }
}

/*
  Host launcher:

  - Computes launch configuration that fits the dynamic shared memory requirements.
  - Zeroes the output histogram buffer (device memory) before launching the kernel.
  - Adapts the number of warps per block to the device's maximum dynamic shared memory per block.
  - Sets the kernel's dynamic shared memory attribute to allow using more than the default shared memory per block if available.

  Parameters:
    input      - device pointer to input chars (cudaMalloc'ed)
    histogram  - device pointer to output histogram of size (to - from + 1) unsigned ints (cudaMalloc'ed)
    inputSize  - number of chars in 'input'
    from, to   - inclusive character range [from, to], with 0 <= from < to <= 255

  Synchronization:
    This function does not synchronize the device. The caller is responsible for synchronization.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int bins = to - from + 1;

    // Edge cases: empty input or invalid bin count (shouldn't happen per spec but guard anyway)
    if (bins <= 0) {
        return;
    }

    // Zero the output histogram (important since kernel atomically adds per-block totals)
    cudaMemset(histogram, 0, (size_t)bins * sizeof(unsigned int));

    // Determine shared memory requirements and select a block size (multiple of 32).
    int device = 0;
    cudaGetDevice(&device);

    int maxOptinShmem = 0;
    cudaDeviceGetAttribute(&maxOptinShmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int maxDefaultShmem = 0;
    cudaDeviceGetAttribute(&maxDefaultShmem, cudaDevAttrMaxSharedMemoryPerBlock, device);

    size_t maxAllowedShmem = (size_t)((maxOptinShmem > 0) ? maxOptinShmem : maxDefaultShmem);

    // Per-warp shared memory footprint = bins * 32 lanes * 4 bytes
    const size_t perWarpShared = (size_t)bins * 32u * sizeof(unsigned int);

    // Start with a default of 4 warps per block (128 threads), then clamp to fit shared memory.
    int warpsPerBlock = 4;
    // Ensure at least 1 warp per block
    if (warpsPerBlock < 1) warpsPerBlock = 1;

    // Reduce warps if shared memory would exceed the allowed limit
    while ((size_t)warpsPerBlock * perWarpShared > maxAllowedShmem && warpsPerBlock > 1) {
        --warpsPerBlock;
    }

    // If bins is large and only 1 warp fits, this still works (blockDim=32).
    if ((size_t)warpsPerBlock * perWarpShared > maxAllowedShmem) {
        // As a last resort (should not happen for bins <= 256 on modern GPUs), cap to a single warp.
        warpsPerBlock = 1;
    }

    const int blockSize = warpsPerBlock * 32;
    const size_t dynamicShmemBytes = (size_t)warpsPerBlock * perWarpShared;

    // If dynamic shared memory needed exceeds the default limit, try to opt-in to a higher limit
    if (dynamicShmemBytes > (size_t)maxDefaultShmem && maxOptinShmem > 0) {
        // Request the needed dynamic shared memory size for this kernel
        cudaFuncSetAttribute(histogram_range_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)dynamicShmemBytes);
    }

    // Choose grid size based on input size and work per block
    unsigned long long itemsPerBlock = (unsigned long long)blockSize * (unsigned long long)itemsPerThread;
    unsigned int gridSize = (itemsPerBlock > 0)
                              ? (unsigned int)(((unsigned long long)inputSize + itemsPerBlock - 1ULL) / itemsPerBlock)
                              : 1U;
    if (gridSize == 0) gridSize = 1;

    // Launch kernel
    histogram_range_kernel<<<gridSize, blockSize, dynamicShmemBytes>>>(input, histogram, inputSize, from, to);
}