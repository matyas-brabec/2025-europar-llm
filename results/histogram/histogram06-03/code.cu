#include <cuda_runtime.h>

// Compute a histogram over a restricted character range using 32-way shared-memory privatization.
// The kernel uses per-lane sub-histograms in shared memory to minimize intra-warp bank conflicts.
// Each of the 32 per-lane copies starts at a different bank by using a stride that is 1 (mod 32).
// After per-block accumulation, the 32 copies are reduced and atomically accumulated to the global histogram.

namespace {

// Tuneable constant: number of characters processed per thread per outer iteration.
// Chosen for modern NVIDIA data-center GPUs (A100/H100) and large inputs.
// Larger values increase per-thread work and can reduce overhead; 16 is a solid default for 1-byte loads.
constexpr int itemsPerThread = 16;

// Compute the per-copy stride in shared memory, in 32-bit words, for 'numBins' bins.
// We pad to the next multiple of 32 and add +1 so that stride % 32 == 1.
// This ensures that the base of copy 'lane' starts at bank 'lane' (avoiding intra-warp bank conflicts
// when all lanes update the same bin index).
static __host__ __device__ __forceinline__ int compute_stride(int numBins) {
    // round up to multiple of 32, then add 1 to make stride ≡ 1 (mod 32)
    return ((numBins + 31) & ~31) + 1;
}

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // NOTE: Assumes 0 <= from < to <= 255 (as specified by the problem).
    const int numBins = to - from + 1;
    const int stride  = compute_stride(numBins);             // stride in 32-bit elements
    const int lane    = threadIdx.x & 31;                    // warp lane id [0,31]
    extern __shared__ unsigned int shmem[];                  // size: 32 * stride unsigned ints

    // Zero all shared-memory bins for this block across all 32 copies.
    for (int i = threadIdx.x; i < 32 * stride; i += blockDim.x) {
        shmem[i] = 0u;
    }
    __syncthreads();

    // Base pointer for this thread's per-lane copy (each warp lane updates its own copy).
    // With stride ≡ 1 mod 32, the base of copy 'lane' maps to bank 'lane', minimizing intra-warp conflicts.
    unsigned int* __restrict__ smem_lane_hist = shmem + lane * stride;

    // Process the input in block-strided chunks, with itemsPerThread contiguous elements per thread.
    // For each k in [0, itemsPerThread), threads of a block access consecutive bytes for coalesced loads.
    const size_t blockSpan = static_cast<size_t>(blockDim.x) * static_cast<size_t>(itemsPerThread);
    const size_t gridSpan  = static_cast<size_t>(gridDim.x)  * blockSpan;

    for (size_t blockStart = static_cast<size_t>(blockIdx.x) * blockSpan;
         blockStart < inputSize;
         blockStart += gridSpan)
    {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            size_t idx = blockStart + static_cast<size_t>(k) * blockDim.x + threadIdx.x;
            if (idx < inputSize) {
                // Convert to unsigned to treat chars as bytes [0,255] regardless of signedness of 'char'.
                unsigned char uc = static_cast<unsigned char>(input[idx]);

                // Fast in-range check: compute bin = uc - from, then test 0 <= bin < numBins via unsigned compare.
                int bin = static_cast<int>(uc) - from;
                if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                    // Shared-memory accumulation. Multiple warps in the same block share the 32 per-lane copies,
                    // so we must use atomicAdd to avoid inter-warp races. Intra-warp bank conflicts are minimized
                    // by placing each copy at a separate bank base as explained above.
                    atomicAdd(&smem_lane_hist[bin], 1u);
                }
            }
        }
    }
    __syncthreads();

    // Reduce the 32 per-lane copies into global memory.
    // Each thread accumulates a subset of bins to minimize global atomics: total atomics per block = numBins.
    for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
        unsigned int sum = 0;
        // Sum the bin across all 32 lane-local copies.
        #pragma unroll
        for (int copy = 0; copy < 32; ++copy) {
            sum += shmem[copy * stride + b];
        }
        // Atomically add the block's total for this bin to the global histogram.
        atomicAdd(&global_hist[b], sum);
    }
}

} // anonymous namespace

// Host entry point.
// input:      device pointer to input characters (cudaMalloc'd)
// histogram:  device pointer to output histogram buffer with length (to - from + 1) (cudaMalloc'd)
// inputSize:  number of characters in 'input'
// from, to:   inclusive range [from, to] to compute (0 <= from < to <= 255)
//
// This function zeroes the output histogram, configures and launches the kernel using dynamic shared memory,
// and relies on the caller for stream/synchronization management as specified.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (!input || !histogram || inputSize == 0 || to < from) {
        return;
    }

    const int numBins = to - from + 1;

    // Zero the output histogram (device memory).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Determine reasonable launch configuration for modern data-center GPUs.
    // We choose 256 threads per block, which balances occupancy and per-thread register usage well.
    const int threadsPerBlock = 256;

    // Compute the per-block work size and select a grid size large enough to saturate the GPU.
    // We ensure at least 8 blocks per SM (typical good default) while also covering the input.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const int smCount = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 80; // fallback guess
    const int minBlocksForSaturation = smCount * 8;

    const size_t blockWork = static_cast<size_t>(threadsPerBlock) * static_cast<size_t>(itemsPerThread);
    int blocksBySize = static_cast<int>((static_cast<size_t>(inputSize) + blockWork - 1) / blockWork);
    if (blocksBySize < 1) blocksBySize = 1;

    int gridDimX = blocksBySize < minBlocksForSaturation ? minBlocksForSaturation : blocksBySize;
    if (prop.maxGridSize[0] > 0 && gridDimX > prop.maxGridSize[0]) {
        gridDimX = prop.maxGridSize[0];
    }

    // Dynamic shared memory size: 32 copies, each with 'stride' 32-bit elements.
    const int stride = compute_stride(numBins);
    const size_t sharedMemBytes = static_cast<size_t>(32) * static_cast<size_t>(stride) * sizeof(unsigned int);

    // Launch the kernel. Caller handles synchronization if needed.
    histogram_range_kernel<<<gridDimX, threadsPerBlock, sharedMemBytes>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}