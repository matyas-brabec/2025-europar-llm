#include <cuda_runtime.h>

namespace
{
// Tuned defaults for large inputs on modern data-center GPUs (A100/H100 class):
// - 256 threads/block keeps enough warps resident without doubling inter-warp
//   contention on the 32 lane-striped shared-memory replicas.
// - 16 items/thread amortizes address arithmetic and loop overhead while keeping
//   register pressure low under the shared-memory footprint of this kernel.
constexpr int itemsPerThread         = 16;
constexpr int histogramReplicaCount  = 32; // one replica per warp lane
constexpr int histogramReplicaStride = 32; // 32 uint32 words = 128 B
constexpr int blockSize              = 256;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(histogramReplicaCount == 32, "This kernel assumes 32-thread warps.");
static_assert(histogramReplicaStride == 32, "Replica stride must be 32 words.");
static_assert(blockSize % histogramReplicaCount == 0, "blockSize must be a multiple of 32.");

__device__ __forceinline__ void tally_byte(
    const unsigned int byteValue,
    const unsigned int from_u,
    const unsigned int numBins_u,
    unsigned int* const s_lane_hist)
{
    // Unsigned subtraction turns the range test into a single comparison:
    // if byteValue < from_u, the subtraction underflows and the comparison fails.
    const unsigned int bin = byteValue - from_u;
    if (bin < numBins_u)
    {
        atomicAdd(&s_lane_hist[bin * histogramReplicaStride], 1u);
    }
}

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // A100/H100 target path: use the hardware-supported warp reduction intrinsic.
    return __reduce_add_sync(0xffffffffu, value);
#else
    // Portable fallback for older compilation targets.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
#endif
}

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int numBins)
{
    // Shared-memory layout: [bin][replica].
    //
    // Each bin owns 32 replicas, and replicas are spaced by 32 uint32 words
    // (128 bytes). For any fixed bin, replica r maps to a unique shared-memory
    // bank modulo a constant base rotation. By choosing replica = warp lane,
    // all concurrent updates from a warp hit distinct banks, eliminating
    // intra-warp bank conflicts while still allowing a block-private histogram.
    extern __shared__ unsigned int s_hist[];

    const unsigned int lane      = threadIdx.x & (histogramReplicaCount - 1);
    unsigned int* const s_lane_hist = s_hist + lane;
    const unsigned int numBins_u = static_cast<unsigned int>(numBins);

    // Parallel initialization of the block-private replicated histogram.
    for (unsigned int i = threadIdx.x; i < numBins_u * histogramReplicaStride; i += BLOCK_SIZE)
    {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // Grid-stride over tiles. Within each tile, a thread processes items that are
    // striped by BLOCK_SIZE so that each inner-loop load is fully coalesced.
    const size_t blockTile   = static_cast<size_t>(BLOCK_SIZE) * itemsPerThread;
    const size_t threadBase  = static_cast<size_t>(blockIdx.x) * blockTile + threadIdx.x;
    const size_t gridStride  = static_cast<size_t>(gridDim.x) * blockTile;
    const size_t fullTileSpan = static_cast<size_t>(itemsPerThread - 1) * BLOCK_SIZE;
    const size_t inputSize64 = static_cast<size_t>(inputSize);

    size_t base = threadBase;

    // Fast path: full tiles need no per-item bounds checks.
    for (; base + fullTileSpan < inputSize64; base += gridStride)
    {
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item)
        {
            const size_t idx = base + static_cast<size_t>(item) * BLOCK_SIZE;
            tally_byte(static_cast<unsigned int>(inputBytes[idx]), from_u, numBins_u, s_lane_hist);
        }
    }

    // Tail path: only the final partial tile per thread needs bounds checks.
    for (; base < inputSize64; base += gridStride)
    {
        #pragma unroll
        for (int item = 0; item < itemsPerThread; ++item)
        {
            const size_t idx = base + static_cast<size_t>(item) * BLOCK_SIZE;
            if (idx < inputSize64)
            {
                tally_byte(static_cast<unsigned int>(inputBytes[idx]), from_u, numBins_u, s_lane_hist);
            }
        }
    }

    __syncthreads();

    // Reduction across the 32 lane-striped replicas.
    //
    // Each warp reduces one bin at a time:
    // - lane r reads replica r (conflict-free),
    // - the warp sums the 32 replicas,
    // - lane 0 atomically adds the block contribution to global memory.
    const unsigned int warpId = threadIdx.x >> 5;
    constexpr unsigned int warpsPerBlock = BLOCK_SIZE / histogramReplicaCount;

    for (int bin = static_cast<int>(warpId); bin < numBins; bin += static_cast<int>(warpsPerBlock))
    {
        unsigned int value = s_lane_hist[static_cast<unsigned int>(bin) * histogramReplicaStride];
        value = warp_reduce_sum(value);

        if (lane == 0 && value != 0u)
        {
            atomicAdd(&histogram[bin], value);
        }
    }
}
} // namespace

void run_histogram(
    const char *input,
    unsigned int *histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // The problem statement guarantees valid arguments, but a small guard keeps
    // the function safe for accidental misuse.
    if (from < 0 || to > 255 || from > to)
    {
        return;
    }

    const int numBins = to - from + 1;

    // The output must represent exactly this invocation, so clear it first.
    // Stream 0 is used because the function signature does not expose a stream;
    // caller-managed synchronization is intentionally left to the caller.
    cudaMemsetAsync(
        histogram,
        0,
        static_cast<size_t>(numBins) * sizeof(unsigned int),
        0);

    if (inputSize == 0)
    {
        return;
    }

    // This kernel is shared-memory heavy and streams the input once, so asking
    // the runtime for the maximum shared-memory carveout is the right default on
    // architectures with configurable L1/shared partitioning.
    cudaFuncSetAttribute(
        histogram_range_kernel<blockSize>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount <= 0)
    {
        smCount = 1;
    }

    // Max dynamic shared-memory requirement:
    // range_size * 32 replicas * 4 bytes.
    // Since range_size <= 256, the worst case is 32 KiB/block.
    const size_t sharedMemBytes =
        static_cast<size_t>(numBins) * histogramReplicaStride * sizeof(unsigned int);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<blockSize>,
        blockSize,
        sharedMemBytes);
    if (activeBlocksPerSM <= 0)
    {
        activeBlocksPerSM = 1;
    }

    // Because the kernel uses a grid-stride loop, one occupancy-saturating wave
    // of blocks is enough: resident blocks keep pulling tiles until the entire
    // input is consumed, while minimizing the number of block-to-global merges.
    const size_t blockTile = static_cast<size_t>(blockSize) * itemsPerThread;
    const unsigned int blocksNeeded =
        static_cast<unsigned int>((static_cast<size_t>(inputSize) + blockTile - 1) / blockTile);

    unsigned int blocks = static_cast<unsigned int>(smCount * activeBlocksPerSM);
    if (blocks == 0u)
    {
        blocks = 1u;
    }
    if (blocks > blocksNeeded)
    {
        blocks = blocksNeeded;
    }
    if (blocks == 0u)
    {
        blocks = 1u;
    }

    histogram_range_kernel<blockSize><<<blocks, blockSize, sharedMemBytes, 0>>>(
        input,
        histogram,
        inputSize,
        from,
        numBins);
}