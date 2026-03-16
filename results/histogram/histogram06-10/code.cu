#include <cuda_runtime.h>
#include <cstddef>

namespace
{
// Range-restricted byte histogram optimized for recent NVIDIA data-center GPUs.
//
// High-level strategy:
//  1) Each thread block builds a block-private histogram in shared memory.
//  2) The shared histogram is replicated 32 times: one logical copy per warp lane.
//  3) The copies are laid out with a stride such that stride % 32 == 1.
//     Therefore, for any fixed bin b, the 32 addresses
//         copy * stride + b
//     map to banks
//         (copy + b) % 32,
//     i.e. one distinct bank per copy. This removes the classic intra-warp bank
//     conflict when many lanes hit the same histogram bin.
//  4) Threads index the shared copies by lane id, not by warp id. Threads with
//     the same lane in different warps therefore share a copy; shared-memory
//     atomics are still required, but hot-bin updates no longer collide inside
//     a warp on the shared-memory banks.
//  5) After the block has processed its grid-stride slice of the input, one
//     thread per output bin reduces the 32 shared copies and atomically adds
//     the block result to the global histogram.
//
// Tuning choices for A100/H100-class GPUs:
//  * blockSize = 256 gives 8 warps/block. This is a good balance between
//    occupancy and contention on a given lane-indexed shared copy.
//  * itemsPerThread = 16 is a good default for large inputs: enough work to
//    amortize loop/control overhead without driving register pressure or code
//    size too high.
constexpr unsigned int histogramCopies = 32u;
constexpr unsigned int histogramCopiesMask = histogramCopies - 1u;
constexpr int          blockSize        = 256;
constexpr int          itemsPerThread   = 16;

static_assert(sizeof(unsigned int) == 4,
              "This layout assumes 32-bit counters (one shared-memory bank slot per counter).");
static_assert((histogramCopies & histogramCopiesMask) == 0u,
              "histogramCopies must be a power of two.");
static_assert(blockSize % histogramCopies == 0,
              "blockSize must be a whole number of warps.");
static_assert(blockSize >= 256,
              "blockSize must cover the full maximum histogram range (256 bins).");
static_assert(itemsPerThread > 0,
              "itemsPerThread must be positive.");

// Smallest stride >= range such that stride % 32 == 1.
// This is the bank-rotation trick that places the same bin across the 32 copies
// into 32 different banks.
inline unsigned int compute_shared_stride(const unsigned int range)
{
    return range + ((1u - (range & histogramCopiesMask)) & histogramCopiesMask);
}

__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 const unsigned int inputSize,
                                 const unsigned int from,
                                 const unsigned int range,
                                 const unsigned int stride)
{
    extern __shared__ unsigned int sharedHist[];

    const unsigned int lane = threadIdx.x & histogramCopiesMask;

    // Treat the input as unsigned bytes so values 128..255 are handled correctly
    // regardless of whether plain 'char' is signed on the compilation target.
    const unsigned char* __restrict__ bytes =
        reinterpret_cast<const unsigned char*>(input);

    // Lane-private within a warp, but shared by the same lane across all warps
    // in the block. Hence the updates still use shared-memory atomics.
    unsigned int* const lanePrivateHist = sharedHist + lane * stride;

    const unsigned int totalSharedCounters = histogramCopies * stride;
    const size_t       n                   = static_cast<size_t>(inputSize);
    const size_t       blockTile           = static_cast<size_t>(blockDim.x) * itemsPerThread;
    const size_t       gridTile            = static_cast<size_t>(gridDim.x) * blockTile;

    // Zero the block-private histograms.
    for (unsigned int i = threadIdx.x; i < totalSharedCounters; i += blockDim.x)
    {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over block tiles.
    //
    // For each unrolled "item" iteration, threads in a warp access consecutive
    // bytes, so global loads are naturally coalesced.
    for (size_t threadBase = static_cast<size_t>(blockIdx.x) * blockTile + threadIdx.x;
         threadBase < n;
         threadBase += gridTile)
    {
#pragma unroll
        for (int item = 0; item < itemsPerThread; ++item)
        {
            const size_t pos = threadBase + static_cast<size_t>(item) * blockDim.x;
            if (pos < n)
            {
                // Unsigned subtraction folds the inclusive range test
                //     from <= byte <= to
                // into the single comparison
                //     (byte - from) < range.
                const unsigned int bin = static_cast<unsigned int>(bytes[pos]) - from;
                if (bin < range)
                {
                    atomicAdd(&lanePrivateHist[bin], 1u);
                }
            }
        }
    }
    __syncthreads();

    // Final block-local reduction:
    // blockSize is 256, and the requested range is at most 256 bins, so one
    // thread can reduce exactly one output bin.
    const unsigned int bin = threadIdx.x;
    if (bin < range)
    {
        unsigned int sum = 0u;

#pragma unroll
        for (unsigned int copy = 0; copy < histogramCopies; ++copy)
        {
            sum += sharedHist[copy * stride + bin];
        }

        // histogram[bin] corresponds to input byte value (from + bin).
        if (sum != 0u)
        {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

} // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // This interface has no stream parameter, so the default stream is used.
    // The function intentionally only enqueues work; caller-managed
    // synchronization/error observation happens outside this function.
    const cudaStream_t stream = 0;

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range  = static_cast<unsigned int>(to - from + 1);
    const unsigned int stride = compute_shared_stride(range);

    // Shared-memory footprint scales with the requested range, not with a fixed
    // 256-bin histogram.
    const size_t sharedBytes =
        static_cast<size_t>(histogramCopies) * static_cast<size_t>(stride) * sizeof(unsigned int);

    // The output histogram contains only bins [from, to], so zero exactly that range.
    (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), stream);

    if (inputSize == 0u)
    {
        return;
    }

    // Prefer shared memory over L1: this kernel is dominated by shared-memory
    // atomics, while input reads are a simple streaming pass.
    (void)cudaFuncSetAttribute(histogram_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    int device = 0;
    (void)cudaGetDevice(&device);

    int smCount = 1;
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1)
    {
        smCount = 1;
    }

    int activeBlocksPerSM = 1;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
                                                        histogram_kernel,
                                                        blockSize,
                                                        sharedBytes);
    if (activeBlocksPerSM < 1)
    {
        activeBlocksPerSM = 1;
    }

    const size_t blockTile = static_cast<size_t>(blockSize) * itemsPerThread;
    const unsigned int blocksForInput =
        static_cast<unsigned int>((static_cast<size_t>(inputSize) + blockTile - 1u) / blockTile);

    // Launch only one occupancy-saturating wave of blocks. Because each block
    // performs one final global reduction of its private histogram, using more
    // blocks than can reside concurrently only increases global atomic traffic;
    // the grid-stride loop keeps the resident blocks busy on large inputs.
    const unsigned int residentBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSM);

    const unsigned int gridBlocks =
        (blocksForInput < residentBlocks) ? blocksForInput : residentBlocks;

    histogram_kernel<<<gridBlocks, blockSize, sharedBytes, stream>>>(
        input,
        histogram,
        inputSize,
        from_u,
        range,
        stride);
}