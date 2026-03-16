#include <cuda_runtime.h>

// Range-restricted byte histogram for device-resident input.
//
// Design choices aimed at A100/H100-class GPUs:
// - Shared-memory privatization is done per warp, not just per block.
//   The histogram range is at most 256 bins, so one full 256-bin replica is only 1 KiB.
//   With 256 threads/block (= 8 warps), the entire shared working set is just 8 KiB/block,
//   which is small enough to keep occupancy high while sharply reducing shared-atomic contention.
// - itemsPerThread is the main tuning knob. 16 is a strong default on modern NVIDIA GPUs for
//   large inputs: it gives enough ILP to amortize index math and, in the fast path below,
//   maps naturally to one 16-byte vector load per thread.
// - blockSize is 256. That is a good default for histogram kernels on recent data-center GPUs,
//   and it also matches the maximum possible number of output bins so the final reduction can
//   assign one thread to each possible output bin.

namespace {

constexpr int itemsPerThread = 16;
constexpr int blockSize      = 256;
constexpr int warpSizeConst  = 32;
constexpr int maxBins        = 256;

constexpr unsigned int bytesPerBlockPerIteration =
    static_cast<unsigned int>(blockSize * itemsPerThread);

static_assert(blockSize % warpSizeConst == 0, "blockSize must be a multiple of 32.");
static_assert(blockSize >= maxBins, "blockSize must be at least 256.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

// Single-bin update helper.
// The range check uses unsigned arithmetic so one comparison covers both bounds:
//   bin = value - from;
//   bin < numBins   <=>   from <= value <= from + numBins - 1
__device__ __forceinline__
void count_byte(const unsigned int byteValue,
                const unsigned int from,
                const unsigned int numBins,
                unsigned int* const warpHist)
{
    const unsigned int bin = byteValue - from;
    if (bin < numBins) {
        atomicAdd(&warpHist[bin], 1u);
    }
}

// Process four packed bytes held in one 32-bit word.
// NVIDIA GPUs are little-endian, so extracting 8-bit lanes this way preserves byte order.
__device__ __forceinline__
void count_packed32(const unsigned int packed,
                    const unsigned int from,
                    const unsigned int numBins,
                    unsigned int* const warpHist)
{
    count_byte((packed      ) & 0xFFu, from, numBins, warpHist);
    count_byte((packed >>  8) & 0xFFu, from, numBins, warpHist);
    count_byte((packed >> 16) & 0xFFu, from, numBins, warpHist);
    count_byte((packed >> 24) & 0xFFu, from, numBins, warpHist);
}

template <int ITEMS_PER_THREAD, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int numBins)
{
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32.");
    static_assert(BLOCK_SIZE >= maxBins, "BLOCK_SIZE must be at least 256.");
    static_assert(ITEMS_PER_THREAD > 0, "ITEMS_PER_THREAD must be positive.");

    constexpr int kWarpsPerBlock       = BLOCK_SIZE / 32;
    constexpr int kReplicaBins         = maxBins;
    constexpr bool kVectorizableItems  = ((ITEMS_PER_THREAD % 16) == 0);
    constexpr int kVecLoadsPerThread   = ITEMS_PER_THREAD / 16;

    // One fully private 256-bin histogram replica per warp.
    __shared__ unsigned int sharedHistograms[kWarpsPerBlock][kReplicaBins];

    // Zero the entire shared working set. This is only 8 KiB with the chosen defaults,
    // so clearing all replica bins is cheap and keeps address math simple.
    unsigned int* const sharedFlat = &sharedHistograms[0][0];
    #pragma unroll
    for (int i = threadIdx.x; i < kWarpsPerBlock * kReplicaBins; i += BLOCK_SIZE) {
        sharedFlat[i] = 0u;
    }
    __syncthreads();

    const unsigned int warpId = threadIdx.x >> 5;
    unsigned int* const warpHist = sharedHistograms[warpId];

    // Reinterpret as unsigned bytes so ordinal values 128..255 are handled correctly
    // regardless of the signedness of plain char.
    const unsigned char* const data = reinterpret_cast<const unsigned char*>(input);
    const size_t n = static_cast<size_t>(inputSize);

    // Fast path: if itemsPerThread is a multiple of 16 and the pointer is 16-byte aligned,
    // each thread fetches its chunk with 16-byte vector loads.
    //
    // cudaMalloc provides strong alignment, so the intended use hits this path. The explicit
    // alignment check keeps the code correct even if a sub-pointer is ever passed in.
    const bool useVectorFastPath =
        kVectorizableItems && ((reinterpret_cast<size_t>(data) & 0xFu) == 0u);

    if (useVectorFastPath) {
        const size_t gridStrideBytes =
            static_cast<size_t>(gridDim.x) * BLOCK_SIZE * ITEMS_PER_THREAD;

        for (size_t base =
                 (static_cast<size_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x) * ITEMS_PER_THREAD;
             base < n;
             base += gridStrideBytes)
        {
            if (base + ITEMS_PER_THREAD <= n) {
                const uint4* const vectorPtr =
                    reinterpret_cast<const uint4*>(data + base);

                #pragma unroll
                for (int vec = 0; vec < kVecLoadsPerThread; ++vec) {
                    const uint4 packed = vectorPtr[vec];
                    count_packed32(packed.x, from, numBins, warpHist);
                    count_packed32(packed.y, from, numBins, warpHist);
                    count_packed32(packed.z, from, numBins, warpHist);
                    count_packed32(packed.w, from, numBins, warpHist);
                }
            } else {
                // Only the final partial chunk reaches this path.
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    const size_t idx = base + static_cast<size_t>(i);
                    if (idx < n) {
                        count_byte(data[idx], from, numBins, warpHist);
                    }
                }
            }
        }
    } else {
        // Generic fallback when itemsPerThread is changed to a non-16-multiple or the input
        // pointer is not 16-byte aligned. The interleaved access pattern keeps scalar loads
        // coalesced across the warp.
        const size_t gridStride =
            static_cast<size_t>(gridDim.x) * BLOCK_SIZE * ITEMS_PER_THREAD;

        for (size_t base =
                 static_cast<size_t>(blockIdx.x) * BLOCK_SIZE * ITEMS_PER_THREAD + threadIdx.x;
             base < n;
             base += gridStride)
        {
            #pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                const size_t idx = base + static_cast<size_t>(item) * BLOCK_SIZE;
                if (idx < n) {
                    count_byte(data[idx], from, numBins, warpHist);
                }
            }
        }
    }

    __syncthreads();

    // Because the maximum number of bins is 256 and BLOCK_SIZE is also 256, one thread can own
    // one final output bin during the block reduction.
    if (threadIdx.x < numBins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHistograms[w][threadIdx.x];
        }

        if (sum != 0u) {
            // Multiple blocks contribute to the same output bins, so the block flush uses
            // global atomics after the much larger amount of per-byte traffic has already
            // been absorbed in shared memory.
            atomicAdd(&histogram[threadIdx.x], sum);
        }
    }
}

} // anonymous namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The prompt guarantees: 0 <= from < to <= 255.
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The requested output is a fresh histogram for this call, not an accumulation.
    // This memset is asynchronous with respect to the host; the caller owns synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    // Launch one occupancy-saturating wave of blocks.
    // The kernel itself uses a grid-stride loop, so one full resident wave is enough to keep
    // all SMs busy on large inputs without oversharding the work into unnecessary block flushes.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel<itemsPerThread, blockSize>,
        blockSize,
        0);

    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const unsigned int blocksNeeded =
        (inputSize + bytesPerBlockPerIteration - 1u) / bytesPerBlockPerIteration;

    unsigned int gridSize =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSM);

    if (gridSize > blocksNeeded) {
        gridSize = blocksNeeded;
    }
    if (gridSize == 0u) {
        gridSize = 1u;
    }

    histogram_range_kernel<itemsPerThread, blockSize>
        <<<gridSize, blockSize>>>(
            input,
            histogram,
            inputSize,
            static_cast<unsigned int>(from),
            numBins);

    // Intentionally no synchronization here; the caller requested enqueue-only behavior.
}