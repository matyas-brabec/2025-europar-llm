#include <cuda_runtime.h>
#include <cstdint>

// Compute a histogram over a runtime-selected inclusive byte range [from, to].
// Output bin i corresponds to byte value (from + i).
//
// Optimization strategy tuned for modern datacenter GPUs (A100/H100 class):
// - 256 threads/block -> 8 warps/block.
// - One warp-private 256-bin histogram in shared memory per warp.
//   This costs only 8 KiB/block (8 warps * 256 bins * 4 bytes) and removes
//   inter-warp contention inside the block.
// - On Volta+ the update path uses __match_any_sync() to collapse duplicate bins
//   within the active subset of a warp to one shared-memory increment.
// - itemsPerThread = 16 is a good default for large inputs on recent GPUs:
//   it provides useful ILP, amortizes loop/index overhead, and maps naturally
//   to aligned 16-byte vector loads from cudaMalloc memory.
constexpr int itemsPerThread = 16;
constexpr int blockThreads   = 256;
constexpr int vectorLoadBytes = 16;

constexpr unsigned int warpSizeConst  = 32u;
constexpr unsigned int warpsPerBlock  = static_cast<unsigned int>(blockThreads) / warpSizeConst;
constexpr unsigned int fullCharDomain = 256u;
constexpr unsigned int invalidBin     = 0xFFFFFFFFu;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(blockThreads % static_cast<int>(warpSizeConst) == 0,
              "blockThreads must be a multiple of warp size.");
static_assert(blockThreads >= static_cast<int>(fullCharDomain),
              "blockThreads must be at least 256 so one thread can reduce one output bin.");
static_assert(vectorLoadBytes == static_cast<int>(sizeof(uint4)),
              "uint4 is expected to be a 16-byte vector load.");

// Add one logical histogram hit into a warp-private shared-memory histogram.
// Fast path (sm_70+): collapse all lanes that target the same bin into one leader
// update. Because each warp owns its own histogram segment, the shared-memory
// increment can be a plain add instead of an atomic.
// Fallback path: shared atomic within the warp-private histogram.
static __device__ __forceinline__
void warp_histogram_add_key(unsigned int key,
                            unsigned int range,
                            unsigned int* warpHist,
                            unsigned int lane,
                            unsigned int activeMask)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const unsigned int bin   = (key < range) ? key : invalidBin;
    const unsigned int peers = __match_any_sync(activeMask, bin);
    if (bin != invalidBin && lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        warpHist[bin] += static_cast<unsigned int>(__popc(peers));
    }
#else
    (void)lane;
    (void)activeMask;
    if (key < range) {
        atomicAdd(warpHist + key, 1u);
    }
#endif
}

// Process four packed bytes loaded from global memory.
static __device__ __forceinline__
void process_packed_u32(uint32_t word,
                        unsigned int from_u,
                        unsigned int range,
                        unsigned int* warpHist,
                        unsigned int lane,
                        unsigned int activeMask)
{
    warp_histogram_add_key(((word      ) & 0xFFu) - from_u, range, warpHist, lane, activeMask);
    warp_histogram_add_key(((word >>  8) & 0xFFu) - from_u, range, warpHist, lane, activeMask);
    warp_histogram_add_key(((word >> 16) & 0xFFu) - from_u, range, warpHist, lane, activeMask);
    warp_histogram_add_key(((word >> 24) & 0xFFu) - from_u, range, warpHist, lane, activeMask);
}

template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(blockThreads)
void histogram_kernel(const char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      unsigned int inputSize,
                      unsigned int from_u,
                      unsigned int range)
{
    // Full 256-bin warp-private histograms keep addressing fixed and cheap.
    // Only bins [0, range) are initialized and reduced because only those map
    // to the user-requested output subrange.
    __shared__ unsigned int s_hist[warpsPerBlock * fullCharDomain];

    const unsigned int tid    = static_cast<unsigned int>(threadIdx.x);
    const unsigned int warpId = tid >> 5;
    const unsigned int lane   = tid & (warpSizeConst - 1u);

    unsigned int* const warpHist = s_hist + warpId * fullCharDomain;
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // Each warp initializes only its own histogram segment.
    for (unsigned int bin = lane; bin < range; bin += warpSizeConst) {
        warpHist[bin] = 0u;
    }
    __syncthreads();

    const uint64_t size64      = static_cast<uint64_t>(inputSize);
    const uint64_t fullChunk   = static_cast<uint64_t>(ITEMS_PER_THREAD);
    const uint64_t chunkStride = static_cast<uint64_t>(gridDim.x) *
                                 static_cast<uint64_t>(blockThreads) *
                                 static_cast<uint64_t>(ITEMS_PER_THREAD);

    // Thread-contiguous chunking is chosen so the main path can use aligned
    // 16-byte vector loads when ITEMS_PER_THREAD is a multiple of 16. The prompt
    // states that input comes from cudaMalloc, so the base pointer is suitably
    // aligned and idx stays aligned because it advances in multiples of 16 for
    // the default itemsPerThread = 16.
    uint64_t idx = (static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockThreads) +
                    static_cast<uint64_t>(tid)) *
                   static_cast<uint64_t>(ITEMS_PER_THREAD);

    // Main path: full chunks only.
    if ((ITEMS_PER_THREAD % vectorLoadBytes) == 0) {
        if (size64 >= fullChunk) {
            for (; idx + fullChunk <= size64; idx += chunkStride) {
                const unsigned int activeMask = __activemask();
                const size_t base = static_cast<size_t>(idx);
                const uint4* const vec = reinterpret_cast<const uint4*>(inputBytes + base);

#pragma unroll
                for (int v = 0; v < ITEMS_PER_THREAD / vectorLoadBytes; ++v) {
                    const uint4 packed = vec[v];
                    process_packed_u32(packed.x, from_u, range, warpHist, lane, activeMask);
                    process_packed_u32(packed.y, from_u, range, warpHist, lane, activeMask);
                    process_packed_u32(packed.z, from_u, range, warpHist, lane, activeMask);
                    process_packed_u32(packed.w, from_u, range, warpHist, lane, activeMask);
                }
            }
        }
    } else {
        if (size64 >= fullChunk) {
            for (; idx + fullChunk <= size64; idx += chunkStride) {
                const unsigned int activeMask = __activemask();
                const size_t base = static_cast<size_t>(idx);

#pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
                    const unsigned int key =
                        static_cast<unsigned int>(inputBytes[base + static_cast<size_t>(j)]) - from_u;
                    warp_histogram_add_key(key, range, warpHist, lane, activeMask);
                }
            }
        }
    }

    // Tail path: at most one partial chunk per thread remains after the full-chunk loop.
    // Lanes that have run out of valid input still participate in the warp-level grouping
    // with an invalid bin so the active subset remains well-defined.
    for (; idx < size64; idx += chunkStride) {
        const unsigned int activeMask = __activemask();
        const size_t base = static_cast<size_t>(idx);
        const unsigned int remaining = static_cast<unsigned int>(size64 - idx);

#pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            unsigned int key = invalidBin;
            if (static_cast<unsigned int>(j) < remaining) {
                key = static_cast<unsigned int>(inputBytes[base + static_cast<size_t>(j)]) - from_u;
            }
            warp_histogram_add_key(key, range, warpHist, lane, activeMask);
        }
    }

    __syncthreads();

    // One thread per output bin reduces the warp-private histograms for this block.
    // If there is only one block in the grid, a direct store is enough because the
    // host launcher clears the output histogram beforehand.
    if (tid < range) {
        unsigned int sum = 0u;
#pragma unroll
        for (unsigned int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * fullCharDomain + tid];
        }

        if (sum != 0u) {
            if (gridDim.x == 1u) {
                histogram[tid] = sum;
            } else {
                atomicAdd(histogram + tid, sum);
            }
        }
    }
}

static inline unsigned int ceil_div_u32(unsigned int x, unsigned int y)
{
    return (x == 0u) ? 0u : (1u + (x - 1u) / y);
}

// input and histogram are device pointers allocated with cudaMalloc.
// The function clears the output histogram and then launches the kernel.
// Both operations are asynchronous; synchronization is intentionally left
// to the caller as requested.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range  = static_cast<unsigned int>(to - from + 1);

    // The kernel accumulates into the destination histogram, so clear it first.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        device = 0;
    }

    int smCount = 0;
    if (cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        smCount <= 0) {
        smCount = 1;
    }

    int activeBlocksPerSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocksPerSM,
            histogram_kernel<itemsPerThread>,
            blockThreads,
            0) != cudaSuccess || activeBlocksPerSM <= 0) {
        activeBlocksPerSM = 1;
    }

    // Grid-stride looping means an occupancy-sized grid is enough for large inputs.
    const unsigned int blockChunk     = static_cast<unsigned int>(blockThreads) *
                                        static_cast<unsigned int>(itemsPerThread);
    const unsigned int blocksNeeded   = ceil_div_u32(inputSize, blockChunk);
    const unsigned int residentBlocks = static_cast<unsigned int>(smCount) *
                                        static_cast<unsigned int>(activeBlocksPerSM);
    const unsigned int grid           = (blocksNeeded < residentBlocks) ? blocksNeeded
                                                                        : residentBlocks;

    if (grid == 0u) {
        return;
    }

    histogram_kernel<itemsPerThread><<<grid, blockThreads>>>(
        input,
        histogram,
        inputSize,
        from_u,
        range);
}