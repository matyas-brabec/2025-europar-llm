#include <cuda_runtime.h>
#include <cstddef>

// Range-restricted byte histogram for GPU-resident input.
//
// Performance strategy:
// - Each warp owns a private histogram in shared memory, so updates never
//   contend across warps within a block.
// - __match_any_sync collapses equal bins among currently active lanes; only the
//   leader of each equal-value group performs the shared-memory increment.
// - After the scan, the block merges its warp-private histograms and updates the
//   global histogram once per non-zero bin.
// - Host launch logic uses:
//     * 256-thread blocks for medium inputs to expose more CTAs,
//     * 512-thread blocks for larger inputs to reduce the cost of the per-block
//       final merge once there is enough work to keep all SMs busy.
//
// This requires Volta+ warp-match intrinsics; the target hardware here is a
// modern data-center GPU such as A100/H100.

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxBins = 256;
constexpr unsigned int kVectorBytes = 16u;    // One uint4 load = 16 input bytes.
constexpr unsigned int kInvalidBin = 0xFFFFFFFFu;

static_assert(sizeof(uint4) == kVectorBytes, "uint4 must be 16 bytes.");

__device__ __forceinline__
void accumulate_bin_warp_private(unsigned int* warpHistogram,
                                 unsigned int relativeBin,
                                 unsigned int binCount,
                                 unsigned int lane)
{
    // All currently active lanes participate. Using __activemask() keeps the
    // warp match correct in partially active final grid-stride iterations.
    const unsigned int active = __activemask();
    const unsigned int key = (relativeBin < binCount) ? relativeBin : kInvalidBin;
    const unsigned int peers = __match_any_sync(active, key);

    // Exactly one lane per distinct valid bin performs the increment by the size
    // of its peer group. Because each warp writes only to its own shared-memory
    // slice, a normal shared-memory increment is sufficient here.
    if (key != kInvalidBin &&
        lane == static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1)) {
        warpHistogram[key] += static_cast<unsigned int>(__popc(peers));
    }
}

__device__ __forceinline__
void process_packed_u32(unsigned int* warpHistogram,
                        unsigned int packed,
                        unsigned int from,
                        unsigned int binCount,
                        unsigned int lane)
{
    // Each 32-bit word contains four text bytes. Order does not matter for a
    // histogram, so simple bit extraction is sufficient.
    accumulate_bin_warp_private(warpHistogram, ((packed      ) & 0xFFu) - from, binCount, lane);
    accumulate_bin_warp_private(warpHistogram, ((packed >>  8) & 0xFFu) - from, binCount, lane);
    accumulate_bin_warp_private(warpHistogram, ((packed >> 16) & 0xFFu) - from, binCount, lane);
    accumulate_bin_warp_private(warpHistogram, ((packed >> 24) & 0xFFu) - from, binCount, lane);
}

template <int BlockSize>
__global__ __launch_bounds__(BlockSize)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int binCount)
{
    static_assert(BlockSize % kWarpSize == 0, "Block size must be a multiple of 32.");
    static_assert(BlockSize >= kMaxBins,
                  "Block size must be at least the maximum number of output bins.");

    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;
    constexpr unsigned int kBlockSizeU = static_cast<unsigned int>(BlockSize);

    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & (kWarpSize - 1u);
    const unsigned int warpId = tid >> 5;
    unsigned int* const warpHistogram = sharedHistogram + warpId * binCount;

    // Zero the block's warp-private histograms in shared memory.
    const unsigned int totalSharedBins =
        static_cast<unsigned int>(kWarpsPerBlock) * binCount;
    for (unsigned int i = tid; i < totalSharedBins; i += kBlockSizeU) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    const unsigned int globalThread = blockIdx.x * kBlockSizeU + tid;
    const unsigned int globalStride = gridDim.x * kBlockSizeU;

    // Align to 16 bytes for vector loads. cudaMalloc already provides strong
    // alignment, but handling an arbitrary subspan keeps the kernel correct even
    // if the caller passes an offset pointer.
    const size_t inputAddress = reinterpret_cast<size_t>(input);
    const unsigned int misalignment =
        static_cast<unsigned int>(inputAddress & static_cast<size_t>(kVectorBytes - 1u));
    unsigned int prefixBytes = (kVectorBytes - misalignment) & (kVectorBytes - 1u);
    if (prefixBytes > inputSize) {
        prefixBytes = inputSize;
    }

    // Scalar prefix before the aligned vector region. prefixBytes < 16, so only
    // the first few global threads ever participate here.
    if (globalThread < prefixBytes) {
        accumulate_bin_warp_private(
            warpHistogram,
            static_cast<unsigned int>(input[globalThread]) - from,
            binCount,
            lane);
    }

    const unsigned char* const alignedInput = input + prefixBytes;
    const unsigned int remainingBytes = inputSize - prefixBytes;
    const unsigned int vectorCount = remainingBytes / kVectorBytes;
    const uint4* const vectorInput = reinterpret_cast<const uint4*>(alignedInput);

    // Main vectorized scan: one 16-byte load per iteration, then update the
    // warp-private shared-memory histogram.
    for (unsigned int vec = globalThread; vec < vectorCount; vec += globalStride) {
        const uint4 packed = vectorInput[vec];
        process_packed_u32(warpHistogram, packed.x, from, binCount, lane);
        process_packed_u32(warpHistogram, packed.y, from, binCount, lane);
        process_packed_u32(warpHistogram, packed.z, from, binCount, lane);
        process_packed_u32(warpHistogram, packed.w, from, binCount, lane);
    }

    // Scalar tail after the vector region. tailBytes < 16, so again only the
    // first few global threads participate.
    const unsigned int tailStart = prefixBytes + vectorCount * kVectorBytes;
    const unsigned int tailBytes = inputSize - tailStart;
    if (globalThread < tailBytes) {
        accumulate_bin_warp_private(
            warpHistogram,
            static_cast<unsigned int>(input[tailStart + globalThread]) - from,
            binCount,
            lane);
    }

    __syncthreads();

    // The requested range is at most 256 bins wide, so a single block always has
    // enough threads to assign one thread per output bin for the final merge.
    if (tid < binCount) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += sharedHistogram[static_cast<unsigned int>(w) * binCount + tid];
        }

        // For a single-block launch, write the complete histogram directly and
        // avoid global atomics and a separate memset. For multi-block launches,
        // accumulate into a zeroed output array.
        if (gridDim.x == 1) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

template <int BlockSize>
inline void launch_histogram_kernel(const unsigned char* input,
                                    unsigned int* histogram,
                                    unsigned int inputSize,
                                    unsigned int from,
                                    unsigned int binCount,
                                    size_t histogramBytes,
                                    unsigned int workUnits16,
                                    unsigned int smCount)
{
    static_assert(BlockSize % kWarpSize == 0, "Block size must be a multiple of 32.");
    static_assert(BlockSize >= kMaxBins,
                  "Block size must be at least the maximum number of output bins.");

    constexpr int kWarpsPerBlock = BlockSize / kWarpSize;
    constexpr unsigned int kBlockSizeU = static_cast<unsigned int>(BlockSize);
    constexpr unsigned int kBlocksPerSm = 2048u / kBlockSizeU;

    unsigned int blocksForWork = (workUnits16 + kBlockSizeU - 1u) / kBlockSizeU;
    if (blocksForWork == 0u) {
        blocksForWork = 1u;
    }

    // smCount == 0 is used intentionally for the small-input fast paths below:
    // it forces a single-block launch, which lets the kernel write the full
    // histogram directly without a prior memset.
    const unsigned int maxBlocks = (smCount != 0u) ? (smCount * kBlocksPerSm) : 1u;

    unsigned int gridSize = blocksForWork;
    if (gridSize > maxBlocks) {
        gridSize = maxBlocks;
    }
    if (gridSize == 0u) {
        gridSize = 1u;
    }

    if (gridSize != 1u) {
        // Multi-block launches atomically accumulate into the output array, so it
        // must be zeroed first. The operation is intentionally async; the caller
        // owns any required synchronization.
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
    }

    const size_t sharedMemoryBytes =
        static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(binCount) * sizeof(unsigned int);

    histogram_range_kernel<BlockSize>
        <<<gridSize, BlockSize, sharedMemoryBytes, 0>>>(
            input,
            histogram,
            inputSize,
            from,
            binCount);
}

}  // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (histogram == nullptr) {
        return;
    }
    if (from < 0 || to < from || to > 255) {
        return;
    }

    const unsigned int binCount = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes =
        static_cast<size_t>(binCount) * sizeof(unsigned int);

    // An empty input still needs a valid zero histogram.
    if (input == nullptr || inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogramBytes, 0);
        return;
    }

    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int workUnits16 =
        (inputSize >> 4) + static_cast<unsigned int>((inputSize & (kVectorBytes - 1u)) != 0u);
    const unsigned char* const inputBytes =
        reinterpret_cast<const unsigned char*>(input);

    // Small fast paths:
    // - <= 256 vector units  -> one 256-thread block
    // - <= 512 vector units  -> one 512-thread block
    //
    // Both avoid a pre-launch memset because the single block writes the entire
    // output histogram directly.
    if (workUnits16 <= 256u) {
        launch_histogram_kernel<256>(
            inputBytes,
            histogram,
            inputSize,
            fromU,
            binCount,
            histogramBytes,
            workUnits16,
            0u);
        return;
    }

    if (workUnits16 <= 512u) {
        launch_histogram_kernel<512>(
            inputBytes,
            histogram,
            inputSize,
            fromU,
            binCount,
            histogramBytes,
            workUnits16,
            0u);
        return;
    }

    int currentDevice = 0;
    cudaGetDevice(&currentDevice);

    int smCountInt = 0;
    cudaDeviceGetAttribute(&smCountInt, cudaDevAttrMultiProcessorCount, currentDevice);
    const unsigned int smCount =
        (smCountInt > 0) ? static_cast<unsigned int>(smCountInt) : 0u;

    // Adaptive block-size selection:
    // - 256 threads exposes more CTAs for medium-sized inputs.
    // - 512 threads cuts block-final merge traffic once there is enough work to
    //   keep all SMs busy even with the larger block.
    if (smCount != 0u && workUnits16 >= smCount * 512u) {
        launch_histogram_kernel<512>(
            inputBytes,
            histogram,
            inputSize,
            fromU,
            binCount,
            histogramBytes,
            workUnits16,
            smCount);
    } else {
        launch_histogram_kernel<256>(
            inputBytes,
            histogram,
            inputSize,
            fromU,
            binCount,
            histogramBytes,
            workUnits16,
            smCount);
    }

    // No synchronization here by design; the caller requested to handle it.
}