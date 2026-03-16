#include <cuda_runtime.h>

// Range-restricted byte histogram optimized for modern NVIDIA data-center GPUs.
//
// Design:
// - Each block privatizes the requested histogram range in shared memory.
// - To eliminate shared-memory bank conflicts during updates, the shared histogram stores 32
//   interleaved copies. Copy c of bin b lives at s_hist[b * 32 + c], and thread t always updates
//   copy (t & 31). Within a warp, every lane therefore maps to a distinct bank.
// - Input is processed with a grid-stride loop. The compile-time constant itemsPerThread controls
//   how much input each thread consumes. 16 is a good default on A100/H100-class GPUs for this
//   shared-atomic-heavy kernel: it provides useful ILP without materially hurting occupancy.
// - Reduction back to global memory is done warp-by-warp: each warp reduces one bin at a time by
//   reading the 32 copies conflict-free, then emits a single global atomic add.
//
// Full-range worst-case shared-memory footprint:
// 256 bins * 32 copies * 4 bytes = 32 KiB per block.

constexpr int kHistogramCopies      = 32;
constexpr int kHistogramCopiesLog2  = 5;
constexpr unsigned int kCopyMask    = static_cast<unsigned int>(kHistogramCopies - 1);
constexpr int kBlockSize            = 256;
constexpr int kMinBlocksPerSM       = 5;
constexpr int itemsPerThread        = 16;
constexpr int kVectorWidth          = 4;
constexpr int kVectorsPerThread     = itemsPerThread / kVectorWidth;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(kHistogramCopies == (1 << kHistogramCopiesLog2),
              "kHistogramCopiesLog2 must match kHistogramCopies.");
static_assert((kHistogramCopies & (kHistogramCopies - 1)) == 0,
              "kHistogramCopies must be a power of two.");
static_assert(kBlockSize % kHistogramCopies == 0,
              "The block size must be a whole number of warps.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(itemsPerThread % kVectorWidth == 0,
              "The vectorized load path requires itemsPerThread to be a multiple of 4.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
    // Lane 0 receives the full sum; other lanes hold partial values, which is fine because only
    // lane 0 performs the final global atomic add.
#pragma unroll
    for (int offset = kHistogramCopies / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ void update_private_histogram(
    unsigned int* __restrict__ privateHist,
    const unsigned int from,
    const unsigned int numBins,
    const unsigned char value)
{
    // Unsigned subtraction folds both range checks into one compare:
    //   value in [from, from + numBins - 1]  <=>  (value - from) < numBins
    const unsigned int bin = static_cast<unsigned int>(value) - from;
    if (bin < numBins) {
        // privateHist already points at this thread's copy index (lane), so stepping by 32 lands on
        // the same copy for the next logical histogram bin:
        //   privateHist[bin * 32] == s_hist[bin * 32 + lane]
        atomicAdd(&privateHist[bin << kHistogramCopiesLog2], 1u);
    }
}

__device__ __forceinline__ void update_private_histogram4(
    unsigned int* __restrict__ privateHist,
    const unsigned int from,
    const unsigned int numBins,
    const uchar4 values)
{
    update_private_histogram(privateHist, from, numBins, values.x);
    update_private_histogram(privateHist, from, numBins, values.y);
    update_private_histogram(privateHist, from, numBins, values.z);
    update_private_histogram(privateHist, from, numBins, values.w);
}

__global__ __launch_bounds__(kBlockSize, kMinBlocksPerSM)
void range_histogram_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int from,
                            const unsigned int numBins)
{
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & kCopyMask;
    const unsigned int warpId        = tid >> kHistogramCopiesLog2;
    const unsigned int warpsPerBlock = blockDim.x >> kHistogramCopiesLog2;

    // Offset the shared histogram base by the thread's copy index. This lets the hot update path
    // use a single shifted index per bin.
    unsigned int* const privateHist = s_hist + lane;

    // Zero the block-private histogram copies.
    const unsigned int sharedWords = numBins << kHistogramCopiesLog2;
    for (unsigned int i = tid; i < sharedWords; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned char* __restrict__ input_u8 =
        reinterpret_cast<const unsigned char*>(input);

    // The API contract states that input points to a cudaMalloc allocation, so the base pointer is
    // naturally aligned. Because each thread starts at a 4-byte-aligned index and advances in
    // 4-byte steps, these uchar4 loads stay aligned and reduce load instruction count.
    const uchar4* __restrict__ input_vec =
        reinterpret_cast<const uchar4*>(input_u8);

    const unsigned long long inputSize64 =
        static_cast<unsigned long long>(inputSize);

    const unsigned long long blockVectorStride =
        static_cast<unsigned long long>(blockDim.x) * static_cast<unsigned long long>(kVectorWidth);
    const unsigned long long workPerBlock =
        blockVectorStride * static_cast<unsigned long long>(kVectorsPerThread);

    const unsigned long long base0 =
        static_cast<unsigned long long>(blockIdx.x) * workPerBlock +
        static_cast<unsigned long long>(tid) * static_cast<unsigned long long>(kVectorWidth);

    const unsigned long long gridStride =
        static_cast<unsigned long long>(gridDim.x) * workPerBlock;

    // Grid-stride traversal of the input.
    for (unsigned long long base = base0; base < inputSize64; base += gridStride) {
#pragma unroll
        for (int v = 0; v < kVectorsPerThread; ++v) {
            const unsigned long long idx64 =
                base + static_cast<unsigned long long>(v) * blockVectorStride;

            if (idx64 < inputSize64) {
                const unsigned int idx = static_cast<unsigned int>(idx64);
                const unsigned int remaining = inputSize - idx;

                if (remaining >= static_cast<unsigned int>(kVectorWidth)) {
                    const uchar4 bytes = input_vec[idx >> 2];
                    update_private_histogram4(privateHist, from, numBins, bytes);
                } else {
                    // Final partial vector near EOF.
                    const unsigned char* const tail = input_u8 + idx;
                    if (remaining > 0u) update_private_histogram(privateHist, from, numBins, tail[0]);
                    if (remaining > 1u) update_private_histogram(privateHist, from, numBins, tail[1]);
                    if (remaining > 2u) update_private_histogram(privateHist, from, numBins, tail[2]);
                }
            }
        }
    }

    __syncthreads();

    // Conflict-free reduction of the 32 copies:
    // one warp handles one logical bin at a time, and lane c reads copy c.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock) {
        const unsigned int binBase = bin << kHistogramCopiesLog2;
        unsigned int count = s_hist[binBase + lane];
        count = warp_reduce_sum(count);

        if (lane == 0u && count != 0u) {
            atomicAdd(&histogram[bin], count);
        }
    }
}

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Precondition from the interface contract:
    //   - input and histogram are device pointers allocated with cudaMalloc
    //   - 0 <= from < to <= 255
    // The caller also owns synchronization, so this function only enqueues work.

    const unsigned int fromU   = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // The kernel only accumulates, so start from a clean output histogram.
    const cudaStream_t stream = 0;
    cudaMemsetAsync(histogram,
                    0,
                    static_cast<size_t>(numBins) * sizeof(unsigned int),
                    stream);

    if (inputSize == 0u) {
        return;
    }

    // Favor a shared-memory-heavy carveout: this kernel is dominated by shared atomics while the
    // input read stream is one-pass and cache-light. This is a performance hint, not a requirement.
    (void)cudaFuncSetAttribute(range_histogram_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    // Shared memory requested in the prompt: 32 copies of the selected histogram range.
    // Worst case (full 0..255 range) is 32 KiB/block, so no max-dynamic-shmem opt-in is needed.
    const size_t sharedBytes =
        static_cast<size_t>(numBins) *
        static_cast<size_t>(kHistogramCopies) *
        sizeof(unsigned int);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
                                                  range_histogram_kernel,
                                                  kBlockSize,
                                                  sharedBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    int gridSize = smCount * activeBlocksPerSM;

    // Avoid launching blocks that would only pay the shared-histogram init/reduction cost without
    // owning any initial input tile.
    const unsigned long long workPerBlock =
        static_cast<unsigned long long>(kBlockSize) *
        static_cast<unsigned long long>(itemsPerThread);

    const int maxUsefulBlocks = static_cast<int>(
        (static_cast<unsigned long long>(inputSize) + workPerBlock - 1ull) / workPerBlock);

    if (gridSize > maxUsefulBlocks) {
        gridSize = maxUsefulBlocks;
    }
    if (gridSize < 1) {
        gridSize = 1;
    }

    range_histogram_kernel<<<gridSize, kBlockSize, sharedBytes, stream>>>(
        input,
        histogram,
        inputSize,
        fromU,
        numBins);
}