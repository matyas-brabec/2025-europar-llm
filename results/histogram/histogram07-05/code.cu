#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Restricted byte histogram for modern NVIDIA data-center GPUs.
//
// Design summary:
//   - The output is histogram[bin] where bin = byte_value - from.
//   - Each block owns a privatized histogram in shared memory.
//   - To avoid shared-memory bank conflicts during updates, the block-private histogram is replicated
//     32 times. A thread uses copy `threadIdx.x % 32`.
//   - The requested layout is used exactly:
//         offset(bin, copy) = bin * 32 + copy
//     Because the bank index is the low 5 bits, copy `c` always lives in bank `c`, so a warp can
//     update the privatized histograms with no intra-warp bank conflicts.
//   - The tuned default `itemsPerThread = 16` is a good fit for A100/H100-class GPUs: it maps to one
//     aligned 16-byte vector load per thread while amortizing block init/reduction overhead.
//   - The grid is sized by occupancy because this is a grid-stride kernel; launching only enough
//     resident blocks minimizes the number of per-block global merge atomics.

constexpr int histogramCopies = 32;
constexpr int blockSize       = 256;
constexpr int itemsPerThread  = 16;

static_assert(histogramCopies == 32,
              "This kernel relies on exactly 32 shared-memory histogram copies.");
static_assert((blockSize % 32) == 0,
              "blockSize must be a multiple of 32 so whole warps map cleanly to the 32 copies.");
static_assert(itemsPerThread > 0,
              "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value)
{
    // Portable warp reduction; the merge phase is tiny relative to the byte-processing phase.
    #pragma unroll
    for (int offset = histogramCopies / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset, histogramCopies);
    }
    return value;
}

__device__ __forceinline__ void accumulate_byte(
    unsigned char value,
    unsigned int from_u,
    unsigned int range_u,
    unsigned int* __restrict__ shared_hist_lane)
{
    // One unsigned comparison implements the inclusive range test:
    //   bin = value - from
    //   bin < range   <=>   from <= value <= to
    //
    // The input is interpreted as unsigned bytes so signed `char` does not mis-handle values > 127.
    const unsigned int bin = static_cast<unsigned int>(value) - from_u;
    if (bin < range_u) {
        // `shared_hist_lane` already points at this thread's copy (lane/copy number),
        // so only the requested `bin * 32` stride remains.
        atomicAdd(shared_hist_lane + bin * histogramCopies, 1u);
    }
}

__device__ __forceinline__ void accumulate_packed_u32(
    uint32_t packed,
    unsigned int from_u,
    unsigned int range_u,
    unsigned int* __restrict__ shared_hist_lane)
{
    // The tuned path loads a `uint4` (16 bytes). Each 32-bit word contains 4 consecutive bytes.
    // NVIDIA GPUs are little-endian, so repeatedly consuming the low byte is the natural unpacking.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int bin = (packed & 0xFFu) - from_u;
        if (bin < range_u) {
            atomicAdd(shared_hist_lane + bin * histogramCopies, 1u);
        }
        packed >>= 8;
    }
}

template<int ItemsPerThread>
struct ChunkProcessor {
    static __device__ __forceinline__ void full(
        const unsigned char* __restrict__ input,
        uint64_t idx,
        unsigned int from_u,
        unsigned int range_u,
        unsigned int* __restrict__ shared_hist_lane,
        bool /*can_vectorize*/)
    {
        // Generic fallback if `itemsPerThread` is changed away from the tuned default.
        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            accumulate_byte(input[idx + static_cast<uint64_t>(i)],
                            from_u, range_u, shared_hist_lane);
        }
    }
};

template<>
struct ChunkProcessor<16> {
    static __device__ __forceinline__ void full(
        const unsigned char* __restrict__ input,
        uint64_t idx,
        unsigned int from_u,
        unsigned int range_u,
        unsigned int* __restrict__ shared_hist_lane,
        bool can_vectorize)
    {
        if (can_vectorize) {
            // Tuned default path: one aligned 16-byte global load per thread.
            const uint4 v = reinterpret_cast<const uint4*>(input + idx)[0];
            accumulate_packed_u32(v.x, from_u, range_u, shared_hist_lane);
            accumulate_packed_u32(v.y, from_u, range_u, shared_hist_lane);
            accumulate_packed_u32(v.z, from_u, range_u, shared_hist_lane);
            accumulate_packed_u32(v.w, from_u, range_u, shared_hist_lane);
            return;
        }

        // Robust fallback for the uncommon case where the caller passes a sub-pointer that is not
        // 16-byte aligned. Correctness is unchanged; only the vector-load fast path is disabled.
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            accumulate_byte(input[idx + static_cast<uint64_t>(i)],
                            from_u, range_u, shared_hist_lane);
        }
    }
};

template<int ItemsPerThread>
static __device__ __forceinline__ void process_tail_chunk(
    const unsigned char* __restrict__ input,
    uint64_t idx,
    uint64_t input_size,
    unsigned int from_u,
    unsigned int range_u,
    unsigned int* __restrict__ shared_hist_lane)
{
    // Only the final partial chunk for a thread reaches this path, so a simple unrolled scalar tail
    // is the best trade-off.
    #pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        const uint64_t pos = idx + static_cast<uint64_t>(i);
        if (pos < input_size) {
            accumulate_byte(input[pos], from_u, range_u, shared_hist_lane);
        }
    }
}

template<int ItemsPerThread>
__global__ __launch_bounds__(blockSize)
void histogram_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    extern __shared__ unsigned int shared_hist[];

    const unsigned int from_u  = static_cast<unsigned int>(from);
    const unsigned int range_u = static_cast<unsigned int>(to - from) + 1u;

    // Copy selection required by the prompt: threadIdx.x % 32.
    const unsigned int lane = threadIdx.x & 31u;

    // Shared layout is:
    //   bin 0: copy0 copy1 ... copy31
    //   bin 1: copy0 copy1 ... copy31
    //   ...
    //
    // So `shared_hist_lane = shared_hist + lane` means all accesses by this thread are to:
    //   shared_hist[bin * 32 + lane]
    // Every warp lane maps to a unique bank, eliminating intra-warp bank conflicts.
    unsigned int* const shared_hist_lane = shared_hist + lane;

    const unsigned int shared_entries = range_u * histogramCopies;

    // Zero the block-private histogram copies.
    for (unsigned int i = threadIdx.x; i < shared_entries; i += blockDim.x) {
        shared_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned char* const input_u8 = reinterpret_cast<const unsigned char*>(input);
    const uint64_t input_size  = static_cast<uint64_t>(inputSize);
    const uint64_t block_work  = static_cast<uint64_t>(blockDim.x) *
                                 static_cast<uint64_t>(ItemsPerThread);
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x) * block_work;

    // Each thread owns a contiguous chunk of `itemsPerThread` bytes. Across a warp, those chunks are
    // adjacent, so the global loads are naturally coalesced.
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * block_work +
                   static_cast<uint64_t>(threadIdx.x) *
                   static_cast<uint64_t>(ItemsPerThread);

    // When ItemsPerThread == 16, every full chunk begins at a multiple of 16, so alignment of the
    // base pointer alone determines whether all full-chunk vector loads are legal.
    const bool can_vectorize =
        (ItemsPerThread == 16) &&
        ((reinterpret_cast<uintptr_t>(input_u8) & static_cast<uintptr_t>(0x0F)) == 0u);

    for (; idx + static_cast<uint64_t>(ItemsPerThread) <= input_size; idx += grid_stride) {
        ChunkProcessor<ItemsPerThread>::full(
            input_u8, idx, from_u, range_u, shared_hist_lane, can_vectorize);
    }

    if (idx < input_size) {
        process_tail_chunk<ItemsPerThread>(
            input_u8, idx, input_size, from_u, range_u, shared_hist_lane);
    }

    __syncthreads();

    // Final merge:
    // each warp reduces one bin at a time. Lane `c` reads copy `c`, so the readback is also
    // bank-conflict-free under the [bin * 32 + copy] layout.
    const unsigned int warp            = threadIdx.x >> 5;
    const unsigned int warps_per_block = blockDim.x >> 5;

    for (unsigned int bin = warp; bin < range_u; bin += warps_per_block) {
        unsigned int sum = shared_hist[bin * histogramCopies + lane];
        sum = warp_reduce_sum(sum);
        if (lane == 0u && sum != 0u) {
            // One global atomic per (block, bin) keeps expensive global serialization small.
            atomicAdd(histogram + bin, sum);
        }
    }
}

} // anonymous namespace

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // Output size is exactly the requested inclusive range [from, to].
    const unsigned int range = static_cast<unsigned int>(to - from) + 1u;

    // The kernel accumulates into global memory with atomicAdd, so the output histogram must start at 0.
    // This is intentionally asynchronous; the caller requested to handle synchronization externally.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), 0);

    if (inputSize == 0u) {
        return;
    }

    // Shared memory required by the prompt:
    //   (to - from + 1) bins * 32 copies * sizeof(unsigned int)
    //
    // Worst case:
    //   256 bins * 32 copies * 4 bytes = 32 KiB
    //
    // That comfortably fits on modern GPUs, so no opt-in for >48 KiB dynamic shared memory is needed.
    const size_t sharedMemBytes =
        static_cast<size_t>(range) * histogramCopies * sizeof(unsigned int);

    // This kernel is shared-memory heavy and streams the input once. On A100/H100-class GPUs,
    // preferring shared memory over L1 is typically the right carveout.
    cudaFuncSetAttribute(
        histogram_kernel<itemsPerThread>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    // Because the kernel is grid-stride, launching only enough resident blocks to fill the machine
    // is better than launching one block per input tile: fewer blocks means fewer final global merges.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        histogram_kernel<itemsPerThread>,
        blockSize,
        sharedMemBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSm < 1) {
        activeBlocksPerSm = 1;
    }

    const uint64_t workPerBlock =
        static_cast<uint64_t>(blockSize) * static_cast<uint64_t>(itemsPerThread);
    const uint64_t blocksNeeded =
        (static_cast<uint64_t>(inputSize) + workPerBlock - 1ull) / workPerBlock;

    uint64_t grid = static_cast<uint64_t>(smCount) *
                    static_cast<uint64_t>(activeBlocksPerSm);
    if (grid > blocksNeeded) {
        grid = blocksNeeded;
    }
    if (grid == 0u) {
        grid = 1u;
    }

    // No explicit synchronization here; the caller asked for asynchronous behavior.
    histogram_kernel<itemsPerThread>
        <<<static_cast<unsigned int>(grid), blockSize, sharedMemBytes, 0>>>(
            input, histogram, inputSize, from, to);
}