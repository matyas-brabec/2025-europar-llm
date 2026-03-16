#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kHistogramBlockSize = 256;
constexpr unsigned int kVectorBytes = 16u;
constexpr unsigned int kInvalidBinSentinel = ~0u;

static_assert((kHistogramBlockSize % kWarpSize) == 0, "Block size must be a multiple of warp size.");
static_assert(sizeof(uint4) == kVectorBytes, "Expected uint4 to be 16 bytes.");

// Warp-aggregate equal-bin updates so only one lane performs the shared-memory atomicAdd.
// Out-of-range bytes all use the same sentinel key, which collapses them into a single no-op group.
__device__ __forceinline__ void warp_aggregated_shared_increment(
    unsigned int bin,
    bool valid,
    int lane,
    unsigned int* warp_hist)
{
    const unsigned int active = __activemask();
    const unsigned int group = __match_any_sync(active, valid ? bin : kInvalidBinSentinel);

    if (valid && lane == (__ffs(group) - 1)) {
        atomicAdd(warp_hist + bin, static_cast<unsigned int>(__popc(group)));
    }
}

__device__ __forceinline__ void accumulate_byte(
    unsigned char value,
    unsigned int from_u,
    unsigned int bins_u,
    int lane,
    unsigned int* warp_hist)
{
    const unsigned int bin = static_cast<unsigned int>(value) - from_u;
    warp_aggregated_shared_increment(bin, bin < bins_u, lane, warp_hist);
}

__device__ __forceinline__ void accumulate_word(
    unsigned int word,
    unsigned int from_u,
    unsigned int bins_u,
    int lane,
    unsigned int* warp_hist)
{
    accumulate_byte(static_cast<unsigned char>( word        & 0xFFu), from_u, bins_u, lane, warp_hist);
    accumulate_byte(static_cast<unsigned char>((word >>  8) & 0xFFu), from_u, bins_u, lane, warp_hist);
    accumulate_byte(static_cast<unsigned char>((word >> 16) & 0xFFu), from_u, bins_u, lane, warp_hist);
    accumulate_byte(static_cast<unsigned char>((word >> 24) & 0xFFu), from_u, bins_u, lane, warp_hist);
}

/*
  Optimized histogram kernel for a contiguous byte range [from, from + bins - 1].

  Strategy:
  - Each warp owns a private histogram slice in shared memory, which removes inter-warp contention.
  - Within a warp, __match_any_sync() combines lanes that hit the same bin, reducing shared atomics.
  - Input is consumed mostly as aligned 16-byte vectors (uint4). Tiny scalar prologue/tail code
    handles arbitrary input pointer alignment without sacrificing vectorized bulk loads.
  - Each block flushes its shared histogram to global memory once, minimizing global atomics.

  Notes:
  - Bytes are interpreted as unsigned char values in [0, 255], independent of host/device char signedness.
  - The global histogram must be zero-initialized before launch.
  - This kernel targets modern NVIDIA GPUs (Volta+); A100/H100 are the intended targets.
*/
template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int bins)
{
    static_assert((BLOCK_SIZE % kWarpSize) == 0, "BLOCK_SIZE must be a multiple of warp size.");
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / kWarpSize;

    extern __shared__ unsigned int s_private_hist[];

    const int tid = static_cast<int>(threadIdx.x);
    const unsigned int gtid =
        static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(BLOCK_SIZE) +
        static_cast<unsigned int>(tid);
    const unsigned int stride =
        static_cast<unsigned int>(gridDim.x) * static_cast<unsigned int>(BLOCK_SIZE);

    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int bins_u = static_cast<unsigned int>(bins);

    unsigned int* const warp_hist = s_private_hist + warp_id * bins;
    const int total_private_bins = WARPS_PER_BLOCK * bins;

    for (int i = tid; i < total_private_bins; i += BLOCK_SIZE) {
        s_private_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned char* const input_bytes = reinterpret_cast<const unsigned char*>(input);

    // Scalar prologue: move to a 16-byte aligned address so the bulk loop can use uint4 loads.
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(input_bytes);
    const unsigned int misalignment =
        static_cast<unsigned int>(base_addr & static_cast<uintptr_t>(kVectorBytes - 1u));
    unsigned int prologue = (kVectorBytes - misalignment) & (kVectorBytes - 1u);
    if (prologue > inputSize) {
        prologue = inputSize;
    }

    // Prologue and tail are each < 16 bytes, so the first few global threads handle them directly.
    if (gtid < prologue) {
        accumulate_byte(input_bytes[gtid], from_u, bins_u, lane, warp_hist);
    }

    const unsigned int remaining = inputSize - prologue;
    const unsigned int vector_count = remaining / kVectorBytes;
    const unsigned int tail_count = remaining % kVectorBytes;
    const unsigned int tail_start = inputSize - tail_count;

    const unsigned char* const aligned_bytes = input_bytes + prologue;
    const uint4* const aligned_input4 = reinterpret_cast<const uint4*>(aligned_bytes);

    for (unsigned int vec_idx = gtid; vec_idx < vector_count; vec_idx += stride) {
        const uint4 v = aligned_input4[vec_idx];
        accumulate_word(v.x, from_u, bins_u, lane, warp_hist);
        accumulate_word(v.y, from_u, bins_u, lane, warp_hist);
        accumulate_word(v.z, from_u, bins_u, lane, warp_hist);
        accumulate_word(v.w, from_u, bins_u, lane, warp_hist);
    }

    if (gtid < tail_count) {
        accumulate_byte(input_bytes[tail_start + gtid], from_u, bins_u, lane, warp_hist);
    }

    __syncthreads();

    // By contract, bins <= 256, so one thread can flush one output bin.
    if (tid < bins) {
        const int bin = tid;
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += s_private_hist[w * bins + bin];
        }

        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

}  // namespace

/*
  Host wrapper:
  - Zeros the output histogram because the kernel accumulates into it with atomicAdd.
  - Uses the default stream because the API does not expose a stream parameter.
  - Does not synchronize; the caller is responsible for synchronization and optional error checks.
  - Caps the grid to the number of concurrently resident blocks to avoid unnecessary per-block flush traffic.
*/
void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // The problem statement promises a valid range; this guard just avoids accidental misuse.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int bins = to - from + 1;
    const size_t histogram_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    cudaMemsetAsync(histogram, 0, histogram_bytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t shared_bytes =
        static_cast<size_t>(kHistogramBlockSize / kWarpSize) *
        static_cast<size_t>(bins) *
        sizeof(unsigned int);

    // One block nominally covers 256 vector loads (4096 bytes of bulk data) before it needs to loop.
    // Prologue/tail are tiny and intentionally ignored in this heuristic.
    const size_t vector_chunks = static_cast<size_t>(inputSize) / kVectorBytes;
    int blocks_for_input = static_cast<int>(
        (vector_chunks + static_cast<size_t>(kHistogramBlockSize) - 1u) /
        static_cast<size_t>(kHistogramBlockSize));
    if (blocks_for_input < 1) {
        blocks_for_input = 1;
    }

    int grid_size = blocks_for_input;

    if (grid_size > 1) {
        int device = 0;
        int sm_count = 1;
        int active_blocks_per_sm = 1;

        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            histogram_range_kernel<kHistogramBlockSize>,
            kHistogramBlockSize,
            shared_bytes);

        if (sm_count < 1) {
            sm_count = 1;
        }
        if (active_blocks_per_sm < 1) {
            active_blocks_per_sm = 1;
        }

        const int max_concurrent_blocks = sm_count * active_blocks_per_sm;
        if (grid_size > max_concurrent_blocks) {
            grid_size = max_concurrent_blocks;
        }
        if (grid_size < 1) {
            grid_size = 1;
        }
    }

    histogram_range_kernel<kHistogramBlockSize>
        <<<grid_size, kHistogramBlockSize, shared_bytes, 0>>>(
            input,
            histogram,
            inputSize,
            from,
            bins);
}