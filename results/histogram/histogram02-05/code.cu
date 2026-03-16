#include <cuda_runtime.h>

namespace detail {

// Tuned for modern NVIDIA data-center GPUs (A100/H100 class):
// - 256 threads/block gives exactly one thread per possible byte value in the final reduction.
// - 8 warps/block is enough to keep the machine busy while keeping shared-memory usage tiny.
// - Each thread vector-loads 16 bytes at a time via uint4.
constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr unsigned int kVectorBytes = sizeof(uint4);
constexpr unsigned int kBytesPerBlockIteration =
    static_cast<unsigned int>(kBlockThreads) * kVectorBytes;

// For this kernel, launching at the absolute occupancy limit is not always optimal:
// every extra block adds another per-block reduction into the global histogram.
// On Ampere/Hopper, 4 CTAs/SM (32 resident warps/SM) is typically enough to saturate
// memory throughput for this workload while reducing global atomic pressure.
constexpr unsigned int kTargetBlocksPerSM = 4u;

static_assert(kBlockThreads % kWarpSize == 0,
              "Block size must be a multiple of the warp size.");
static_assert(kBlockThreads >= 256,
              "Block size must be at least 256 so one thread can reduce each possible byte bin.");

// Count one byte into this warp's shared-memory histogram.
//
// The histogram is private per warp, which removes inter-warp contention.
// __match_any_sync() collapses identical byte values inside a warp, so a whole group
// of equal characters becomes one shared-memory atomic instead of up to 32 atomics.
//
// A shared-memory atomic is intentionally retained here even though the histogram is
// warp-private: on Volta+ GPUs with independent thread scheduling, atomics keep the
// updates correct without relying on undocumented warp-synchronous memory behavior.
__device__ __forceinline__ void accumulate_byte(
    unsigned int value,
    unsigned int from,
    unsigned int bins,
    unsigned int lane,
    unsigned int* warp_hist)
{
    const unsigned int bin = value - from;
    if (bin < bins) {
        const unsigned int active = __activemask();
        const unsigned int peers = __match_any_sync(active, bin);
        const unsigned int leader = static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1);
        if (lane == leader) {
            atomicAdd(warp_hist + bin, static_cast<unsigned int>(__popc(peers)));
        }
    }
}

// Unpack four bytes from a 32-bit word and feed them to the byte accumulator.
// NVIDIA GPUs are little-endian, so the low byte is the first byte in memory.
__device__ __forceinline__ void accumulate_word(
    unsigned int word,
    unsigned int from,
    unsigned int bins,
    unsigned int lane,
    unsigned int* warp_hist)
{
    accumulate_byte((word >>  0) & 0xFFu, from, bins, lane, warp_hist);
    accumulate_byte((word >>  8) & 0xFFu, from, bins, lane, warp_hist);
    accumulate_byte((word >> 16) & 0xFFu, from, bins, lane, warp_hist);
    accumulate_byte((word >> 24) & 0xFFu, from, bins, lane, warp_hist);
}

// One-pass histogram kernel:
// 1) each warp accumulates into its own shared-memory histogram,
// 2) the block reduces warp-private histograms,
// 3) the block contributes to the final global histogram.
//
// The requested histogram range is continuous and at most 256 bins wide, so the shared
// footprint is very small: warps_per_block * bins * sizeof(unsigned int), i.e. at most 8 KB.
template <int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS)
__global__ void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from,
    unsigned int bins)
{
    constexpr int kWarps = BLOCK_THREADS / kWarpSize;
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    // Layout in shared memory:
    //   [warp0 bins][warp1 bins][warp2 bins]...[warpN bins]
    unsigned int* const warp_hist = s_hist + warp * bins;

    // Zero only this warp's private histogram. Because the histogram is warp-private,
    // a warp-level sync is sufficient before accumulation starts.
    for (unsigned int bin = lane; bin < bins; bin += static_cast<unsigned int>(kWarpSize)) {
        warp_hist[bin] = 0u;
    }
    __syncwarp();

    // Treat the input as raw bytes. This is important because plain 'char' may be signed,
    // but histogram bins are defined over ordinal byte values 0..255.
    const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(input);

    const unsigned int global_thread = blockIdx.x * BLOCK_THREADS + tid;
    const unsigned int total_threads = gridDim.x * BLOCK_THREADS;

    // Alignment-safe vectorization:
    // cudaMalloc() pointers are well aligned, but handling an arbitrary sub-buffer pointer
    // costs almost nothing and preserves correctness for all callers.
    const unsigned int misalignment =
        static_cast<unsigned int>(reinterpret_cast<size_t>(bytes) & (kVectorBytes - 1u));
    const unsigned int prefix = (kVectorBytes - misalignment) & (kVectorBytes - 1u);
    const unsigned int scalar_prefix = (prefix < inputSize) ? prefix : inputSize;

    // Scalar prefix until the pointer is 16-byte aligned.
    for (unsigned int i = global_thread; i < scalar_prefix; i += total_threads) {
        accumulate_byte(static_cast<unsigned int>(bytes[i]), from, bins, lane, warp_hist);
    }

    // Vectorized main loop: one uint4 == 16 input bytes.
    const unsigned int aligned_size = inputSize - scalar_prefix;
    const unsigned int vec_count = aligned_size / kVectorBytes;
    const uint4* const vec_input = reinterpret_cast<const uint4*>(bytes + scalar_prefix);

    for (unsigned int i = global_thread; i < vec_count; i += total_threads) {
        const uint4 v = vec_input[i];

        accumulate_word(v.x, from, bins, lane, warp_hist);
        accumulate_word(v.y, from, bins, lane, warp_hist);
        accumulate_word(v.z, from, bins, lane, warp_hist);
        accumulate_word(v.w, from, bins, lane, warp_hist);
    }

    // Scalar tail for the final 0..15 bytes.
    const unsigned int tail_start = scalar_prefix + vec_count * kVectorBytes;
    for (unsigned int i = tail_start + global_thread; i < inputSize; i += total_threads) {
        accumulate_byte(static_cast<unsigned int>(bytes[i]), from, bins, lane, warp_hist);
    }

    // Make every warp's shared histogram visible before the block-wide reduction.
    __syncthreads();

    // Final block reduction:
    // BLOCK_THREADS == 256 and bins <= 256, so at most one bin per thread.
    if (tid < bins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            sum += s_hist[w * bins + tid];
        }

        // If only one block was launched, the kernel can overwrite the output directly and
        // the host-side memset can be skipped entirely. Otherwise each block atomically
        // contributes its partial sum to the global histogram.
        if (gridDim.x == 1u) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

__host__ __forceinline__ unsigned int ceil_div_u32(unsigned int n, unsigned int d)
{
    return (n + d - 1u) / d;
}

inline unsigned int choose_grid(unsigned int inputSize, size_t shared_bytes)
{
    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    int active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        histogram_range_kernel<kBlockThreads>,
        kBlockThreads,
        shared_bytes);

    unsigned int blocks_per_sm =
        static_cast<unsigned int>((active_blocks_per_sm > 0) ? active_blocks_per_sm : 1);
    if (blocks_per_sm > kTargetBlocksPerSM) {
        blocks_per_sm = kTargetBlocksPerSM;
    }

    const unsigned int max_launch_blocks =
        static_cast<unsigned int>((sm_count > 0) ? sm_count : 1) * blocks_per_sm;

    const unsigned int blocks_for_coverage =
        ceil_div_u32(inputSize, kBytesPerBlockIteration);

    return (blocks_for_coverage < max_launch_blocks) ? blocks_for_coverage : max_launch_blocks;
}

} // namespace detail

// Launch helper expected by the prompt.
//
// Notes:
// - input and histogram are assumed to be device pointers allocated by cudaMalloc().
// - histogram has exactly (to - from + 1) elements.
// - the function enqueues work on stream 0 and performs no host/device synchronization;
//   the caller remains responsible for synchronization and error handling policy.
void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const size_t histogram_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // Empty input is a pure "write zeros" case.
    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogram_bytes);
        return;
    }

    const size_t shared_bytes =
        static_cast<size_t>(detail::kWarpsPerBlock) *
        static_cast<size_t>(bins) *
        sizeof(unsigned int);

    const unsigned int grid = detail::choose_grid(inputSize, shared_bytes);

    // For a single-block launch, the kernel overwrites every output bin directly,
    // so the explicit memset can be skipped.
    if (grid != 1u) {
        cudaMemsetAsync(histogram, 0, histogram_bytes);
    }

    detail::histogram_range_kernel<detail::kBlockThreads>
        <<<grid, detail::kBlockThreads, shared_bytes>>>(
            input,
            histogram,
            inputSize,
            static_cast<unsigned int>(static_cast<unsigned char>(from)),
            bins);
}