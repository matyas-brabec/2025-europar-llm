#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Fixed at 256 because:
// 1) the byte domain is exactly [0, 255], so threadIdx.x can map directly to byte values
//    during shared-memory initialization, and
// 2) the requested output range can never exceed 256 bins, so the same block size lets the
//    first `range_size` threads perform the final reduction, one thread per output bin.
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr unsigned int kByteValueCount = 256u;

static_assert(kBlockSize == static_cast<int>(kByteValueCount),
              "kBlockSize must be 256.");
static_assert((kBlockSize % kWarpSize) == 0,
              "kBlockSize must be a multiple of the warp size.");

inline unsigned int div_up_u32(const unsigned int n, const unsigned int d)
{
    // Overflow-safe ceil(n / d) for unsigned 32-bit values.
    return (n == 0u) ? 0u : 1u + (n - 1u) / d;
}

static __device__ __forceinline__ void accumulate_byte(
    const unsigned int byte_value,
    const unsigned int from,
    const unsigned int range_size,
    const unsigned int active_mask,
    const unsigned int lane,
    unsigned int* __restrict__ warp_hist)
{
    const unsigned int output_bin = byte_value - from;
    const bool in_range = output_bin < range_size;

    // Only lanes whose byte is inside [from, to] participate.
    // __match_any_sync groups equal byte values inside the active valid lanes, so exactly
    // one leader lane updates the warp-private shared histogram and adds the whole group's
    // population at once. This removes both global atomics in the hot path and nearly all
    // shared-memory update contention.
    const unsigned int valid_mask = __ballot_sync(active_mask, in_range);
    if (in_range) {
        const unsigned int peers = __match_any_sync(valid_mask, byte_value);
        const unsigned int leader = static_cast<unsigned int>(__ffs(peers) - 1);
        if (lane == leader) {
            // Safe without atomics:
            // - this histogram is private to one warp, and
            // - within this helper invocation there is exactly one leader per distinct byte.
            warp_hist[byte_value] += __popc(peers);
        }
    }
}

static __device__ __forceinline__ void accumulate_word(
    const unsigned int word,
    const unsigned int from,
    const unsigned int range_size,
    const unsigned int active_mask,
    const unsigned int lane,
    unsigned int* __restrict__ warp_hist)
{
    accumulate_byte( word        & 0xFFu, from, range_size, active_mask, lane, warp_hist);
    accumulate_byte((word >>  8) & 0xFFu, from, range_size, active_mask, lane, warp_hist);
    accumulate_byte((word >> 16) & 0xFFu, from, range_size, active_mask, lane, warp_hist);
    accumulate_byte((word >> 24) & 0xFFu, from, range_size, active_mask, lane, warp_hist);
}

// Range-restricted byte histogram.
// Optimization strategy:
// - Each warp owns a private 256-bin histogram in shared memory.
// - Using full 256-bin warp histograms costs only 8 KiB/block at 256 threads, but it
//   enables direct indexing by raw byte value and very simple block-wide init/reduction.
// - The input is processed with aligned 32-bit loads for the bulk path.
// - Each block emits at most one global atomic add per requested output bin.
__global__ __launch_bounds__(kBlockSize)
void histogram_range_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    const unsigned int input_size,
    const unsigned int from,
    const unsigned int range_size)
{
    __shared__ unsigned int shared_hist[kWarpsPerBlock][kByteValueCount];

    // One thread per possible byte value; each thread clears that value across all warp
    // histograms. This is conflict-free and maps perfectly onto the 256-thread block.
    #pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
        shared_hist[w][threadIdx.x] = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned int warp = threadIdx.x >> 5;
    unsigned int* const warp_hist = shared_hist[warp];

    const unsigned char* const input_u8 =
        reinterpret_cast<const unsigned char*>(input);

    const unsigned int global_thread =
        static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(kBlockSize) + threadIdx.x;
    const unsigned int global_stride =
        static_cast<unsigned int>(gridDim.x) * static_cast<unsigned int>(kBlockSize);

    // Peel a tiny scalar prefix so the bulk path can use aligned 32-bit loads even if the
    // caller passes a sub-buffer pointer. For a plain cudaMalloc base pointer, the prefix
    // is normally zero.
    const uintptr_t input_addr = reinterpret_cast<uintptr_t>(input_u8);
    const unsigned int prefix = static_cast<unsigned int>((4u - (input_addr & 3u)) & 3u);
    const unsigned int aligned_prefix = (prefix < input_size) ? prefix : input_size;

    for (unsigned int idx = global_thread; idx < aligned_prefix; idx += global_stride) {
        const unsigned int active = __activemask();
        accumulate_byte(static_cast<unsigned int>(input_u8[idx]),
                        from, range_size, active, lane, warp_hist);
    }

    const unsigned int remaining = input_size - aligned_prefix;
    const unsigned int num_words = remaining >> 2;
    const unsigned int* const input_words =
        reinterpret_cast<const unsigned int*>(input_u8 + aligned_prefix);

    for (unsigned int word_idx = global_thread; word_idx < num_words; word_idx += global_stride) {
        const unsigned int active = __activemask();
        const unsigned int word = input_words[word_idx];
        accumulate_word(word, from, range_size, active, lane, warp_hist);
    }

    const unsigned int tail_base = aligned_prefix + (num_words << 2);
    for (unsigned int idx = tail_base + global_thread; idx < input_size; idx += global_stride) {
        const unsigned int active = __activemask();
        accumulate_byte(static_cast<unsigned int>(input_u8[idx]),
                        from, range_size, active, lane, warp_hist);
    }

    __syncthreads();

    // One thread per requested output bin. Because the range length is at most 256, the
    // first `range_size` threads can reduce the requested bins directly.
    if (threadIdx.x < range_size) {
        const unsigned int byte_value = from + threadIdx.x;
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_hist[w][byte_value];
        }

        if (sum != 0u) {
            atomicAdd(histogram + threadIdx.x, sum);
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
    // The prompt guarantees valid inputs, but this defensive guard avoids undefined
    // behavior if the function is called incorrectly.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range_size = static_cast<unsigned int>(to - from) + 1u;

    // The caller provides device pointers. Clear the output asynchronously, then launch
    // the kernel in the same default stream. By contract, synchronization is the caller's
    // responsibility, so this function deliberately does not synchronize.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range_size) * sizeof(unsigned int));
    if (inputSize == 0u) {
        return;
    }

    int device = 0;
    int sm_count = 1;
    int blocks_per_sm = 4;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        histogram_range_kernel,
        kBlockSize,
        0);

    if (sm_count < 1) {
        sm_count = 1;
    }
    if (blocks_per_sm < 1) {
        blocks_per_sm = 4;
    }

    // One resident wave is sufficient because the kernel uses a grid-stride loop; more
    // blocks would mostly increase the number of final block-to-global atomic updates.
    // For small inputs, cap the grid so we do not launch many nearly idle blocks.
    unsigned int grid =
        static_cast<unsigned int>(sm_count) * static_cast<unsigned int>(blocks_per_sm);
    const unsigned int blocks_for_input =
        div_up_u32(inputSize, static_cast<unsigned int>(kBlockSize));

    if (blocks_for_input < grid) {
        grid = blocks_for_input;
    }
    if (grid == 0u) {
        grid = 1u;
    }

    histogram_range_kernel<<<grid, kBlockSize>>>(
        input,
        histogram,
        inputSize,
        from_u,
        range_size);
}