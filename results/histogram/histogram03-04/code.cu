#include <cuda_runtime.h>

// Internal implementation notes:
//
// 1) The histogram range is always at most 256 bins, so a 256-thread block is a good fit:
//    the final flush stage can map one thread to one possible output bin.
//
// 2) The hot counting path uses shared-memory privatization at warp granularity:
//      - each warp owns its own private histogram in shared memory,
//      - threads update only that private histogram while scanning input,
//      - the block merges those warp-private histograms into the global output once at the end.
//    This replaces O(inputSize) global atomics with only O(gridDim.x * range) global atomics.
//
// 3) The kernel reads input through 128-bit vector loads (uint4) when possible.
//    To keep that path correct even if the caller passes an interior device pointer that is not
//    16-byte aligned, the kernel first consumes a tiny scalar prefix until the address is aligned.

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr unsigned int kMaxHistogramBins = 256u;
constexpr unsigned int kVectorBytes = sizeof(uint4);
constexpr size_t kWorkPerBlock =
    static_cast<size_t>(kBlockSize) * static_cast<size_t>(kVectorBytes);

static_assert(kBlockSize % kWarpSize == 0,
              "Block size must be an integer number of warps.");
static_assert(kBlockSize >= static_cast<int>(kMaxHistogramBins),
              "The flush stage expects at least one thread per possible output bin.");

// Unsigned subtraction turns the membership test for [from_u, from_u + range) into one compare.
// Values below from_u underflow and automatically fail the compare.
__device__ __forceinline__ void accumulate_byte(const unsigned int c,
                                                const unsigned int from_u,
                                                const unsigned int range,
                                                unsigned int* const warp_hist)
{
    const unsigned int rel = c - from_u;
    if (rel < range) {
        atomicAdd(&warp_hist[rel], 1u);
    }
}

__device__ __forceinline__ void accumulate_word(unsigned int word,
                                                const unsigned int from_u,
                                                const unsigned int range,
                                                unsigned int* const warp_hist)
{
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        accumulate_byte(word & 0xFFu, from_u, range, warp_hist);
        word >>= 8;
    }
}

__device__ __forceinline__ void accumulate_vec16(const uint4& v,
                                                 const unsigned int from_u,
                                                 const unsigned int range,
                                                 unsigned int* const warp_hist)
{
    accumulate_word(v.x, from_u, range, warp_hist);
    accumulate_word(v.y, from_u, range, warp_hist);
    accumulate_word(v.z, from_u, range, warp_hist);
    accumulate_word(v.w, from_u, range, warp_hist);
}

// Clear all warp-private histograms in parallel.
__device__ __forceinline__ void zero_private_hist(unsigned int* const shared_hist,
                                                  const unsigned int total_private_bins)
{
    for (unsigned int i = threadIdx.x; i < total_private_bins; i += kBlockSize) {
        shared_hist[i] = 0u;
    }
}

// Collapse all warp-private histograms in the block into the final output.
//
// If the grid has only one block, every output bin is fully determined inside this block, so
// we can store directly and skip both the host-side memset and global atomics.
// Otherwise each block contributes a partial histogram and must atomically merge into global memory.
__device__ __forceinline__ void flush_private_hist_to_global(const unsigned int* const shared_hist,
                                                             unsigned int* const histogram,
                                                             const unsigned int range)
{
    const unsigned int tid = threadIdx.x;

    if (tid < range) {
        unsigned int sum = 0u;

#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_hist[static_cast<unsigned int>(w) * range + tid];
        }

        if (gridDim.x == 1) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

// Per-warp shared-memory-privatized histogram kernel.
__global__ __launch_bounds__(kBlockSize)
void histogram_kernel(const char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      const unsigned int inputSize,
                      const unsigned int from_u,
                      const unsigned int range)
{
    extern __shared__ unsigned int shared_hist[];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int total_private_bins =
        static_cast<unsigned int>(kWarpsPerBlock) * range;

    zero_private_hist(shared_hist, total_private_bins);
    __syncthreads();

    unsigned int* const warp_hist = shared_hist + warp_id * range;

    const unsigned int global_thread =
        static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(kBlockSize) + tid;
    const unsigned int global_stride =
        static_cast<unsigned int>(gridDim.x) * static_cast<unsigned int>(kBlockSize);

    const unsigned char* const input_bytes =
        reinterpret_cast<const unsigned char*>(input);

    // Align to 16 bytes so the main loop can legally use uint4 vector loads.
    const size_t input_addr = reinterpret_cast<size_t>(input_bytes);
    const size_t vector_mask = static_cast<size_t>(kVectorBytes - 1u);
    const size_t misalignment = input_addr & vector_mask;
    unsigned int prefix = static_cast<unsigned int>(
        (static_cast<size_t>(kVectorBytes) - misalignment) & vector_mask);
    if (prefix > inputSize) {
        prefix = inputSize;
    }

    // Small unaligned prefix.
    for (unsigned int idx = global_thread; idx < prefix; idx += global_stride) {
        accumulate_byte(static_cast<unsigned int>(input_bytes[idx]), from_u, range, warp_hist);
    }

    // Main aligned vectorized body.
    const unsigned char* const aligned_bytes = input_bytes + prefix;
    const unsigned int aligned_size = inputSize - prefix;
    const unsigned int vec_count = aligned_size / kVectorBytes;
    const uint4* const vec_input = reinterpret_cast<const uint4*>(aligned_bytes);

    for (unsigned int vec_idx = global_thread; vec_idx < vec_count; vec_idx += global_stride) {
        const uint4 v = vec_input[vec_idx];
        accumulate_vec16(v, from_u, range, warp_hist);
    }

    // Final scalar tail.
    const unsigned int tail_base = prefix + vec_count * kVectorBytes;
    for (unsigned int idx = tail_base + global_thread; idx < inputSize; idx += global_stride) {
        accumulate_byte(static_cast<unsigned int>(input_bytes[idx]), from_u, range, warp_hist);
    }

    __syncthreads();
    flush_private_hist_to_global(shared_hist, histogram, range);
}

// The kernel uses a grid-stride loop, so one fully resident wave of blocks is enough to process
// arbitrarily large inputs. Capping the grid to the number of simultaneously resident blocks
// avoids paying extra final global atomics for surplus blocks.
inline unsigned int compute_resident_grid_cap(const size_t shared_mem_bytes)
{
    int active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        histogram_kernel,
        kBlockSize,
        shared_mem_bytes);

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const unsigned int active =
        static_cast<unsigned int>((active_blocks_per_sm > 0) ? active_blocks_per_sm : 1);
    const unsigned int sms =
        static_cast<unsigned int>((sm_count > 0) ? sm_count : 1);

    return active * sms;
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The prompt guarantees valid bounds, but keep a defensive guard so accidental misuse does
    // not turn into an out-of-range memset or shared-memory size.
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    const size_t histogram_bytes =
        static_cast<size_t>(range) * sizeof(unsigned int);
    const size_t shared_mem_bytes =
        static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(range) * sizeof(unsigned int);

    // The interface does not expose a stream parameter, so this function uses stream 0.
    // The caller explicitly owns synchronization, so there is intentionally no cudaDeviceSynchronize().
    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, histogram_bytes, 0);
        return;
    }

    // Fast path: if one block is enough, skip the extra memset.
    // The kernel detects gridDim.x == 1 and overwrites every output bin directly.
    const size_t blocks_for_input =
        (static_cast<size_t>(inputSize) + kWorkPerBlock - 1u) / kWorkPerBlock;

    if (blocks_for_input <= 1u) {
        histogram_kernel<<<1, kBlockSize, shared_mem_bytes, 0>>>(
            input, histogram, inputSize, from_u, range);
        return;
    }

    // Multi-block path: zero the output first, then launch at most one resident wave of blocks.
    // The memset is enqueued before the occupancy/device queries so the device can start it while
    // the host computes the launch configuration.
    cudaMemsetAsync(histogram, 0, histogram_bytes, 0);

    const unsigned int resident_grid_cap = compute_resident_grid_cap(shared_mem_bytes);
    unsigned int grid =
        (blocks_for_input < static_cast<size_t>(resident_grid_cap))
            ? static_cast<unsigned int>(blocks_for_input)
            : resident_grid_cap;

    if (grid == 0u) {
        grid = 1u;
    }

    histogram_kernel<<<grid, kBlockSize, shared_mem_bytes, 0>>>(
        input, histogram, inputSize, from_u, range);
}