#include <cuda_runtime.h>
#include <stdint.h>

// Restricted byte-range histogram for a device-resident char buffer.
//
// Design choices aimed at modern data-center GPUs (A100/H100):
// - The requested histogram has at most 256 bins, so a full per-block histogram fits easily in shared memory.
// - To cut shared-memory contention further, each warp owns a private histogram slice in shared memory.
// - The input is consumed with 16-byte vector loads (uint4) after peeling a tiny scalar prefix to guarantee
//   alignment even if the caller passes an offset pointer into a cudaMalloc allocation.
// - Within each warp, __match_any_sync aggregates identical byte values so repeated lanes issue one shared
//   atomicAdd with the popcount of participating lanes instead of many contending atomics.
// - The host wrapper launches only one occupancy-sized wave of blocks because the kernel already uses
//   grid-stride loops; oversubscribing the grid would only increase the final global histogram merge cost.

namespace
{
constexpr int          kWarpSize       = 32;
constexpr int          kBlockThreads   = 256;
constexpr int          kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr unsigned int kBytesPerVector = 16u;
constexpr int          kMinBlocksPerSM = 4;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");
static_assert(sizeof(uint4) == kBytesPerVector, "uint4 must be 16 bytes.");

// Update one warp-private shared-memory histogram bin.
//
// active_mask is passed in from the surrounding loop iteration instead of recomputing it here.
// That is both cheaper and correct for partially active warps in the last grid-stride iteration.
static __device__ __forceinline__
void update_warp_private_hist(unsigned int* __restrict__ warp_hist,
                              int from,
                              unsigned int num_bins,
                              unsigned int byte_value,
                              unsigned int active_mask,
                              int lane)
{
    const int  bin      = static_cast<int>(byte_value) - from;
    const bool in_range = static_cast<unsigned int>(bin) < num_bins;

    const unsigned int valid_mask = __ballot_sync(active_mask, in_range);
    if (!in_range)
    {
        return;
    }

    const unsigned int peer_mask   = __match_any_sync(valid_mask, bin);
    const int          leader_lane = __ffs(static_cast<int>(peer_mask)) - 1;

    if (lane == leader_lane)
    {
        atomicAdd(warp_hist + bin, static_cast<unsigned int>(__popc(peer_mask)));
    }
}

// Process four adjacent bytes packed into one 32-bit word.
static __device__ __forceinline__
void process_packed_u32(unsigned int packed,
                        unsigned int* __restrict__ warp_hist,
                        int from,
                        unsigned int num_bins,
                        unsigned int active_mask,
                        int lane)
{
    update_warp_private_hist(warp_hist, from, num_bins,  packed        & 0xFFu, active_mask, lane);
    update_warp_private_hist(warp_hist, from, num_bins, (packed >>  8) & 0xFFu, active_mask, lane);
    update_warp_private_hist(warp_hist, from, num_bins, (packed >> 16) & 0xFFu, active_mask, lane);
    update_warp_private_hist(warp_hist, from, num_bins,  packed >> 24,          active_mask, lane);
}

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, kMinBlocksPerSM)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int input_size,
                            int from,
                            unsigned int num_bins)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;

    // Shared layout is [warp0 bins][warp1 bins]...[warpN bins].
    extern __shared__ unsigned int s_warp_histograms[];

    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid & (kWarpSize - 1);
    const int warp_id  = tid >> 5;
    const int bins_i   = static_cast<int>(num_bins);
    const int sh_count = WARPS_PER_BLOCK * bins_i;

    unsigned int* const warp_hist = s_warp_histograms + warp_id * bins_i;

    // Zero the warp-private block histogram.
    for (int i = tid; i < sh_count; i += BLOCK_THREADS)
    {
        s_warp_histograms[i] = 0u;
    }
    __syncthreads();

    const unsigned int global_thread = static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(BLOCK_THREADS) +
                                       static_cast<unsigned int>(tid);
    const unsigned int total_threads = static_cast<unsigned int>(gridDim.x) * static_cast<unsigned int>(BLOCK_THREADS);

    // Interpret the input as raw bytes. This avoids any signed-char ambiguity for values 128..255.
    const unsigned char* const input_bytes = reinterpret_cast<const unsigned char*>(input);

    // Peel a small scalar prefix to guarantee 16-byte alignment for the main uint4 vectorized path.
    const uintptr_t    addr         = reinterpret_cast<uintptr_t>(input_bytes);
    const unsigned int misalignment = static_cast<unsigned int>(addr & static_cast<uintptr_t>(kBytesPerVector - 1u));

    unsigned int scalar_prefix = 0u;
    if (misalignment != 0u)
    {
        scalar_prefix = kBytesPerVector - misalignment;
        if (scalar_prefix > input_size)
        {
            scalar_prefix = input_size;
        }
    }

    // Scalar prefix.
    for (unsigned int idx = global_thread; idx < scalar_prefix; idx += total_threads)
    {
        const unsigned int active_mask = __activemask();
        update_warp_private_hist(
            warp_hist,
            from,
            num_bins,
            static_cast<unsigned int>(input_bytes[idx]),
            active_mask,
            lane);
    }

    // Vectorized main body: 16 bytes per thread per iteration.
    const unsigned int   main_bytes    = input_size - scalar_prefix;
    const unsigned int   num_vecs      = main_bytes / kBytesPerVector;
    const unsigned char* aligned_bytes = input_bytes + scalar_prefix;
    const uint4* const   input_vec     = reinterpret_cast<const uint4*>(aligned_bytes);

    for (unsigned int vec = global_thread; vec < num_vecs; vec += total_threads)
    {
        const unsigned int active_mask = __activemask();
        const uint4        v           = input_vec[vec];

        process_packed_u32(v.x, warp_hist, from, num_bins, active_mask, lane);
        process_packed_u32(v.y, warp_hist, from, num_bins, active_mask, lane);
        process_packed_u32(v.z, warp_hist, from, num_bins, active_mask, lane);
        process_packed_u32(v.w, warp_hist, from, num_bins, active_mask, lane);
    }

    // Scalar tail after the vectorized section.
    const unsigned int tail_base  = scalar_prefix + num_vecs * kBytesPerVector;
    const unsigned int tail_bytes = main_bytes & (kBytesPerVector - 1u);

    for (unsigned int t = global_thread; t < tail_bytes; t += total_threads)
    {
        const unsigned int active_mask = __activemask();
        update_warp_private_hist(
            warp_hist,
            from,
            num_bins,
            static_cast<unsigned int>(input_bytes[tail_base + t]),
            active_mask,
            lane);
    }

    __syncthreads();

    // Reduce warp-private shared histograms into the final global histogram.
    // One atomicAdd per non-zero bin per block is much cheaper than atomically updating global memory per byte.
    for (unsigned int bin = static_cast<unsigned int>(tid); bin < num_bins; bin += static_cast<unsigned int>(BLOCK_THREADS))
    {
        const int bin_i = static_cast<int>(bin);
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w)
        {
            sum += s_warp_histograms[w * bins_i + bin_i];
        }

        if (sum != 0u)
        {
            atomicAdd(histogram + bin, sum);
        }
    }
}

} // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // The problem statement guarantees a valid inclusive range, but this keeps the wrapper safe if misused.
    // Supporting from == to is harmless even though the prompt says from < to.
    if (from < 0 || to > 255 || from > to)
    {
        return;
    }

    const unsigned int num_bins = static_cast<unsigned int>(to - from + 1);

    // As requested, this wrapper is asynchronous: it does not synchronize and it does not force error handling.
    // The caller can observe launch/memset errors later via cudaGetLastError/cudaStreamSynchronize/etc.
    (void)cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(unsigned int), 0);

    if (inputSize == 0u)
    {
        return;
    }

    const size_t shared_bytes = static_cast<size_t>(kWarpsPerBlock) *
                                static_cast<size_t>(num_bins) *
                                sizeof(unsigned int);

    int device = 0;
    int sm_count = 1;
    int active_blocks_per_sm = 1;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        histogram_range_kernel<kBlockThreads>,
        kBlockThreads,
        shared_bytes);

    if (sm_count < 1)
    {
        sm_count = 1;
    }
    if (active_blocks_per_sm < 1)
    {
        active_blocks_per_sm = 1;
    }

    // Approximate work in units of 16-byte chunks, matching the main vectorized path.
    const unsigned int work_chunks =
        (inputSize / kBytesPerVector) +
        (((inputSize & (kBytesPerVector - 1u)) != 0u) ? 1u : 0u);

    const unsigned int blocks_by_work =
        (work_chunks + static_cast<unsigned int>(kBlockThreads) - 1u) /
        static_cast<unsigned int>(kBlockThreads);

    // Launch one occupancy-sized wave of CTAs at most. The kernel is grid-stride, so more CTAs would only add
    // extra block-level global histogram merges without increasing steady-state throughput.
    const unsigned int max_resident_blocks =
        static_cast<unsigned int>(sm_count) *
        static_cast<unsigned int>(active_blocks_per_sm);

    unsigned int grid = (blocks_by_work < max_resident_blocks) ? blocks_by_work : max_resident_blocks;
    if (grid == 0u)
    {
        grid = 1u;
    }

    histogram_range_kernel<kBlockThreads>
        <<<static_cast<int>(grid), kBlockThreads, shared_bytes, 0>>>(
            input,
            histogram,
            inputSize,
            from,
            num_bins);
}