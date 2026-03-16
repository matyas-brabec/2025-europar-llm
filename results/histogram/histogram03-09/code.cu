#include <cuda_runtime.h>
#include <stdint.h>

// Restricted-range text histogram optimized for modern NVIDIA data-center GPUs.
//
// High-level strategy:
//   1) Each block keeps one private histogram per warp in shared memory.
//      With 256 threads/block this is 8 warp-private histograms, so the shared-memory
//      footprint is only 8 * range_len * sizeof(unsigned int), at most 8 KiB.
//   2) Input is scanned with grid-stride loops. The main path uses 16-byte vector loads
//      (uint4) for good memory throughput. A tiny scalar prefix keeps the code correct
//      even if the caller passes a misaligned sub-buffer pointer.
//   3) Matching bytes update the warp-private shared histogram. On Volta+ we first
//      aggregate equal bins within the warp via __match_any_sync(), so one shared atomic
//      can account for multiple lanes hitting the same character.
//   4) After all input has been consumed, each block reduces its 8 private histograms
//      and emits a single global update per output bin. This cuts global atomic traffic
//      from O(number_of_matching_characters) to O(number_of_blocks * range_len).

constexpr int          kHistogramBlockThreads = 256;
constexpr unsigned int kHistogramBlockThreadsU = 256u;
constexpr int          kHistogramWarpSize = 32;
constexpr unsigned int kHistogramWarpMask = 31u;
constexpr int          kHistogramWarpsPerBlock = kHistogramBlockThreads / kHistogramWarpSize;
constexpr unsigned int kHistogramVectorBytes = 16u;
constexpr unsigned int kHistogramVectorMask = kHistogramVectorBytes - 1u;
constexpr unsigned int kHistogramVectorShift = 4u;

static_assert(kHistogramBlockThreads % kHistogramWarpSize == 0,
              "Block size must be a multiple of warp size.");
static_assert(kHistogramBlockThreads >= 256,
              "The final reduction expects at least one thread per possible output bin.");
static_assert(sizeof(uint4) == kHistogramVectorBytes,
              "This implementation assumes 16-byte uint4 vector loads.");

// Map an input byte to a histogram bin in [0, range_len) or return -1 if the byte
// is outside the requested inclusive range [from_u, from_u + range_len - 1].
__device__ __forceinline__
int restricted_bin(const unsigned char c,
                   const unsigned int from_u,
                   const unsigned int range_len)
{
    const unsigned int bin = static_cast<unsigned int>(c) - from_u;
    return (bin < range_len) ? static_cast<int>(bin) : -1;
}

// Increment one bin in the warp-private shared histogram.
// Modern path:
//   - __match_any_sync() groups lanes with identical bin values.
//   - Only one leader lane performs the shared atomicAdd with the group population.
// Fallback path:
//   - One shared atomic per matching byte.
__device__ __forceinline__
void warp_aggregated_shared_increment(const int bin,
                                      unsigned int * const warp_hist,
                                      const unsigned int lane)
{
#if __CUDA_ARCH__ >= 700
    const unsigned int active = __activemask();
    const unsigned int peers  = __match_any_sync(active, bin);
    const int leader_lane     = __ffs(static_cast<int>(peers)) - 1;

    if (bin >= 0 && static_cast<int>(lane) == leader_lane) {
        atomicAdd(warp_hist + bin, static_cast<unsigned int>(__popc(peers)));
    }
#else
    (void)lane;
    if (bin >= 0) {
        atomicAdd(warp_hist + bin, 1u);
    }
#endif
}

__device__ __forceinline__
void process_byte(const unsigned char c,
                  const unsigned int from_u,
                  const unsigned int range_len,
                  unsigned int * const warp_hist,
                  const unsigned int lane)
{
    warp_aggregated_shared_increment(restricted_bin(c, from_u, range_len), warp_hist, lane);
}

// Process four bytes packed in one 32-bit word. Byte order does not matter for the
// histogram, so simple bit extraction is sufficient.
__device__ __forceinline__
void process_u32(const unsigned int word,
                 const unsigned int from_u,
                 const unsigned int range_len,
                 unsigned int * const warp_hist,
                 const unsigned int lane)
{
    process_byte(static_cast<unsigned char>( word        & 0xFFu), from_u, range_len, warp_hist, lane);
    process_byte(static_cast<unsigned char>((word >>  8) & 0xFFu), from_u, range_len, warp_hist, lane);
    process_byte(static_cast<unsigned char>((word >> 16) & 0xFFu), from_u, range_len, warp_hist, lane);
    process_byte(static_cast<unsigned char>((word >> 24) & 0xFFu), from_u, range_len, warp_hist, lane);
}

__global__ __launch_bounds__(kHistogramBlockThreads)
void histogram_range_kernel(const char * __restrict__ input,
                            unsigned int * __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int from_u,
                            const unsigned int range_len)
{
    // Shared memory layout:
    //   [ warp0 bins ][ warp1 bins ] ... [ warp7 bins ]
    // Each slice has exactly range_len counters, so narrow ranges pay less zeroing cost.
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid     = threadIdx.x;
    const unsigned int lane    = tid & kHistogramWarpMask;
    const unsigned int warp_id = tid >> 5;

    unsigned int * const warp_hist = s_hist + warp_id * range_len;
    const unsigned int shared_bins = static_cast<unsigned int>(kHistogramWarpsPerBlock) * range_len;

    // Zero the block-private shared histograms.
    for (unsigned int i = tid; i < shared_bins; i += kHistogramBlockThreadsU) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int global_tid    = blockIdx.x * kHistogramBlockThreadsU + tid;
    const unsigned int total_threads = gridDim.x  * kHistogramBlockThreadsU;

    // cudaMalloc() base pointers are naturally well aligned, but this scalar prefix keeps
    // the vectorized path correct even if the caller passes an offset sub-buffer pointer.
    const uintptr_t address = reinterpret_cast<uintptr_t>(input);
    unsigned int scalar_prefix =
        static_cast<unsigned int>((kHistogramVectorBytes - (address & kHistogramVectorMask)) &
                                  kHistogramVectorMask);
    if (scalar_prefix > inputSize) {
        scalar_prefix = inputSize;
    }

    // Small misalignment prefix.
    for (unsigned int i = global_tid; i < scalar_prefix; i += total_threads) {
        process_byte(static_cast<unsigned char>(input[i]), from_u, range_len, warp_hist, lane);
    }

    // Main vectorized body.
    const char * const aligned_input = input + scalar_prefix;
    const unsigned int aligned_size  = inputSize - scalar_prefix;
    const uint4 * const vec_input    = reinterpret_cast<const uint4 *>(aligned_input);
    const unsigned int vec_count     = aligned_size >> kHistogramVectorShift;

    for (unsigned int vec_idx = global_tid; vec_idx < vec_count; vec_idx += total_threads) {
        const uint4 v = vec_input[vec_idx];
        process_u32(v.x, from_u, range_len, warp_hist, lane);
        process_u32(v.y, from_u, range_len, warp_hist, lane);
        process_u32(v.z, from_u, range_len, warp_hist, lane);
        process_u32(v.w, from_u, range_len, warp_hist, lane);
    }

    // Tail after the vectorized body.
    const unsigned int tail_start = vec_count << kHistogramVectorShift;
    for (unsigned int i = tail_start + global_tid; i < aligned_size; i += total_threads) {
        process_byte(static_cast<unsigned char>(aligned_input[i]), from_u, range_len, warp_hist, lane);
    }

    __syncthreads();

    // One thread per output bin (range_len <= 256 by problem statement).
    // Sum the 8 warp-private counters and emit one global update per block/bin.
    if (tid < range_len) {
        const unsigned int histogram_bin = tid;
        const unsigned int * const bin_base = s_hist + histogram_bin;

        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < kHistogramWarpsPerBlock; ++w) {
            sum += bin_base[static_cast<unsigned int>(w) * range_len];
        }

        if (sum != 0u) {
            // Fast path for tiny launches: if there is only one block, the output bin is
            // written directly because the histogram was already zeroed before launch.
            if (gridDim.x == 1u) {
                histogram[histogram_bin] = sum;
            } else {
                atomicAdd(histogram + histogram_bin, sum);
            }
        }
    }
}

// Host entry point requested by the problem statement.
//
// Notes:
//   - input and histogram are assumed to be device pointers allocated with cudaMalloc().
//   - from and to are assumed valid per the problem statement.
//   - The function uses the default stream because the signature does not provide one.
//   - The caller handles synchronization.
//   - The histogram is explicitly zeroed here because the kernel performs accumulation.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const unsigned int from_u    = static_cast<unsigned int>(from);
    const unsigned int range_len = static_cast<unsigned int>(to - from + 1);
    const size_t histogram_bytes = static_cast<size_t>(range_len) * sizeof(unsigned int);

    (void)cudaMemsetAsync(histogram, 0, histogram_bytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t dynamic_smem_bytes =
        static_cast<size_t>(kHistogramWarpsPerBlock) *
        static_cast<size_t>(range_len) *
        sizeof(unsigned int);

    // Grid-stride loops mean we only need enough blocks to fill the machine once.
    // Occupancy is queried so launch size adapts if compiler register usage changes.
    int device = 0;
    (void)cudaGetDevice(&device);

    int sm_count = 0;
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    int blocks_per_sm = 0;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        histogram_range_kernel,
        kHistogramBlockThreads,
        dynamic_smem_bytes);
    if (blocks_per_sm < 1) {
        blocks_per_sm = 1;
    }

    const int max_resident_blocks = sm_count * blocks_per_sm;

    // Rough work estimate in 16-byte chunks; enough to avoid launching far more blocks
    // than the input can use while still saturating the GPU for large inputs.
    const unsigned int work_items =
        (inputSize + kHistogramVectorBytes - 1u) / kHistogramVectorBytes;

    int blocks_for_work = static_cast<int>(
        (work_items + kHistogramBlockThreadsU - 1u) / kHistogramBlockThreadsU);
    if (blocks_for_work < 1) {
        blocks_for_work = 1;
    }

    int grid = blocks_for_work;
    if (grid > max_resident_blocks) {
        grid = max_resident_blocks;
    }
    if (grid < 1) {
        grid = 1;
    }

    histogram_range_kernel<<<grid, kHistogramBlockThreads, dynamic_smem_bytes>>>(
        input,
        histogram,
        inputSize,
        from_u,
        range_len);
}