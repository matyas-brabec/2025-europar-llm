#include <cuda_runtime.h>

namespace {

// Range-limited byte histogram tuned for modern data-center GPUs:
//
// - 256-thread blocks => 8 warps/block.
// - Each warp owns a private 256-bin histogram in shared memory.
//   That keeps inter-warp contention at zero while staying tiny:
//   8 warps * 256 bins * 4 bytes = 8 KiB/block.
// - The main path processes input in 512-byte warp tiles:
//   each lane loads 16 bytes (one uint4), so all 32 lanes stay converged.
// - Within a converged warp, __match_any_sync() collapses duplicate byte values;
//   only one lane per distinct value updates shared memory, and those updates
//   are plain increments (no atomics needed).
// - The leftover tail is < 512 bytes and may diverge a warp. For that small
//   region we deliberately use shared-memory atomics for correctness under
//   independent thread scheduling.
// - Final block reduction emits one global atomic per nonzero bin (or plain
//   stores when the launch uses a single block).

constexpr unsigned int WARP_THREADS             = 32u;
constexpr unsigned int BLOCK_THREADS            = 256u;
constexpr unsigned int WARPS_PER_BLOCK          = BLOCK_THREADS / WARP_THREADS;
constexpr unsigned int MAX_BINS                 = 256u;
constexpr unsigned int BYTES_PER_LANE           = sizeof(uint4);                 // 16
constexpr unsigned int BYTES_PER_WARP           = WARP_THREADS * BYTES_PER_LANE; // 512
constexpr unsigned int BYTES_PER_BLOCK          = BLOCK_THREADS * BYTES_PER_LANE;// 4096
constexpr unsigned int FULL_MASK                = 0xFFFFFFFFu;
constexpr unsigned int INVALID_BIN              = 0xFFFFFFFFu;
constexpr unsigned int FALLBACK_BLOCKS_PER_SM   = 8u;
constexpr unsigned int FALLBACK_MAX_GRID_BLOCKS = 1024u;

static_assert(BLOCK_THREADS % WARP_THREADS == 0u, "Block size must be a whole number of warps.");
static_assert(MAX_BINS == 256u, "This kernel assumes byte-valued input.");

__device__ __forceinline__
void update_byte_fullwarp(unsigned int* __restrict__ local_hist,
                          unsigned int lane,
                          unsigned int byte_value,
                          unsigned int from,
                          unsigned int num_bins)
{
    const unsigned int bin = byte_value - from;
    const unsigned int key = (bin < num_bins) ? bin : INVALID_BIN;

    // Full-warp call: every lane participates, so FULL_MASK is valid here.
    const unsigned int peers = __match_any_sync(FULL_MASK, key);

    // One leader per distinct key performs the increment by the group size.
    // INVALID_BIN is ignored, so out-of-range bytes do not touch the histogram.
    if ((key != INVALID_BIN) && (lane == static_cast<unsigned int>(__ffs(peers) - 1))) {
        local_hist[key] += __popc(peers);
    }
}

__device__ __forceinline__
void process_word_fullwarp(unsigned int* __restrict__ local_hist,
                           unsigned int lane,
                           unsigned int word,
                           unsigned int from,
                           unsigned int num_bins)
{
    update_byte_fullwarp(local_hist, lane,  word        & 0xFFu, from, num_bins);
    update_byte_fullwarp(local_hist, lane, (word >>  8) & 0xFFu, from, num_bins);
    update_byte_fullwarp(local_hist, lane, (word >> 16) & 0xFFu, from, num_bins);
    update_byte_fullwarp(local_hist, lane, (word >> 24) & 0xFFu, from, num_bins);
}

__global__ __launch_bounds__(BLOCK_THREADS)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int num_bins)
{
    __shared__ unsigned int warp_hist[WARPS_PER_BLOCK][MAX_BINS];

    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & (WARP_THREADS - 1u);

    unsigned int* const local_hist = warp_hist[warp];

    // Only bins [0, num_bins) are ever touched/read for this launch, so zero only those.
    #pragma unroll
    for (unsigned int bin = lane; bin < num_bins; bin += WARP_THREADS) {
        local_hist[bin] = 0u;
    }

    // Warp-private histogram slices are independent, so a warp-level sync is sufficient here.
    __syncwarp();

    // Treat input chars as raw bytes in [0,255].
    const unsigned char* const input_bytes = reinterpret_cast<const unsigned char*>(input);

    // Fast path over full 512-byte warp tiles.
    // Because the prompt states the buffer itself is cudaMalloc'ed, the base pointer is
    // sufficiently aligned for 16-byte vector loads, and tile starts are multiples of 512.
    const size_t vectorized_end = static_cast<size_t>(inputSize) &
                                  ~(static_cast<size_t>(BYTES_PER_WARP) - 1);

    const size_t warp_index  = static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const size_t warp_stride = static_cast<size_t>(gridDim.x) * WARPS_PER_BLOCK * BYTES_PER_WARP;

    for (size_t base = warp_index * BYTES_PER_WARP; base < vectorized_end; base += warp_stride) {
        const uint4* const vec_ptr = reinterpret_cast<const uint4*>(input_bytes + base);
        const uint4 v = vec_ptr[lane];

        process_word_fullwarp(local_hist, lane, v.x, from, num_bins);
        process_word_fullwarp(local_hist, lane, v.y, from, num_bins);
        process_word_fullwarp(local_hist, lane, v.z, from, num_bins);
        process_word_fullwarp(local_hist, lane, v.w, from, num_bins);
    }

    // Tail path: < 512 bytes remain. This loop can diverge within a warp, so use shared-memory
    // atomics for correctness; the tail is tiny enough that the extra cost is negligible.
    const size_t global_thread = static_cast<size_t>(blockIdx.x) * BLOCK_THREADS + tid;
    const size_t thread_stride = static_cast<size_t>(gridDim.x) * BLOCK_THREADS;

    for (size_t i = vectorized_end + global_thread; i < static_cast<size_t>(inputSize); i += thread_stride) {
        const unsigned int bin = static_cast<unsigned int>(input_bytes[i]) - from;
        if (bin < num_bins) {
            atomicAdd(local_hist + bin, 1u);
        }
    }

    __syncthreads();

    // Reduce the per-warp histograms into the final output.
    if (tid < num_bins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (unsigned int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += warp_hist[w][tid];
        }

        // If only one block was launched, no global atomic is needed.
        if (gridDim.x == 1) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
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
    // The output buffer is specified as the result histogram, not as an accumulator,
    // so clear it here. This is asynchronous; the caller owns synchronization.
    const unsigned int num_bins = static_cast<unsigned int>(to - from + 1);
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    // Cache launch heuristics per host thread and per current device.
    // The kernel shape is fixed, so occupancy is stable for a given device/binary.
    thread_local int cached_device        = -1;
    thread_local int cached_sms           = 0;
    thread_local int cached_active_blocks = 0;

    int device = 0;
    cudaGetDevice(&device);

    if (device != cached_device) {
        cached_device        = device;
        cached_sms           = 0;
        cached_active_blocks = 0;
    }

    if (cached_sms <= 0) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        if (sms > 0) {
            cached_sms = sms;
        }
    }

    if (cached_active_blocks <= 0) {
        int active_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            histogram_range_kernel,
            static_cast<int>(BLOCK_THREADS),
            0);
        if (active_blocks > 0) {
            cached_active_blocks = active_blocks;
        }
    }

    // 4 KiB/block is the natural work quantum of the vectorized path.
    const unsigned int blocks_needed = 1u + (inputSize - 1u) / BYTES_PER_BLOCK;

    unsigned int max_grid_blocks = FALLBACK_MAX_GRID_BLOCKS;
    if (cached_sms > 0) {
        const unsigned int blocks_per_sm =
            (cached_active_blocks > 0)
                ? static_cast<unsigned int>(cached_active_blocks)
                : FALLBACK_BLOCKS_PER_SM;

        max_grid_blocks = static_cast<unsigned int>(cached_sms) * blocks_per_sm;
    }

    const unsigned int grid_blocks =
        (blocks_needed < max_grid_blocks) ? blocks_needed : max_grid_blocks;

    histogram_range_kernel<<<grid_blocks, BLOCK_THREADS>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        num_bins);

    // Intentionally no synchronization and no error consumption here:
    // the caller is responsible for synchronization and optional error handling.
}