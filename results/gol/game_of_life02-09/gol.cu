#include <cstdint>
#include <cuda_runtime.h>

// One Conway step directly on the bit-packed device representation.
//
// Layout / bit order:
// - The grid is row-major in 64-bit words.
// - Bit i of each word stores cell x = word_base + i, i.e. LSB -> increasing x.
//
// Performance-oriented design for A100/H100-class GPUs:
// - One lane updates one 64-bit word = 64 cells at once.
// - Horizontal neighbor words come from warp shuffles, so only subgroup-boundary
//   lanes issue extra global loads.
// - No shared memory: for this access pattern, shuffles + hardware caches are
//   sufficient and simpler.
// - Neighbor counts are formed with bit-sliced boolean arithmetic. Only counts
//   2 and 3 matter for Life, so keeping the count modulo 8 is enough; the only
//   wrapped case is count 8 -> 0, which is harmless for the rule.
// - The 1024x1024 case (16 words/row) is specialized so each hardware warp
//   updates two rows via two independent 16-lane subgroups.

namespace {

using u64 = std::uint64_t;

constexpr int kWarpSize               = 32;
constexpr int kHalfWarpSize           = 16;
constexpr int kWarpsPerBlock          = 8;
constexpr int kBlockThreads           = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kWarpWordShift = 5u;  // log2(32 words)
constexpr unsigned int kHalfRowShift  = 4u;  // log2(16 words)
constexpr u64 kHalfWordsPerRow        = 1ull << kHalfRowShift;

// Majority of three 64-bit bitboards. Written in a form NVCC lowers well.
__device__ __forceinline__ u64 majority3(u64 a, u64 b, u64 c) {
    return (a & b) | (c & (a | b));
}

// Load the current word and its left/right neighbor words from the same row.
// The subgroup width is either 32 (full warp) or 16 (half warp specialization).
// __activemask() is intentionally captured here so the same helper remains
// correct inside the half-warp kernel's top/bottom boundary-divergent branches.
template <int GROUP_WIDTH>
__device__ __forceinline__ void load_triplet(
    const u64* __restrict__ input,
    u64 word_index,
    u64 x_word,
    u64 words_per_row,
    unsigned int lane_local,
    u64& left,
    u64& center,
    u64& right)
{
    const unsigned int mask = __activemask();

    center = input[word_index];

    left = __shfl_up_sync(mask, center, 1, GROUP_WIDTH);
    if (lane_local == 0) {
        left = (x_word != 0) ? input[word_index - 1] : 0ull;
    }

    right = __shfl_down_sync(mask, center, 1, GROUP_WIDTH);
    if (lane_local == static_cast<unsigned int>(GROUP_WIDTH - 1)) {
        right = (x_word + 1 < words_per_row) ? input[word_index + 1] : 0ull;
    }
}

// Encode a neighboring row's 3-cell horizontal stencil (west, center, east)
// into two bitplanes: lo + 2*hi, i.e. the population count 0..3 for each bit.
template <int GROUP_WIDTH>
__device__ __forceinline__ void encode_three_cell_row(
    const u64* __restrict__ input,
    u64 word_index,
    u64 x_word,
    u64 words_per_row,
    unsigned int lane_local,
    u64& lo,
    u64& hi)
{
    u64 left, center, right;
    load_triplet<GROUP_WIDTH>(input, word_index, x_word, words_per_row, lane_local,
                              left, center, right);

    const u64 west = (center << 1) | (left >> 63);
    const u64 east = (center >> 1) | (right << 63);

    lo = west ^ center ^ east;
    hi = majority3(west, center, east);
}

// Encode the current row's horizontal neighbors (west/east only; the center
// cell itself is excluded from the neighbor count) into two bitplanes.
template <int GROUP_WIDTH>
__device__ __forceinline__ void encode_two_cell_row(
    const u64* __restrict__ input,
    u64 word_index,
    u64 x_word,
    u64 words_per_row,
    unsigned int lane_local,
    u64& self_word,
    u64& lo,
    u64& hi)
{
    u64 left, center, right;
    load_triplet<GROUP_WIDTH>(input, word_index, x_word, words_per_row, lane_local,
                              left, center, right);

    self_word = center;

    const u64 west = (center << 1) | (left >> 63);
    const u64 east = (center >> 1) | (right << 63);

    lo = west ^ east;
    hi = west & east;
}

// Combine:
//   north_count = n0 + 2*n1   (0..3)
//   mid_count   = m0 + 2*m1   (0..2)
//   south_count = s0 + 2*s1   (0..3)
// into the 3 low bits of the total neighbor count.
//
// Count 8 wraps to 0 modulo 8; that is fine because the Life rule only accepts
// counts 2 and 3. The final boolean is:
//   next = (count == 3) | (self & (count == 2))
//        = cnt1 & ~cnt2 & (cnt0 | self)
__device__ __forceinline__ u64 finalize_word(
    u64 self_word,
    u64 n0, u64 n1,
    u64 m0, u64 m1,
    u64 s0, u64 s1)
{
    const u64 cnt0 = n0 ^ m0 ^ s0;
    const u64 carry0 = majority3(n0, m0, s0);

    const u64 pair01_lo = n1 ^ m1;
    const u64 pair01_hi = n1 & m1;
    const u64 pair23_lo = s1 ^ carry0;
    const u64 pair23_hi = s1 & carry0;

    const u64 cnt1 = pair01_lo ^ pair23_lo;
    const u64 cnt2 = pair01_hi ^ pair23_hi ^ (pair01_lo & pair23_lo);

    return (cnt1 & ~cnt2) & (cnt0 | self_word);
}

// Full-warp kernel for rows with at least 32 packed words (grid >= 2048).
// Task space is "one 32-word tile of one row per hardware warp". Because the
// row length is a power of two, row index and x-within-row are recovered with
// shifts/masks instead of division/modulo.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel_full(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    u64 rows,
    unsigned int words_per_row_shift)
{
    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;

    const u64 words_per_row = 1ull << words_per_row_shift;
    const u64 row_mask      = words_per_row - 1ull;
    const unsigned int task_shift = words_per_row_shift - kWarpWordShift;
    const u64 total_tasks = rows << task_shift;

    const u64 warp_id     = static_cast<u64>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const u64 warp_stride = static_cast<u64>(gridDim.x) * kWarpsPerBlock;

    for (u64 task = warp_id; task < total_tasks; task += warp_stride) {
        const u64 word_index = (task << kWarpWordShift) | static_cast<u64>(lane);
        const u64 y          = task >> task_shift;
        const u64 x_word     = word_index & row_mask;

        u64 n0 = 0ull, n1 = 0ull;
        if (y != 0) {
            encode_three_cell_row<kWarpSize>(
                input, word_index - words_per_row, x_word, words_per_row, lane, n0, n1);
        }

        u64 self_word, m0, m1;
        encode_two_cell_row<kWarpSize>(
            input, word_index, x_word, words_per_row, lane, self_word, m0, m1);

        u64 s0 = 0ull, s1 = 0ull;
        if (y + 1 < rows) {
            encode_three_cell_row<kWarpSize>(
                input, word_index + words_per_row, x_word, words_per_row, lane, s0, s1);
        }

        output[word_index] = finalize_word(self_word, n0, n1, m0, m1, s0, s1);
    }
}

// Specialized kernel for exactly 16 packed words per row (1024 cells/row).
// Each hardware warp updates two rows at once using two 16-lane subgroups.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel_half(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    u64 rows)
{
    const unsigned int lane       = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const unsigned int subwarp    = lane >> kHalfRowShift;            // 0 or 1
    const unsigned int lane16     = lane & (kHalfWarpSize - 1);      // 0..15
    const u64 x_word              = static_cast<u64>(lane16);

    const u64 pair_count  = rows >> 1; // rows is guaranteed even
    const u64 pair_id     = static_cast<u64>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const u64 pair_stride = static_cast<u64>(gridDim.x) * kWarpsPerBlock;

    for (u64 pair = pair_id; pair < pair_count; pair += pair_stride) {
        const u64 y = (pair << 1) | static_cast<u64>(subwarp);

        // One pair contains two rows * 16 words = 32 words total.
        const u64 word_index =
            (pair << kWarpWordShift) |
            (static_cast<u64>(subwarp) << kHalfRowShift) |
            x_word;

        u64 n0 = 0ull, n1 = 0ull;
        if (y != 0) {
            encode_three_cell_row<kHalfWarpSize>(
                input, word_index - kHalfWordsPerRow, x_word, kHalfWordsPerRow, lane16, n0, n1);
        }

        u64 self_word, m0, m1;
        encode_two_cell_row<kHalfWarpSize>(
            input, word_index, x_word, kHalfWordsPerRow, lane16, self_word, m0, m1);

        u64 s0 = 0ull, s1 = 0ull;
        if (y + 1 < rows) {
            encode_three_cell_row<kHalfWarpSize>(
                input, word_index + kHalfWordsPerRow, x_word, kHalfWordsPerRow, lane16, s0, s1);
        }

        output[word_index] = finalize_word(self_word, n0, n1, m0, m1, s0, s1);
    }
}

inline u64 ceil_div_u64(u64 n, u64 d) {
    return (n + d - 1ull) / d;
}

inline unsigned int log2_pow2_u64(u64 x) {
    unsigned int s = 0;
    while (x > 1ull) {
        x >>= 1;
        ++s;
    }
    return s;
}

// For grid-stride kernels, launching roughly one fully resident wave of blocks
// avoids underfilling the GPU without overscheduling far beyond residency.
inline unsigned int choose_grid_blocks(u64 task_count, int sm_count, int blocks_per_sm) {
    const u64 blocks_needed = ceil_div_u64(task_count, static_cast<u64>(kWarpsPerBlock));
    const u64 resident_blocks =
        static_cast<u64>(sm_count > 0 ? sm_count : 1) *
        static_cast<u64>(blocks_per_sm > 0 ? blocks_per_sm : 1);

    u64 blocks = (blocks_needed < resident_blocks) ? blocks_needed : resident_blocks;
    if (blocks == 0) blocks = 1;
    return static_cast<unsigned int>(blocks);
}

// Cached per host thread / current device. These queries are host-side only,
// and caching keeps repeated time-step launches cheap.
struct LaunchConfigCache {
    int device = -1;
    int sm_count = 0;
    int full_blocks_per_sm = 0;
    int half_blocks_per_sm = 0;
};

inline const LaunchConfigCache& get_launch_config_cache() {
    thread_local LaunchConfigCache cache;

    int device = 0;
    cudaGetDevice(&device);

    if (device != cache.device) {
        cache.device = device;

        cudaDeviceGetAttribute(&cache.sm_count, cudaDevAttrMultiProcessorCount, device);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cache.full_blocks_per_sm,
            game_of_life_kernel_full,
            kBlockThreads,
            0);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cache.half_blocks_per_sm,
            game_of_life_kernel_half,
            kBlockThreads,
            0);

        if (cache.sm_count < 1) cache.sm_count = 1;
        if (cache.full_blocks_per_sm < 1) cache.full_blocks_per_sm = 1;
        if (cache.half_blocks_per_sm < 1) cache.half_blocks_per_sm = 1;
    }

    return cache;
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Hot-path launcher: no synchronization and no error checking here by design.
    // The caller owns synchronization/error semantics and must provide distinct
    // input/output device buffers.

    const u64 rows = static_cast<u64>(grid_dimensions);
    const u64 words_per_row = rows >> 6; // 64 cells per 64-bit word

    // words_per_row is also a power of two because grid_dimensions is.
    const unsigned int words_per_row_shift = log2_pow2_u64(words_per_row);

    const auto& cache = get_launch_config_cache();
    const dim3 block(kWarpSize, kWarpsPerBlock, 1);

    if (words_per_row == kHalfWordsPerRow) {
        const u64 pair_count = rows >> 1;
        const unsigned int grid_blocks =
            choose_grid_blocks(pair_count, cache.sm_count, cache.half_blocks_per_sm);

        game_of_life_kernel_half<<<grid_blocks, block>>>(input, output, rows);
    } else {
        const u64 task_count = rows << (words_per_row_shift - kWarpWordShift);
        const unsigned int grid_blocks =
            choose_grid_blocks(task_count, cache.sm_count, cache.full_blocks_per_sm);

        game_of_life_kernel_full<<<grid_blocks, block>>>(input, output, rows, words_per_row_shift);
    }
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// This label is used by the framework to identify the memory layout of the input and output arrays.
// MEMORY_LAYOUT: ROWS
