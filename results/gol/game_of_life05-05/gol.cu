#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Optimized Conway's Game of Life for a bit-packed square grid.
//
// Key choices made for throughput on modern data-center GPUs:
// - The kernel operates directly on the packed representation; no unpacking pass.
// - One thread updates one 64-bit word, i.e. 64 cells, so writes are naturally race-free.
// - Shared memory is intentionally avoided. Left/right word exchange is done with warp
//   shuffles, which is cheaper than staging 3x3 neighborhoods in shared memory here.
// - Neighbor counts are accumulated bit-slice style with carry-save adders, so all 64
//   cells in a word are updated simultaneously with integer logic.

namespace detail {

using u64 = std::uint64_t;
using usize = std::size_t;

static_assert(sizeof(u64) == 8, "This implementation requires 64-bit words.");

constexpr int kCellsPerWordShift = 6;  // 2^6 = 64 cells per packed word.
constexpr int kBlockThreads = 256;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

// Host-side helper. The input is guaranteed to be a power of two.
inline int log2_pow2(int value) noexcept {
    int shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

// Bit 0 is treated as the leftmost cell in the 64-cell span, matching the problem
// statement: bit 0 needs words to the left, bit 63 needs words to the right.
__device__ __forceinline__ u64 neighbor_from_left(u64 word, u64 left_word) {
    return (word << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 neighbor_from_right(u64 word, u64 right_word) {
    return (word >> 1) | (right_word << 63);
}

// Carry-save add of three 1-bit planes across all 64 bit positions in parallel.
// Written in canonical XOR/AND/OR form so NVCC can lower it efficiently on modern GPUs.
__device__ __forceinline__ void csa(u64& carry, u64& sum, u64 a, u64 b, u64 c) {
    const u64 ab_xor = a ^ b;
    sum = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// Sum the three neighbors contributed by a row above or below:
//   upper-left / upper / upper-right   or   lower-left / lower / lower-right
// The result is returned as two bit-planes: ones and twos.
__device__ __forceinline__ void row_sum3(
    u64 row_center_word,
    u64 row_left_word,
    u64 row_right_word,
    u64& ones,
    u64& twos) {
    const u64 left = neighbor_from_left(row_center_word, row_left_word);
    const u64 right = neighbor_from_right(row_center_word, row_right_word);
    csa(twos, ones, left, row_center_word, right);
}

// Sum the two horizontal neighbors in the current row. The center cell itself is
// intentionally excluded because Conway counts only neighbors.
__device__ __forceinline__ void row_sum2(
    u64 row_center_word,
    u64 row_left_word,
    u64 row_right_word,
    u64& ones,
    u64& twos) {
    const u64 left = neighbor_from_left(row_center_word, row_left_word);
    const u64 right = neighbor_from_right(row_center_word, row_right_word);
    ones = left ^ right;
    twos = left & right;
}

template <int SUBWARP_SHIFT>
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int words_shift,
    int grid_dim) {
    constexpr int SUBWARP = 1 << SUBWARP_SHIFT;
    constexpr int GROUPS_PER_BLOCK = kBlockThreads / SUBWARP;

    const int lane = threadIdx.x & (SUBWARP - 1);
    const int subgroup = threadIdx.x >> SUBWARP_SHIFT;

    // The launch is linearized over contiguous row-segments instead of using grid.y.
    // This avoids the 65535 grid.y limit for very large boards while preserving
    // contiguous memory access within each subgroup.
    const usize segment =
        static_cast<usize>(blockIdx.x) * GROUPS_PER_BLOCK + static_cast<usize>(subgroup);

    const int words_per_row = 1 << words_shift;
    const int segment_row_shift = words_shift - SUBWARP_SHIFT;
    const usize segment_row_mask = static_cast<usize>((words_per_row >> SUBWARP_SHIFT) - 1);

    const usize y = segment >> segment_row_shift;
    const int x = static_cast<int>((segment & segment_row_mask) << SUBWARP_SHIFT) | lane;

    const int last_word = words_per_row - 1;
    const usize last_row = static_cast<usize>(grid_dim - 1);

    const bool has_left = (x != 0);
    const bool has_right = (x != last_word);
    const bool has_up = (y != 0);
    const bool has_down = (y != last_row);

    const usize row_base = y << words_shift;
    const usize idx = row_base | static_cast<usize>(x);
    const u64* p = input + idx;

    // Common-case traffic is just three coalesced loads per thread: above/current/below.
    // Interior left/right words are supplied via shuffles; only subgroup boundary lanes
    // perform extra global loads to cross subgroup boundaries or handle true edges.
    const u64 center = p[0];
    const u64 above = has_up ? p[-words_per_row] : 0ULL;
    const u64 below = has_down ? p[words_per_row] : 0ULL;

    // Exact power-of-two launch geometry means every subgroup is full, so a constant
    // full-warp mask is valid here. width=16/32 partitions the warp into subgroups.
    u64 above_left = __shfl_up_sync(kFullWarpMask, above, 1, SUBWARP);
    u64 above_right = __shfl_down_sync(kFullWarpMask, above, 1, SUBWARP);
    u64 center_left = __shfl_up_sync(kFullWarpMask, center, 1, SUBWARP);
    u64 center_right = __shfl_down_sync(kFullWarpMask, center, 1, SUBWARP);
    u64 below_left = __shfl_up_sync(kFullWarpMask, below, 1, SUBWARP);
    u64 below_right = __shfl_down_sync(kFullWarpMask, below, 1, SUBWARP);

    // Handle the special cases called out in the prompt:
    // - bit 0 needs the three words to the left (above-left / left / below-left)
    // - bit 63 needs the three words to the right (above-right / right / below-right)
    if (lane == 0) {
        above_left = (has_up && has_left) ? p[-words_per_row - 1] : 0ULL;
        center_left = has_left ? p[-1] : 0ULL;
        below_left = (has_down && has_left) ? p[words_per_row - 1] : 0ULL;
    }

    if (lane == SUBWARP - 1) {
        above_right = (has_up && has_right) ? p[-words_per_row + 1] : 0ULL;
        center_right = has_right ? p[1] : 0ULL;
        below_right = (has_down && has_right) ? p[words_per_row + 1] : 0ULL;
    }

    u64 top_ones, top_twos;
    row_sum3(above, above_left, above_right, top_ones, top_twos);

    u64 mid_ones, mid_twos;
    row_sum2(center, center_left, center_right, mid_ones, mid_twos);

    u64 bot_ones, bot_twos;
    row_sum3(below, below_left, below_right, bot_ones, bot_twos);

    // Add the three row-contribution bit-planes.
    //
    // After these two CSA stages:
    // - count_bit0 is the low bit of the 8-neighbor count.
    // - count_bit1 is 1 for counts 2,3,6,7.
    // - carry_to_fours is 1 for counts 5,6,7,8.
    //
    // Therefore:
    //   count_bit1 & ~carry_to_fours   <=>   neighbor count is exactly 2 or 3.
    u64 carry_to_twos, count_bit0;
    csa(carry_to_twos, count_bit0, top_ones, mid_ones, bot_ones);

    u64 carry_to_fours, twos_sum_lsb;
    csa(carry_to_fours, twos_sum_lsb, top_twos, mid_twos, bot_twos);

    const u64 count_bit1 = twos_sum_lsb ^ carry_to_twos;
    const u64 two_or_three = count_bit1 & ~carry_to_fours;

    // Game of Life rule:
    // - count == 3: born regardless of current state  -> count_bit0 == 1
    // - count == 2: survive only if already alive     -> count_bit0 == 0
    output[idx] = two_or_three & (count_bit0 | center);
}

}  // namespace detail

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is guaranteed to be a power of two, so words_per_row and the
    // number of row-segments are also powers of two. That lets the kernel recover
    // x/y entirely with shifts and masks and lets the launch cover the board exactly
    // with no tail handling.
    const int words_shift = detail::log2_pow2(grid_dimensions) - detail::kCellsPerWordShift;
    const dim3 block(detail::kBlockThreads);

    // Prefer full 32-lane subgroups whenever a row has at least 32 words.
    // The only smaller valid case under the stated constraints is 16 words/row
    // (a 1024x1024 grid), where 16-lane subgroups avoid wasting half a warp.
    if (words_shift >= 5) {
        constexpr int kSubwarpShift = 5;
        constexpr int kGroupsPerBlock = detail::kBlockThreads >> kSubwarpShift;

        const std::size_t total_segments =
            static_cast<std::size_t>(grid_dimensions) << (words_shift - kSubwarpShift);
        const dim3 grid(static_cast<unsigned int>(total_segments / kGroupsPerBlock));

        // As required, synchronization is left entirely to the caller.
        detail::game_of_life_kernel<kSubwarpShift><<<grid, block>>>(
            input, output, words_shift, grid_dimensions);
    } else {
        constexpr int kSubwarpShift = 4;
        constexpr int kGroupsPerBlock = detail::kBlockThreads >> kSubwarpShift;

        const std::size_t total_segments =
            static_cast<std::size_t>(grid_dimensions) << (words_shift - kSubwarpShift);
        const dim3 grid(static_cast<unsigned int>(total_segments / kGroupsPerBlock));

        detail::game_of_life_kernel<kSubwarpShift><<<grid, block>>>(
            input, output, words_shift, grid_dimensions);
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
