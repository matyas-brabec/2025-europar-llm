#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// High-performance bit-packed Conway step for modern NVIDIA GPUs.
//
// Design choices:
// - One CUDA thread updates one 64-bit packed word (64 cells), so no atomics are needed.
// - Bit k of a word stores cell x = word_x * 64 + k, i.e. bit 0 is the leftmost cell in the word
//   and bit 63 is the rightmost cell in the word.
// - grid_dimensions is a power of two and > 512, so every row contains an integer number of
//   fully-used 64-bit words; no tail masking is required.
// - Shared/texture memory are intentionally not used. The access pattern is regular, neighboring
//   threads naturally reuse the same cache lines, and direct global loads keep the kernel simpler.
// - A 1D launch is used on purpose: because words_per_row is also a power of two, recovering
//   (row, word_x) from the linear thread index is just shift+mask, and this avoids 2D-grid limits
//   for very large boards.

namespace detail {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr int kThreadsPerBlock = 256;

// Carry-save adder for bit-sliced arithmetic.
// For every bit position independently:
//   a + b + c = sum + 2 * carry
__device__ __forceinline__ void csa(u64& carry, u64& sum, const u64 a, const u64 b, const u64 c) {
    const u64 ab_xor = a ^ b;
    sum = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// Evaluate one 64-cell word given the surrounding 3x3 word neighborhood.
//
// The "special handling" for bit 0 / bit 63 requested by the prompt is encoded by these stitched
// shifts:
//   west  = (center << 1) | (left  >> 63)
//   east  = (center >> 1) | (right << 63)
// and likewise for the rows above/below. Therefore:
// - bit 0 can see above_left / left / below_left
// - bit 63 can see above_right / right / below_right
//
// Neighbor counts are computed in bit-sliced form with carry-save adders, so all 64 cells are
// updated in parallel using integer logic only.
//
// Only the low three bits of the 8-neighbor count are needed:
// - count 2 => 0b010
// - count 3 => 0b011
// - count 8 => 0b000 modulo 8, which is harmless because Life only distinguishes 2 and 3.
__device__ __forceinline__ u64 evaluate_neighborhood(
    const u64 above_left,  const u64 above,  const u64 above_right,
    const u64 left,        const u64 center, const u64 right,
    const u64 below_left,  const u64 below,  const u64 below_right)
{
    // Top row of neighbors: NW, N, NE.
    const u64 north_west = (above << 1) | (above_left >> 63);
    const u64 north_east = (above >> 1) | (above_right << 63);
    u64 carry_top, sum_top;
    csa(carry_top, sum_top, above, north_west, north_east);

    // Bottom row of neighbors: SW, S, SE.
    const u64 south_west = (below << 1) | (below_left >> 63);
    const u64 south_east = (below >> 1) | (below_right << 63);
    u64 carry_bottom, sum_bottom;
    csa(carry_bottom, sum_bottom, below, south_west, south_east);

    // Same-row neighbors: W, E.
    const u64 west = (center << 1) | (left >> 63);
    const u64 east = (center >> 1) | (right << 63);
    const u64 sum_lr = west ^ east;
    const u64 carry_lr = west & east;

    // Combine the low bits from the three row groups.
    u64 carry_low, ones;
    csa(carry_low, ones, sum_top, sum_bottom, sum_lr);

    // Combine the carry bits from the three row groups. These are already weight-2 contributions.
    u64 carry_high, twos_partial;
    csa(carry_high, twos_partial, carry_top, carry_bottom, carry_lr);

    // Add carry_low (another weight-2 contribution) to the partial twos/fours representation.
    const u64 twos  = twos_partial ^ carry_low;
    const u64 fours = carry_high ^ (twos_partial & carry_low);

    // Conway rule:
    //   next = (count == 3) || (alive && count == 2)
    // With bit-sliced count bits (ones, twos, fours):
    //   count == 2 -> ones=0, twos=1, fours=0
    //   count == 3 -> ones=1, twos=1, fours=0
    return twos & ~fours & (ones | center);
}

// Fast path for non-boundary words: all 3x3 surrounding words exist, so loads are unconditional.
__device__ __forceinline__ u64 update_word_interior(
    const u64* __restrict__ input,
    const std::size_t idx,
    const std::size_t row_words)
{
    const u64* const above_row = input + idx - row_words - 1;
    const u64* const here_row  = input + idx - 1;
    const u64* const below_row = input + idx + row_words - 1;

    return evaluate_neighborhood(
        above_row[0], above_row[1], above_row[2],
        here_row[0],  here_row[1],  here_row[2],
        below_row[0], below_row[1], below_row[2]);
}

// Boundary path: words beyond the board are treated as zero, exactly matching
// "all cells outside the grid are dead".
__device__ __forceinline__ u64 update_word_boundary(
    const u64* __restrict__ input,
    const std::size_t idx,
    const int x,
    const int y,
    const int last_word,
    const int last_row,
    const std::size_t row_words)
{
    const bool has_left  = (x != 0);
    const bool has_right = (x != last_word);
    const bool has_up    = (y != 0);
    const bool has_down  = (y != last_row);

    const u64 center      = input[idx];
    const u64 left        = has_left               ? input[idx - 1] : 0ull;
    const u64 right       = has_right              ? input[idx + 1] : 0ull;
    const u64 above       = has_up                 ? input[idx - row_words] : 0ull;
    const u64 below       = has_down               ? input[idx + row_words] : 0ull;
    const u64 above_left  = (has_up   && has_left) ? input[idx - row_words - 1] : 0ull;
    const u64 above_right = (has_up   && has_right)? input[idx - row_words + 1] : 0ull;
    const u64 below_left  = (has_down && has_left) ? input[idx + row_words - 1] : 0ull;
    const u64 below_right = (has_down && has_right)? input[idx + row_words + 1] : 0ull;

    return evaluate_neighborhood(
        above_left, above, above_right,
        left, center, right,
        below_left, below, below_right);
}

// Template keeps the common case on 32-bit indices while still supporting very large boards.
template <typename IndexT>
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const IndexT total_words,
    const std::size_t row_words,
    const int row_shift,
    const int last_word,
    const int last_row)
{
    const IndexT linear =
        static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(kThreadsPerBlock) +
        static_cast<IndexT>(threadIdx.x);

    if (linear >= total_words) {
        return;
    }

    const IndexT word_mask = static_cast<IndexT>(last_word);
    const int x = static_cast<int>(linear & word_mask);
    const int y = static_cast<int>(linear >> row_shift);
    const std::size_t idx = static_cast<std::size_t>(linear);

    // Nearly all words are interior for large boards, so keep a fast branch-free interior path
    // and only fall back to zero-extended loads on the actual boundary.
    const bool boundary = (x == 0) || (x == last_word) || (y == 0) || (y == last_row);

    if (!boundary) {
        output[idx] = update_word_interior(input, idx, row_words);
    } else {
        output[idx] = update_word_boundary(input, idx, x, y, last_word, last_row, row_words);
    }
}

} // namespace detail

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Preconditions guaranteed by the problem statement:
    // - input and output are valid cudaMalloc device buffers
    // - the board is square
    // - grid_dimensions is a power of two and > 512
    // - input and output are distinct buffers (in-place update is intentionally unsupported)
    //
    // No synchronization or launch-error handling is done here because the caller handles
    // synchronization and the task is explicitly performance-focused.

    const int words_per_row = grid_dimensions >> 6;
    const int last_word = words_per_row - 1;
    const int last_row = grid_dimensions - 1;
    const std::size_t row_words = static_cast<std::size_t>(words_per_row);

    const std::uint64_t total_words =
        static_cast<std::uint64_t>(words_per_row) * static_cast<std::uint64_t>(grid_dimensions);

    // words_per_row is a power of two, so:
    //   word_x = linear & (words_per_row - 1)
    //   row    = linear >> row_shift
    int row_shift = 0;
    for (int tmp = words_per_row; tmp > 1; tmp >>= 1) {
        ++row_shift;
    }

    const std::uint64_t threads_per_block = static_cast<std::uint64_t>(detail::kThreadsPerBlock);
    const unsigned int blocks =
        static_cast<unsigned int>((total_words + threads_per_block - 1ull) / threads_per_block);

    // Use 32-bit indexing whenever possible; only extremely large boards need the 64-bit variant.
    if (total_words <= 0xFFFFFFFFull) {
        detail::game_of_life_kernel<detail::u32><<<blocks, detail::kThreadsPerBlock>>>(
            input,
            output,
            static_cast<detail::u32>(total_words),
            row_words,
            row_shift,
            last_word,
            last_row);
    } else {
        detail::game_of_life_kernel<detail::u64><<<blocks, detail::kThreadsPerBlock>>>(
            input,
            output,
            static_cast<detail::u64>(total_words),
            row_words,
            row_shift,
            last_word,
            last_row);
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
