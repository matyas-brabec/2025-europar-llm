#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// High-throughput one-step Conway's Game of Life for a square, bit-packed grid.
//
// Design choices driven purely by throughput:
//   * one thread updates one 64-bit word (64 cells), so no atomics are needed.
//   * the 64 cells in a word are updated simultaneously with bit-sliced boolean arithmetic;
//     there is no per-cell loop and no per-cell popcount.
//   * shared/texture memory are intentionally avoided. Horizontal reuse is captured with
//     warp shuffles, while the remaining vertical loads are cache-friendly on modern
//     datacenter GPUs.
//
// Outside-the-grid cells are dead, so missing rows/words are injected as zeros.

namespace
{
using u64 = unsigned long long;

constexpr int kThreadsPerBlock = 256;
constexpr int kCellsPerWordLog2 = 6;  // 64 packed cells per 64-bit word.
constexpr int kWordEdgeBit = 63;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "Expected 64-bit words.");
static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0, "Block size must be a power of two.");
static_assert((kThreadsPerBlock % 32) == 0, "Block size must be warp aligned.");

__device__ __forceinline__ u64 load_word(const std::uint64_t* __restrict__ words, std::size_t idx)
{
    return static_cast<u64>(words[idx]);
}

__device__ __forceinline__ u64 align_west(const u64 center_word, const u64 left_word)
{
    // Bit i receives the state of cell i-1.
    // The only cross-word case is bit 0, which pulls bit 63 from the word to the left.
    return (center_word << 1) | (left_word >> kWordEdgeBit);
}

__device__ __forceinline__ u64 align_east(const u64 center_word, const u64 right_word)
{
    // Bit i receives the state of cell i+1.
    // The only cross-word case is bit 63, which pulls bit 0 from the word to the right.
    return (center_word >> 1) | (right_word << kWordEdgeBit);
}

__device__ __forceinline__ void csa3(const u64 a, const u64 b, const u64 c, u64& carry, u64& sum)
{
    // Carry-save adder on three 1-bit planes:
    //   a + b + c = sum + 2*carry
    // performed independently for all 64 bit positions.
    const u64 ab_xor = a ^ b;
    sum = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

__device__ __forceinline__ void sum_three_adjacent(
    const u64 left_word,
    const u64 center_word,
    const u64 right_word,
    u64& high_bit,
    u64& low_bit)
{
    // Counts the three horizontally adjacent cells for every bit position in one row:
    //   [x-1, x, x+1] -> low_bit + 2*high_bit
    const u64 west = align_west(center_word, left_word);
    const u64 east = align_east(center_word, right_word);
    csa3(west, center_word, east, high_bit, low_bit);
}

__device__ __forceinline__ u64 one_of_four(const u64 a, const u64 b, const u64 c, const u64 d)
{
    // Exact-one detector for four 1-bit planes.
    // Used for H = carry0 + top_hi + mid_hi + bot_hi.
    const u64 ab_xor = a ^ b;
    const u64 ab_carry = a & b;
    const u64 cd_xor = c ^ d;
    const u64 cd_carry = c & d;
    return (ab_xor ^ cd_xor) & ~(ab_carry | cd_carry | (ab_xor & cd_xor));
}

__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    const std::size_t words_per_row,
    const std::size_t word_mask,
    const std::size_t last_row_base)
{
    // One thread updates one packed 64-cell word.
    //
    // Horizontal neighbor words are obtained from adjacent lanes via warp shuffles whenever
    // they live inside the same warp. Only warp-boundary lanes (0 and 31) need a global-load
    // fallback, and only when the neighboring word is still inside the same row.
    //
    // The problem constraints imply:
    //   grid_dimensions = 2^k, k >= 10
    //   words_per_row   = grid_dimensions / 64 = 2^(k-6)
    //   total_words     = grid_dimensions * words_per_row = 2^(2k-6)
    //
    // So total_words is a power of two and at least 2^14. With a power-of-two block size
    // (256), the launch covers the domain exactly: no tail threads and every warp is full.
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kThreadsPerBlock) +
        static_cast<std::size_t>(threadIdx.x);

    const std::size_t x = idx & word_mask;
    const bool has_left = x != 0;
    const bool has_right = x != word_mask;
    const bool has_up = idx >= words_per_row;
    const bool has_down = idx < last_row_base;

    // Missing rows are zero because cells outside the grid are dead.
    const u64 center = load_word(input, idx);
    const u64 above = has_up ? load_word(input, idx - words_per_row) : 0ull;
    const u64 below = has_down ? load_word(input, idx + words_per_row) : 0ull;

    // Warp-local horizontal reuse. For words_per_row == 16, a warp spans two rows, but the
    // has_left/has_right predicates prevent crossing the row boundary at lanes 15/16.
    const u64 center_prev = __shfl_up_sync(kFullWarpMask, center, 1);
    const u64 center_next = __shfl_down_sync(kFullWarpMask, center, 1);
    const u64 above_prev = __shfl_up_sync(kFullWarpMask, above, 1);
    const u64 above_next = __shfl_down_sync(kFullWarpMask, above, 1);
    const u64 below_prev = __shfl_up_sync(kFullWarpMask, below, 1);
    const u64 below_next = __shfl_down_sync(kFullWarpMask, below, 1);

    const unsigned int lane = threadIdx.x & 31u;

    const bool use_shfl_left = has_left && (lane != 0u);
    const bool use_shfl_right = has_right && (lane != 31u);
    const bool load_left = has_left && (lane == 0u);
    const bool load_right = has_right && (lane == 31u);

    // Missing left/right words are zero for the same "outside the grid is dead" reason.
    const u64 left_word = use_shfl_left
        ? center_prev
        : (load_left ? load_word(input, idx - 1) : 0ull);
    const u64 right_word = use_shfl_right
        ? center_next
        : (load_right ? load_word(input, idx + 1) : 0ull);

    const u64 above_left = use_shfl_left
        ? above_prev
        : ((load_left && has_up) ? load_word(input, idx - words_per_row - 1) : 0ull);
    const u64 above_right = use_shfl_right
        ? above_next
        : ((load_right && has_up) ? load_word(input, idx - words_per_row + 1) : 0ull);

    const u64 below_left = use_shfl_left
        ? below_prev
        : ((load_left && has_down) ? load_word(input, idx + words_per_row - 1) : 0ull);
    const u64 below_right = use_shfl_right
        ? below_next
        : ((load_right && has_down) ? load_word(input, idx + words_per_row + 1) : 0ull);

    // Row-wise neighbor counts:
    //   top    = above-left + above + above-right
    //   middle = left + right
    //   bottom = below-left + below + below-right
    u64 top_hi, top_lo;
    sum_three_adjacent(above_left, above, above_right, top_hi, top_lo);

    const u64 west = align_west(center, left_word);
    const u64 east = align_east(center, right_word);
    const u64 mid_lo = west ^ east;  // 0/1 count bit for the two same-row side neighbors.
    const u64 mid_hi = west & east;  // 2's bit for the two same-row side neighbors.

    u64 bot_hi, bot_lo;
    sum_three_adjacent(below_left, below, below_right, bot_hi, bot_lo);

    // Sum the low bits from the three row contributions:
    //   top_lo + mid_lo + bot_lo = count0 + 2*carry0
    u64 carry0, count0;
    csa3(top_lo, mid_lo, bot_lo, carry0, count0);

    // Total neighbor count can now be written as:
    //   neighbors = count0 + 2 * H
    // where
    //   H = carry0 + top_hi + mid_hi + bot_hi
    //
    // The only counts that matter are 2 and 3:
    //   2 -> H == 1 and count0 == 0
    //   3 -> H == 1 and count0 == 1
    //
    // So the final rule becomes:
    //   next = (H == 1) & (count0 | center)
    // which simultaneously implements:
    //   * birth on exactly 3 neighbors
    //   * survival on exactly 2 or 3 neighbors
    const u64 high_sum_eq_one = one_of_four(carry0, top_hi, mid_hi, bot_hi);
    const u64 next = high_sum_eq_one & (count0 | center);

    output[idx] = static_cast<std::uint64_t>(next);
}
}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The caller guarantees:
    //   * grid_dimensions is a power of two
    //   * grid_dimensions > 512
    //   * input and output are device pointers allocated by cudaMalloc
    //
    // No synchronization or error checking is done here by design: the caller owns
    // synchronization, and raw step throughput is the only concern.
    const std::size_t words_per_row = static_cast<std::size_t>(grid_dimensions) >> kCellsPerWordLog2;
    const std::size_t word_mask = words_per_row - 1;
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * words_per_row;
    const std::size_t last_row_base = total_words - words_per_row;

    // Exact division under the stated constraints because total_words is a power of two
    // and kThreadsPerBlock is also a power of two.
    const unsigned int blocks = static_cast<unsigned int>(total_words / kThreadsPerBlock);

    game_of_life_kernel<<<blocks, kThreadsPerBlock>>>(
        input,
        output,
        words_per_row,
        word_mask,
        last_row_base);
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
