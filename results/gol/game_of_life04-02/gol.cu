#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// High-throughput Conway's Game of Life step for a bit-packed square grid.
//
// Design choices driven purely by performance on modern data-center GPUs:
// - One CUDA thread updates exactly one 64-bit word (64 cells), so no atomics are needed.
// - Shared memory is intentionally avoided: each thread touches only a 3x3 word neighborhood,
//   and the hardware caches on A100/H100-class GPUs handle the reuse well enough that extra
//   staging usually adds complexity without helping.
// - A naive implementation would iterate over 64 bits and use __popc/__popcll on extracted
//   neighborhoods. That is much better than scalar per-cell masking, but a fully bit-sliced
//   implementation is faster still: it updates all 64 cells in parallel using Boolean ops and
//   a carry-save adder tree.
//
// Bit layout assumption used here:
// - Bit i in a 64-bit word corresponds to column (word_base + i).
// - Therefore, shifting a row word left by 1 aligns the west neighbor for each cell lane.
// - Shifting a row word right by 1 aligns the east neighbor for each cell lane.

namespace {
using u64 = std::uint64_t;

constexpr int kBlockSize = 256;

// Align the west neighbor of every cell in a 64-bit word into the current word's bit lanes.
// left_word contributes only to bit 0 (the cross-word boundary case).
__device__ __forceinline__ u64 align_west(u64 center_word, u64 left_word) {
    return (center_word << 1) | (left_word >> 63);
}

// Align the east neighbor of every cell in a 64-bit word into the current word's bit lanes.
// right_word contributes only to bit 63 (the cross-word boundary case).
__device__ __forceinline__ u64 align_east(u64 center_word, u64 right_word) {
    return (center_word >> 1) | (right_word << 63);
}

// 64-lane carry-save adder.
// For every bit position i independently:
//     a_i + b_i + c_i = sum_i + 2 * carry_i
//
// This is the core primitive that lets us sum eight 1-bit neighbor bitboards for 64 cells
// in parallel without looping over the individual bits.
__device__ __forceinline__ void csa(u64& carry, u64& sum, u64 a, u64 b, u64 c) {
    const u64 ab_xor = a ^ b;
    carry = (a & b) | (ab_xor & c);
    sum   = ab_xor ^ c;
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         std::size_t total_words,
                         std::size_t stride_words) {
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (linear >= total_words) {
        return;
    }

    // Because the grid width is a power of two and divisible by 64, the number of 64-bit words
    // per row is also a power of two. That makes x-within-row a cheap bit-mask operation.
    const std::size_t row_mask = stride_words - 1;
    const std::size_t x_in_row = linear & row_mask;

    const bool has_left  = x_in_row != 0;
    const bool has_right = x_in_row != row_mask;
    const bool has_up    = linear >= stride_words;
    const bool has_down  = linear < (total_words - stride_words);

    const u64 center = input[linear];

    // Same-row neighbor words. Outside-grid words are treated as zero.
    u64 left_word  = 0;
    u64 right_word = 0;
    if (has_left)  left_word  = input[linear - 1];
    if (has_right) right_word = input[linear + 1];

    // Same-row aligned neighbor bitboards.
    const u64 west = align_west(center, left_word);
    const u64 east = align_east(center, right_word);

    // Top row: produce two bitboards representing the per-cell sum of (NW, N, NE):
    //   top_sum   = low bit of that 0..3 count
    //   top_carry = high bit of that 0..3 count
    u64 top_sum   = 0;
    u64 top_carry = 0;
    if (has_up) {
        const std::size_t up = linear - stride_words;

        const u64 up_center = input[up];
        u64 up_left  = 0;
        u64 up_right = 0;
        if (has_left)  up_left  = input[up - 1];
        if (has_right) up_right = input[up + 1];

        csa(top_carry, top_sum,
            align_west(up_center, up_left),
            up_center,
            align_east(up_center, up_right));
    }

    // Bottom row: same idea for (SW, S, SE).
    u64 bottom_sum   = 0;
    u64 bottom_carry = 0;
    if (has_down) {
        const std::size_t down = linear + stride_words;

        const u64 down_center = input[down];
        u64 down_left  = 0;
        u64 down_right = 0;
        if (has_left)  down_left  = input[down - 1];
        if (has_right) down_right = input[down + 1];

        csa(bottom_carry, bottom_sum,
            align_west(down_center, down_left),
            down_center,
            align_east(down_center, down_right));
    }

    // Middle row contributes only W and E.
    // For each cell lane:
    //   west_i + east_i = middle_sum_i + 2 * middle_carry_i
    const u64 middle_sum   = west ^ east;
    const u64 middle_carry = west & east;

    // Combine the three low-bit sources (top_sum, bottom_sum, middle_sum).
    // Result:
    //   ones   = 1's bit of the total 0..8 neighbor count
    //   carry2 = carry into the 2's place from those low-bit additions
    u64 carry2 = 0;
    u64 ones   = 0;
    csa(carry2, ones, top_sum, bottom_sum, middle_sum);

    // Combine the existing 2's-place contributors.
    // Result:
    //   twos_base = one source of the 2's bit
    //   carry4    = carry into the 4's place
    u64 carry4   = 0;
    u64 twos_base = 0;
    csa(carry4, twos_base, top_carry, bottom_carry, middle_carry);

    // Merge the two different sources of the 2's bit.
    const u64 twos = twos_base ^ carry2;
    const u64 carry4_from_twos = twos_base & carry2;

    // Final 4's and 8's bits of the 0..8 neighbor count.
    const u64 fours  = carry4 ^ carry4_from_twos;
    const u64 eights = carry4 & carry4_from_twos;

    // Conway rule in bit-sliced form:
    // - count == 3  => ones=1, twos=1, higher bits clear => cell becomes alive
    // - count == 2  => ones=0, twos=1, higher bits clear => cell keeps current state
    //
    // Therefore:
    //   count in {2,3} <=> twos == 1 and no 4's/8's bits are set
    //   next = (count==3) | (current & count==2)
    //        = count_is_2_or_3 & (ones | current)
    const u64 count_is_2_or_3 = twos & ~(fours | eights);
    output[linear] = count_is_2_or_3 & (ones | center);
}
}  // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // Expected by contract:
    // - input and output are device pointers allocated by cudaMalloc
    // - input and output do not alias (Game of Life is not an in-place update)
    // - grid_dimensions is the width/height of the square grid
    // - grid_dimensions is a power of two, > 512, and thus divisible by 64
    //
    // The launch is intentionally asynchronous. No error checking or synchronization is done here
    // because the caller owns stream ordering and synchronization policy.

    const std::size_t n = static_cast<std::size_t>(grid_dimensions);
    const std::size_t stride_words = n >> 6;       // grid_dimensions / 64
    const std::size_t total_words  = n * stride_words;

    const unsigned int blocks = static_cast<unsigned int>(
        (total_words + static_cast<std::size_t>(kBlockSize) - 1) /
        static_cast<std::size_t>(kBlockSize));

    game_of_life_kernel<<<blocks, kBlockSize>>>(input, output, total_words, stride_words);
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
