#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

/*
  Bit-sliced Conway's Game of Life for a square, bit-packed board.

  Layout assumed by the prompt:
    - The board is row-major.
    - One 64-bit word stores 64 consecutive x positions from one row.
    - Bit j of word (row, word_col) is cell x = 64 * word_col + j.
    - Cells outside the board are dead.

  The kernel processes 64 cells at once per thread by treating each bit position as an independent
  1-bit lane. Neighbor counts are accumulated with carry-save adder logic:
      sum   = a ^ b ^ c
      carry = majority(a,b,c) = (a & b) | ((a ^ b) & c)
  applied bitwise across the full 64-bit word.

  The "special handling" for bit 0 and bit 63 is exactly the cross-word splice done by align_west()
  and align_east():
    - west neighbor of bit 0 comes from bit 63 of the word on the left
    - east neighbor of bit 63 comes from bit 0 of the word on the right

  This implementation intentionally stays with plain global memory; for this access pattern on modern
  A100/H100-class GPUs, regular coalesced loads plus cheap bitwise ALU is the fast path, and shared
  or texture memory only adds complexity.
*/

namespace {

using u64 = std::uint64_t;

constexpr int kBlockSize = 256;

// Power-of-two CTA cap.
// Boards up to 16384x16384 cells launch literally one thread per 64-bit word.
// Larger boards keep the same per-word ownership but use a grid-stride loop to avoid launching
// millions of CTAs. Keeping the cap as a power of two also makes the capped grid-stride a power
// of two, which matches the power-of-two row pitch nicely.
constexpr unsigned int kMaxBlocks = 1u << 14;  // 16384

static_assert((kBlockSize & (kBlockSize - 1)) == 0, "kBlockSize must be a power of two.");
static_assert((kMaxBlocks & (kMaxBlocks - 1u)) == 0u, "kMaxBlocks must be a power of two.");
static_assert(sizeof(u64) == 8, "u64 must be 64 bits.");

// 3:2 carry-save adder applied bitwise to all 64 lanes at once.
// For each bit position this is a full adder:
//   sum   = a ^ b ^ c
//   carry = majority(a, b, c)
// The carry represents the next higher count bit.
static __device__ __forceinline__
void csa3(const u64 a, const u64 b, const u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// Align the west neighbors of the 64 cells owned by the current word.
// Bit 0 pulls in bit 63 from the word on the left; all other bits come from a 1-bit left shift.
static __device__ __forceinline__
u64 align_west(const u64 center, const u64 left) {
    return (center << 1) | (left >> 63);
}

// Align the east neighbors of the 64 cells owned by the current word.
// Bit 63 pulls in bit 0 from the word on the right; all other bits come from a 1-bit right shift.
static __device__ __forceinline__
u64 align_east(const u64 center, const u64 right) {
    return (center >> 1) | (right << 63);
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         std::size_t total_words,
                         unsigned int words_per_row) {
    // words_per_row is guaranteed to be a power of two, so words_per_row - 1 is both:
    //   - the last valid word index within a row
    //   - the bitmask that extracts the intra-row word coordinate from a flattened index
    const std::size_t row_stride = static_cast<std::size_t>(words_per_row);
    const std::size_t bottom_limit = total_words - row_stride;
    const unsigned int last_col = words_per_row - 1u;

    const std::size_t thread_index =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t thread_stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    // Each loop iteration still updates exactly one 64-bit word, and each output word is written
    // by exactly one thread. The loop only amortizes launch/scheduling overhead on very large boards.
    for (std::size_t idx = thread_index; idx < total_words; idx += thread_stride) {
        const unsigned int col = static_cast<unsigned int>(idx) & last_col;
        const bool has_left  = (col != 0u);
        const bool has_right = (col != last_col);
        const bool has_up    = (idx >= row_stride);
        const bool has_down  = (idx < bottom_limit);

        // Current word: also serves as the "alive" mask for the final rule application.
        const u64 center = input[idx];

        // Missing neighbor words beyond the border are treated as zero, which exactly implements
        // "cells outside the grid are dead".
        u64 left = 0;
        u64 right = 0;
        if (has_left)  left  = input[idx - 1];
        if (has_right) right = input[idx + 1];

        // Same-row horizontal neighbors (W, E): 0..2 neighbors, encoded as two bit-planes.
        const u64 west  = align_west(center, left);
        const u64 east  = align_east(center, right);
        const u64 side0 = west ^ east;   // bit 0 of the 0..2 count
        const u64 side1 = west & east;   // bit 1 of the 0..2 count

        // North row (NW, N, NE): 0..3 neighbors, encoded as two bit-planes.
        u64 top0 = 0;
        u64 top1 = 0;
        if (has_up) {
            const std::size_t up_idx = idx - row_stride;
            const u64 up = input[up_idx];

            u64 up_left = 0;
            u64 up_right = 0;
            if (has_left)  up_left  = input[up_idx - 1];
            if (has_right) up_right = input[up_idx + 1];

            csa3(align_west(up, up_left), up, align_east(up, up_right), top0, top1);
        }

        // South row (SW, S, SE): 0..3 neighbors, encoded as two bit-planes.
        u64 bottom0 = 0;
        u64 bottom1 = 0;
        if (has_down) {
            const std::size_t down_idx = idx + row_stride;
            const u64 down = input[down_idx];

            u64 down_left = 0;
            u64 down_right = 0;
            if (has_left)  down_left  = input[down_idx - 1];
            if (has_right) down_right = input[down_idx + 1];

            csa3(align_west(down, down_left), down, align_east(down, down_right), bottom0, bottom1);
        }

        // Combine the three groups:
        //   top    : 0..3  -> (top1 top0)
        //   sides  : 0..2  -> (side1 side0)
        //   bottom : 0..3  -> (bottom1 bottom0)
        //
        // First combine the low bits of the three groups; this yields the full count LSB (count0)
        // plus a carry plane (carry0).
        u64 count0;
        u64 carry0;
        csa3(top0, bottom0, side0, count0, carry0);

        // Then combine the remaining carry planes. carry1 is the next count bit, carry2 is the bit
        // above that. We do not need the count bit for value 8 explicitly: counts 2 and 3 are
        // identified entirely by the low three bits, and count 8 has low three bits 000 so it
        // cannot alias either case.
        u64 carry1;
        u64 carry2;
        csa3(carry0, top1, bottom1, carry1, carry2);

        const u64 count1 = carry1 ^ side1;
        const u64 count2 = carry2 ^ (carry1 & side1);

        // Counts 2 and 3 are exactly the cases:
        //   count2 count1 = 0 1
        const u64 count_is_2_or_3 = count1 & ~count2;

        // Game of Life rule in factorized bit-sliced form:
        //   next = (count == 3) | (alive & (count == 2))
        //        = (count is 2 or 3) & (alive | count0)
        // because count0 distinguishes 2 (count0=0) from 3 (count0=1).
        output[idx] = count_is_2_or_3 & (center | count0);
    }
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The prompt guarantees valid power-of-two dimensions > 512, but these guards keep the function
    // well-defined for accidental misuse without adding any measurable cost.
    if (grid_dimensions <= 0) {
        return;
    }

    const unsigned int words_per_row = static_cast<unsigned int>(grid_dimensions) >> 6;
    if (words_per_row == 0u) {
        return;
    }

    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Launch literally one thread per word when the board is small enough.
    // Once the board is larger, cap the grid and let the kernel's grid-stride loop carry the rest.
    // The caller handles synchronization; this function only launches the kernel.
    const std::size_t required_blocks =
        (total_words + static_cast<std::size_t>(kBlockSize) - 1u) /
        static_cast<std::size_t>(kBlockSize);

    const unsigned int blocks = static_cast<unsigned int>(
        required_blocks < static_cast<std::size_t>(kMaxBlocks)
            ? required_blocks
            : static_cast<std::size_t>(kMaxBlocks));

    game_of_life_kernel<<<blocks, kBlockSize>>>(input, output, total_words, words_per_row);
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
