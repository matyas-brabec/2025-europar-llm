#include <cuda_runtime.h>
#include <cstdint>

namespace {

// One-step Conway's Game of Life on a bit-packed square grid.
//
// Design points chosen strictly for throughput on modern NVIDIA data-center GPUs:
//   - One thread updates one 64-bit packed word = 64 cells.
//   - No shared/texture memory: horizontal reuse is captured with warp shuffles instead.
//   - Neighbor counts are accumulated with carry-save full/half adders so all 64 cells in a
//     word are processed in parallel with simple bitwise logic.
//
// Bit convention inside each 64-bit word:
//   - bit 0  = leftmost cell of the 64-cell run
//   - bit 63 = rightmost cell of the 64-cell run
//
// This matches the prompt's requirement that bit 0 depends on words to the left and bit 63 on
// words to the right.

using u64 = std::uint64_t;

constexpr int BLOCK_THREADS = 256;
constexpr int BLOCK_SHIFT   = 8;
constexpr unsigned int FULL_MASK = 0xFFFFFFFFu;

static_assert(BLOCK_THREADS == (1 << BLOCK_SHIFT),
              "BLOCK_THREADS and BLOCK_SHIFT must remain consistent.");
static_assert((BLOCK_THREADS % 32) == 0,
              "BLOCK_THREADS must be a whole number of warps.");
static_assert(sizeof(u64) == 8, "Expected 64-bit packed words.");

// 2-input bit-sliced adder.
// For every bit position independently:
//   sum   = a xor b
//   carry = a & b
// carry has twice the weight of sum in the population-count reduction.
__device__ __forceinline__ void half_adder(const u64 a, const u64 b,
                                           u64& sum, u64& carry) {
    sum   = a ^ b;
    carry = a & b;
}

// 3-input bit-sliced full adder.
// For every bit position independently:
//   sum   = a xor b xor c
//   carry = majority(a, b, c)
// carry again has twice the weight of sum.
__device__ __forceinline__ void full_adder(const u64 a, const u64 b, const u64 c,
                                           u64& sum, u64& carry) {
    const u64 axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (axb & c);
}

template <int SHUFFLE_WIDTH>
__global__ __launch_bounds__(BLOCK_THREADS)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         const unsigned int words_per_row,
                         const unsigned int word_mask,
                         const unsigned int last_row_start) {
    // One thread per packed 64-cell word.
    //
    // The problem constraints guarantee:
    //   grid_dimensions = 2^k, k >= 10
    //   words_per_row   = grid_dimensions / 64 = 2^(k-6)
    //   total_words     = grid_dimensions * words_per_row = 2^(2k-6)
    //
    // Therefore total_words is always a multiple of 256, so this launch has no tail block and
    // needs no bounds check.
    //
    // Under the stated A100/H100-class memory envelope, even the largest admissible power-of-two
    // grid still has valid packed-word indices in [0, 2^32 - 1], so 32-bit index arithmetic is
    // sufficient and cheaper than 64-bit indexing here.
    const unsigned int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    // words_per_row is a power of two, so x = idx mod words_per_row is just a mask.
    const unsigned int x = idx & word_mask;

    const bool has_left  = (x != 0u);
    const bool has_right = (x != word_mask);
    const bool has_up    = (idx >= words_per_row);
    const bool has_down  = (idx < last_row_start);

    const unsigned int up_idx   = idx - words_per_row;   // Wrap on top row is harmless; guarded by has_up.
    const unsigned int down_idx = idx + words_per_row;   // Wrap on bottom row is harmless; guarded by has_down.

    // Mandatory global loads: the current word plus the vertically adjacent words.
    const u64 current = input[idx];

    u64 north = 0ULL;
    if (has_up) {
        north = input[up_idx];
    }

    u64 south = 0ULL;
    if (has_down) {
        south = input[down_idx];
    }

    // Horizontal neighboring words are usually obtained with warp shuffles instead of extra
    // global loads. Only lanes at the edge of a shuffle segment fall back to global memory.
    //
    // Two segment widths are enough:
    //   SHUFFLE_WIDTH == 16 : only for the smallest legal grid (1024x1024 => 16 words/row),
    //                         so each half warp corresponds to one complete row.
    //   SHUFFLE_WIDTH == 32 : all larger grids; each warp is a 32-word row segment.
    //
    // For rows wider than one segment, segment-edge lanes load the missing left/right words from
    // global memory. This avoids shared memory while preserving correctness across warp and block
    // boundaries.
    const unsigned int lane         = threadIdx.x & 31u;
    constexpr unsigned int SEG_MASK = static_cast<unsigned int>(SHUFFLE_WIDTH - 1);
    const unsigned int segment_lane = lane & SEG_MASK;

    u64 left_current  = __shfl_up_sync(FULL_MASK, current, 1, SHUFFLE_WIDTH);
    u64 right_current = __shfl_down_sync(FULL_MASK, current, 1, SHUFFLE_WIDTH);
    u64 left_north    = __shfl_up_sync(FULL_MASK, north,   1, SHUFFLE_WIDTH);
    u64 right_north   = __shfl_down_sync(FULL_MASK, north, 1, SHUFFLE_WIDTH);
    u64 left_south    = __shfl_up_sync(FULL_MASK, south,   1, SHUFFLE_WIDTH);
    u64 right_south   = __shfl_down_sync(FULL_MASK, south, 1, SHUFFLE_WIDTH);

    if (segment_lane == 0u) {
        if (has_left) {
            left_current = input[idx - 1u];
            left_north   = has_up   ? input[up_idx   - 1u] : 0ULL;
            left_south   = has_down ? input[down_idx - 1u] : 0ULL;
        } else {
            left_current = 0ULL;
            left_north   = 0ULL;
            left_south   = 0ULL;
        }
    }

    if (segment_lane == SEG_MASK) {
        if (has_right) {
            right_current = input[idx + 1u];
            right_north   = has_up   ? input[up_idx   + 1u] : 0ULL;
            right_south   = has_down ? input[down_idx + 1u] : 0ULL;
        } else {
            right_current = 0ULL;
            right_north   = 0ULL;
            right_south   = 0ULL;
        }
    }

    // Align each neighbor direction to the destination bit position.
    //
    // Example for the west neighbor:
    //   aligned_west[i] = current[i-1] for i > 0
    //   aligned_west[0] = left_current[63]
    //
    // This is exactly the special handling required for bits 0 and 63 at word boundaries.
    //
    // We then reduce the 8 aligned neighbor bitplanes with a carry-save adder tree:
    //   top row    : NW + N + NE  -> top_ones,    top_twos
    //   bottom row : SW + S + SE  -> bottom_ones, bottom_twos
    //   same row   : W  + E       -> horiz_ones,  horiz_twos
    //
    // Finally:
    //   count_bit0    = bit 0 of the neighbor count
    //   count_bit1    = bit 1 of the neighbor count, provided count < 4
    //   at_least_four = count in [4, 8]
    //
    // Game of Life then becomes:
    //   next = (count == 3) | (current & (count == 2))
    //        = count_bit1 & ~at_least_four & (count_bit0 | current)
    u64 top_ones, top_twos;
    full_adder((north << 1) | (left_north >> 63),    // NW
               north,                                // N
               (north >> 1) | (right_north << 63),   // NE
               top_ones, top_twos);

    u64 bottom_ones, bottom_twos;
    full_adder((south << 1) | (left_south >> 63),    // SW
               south,                                // S
               (south >> 1) | (right_south << 63),   // SE
               bottom_ones, bottom_twos);

    u64 horiz_ones, horiz_twos;
    half_adder((current << 1) | (left_current >> 63),    // W
               (current >> 1) | (right_current << 63),   // E
               horiz_ones, horiz_twos);

    u64 count_bit0, twos_from_ones;
    full_adder(top_ones, bottom_ones, horiz_ones, count_bit0, twos_from_ones);

    u64 twos_lsb, fours_from_twos;
    full_adder(top_twos, bottom_twos, horiz_twos, twos_lsb, fours_from_twos);

    u64 count_bit1, fours_from_merge;
    half_adder(twos_lsb, twos_from_ones, count_bit1, fours_from_merge);

    const u64 at_least_four = fours_from_twos | fours_from_merge;
    output[idx] = count_bit1 & ~at_least_four & (count_bit0 | current);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees:
    //   - grid_dimensions is a power of two
    //   - grid_dimensions > 512
    //   - input/output are device pointers from cudaMalloc
    //   - input and output are distinct ping-pong buffers
    //
    // Therefore:
    //   words_per_row = grid_dimensions / 64 is a power of two and at least 16
    //   total_words   = grid_dimensions * words_per_row is an exact multiple of 256
    //
    // No host-side synchronization or error checking is performed here because the prompt states
    // that synchronization is handled by the caller and only performance matters.

    const unsigned int n = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = n >> 6;

    const std::uint64_t total_words =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(words_per_row);

    const unsigned int word_mask      = words_per_row - 1u;
    const unsigned int last_row_start = static_cast<unsigned int>(total_words - words_per_row);
    const unsigned int grid_blocks    = static_cast<unsigned int>(total_words >> BLOCK_SHIFT);

    if (words_per_row == 16u) {
        game_of_life_kernel<16><<<grid_blocks, BLOCK_THREADS>>>(
            input, output, words_per_row, word_mask, last_row_start);
    } else {
        game_of_life_kernel<32><<<grid_blocks, BLOCK_THREADS>>>(
            input, output, words_per_row, word_mask, last_row_start);
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
