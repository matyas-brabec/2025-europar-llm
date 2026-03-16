#include <cstdint>
#include <cuda_runtime.h>

// High-throughput one-step Conway's Game of Life for a square, bit-packed board.
//
// Layout:
// - The board is stored row-major.
// - Each 64-bit word stores 64 consecutive cells from one row.
// - Bit i in a word corresponds to x = word_base + i.
// - Bit value 1 means alive, 0 means dead.
//
// Mapping:
// - One CUDA thread updates one 64-bit word, so no atomics are required.
// - The kernel is intentionally register-only plus warp shuffles:
//   shared memory is unnecessary here and would only add complexity.
//
// Boundary handling:
// - Cells outside the board are dead.
// - For bit 0 and bit 63, horizontal neighbors may come from the left/right words of
//   the same row, and likewise for the rows above and below.
// - The west/east alignment helpers below import those cross-word boundary bits.

namespace {
using u64 = std::uint64_t;

constexpr int kBlockSize = 256;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert((kBlockSize % 32) == 0, "kBlockSize must be a multiple of the warp size.");

// 3-input majority. On modern NVIDIA GPUs this boolean pattern maps cleanly to LOP3.
__device__ __forceinline__ u64 majority3_u64(const u64 a, const u64 b, const u64 c) {
    return (a & b) | (c & (a | b));
}

// 64-bit warp shuffles. The explicit cast keeps the overload selection unambiguous.
__device__ __forceinline__ u64 shfl_up_u64(const u64 v, const unsigned int delta) {
    return static_cast<u64>(
        __shfl_up_sync(kFullWarpMask, static_cast<unsigned long long>(v), delta));
}

__device__ __forceinline__ u64 shfl_down_u64(const u64 v, const unsigned int delta) {
    return static_cast<u64>(
        __shfl_down_sync(kFullWarpMask, static_cast<unsigned long long>(v), delta));
}

// Bit i in the result is the west neighbor of cell i.
// Since bit 0 is the leftmost cell within the 64-cell span, west is a logical left shift.
// The new bit 0 comes from bit 63 of the word immediately to the left.
__device__ __forceinline__ u64 west_aligned(const u64 word, const u64 left_word) {
    return (word << 1) | (left_word >> 63);
}

// Bit i in the result is the east neighbor of cell i.
// East is a logical right shift, with the new bit 63 imported from bit 0 of the word
// immediately to the right.
__device__ __forceinline__ u64 east_aligned(const u64 word, const u64 right_word) {
    return (word >> 1) | (right_word << 63);
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              const u64 total_words,
                              const unsigned int words_per_row) {
    // Linear indexing avoids 2D-grid limits for very large boards and avoids idle threads
    // when a row contains only a small number of 64-bit words.
    const u64 idx = static_cast<u64>(blockIdx.x) * static_cast<u64>(kBlockSize) + threadIdx.x;
    const u64 stride = static_cast<u64>(words_per_row);

    // Because words_per_row is a power of two, x = idx % words_per_row is just a mask.
    // Using x rather than lane id for row-edge tests also handles the smallest valid board
    // (1024x1024, 16 words per row), where one warp spans two rows.
    const unsigned int row_mask = words_per_row - 1u;
    const unsigned int x = static_cast<unsigned int>(idx) & row_mask;
    const unsigned int lane = threadIdx.x & 31u;

    const bool at_row_start = (x == 0u);
    const bool at_row_end = (x == row_mask);
    const bool has_north = (idx >= stride);
    const bool has_south = (idx < (total_words - stride));

    // Outside-of-grid words are treated as zero, exactly matching the problem statement.
    //
    // Each thread loads only the vertically aligned words:
    //   north/current/south at the same x position.
    // Horizontal neighbor words are recovered via warp shuffles. Only warp-edge lanes that
    // are not also row edges need a fallback global load.
    const u64 north = has_north ? input[idx - stride] : 0ull;
    const u64 alive = input[idx];
    const u64 south = has_south ? input[idx + stride] : 0ull;

    const u64 north_from_prev_lane = shfl_up_u64(north, 1);
    const u64 north_from_next_lane = shfl_down_u64(north, 1);
    const u64 alive_from_prev_lane = shfl_up_u64(alive, 1);
    const u64 alive_from_next_lane = shfl_down_u64(alive, 1);
    const u64 south_from_prev_lane = shfl_up_u64(south, 1);
    const u64 south_from_next_lane = shfl_down_u64(south, 1);

    u64 north_left = 0ull;
    u64 north_right = 0ull;
    u64 alive_left = 0ull;
    u64 alive_right = 0ull;
    u64 south_left = 0ull;
    u64 south_right = 0ull;

    if (!at_row_start) {
        north_left = (lane != 0u)
                         ? north_from_prev_lane
                         : (has_north ? input[idx - stride - 1ull] : 0ull);
        alive_left = (lane != 0u) ? alive_from_prev_lane : input[idx - 1ull];
        south_left = (lane != 0u)
                         ? south_from_prev_lane
                         : (has_south ? input[idx + stride - 1ull] : 0ull);
    }

    if (!at_row_end) {
        north_right = (lane != 31u)
                          ? north_from_next_lane
                          : (has_north ? input[idx - stride + 1ull] : 0ull);
        alive_right = (lane != 31u) ? alive_from_next_lane : input[idx + 1ull];
        south_right = (lane != 31u)
                          ? south_from_next_lane
                          : (has_south ? input[idx + stride + 1ull] : 0ull);
    }

    // Build the eight aligned neighbor bitboards for this word:
    //
    //   north_west  north  north_east
    //   west               east
    //   south_west  south  south_east
    //
    // Bit i in each mask corresponds to the neighbor of cell i.
    const u64 north_west = west_aligned(north, north_left);
    const u64 north_east = east_aligned(north, north_right);
    const u64 west = west_aligned(alive, alive_left);
    const u64 east = east_aligned(alive, alive_right);
    const u64 south_west = west_aligned(south, south_left);
    const u64 south_east = east_aligned(south, south_right);

    // Bit-sliced population count over all 64 cells in parallel.
    //
    // First compress each row neighborhood into low/high bit planes:
    // - top_lo/top_hi encode (north_west + north + north_east)
    // - mid_lo/mid_hi encode (west + east)
    // - bot_lo/bot_hi encode (south_west + south + south_east)
    //
    // For a 3-input sum, parity is the low bit and majority is the high bit:
    //   a + b + c = (a ^ b ^ c) + 2*majority(a,b,c)
    const u64 top_lo = north_west ^ north ^ north_east;
    const u64 top_hi = majority3_u64(north_west, north, north_east);

    const u64 mid_lo = west ^ east;
    const u64 mid_hi = west & east;

    const u64 bot_lo = south_west ^ south ^ south_east;
    const u64 bot_hi = majority3_u64(south_west, south, south_east);

    // Sum the three low planes:
    //   top_lo + mid_lo + bot_lo = count_bit0 + 2*carry_to_twos
    const u64 count_bit0 = top_lo ^ mid_lo ^ bot_lo;
    const u64 carry_to_twos = majority3_u64(top_lo, mid_lo, bot_lo);

    // Now sum the four weight-2 contributors:
    //   carry_to_twos + top_hi + mid_hi + bot_hi
    //
    // This yields the upper three bits of the full 4-bit neighbor count:
    //   count = count_bit0 + 2*count_bit1 + 4*count_bit2 + 8*count_bit3
    const u64 pair_lo0 = carry_to_twos ^ top_hi;
    const u64 pair_hi0 = carry_to_twos & top_hi;
    const u64 pair_lo1 = mid_hi ^ bot_hi;
    const u64 pair_hi1 = mid_hi & bot_hi;

    const u64 count_bit1 = pair_lo0 ^ pair_lo1;
    const u64 carry_to_fours = pair_lo0 & pair_lo1;
    const u64 count_bit2 = carry_to_fours ^ pair_hi0 ^ pair_hi1;
    const u64 count_bit3 = majority3_u64(carry_to_fours, pair_hi0, pair_hi1);

    // Game of Life rule:
    //   next = (count == 3) | (alive & count == 2)
    //
    // With count bits c3:c2:c1:c0 this simplifies to:
    //   next = ~(c2 | c3) & c1 & (c0 | alive)
    const u64 count_is_2_or_3 = (~(count_bit2 | count_bit3)) & count_bit1;
    output[idx] = count_is_2_or_3 & (count_bit0 | alive);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The intended usage is a ping-pong update, so input and output are treated as
    // non-overlapping device buffers.
    //
    // words_per_row = grid_dimensions / 64
    // total_words   = grid_dimensions * words_per_row = grid_dimensions^2 / 64
    //
    // Because grid_dimensions is guaranteed to be a power of two > 512:
    // - words_per_row is also a power of two,
    // - total_words is a power of two,
    // - total_words is therefore divisible by the fixed 256-thread block size.
    //
    // That exact divisibility is why the kernel can use full-warp shuffles with no
    // tail handling and no bounds check on idx.
    const unsigned int words_per_row = static_cast<unsigned int>(grid_dimensions) >> 6;
    const std::uint64_t total_words =
        static_cast<std::uint64_t>(grid_dimensions) * static_cast<std::uint64_t>(words_per_row);
    const unsigned int blocks = static_cast<unsigned int>(total_words / kBlockSize);

    // No synchronization or error polling here by design; the caller owns stream ordering
    // and any required synchronization.
    game_of_life_step_kernel<<<blocks, kBlockSize>>>(input, output, total_words, words_per_row);
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
