#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

using u64 = std::uint64_t;

// The kernel is intentionally built around 256-thread blocks.
// That is a good fit for modern data-center GPUs while keeping enough warps
// resident to hide memory latency.
constexpr int kBlockThreads   = 256;
constexpr int kSegmentWidth16 = 16;
constexpr int kBlockY16       = 16;
constexpr int kSegmentWidth32 = 32;
constexpr int kBlockY32       = 8;

static_assert(kSegmentWidth16 * kBlockY16 == kBlockThreads, "Unexpected 16-wide block shape.");
static_assert(kSegmentWidth32 * kBlockY32 == kBlockThreads, "Unexpected 32-wide block shape.");

// Bit 0 is treated as the leftmost cell inside a 64-bit word and bit 63 as the rightmost.
// align_west() produces, for every bit position i, the state of the west neighbor (x - 1)
// aligned back to bit i. The carry-in for bit 0 comes from the previous 64-bit word.
__device__ __forceinline__ u64 align_west(u64 center, u64 west_word) {
    return (center << 1) | (west_word >> 63);
}

// align_east() produces, for every bit position i, the state of the east neighbor (x + 1)
// aligned back to bit i. The carry-in for bit 63 comes from the next 64-bit word.
__device__ __forceinline__ u64 align_east(u64 center, u64 east_word) {
    return (center >> 1) | (east_word << 63);
}

// Bit-sliced half adder: every one of the 64 bit positions behaves like an independent 1-bit lane.
// Integer '+' cannot be used here because carries would leak between different cell positions.
__device__ __forceinline__ void add2(u64 a, u64 b, u64& sum, u64& carry) {
    sum   = a ^ b;
    carry = a & b;
}

// Bit-sliced full adder for three 1-bit planes.
__device__ __forceinline__ void add3(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 x = a ^ b;
    sum   = x ^ c;
    carry = (a & b) | (x & c);
}

// For four 1-bit planes a, b, c, d:
//   parity = (a + b + c + d) & 1
//   ge2    = (a + b + c + d) >= 2
//
// In this kernel these four inputs are the contributors to the "2's" bit of the
// total neighbor count. Therefore ge2 is exactly the mask "neighbor_count >= 4",
// which is all Conway's rule needs above 3.
__device__ __forceinline__ void parity_and_ge2_of4(
    u64 a, u64 b, u64 c, u64 d,
    u64& parity, u64& ge2)
{
    const u64 p = a ^ b;
    const u64 q = c ^ d;
    parity = p ^ q;
    ge2    = (a & b) | (c & d) | (p & q);
}

// SEGMENT_WIDTH is the X extent of one independently shuffled row segment.
//   * SEGMENT_WIDTH == 32:
//       one whole warp maps to one row segment.
//   * SEGMENT_WIDTH == 16:
//       one warp maps to two row segments, but the shuffle width keeps the two
//       16-lane halves fully independent.
//
// This lets interior threads obtain west/east neighboring words via warp shuffles,
// reducing the common-case global loads from 9 words down to 3 words per thread
// (north/current/south). Only segment-boundary lanes fall back to global memory
// for the extra west/east words.
template <int SEGMENT_WIDTH, int BLOCK_Y>
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int grid_dimensions,
    int words_per_row)
{
    static_assert(SEGMENT_WIDTH * BLOCK_Y == kBlockThreads, "Unexpected block shape.");
    static_assert(SEGMENT_WIDTH == 16 || SEGMENT_WIDTH == 32, "Unexpected shuffle width.");

    const int lane = static_cast<int>(threadIdx.x);
    const int col  = static_cast<int>(blockIdx.x) * SEGMENT_WIDTH + lane;
    const int row  = static_cast<int>(blockIdx.y) * BLOCK_Y + static_cast<int>(threadIdx.y);

    // run_game_of_life() exploits the problem guarantees:
    //   - grid_dimensions is a power of 2 and > 512
    //   - words_per_row = grid_dimensions / 64 is therefore also a power of 2
    //   - words_per_row is either 16 (only for 1024x1024) or a multiple of 32
    //
    // That allows exact tiling with no out-of-range threads:
    //   - SEGMENT_WIDTH = 16 when words_per_row == 16
    //   - SEGMENT_WIDTH = 32 otherwise
    // grid_dimensions is also an exact multiple of BLOCK_Y, so no Y-boundary masking
    // is needed beyond the Game-of-Life edge rules themselves.
    const bool has_top    = (row != 0);
    const bool has_bottom = (row + 1 != grid_dimensions);
    const bool has_left   = (col != 0);
    const bool has_right  = (col + 1 != words_per_row);

    const std::size_t stride   = static_cast<std::size_t>(words_per_row);
    const std::size_t row_base = static_cast<std::size_t>(row) * stride;

    const u64* const row_ptr = input + row_base;
    const u64* const top_ptr = has_top    ? (row_ptr - stride) : nullptr;
    const u64* const bot_ptr = has_bottom ? (row_ptr + stride) : nullptr;

    // Mandatory loads for this output word: north/current/south in the same column.
    const u64 top = has_top    ? top_ptr[col] : 0ull;
    const u64 mid = row_ptr[col];
    const u64 bot = has_bottom ? bot_ptr[col] : 0ull;

    // Latest CUDA allows 64-bit values in shuffles directly.
    // The width parameter keeps 16-lane specializations confined to half-warps.
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    const u64 top_from_west = __shfl_up_sync  (kFullMask, top, 1, SEGMENT_WIDTH);
    const u64 top_from_east = __shfl_down_sync(kFullMask, top, 1, SEGMENT_WIDTH);
    const u64 mid_from_west = __shfl_up_sync  (kFullMask, mid, 1, SEGMENT_WIDTH);
    const u64 mid_from_east = __shfl_down_sync(kFullMask, mid, 1, SEGMENT_WIDTH);
    const u64 bot_from_west = __shfl_up_sync  (kFullMask, bot, 1, SEGMENT_WIDTH);
    const u64 bot_from_east = __shfl_down_sync(kFullMask, bot, 1, SEGMENT_WIDTH);

    // Default west/east words come from neighboring lanes inside the same shuffled segment.
    // For the segment boundary lanes, fall back to explicit global loads if the board actually
    // has a west/east word there; otherwise keep zero to model "outside the grid is dead".
    u64 west_top_word = (lane != 0) ? top_from_west : 0ull;
    u64 west_mid_word = (lane != 0) ? mid_from_west : 0ull;
    u64 west_bot_word = (lane != 0) ? bot_from_west : 0ull;

    if (lane == 0 && has_left) {
        const int west_col = col - 1;
        if (has_top)    west_top_word = top_ptr[west_col];
        west_mid_word = row_ptr[west_col];
        if (has_bottom) west_bot_word = bot_ptr[west_col];
    }

    u64 east_top_word = (lane + 1 != SEGMENT_WIDTH) ? top_from_east : 0ull;
    u64 east_mid_word = (lane + 1 != SEGMENT_WIDTH) ? mid_from_east : 0ull;
    u64 east_bot_word = (lane + 1 != SEGMENT_WIDTH) ? bot_from_east : 0ull;

    if (lane + 1 == SEGMENT_WIDTH && has_right) {
        const int east_col = col + 1;
        if (has_top)     east_top_word = top_ptr[east_col];
        east_mid_word = row_ptr[east_col];
        if (has_bottom)  east_bot_word = bot_ptr[east_col];
    }

    // Build the six horizontally shifted bitboards that complete the 3x3 neighborhood.
    // Each bit position now sees its own aligned north-west / west / south-west etc.
    const u64 top_l = align_west(top, west_top_word);
    const u64 top_r = align_east(top, east_top_word);
    const u64 mid_l = align_west(mid, west_mid_word);
    const u64 mid_r = align_east(mid, east_mid_word);
    const u64 bot_l = align_west(bot, west_bot_word);
    const u64 bot_r = align_east(bot, east_bot_word);

    // First stage: horizontal population counts per row.
    //
    // top_count    = top_l + top + top_r      in [0, 3]
    // middle_count = mid_l + mid_r            in [0, 2]   (center cell excluded)
    // bottom_count = bot_l + bot + bot_r      in [0, 3]
    //
    // Each count is represented in bit-sliced form:
    //   row0 -> low bit of the per-cell row sum
    //   row1 -> high bit of the per-cell row sum
    u64 top0, top1;
    add3(top_l, top, top_r, top0, top1);

    u64 mid0, mid1;
    add2(mid_l, mid_r, mid0, mid1);

    u64 bot0, bot1;
    add3(bot_l, bot, bot_r, bot0, bot1);

    // Second stage: add the three row counts vertically.
    //
    // count0 is the exact LSB of the total 8-neighbor count.
    // carry_low/top1/mid1/bot1 are the four contributors to the "2's" bit.
    u64 count0, carry_low;
    add3(top0, mid0, bot0, count0, carry_low);

    // count1 is the exact second bit of the total neighbor count.
    // ge4 is the exact mask "neighbor_count >= 4".
    u64 count1, ge4;
    parity_and_ge2_of4(carry_low, top1, mid1, bot1, count1, ge4);

    // Conway rule:
    //   next = (count == 3) | (alive & count == 2)
    //
    // Counts 2 and 3 share:
    //   ge4    == 0
    //   count1 == 1
    //
    // count0 distinguishes them:
    //   count0 == 0 -> exactly 2
    //   count0 == 1 -> exactly 3
    const u64 two_or_three = ~ge4 & count1;
    output[row_base + static_cast<std::size_t>(col)] = two_or_three & (count0 | mid);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller owns synchronization/stream semantics; this function only enqueues one kernel.
    //
    // Because grid_dimensions is a power of 2 and > 512:
    //   words_per_row = grid_dimensions / 64
    // is also a power of 2, therefore either:
    //   * 16  (only for a 1024x1024 grid), or
    //   * a multiple of 32
    //
    // We use that to pick an exact launch shape with no out-of-range threads.
    const int words_per_row = grid_dimensions >> 6;

    if (words_per_row == kSegmentWidth16) {
        const dim3 block(kSegmentWidth16, kBlockY16, 1);
        const dim3 grid(1u, static_cast<unsigned int>(grid_dimensions / kBlockY16), 1);
        game_of_life_kernel<kSegmentWidth16, kBlockY16>
            <<<grid, block>>>(input, output, grid_dimensions, words_per_row);
    } else {
        const dim3 block(kSegmentWidth32, kBlockY32, 1);
        const dim3 grid(
            static_cast<unsigned int>(words_per_row / kSegmentWidth32),
            static_cast<unsigned int>(grid_dimensions / kBlockY32),
            1);
        game_of_life_kernel<kSegmentWidth32, kBlockY32>
            <<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
