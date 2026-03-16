#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Use an explicit 64-bit type that has CUDA shuffle overloads on all supported hosts.
using word_t = unsigned long long;

// 256 threads/block = 8 warps/block.
// No shared memory is used, so occupancy is typically limited only by registers.
constexpr unsigned kWarpSize      = 32u;
constexpr unsigned kFullMask      = 0xffffffffu;
constexpr int      kHalfWarpWidth = 16;
constexpr int      kBlockThreads  = 256;
constexpr unsigned kWarpsPerBlock = static_cast<unsigned>(kBlockThreads) / kWarpSize;
constexpr std::size_t kHalfStrideWords = 16;  // 1024x1024 board => 16 packed words/row.

static_assert(kBlockThreads % static_cast<int>(kWarpSize) == 0, "Block size must be warp-aligned.");
static_assert(sizeof(word_t) == 8, "word_t must be 64 bits.");
static_assert(sizeof(std::uint64_t) == 8, "std::uint64_t must be 64 bits.");

// Read one packed 64-cell word from global memory.
__device__ __forceinline__ word_t load_word(const std::uint64_t* __restrict__ base, std::size_t idx) {
    return static_cast<word_t>(base[idx]);
}

// Two-input half adder, lane-wise across all 64 bit positions.
__device__ __forceinline__ void half_add(word_t& carry, word_t& sum, word_t a, word_t b) {
    sum   = a ^ b;
    carry = a & b;
}

// Three-input carry-save adder, lane-wise across all 64 bit positions.
// For every bit position independently: a + b + c == sum + 2*carry.
__device__ __forceinline__ void csa(word_t& carry, word_t& sum, word_t a, word_t b, word_t c) {
    const word_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// Compute the next Game-of-Life state for one packed 64-cell word.
//
// The nine arguments are the raw neighboring packed words in the 3x3 word stencil:
//
//   top_left_word   top_word   top_right_word
//   mid_left_word   mid_word   mid_right_word
//   bot_left_word   bot_word   bot_right_word
//
// Each of those raw words is aligned to the current word's bit positions with shifts.
// The inserted carry bit from the adjacent word handles bit 0 / bit 63 exactly:
// - (row << 1) | (left >> 63)  supplies the "west" neighbor for every bit lane,
//   including bit 0 from the word to the left.
// - (row >> 1) | (right << 63) supplies the "east" neighbor for every bit lane,
//   including bit 63 from the word to the right.
//
// After alignment, the eight neighbors are counted with a bit-sliced carry-save adder
// network. This counts all 64 cells in parallel using only bitwise integer ops.
__device__ __forceinline__ word_t evolve_from_raw_neighbors(
    word_t top_left_word, word_t top_word, word_t top_right_word,
    word_t mid_left_word, word_t mid_word, word_t mid_right_word,
    word_t bot_left_word, word_t bot_word, word_t bot_right_word)
{
    // Horizontally align neighbors from the three rows around the current word.
    const word_t nw = (top_word << 1) | (top_left_word >> 63);
    const word_t ne = (top_word >> 1) | (top_right_word << 63);
    const word_t w  = (mid_word << 1) | (mid_left_word >> 63);
    const word_t e  = (mid_word >> 1) | (mid_right_word << 63);
    const word_t sw = (bot_word << 1) | (bot_left_word >> 63);
    const word_t se = (bot_word >> 1) | (bot_right_word << 63);

    // Count the 8 neighbors with a small CSA tree.
    // Grouping:
    //   top row:    nw + n + ne
    //   bottom row: sw + s + se
    //   middle row: w + e
    //
    // Final count representation:
    //   count = ones + 2*twos + 4*fours + 8*eights
    word_t c_top, s_top;
    csa(c_top, s_top, nw, top_word, ne);

    word_t c_bot, s_bot;
    csa(c_bot, s_bot, sw, bot_word, se);

    word_t c_mid, s_mid;
    half_add(c_mid, s_mid, w, e);

    word_t c_ones, ones;
    csa(c_ones, ones, s_top, s_bot, s_mid);

    word_t c_twos_hi, s_twos_lo;
    csa(c_twos_hi, s_twos_lo, c_top, c_bot, c_mid);

    word_t c_twos_carry, twos;
    half_add(c_twos_carry, twos, s_twos_lo, c_ones);

    word_t eights, fours;
    half_add(eights, fours, c_twos_hi, c_twos_carry);

    // Life rule:
    // - count == 3 => alive
    // - count == 2 => preserve current state
    //
    // `twos & ~(fours | eights)` selects counts 2 or 3.
    // Within that subset, `ones | mid_word` distinguishes:
    //   count 2: ones=0 => keep only if currently alive
    //   count 3: ones=1 => always alive
    const word_t two_or_three = twos & ~(fours | eights);
    return two_or_three & (ones | mid_word);
}

// Special-case kernel for the only legal board width below one full warp of words:
// 1024x1024 cells => 16 packed 64-bit words/row.
//
// One warp processes two rows:
// - lanes  0..15 -> one full 16-word row
// - lanes 16..31 -> the next full 16-word row
//
// Shuffles use width=16 so the lower and upper half-warp act as independent row groups.
// This avoids wasting 16 inactive lanes per row and removes all inter-segment boundary loads
// because the whole row fits inside one half-warp.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel_halfwarp16(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int grid_dimensions,
    std::size_t total_warps)
{
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & (kWarpSize - 1u);
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kBlockThreads) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t warp_id = global_thread >> 5;

    if (warp_id >= total_warps) {
        return;
    }

    const unsigned rows     = static_cast<unsigned>(grid_dimensions);
    const unsigned last_row = rows - 1u;

    const unsigned half    = lane >> 4;   // 0 for lower half-warp, 1 for upper half-warp
    const unsigned sublane = lane & 15u;  // 0..15 within its row
    const unsigned row     = static_cast<unsigned>(warp_id << 1) + half;

    // Because each warp covers 32 consecutive packed words and each row has 16 words,
    // the linear word index is simply warp_id*32 + lane.
    const std::size_t idx = (warp_id << 5) + static_cast<std::size_t>(lane);

    const bool has_top    = (row != 0u);
    const bool has_bottom = (row != last_row);

    const word_t top = has_top    ? load_word(input, idx - kHalfStrideWords) : 0ULL;
    const word_t mid =                 load_word(input, idx);
    const word_t bot = has_bottom ? load_word(input, idx + kHalfStrideWords) : 0ULL;

    // Within each half-warp, neighboring words are already in neighboring lanes.
    word_t top_left_word = __shfl_up_sync(kFullMask, top, 1, kHalfWarpWidth);
    word_t mid_left_word = __shfl_up_sync(kFullMask, mid, 1, kHalfWarpWidth);
    word_t bot_left_word = __shfl_up_sync(kFullMask, bot, 1, kHalfWarpWidth);

    word_t top_right_word = __shfl_down_sync(kFullMask, top, 1, kHalfWarpWidth);
    word_t mid_right_word = __shfl_down_sync(kFullMask, mid, 1, kHalfWarpWidth);
    word_t bot_right_word = __shfl_down_sync(kFullMask, bot, 1, kHalfWarpWidth);

    // Entire row fits in the half-warp, so row edges map directly to zero.
    if (sublane == 0u) {
        top_left_word = 0ULL;
        mid_left_word = 0ULL;
        bot_left_word = 0ULL;
    }
    if (sublane == 15u) {
        top_right_word = 0ULL;
        mid_right_word = 0ULL;
        bot_right_word = 0ULL;
    }

    const word_t next = evolve_from_raw_neighbors(
        top_left_word, top, top_right_word,
        mid_left_word, mid, mid_right_word,
        bot_left_word, bot, bot_right_word);

    output[idx] = static_cast<std::uint64_t>(next);
}

// General kernel for rows that are at least 32 packed words wide.
//
// One warp processes one contiguous 32-word row segment.
// Most left/right neighboring words are exchanged with warp shuffles, so each thread
// normally performs only 3 global reads: row above, current row, row below.
// Only lane 0 and lane 31 need extra global reads at segment boundaries.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel_segments32(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int grid_dimensions,
    int segments_shift,
    unsigned int segments_mask,
    std::size_t total_warps)
{
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & (kWarpSize - 1u);
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kBlockThreads) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t warp_id = global_thread >> 5;

    if (warp_id >= total_warps) {
        return;
    }

    const unsigned rows     = static_cast<unsigned>(grid_dimensions);
    const unsigned last_row = rows - 1u;

    // Since segments_per_row is a power of two, row/segment decoding is shift+mask.
    const unsigned segment = static_cast<unsigned>(warp_id) & segments_mask;
    const unsigned row     = static_cast<unsigned>(warp_id >> segments_shift);

    // Each warp owns 32 consecutive packed words in memory, so the global word index is
    // again warp_id*32 + lane.
    const std::size_t idx = (warp_id << 5) + static_cast<std::size_t>(lane);

    // words_per_row = 32 * segments_per_row = 2^(segments_shift + 5)
    const std::size_t stride = std::size_t{1} << (segments_shift + 5);

    const bool has_top    = (row != 0u);
    const bool has_bottom = (row != last_row);

    const word_t top = has_top    ? load_word(input, idx - stride) : 0ULL;
    const word_t mid =                 load_word(input, idx);
    const word_t bot = has_bottom ? load_word(input, idx + stride) : 0ULL;

    // Pull neighboring words from adjacent lanes for the common case.
    word_t top_left_word = __shfl_up_sync(kFullMask, top, 1);
    word_t mid_left_word = __shfl_up_sync(kFullMask, mid, 1);
    word_t bot_left_word = __shfl_up_sync(kFullMask, bot, 1);

    word_t top_right_word = __shfl_down_sync(kFullMask, top, 1);
    word_t mid_right_word = __shfl_down_sync(kFullMask, mid, 1);
    word_t bot_right_word = __shfl_down_sync(kFullMask, bot, 1);

    // Lane 0 needs the real left neighboring word only when this segment is not the first
    // segment in the row. Otherwise the outside of the board is dead => zero.
    if (lane == 0u) {
        if (segment != 0u) {
            top_left_word = has_top    ? load_word(input, idx - stride - 1) : 0ULL;
            mid_left_word =               load_word(input, idx - 1);
            bot_left_word = has_bottom ? load_word(input, idx + stride - 1) : 0ULL;
        } else {
            top_left_word = 0ULL;
            mid_left_word = 0ULL;
            bot_left_word = 0ULL;
        }
    }

    // Lane 31 needs the real right neighboring word only when this segment is not the last
    // segment in the row. Otherwise the outside of the board is dead => zero.
    if (lane == (kWarpSize - 1u)) {
        if (segment != segments_mask) {
            top_right_word = has_top    ? load_word(input, idx - stride + 1) : 0ULL;
            mid_right_word =               load_word(input, idx + 1);
            bot_right_word = has_bottom ? load_word(input, idx + stride + 1) : 0ULL;
        } else {
            top_right_word = 0ULL;
            mid_right_word = 0ULL;
            bot_right_word = 0ULL;
        }
    }

    const word_t next = evolve_from_raw_neighbors(
        top_left_word, top, top_right_word,
        mid_left_word, mid, mid_right_word,
        bot_left_word, bot, bot_right_word);

    output[idx] = static_cast<std::uint64_t>(next);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees:
    // - square board
    // - power-of-two dimensions
    // - dimensions > 512
    // - input/output are device pointers allocated with cudaMalloc
    //
    // This function intentionally does not synchronize and does not check launch errors,
    // because the prompt states that synchronization is handled by the caller and only
    // simulation throughput matters.

    const int words_per_row = grid_dimensions >> 6;

    // The only legal sub-32-word case is 16 words/row (1024x1024 board).
    // Use the dedicated half-warp kernel so every lane remains useful.
    if (words_per_row == 16) {
        const std::size_t total_warps = static_cast<std::size_t>(grid_dimensions) >> 1;  // 2 rows/warp
        const unsigned int grid_x = static_cast<unsigned int>(
            (total_warps + static_cast<std::size_t>(kWarpsPerBlock) - 1u) /
            static_cast<std::size_t>(kWarpsPerBlock));

        game_of_life_kernel_halfwarp16<<<grid_x, kBlockThreads>>>(input, output, grid_dimensions, total_warps);
        return;
    }

    // For all larger boards, row width is 32, 64, 128, ...
    // segments_per_row = words_per_row / 32 = 2^segments_shift.
    int segments_shift = 0;
    for (int words = words_per_row; words > 32; words >>= 1) {
        ++segments_shift;
    }

    const unsigned int segments_mask = (1u << segments_shift) - 1u;
    const std::size_t total_warps = static_cast<std::size_t>(grid_dimensions) << segments_shift;
    const unsigned int grid_x = static_cast<unsigned int>(
        (total_warps + static_cast<std::size_t>(kWarpsPerBlock) - 1u) /
        static_cast<std::size_t>(kWarpsPerBlock));

    // 1D launch is deliberate:
    // it keeps the warp-to-segment mapping trivial and avoids 2D grid dimension limits
    // on very large boards.
    game_of_life_kernel_segments32<<<grid_x, kBlockThreads>>>(
        input, output, grid_dimensions, segments_shift, segments_mask, total_warps);
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
