#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Bit-packed Game of Life on a square grid.
//
// Encoding assumption:
//   - Each 64-bit word stores 64 consecutive cells from one row.
//   - Bit i corresponds to the cell at x = word_base + i (LSB -> increasing x).
//
// Kernel mapping:
//   - One warp handles one horizontal row segment of up to 32 packed words.
//   - Each lane owns one 64-bit word = 64 cells.
//   - Only three center words are fetched from global memory per lane: row-1, row, row+1.
//   - Left/right neighbor words are obtained with warp shuffles; only segment-edge lanes
//     perform one extra global load when a neighbor word sits in an adjacent segment.
//   - Neighbor population counts are computed in bit-sliced form with a carry-save-adder tree.
//     This avoids unpacking bits or looping over individual cells.
//
// Shared memory is intentionally not used: on recent data-center GPUs, shuffle-based horizontal
// exchange plus normal caching is simpler and fast for this access pattern.

using u64 = std::uint64_t;

static_assert(sizeof(u64) == 8, "Expected 64-bit packed words.");

constexpr int kCellsPerWordShift = 6;   // 64 cells per packed word.
constexpr int kWarpSize          = 32;  // NVIDIA warp size.
constexpr int kWarpsPerBlock     = 8;   // 256 threads/block.
constexpr int kBlockThreads      = kWarpSize * kWarpsPerBlock;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

// 3:2 carry-save adder on 64 independent bit lanes.
// For each bit position:
//   a + b + c = sum + 2 * carry
static __device__ __forceinline__
void csa3(const u64 a, const u64 b, const u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (c & ab_xor);
}

__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         const int grid_dimensions) {
    const int words_per_row = grid_dimensions >> kCellsPerWordShift;

    const int lane          = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int row           = static_cast<int>(blockIdx.y) * kWarpsPerBlock + warp_in_block;

    if (row >= grid_dimensions) {
        return;
    }

    const int segment_word   = static_cast<int>(blockIdx.x) * kWarpSize;
    const int remaining      = words_per_row - segment_word;
    const int tile_words     = (remaining >= kWarpSize) ? kWarpSize : remaining;
    const bool active        = lane < tile_words;
    const unsigned lane_mask = __ballot_sync(kFullWarpMask, active);

    if (!active) {
        return;
    }

    const bool first_lane        = (lane == 0);
    const bool last_active_lane  = (lane == tile_words - 1);
    const bool has_segment_left  = (segment_word > 0);
    const bool has_segment_right = (segment_word + tile_words < words_per_row);

    const std::size_t row_base =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row);

    const u64* const row_seg = input  + row_base + static_cast<std::size_t>(segment_word);
    u64* const       out_seg = output + row_base + static_cast<std::size_t>(segment_word);

    // Outside-grid cells are defined as dead, so missing rows/words are injected as zero.
    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < grid_dimensions);

    const u64* const top_seg = has_top    ? (row_seg - words_per_row) : nullptr;
    const u64* const bot_seg = has_bottom ? (row_seg + words_per_row) : nullptr;

    const u64 top_center = has_top    ? top_seg[lane] : 0ULL;
    const u64 mid_center = row_seg[lane];
    const u64 bot_center = has_bottom ? bot_seg[lane] : 0ULL;

    // First CSA group: north, south, west
    u64 sum0, carry0;
    {
        const u64 mid_left =
            first_lane
                ? (has_segment_left ? row_seg[-1] : 0ULL)
                : __shfl_up_sync(lane_mask, mid_center, 1);

        const u64 west = (mid_center << 1) | (mid_left >> 63);
        csa3(top_center, bot_center, west, sum0, carry0);
    }

    // Second CSA group: east, northwest, northeast
    u64 sum1, carry1;
    {
        const u64 mid_right =
            last_active_lane
                ? (has_segment_right ? row_seg[tile_words] : 0ULL)
                : __shfl_down_sync(lane_mask, mid_center, 1);

        const u64 east = (mid_center >> 1) | (mid_right << 63);

        const u64 top_left =
            has_top
                ? (first_lane
                       ? (has_segment_left ? top_seg[-1] : 0ULL)
                       : __shfl_up_sync(lane_mask, top_center, 1))
                : 0ULL;

        const u64 northwest = (top_center << 1) | (top_left >> 63);

        const u64 top_right =
            has_top
                ? (last_active_lane
                       ? (has_segment_right ? top_seg[tile_words] : 0ULL)
                       : __shfl_down_sync(lane_mask, top_center, 1))
                : 0ULL;

        const u64 northeast = (top_center >> 1) | (top_right << 63);

        csa3(east, northwest, northeast, sum1, carry1);
    }

    // Remaining two neighbors: southwest and southeast
    u64 sum_sw_se, carry_sw_se;
    {
        const u64 bot_left =
            has_bottom
                ? (first_lane
                       ? (has_segment_left ? bot_seg[-1] : 0ULL)
                       : __shfl_up_sync(lane_mask, bot_center, 1))
                : 0ULL;

        const u64 southwest = (bot_center << 1) | (bot_left >> 63);

        const u64 bot_right =
            has_bottom
                ? (last_active_lane
                       ? (has_segment_right ? bot_seg[tile_words] : 0ULL)
                       : __shfl_down_sync(lane_mask, bot_center, 1))
                : 0ULL;

        const u64 southeast = (bot_center >> 1) | (bot_right << 63);

        sum_sw_se   = southwest ^ southeast;
        carry_sw_se = southwest & southeast;
    }

    // Add the 8 one-bit neighbor boards:
    //   count = ones_bit + 2*twos_bit + higher_bits
    //
    // We do not materialize the full 4-bit count because the Life rule only needs:
    //   count == 2  -> survive if currently alive
    //   count == 3  -> always alive
    //
    // That condition is equivalent to:
    //   twos_bit == 1, no 4/8 bits set, and (ones_bit == 1 || current_cell == 1)
    u64 ones_bit, carry_to_twos;
    csa3(sum0, sum1, sum_sw_se, ones_bit, carry_to_twos);

    u64 twos_partial, fours_partial;
    csa3(carry0, carry1, carry_sw_se, twos_partial, fours_partial);

    const u64 twos_bit         = twos_partial ^ carry_to_twos;
    const u64 high_bits_nonzero = fours_partial | (twos_partial & carry_to_twos);

    out_seg[lane] = twos_bit & ~high_bits_nonzero & (ones_bit | mid_center);
}

} // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The caller handles synchronization and any desired error policy.
    // This function only enqueues one step asynchronously.
    const int words_per_row = grid_dimensions >> kCellsPerWordShift;

    const dim3 block(kBlockThreads, 1, 1);
    const dim3 grid((words_per_row + kWarpSize - 1) / kWarpSize,
                    (grid_dimensions + kWarpsPerBlock - 1) / kWarpsPerBlock,
                    1);

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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
