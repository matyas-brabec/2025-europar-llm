#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/*
 * High-throughput bit-packed Conway step.
 *
 * Key choices:
 * - One thread updates exactly one 64-bit word = 64 cells.
 * - Horizontal reuse is handled with warp shuffles instead of shared memory.
 * - Vertical reuse is left to the hardware caches; no shared/texture memory is used.
 * - Neighbor counts are computed with bit-sliced full-adder logic, so each 64-bit
 *   operation updates 64 independent cells in parallel.
 *
 * The implementation intentionally exploits the stated constraints:
 * - grid_dimensions is a power of two and > 512
 * - therefore words_per_row = grid_dimensions / 64 is either 16 (the 1024x1024 case)
 *   or a multiple of 32
 *
 * Two exact kernels are used:
 * - 32x8 tiles for all boards with at least 32 words/row: one warp == one row tile
 * - 16x16 tiles for the 1024x1024 board: width-16 shuffles split each warp into
 *   two independent 16-lane row tiles
 *
 * This removes hot-path bounds checks entirely.
 */

namespace {

using u64 = unsigned long long;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "Expected 64-bit words.");

constexpr unsigned int kFullMask      = 0xFFFFFFFFu;
constexpr unsigned int kMaxExactGridY = 1u << 15;  // Largest power of two <= 65535.

constexpr int kSmallTileX = 16;
constexpr int kSmallTileY = 16;
constexpr int kLargeTileX = 32;
constexpr int kLargeTileY = 8;

static_assert(kSmallTileX * kSmallTileY == 256, "Small kernel must use 256 threads/block.");
static_assert(kLargeTileX * kLargeTileY == 256, "Large kernel must use 256 threads/block.");

// Per-bit full adder for three 64-bit bitmasks:
//   a + b + c = sum + 2*carry
// Every bit position is an independent 1-bit addition, so this computes 64
// cell-neighbor additions in parallel.
__device__ __forceinline__ void full_adder(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (axb & c);  // majority(a, b, c)
}

// Builds the contribution of one board row to the 8-neighbor count.
//
// INCLUDE_CENTER == true:
//   count west + center + east
//   used for the north and south rows
//
// INCLUDE_CENTER == false:
//   count west + east only
//   used for the current row because the cell itself is not its own neighbor
//
// The helper is called unconditionally by every thread. A ballot computes the
// exact participating lane mask for this invocation; this is required for the
// 16-word kernel, where a warp contains two independent 16-lane row tiles and
// only one of them may exist at the top/bottom edge.
template <int TileX, bool INCLUDE_CENTER>
__device__ __forceinline__ void accumulate_row(
    const std::uint64_t* __restrict__ input,
    std::size_t center_idx,
    u64 center_word,
    int lane_in_tile,
    bool row_exists,
    bool has_left,
    bool has_right,
    u64& sum,
    u64& carry)
{
    const unsigned int active_mask = __ballot_sync(kFullMask, row_exists);

    if (!row_exists) {
        sum = 0ULL;
        carry = 0ULL;
        return;
    }

    // Shuffles are issued unconditionally for the participating lanes.
    // Boundary lanes may receive undefined shuffle results when the source lane
    // would be outside the tile; those lanes never use that value and instead
    // fall back to a global load (or zero at the global board edge).
    const u64 left_from_lane  = __shfl_up_sync(active_mask, center_word, 1, TileX);
    const u64 right_from_lane = __shfl_down_sync(active_mask, center_word, 1, TileX);

    u64 left_word = 0ULL;
    u64 right_word = 0ULL;

    if (has_left) {
        left_word = (lane_in_tile > 0)
            ? left_from_lane
            : static_cast<u64>(input[center_idx - 1]);
    }

    if (has_right) {
        right_word = (lane_in_tile + 1 < TileX)
            ? right_from_lane
            : static_cast<u64>(input[center_idx + 1]);
    }

    // Bit 0 and bit 63 need cross-word handling:
    // - west neighbor of bit 0 comes from bit 63 of the word on the left
    // - east neighbor of bit 63 comes from bit 0 of the word on the right
    const u64 west = (center_word << 1) | (left_word >> 63);
    const u64 east = (center_word >> 1) | (right_word << 63);

    if (INCLUDE_CENTER) {
        full_adder(west, center_word, east, sum, carry);
    } else {
        sum   = west ^ east;
        carry = west & east;
    }
}

template <int TileX, int TileY>
__global__ __launch_bounds__(256)
void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    unsigned int grid_dimensions)
{
    static_assert((TileX & (TileX - 1)) == 0, "TileX must be a power of two.");
    static_assert((TileY & (TileY - 1)) == 0, "TileY must be a power of two.");
    static_assert(TileX * TileY == 256, "Kernel launch configuration must use 256 threads/block.");

    const unsigned int words_per_row = grid_dimensions >> 6;

    // Launch geometry is exact for the stated constraints:
    // - x tiles exactly cover the row
    // - y tiles exactly cover the board height
    // Thus no bounds checks are needed in the hot path.
    const unsigned int x = blockIdx.x * TileX + threadIdx.x;
    const unsigned int tile_y = blockIdx.y + blockIdx.z * gridDim.y;
    const unsigned int y = tile_y * TileY + threadIdx.y;

    const bool has_left  = (x != 0u);
    const bool has_right = (x + 1u < words_per_row);
    const bool has_north = (y != 0u);
    const bool has_south = (y + 1u < grid_dimensions);

    const int lane_in_tile = static_cast<int>(threadIdx.x);

    const std::size_t stride = static_cast<std::size_t>(words_per_row);
    const std::size_t idx = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
    const std::size_t north_idx = idx - stride;
    const std::size_t south_idx = idx + stride;

    u64 north = 0ULL;
    if (has_north) {
        north = static_cast<u64>(input[north_idx]);
    }

    const u64 cur = static_cast<u64>(input[idx]);

    u64 south = 0ULL;
    if (has_south) {
        south = static_cast<u64>(input[south_idx]);
    }

    u64 top_sum, top_carry;
    accumulate_row<TileX, true>(
        input, north_idx, north, lane_in_tile, has_north, has_left, has_right, top_sum, top_carry);

    u64 mid_sum, mid_carry;
    accumulate_row<TileX, false>(
        input, idx, cur, lane_in_tile, true, has_left, has_right, mid_sum, mid_carry);

    u64 bot_sum, bot_carry;
    accumulate_row<TileX, true>(
        input, south_idx, south, lane_in_tile, has_south, has_left, has_right, bot_sum, bot_carry);

    // Bit-sliced count tree:
    //
    //   top    = NW + N + NE     = top_sum + 2*top_carry
    //   middle = W + E           = mid_sum + 2*mid_carry
    //   bottom = SW + S + SE     = bot_sum + 2*bot_carry
    //
    // Combine low bits and high bits separately:
    //   low  = top_sum   + mid_sum   + bot_sum
    //   high = top_carry + mid_carry + bot_carry
    //
    // We only need:
    //   s0  = count bit 0
    //   q0  = count bit 1
    //   ge4 = (count >= 4)
    //
    // Then the Game of Life rule is:
    //   next = (count == 3) | (current & count == 2)
    //        = q0 & ~ge4 & (s0 | current)
    u64 s0, c0;
    full_adder(top_sum, mid_sum, bot_sum, s0, c0);

    u64 s1, c1;
    full_adder(top_carry, mid_carry, bot_carry, s1, c1);

    const u64 q0  = c0 ^ s1;
    const u64 ge4 = (c0 & s1) | c1;
    const u64 next = q0 & (s0 | cur) & ~ge4;

    output[idx] = static_cast<std::uint64_t>(next);
}

template <int TileX, int TileY>
inline void launch_exact_tiled_kernel(
    const std::uint64_t* input,
    std::uint64_t* output,
    unsigned int grid_dimensions)
{
    static_assert((TileX & (TileX - 1)) == 0, "TileX must be a power of two.");
    static_assert((TileY & (TileY - 1)) == 0, "TileY must be a power of two.");
    static_assert(TileX * TileY == 256, "Launch configuration must use 256 threads/block.");

    const unsigned int words_per_row = grid_dimensions >> 6;
    const unsigned int tiles_x = words_per_row / TileX;
    const unsigned int row_tiles = grid_dimensions / TileY;

    // row_tiles is a power of two. Choosing grid.y as either row_tiles itself or
    // 32768 (also a power of two) keeps grid_z integral, so the Y decomposition
    // is exact and the kernel needs no Y bounds checks.
    const unsigned int grid_y = (row_tiles > kMaxExactGridY) ? kMaxExactGridY : row_tiles;
    const unsigned int grid_z = row_tiles / grid_y;

    const dim3 block(TileX, TileY, 1);
    const dim3 grid(tiles_x, grid_y, grid_z);

    game_of_life_kernel<TileX, TileY><<<grid, block>>>(input, output, grid_dimensions);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    if (grid_dimensions <= 0) {
        return;
    }

    const unsigned int n = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = n >> 6;

    // The constraints guarantee:
    // - words_per_row == 16 for the smallest legal board (1024x1024)
    // - otherwise words_per_row is a multiple of 32
    //
    // The 16x16 kernel avoids half-empty 32-lane row tiles on the 1024x1024 case.
    if (words_per_row == 16u) {
        launch_exact_tiled_kernel<kSmallTileX, kSmallTileY>(input, output, n);
    } else {
        launch_exact_tiled_kernel<kLargeTileX, kLargeTileY>(input, output, n);
    }

    // Deliberately no synchronization or error-query call here.
    // The caller owns synchronization/error handling, and only step throughput
    // matters for this function.
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
