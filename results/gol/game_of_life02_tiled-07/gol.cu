#include <cstdint>
#include <cuda_runtime.h>

/*
 * Optimized CUDA Game of Life update for the problem's native bit-packed format.
 *
 * Storage model assumed by this implementation:
 *   - The board is partitioned into 8x8 tiles stored row-major.
 *   - Each 64-bit word stores one 8x8 tile in row-major bit order:
 *       bit ((local_y << 3) | local_x) == cell (local_x, local_y)
 *     with local_x in [0, 7] and local_y in [0, 7].
 *
 * Performance strategy:
 *   - One CUDA thread updates one 8x8 output tile, i.e. 64 cells at once.
 *   - The eight neighbor directions are formed as aligned 64-bit masks and then
 *     summed with bit-sliced boolean arithmetic ("SWAR" / bitboard style).
 *   - Horizontal tile neighbors are exchanged with warp shuffles so that the
 *     common path needs only three coalesced global loads per thread:
 *       north / center / south
 *   - Only lane 0 and lane 31 of each warp perform a few extra loads to bridge
 *     warp boundaries or inject zeroes at the outer board boundary.
 *   - No shared memory or texture memory is used, per the prompt.
 *
 * This warp mapping relies directly on the stated constraints:
 *   grid_dimensions is a power of two > 512
 *   => tiles_per_dim = grid_dimensions / 8 is a power of two >= 128
 *   => every tile row length is a multiple of 32
 *   => a warp of 32 consecutive thread/tile indices never crosses a tile-row boundary.
 */

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr u32 kFullWarpMask = 0xFFFFFFFFu;

static_assert((kThreadsPerBlock % kWarpSize) == 0,
              "kThreadsPerBlock must be a whole number of warps.");

// Bit masks for the standard row-major 8x8 bitboard layout.
constexpr u64 COL0_MASK     = 0x0101010101010101ULL;  // x == 0
constexpr u64 COL7_MASK     = 0x8080808080808080ULL;  // x == 7
constexpr u64 NOT_COL0_MASK = 0xFEFEFEFEFEFEFEFEULL;  // x != 0
constexpr u64 NOT_COL7_MASK = 0x7F7F7F7F7F7F7F7FULL;  // x != 7

// Individual corner bits inside a tile.
constexpr u64 BIT_R0_C0 = 0x0000000000000001ULL;
constexpr u64 BIT_R0_C7 = 0x0000000000000080ULL;
constexpr u64 BIT_R7_C0 = 0x0100000000000000ULL;
constexpr u64 BIT_R7_C7 = 0x8000000000000000ULL;

/*
 * Compute the next-state 8x8 tile from its 3x3 input-tile neighborhood.
 *
 * The implementation forms eight aligned neighbor masks:
 *   nbr_n, nbr_s, nbr_e, nbr_w, nbr_nw, nbr_ne, nbr_sw, nbr_se
 *
 * In each mask, bit i answers "is the corresponding neighbor of cell i alive?".
 * The eight 1-bit values per cell are then summed with a small carry-save / full-adder
 * tree, producing the 4-bit neighbor count in bit-sliced form across all 64 cells.
 *
 * Finally, the Life rule is applied in fully bit-parallel form:
 *   next = (count == 3) | (current & (count == 2))
 */
__device__ __forceinline__ u64 step_tile_8x8(
    const u64 tile_c,
    const u64 tile_n,
    const u64 tile_s,
    const u64 tile_w,
    const u64 tile_e,
    const u64 tile_nw,
    const u64 tile_ne,
    const u64 tile_sw,
    const u64 tile_se)
{
    // North / south aligned to destination cell positions.
    const u64 nbr_n = (tile_c << 8) | (tile_n >> 56);
    const u64 nbr_s = (tile_c >> 8) | (tile_s << 56);

    // Pairwise add N + S.
    const u64 p01 = nbr_n ^ nbr_s;
    const u64 c01 = nbr_n & nbr_s;

    // East / west aligned to destination cell positions.
    u64 x0 = ((tile_c & NOT_COL0_MASK) >> 1) | ((tile_e & COL0_MASK) << 7);
    u64 x1 = ((tile_c & NOT_COL7_MASK) << 1) | ((tile_w & COL7_MASK) >> 7);

    // Pairwise add E + W.
    const u64 p23 = x0 ^ x1;
    const u64 c23 = x0 & x1;

    // First four directions -> partial count a2:a1:a0 (0..4 per cell).
    u64 carry = p01 & p23;
    u64 t = c01 ^ c23;
    const u64 a0 = p01 ^ p23;
    const u64 a1 = t ^ carry;
    const u64 a2 = (c01 & c23) | (carry & t);

    // Diagonals reuse the already-aligned north/south masks:
    //   NW = west-shifted north + west-edge injection + NW corner injection
    //   NE = east-shifted north + east-edge injection + NE corner injection
    x0 = ((nbr_n & NOT_COL7_MASK) << 1)
       | ((tile_w & COL7_MASK) << 1)
       | ((tile_nw & BIT_R7_C7) >> 63);

    x1 = ((nbr_n & NOT_COL0_MASK) >> 1)
       | ((tile_e & COL0_MASK) << 15)
       | ((tile_ne & BIT_R7_C0) >> 49);

    const u64 p45 = x0 ^ x1;
    const u64 c45 = x0 & x1;

    x0 = ((nbr_s & NOT_COL7_MASK) << 1)
       | ((tile_w & COL7_MASK) >> 15)
       | ((tile_sw & BIT_R0_C7) << 49);

    x1 = ((nbr_s & NOT_COL0_MASK) >> 1)
       | ((tile_e & COL0_MASK) >> 1)
       | ((tile_se & BIT_R0_C0) << 63);

    const u64 p67 = x0 ^ x1;
    const u64 c67 = x0 & x1;

    // Last four directions -> partial count b2:b1:b0 (0..4 per cell).
    carry = p45 & p67;
    t = c45 ^ c67;
    const u64 b0 = p45 ^ p67;
    const u64 b1 = t ^ carry;
    const u64 b2 = (c45 & c67) | (carry & t);

    // Add the two 3-bit partial counts -> full 4-bit count count3:count2:count1:count0.
    carry = a0 & b0;
    t = a1 ^ b1;
    const u64 count0 = a0 ^ b0;
    const u64 count1 = t ^ carry;
    const u64 carry1 = (a1 & b1) | (carry & t);

    t = a2 ^ b2;
    const u64 count2 = t ^ carry1;
    const u64 count3 = (a2 & b2) | (carry1 & t);

    // Bit-sliced Life rule:
    //   count == 2 or 3  <=>  count1 == 1 and no higher count bits are set
    //   count == 3       <=>  count0 == 1
    //   count == 2       <=>  count0 == 0
    //
    // Therefore:
    //   next = (count == 3) | (current & (count == 2))
    //        = count1 & ~(count2 | count3) & (count0 | current)
    return count1 & ~(count2 | count3) & (count0 | tile_c);
}

__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const u32 tiles_per_dim,
    const u32 tiles_mask,
    const u64 south_limit)
{
    const u64 idx =
        static_cast<u64>(blockIdx.x) * static_cast<u64>(kThreadsPerBlock) +
        static_cast<u64>(threadIdx.x);

    const u64 stride = static_cast<u64>(tiles_per_dim);
    const u32 lane = static_cast<u32>(threadIdx.x) & static_cast<u32>(kWarpSize - 1);

    // Out-of-grid tiles are treated as all-zero (dead).
    const bool has_n = idx >= stride;
    const bool has_s = idx < south_limit;

    const u64 tile_c = input[idx];
    const u64 tile_n = has_n ? input[idx - stride] : 0ULL;
    const u64 tile_s = has_s ? input[idx + stride] : 0ULL;

    // Fast horizontal neighbor exchange through the warp.
    // The problem constraints guarantee that a warp never crosses a tile-row boundary.
    u64 tile_w  = __shfl_up_sync  (kFullWarpMask, tile_c, 1, kWarpSize);
    u64 tile_e  = __shfl_down_sync(kFullWarpMask, tile_c, 1, kWarpSize);
    u64 tile_nw = __shfl_up_sync  (kFullWarpMask, tile_n, 1, kWarpSize);
    u64 tile_ne = __shfl_down_sync(kFullWarpMask, tile_n, 1, kWarpSize);
    u64 tile_sw = __shfl_up_sync  (kFullWarpMask, tile_s, 1, kWarpSize);
    u64 tile_se = __shfl_down_sync(kFullWarpMask, tile_s, 1, kWarpSize);

    // Only warp-edge lanes need true cross-warp loads.
    if (lane == 0u) {
        const bool has_w = (static_cast<u32>(idx) & tiles_mask) != 0u;
        if (has_w) {
            const u64 idx_left = idx - 1ULL;
            tile_w  = input[idx_left];
            tile_nw = has_n ? input[idx_left - stride] : 0ULL;
            tile_sw = has_s ? input[idx_left + stride] : 0ULL;
        } else {
            tile_w = 0ULL;
            tile_nw = 0ULL;
            tile_sw = 0ULL;
        }
    }

    if (lane == static_cast<u32>(kWarpSize - 1)) {
        const bool has_e = (static_cast<u32>(idx) & tiles_mask) != tiles_mask;
        if (has_e) {
            const u64 idx_right = idx + 1ULL;
            tile_e  = input[idx_right];
            tile_ne = has_n ? input[idx_right - stride] : 0ULL;
            tile_se = has_s ? input[idx_right + stride] : 0ULL;
        } else {
            tile_e = 0ULL;
            tile_ne = 0ULL;
            tile_se = 0ULL;
        }
    }

    output[idx] = step_tile_8x8(
        tile_c,
        tile_n, tile_s,
        tile_w, tile_e,
        tile_nw, tile_ne,
        tile_sw, tile_se);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Enqueue exactly one thread per 8x8 tile on the default stream.
    // The caller is responsible for any host-device synchronization.
    const u32 tiles_per_dim = static_cast<u32>(grid_dimensions) >> 3;
    const u32 tiles_mask = tiles_per_dim - 1u;
    const u64 total_tiles = static_cast<u64>(tiles_per_dim) * static_cast<u64>(tiles_per_dim);
    const u64 south_limit = total_tiles - static_cast<u64>(tiles_per_dim);

    // Under the problem constraints, total_tiles is always a multiple of 256,
    // so we launch an exact number of threads and omit a bounds check in the kernel.
    const u32 blocks = static_cast<u32>(total_tiles / static_cast<u64>(kThreadsPerBlock));

    game_of_life_kernel<<<blocks, kThreadsPerBlock>>>(
        input,
        output,
        tiles_per_dim,
        tiles_mask,
        south_limit);
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
// MEMORY_LAYOUT: TILES
