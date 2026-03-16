#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Conway's Game of Life, one step, for a square grid bit-packed as 8x8 tiles.
//
// Assumed layout (the only layout directly implied by "one uint64_t per 8x8 tile"):
//   - Tiles are stored row-major in a (grid_dimensions / 8) x (grid_dimensions / 8) array.
//   - Inside each 64-bit tile, bit (local_y * 8 + local_x) stores cell (local_x, local_y),
//     where local_x increases to the right and local_y increases downward.
//   - Therefore each tile row is exactly one byte.
//
// Performance strategy:
//   - One thread computes one output tile (64 cells).
//   - blockDim.x == 32, so each warp spans 32 consecutive tiles along x.
//     Horizontal neighbor tiles are therefore usually obtained with warp shuffles instead of
//     redundant global loads; only lane 0 / lane 31 need fallback loads because shuffles do
//     not cross warp boundaries.
//   - No shared memory is used: the problem statement explicitly says it is unnecessary here,
//     and warp shuffles already remove most horizontal load redundancy while keeping the code
//     simple and fast on modern data-center GPUs.
//   - Within a tile, all 64 cells are updated in parallel with bitwise logic. Neighbor counts
//     are accumulated as three bit-planes (ones/twos/fours) modulo 8. Count 8 wraps to 0,
//     which is harmless because Conway's rule only cares about neighbor counts 2 and 3.

namespace {

using word_t = std::uint64_t;
static_assert(sizeof(word_t) == sizeof(unsigned long long), "Expected 64-bit words.");

constexpr int kTileEdge = 8;
constexpr int kBlockX   = 32;  // exactly one warp in x
constexpr int kBlockY   = 8;
constexpr int kBlockThreads = kBlockX * kBlockY;
static_assert(kBlockX == 32, "This kernel relies on blockDim.x being exactly one warp.");

constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

// Bit masks for the 8x8 tile encoding described above.
constexpr word_t kCol0 = 0x0101010101010101ULL;
constexpr word_t kCol7 = 0x8080808080808080ULL;
constexpr word_t kNotCol0 = 0xFEFEFEFEFEFEFEFEULL;
constexpr word_t kNotCol7 = 0x7F7F7F7F7F7F7F7FULL;
constexpr word_t kRow0 = 0x00000000000000FFULL;

// Low-byte masks used after extracting a tile row into the low byte.
constexpr word_t kRowByteNoCol0 = 0x00000000000000FEULL;  // x = 1..7
constexpr word_t kRowByteNoCol7 = 0x000000000000007FULL;  // x = 0..6

// Individual bit masks used for the diagonal corner fills.
constexpr word_t kBit0  = 0x0000000000000001ULL;
constexpr word_t kBit7  = 0x0000000000000080ULL;
constexpr word_t kBit56 = 0x0100000000000000ULL;

__device__ __forceinline__ word_t shfl_up_u64(const unsigned mask, const word_t value, const unsigned delta) {
    return static_cast<word_t>(
        __shfl_up_sync(mask, static_cast<unsigned long long>(value), static_cast<int>(delta)));
}

__device__ __forceinline__ word_t shfl_down_u64(const unsigned mask, const word_t value, const unsigned delta) {
    return static_cast<word_t>(
        __shfl_down_sync(mask, static_cast<unsigned long long>(value), static_cast<int>(delta)));
}

__device__ __forceinline__ void add_neighbor_mod8(
    const word_t neighbor,
    word_t& ones,
    word_t& twos,
    word_t& fours)
{
    // Bit-sliced increment of a 3-bit counter.
    //
    // Every bit position in "neighbor" is one 1-bit addend for the corresponding cell.
    // The counter is kept modulo 8:
    //   count 0..7 are represented exactly by (ones, twos, fours),
    //   count 8 wraps to 0 (overflow discarded).
    //
    // This is valid for Game of Life because the rule only distinguishes:
    //   - exactly 2 neighbors
    //   - exactly 3 neighbors
    //   - everything else
    // and count 8 belongs to "everything else".
    word_t carry = ones & neighbor;
    ones ^= neighbor;

    word_t carry2 = twos & carry;
    twos ^= carry;

    fours ^= carry2;
}

__device__ __forceinline__ word_t evolve_tile(
    const word_t nw, const word_t n, const word_t ne,
    const word_t w,  const word_t c, const word_t e,
    const word_t sw, const word_t s, const word_t se)
{
    // Pre-extract/reuse the pieces that appear in multiple directions.
    const word_t c_no_col7 = c & kNotCol7;
    const word_t c_no_col0 = c & kNotCol0;
    const word_t w_col7    = w & kCol7;
    const word_t e_col0    = e & kCol0;
    const word_t n_row7    = n >> 56;   // north tile's last row, moved into the low byte
    const word_t s_row0    = s & kRow0; // south tile's first row, already in the low byte

    word_t ones  = 0;
    word_t twos  = 0;
    word_t fours = 0;
    word_t neighbor;

    // Cardinal directions.
    neighbor = (c << 8) | n_row7;                      // north
    add_neighbor_mod8(neighbor, ones, twos, fours);

    neighbor = (c >> 8) | (s_row0 << 56);             // south
    add_neighbor_mod8(neighbor, ones, twos, fours);

    neighbor = (c_no_col7 << 1) | (w_col7 >> 7);      // west
    add_neighbor_mod8(neighbor, ones, twos, fours);

    neighbor = (c_no_col0 >> 1) | (e_col0 << 7);      // east
    add_neighbor_mod8(neighbor, ones, twos, fours);

    // Diagonals.
    neighbor = (c_no_col7 << 9)                        // current tile interior contribution
             | ((n_row7 & kRowByteNoCol7) << 1)       // north tile, row 7, x = 0..6 -> x = 1..7
             | (w_col7 << 1)                          // west tile, col 7, y = 0..6 -> y = 1..7
             | (nw >> 63);                            // north-west corner tile, bit 63 -> bit 0
    add_neighbor_mod8(neighbor, ones, twos, fours);   // north-west

    neighbor = (c_no_col0 << 7)
             | ((n_row7 & kRowByteNoCol0) >> 1)       // north tile, row 7, x = 1..7 -> x = 0..6
             | (e_col0 << 15)                         // east tile, col 0, y = 0..6 -> y = 1..7
             | ((ne & kBit56) >> 49);                 // north-east corner tile, bit 56 -> bit 7
    add_neighbor_mod8(neighbor, ones, twos, fours);   // north-east

    neighbor = (c_no_col7 >> 7)
             | ((s_row0 & kRowByteNoCol7) << 57)      // south tile, row 0, x = 0..6 -> x = 1..7
             | (w_col7 >> 15)                         // west tile, col 7, y = 1..7 -> y = 0..6
             | ((sw & kBit7) << 49);                  // south-west corner tile, bit 7 -> bit 56
    add_neighbor_mod8(neighbor, ones, twos, fours);   // south-west

    neighbor = (c_no_col0 >> 9)
             | ((s_row0 & kRowByteNoCol0) << 55)      // south tile, row 0, x = 1..7 -> x = 0..6
             | (e_col0 >> 1)                          // east tile, col 0, y = 1..7 -> y = 0..6
             | ((se & kBit0) << 63);                  // south-east corner tile, bit 0 -> bit 63
    add_neighbor_mod8(neighbor, ones, twos, fours);   // south-east

    // Conway rule:
    //   alive next step iff neighbor_count == 3, or (currently alive and neighbor_count == 2)
    //
    // With the bit-sliced counter:
    //   count == 2  -> ones=0, twos=1, fours=0
    //   count == 3  -> ones=1, twos=1, fours=0
    //
    // So "neighbor_count is 2 or 3" is: twos & ~fours
    // and then (count == 3) OR (current cell alive) becomes: ones | c
    const word_t life_mask = twos & ~fours;
    return life_mask & (ones | c);
}

__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(
    const word_t* __restrict__ input,
    word_t* __restrict__ output,
    const int tiles_per_side)
{
    // Because blockDim.x == 32 and tiles_per_side is a power of two >= 64, each warp covers
    // exactly one contiguous 32-tile segment in x, with no partial x-warps.
    const int tx = static_cast<int>(blockIdx.x) * kBlockX + static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(blockIdx.y) * kBlockY + static_cast<int>(threadIdx.y);

    // This branch is warp-uniform because threadIdx.x spans the whole warp while threadIdx.y
    // is constant within a warp when blockDim.x == 32.
    if (ty >= tiles_per_side) {
        return;
    }

    const std::size_t stride = static_cast<std::size_t>(tiles_per_side);
    const std::size_t idx = static_cast<std::size_t>(ty) * stride + static_cast<std::size_t>(tx);

    // Missing neighboring tiles are treated as zero, which exactly implements "cells outside
    // the grid are dead" while still preserving all intra-tile neighbors.
    const bool has_n = (ty > 0);
    const bool has_s = (ty + 1 < tiles_per_side);

    const word_t c = input[idx];
    const word_t n = has_n ? input[idx - stride] : 0;
    const word_t s = has_s ? input[idx + stride] : 0;

    // Horizontal neighbors usually come from warp shuffles of the current/adjacent rows.
    // lane 0 / lane 31 are fixed up below because shuffles do not cross warp boundaries.
    word_t w  = shfl_up_u64(kFullWarpMask, c, 1);
    word_t e  = shfl_down_u64(kFullWarpMask, c, 1);
    word_t nw = shfl_up_u64(kFullWarpMask, n, 1);
    word_t ne = shfl_down_u64(kFullWarpMask, n, 1);
    word_t sw = shfl_up_u64(kFullWarpMask, s, 1);
    word_t se = shfl_down_u64(kFullWarpMask, s, 1);

    const int lane = static_cast<int>(threadIdx.x);

    if (lane == 0) {
        if (tx > 0) {
            w  = input[idx - 1];
            nw = has_n ? input[idx - stride - 1] : 0;
            sw = has_s ? input[idx + stride - 1] : 0;
        } else {
            w = nw = sw = 0;
        }
    }

    if (lane == 31) {
        if (tx + 1 < tiles_per_side) {
            e  = input[idx + 1];
            ne = has_n ? input[idx - stride + 1] : 0;
            se = has_s ? input[idx + stride + 1] : 0;
        } else {
            e = ne = se = 0;
        }
    }

    output[idx] = evolve_tile(nw, n, ne, w, c, e, sw, s, se);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees valid device pointers allocated by cudaMalloc and handles all
    // synchronization externally. This function intentionally only enqueues the kernel.
    //
    // Standard double-buffered Game of Life is assumed: input and output must not overlap.
    if (grid_dimensions <= 0) {
        return;
    }

    const int tiles_per_side = grid_dimensions / kTileEdge;
    if (tiles_per_side <= 0) {
        return;
    }

    // Problem constraints guarantee:
    //   - grid_dimensions is a power of two and > 512
    //   - therefore tiles_per_side is also a power of two and >= 64
    //   - therefore tiles_per_side is divisible by 32, so grid.x is exact
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        static_cast<unsigned>(tiles_per_side / kBlockX),
        static_cast<unsigned>((tiles_per_side + kBlockY - 1) / kBlockY));

    game_of_life_kernel<<<grid, block>>>(input, output, tiles_per_side);
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
