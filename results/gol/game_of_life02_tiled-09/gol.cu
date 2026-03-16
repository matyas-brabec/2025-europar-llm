#include <cstdint>
#include <cuda_runtime.h>

// CUDA Game of Life step for an 8x8-tiled, bit-packed board.
//
// Encoding assumed by this implementation:
//   - Tiles are stored row-major in memory.
//   - Inside each 64-bit tile, bit (local_y * 8 + local_x) is the cell at
//     tile-local coordinates (local_x, local_y).
//   - Bit 0 is the tile-local top-left cell; increasing the bit index moves
//     right within a row, then down to the next row.
//
// This bit order is what makes the shift/mask logic below correct. If a caller
// uses a different intra-tile bit order, only the masks/shifts in evolve_tile()
// need to change; the kernel launch structure can stay the same.
//
// Performance strategy:
//   - One thread updates one 8x8 tile (64 cells at once).
//   - Each warp spans 32 consecutive tiles in one tile row.
//   - Every thread only loads its north/center/south tiles directly.
//   - West/east tiles and corner tiles are obtained mostly through warp shuffles;
//     only lane 0 and lane 31 issue the extra cross-warp loads.
//   - All 64 cells of a tile are evolved in parallel using SWAR / bit-sliced
//     arithmetic. There are no per-cell loops and no shared memory.
//
// The prompt guarantees:
//   - grid_dimensions is a power of 2
//   - grid_dimensions > 512
// Therefore:
//   - tiles_per_row = grid_dimensions / 8 is a power of 2 >= 64
//   - tiles_per_row is divisible by both 32 and 8
// So a 32x8 block tiles the domain exactly: no tail predicates, no partial warps.

namespace {

using u64 = std::uint64_t;

constexpr int kBlockXLog2 = 5;               // 32 tiles per warp in x
constexpr int kBlockYLog2 = 3;               // 8 warps per CTA in y
constexpr int kBlockX     = 1 << kBlockXLog2;
constexpr int kBlockY     = 1 << kBlockYLog2;
constexpr int kBlockThreads = kBlockX * kBlockY;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert(kBlockX == 32, "This kernel maps exactly one warp to one tile row.");

// Tile masks for the row-major, bit0=top-left 8x8 packing described above.
constexpr u64 kCol0Mask   = 0x0101010101010101ULL;
constexpr u64 kCol7Mask   = 0x8080808080808080ULL;
constexpr u64 kNoCol0Mask = 0xFEFEFEFEFEFEFEFEULL;
constexpr u64 kNoCol7Mask = 0x7F7F7F7F7F7F7F7FULL;

// Low-byte masks. These are used after extracting a tile edge row into the low byte.
constexpr u64 kRow0Mask   = 0x00000000000000FFULL;
constexpr u64 kRow0NoCol0 = 0x00000000000000FEULL;
constexpr u64 kRow0NoCol7 = 0x000000000000007FULL;
constexpr u64 kRow0Col7   = 0x0000000000000080ULL;

// High-position corner mask used directly on an unshifted neighbor tile.
constexpr u64 kRow7Col0   = 0x0100000000000000ULL;

__device__ __forceinline__ u64 ro_load(const u64* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ __forceinline__ u64 majority(u64 a, u64 b, u64 c) {
    // Per-bit majority: the carry of adding three 1-bit values.
    return (a & b) | (a & c) | (b & c);
}

__device__ __forceinline__ u64 evolve_tile(
    u64 center,
    u64 north,
    u64 south,
    u64 west,
    u64 east,
    u64 northwest,
    u64 northeast,
    u64 southwest,
    u64 southeast)
{
    // Reused masked versions. Precomputing these saves several repeated ANDs.
    const u64 center_no_col0 = center & kNoCol0Mask;
    const u64 center_no_col7 = center & kNoCol7Mask;
    const u64 east_col0      = east   & kCol0Mask;
    const u64 west_col7      = west   & kCol7Mask;

    // Extract the halo rows that are needed multiple times.
    const u64 north_row7 = north >> 56;          // north tile's bottom row -> low byte
    const u64 south_row0 = south & kRow0Mask;    // south tile's top row    -> low byte

    // Build the eight directional bitboards.
    // For each direction, a set bit means "the neighbor in that direction is alive"
    // for the corresponding cell position inside the current 8x8 tile.
    const u64 dir_n  = (center << 8) | north_row7;
    const u64 dir_s  = (center >> 8) | (south_row0 << 56);
    const u64 dir_e  = (center_no_col0 >> 1) | (east_col0 << 7);
    const u64 dir_w  = (center_no_col7 << 1) | (west_col7 >> 7);

    const u64 dir_ne = (center_no_col0 << 7)
                     | (north_row7 >> 1)
                     | (east_col0 << 15)
                     | ((northeast & kRow7Col0) >> 49);

    const u64 dir_nw = (center_no_col7 << 9)
                     | ((north_row7 & kRow0NoCol7) << 1)
                     | (west_col7 << 1)
                     | (northwest >> 63);

    const u64 dir_se = (center_no_col0 >> 9)
                     | ((south_row0 & kRow0NoCol0) << 55)
                     | (east_col0 >> 1)
                     | (southeast << 63);

    const u64 dir_sw = (center_no_col7 >> 7)
                     | ((south_row0 & kRow0NoCol7) << 57)
                     | (west_col7 >> 15)
                     | ((southwest & kRow0Col7) << 49);

    // Compress N/S/E/W into a 3-bit per-cell count (count_a2:count_a1:count_a0).
    const u64 ns_xor  = dir_n ^ dir_s;
    const u64 ns_and  = dir_n & dir_s;
    const u64 ew_xor  = dir_e ^ dir_w;
    const u64 ew_and  = dir_e & dir_w;

    const u64 count_a0 = ns_xor ^ ew_xor;
    const u64 carry_a  = ns_xor & ew_xor;
    const u64 count_a1 = ns_and ^ ew_and ^ carry_a;
    const u64 count_a2 = majority(ns_and, ew_and, carry_a);

    // Compress NE/NW/SE/SW into another 3-bit per-cell count (count_b2:count_b1:count_b0).
    const u64 nwne_xor = dir_ne ^ dir_nw;
    const u64 nwne_and = dir_ne & dir_nw;
    const u64 swse_xor = dir_se ^ dir_sw;
    const u64 swse_and = dir_se & dir_sw;

    const u64 count_b0 = nwne_xor ^ swse_xor;
    const u64 carry_b  = nwne_xor & swse_xor;
    const u64 count_b1 = nwne_and ^ swse_and ^ carry_b;
    const u64 count_b2 = majority(nwne_and, swse_and, carry_b);

    // Add the two 3-bit counters to obtain the full 0..8 neighbor count.
    const u64 count_bit0 = count_a0 ^ count_b0;
    const u64 carry_1    = count_a0 & count_b0;

    const u64 sum_1      = count_a1 ^ count_b1;
    const u64 count_bit1 = sum_1 ^ carry_1;
    const u64 carry_2    = majority(count_a1, count_b1, carry_1);

    const u64 sum_2      = count_a2 ^ count_b2;
    const u64 count_bit2 = sum_2 ^ carry_2;
    const u64 count_bit3 = majority(count_a2, count_b2, carry_2);

    // Conway rule in binary form:
    //   next = (count == 3) | (alive & (count == 2))
    //
    // For counts 0..8:
    //   count in {2,3} iff count_bit1 = 1 and count_bit2 = count_bit3 = 0.
    //   count_bit0 distinguishes 2 from 3.
    //
    // Therefore:
    //   next = !count_bit2 & !count_bit3 & count_bit1 & (alive | count_bit0)
    return (~(count_bit2 | count_bit3)) & count_bit1 & (center | count_bit0);
}

__global__ __launch_bounds__(kBlockThreads, 2)
void game_of_life_step_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int tiles_per_row)
{
    // Because blockDim.x == 32, each warp corresponds to a single tile row
    // inside the CTA, which makes left/right neighbor exchange a plain shuffle.
    const unsigned lane = threadIdx.x;
    const int x = (static_cast<int>(blockIdx.x) << kBlockXLog2) + static_cast<int>(lane);
    const int y = (static_cast<int>(blockIdx.y) << kBlockYLog2) + static_cast<int>(threadIdx.y);
    const int idx = y * tiles_per_row + x;

    // Grid boundary conditions: everything outside the board is dead.
    const bool has_n = (y != 0);
    const bool has_s = (y + 1 != tiles_per_row);
    const bool has_w = (x != 0);
    const bool has_e = (x + 1 != tiles_per_row);

    const int north_idx = idx - tiles_per_row;
    const int south_idx = idx + tiles_per_row;

    // Mandatory loads: north / center / south tiles for this x position.
    u64 north  = has_n ? ro_load(input + north_idx) : 0ULL;
    u64 center = ro_load(input + idx);
    u64 south  = has_s ? ro_load(input + south_idx) : 0ULL;

    // Neighbor tiles inside the same warp are exchanged through shuffles.
    u64 north_w = __shfl_up_sync(kFullWarpMask, north, 1);
    u64 center_w = __shfl_up_sync(kFullWarpMask, center, 1);
    u64 south_w = __shfl_up_sync(kFullWarpMask, south, 1);

    u64 north_e = __shfl_down_sync(kFullWarpMask, north, 1);
    u64 center_e = __shfl_down_sync(kFullWarpMask, center, 1);
    u64 south_e = __shfl_down_sync(kFullWarpMask, south, 1);

    // Warp-edge lanes need explicit cross-warp loads (or zeros at the true grid edge).
    if (lane == 0) {
        north_w  = (has_w && has_n) ? ro_load(input + north_idx - 1) : 0ULL;
        center_w = has_w            ? ro_load(input + idx - 1)       : 0ULL;
        south_w  = (has_w && has_s) ? ro_load(input + south_idx - 1) : 0ULL;
    }

    if (lane == kBlockX - 1) {
        north_e  = (has_e && has_n) ? ro_load(input + north_idx + 1) : 0ULL;
        center_e = has_e            ? ro_load(input + idx + 1)       : 0ULL;
        south_e  = (has_e && has_s) ? ro_load(input + south_idx + 1) : 0ULL;
    }

    output[idx] = evolve_tile(
        center,
        north,
        south,
        center_w,
        center_e,
        north_w,
        north_e,
        south_w,
        south_e);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // One word stores one 8x8 tile, so the number of tiles per side is / 8.
    // The problem constraints make this an exact power-of-two division.
    const int tiles_per_row = grid_dimensions >> 3;

    // Exact tiling of the tile-grid:
    //   - x: 32 tiles per block (one warp-wide tile row)
    //   - y:  8 tile rows per block
    // No host-device synchronization is performed here; the caller owns that.
    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(
        static_cast<unsigned int>(tiles_per_row >> kBlockXLog2),
        static_cast<unsigned int>(tiles_per_row >> kBlockYLog2),
        1u);

    // Input and output are expected to be distinct device buffers (standard ping-pong step).
    game_of_life_step_kernel<<<grid, block>>>(input, output, tiles_per_row);
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
