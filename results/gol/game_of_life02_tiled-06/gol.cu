#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/*
Optimized Conway's Game of Life for a grid stored as row-major 8x8 tiles.

Assumed layout:
  - The tile array itself is row-major over 8x8 tiles.
  - Inside each 64-bit tile, bit index (local_y << 3) | local_x encodes cell (local_x, local_y).
    Equivalently: each tile is eight 8-bit rows packed from the least-significant byte upward.
    This is the common "bitboard" convention where moving north is << 8 and moving east is >> 1.

Performance strategy:
  - One CUDA thread updates one 8x8 tile, so 64 cells are processed in parallel with bitwise logic.
  - blockDim.x == 32, so every warp covers 32 adjacent tiles in x.
    Horizontal neighbor tiles are exchanged with warp shuffles; only the two warp-edge lanes
    need extra global loads.
  - The eight neighbor planes are added with bit-sliced arithmetic; shared/texture memory are
    intentionally avoided because they are unnecessary here for this layout.

The kernel expects input and output to be distinct device buffers.
*/

namespace {

using u64 = unsigned long long;

constexpr int kTileSide      = 8;
constexpr int kTileSideShift = 3;

constexpr int kBlockX      = 32;  // exactly one warp in x
constexpr int kBlockXShift = 5;
constexpr int kBlockY      = 8;   // 8 warp-rows per block => 256 threads/block
constexpr int kBlockYShift = 3;

constexpr int kThreadsPerBlock = kBlockX * kBlockY;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "64-bit type mismatch.");
static_assert(kTileSide == 8, "This implementation is specialized for 8x8 packed tiles.");
static_assert(kBlockX == 32, "blockDim.x must equal one warp.");
static_assert((1 << kTileSideShift) == kTileSide, "Tile shift mismatch.");
static_assert((1 << kBlockXShift) == kBlockX, "Block X shift mismatch.");
static_assert((1 << kBlockYShift) == kBlockY, "Block Y shift mismatch.");

// Column masks inside an 8x8 tile.
constexpr u64 kCol0    = 0x0101010101010101ULL; // x == 0 in each row
constexpr u64 kCol7    = kCol0 << 7;            // x == 7 in each row
constexpr u64 kNotCol0 = ~kCol0;
constexpr u64 kNotCol7 = ~kCol7;

constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

// Majority of three 64-bit bitboards.
// On modern NVIDIA GPUs this typically lowers to efficient LOP3-style logic.
__device__ __forceinline__ u64 majority3(u64 a, u64 b, u64 c) {
    return (a & b) | (c & (a | b));
}

// Aligned-neighbor extraction helpers.
// The returned bitboard contains, at each cell position, the state of that neighbor.
// Missing tiles are passed as zero, which naturally implements the required dead boundary.
__device__ __forceinline__ u64 align_west(u64 tile, u64 west_tile) {
    return ((tile & kNotCol7) << 1) | ((west_tile & kCol7) >> 7);
}

__device__ __forceinline__ u64 align_east(u64 tile, u64 east_tile) {
    return ((tile & kNotCol0) >> 1) | ((east_tile & kCol0) << 7);
}

__device__ __forceinline__ u64 align_north(u64 tile, u64 north_tile) {
    return (tile << kTileSide) | (north_tile >> (64 - kTileSide));
}

__device__ __forceinline__ u64 align_south(u64 tile, u64 south_tile) {
    return (tile >> kTileSide) | (south_tile << (64 - kTileSide));
}

// 64 parallel 4-input additions.
// For every bit position, the outputs are the 1/2/4 bits of the local 0..4 population count.
__device__ __forceinline__ void sum4(
    u64 a, u64 b, u64 c, u64 d,
    u64& ones, u64& twos, u64& fours)
{
    const u64 ab_xor = a ^ b;
    const u64 ab_and = a & b;
    const u64 cd_xor = c ^ d;
    const u64 cd_and = c & d;

    ones = ab_xor ^ cd_xor;

    const u64 carry_into_twos = ab_xor & cd_xor;
    twos  = ab_and ^ cd_and ^ carry_into_twos;
    fours = majority3(ab_and, cd_and, carry_into_twos);
}

__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int tiles_per_row)
{
    // Because blockDim.x == 32, each warp is one contiguous strip of tiles in x.
    const int lane   = static_cast<int>(threadIdx.x);
    const int tile_x = (static_cast<int>(blockIdx.x) << kBlockXShift) + lane;
    const int tile_y = (static_cast<int>(blockIdx.y) << kBlockYShift) + static_cast<int>(threadIdx.y);

    const int last_tile = tiles_per_row - 1;

    const bool has_north = (tile_y != 0);
    const bool has_south = (tile_y != last_tile);
    const bool has_west  = (tile_x != 0);
    const bool has_east  = (tile_x != last_tile);

    const std::size_t linear_tile =
        static_cast<std::size_t>(tile_y) * static_cast<std::size_t>(tiles_per_row) +
        static_cast<std::size_t>(tile_x);

    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(tiles_per_row);
    const std::uint64_t* const center_ptr = input + linear_tile;

    // Vertical neighbors are loaded directly.
    const u64 tile_center = static_cast<u64>(*center_ptr);
    const u64 tile_north  = has_north ? static_cast<u64>(*(center_ptr - stride)) : 0ULL;
    const u64 tile_south  = has_south ? static_cast<u64>(*(center_ptr + stride)) : 0ULL;

    // Horizontal neighbors mostly come from neighboring lanes in the same warp.
    // Only warp-edge lanes need explicit global loads to cross the 32-tile warp boundary.
    u64 tile_west = __shfl_up_sync(kFullWarpMask,   tile_center, 1);
    u64 tile_east = __shfl_down_sync(kFullWarpMask, tile_center, 1);

    u64 tile_nw   = __shfl_up_sync(kFullWarpMask,   tile_north,  1);
    u64 tile_ne   = __shfl_down_sync(kFullWarpMask, tile_north,  1);

    u64 tile_sw   = __shfl_up_sync(kFullWarpMask,   tile_south,  1);
    u64 tile_se   = __shfl_down_sync(kFullWarpMask, tile_south,  1);

    if (lane == 0) {
        tile_west = has_west ? static_cast<u64>(*(center_ptr - 1)) : 0ULL;
        tile_nw   = (has_north && has_west) ? static_cast<u64>(*(center_ptr - stride - 1)) : 0ULL;
        tile_sw   = (has_south && has_west) ? static_cast<u64>(*(center_ptr + stride - 1)) : 0ULL;
    }

    if (lane == kBlockX - 1) {
        tile_east = has_east ? static_cast<u64>(*(center_ptr + 1)) : 0ULL;
        tile_ne   = (has_north && has_east) ? static_cast<u64>(*(center_ptr - stride + 1)) : 0ULL;
        tile_se   = (has_south && has_east) ? static_cast<u64>(*(center_ptr + stride + 1)) : 0ULL;
    }

    // Build the eight aligned neighbor planes for the 64 cells in this tile.
    const u64 north_row      = align_north(tile_center, tile_north);
    const u64 north_row_west = align_north(tile_west,   tile_nw);
    const u64 north_row_east = align_north(tile_east,   tile_ne);

    const u64 neighbor_nw = align_west(north_row, north_row_west);
    const u64 neighbor_ne = align_east(north_row, north_row_east);
    const u64 neighbor_w  = align_west(tile_center, tile_west);

    const u64 south_row      = align_south(tile_center, tile_south);
    const u64 south_row_west = align_south(tile_west,   tile_sw);
    const u64 south_row_east = align_south(tile_east,   tile_se);

    const u64 neighbor_sw = align_west(south_row, south_row_west);
    const u64 neighbor_se = align_east(south_row, south_row_east);
    const u64 neighbor_e  = align_east(tile_center, tile_east);

    // Sum the eight neighbor planes as two independent 4-input groups.
    // The center tile is intentionally excluded from this count.
    u64 ones0, twos0, fours0;
    sum4(neighbor_nw, north_row, neighbor_ne, neighbor_w, ones0, twos0, fours0);

    u64 ones1, twos1, fours1;
    sum4(neighbor_e, neighbor_sw, south_row, neighbor_se, ones1, twos1, fours1);

    // Combine the two 0..4 partial counts into the total 0..8 count.
    const u64 ones_bit         = ones0 ^ ones1;
    const u64 carry_into_twos  = ones0 & ones1;
    const u64 twos_bit         = twos0 ^ twos1 ^ carry_into_twos;
    const u64 carry_into_fours = majority3(twos0, twos1, carry_into_twos);

    // twos_bit is 1 for total counts {2,3,6,7}.
    // Any contribution in the 4's place means the total is >= 4, so filtering that out
    // leaves exactly {2,3}. The 1's bit then distinguishes 3 from 2:
    //   - 3  -> ones_bit = 1 => birth or survive
    //   - 2  -> ones_bit = 0 => survive only if currently alive
    const u64 ge4             = fours0 | fours1 | carry_into_fours;
    const u64 count_is_2_or_3 = twos_bit & ~ge4;
    const u64 next_tile       = count_is_2_or_3 & (ones_bit | tile_center);

    output[linear_tile] = static_cast<std::uint64_t>(next_tile);
}

} // anonymous namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is guaranteed to be a power of two > 512.
    // Therefore tiles_per_row = grid_dimensions / 8 is also a power of two and is divisible
    // by both 32 and 8, so this launch covers the tile grid exactly with no bounds checks.
    const int tiles_per_row = grid_dimensions >> kTileSideShift;

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        static_cast<unsigned int>(tiles_per_row >> kBlockXShift),
        static_cast<unsigned int>(tiles_per_row >> kBlockYShift));

    // Launch only; synchronization and any desired error handling are intentionally left to the caller.
    game_of_life_kernel<<<grid, block>>>(input, output, tiles_per_row);
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
