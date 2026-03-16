#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Bit layout assumed inside each 8x8 tile word:
//
//   bit (row * 8 + col)  <=>  local cell (col, row)
//
// So each tile row is exactly one byte:
//   bits   0.. 7 -> row 0
//   bits   8..15 -> row 1
//   ...
//   bits  56..63 -> row 7
//
// The full board is a row-major array of these 8x8 tiles.
//
// Performance choices:
// - One thread updates one 8x8 tile entirely in registers.
// - No shared memory is used; the 3x3 tile neighborhood is tiny and hardware caches
//   handle the overlap well on modern data-center GPUs.
// - A single kernel is used with a fast interior path and a rare boundary path.
// - A 1D launch is used to avoid grid-y limits on very large boards. Because the tile
//   row width is a power of two, x = tile_idx & (stride - 1) is enough to recover the
//   tile x-coordinate without division.
// - kBlockSize is 128, matching the minimum legal tile-row width. Since every legal
//   tile-row width is a power of two >= 128, blocks never cross tile-row boundaries.

using u64 = std::uint64_t;

constexpr int kBlockShift = 7;
constexpr int kBlockSize  = 1 << kBlockShift;
static_assert((kBlockSize & 31) == 0, "Block size must be a whole number of warps.");

constexpr u64 kCol0    = 0x0101010101010101ULL; // column 0 in every row byte
constexpr u64 kCol7    = 0x8080808080808080ULL; // column 7 in every row byte
constexpr u64 kNotCol0 = 0xFEFEFEFEFEFEFEFEULL; // clear column 0 in every row byte
constexpr u64 kNotCol7 = 0x7F7F7F7F7F7F7F7FULL; // clear column 7 in every row byte
constexpr u64 kLowByteCol7 = 0x0000000000000080ULL; // bit 7 of the low byte

// Half adder for 64 independent 1-bit lanes packed into a 64-bit word.
__device__ __forceinline__ void half_add(u64 a, u64 b, u64& sum, u64& carry) {
    sum   = a ^ b;
    carry = a & b;
}

// Full adder for 64 independent 1-bit lanes packed into a 64-bit word.
// The compiler maps this very well to modern NVIDIA ternary-logic instructions.
__device__ __forceinline__ void full_add(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 ab = a ^ b;
    sum   = ab ^ c;
    carry = (a & b) | (ab & c);
}

// Evolve one 8x8 tile. The 3x3 neighborhood is passed as 9 tile words:
//
//   nw  n  ne
//    w  c   e
//   sw  s  se
//
// All neighbor generation and counting is fully bit-parallel across the 64 cells.
__device__ __forceinline__ u64 evolve_tile(
    u64 nw, u64 n, u64 ne,
    u64 w,  u64 c, u64 e,
    u64 sw, u64 s, u64 se)
{
    const u64 west_edge = w & kCol7;
    const u64 east_edge = e & kCol0;

    // Cardinal neighbors aligned to the destination cell positions.
    const u64 north = (c << 8) | (n >> 56);
    const u64 south = (c >> 8) | (s << 56);

    // Carry-save adder network for the 8 one-bit neighbor boards.
    // We only keep the count modulo 8 (ones/twos/fours). That is sufficient because
    // the true count range is 0..8, and the only truncated value is 8 -> 0 mod 8;
    // 8 is neither 2 nor 3, which are the only counts Conway's rules care about.
    u64 s0, c0;
    {
        const u64 east = ((c >> 1) & kNotCol7) | (east_edge << 7);
        full_add(north, south, east, s0, c0);
    }

    u64 s1, c1;
    {
        const u64 west       = ((c << 1) & kNotCol0) | (west_edge >> 7);
        const u64 north_east = ((north >> 1) & kNotCol7) | (east_edge << 15) | ((ne >> 49) & kLowByteCol7);
        const u64 north_west = ((north << 1) & kNotCol0) | (west_edge << 1) | (nw >> 63);
        full_add(west, north_east, north_west, s1, c1);
    }

    u64 s2, c2;
    {
        const u64 south_east = ((south >> 1) & kNotCol7) | (east_edge >> 1) | ((se & 1ULL) << 63);
        const u64 south_west = ((south << 1) & kNotCol0) | (west_edge >> 15) | ((sw & kLowByteCol7) << 49);
        full_add(south_east, south_west, s0, s2, c2);
    }

    u64 ones, carry_to_twos;
    half_add(s1, s2, ones, carry_to_twos);

    u64 twos_partial, carry_to_fours_a;
    full_add(c0, c1, c2, twos_partial, carry_to_fours_a);

    u64 twos, carry_to_fours_b;
    half_add(twos_partial, carry_to_twos, twos, carry_to_fours_b);

    const u64 fours = carry_to_fours_a ^ carry_to_fours_b;

    // Conway rule:
    //   next = (count == 3) | (alive & (count == 2))
    // With the bit-planes above this simplifies to:
    //   next = twos & ~fours & (ones | alive)
    return twos & ~fours & (ones | c);
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    u64 stride,          // tiles per row = grid_dimensions / 8
    u64 x_mask,          // stride - 1, valid because stride is a power of two
    u64 last_row_start)  // first tile index of the last tile row
{
    // Exact launch shape: because stride is a power of two >= 128, the total tile count
    // is an exact multiple of kBlockSize, so no tail guard is needed.
    const u64 tile_idx =
        (static_cast<u64>(blockIdx.x) << kBlockShift) + static_cast<u64>(threadIdx.x);

    const u64 x = tile_idx & x_mask;

    const bool has_w = (x != 0ULL);
    const bool has_e = (x != x_mask);
    const bool has_n = (tile_idx >= stride);
    const bool has_s = (tile_idx < last_row_start);

    const u64 c = input[tile_idx];

    u64 nw, n, ne;
    u64 w,  e;
    u64 sw, s, se;

    // Fast path for the overwhelming majority of tiles: all 8 neighboring tiles exist.
    if (has_w && has_e && has_n && has_s) {
        const u64 north_idx = tile_idx - stride;
        const u64 south_idx = tile_idx + stride;

        nw = input[north_idx - 1];
        n  = input[north_idx];
        ne = input[north_idx + 1];

        w  = input[tile_idx - 1];
        e  = input[tile_idx + 1];

        sw = input[south_idx - 1];
        s  = input[south_idx];
        se = input[south_idx + 1];
    } else {
        // Boundary path: cells outside the grid are dead, so missing neighboring tiles
        // are simply injected as zero. No toroidal wrap-around.
        w = has_w ? input[tile_idx - 1] : 0ULL;
        e = has_e ? input[tile_idx + 1] : 0ULL;

        if (has_n) {
            const u64 north_idx = tile_idx - stride;
            n  = input[north_idx];
            nw = has_w ? input[north_idx - 1] : 0ULL;
            ne = has_e ? input[north_idx + 1] : 0ULL;
        } else {
            nw = 0ULL;
            n  = 0ULL;
            ne = 0ULL;
        }

        if (has_s) {
            const u64 south_idx = tile_idx + stride;
            s  = input[south_idx];
            sw = has_w ? input[south_idx - 1] : 0ULL;
            se = has_e ? input[south_idx + 1] : 0ULL;
        } else {
            sw = 0ULL;
            s  = 0ULL;
            se = 0ULL;
        }
    }

    output[tile_idx] = evolve_tile(nw, n, ne, w, c, e, sw, s, se);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Per the problem statement:
    // - input/output are already device pointers allocated by cudaMalloc
    // - grid_dimensions is a power of two and > 512
    // - the caller handles any synchronization around this launch
    //
    // input and output are assumed to be distinct buffers (double-buffered simulation step).

    const u64 stride = static_cast<u64>(grid_dimensions) >> 3; // tiles per row
    const u64 num_tiles = stride * stride;
    const u64 x_mask = stride - 1;
    const u64 last_row_start = num_tiles - stride;

    // Exact because num_tiles is a power of two and kBlockSize == 128.
    const unsigned int grid_blocks = static_cast<unsigned int>(num_tiles >> kBlockShift);

    game_of_life_kernel<<<grid_blocks, kBlockSize>>>(input, output, stride, x_mask, last_row_start);
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
