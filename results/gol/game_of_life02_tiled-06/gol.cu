#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of a single step of Conway's Game of Life on a bit-packed grid.
//
// Grid encoding and layout
// - The grid is square with "grid_dimensions" cells per side (power of 2, > 512).
// - Cells are bit-packed into 64-bit words; each uint64_t encodes an 8x8 tile of cells.
// - Within each 64-bit tile word, bits are laid out in row-major order:
//     bit index = row * 8 + col, where row and col are in [0,7].
//     bit 0 corresponds to row=0, col=0 (top-left of the tile).
// - The array of tiles is arranged in row-major order:
//     tiles_per_row = grid_dimensions / 8;
//     index of tile at (tile_y, tile_x) = tile_y * tiles_per_row + tile_x.
//
// Boundary conditions
// - All cells outside the grid are considered dead. Neighbor lookups beyond grid edges
//   treat the out-of-bounds tiles as zero.
//
// Algorithm overview
// - Each thread processes one 8x8 tile, loading its 3x3 tile neighborhood (9 words).
// - For each of the three tile rows (top: NW,N,NE; middle: W,C,E; bottom: SW,S,SE),
//   we compute the horizontal 3-neighbor sum (west + center + east) per bit position
//   using carry-free bit-sliced addition. For the middle row we exclude the center
//   bitboard (so we only sum left and right), thereby excluding the cell itself
//   from the overall neighbor count.
// - We then vertically align the top row's horizontal sum "down by 1 cell" and the
//   bottom row's horizontal sum "up by 1 cell" into the center tile's coordinate frame,
//   all done in-register, including proper cross-tile handling for row 0/row 7 bytes.
// - Next, we add the three 2-bit numbers (top, middle, bottom) per bit position using
//   bit-sliced adders, producing a 4-bit neighbor count (0..8).
// - The next state per bit is computed by the Life rule:
//     alive_next = (count == 3) | (alive_current & (count == 2))
//
// Performance notes
// - No shared or texture memory is used; global reads are coalesced by assigning one
//   tile per thread and using a grid-stride loop.
// - All neighbor computations are performed with 64-bit bitwise operations, minimizing
//   branches and avoiding per-cell work.
//
// Assumptions
// - grid_dimensions is divisible by 8.
// - input and output are allocated with cudaMalloc and large enough to hold all tiles.

namespace {

// Column and row masks within an 8x8 bitboard (one uint64_t)
__device__ __constant__ uint64_t COL_0      = 0x0101010101010101ull; // bit 0 in each byte (col 0)
__device__ __constant__ uint64_t COL_7      = 0x8080808080808080ull; // bit 7 in each byte (col 7)
__device__ __constant__ uint64_t ROW_0      = 0x00000000000000FFull; // low byte (row 0)
__device__ __constant__ uint64_t ROW_7      = 0xFF00000000000000ull; // high byte (row 7)
__device__ __constant__ uint64_t NOT_COL_0  = 0xFEFEFEFEFEFEFEFEull; // ~COL_0
__device__ __constant__ uint64_t NOT_COL_7  = 0x7F7F7F7F7F7F7F7Full; // ~COL_7
__device__ __constant__ uint64_t NOT_ROW_0  = 0xFFFFFFFFFFFFFF00ull; // ~ROW_0
__device__ __constant__ uint64_t NOT_ROW_7  = 0x00FFFFFFFFFFFFFFull; // ~ROW_7

// Compute, in the coordinate frame of the center tile "C", the horizontal 3-neighbor sum
// (west + center + east) per cell, pulling in boundary bits from the left tile "L" and
// right tile "R" as needed.
// - If include_center is true, the "center" term is C (used for top/bottom rows).
// - If include_center is false, the "center" term is 0 (used for middle row to exclude
//   the cell itself from the overall neighbor count).
// Returns two 64-bit bitboards via references:
// - sum_lo: the LSB of the per-cell horizontal sum (0..3)
// - sum_hi: the MSB of the per-cell horizontal sum (0..3)
__device__ __forceinline__
void horizontal_sum3(uint64_t L, uint64_t C, uint64_t R, bool include_center,
                     uint64_t &sum_lo, uint64_t &sum_hi)
{
    // West neighbor contributions: shift source bits east by 1 cell,
    // bringing in col7 from L into col0 of C.
    uint64_t left_from  = ((C & NOT_COL_7) << 1) | ((L & COL_7) >> 7);

    // East neighbor contributions: shift source bits west by 1 cell,
    // bringing in col0 from R into col7 of C.
    uint64_t right_from = ((C & NOT_COL_0) >> 1) | ((R & COL_0) << 7);

    // Center contribution (either C or 0)
    uint64_t center = include_center ? C : 0ull;

    // Carry-free 1-bit + 1-bit + 1-bit addition:
    // Sum bit (LSB): XOR of the three inputs.
    // Carry bit (MSB): majority function over the three inputs.
    sum_lo = left_from ^ center ^ right_from;
    sum_hi = (left_from & center) | (left_from & right_from) | (center & right_from);
}

// Shift a 64-bit bitboard "x" one row SOUTH (down) into the coordinate frame of the tile
// below. This maps row 7 into row 0 of the destination frame and shifts rows 0..6 down by 1.
// For our usage, it maps the "top tile row" (N row) into the center tile's frame.
__device__ __forceinline__
uint64_t shift_south_1row_to_center(uint64_t x)
{
    return ((x & NOT_ROW_7) << 8) | ((x & ROW_7) >> 56);
}

// Shift a 64-bit bitboard "x" one row NORTH (up) into the coordinate frame of the tile
// above. This maps row 0 into row 7 of the destination frame and shifts rows 1..7 up by 1.
// For our usage, it maps the "bottom tile row" (S row) into the center tile's frame.
__device__ __forceinline__
uint64_t shift_north_1row_to_center(uint64_t x)
{
    return ((x & NOT_ROW_0) >> 8) | ((x & ROW_0) << 56);
}

// Add two per-cell 2-bit numbers (a1:a0) + (b1:b0) using bit-sliced addition.
// Produces a 3-bit result per cell via references: s2:s1:s0.
__device__ __forceinline__
void add_2bit_numbers(uint64_t a0, uint64_t a1,
                      uint64_t b0, uint64_t b1,
                      uint64_t &s0, uint64_t &s1, uint64_t &s2)
{
    // Add LSBs
    uint64_t s0_    = a0 ^ b0;
    uint64_t carry0 = a0 & b0;

    // Add the "second bits" plus carry0: (a1 + b1 + carry0)
    uint64_t t      = a1 ^ b1;
    uint64_t s1_    = t ^ carry0;
    uint64_t carry1 = (a1 & b1) | (t & carry0);

    // Third bit is the carry from the previous addition
    s0 = s0_;
    s1 = s1_;
    s2 = carry1;
}

// Add a 2-bit number (c1:c0) to a 3-bit number (s2:s1:s0) using bit-sliced addition.
// Produces a 4-bit result per cell via references: o3:o2:o1:o0.
__device__ __forceinline__
void add_2bit_to_3bit(uint64_t s0, uint64_t s1, uint64_t s2,
                      uint64_t c0, uint64_t c1,
                      uint64_t &o0, uint64_t &o1, uint64_t &o2, uint64_t &o3)
{
    // Add LSBs
    uint64_t t0     = s0 ^ c0;
    uint64_t carry0 = s0 & c0;

    // Add next bits plus carry
    uint64_t t1x    = s1 ^ c1;
    uint64_t t1     = t1x ^ carry0;
    uint64_t carry1 = (s1 & c1) | (t1x & carry0);

    // Add MSBs and propagate final carry
    uint64_t t2     = s2 ^ carry1;
    uint64_t t3     = s2 & carry1;

    o0 = t0;
    o1 = t1;
    o2 = t2;
    o3 = t3;
}

__global__ void life_kernel(const std::uint64_t* __restrict__ in,
                            std::uint64_t* __restrict__ out,
                            int tiles_per_row, int tiles_per_col)
{
    int total_tiles = tiles_per_row * tiles_per_col;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int t = tid; t < total_tiles; t += stride) {
        // Compute tile coordinates
        int ty = t / tiles_per_row;
        int tx = t - ty * tiles_per_row;

        // Load 3x3 neighborhood of tiles, with dead padding outside the grid.
        // Center row
        const std::uint64_t C  = in[t];
        const std::uint64_t W  = (tx > 0) ? in[t - 1] : 0ull;
        const std::uint64_t E  = (tx + 1 < tiles_per_row) ? in[t + 1] : 0ull;

        // Row above
        const int row_stride = tiles_per_row;
        const std::uint64_t N  = (ty > 0) ? in[t - row_stride] : 0ull;
        const std::uint64_t NW = (ty > 0 && tx > 0) ? in[t - row_stride - 1] : 0ull;
        const std::uint64_t NE = (ty > 0 && tx + 1 < tiles_per_row) ? in[t - row_stride + 1] : 0ull;

        // Row below
        const std::uint64_t S  = (ty + 1 < tiles_per_col) ? in[t + row_stride] : 0ull;
        const std::uint64_t SW = (ty + 1 < tiles_per_col && tx > 0) ? in[t + row_stride - 1] : 0ull;
        const std::uint64_t SE = (ty + 1 < tiles_per_col && tx + 1 < tiles_per_row) ? in[t + row_stride + 1] : 0ull;

        // Horizontal 3-sums for the top, middle, and bottom tile rows.
        // Top row: include center (N) to count vertical neighbors from above.
        std::uint64_t top_lo, top_hi;
        horizontal_sum3(NW, N, NE, /*include_center=*/true, top_lo, top_hi);

        // Middle row: exclude center (C) to avoid counting the cell itself.
        std::uint64_t mid_lo, mid_hi;
        horizontal_sum3(W, C, E,  /*include_center=*/false, mid_lo, mid_hi);

        // Bottom row: include center (S) to count vertical neighbors from below.
        std::uint64_t bot_lo, bot_hi;
        horizontal_sum3(SW, S, SE, /*include_center=*/true, bot_lo, bot_hi);

        // Vertically align top and bottom row contributions into the center tile's frame.
        // Top row (above) shifts one row SOUTH into center tile.
        std::uint64_t n_lo = shift_south_1row_to_center(top_lo);
        std::uint64_t n_hi = shift_south_1row_to_center(top_hi);

        // Bottom row (below) shifts one row NORTH into center tile.
        std::uint64_t s_lo = shift_north_1row_to_center(bot_lo);
        std::uint64_t s_hi = shift_north_1row_to_center(bot_hi);

        // Sum vertically: (n_lo:n_hi) + (mid_lo:mid_hi) => s2:s1:s0
        std::uint64_t v0, v1, v2;
        add_2bit_numbers(n_lo, n_hi, mid_lo, mid_hi, v0, v1, v2);

        // Then add bottom row: + (s_lo:s_hi) => t3:t2:t1:t0 (4-bit neighbor count per cell)
        std::uint64_t t0, t1, t2, t3;
        add_2bit_to_3bit(v0, v1, v2, s_lo, s_hi, t0, t1, t2, t3);

        // Apply Life rule: next = (count == 3) | (alive & (count == 2))
        // count == 3: t3=0, t2=0, t1=1, t0=1
        // count == 2: t3=0, t2=0, t1=1, t0=0
        // Compute equality tests with bitwise logic.
        std::uint64_t not_t3 = ~t3;
        std::uint64_t not_t2 = ~t2;

        std::uint64_t eq3 = not_t3 & not_t2 & t1 & t0;
        std::uint64_t eq2 = not_t3 & not_t2 & t1 & (~t0);

        std::uint64_t next = eq3 | (C & eq2);

        // Store the updated tile
        out[t] = next;
    }
}

} // anonymous namespace

// Host function to run one generation step of Conway's Game of Life.
// - input:  pointer to device memory holding the input grid (bit-packed tiles).
// - output: pointer to device memory to store the output grid (bit-packed tiles).
// - grid_dimensions: number of cells per side; must be a power of 2 and divisible by 8.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    if (grid_dimensions <= 0) return;

    const int tiles_per_row = grid_dimensions >> 3; // divide by 8
    const int tiles_per_col = tiles_per_row;
    const int total_tiles   = tiles_per_row * tiles_per_col;

    // Choose a reasonable launch configuration; kernel uses a grid-stride loop.
    const int block_size = 256;
    int grid_size = (total_tiles + block_size - 1) / block_size;
    // Cap grid size to avoid excessive launch dimensions (modern GPUs support large grids).
    if (grid_size > 131072) grid_size = 131072;

    life_kernel<<<grid_size, block_size>>>(input, output, tiles_per_row, tiles_per_col);
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
