#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one simulation step of Conway's Game of Life on a bit-packed grid.
// The grid is divided into 8x8 tiles, each stored in one 64-bit word (std::uint64_t).
// Bit layout inside a tile:
//   - Bits are arranged row-major within the tile.
//   - For a tile, bit index b = (row * 8 + col), with row, col in [0,7].
//   - Within each row, bit 0 corresponds to col 0 (leftmost), bit 7 to col 7 (rightmost).
//
// Memory layout across tiles:
//   - Tiles are stored row-major. For a grid of dimension N (power of 2), there are T = N / 8 tiles per dimension.
//   - Tile at (ty, tx) (0-based) is at index ty * T + tx.
//
// Boundary handling:
//   - Cells outside the grid are considered dead (0). For tiles at the edges, neighbor tiles beyond the boundary
//     are treated as zero (no wrap-around).
//
// Performance approach:
//   - No shared/texture memory; use global memory only with coalesced loads where possible.
//   - Each CUDA thread computes one output tile using bit-parallel operations.
//   - Neighbor counts are obtained using carry-save adders (CSA) on eight 64-bit bitboards corresponding to the
//     eight neighbor directions. This avoids per-bit loops and prevents cross-bit carries.
//
// Kernel launch:
//   - A 2D grid of blocks covers the T x T tiles.
//   - Block size chosen as (32, 8) for good occupancy and coalescing on A100/H100 GPUs.

namespace {

__device__ __forceinline__ void csa(std::uint64_t &h, std::uint64_t &l,
                                    const std::uint64_t a,
                                    const std::uint64_t b,
                                    const std::uint64_t c)
{
    // Carry-save adder for three bitboards:
    // l = (a ^ b ^ c)      -> parity (sum bit, weight 1)
    // h = majority(a,b,c)  -> carry bit to next weight (weight 2)
    const std::uint64_t u = a ^ b;
    l = u ^ c;
    h = (a & b) | (u & c);
}

// Masks for per-row boundary handling within an 8x8 tile.
// ROW_LSB_MASK has the least significant bit of each 8-bit row set (col 0 of each row).
// ROW_MSB_MASK has the most significant bit of each 8-bit row set (col 7 of each row).
static __device__ __forceinline__ std::uint64_t row_lsb_mask()
{
    return 0x0101010101010101ull;
}
static __device__ __forceinline__ std::uint64_t row_msb_mask()
{
    return 0x8080808080808080ull;
}

// Horizontal shift by 1 to the west (i.e., source cells are to the east of targets).
// For each row, perform a >> 1 with no wrap across the row boundary, and fill the vacated col 7 -> col 0
// from the MSB column of the west neighbor tile.
__device__ __forceinline__ std::uint64_t shift_west_1(const std::uint64_t center,
                                                      const std::uint64_t west)
{
    const std::uint64_t lsb = row_lsb_mask();
    const std::uint64_t msb = row_msb_mask();
    const std::uint64_t from_center = (center & ~lsb) >> 1;
    const std::uint64_t from_west   = (west   &  msb) >> 7;
    return from_center | from_west;
}

// Horizontal shift by 1 to the east (i.e., source cells are to the west of targets).
// For each row, perform a << 1 with no wrap across the row boundary, and fill the vacated col 0 <- col 7
// from the LSB column of the east neighbor tile.
__device__ __forceinline__ std::uint64_t shift_east_1(const std::uint64_t center,
                                                      const std::uint64_t east)
{
    const std::uint64_t lsb = row_lsb_mask();
    const std::uint64_t msb = row_msb_mask();
    const std::uint64_t from_center = (center & ~msb) << 1;
    const std::uint64_t from_east   = (east   &  lsb) << 7;
    return from_center | from_east;
}

// Vertical shift by 1 to the north (i.e., source cells are to the south of targets).
// Shift center tile up by 8 bits (row r-1 -> r), and fill row 0 from row 7 of the north tile.
__device__ __forceinline__ std::uint64_t shift_north_1(const std::uint64_t center,
                                                       const std::uint64_t north)
{
    return (center << 8) | (north >> 56);
}

// Vertical shift by 1 to the south (i.e., source cells are to the north of targets).
// Shift center tile down by 8 bits (row r+1 -> r), and fill row 7 from row 0 of the south tile.
__device__ __forceinline__ std::uint64_t shift_south_1(const std::uint64_t center,
                                                       const std::uint64_t south)
{
    return (center >> 8) | (south << 56);
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_per_dim)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= tiles_per_dim || ty >= tiles_per_dim) return;

    const int T = tiles_per_dim;
    const int idx = ty * T + tx;

    // Determine the existence of neighbor tiles to respect dead boundaries outside the grid.
    const bool has_w = (tx > 0);
    const bool has_e = (tx + 1 < T);
    const bool has_n = (ty > 0);
    const bool has_s = (ty + 1 < T);

    // Load center and neighbor tiles (zero if out-of-bounds).
    const std::uint64_t tileC  = __ldg(&input[idx]);
    const std::uint64_t tileW  = has_w ? __ldg(&input[idx - 1])       : 0ull;
    const std::uint64_t tileE  = has_e ? __ldg(&input[idx + 1])       : 0ull;
    const std::uint64_t tileN  = has_n ? __ldg(&input[idx - T])       : 0ull;
    const std::uint64_t tileS  = has_s ? __ldg(&input[idx + T])       : 0ull;
    const std::uint64_t tileNW = (has_n && has_w) ? __ldg(&input[idx - T - 1]) : 0ull;
    const std::uint64_t tileNE = (has_n && has_e) ? __ldg(&input[idx - T + 1]) : 0ull;
    const std::uint64_t tileSW = (has_s && has_w) ? __ldg(&input[idx + T - 1]) : 0ull;
    const std::uint64_t tileSE = (has_s && has_e) ? __ldg(&input[idx + T + 1]) : 0ull;

    // Compute horizontally shifted boards for center, north, and south tiles, used to form
    // west/east contributions and the NW/NE/SW/SE diagonals.
    const std::uint64_t wC = shift_west_1(tileC,  tileW);
    const std::uint64_t eC = shift_east_1(tileC,  tileE);
    const std::uint64_t wN = shift_west_1(tileN,  tileNW);
    const std::uint64_t eN = shift_east_1(tileN,  tileNE);
    const std::uint64_t wS = shift_west_1(tileS,  tileSW);
    const std::uint64_t eS = shift_east_1(tileS,  tileSE);

    // Eight neighbor-direction bitboards aligned to the center tile's coordinates.
    const std::uint64_t w  = wC;
    const std::uint64_t e  = eC;
    const std::uint64_t n  = shift_north_1(tileC, tileN);
    const std::uint64_t s  = shift_south_1(tileC, tileS);
    const std::uint64_t nw = shift_north_1(wC, wN);
    const std::uint64_t ne = shift_north_1(eC, eN);
    const std::uint64_t sw = shift_south_1(wC, wS);
    const std::uint64_t se = shift_south_1(eC, eS);

    // Sum eight 1-bit neighbor bitboards using carry-save adders to obtain
    // the count bitplanes b1 (2^0), b2 (2^1), b4 (2^2), b8 (2^3).
    // Pipeline (all operations are bitwise and carry-less within each bit position):
    //   Level 1: group into triples
    std::uint64_t h1, l1; csa(h1, l1, w,  e,  n);
    std::uint64_t h2, l2; csa(h2, l2, s,  nw, ne);
    std::uint64_t h3, l3; csa(h3, l3, sw, se,  0ull);

    //   Level 2: sum the low parts of the triples (weight 1 accumulators)
    std::uint64_t h4, l4; csa(h4, l4, l1, l2, l3); // l4 = parity of all 8 inputs -> b1

    //   Level 3: sum the carries (weight 2 accumulators)
    std::uint64_t h5, l5; csa(h5, l5, h1, h2, h3);
    std::uint64_t h6, l6; csa(h6, l6, l5, h4, 0ull); // l6 = parity of total weight-2 -> b2

    //   Level 4: sum the weight-4 accumulators
    std::uint64_t h7, l7; csa(h7, l7, h5, h6, 0ull); // l7 = parity of weight-4 -> b4, h7 -> b8

    const std::uint64_t b1 = l4;
    const std::uint64_t b2 = l6;
    const std::uint64_t b4 = l7;
    const std::uint64_t b8 = h7;

    // Compute equality masks for counts == 2 and == 3.
    const std::uint64_t not_b1 = ~b1;
    const std::uint64_t not_b4 = ~b4;
    const std::uint64_t not_b8 = ~b8;

    const std::uint64_t eq2 = (not_b1 & b2 & not_b4 & not_b8);
    const std::uint64_t eq3 = (b1     & b2 & not_b4 & not_b8);

    // Apply Game of Life rule:
    // next = (count == 3) | (current & (count == 2))
    const std::uint64_t next = eq3 | (tileC & eq2);

    output[idx] = next;
}

} // anonymous namespace

// Host function to launch a single step of Conway's Game of Life.
//
// Parameters:
// - input:  pointer to device memory grid, bit-packed; each uint64_t encodes an 8x8 tile.
// - output: pointer to device memory grid for next state, same packing as input.
// - grid_dimensions: the width/height of the square grid in cells (power of 2, >= 512).
//
// The function sets up the kernel configuration and launches the kernel.
// No synchronization is performed here; the caller handles any required synchronization.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    const int tiles_per_dim = grid_dimensions >> 3; // grid_dimensions / 8
    dim3 block(32, 8, 1);
    dim3 grid((tiles_per_dim + block.x - 1) / block.x,
              (tiles_per_dim + block.y - 1) / block.y,
              1);

    game_of_life_kernel<<<grid, block>>>(input, output, tiles_per_dim);
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
