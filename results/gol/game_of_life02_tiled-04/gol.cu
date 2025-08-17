#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of a single step of Conway's Game of Life on a bit-packed grid.
// Each 64-bit word encodes an 8x8 tile of cells.
// Bit mapping inside each 64-bit word:
// - Bit index b = y*8 + x (0 <= x,y < 8), with x as the column (left-to-right) and y as the row (top-to-bottom).
// - The least significant bit (bit 0) corresponds to the top-left cell of the 8x8 tile.
// The grid is a square of size grid_dimensions x grid_dimensions, where grid_dimensions is a power of two and > 512.
// The grid is partitioned into tiles of size 8x8; thus, there are tile_dim = grid_dimensions / 8 tiles per dimension.
//
// Memory layout of the tiles is row-major by tile:
// - Tile index for tile coordinates (tx, ty) is idx = ty * tile_dim + tx.
// - For each tile, the 64-bit word packs its 8x8 cells as described above.
//
// Boundary handling: Cells outside the grid are dead (zero). The kernel treats missing neighbor tiles as zero.
//
// Algorithm overview:
// - For each tile, load the 3x3 neighborhood of 64-bit tiles (9 words: center + 8 neighbors).
// - Construct eight 64-bit bitboards representing the presence of alive neighbors in each of the 8 directions for the current tile's 8x8 cells:
//   N, NE, E, SE, S, SW, W, NW. Each bitboard has 1s exactly at positions of cells in the current tile that have a live neighbor in that direction.
//   This requires shifting within the 64-bit word and importing boundary columns/rows from adjacent tiles where needed.
// - Sum the eight direction bitboards using a carry-save adder (CSA) network to obtain, for each bit position (cell), the number of alive neighbors
//   modulo 8 as three bit-planes: ones (1's place), twos (2's place), fours (4's place).
//   Note: counts of 8 neighbors (which would set the 8's place) are not representable in 3 bits but do not affect rules for exactly 2 or 3.
// - Apply Game of Life rules: next = (neighbor_count == 3) | (alive & (neighbor_count == 2)).
// - Store the resulting 64-bit word for the tile.
//
// No shared or texture memory is used; global loads are coalesced when launching with a 2D grid over tiles.

using u64 = std::uint64_t;

__global__ void game_of_life_step_kernel(const u64* __restrict__ in, u64* __restrict__ out, int tile_dim)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tile_dim || ty >= tile_dim) return;

    const int idx = ty * tile_dim + tx;

    // Neighbor tile availability flags
    const bool hasW = (tx > 0);
    const bool hasE = (tx + 1 < tile_dim);
    const bool hasN = (ty > 0);
    const bool hasS = (ty + 1 < tile_dim);

    // Load center and neighbor tiles; missing tiles are treated as zero (dead outside the grid).
    const u64 C  = in[idx];
    const u64 N  = hasN ? in[idx - tile_dim] : 0ull;
    const u64 S  = hasS ? in[idx + tile_dim] : 0ull;
    const u64 W  = hasW ? in[idx - 1]        : 0ull;
    const u64 E  = hasE ? in[idx + 1]        : 0ull;
    const u64 NW = (hasN && hasW) ? in[idx - tile_dim - 1] : 0ull;
    const u64 NE = (hasN && hasE) ? in[idx - tile_dim + 1] : 0ull;
    const u64 SW = (hasS && hasW) ? in[idx + tile_dim - 1] : 0ull;
    const u64 SE = (hasS && hasE) ? in[idx + tile_dim + 1] : 0ull;

    // Column masks within a tile: A = x==0 column, H = x==7 column.
    // We use these to prevent horizontal wrap across row boundaries when shifting.
    const u64 A    = 0x0101010101010101ull; // column x=0 across all rows
    const u64 H    = 0x8080808080808080ull; // column x=7 across all rows
    const u64 notA = ~A;
    const u64 notH = ~H;

    // Build vertically aligned row bitboards for center, and for E/W tiles when needed for diagonals.
    // row_n_X represents the tile X content shifted so that at each position (x,y) it equals the state at (x,y-1) in the original grid.
    // row_s_X represents the tile X content shifted so that at each position (x,y) it equals the state at (x,y+1) in the original grid.
    // Only X in {C, E, W} are needed here; N/S/NE/NW/SE/SW are included via the 56-bit shifts.
    const u64 row_n_C = (C >> 8) | (N >> 56);
    const u64 row_s_C = (C << 8) | (S << 56);

    const u64 row_n_E = (E >> 8) | (NE >> 56);
    const u64 row_n_W = (W >> 8) | (NW >> 56);

    const u64 row_s_E = (E << 8) | (SE << 56);
    const u64 row_s_W = (W << 8) | (SW << 56);

    // Directional neighbor bitboards:
    // For each (x,y) in the current tile, these bitboards have a 1 if the neighbor cell in that direction is alive.
    const u64 Ndir  = row_n_C;
    const u64 Sdir  = row_s_C;

    // Horizontal neighbors: handle cross-tile edges by importing E/W tile boundary columns.
    const u64 Edir  = ((C & notA) >> 1) | ((E & A) << 7);       // b(x+1,y)
    const u64 Wdir  = ((C & notH) << 1) | ((W & H) >> 7);       // b(x-1,y)

    // Diagonals: shift the vertically aligned rows and import E/W neighbors for cross-tile columns.
    const u64 NEdir = ((row_n_C & notA) >> 1) | ((row_n_E & A) << 7); // b(x+1,y-1)
    const u64 NWdir = ((row_n_C & notH) << 1) | ((row_n_W & H) >> 7); // b(x-1,y-1)
    const u64 SEdir = ((row_s_C & notA) >> 1) | ((row_s_E & A) << 7); // b(x+1,y+1)
    const u64 SWdir = ((row_s_C & notH) << 1) | ((row_s_W & H) >> 7); // b(x-1,y+1)

    // Sum the eight neighbor bitboards using a carry-save adder (CSA) tree.
    // We compute three bit-planes per cell: ones (1's place), twos (2's place), fours (4's place).
    // Counts are modulo 8; 8 neighbors (1000b) maps to 000b here, which is fine for detecting exactly 2 or 3.
    u64 s10, c10; // CSA(N, NE, E)
    {
        const u64 a = Ndir, b = NEdir, c = Edir;
        const u64 axb = a ^ b;
        s10 = axb ^ c;
        c10 = (a & b) | (a & c) | (b & c); // weight-2 carries
    }
    u64 s11, c11; // CSA(SE, S, SW)
    {
        const u64 a = SEdir, b = Sdir, c = SWdir;
        const u64 axb = a ^ b;
        s11 = axb ^ c;
        c11 = (a & b) | (a & c) | (b & c); // weight-2 carries
    }
    u64 s12, c12; // CSA(W, NW, 0)
    {
        const u64 a = Wdir, b = NWdir, c = 0ull;
        const u64 axb = a ^ b;
        s12 = axb ^ c;
        c12 = (a & b); // since c==0, carry reduces to a&b
    }

    // Combine the three "sum" bitboards to get ones and a new set of weight-2 carries.
    u64 ones, carry2_from_sums;
    {
        const u64 a = s10, b = s11, c = s12;
        const u64 axb = a ^ b;
        ones = axb ^ c; // 1's place
        carry2_from_sums = (a & b) | (a & c) | (b & c); // weight-2 carries
    }

    // Combine the three original weight-2 carry bitboards to get final twos (weight-2) and partial fours (weight-4).
    u64 sum2_carries, carry4_from_carries;
    {
        const u64 a = c10, b = c11, c = c12;
        const u64 axb = a ^ b;
        sum2_carries = axb ^ c; // 2's place contributions
        carry4_from_carries = (a & b) | (a & c) | (b & c); // 4's place contributions
    }

    // Add the two weight-2 contributions with a half-adder to get final twos and additional fours.
    const u64 twos  = sum2_carries ^ carry2_from_sums;
    const u64 fours = carry4_from_carries | (sum2_carries & carry2_from_sums);

    // Apply Game of Life rules:
    // - exactly two neighbors: (~ones) & twos & (~fours)
    // - exactly three neighbors: ones & twos & (~fours)
    const u64 alive      = C;
    const u64 exactly2   = (~ones) & twos & (~fours);
    const u64 exactly3   = (ones) & twos & (~fours);
    const u64 next_state = exactly3 | (exactly2 & alive);

    out[idx] = next_state;
}

// Host function to launch one simulation step.
// input  - pointer to device memory input grid (bit-packed 8x8 tiles in row-major tile order)
// output - pointer to device memory output grid (same layout)
// grid_dimensions - width/height of the square grid, in cells (power of two and > 512).
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of tiles per dimension (each tile is 8x8 cells).
    const int tile_dim = grid_dimensions >> 3; // divide by 8

    // Use a 2D launch with 256 threads per block for good occupancy on modern data center GPUs.
    const dim3 block(32, 8, 1);
    const dim3 grid((tile_dim + block.x - 1) / block.x,
                    (tile_dim + block.y - 1) / block.y,
                    1);

    game_of_life_step_kernel<<<grid, block>>>(reinterpret_cast<const u64*>(input),
                                              reinterpret_cast<u64*>(output),
                                              tile_dim);
}