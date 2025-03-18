// This CUDA implementation of Conway’s Game of Life processes
// one 8×8 tile per thread. The grid is stored in bit‐packed form,
// with each std::uint64_t encoding an 8×8 block of cells.
// Each cell is represented as a single bit (1 = alive, 0 = dead).
// The new generation is computed according to the usual rules,
// treating all cells outside the grid as dead.
//
// The grid is logically divided into tiles; if the overall grid has
// dimensions “grid_dimensions × grid_dimensions” (in cells) then
// there are (grid_dimensions/8) × (grid_dimensions/8) tiles.
// Each thread loads its tile (tile_mid) plus its 8 immediate neighbors:
// top, bottom, left, right and the 4 diagonal tiles. Boundary conditions
// are handled by substituting zeros where a neighbor is missing.
//
// Within each tile, we process row‐by‐row. For each tile row (0..7)
// the update of each cell (column 0..7) depends on the three “halo” rows:
// the row immediately above, the current row, and the row immediately below.
// For a given row in the 8×8 tile, its “halo” row is built by combining
// the central 8 bits (from the tile itself) with one extra bit on each side,
// taken from the neighboring (left/right) tile, if available.
// We store these “assembled” rows in a 10‐bit integer, where bits 1..8
// (with bit position 1 corresponding to the first column of the tile)
// hold the central cells, bit 9 holds the left halo cell, and bit 0 holds
// the right halo cell.
//
// For each cell in the central tile row (stored at bit position j+1),
// the neighbor sum is computed by summing the three adjacent bits in
// the halo row from above, the two adjacent bits in the current halo row
// (excluding the center), and the three adjacent bits in the halo row below.
// Then the standard rule is applied: the cell becomes alive if the sum
// is exactly 3, or if the sum is 2 and the current cell is alive.
//
// The run_game_of_life() function sets up the 2D grid of threads to cover
// all tiles and launches the CUDA kernel.

#include <cstdint>
#include <cuda_runtime.h>

// Device inline function to extract an 8-bit row from a tile.
// Each tile is stored as a 64‐bit integer. Row “r” is stored in bits [8*r,8*r+7].
// The least‐significant bit of the extracted byte corresponds to column 0.
__device__ __forceinline__
unsigned char getRow(const std::uint64_t tile, int r) {
    return static_cast<unsigned char>((tile >> (r << 3)) & 0xFFULL);
}

// Device inline function to assemble a 10-bit row from a center 8-bit value
// and its left/right halo bits. The returned integer uses bit positions:
// bit 9: left halo, bits 8..1: center row, bit 0: right halo.
__device__ __forceinline__
int assembleRow(unsigned char center, unsigned char left, unsigned char right) {
    return (static_cast<int>(left) << 9) | (static_cast<int>(center) << 1) | static_cast<int>(right);
}

// CUDA kernel to compute one Game of Life step on the bit-packed grid.
// The parameter "tile_count" is the number of tiles per row (grid_dimensions/8).
__global__
void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int tile_count)
{
    // Compute our tile coordinates.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_x >= tile_count || tile_y >= tile_count) return;
    int index = tile_y * tile_count + tile_x;

    // Load the tile for the current block.
    std::uint64_t tile_mid = input[index];

    // Load horizontal and vertical neighbor tiles.
    std::uint64_t tile_top          = (tile_y > 0)              ? input[(tile_y - 1) * tile_count + tile_x] : 0ULL;
    std::uint64_t tile_bottom       = (tile_y < tile_count - 1) ? input[(tile_y + 1) * tile_count + tile_x] : 0ULL;
    std::uint64_t tile_left         = (tile_x > 0)              ? input[tile_y * tile_count + (tile_x - 1)] : 0ULL;
    std::uint64_t tile_right        = (tile_x < tile_count - 1) ? input[tile_y * tile_count + (tile_x + 1)] : 0ULL;
    std::uint64_t tile_top_left     = (tile_y > 0 && tile_x > 0)              ? input[(tile_y - 1) * tile_count + (tile_x - 1)] : 0ULL;
    std::uint64_t tile_top_right    = (tile_y > 0 && tile_x < tile_count - 1) ? input[(tile_y - 1) * tile_count + (tile_x + 1)] : 0ULL;
    std::uint64_t tile_bottom_left  = (tile_y < tile_count - 1 && tile_x > 0) ? input[(tile_y + 1) * tile_count + (tile_x - 1)] : 0ULL;
    std::uint64_t tile_bottom_right = (tile_y < tile_count - 1 && tile_x < tile_count - 1) ? input[(tile_y + 1) * tile_count + (tile_x + 1)] : 0ULL;

    // new_tile will store the updated 8×8 tile.
    std::uint64_t new_tile = 0ULL;

    // Process each row (r = 0..7) of the tile independently.
    for (int r = 0; r < 8; r++) {
        // For the three halo rows: "up", "mid", and "bot".
        // Their "center" bytes come from:
        //   - For the "up" row: if r == 0 then from the bottom row of tile_top;
        //     otherwise from row r-1 of tile_mid.
        //   - The "mid" row always comes from row r of tile_mid.
        //   - For the "bot" row: if r == 7 then from the top row of tile_bottom;
        //     otherwise from row r+1 of tile_mid.
        unsigned char up_center, mid_center, bot_center;
        unsigned char up_left, mid_left, bot_left;
        unsigned char up_right, mid_right, bot_right;

        if (r == 0) {
            // For the row above the first row of the tile.
            up_center = getRow(tile_top, 7);
            // Left halo: use bottom row of tile_top_left if exists.
            up_left  = (tile_y > 0 && tile_x > 0) ? ((getRow(tile_top_left, 7) >> 7) & 1) : 0;
            // Right halo: use bottom row of tile_top_right if exists.
            up_right = (tile_y > 0 && tile_x < tile_count - 1) ? (getRow(tile_top_right, 7) & 1) : 0;
        } else {
            up_center = getRow(tile_mid, r - 1);
            up_left  = (tile_x > 0)         ? ((getRow(tile_left, r - 1) >> 7) & 1) : 0;
            up_right = (tile_x < tile_count - 1) ? (getRow(tile_right, r - 1) & 1) : 0;
        }

        // "Mid" row always comes from tile_mid.
        mid_center = getRow(tile_mid, r);
        mid_left  = (tile_x > 0)         ? ((getRow(tile_left, r) >> 7) & 1) : 0;
        mid_right = (tile_x < tile_count - 1) ? (getRow(tile_right, r) & 1) : 0;

        if (r == 7) {
            bot_center = getRow(tile_bottom, 0);
            bot_left  = (tile_y < tile_count - 1 && tile_x > 0) ? ((getRow(tile_bottom_left, 0) >> 7) & 1) : 0;
            bot_right = (tile_y < tile_count - 1 && tile_x < tile_count - 1) ? (getRow(tile_bottom_right, 0) & 1) : 0;
        } else {
            bot_center = getRow(tile_mid, r + 1);
            bot_left  = (tile_x > 0)         ? ((getRow(tile_left, r + 1) >> 7) & 1) : 0;
            bot_right = (tile_x < tile_count - 1) ? (getRow(tile_right, r + 1) & 1) : 0;
        }

        // Assemble each halo row into a 10-bit integer.
        // Bits: 9 = left halo, 8..1 = center row, 0 = right halo.
        int up_row  = assembleRow(up_center,  up_left,  up_right);
        int mid_row = assembleRow(mid_center, mid_left, mid_right);
        int bot_row = assembleRow(bot_center, bot_left, bot_right);

        // For each cell in the central row (columns 0..7, which are at bit positions 1..8 in mid_row),
        // sum the states of its eight neighbors. For a cell at column j (central bit at position j+1),
        // the neighbors are the three bits from up_row at positions (j, j+1, j+2),
        // the two bits from mid_row at positions j and j+2,
        // and the three bits from bot_row at positions (j, j+1, j+2).
        unsigned char new_row = 0;
        for (int j = 0; j < 8; j++) {
            int cell = (mid_row >> (j + 1)) & 1;
            int sum = ((up_row  >> j)     & 1) + ((up_row  >> (j + 1)) & 1) + ((up_row  >> (j + 2)) & 1)
                    + ((mid_row >> j)     & 1) + ((mid_row >> (j + 2)) & 1)
                    + ((bot_row >> j)     & 1) + ((bot_row >> (j + 1)) & 1) + ((bot_row >> (j + 2)) & 1);
            // Apply Conway's rules:
            // A cell becomes alive if it has exactly 3 neighbors,
            // or if it is already alive and has exactly 2 neighbors.
            int new_cell = (sum == 3) || (sum == 2 && cell == 1);
            new_row |= (new_cell << j);
        }
        // Pack the computed new_row (8 bits) into the corresponding row of new_tile.
        new_tile |= (static_cast<std::uint64_t>(new_row) << (r << 3));
    }
    // Write the new tile to the output grid.
    output[index] = new_tile;
}

// Host function that executes one step of Conway's Game of Life
// using the CUDA kernel above.
// 'input' and 'output' point to device memory containing the bit-packed grid,
// and 'grid_dimensions' is the width/height of the grid in cells.
// The grid is divided into 8×8 tiles.
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of tiles per row.
    int tile_count = grid_dimensions >> 3;  // equivalent to grid_dimensions / 8

    // Define a 2D CUDA grid of threads. Each thread processes one 8×8 tile.
    // Here we choose a 16×16 block size for a good occupancy.
    dim3 blockSize(16, 16);
    dim3 gridSize((tile_count + blockSize.x - 1) / blockSize.x,
                  (tile_count + blockSize.y - 1) / blockSize.y);

    // Launch the kernel.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, tile_count);
    // Note: Host-device synchronization is handled by the caller.
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
