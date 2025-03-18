#include <cuda_runtime.h>
#include <cstdint>

//---------------------------------------------------------------------
// CUDA kernel to compute one generation of Conway's Game of Life.
// The grid is bit‐packed: each std::uint64_t holds an 8×8 tile of cells.
// Within each tile, cell (r,c) (0 ≤ r,c < 8) is stored in bit position (8*r + c),
// where a bit value 1 means alive and 0 means dead.
// For each tile, the kernel gathers its own cell bits and the surrounding
// neighbor cell bits (from adjacent tiles) into a 10×10 padded array.
// Then, it computes the next generation for the 8×8 center cells using
// the standard Game of Life rules and writes the result back in a bit‐packed format.
//---------------------------------------------------------------------
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // Each tile is 8x8 cells. Calculate the number of tiles per row.
    int tiles_per_row = grid_dimensions >> 3; // equivalent to grid_dimensions / 8

    // Determine the tile position (tx, ty) in the grid of tiles.
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tiles_per_row || ty >= tiles_per_row)
        return;

    // Compute the 1D index for the center tile.
    int tile_index = ty * tiles_per_row + tx;
    std::uint64_t tile_center = input[tile_index];

    // Extract the 8 rows (each as an 8-bit value) from the center tile.
    // Each row is stored in 8 bits of the 64-bit value.
    unsigned char center_rows[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        center_rows[r] = (unsigned char)((tile_center >> (8 * r)) & 0xFF);
    }

    // Build a 10x10 padded local cell array.
    // The center 8x8 area (indices [1..8][1..8]) corresponds to the current tile.
    // The outer border is filled from adjacent tiles; if an adjacent tile doesn't exist (edge of grid),
    // the cell is assumed dead (0).
    unsigned char padded[10][10];
    // Initialize the padded array to 0.
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        #pragma unroll
        for (int j = 0; j < 10; j++) {
            padded[i][j] = 0;
        }
    }

    // Fill the center 8x8 region with bits from the current (center) tile.
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            padded[r+1][c+1] = (center_rows[r] >> c) & 1;
        }
    }

    // Helper lambda to extract an 8-bit row from a 64-bit tile,
    // given a row index (0 to 7). (Bit extraction: row = (tile >> (8*row)) & 0xFF)
    auto getRow = [](std::uint64_t tile, int row) -> unsigned char {
        return (unsigned char)((tile >> (8 * row)) & 0xFF);
    };

    // Fill the top padded row (row index 0) from the tile above, if it exists.
    if (ty > 0) {
        std::uint64_t tile_top = input[(ty - 1) * tiles_per_row + tx];
        // The bottom row (row 7) of the top tile corresponds to the padded top row.
        unsigned char top_row = getRow(tile_top, 7);
        for (int c = 0; c < 8; c++) {
            padded[0][c+1] = (top_row >> c) & 1;
        }
    }
    // Fill the bottom padded row (row index 9) from the tile below, if it exists.
    if (ty < tiles_per_row - 1) {
        std::uint64_t tile_bottom = input[(ty + 1) * tiles_per_row + tx];
        // The top row (row 0) of the bottom tile corresponds to the padded bottom row.
        unsigned char bottom_row = getRow(tile_bottom, 0);
        for (int c = 0; c < 8; c++) {
            padded[9][c+1] = (bottom_row >> c) & 1;
        }
    }
    // Fill the left padded column (column index 0) from the tile to the left, if it exists.
    if (tx > 0) {
        std::uint64_t tile_left = input[ty * tiles_per_row + (tx - 1)];
        for (int r = 0; r < 8; r++) {
            // The rightmost bit (bit 7) of each row in the left tile.
            unsigned char left_bit = (unsigned char)((tile_left >> (8 * r + 7)) & 1);
            padded[r+1][0] = left_bit;
        }
    }
    // Fill the right padded column (column index 9) from the tile to the right, if it exists.
    if (tx < tiles_per_row - 1) {
        std::uint64_t tile_right = input[ty * tiles_per_row + (tx + 1)];
        for (int r = 0; r < 8; r++) {
            // The leftmost bit (bit 0) of each row in the right tile.
            unsigned char right_bit = (unsigned char)((tile_right >> (8 * r + 0)) & 1);
            padded[r+1][9] = right_bit;
        }
    }
    // Fill the four corner cells.
    // Top-left corner.
    if (tx > 0 && ty > 0) {
        std::uint64_t tile_top_left = input[(ty - 1) * tiles_per_row + (tx - 1)];
        unsigned char tl_bit = (unsigned char)((tile_top_left >> (8 * 7 + 7)) & 1);
        padded[0][0] = tl_bit;
    }
    // Top-right corner.
    if (tx < tiles_per_row - 1 && ty > 0) {
        std::uint64_t tile_top_right = input[(ty - 1) * tiles_per_row + (tx + 1)];
        unsigned char tr_bit = (unsigned char)((tile_top_right >> (8 * 7 + 0)) & 1);
        padded[0][9] = tr_bit;
    }
    // Bottom-left corner.
    if (tx > 0 && ty < tiles_per_row - 1) {
        std::uint64_t tile_bottom_left = input[(ty + 1) * tiles_per_row + (tx - 1)];
        unsigned char bl_bit = (unsigned char)((tile_bottom_left >> (8 * 0 + 7)) & 1);
        padded[9][0] = bl_bit;
    }
    // Bottom-right corner.
    if (tx < tiles_per_row - 1 && ty < tiles_per_row - 1) {
        std::uint64_t tile_bottom_right = input[(ty + 1) * tiles_per_row + (tx + 1)];
        unsigned char br_bit = (unsigned char)((tile_bottom_right >> (8 * 0 + 0)) & 1);
        padded[9][9] = br_bit;
    }

    // Compute the next generation for each cell in the center 8×8 region.
    // The computed new cell values will be packed into a new 64‐bit tile.
    std::uint64_t result_tile = 0;
    // Loop over the center region: padded rows 1 to 8 and columns 1 to 8.
    for (int r = 1; r <= 8; r++) {
        unsigned char new_row = 0;
        #pragma unroll
        for (int c = 1; c <= 8; c++) {
            // Sum the 8 neighbors.
            int sum =
                padded[r-1][c-1] + padded[r-1][c] + padded[r-1][c+1] +
                padded[r][c-1]               +         padded[r][c+1] +
                padded[r+1][c-1] + padded[r+1][c] + padded[r+1][c+1];
            // Each cell follows the Game of Life rules:
            // - Live cell with 2 or 3 neighbors survives.
            // - Dead cell with exactly 3 neighbors becomes alive.
            // Otherwise, the cell becomes or remains dead.
            unsigned char current = padded[r][c];
            unsigned char new_state = ((current != 0) && (sum == 2 || sum == 3)) || ((current == 0) && (sum == 3));
            // Set the computed state at bit position (c - 1) for the new row.
            new_row |= (new_state << (c - 1));
        }
        // Pack the new row into the result tile at row (r - 1).
        result_tile |= (static_cast<std::uint64_t>(new_row) << (8 * (r - 1)));
    }

    // Write the computed tile to the output grid.
    output[tile_index] = result_tile;
}

//---------------------------------------------------------------------
// Host function that runs one step of Conway's Game of Life.
// It launches the CUDA kernel which processes the grid of 8x8 tiles.
// The grid is assumed to be bit-packed: each std::uint64_t represents an 8×8 tile.
//---------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                   std::uint64_t* output,
                                   int grid_dimensions)
{
    // Calculate the number of tiles per row (grid_dimensions / 8).
    int tiles_per_row = grid_dimensions >> 3;

    // Define a thread block size; chosen as 16x16 for good occupancy.
    dim3 blockDim(16, 16);
    // Calculate grid dimensions based on total number of tiles.
    dim3 gridDim((tiles_per_row + blockDim.x - 1) / blockDim.x,
                 (tiles_per_row + blockDim.y - 1) / blockDim.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled by the caller.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
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
