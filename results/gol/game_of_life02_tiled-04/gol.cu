#include <cuda_runtime.h>
#include <cstdint>

// Device function to extract an 8‐bit row from a 64‐bit tile.
// Each tile encodes an 8x8 block of cells; row 0 is in bits [0,7], row 1 in bits [8,15], etc.
__device__ inline unsigned char get_row(std::uint64_t tile, int r) {
    return (unsigned char)((tile >> (r * 8)) & 0xFF);
}

//---------------------------------------------------------------------
// CUDA kernel implementing one step of Conway's Game of Life.
// The grid is stored in bit‐packed form: each std::uint64_t encodes an 8x8 block.
// For a cell, neighbors outside the grid are considered dead.
// We compute the next state for every tile based on the 8 neighboring tiles.
// For performance, we pre-load the 9 relevant tiles (central and 8 neighbors),
// then for each row within the 8x8 tile we build an effective 10‐bit value
// that includes one extra bit on the left and right. This allows us to
// extract a 3‐bit horizontal window per cell without conditionals.
// The cell update rule is:
//    new = 1 if (neighbor_count == 3) or (current==1 and neighbor_count==2)
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each tile is 8x8 cells.
    // The number of tiles per grid row/column is grid_dimensions/8.
    int tile_grid_dim = grid_dimensions >> 3; // equivalent to grid_dimensions / 8
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_x >= tile_grid_dim || tile_y >= tile_grid_dim)
        return;

    // Compute linear index for the current tile.
    int idx = tile_y * tile_grid_dim + tile_x;

    // Load the central tile and its 8 neighbors.
    // If a neighbor is out-of-bound, use 0 (all cells dead).
    std::uint64_t t_center = input[idx];
    std::uint64_t t_left   = (tile_x > 0) ? input[tile_y * tile_grid_dim + (tile_x - 1)] : 0;
    std::uint64_t t_right  = (tile_x < tile_grid_dim - 1) ? input[tile_y * tile_grid_dim + (tile_x + 1)] : 0;
    std::uint64_t t_top    = (tile_y > 0) ? input[(tile_y - 1) * tile_grid_dim + tile_x] : 0;
    std::uint64_t t_bottom = (tile_y < tile_grid_dim - 1) ? input[(tile_y + 1) * tile_grid_dim + tile_x] : 0;
    std::uint64_t t_top_left = (tile_x > 0 && tile_y > 0) ?
        input[(tile_y - 1) * tile_grid_dim + (tile_x - 1)] : 0;
    std::uint64_t t_top_right = (tile_x < tile_grid_dim - 1 && tile_y > 0) ?
        input[(tile_y - 1) * tile_grid_dim + (tile_x + 1)] : 0;
    std::uint64_t t_bottom_left = (tile_x > 0 && tile_y < tile_grid_dim - 1) ?
        input[(tile_y + 1) * tile_grid_dim + (tile_x - 1)] : 0;
    std::uint64_t t_bottom_right = (tile_x < tile_grid_dim - 1 && tile_y < tile_grid_dim - 1) ?
        input[(tile_y + 1) * tile_grid_dim + (tile_x + 1)] : 0;

    // The new state of the tile will be built in this 64-bit value.
    std::uint64_t new_tile = 0;

    // Process each of the 8 rows in the tile.
    for (int r = 0; r < 8; r++) {
        // We'll build effective 10-bit values representing the horizontal data of a given row.
        // In the effective value, bit positions are as follows:
        //   bit0: left neighbor cell (from adjacent tile, if available)
        //   bits 1..8: the 8 cells in the row (bit0 of row becomes bit1, bit7 becomes bit8)
        //   bit9: right neighbor cell (from adjacent tile, if available)
        //
        // This lets us extract a contiguous 3-bit window for each cell by shifting right by the column index.
        //
        // Determine the "top" row that contributes to the neighbors of cells in row r.
        unsigned char top_row, top_left_row, top_right_row;
        if (r == 0) {
            // For the first row of the current tile, the top neighbors come from the tile above;
            // use row 7 of the above tile (or from the corresponding neighbors of that tile).
            top_row = get_row(t_top, 7);
            top_left_row = get_row(t_top_left, 7);
            top_right_row = get_row(t_top_right, 7);
        } else {
            // For rows 1 to 7, the top neighbor row is the previous row of the central tile,
            // with horizontal neighbors taken from adjacent tiles.
            top_row = get_row(t_center, r - 1);
            top_left_row = (tile_x > 0) ? get_row(t_left, r - 1) : 0;
            top_right_row = (tile_x < tile_grid_dim - 1) ? get_row(t_right, r - 1) : 0;
        }

        // The "mid" row is the current row of the central tile.
        unsigned char mid_row = get_row(t_center, r);
        unsigned char mid_left_row = (tile_x > 0) ? get_row(t_left, r) : 0;
        unsigned char mid_right_row = (tile_x < tile_grid_dim - 1) ? get_row(t_right, r) : 0;

        // Determine the "bottom" row that contributes to the neighbors.
        unsigned char bottom_row, bottom_left_row, bottom_right_row;
        if (r == 7) {
            // For the last row of the tile, the bottom neighbors come from the tile below;
            // use row 0 of that tile.
            bottom_row = get_row(t_bottom, 0);
            bottom_left_row = get_row(t_bottom_left, 0);
            bottom_right_row = get_row(t_bottom_right, 0);
        } else {
            // For other rows, use row r+1 of the central tile.
            bottom_row = get_row(t_center, r + 1);
            bottom_left_row = (tile_x > 0) ? get_row(t_left, r + 1) : 0;
            bottom_right_row = (tile_x < tile_grid_dim - 1) ? get_row(t_right, r + 1) : 0;
        }

        // Construct effective 10-bit values for the top, mid, and bottom rows.
        // For the left neighbor bit, take bit7 of the corresponding left row.
        // For the right neighbor bit, take bit0 of the corresponding right row.
        unsigned int eff_top = (((unsigned int)(top_left_row >> 7)) & 1u)
                             | (((unsigned int)top_row) << 1)
                             | ((((unsigned int)top_right_row) & 1u) << 9);
        unsigned int eff_mid = (((unsigned int)(mid_left_row >> 7)) & 1u)
                             | (((unsigned int)mid_row) << 1)
                             | ((((unsigned int)mid_right_row) & 1u) << 9);
        unsigned int eff_bottom = (((unsigned int)(bottom_left_row >> 7)) & 1u)
                                | (((unsigned int)bottom_row) << 1)
                                | ((((unsigned int)bottom_right_row) & 1u) << 9);

        // This new_row will hold the updated state for row r of the current tile.
        unsigned char new_row = 0;

        // Process each of the 8 columns in this row.
        // For each cell, we extract a 3-bit window from each effective row.
        // The 3-bit window (for a given effective value) is taken as: (eff >> c) & 7.
        // For the mid row window, we subtract the center cell (because we don't count the cell itself).
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            unsigned int top_win = (eff_top >> c) & 7u;
            unsigned int mid_win = (eff_mid >> c) & 7u;
            unsigned int bot_win = (eff_bottom >> c) & 7u;

            // Count the live bits in each 3-bit window.
            unsigned int top_count = ((top_win >> 0) & 1u) + ((top_win >> 1) & 1u) + ((top_win >> 2) & 1u);
            unsigned int mid_count = ((mid_win >> 0) & 1u) + ((mid_win >> 1) & 1u) + ((mid_win >> 2) & 1u);
            // Subtract the current cell from the mid row window.
            mid_count -= ((mid_row >> c) & 1u);
            unsigned int bot_count = ((bot_win >> 0) & 1u) + ((bot_win >> 1) & 1u) + ((bot_win >> 2) & 1u);
            unsigned int total = top_count + mid_count + bot_count;

            // Get current cell state (0 or 1).
            unsigned int cell = (mid_row >> c) & 1u;
            // Apply Conway's rules:
            // A cell becomes alive if it has exactly 3 neighbors, or if it is alive and has exactly 2 neighbors.
            unsigned int new_cell = (total == 3) || (cell && (total == 2));
            new_row |= (new_cell << c);
        }
        // Pack the new 8-bit row into its proper position in the 64-bit output tile.
        new_tile |= ((std::uint64_t)new_row) << (r * 8);
    }

    // Write back the computed tile to the output grid.
    output[idx] = new_tile;
}

//---------------------------------------------------------------------
// Host function to run one Game of Life iteration.
// The input and output grids are stored as bit-packed arrays of std::uint64_t,
// where each element encodes an 8x8 block of cells. Grid dimensions are given in cells.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of tiles in one dimension.
    int tile_grid_dim = grid_dimensions >> 3;  // grid_dimensions / 8

    // Choose block and grid dimensions.
    // We use 16x16 threads per block to cover the tile grid.
    dim3 blockDim(16, 16);
    dim3 gridDim((tile_grid_dim + blockDim.x - 1) / blockDim.x,
                 (tile_grid_dim + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel.
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
