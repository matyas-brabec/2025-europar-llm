// Conway’s Game of Life CUDA implementation with bit‐packed 8x8 tiles.
// Each std::uint64_t encodes an 8×8 block of cells in row‐major order,
// where bit 0 of each 8‐bit “row” corresponds to the left‐most cell 
// and bit 7 to the right‐most cell.
// Neighbor cells outside the grid are considered dead.
//
// For each tile (of 8×8 cells), the next state is computed by examining
// the 8 neighbors of each cell. Since cells near the tile boundary
// may have neighbors in adjacent tiles, we load the 8 neighboring tiles
// (if available) for left, right, top, bottom, and corners.
//
// We pre‐extract the 8 rows (each stored in an 8‐bit value) from the current
// tile and its neighbors. Then, for each row “r” (0 ≤ r < 8) of the current tile,
// we build three “padded” rows (10‐bits wide) representing the row above (padded_up),
// the current row (padded_mid), and the row below (padded_down). In each padded row,
// bits 1..8 correspond to the 8 cells of that row, while bit 0 is the left neighbor
// and bit 9 is the right neighbor, obtained from the adjacent tile’s row (if available).
//
// For a given cell at column j in the current tile row r (which maps to padded_mid bit j+1),
// its eight neighbors come from the bits:
//    padded_up    bits [j, j+1, j+2]
//    padded_mid   bits [j,       j+2]   (skip the center cell at j+1)
//    padded_down  bits [j, j+1, j+2]
// Their sum determines the next state, according to:
//    next_state = 1 if (neighbors == 3) or (neighbors == 2 and cell == 1),
//    else next_state = 0.
//
// The kernel is launched with one thread per tile. The grid (of cells) dimensions
// are given in cells (always a power of 2, ≥512). Because each tile is 8×8 cells,
// the tile grid dimensions are (grid_dimensions/8)×(grid_dimensions/8).
//
// This implementation avoids shared or texture memory and emphasizes
// optimizing the inner loops and memory access for modern NVIDIA GPUs.
#include <cstdint>
#include <cuda_runtime.h>

// __global__ kernel that computes one Game-of-Life update for each 8x8 tile.
// Parameters:
//   input: pointer to bit-packed input grid (each uint64_t holds one 8x8 tile)
//   output: pointer to bit-packed output grid (next state)
//   tiles_dim: number of tiles per row (grid_dimensions/8)
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_dim)
{
    // Compute tile coordinate in the tile grid.
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_row >= tiles_dim || tile_col >= tiles_dim)
        return;
        
    // Compute linear index into the tile array.
    int tile_index = tile_row * tiles_dim + tile_col;
    
    // Each tile is stored as a uint64_t representing 8x8 cells.
    std::uint64_t tile_center = input[tile_index];
    
    // Precompute the 8 rows (8-bit each) for the current tile.
    // Row r is stored in bits [8*r, 8*r+7] of tile_center.
    uint8_t center[8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
    {
        center[r] = (uint8_t)((tile_center >> (r * 8)) & 0xFF);
    }
    
    // For horizontal neighbors, precompute left and right tiles' rows.
    uint8_t left_tile[8], right_tile[8];
    if (tile_col > 0)
    {
        std::uint64_t tile_left = input[tile_row * tiles_dim + (tile_col - 1)];
#pragma unroll
        for (int r = 0; r < 8; ++r)
        {
            left_tile[r] = (uint8_t)((tile_left >> (r * 8)) & 0xFF);
        }
    }
    else
    {
#pragma unroll
        for (int r = 0; r < 8; ++r)
            left_tile[r] = 0;
    }
    
    if (tile_col < tiles_dim - 1)
    {
        std::uint64_t tile_right = input[tile_row * tiles_dim + (tile_col + 1)];
#pragma unroll
        for (int r = 0; r < 8; ++r)
        {
            right_tile[r] = (uint8_t)((tile_right >> (r * 8)) & 0xFF);
        }
    }
    else
    {
#pragma unroll
        for (int r = 0; r < 8; ++r)
            right_tile[r] = 0;
    }
    
    // For vertical neighbors, we need the bottom (for top neighbor) and top (for bottom neighbor)
    // rows of the adjacent tiles.
    uint8_t top_tile_row = 0, top_left_row = 0, top_right_row = 0;
    if (tile_row > 0)
    {
        std::uint64_t tile_top = input[(tile_row - 1) * tiles_dim + tile_col];
        // For current tile row 0, the row above is the bottom row (row 7) of the top tile.
        top_tile_row = (uint8_t)((tile_top >> (7 * 8)) & 0xFF);
        
        if (tile_col > 0)
        {
            std::uint64_t tile_top_left = input[(tile_row - 1) * tiles_dim + (tile_col - 1)];
            top_left_row = (uint8_t)((tile_top_left >> (7 * 8)) & 0xFF);
        }
        
        if (tile_col < tiles_dim - 1)
        {
            std::uint64_t tile_top_right = input[(tile_row - 1) * tiles_dim + (tile_col + 1)];
            top_right_row = (uint8_t)((tile_top_right >> (7 * 8)) & 0xFF);
        }
    }
    
    uint8_t bottom_tile_row = 0, bottom_left_row = 0, bottom_right_row = 0;
    if (tile_row < tiles_dim - 1)
    {
        std::uint64_t tile_bottom = input[(tile_row + 1) * tiles_dim + tile_col];
        // For current tile row 7, the row below is the top row (row 0) of the bottom tile.
        bottom_tile_row = (uint8_t)(tile_bottom & 0xFF);
        
        if (tile_col > 0)
        {
            std::uint64_t tile_bottom_left = input[(tile_row + 1) * tiles_dim + (tile_col - 1)];
            bottom_left_row = (uint8_t)(tile_bottom_left & 0xFF);
        }
        if (tile_col < tiles_dim - 1)
        {
            std::uint64_t tile_bottom_right = input[(tile_row + 1) * tiles_dim + (tile_col + 1)];
            bottom_right_row = (uint8_t)(tile_bottom_right & 0xFF);
        }
    }
    
    // new_tile will accumulate the new state for the current tile.
    std::uint64_t new_tile = 0;
    
    // For each row (r) in the current tile, build padded rows
    // which are 10-bit values: bits 1..8 come from the base 8-bit row,
    // bit 0 is from the left neighbor and bit 9 from the right neighbor.
    // This is done for the row above (padded_up), current row (padded_mid),
    // and row below (padded_down) used for neighbor summing.
#pragma unroll
    for (int r = 0; r < 8; ++r)
    {
        uint16_t padded_up, padded_mid, padded_down;
        // --- Build padded_up row ---
        if (r == 0)
        {
            // For the top row of the current tile, the row above comes from the top neighbor.
            // Use top_tile_row as base. Left border from top-left; right border from top-right.
            uint8_t base = top_tile_row; // 8 bits from top tile's bottom row.
            uint8_t left_bit = (tile_row > 0 && tile_col > 0) ? ((top_left_row >> 7) & 1) : 0;
            uint8_t right_bit = (tile_row > 0 && tile_col < tiles_dim - 1) ? (top_right_row & 1) : 0;
            padded_up = (uint16_t)left_bit | (((uint16_t)base) << 1) | (((uint16_t)right_bit) << 9);
        }
        else
        {
            // For r > 0, the row above is in the current tile (row r-1).
            uint8_t base = center[r - 1];
            uint8_t left_bit = (tile_col > 0) ? ((left_tile[r - 1] >> 7) & 1) : 0;
            uint8_t right_bit = (tile_col < tiles_dim - 1) ? (right_tile[r - 1] & 1) : 0;
            padded_up = (uint16_t)left_bit | (((uint16_t)base) << 1) | (((uint16_t)right_bit) << 9);
        }
        
        // --- Build padded_mid row (current row) ---
        {
            uint8_t base = center[r];
            uint8_t left_bit = (tile_col > 0) ? ((left_tile[r] >> 7) & 1) : 0;
            uint8_t right_bit = (tile_col < tiles_dim - 1) ? (right_tile[r] & 1) : 0;
            padded_mid = (uint16_t)left_bit | (((uint16_t)base) << 1) | (((uint16_t)right_bit) << 9);
        }
        
        // --- Build padded_down row ---
        if (r == 7)
        {
            // For the bottom row of the current tile, get row below from bottom tile.
            uint8_t base = bottom_tile_row; // from bottom tile row 0.
            uint8_t left_bit = (tile_row < tiles_dim - 1 && tile_col > 0) ? ((bottom_left_row >> 7) & 1) : 0;
            uint8_t right_bit = (tile_row < tiles_dim - 1 && tile_col < tiles_dim - 1) ? (bottom_right_row & 1) : 0;
            padded_down = (uint16_t)left_bit | (((uint16_t)base) << 1) | (((uint16_t)right_bit) << 9);
        }
        else
        {
            // Otherwise, the row below is in the current tile (row r+1).
            uint8_t base = center[r + 1];
            uint8_t left_bit = (tile_col > 0) ? ((left_tile[r + 1] >> 7) & 1) : 0;
            uint8_t right_bit = (tile_col < tiles_dim - 1) ? (right_tile[r + 1] & 1) : 0;
            padded_down = (uint16_t)left_bit | (((uint16_t)base) << 1) | (((uint16_t)right_bit) << 9);
        }
        
        // For each cell (column j) in row r, compute the neighbor count and next state.
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int count = 0;
            // Sum three bits from the padded_up row: columns j, j+1, j+2.
            count += (int)((padded_up >> j) & 1);
            count += (int)((padded_up >> (j + 1)) & 1);
            count += (int)((padded_up >> (j + 2)) & 1);
            // From padded_mid row: skip the center cell (at j+1).
            count += (int)((padded_mid >> j) & 1);
            count += (int)((padded_mid >> (j + 2)) & 1);
            // From padded_down row.
            count += (int)((padded_down >> j) & 1);
            count += (int)((padded_down >> (j + 1)) & 1);
            count += (int)((padded_down >> (j + 2)) & 1);
            
            // Extract the current cell value from padded_mid at position (j+1)
            uint8_t current = (uint8_t)((padded_mid >> (j + 1)) & 1);
            // Apply Game-of-Life rules:
            // A cell becomes alive if it has exactly 3 live neighbors,
            // or if it is currently alive and has exactly 2 live neighbors.
            uint8_t new_cell = ((count == 3) || (count == 2 && current)) ? 1 : 0;
            
            // Set the corresponding bit in new_tile (bit position: r*8 + j).
            new_tile |= ((std::uint64_t)new_cell << (r * 8 + j));
        }
    }
    
    // Write the computed tile to output.
    output[tile_index] = new_tile;
}

// Host function that runs one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   input:  pointer to input grid (allocated with cudaMalloc) in bit-packed format.
//   output: pointer to output grid (allocated with cudaMalloc), bit-packed.
//   grid_dimensions: grid width (and height) in cells. Must be a power of 2, ≥512.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Each uint64_t tile represents an 8x8 block.
    int tiles_dim = grid_dimensions / 8;
    
    // Define a 2D thread block. Using 16x16 threads gives a good occupancy.
    dim3 blockDim(16, 16);
    // Compute grid dimensions (in tiles) so that each thread maps to one tile.
    dim3 gridDim((tiles_dim + blockDim.x - 1) / blockDim.x,
                 (tiles_dim + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, tiles_dim);
    
    // No host-device synchronization here; it is handled externally.
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
