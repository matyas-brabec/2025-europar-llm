#include <cstdint>
#include <cuda_runtime.h>

// Kernel to compute one generation of Conway's Game of Life on a bit‐packed grid.
// Each std::uint64_t from the input represents an 8×8 tile. The 64 bits are arranged
// in row‐major order with row 0 in the most significant byte and row 7 in the least significant byte.
// Within each byte, bit 7 is the leftmost cell and bit 0 is the rightmost.
// Each thread processes one tile by reading its own tile (center) and its 8 neighbors:
//   a  b  c
//   d  e  f
//   g  h  i
// Tiles outside the grid are treated as all dead (0).
// For each cell in the central tile, we build an “extended” row (of 10 cells) that consists of
// the proper left and right border bits fetched from neighbor tiles if available.
// Then, using the three extended rows (above, current, below), we sum the eight neighboring bits
// for every cell and apply the Game of Life rules:
//   - Live cell survives if it has 2 or 3 live neighbors.
//   - Dead cell becomes live if it has exactly 3 live neighbors.
//   - Otherwise, the cell dies.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // The grid is divided into 8×8 tiles.
    // Calculate number of tiles per row (and per column) by dividing cell dimensions by 8.
    int tiles_per_row = grid_dimensions >> 3; // grid_dimensions / 8

    // Compute tile coordinates for this thread.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_x >= tiles_per_row || tile_y >= tiles_per_row)
        return;
    
    // Compute linear tile index.
    int tile_index = tile_y * tiles_per_row + tile_x;
    
    // Load neighbor tiles with proper boundary checks.
    // Tiles are arranged as:
    //      a      b      c
    //      d      e      f
    //      g      h      i
    std::uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i_val = 0;
    
    // Top neighbors: if tile_y > 0 then tile b exists.
    if (tile_y > 0) {
        int b_index = (tile_y - 1) * tiles_per_row + tile_x;
        b = input[b_index];
        if (tile_x > 0) {
            int a_index = (tile_y - 1) * tiles_per_row + (tile_x - 1);
            a = input[a_index];
        }
        if (tile_x < tiles_per_row - 1) {
            int c_index = (tile_y - 1) * tiles_per_row + (tile_x + 1);
            c = input[c_index];
        }
    }
    // Left neighbor.
    if (tile_x > 0) {
        int d_index = tile_y * tiles_per_row + (tile_x - 1);
        d = input[d_index];
    }
    // Center tile always exists.
    e = input[tile_index];
    // Right neighbor.
    if (tile_x < tiles_per_row - 1) {
        int f_index = tile_y * tiles_per_row + (tile_x + 1);
        f = input[f_index];
    }
    // Bottom neighbors.
    if (tile_y < tiles_per_row - 1) {
        int h_index = (tile_y + 1) * tiles_per_row + tile_x;
        h = input[h_index];
        if (tile_x > 0) {
            int g_index = (tile_y + 1) * tiles_per_row + (tile_x - 1);
            g = input[g_index];
        }
        if (tile_x < tiles_per_row - 1) {
            int i_index = (tile_y + 1) * tiles_per_row + (tile_x + 1);
            i_val = input[i_index];
        }
    }
    
    // We'll compute the new state for each 8×8 tile and pack the result into a std::uint64_t.
    // Each tile is processed row-by-row. For each row r (0 <= r < 8),
    // we extract the row from the central tile e, and also build its two neighbor rows (above and below)
    // in an "extended" 10-cell row. The extended row contains one extra cell on the left and one on the right.
    // The extra cells are taken from the adjacent tiles d and f (or a/c for the top border and g/i for the bottom border).
    // In our extended row representation, we pack the 10 cells into the lower 10 bits of a uint16_t.
    // The cells are arranged left-to-right such that the leftmost cell is stored in bit position 9,
    // and the rightmost cell in bit position 0.
    // The central 8 cells of the extended row (from index 1 to 8) come from the original row.
    
    std::uint64_t new_tile = 0;
    
    // Loop over each row (r) of the 8×8 central tile.
    for (int r = 0; r < 8; ++r) {
        // Extract the r-th row from the center tile e.
        // In our storage, row 0 is in the most significant byte and row 7 in the least.
        uint8_t cur_row = (uint8_t)(e >> ((7 - r) * 8));
        
        // Build extended row for the current row.
        // Left border: from tile d (if exists), use the rightmost cell of row r.
        int left_current = 0;
        if (tile_x > 0) {
            uint8_t d_row = (uint8_t)(d >> ((7 - r) * 8));
            // In our convention, the rightmost cell is in bit position 0.
            left_current = d_row & 1;
        }
        // Right border: from tile f (if exists), use the leftmost cell of row r.
        int right_current = 0;
        if (tile_x < tiles_per_row - 1) {
            uint8_t f_row = (uint8_t)(f >> ((7 - r) * 8));
            // The leftmost cell is in bit position 7.
            right_current = (f_row >> 7) & 1;
        }
        // Pack the extended current row in a 10-bit value.
        // Place the left border in bit position 9, cur_row in bits 8..1, and right border in bit position 0.
        uint16_t ext_current = ((uint16_t)left_current << 9) | ((uint16_t)cur_row << 1) | (uint16_t)right_current;
        
        // Build the extended row for the "above" neighbors.
        // For cell (r) in the central tile, the above row comes from row (r-1) of tile e,
        // except when r == 0 then use the bottom row of tile b.
        uint16_t ext_above = 0;
        if (r > 0) {
            uint8_t cur_above = (uint8_t)(e >> ((7 - (r - 1)) * 8));
            int left_above = 0;
            if (tile_x > 0) {
                uint8_t d_above = (uint8_t)(d >> ((7 - (r - 1)) * 8));
                left_above = d_above & 1;
            }
            int right_above = 0;
            if (tile_x < tiles_per_row - 1) {
                uint8_t f_above = (uint8_t)(f >> ((7 - (r - 1)) * 8));
                right_above = (f_above >> 7) & 1;
            }
            ext_above = ((uint16_t)left_above << 9) | ((uint16_t)cur_above << 1) | (uint16_t)right_above;
        }
        else {
            // r == 0: use the bottom row of tile b as the "above" row.
            uint8_t cur_above = 0;
            if (tile_y > 0) {
                // In tile b, row 7 (the least significant byte) is the bottom row.
                cur_above = (uint8_t)(b >> (0 * 8));
            }
            int left_above = 0;
            if (tile_x > 0 && tile_y > 0) {
                uint8_t a_row = (uint8_t)(a >> (0 * 8));
                left_above = a_row & 1;
            }
            int right_above = 0;
            if (tile_x < tiles_per_row - 1 && tile_y > 0) {
                uint8_t c_row = (uint8_t)(c >> (0 * 8));
                right_above = (c_row >> 7) & 1;
            }
            ext_above = ((uint16_t)left_above << 9) | ((uint16_t)cur_above << 1) | (uint16_t)right_above;
        }
        
        // Build the extended row for the "below" neighbors.
        // For cell (r) in the central tile, the below row comes from row (r+1) of tile e,
        // except when r == 7 then use the top row of tile h.
        uint16_t ext_below = 0;
        if (r < 7) {
            uint8_t cur_below = (uint8_t)(e >> ((7 - (r + 1)) * 8));
            int left_below = 0;
            if (tile_x > 0) {
                uint8_t d_below = (uint8_t)(d >> ((7 - (r + 1)) * 8));
                left_below = d_below & 1;
            }
            int right_below = 0;
            if (tile_x < tiles_per_row - 1) {
                uint8_t f_below = (uint8_t)(f >> ((7 - (r + 1)) * 8));
                right_below = (f_below >> 7) & 1;
            }
            ext_below = ((uint16_t)left_below << 9) | ((uint16_t)cur_below << 1) | (uint16_t)right_below;
        }
        else {
            // r == 7: use the top row of tile h as the "below" row.
            uint8_t cur_below = 0;
            if (tile_y < tiles_per_row - 1) {
                // In tile h, row 0 is the top row (stored in the most significant byte).
                cur_below = (uint8_t)(h >> (7 * 8));
            }
            int left_below = 0;
            if (tile_x > 0 && tile_y < tiles_per_row - 1) {
                uint8_t g_row = (uint8_t)(g >> (7 * 8));
                left_below = g_row & 1;
            }
            int right_below = 0;
            if (tile_x < tiles_per_row - 1 && tile_y < tiles_per_row - 1) {
                uint8_t i_row = (uint8_t)(i_val >> (7 * 8));
                right_below = (i_row >> 7) & 1;
            }
            ext_below = ((uint16_t)left_below << 9) | ((uint16_t)cur_below << 1) | (uint16_t)right_below;
        }
        
        // Process each of the 8 cells in the current row.
        // In the extended row, the central cells are stored in positions 1 to 8.
        // We represent an extended row in a 10-bit value where the leftmost cell is in bit position 9
        // and the rightmost in bit position 0. To access the cell at extended index k (0 <= k < 10),
        // we extract: (ext >> (9 - k)) & 1.
        uint8_t new_row = 0;
        for (int j = 0; j < 8; ++j)
        {
            // For cell at column j in the central tile, its corresponding extended index is j+1.
            // Its eight neighbors are:
            //   from the above row: indices j, j+1, j+2
            //   from the current row: indices j and j+2 (skip center)
            //   from the below row: indices j, j+1, j+2
            int sum = 0;
            sum += (ext_above >> (9 - j    )) & 1;
            sum += (ext_above >> (9 - (j+1))) & 1;
            sum += (ext_above >> (9 - (j+2))) & 1;
            sum += (ext_current >> (9 - j    )) & 1;
            sum += (ext_current >> (9 - (j+2))) & 1;
            sum += (ext_below >> (9 - j    )) & 1;
            sum += (ext_below >> (9 - (j+1))) & 1;
            sum += (ext_below >> (9 - (j+2))) & 1;
            
            // Get current cell state from the current extended row at index j+1.
            int cur_state = (ext_current >> (9 - (j+1))) & 1;
            
            // Apply Game of Life rules:
            // A cell becomes live if it has exactly 3 live neighbors,
            // or if it is already live and has exactly 2 live neighbors.
            int new_state = (sum == 3) || (cur_state && (sum == 2));
            
            // Place the new state in the new row.
            // In our storage for a tile row, bit 7 is the leftmost cell and bit 0 the rightmost.
            new_row |= (new_state & 1) << (7 - j);
        }
        
        // Pack new_row into the new_tile.
        // Row r of the tile goes into the byte at position (7 - r).
        new_tile |= ((std::uint64_t)new_row) << ((7 - r) * 8);
    }
    
    // Write the computed tile to the output.
    output[tile_index] = new_tile;
}

// This function executes one generation (one step) of Conway's Game of Life using the CUDA kernel.
// The input and output grids are bit‐packed and allocated on the device via cudaMalloc.
// grid_dimensions is the width (and height) of the square grid in cells (always a power of 2).
// Each std::uint64_t in the grid encodes an 8×8 tile.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Calculate how many 8×8 tiles span one row.
    int tiles_per_row = grid_dimensions >> 3; // grid_dimensions / 8

    // Choose block dimensions (tuned for performance on modern NVIDIA GPUs).
    // Here we launch one thread per tile.
    dim3 blockDim(16, 16);
    dim3 gridDim((tiles_per_row + blockDim.x - 1) / blockDim.x,
                 (tiles_per_row + blockDim.y - 1) / blockDim.y);

    // Launch the kernel (synchronization is handled by the caller).
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
