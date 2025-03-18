#include <cstdint>
#include <cuda_runtime.h>

// This kernel computes one iteration of Conway’s Game of Life on a bit‐packed grid.
// The grid is represented as an array of std::uint64_t, each encoding an 8x8 cell tile in row‐major order,
// where bit position (r*8 + c) corresponds to the cell at row r and column c (0 ≤ r,c < 8) in that tile.
// The overall grid dimensions (in cells) are given by grid_dimensions, and grid_dimensions is a multiple of 8.
// Each thread processes one tile. To update the tile correctly, it loads the 3x3 block of tiles that cover
// the current tile and its 8 immediate neighbors (using 0 for out‐of-bound tiles). Then it “unpacks” these
// tiles to assemble a 10x10 patch where the center 8x8 block corresponds to the current tile and the surrounding
// border contains the neighboring cells. Finally, it computes each cell’s new state by summing its 8 neighbors
// and applying the Game of Life rules, then re‐packs the 8x8 block into a std::uint64_t which is written to output.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int tile_count, int grid_dimensions)
{
    // Compute the tile coordinates within the grid of 8x8 tiles.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_x >= tile_count || tile_y >= tile_count)
        return;
        
    // Compute the linear index for the current tile.
    int idx = tile_y * tile_count + tile_x;
    
    // Load the 3x3 block of neighboring tiles into registers.
    // neighbors[1][1] is the current tile.
    // For out‐of-bound neighbor tiles, we substitute a 0 (i.e. all cells dead).
    std::uint64_t neighbors[3][3];
    for (int dy = -1; dy <= 1; dy++) {
        int ty = tile_y + dy;
        for (int dx = -1; dx <= 1; dx++) {
            int tx = tile_x + dx;
            if (tx >= 0 && tx < tile_count && ty >= 0 && ty < tile_count)
                neighbors[dy + 1][dx + 1] = input[ty * tile_count + tx];
            else
                neighbors[dy + 1][dx + 1] = 0;
        }
    }
    
    // Assemble a 10x10 patch of cells.
    // The center 8x8 (patch[1..8][1..8]) corresponds to the current tile.
    // The surrounding border is filled from the neighboring tiles.
    // The bit layout in each tile is assumed to be row-major,
    // where row r’s bits are stored in bits [r*8, r*8+7] with bit c representing column c.
    unsigned char patch[10][10];
    
    // Top row of the patch (row 0):
    // Top-left corner comes from the bottom-right cell of the top-left neighbor.
    patch[0][0] = (unsigned char)((neighbors[0][0] >> (7 * 8 + 7)) & 1ULL);
    // Top-center: fill columns 1..8 from the bottom row of the top neighbor (neighbors[0][1]).
    {
        std::uint64_t top_center = neighbors[0][1];
        // Extract the bottom row (row 7) as an 8-bit value.
        unsigned char row_val = (unsigned char)((top_center >> (7 * 8)) & 0xFFULL);
        for (int c = 0; c < 8; c++) {
            patch[0][c + 1] = (unsigned char)((row_val >> c) & 1);
        }
    }
    // Top-right corner from the bottom-left cell of the top-right neighbor.
    patch[0][9] = (unsigned char)((neighbors[0][2] >> (7 * 8 + 0)) & 1ULL);
    
    // Middle rows of the patch (rows 1 to 8):
    for (int r = 0; r < 8; r++) {
        // Left border cell: from the rightmost column of the left neighbor (neighbors[1][0]).
        patch[r + 1][0] = (unsigned char)((neighbors[1][0] >> (r * 8 + 7)) & 1ULL);
        
        // Center: current tile (neighbors[1][1]).
        {
            std::uint64_t center = neighbors[1][1];
            // Extract row r (an 8-bit value).
            unsigned char row_val = (unsigned char)((center >> (r * 8)) & 0xFFULL);
            for (int c = 0; c < 8; c++) {
                patch[r + 1][c + 1] = (unsigned char)((row_val >> c) & 1);
            }
        }
        
        // Right border cell: from the leftmost column of the right neighbor (neighbors[1][2]).
        patch[r + 1][9] = (unsigned char)((neighbors[1][2] >> (r * 8 + 0)) & 1ULL);
    }
    
    // Bottom row of the patch (row 9):
    // Bottom-left corner from the top-right cell of the bottom-left neighbor.
    patch[9][0] = (unsigned char)((neighbors[2][0] >> (0 * 8 + 7)) & 1ULL);
    // Bottom-center: fill columns 1..8 from the top row of the bottom neighbor (neighbors[2][1]).
    {
        std::uint64_t bottom_center = neighbors[2][1];
        // Extract the top row (row 0) as an 8-bit value.
        unsigned char row_val = (unsigned char)((bottom_center >> (0 * 8)) & 0xFFULL);
        for (int c = 0; c < 8; c++) {
            patch[9][c + 1] = (unsigned char)((row_val >> c) & 1);
        }
    }
    // Bottom-right corner from the top-left cell of the bottom-right neighbor.
    patch[9][9] = (unsigned char)((neighbors[2][2] >> (0 * 8 + 0)) & 1ULL);
    
    // Compute the next state for each cell in the current tile.
    // The new state for a cell is determined by the standard Game of Life rules:
    //   - A live cell remains alive if it has 2 or 3 live neighbors; otherwise, it dies.
    //   - A dead cell becomes alive if it has exactly 3 live neighbors; otherwise, it stays dead.
    std::uint64_t new_tile = 0;
    for (int r = 1; r <= 8; r++) {
        for (int c = 1; c <= 8; c++) {
            int sum = 0;
            // Sum the eight neighbors in the patch.
            sum += patch[r - 1][c - 1];
            sum += patch[r - 1][c];
            sum += patch[r - 1][c + 1];
            sum += patch[r][c - 1];
            sum += patch[r][c + 1];
            sum += patch[r + 1][c - 1];
            sum += patch[r + 1][c];
            sum += patch[r + 1][c + 1];
            
            int current = patch[r][c];
            int new_state = 0;
            if (current == 1) {
                if (sum == 2 || sum == 3)
                    new_state = 1;
            } else {
                if (sum == 3)
                    new_state = 1;
            }
            // Pack the new state into the corresponding bit.
            int bit_pos = (r - 1) * 8 + (c - 1);
            new_tile |= ((std::uint64_t)new_state << bit_pos);
        }
    }
    
    // Write the updated tile to the output grid.
    output[idx] = new_tile;
}

// Host function to run one iteration of Conway's Game of Life.
// The input and output grids are bit-packed arrays (allocated via cudaMalloc) where each
// std::uint64_t represents an 8x8 tile of cells. The grid dimensions (in cells) are provided
// by grid_dimensions, which is guaranteed to be a power of 2 and a multiple of 8.
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate the number of 8x8 tiles along one grid dimension.
    int tile_count = grid_dimensions / 8;
    
    // Set up the CUDA grid and block dimensions.
    // We choose a block size of 16x16 threads; each thread processes one 8x8 tile.
    dim3 blockDim(16, 16);
    dim3 gridDim((tile_count + blockDim.x - 1) / blockDim.x,
                 (tile_count + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel. Any host-device synchronization is managed by the caller.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, tile_count, grid_dimensions);
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
