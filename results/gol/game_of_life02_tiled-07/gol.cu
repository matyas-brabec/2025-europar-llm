#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel that computes one generation of Conway's Game of Life.
// The grid is bit‐packed: each std::uint64_t encodes an 8×8 tile of cells.
// Each bit in a tile is stored in row‐major order with bit index = (row*8 + col).
// The kernel processes one tile per thread. For each tile, it loads its own 8×8 cell block
// and its eight neighboring tiles (if available) to build an extended 10×10 cell array,
// which simplifies access to the 8-neighbor cells for every cell in the tile.
// Then it applies Conway's rules to compute the next state for each cell in the tile,
// repacks the results into a std::uint64_t, and writes it to the output grid.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each tile is 8×8 cells. Compute number of tiles per side.
    int tiles_per_side = grid_dimensions >> 3; // equivalent to grid_dimensions / 8

    // Compute tile coordinates for this thread.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the tile grid.
    if (tile_x >= tiles_per_side || tile_y >= tiles_per_side)
        return;

    // Compute the linear index for the current tile.
    int tile_index = tile_y * tiles_per_side + tile_x;

    // Load the central tile.
    std::uint64_t tile_center = input[tile_index];

    // Initialize neighbor tiles to 0 (interpreted as all dead cells) in case they are out-of-bound.
    std::uint64_t tile_top        = 0, tile_bottom       = 0;
    std::uint64_t tile_left       = 0, tile_right        = 0;
    std::uint64_t tile_top_left   = 0, tile_top_right    = 0;
    std::uint64_t tile_bottom_left = 0, tile_bottom_right = 0;

    // Load neighbor tiles if they exist.
    if (tile_y > 0)
        tile_top = input[(tile_y - 1) * tiles_per_side + tile_x];
    if (tile_y < tiles_per_side - 1)
        tile_bottom = input[(tile_y + 1) * tiles_per_side + tile_x];
    if (tile_x > 0)
        tile_left = input[tile_y * tiles_per_side + (tile_x - 1)];
    if (tile_x < tiles_per_side - 1)
        tile_right = input[tile_y * tiles_per_side + (tile_x + 1)];
    if (tile_y > 0 && tile_x > 0)
        tile_top_left = input[(tile_y - 1) * tiles_per_side + (tile_x - 1)];
    if (tile_y > 0 && tile_x < tiles_per_side - 1)
        tile_top_right = input[(tile_y - 1) * tiles_per_side + (tile_x + 1)];
    if (tile_y < tiles_per_side - 1 && tile_x > 0)
        tile_bottom_left = input[(tile_y + 1) * tiles_per_side + (tile_x - 1)];
    if (tile_y < tiles_per_side - 1 && tile_x < tiles_per_side - 1)
        tile_bottom_right = input[(tile_y + 1) * tiles_per_side + (tile_x + 1)];

    // Build an extended 10×10 array of cells (as unsigned char) to simplify neighbor access.
    // The center 8×8 block (indices [1..8][1..8]) corresponds to the current tile.
    // The border rows and columns are loaded from neighboring tiles (if available).
    unsigned char ext[10][10] = {0};

    // Fill the center region from tile_center.
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // Extract bit at position (i*8 + j) from tile_center.
            ext[i + 1][j + 1] = (tile_center >> (i * 8 + j)) & 1ULL;
        }
    }

    // Fill the top border from tile_top (use bottom row of top tile).
    if (tile_y > 0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            ext[0][j + 1] = (tile_top >> (7 * 8 + j)) & 1ULL;
        }
    }
    // Fill the bottom border from tile_bottom (use top row of bottom tile).
    if (tile_y < tiles_per_side - 1) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            ext[9][j + 1] = (tile_bottom >> (0 * 8 + j)) & 1ULL;
        }
    }
    // Fill the left border from tile_left (use rightmost column of left tile).
    if (tile_x > 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ext[i + 1][0] = (tile_left >> (i * 8 + 7)) & 1ULL;
        }
    }
    // Fill the right border from tile_right (use leftmost column of right tile).
    if (tile_x < tiles_per_side - 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ext[i + 1][9] = (tile_right >> (i * 8 + 0)) & 1ULL;
        }
    }
    // Fill the corner cells.
    if (tile_y > 0 && tile_x > 0)
        ext[0][0] = (tile_top_left >> (7 * 8 + 7)) & 1ULL;
    if (tile_y > 0 && tile_x < tiles_per_side - 1)
        ext[0][9] = (tile_top_right >> (7 * 8 + 0)) & 1ULL;
    if (tile_y < tiles_per_side - 1 && tile_x > 0)
        ext[9][0] = (tile_bottom_left >> (0 * 8 + 7)) & 1ULL;
    if (tile_y < tiles_per_side - 1 && tile_x < tiles_per_side - 1)
        ext[9][9] = (tile_bottom_right >> (0 * 8 + 0)) & 1ULL;

    // Process each cell in the 8×8 current tile.
    // For each cell (located at ext[i][j] for i,j in 1..8), count live neighbors and apply rules.
    std::uint64_t result = 0;
    #pragma unroll
    for (int i = 1; i <= 8; i++) {
        #pragma unroll
        for (int j = 1; j <= 8; j++) {
            // Sum the eight neighbors around cell ext[i][j].
            int live_neighbors =
                ext[i - 1][j - 1] + ext[i - 1][j] + ext[i - 1][j + 1] +
                ext[i][j - 1]                   + ext[i][j + 1] +
                ext[i + 1][j - 1] + ext[i + 1][j] + ext[i + 1][j + 1];

            int current = ext[i][j];
            int new_state = 0;

            // Apply Game of Life rules:
            //   - Live cell survives with 2 or 3 neighbors.
            //   - Dead cell becomes live with exactly 3 neighbors.
            if (current == 1)
                new_state = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
            else
                new_state = (live_neighbors == 3) ? 1 : 0;

            // Set the corresponding bit in the result word.
            int bit_index = (i - 1) * 8 + (j - 1);
            result |= (static_cast<std::uint64_t>(new_state) << bit_index);
        }
    }

    // Write the computed next-generation tile to the output grid.
    output[tile_index] = result;
}

// Host function that launches the kernel to simulate one generation of the Game of Life.
// 'input' and 'output' are pointers to the bit-packed grid allocated on the GPU.
// 'grid_dimensions' is the width/height (number of cells) of the square grid (always a power of 2).
// Each std::uint64_t in the grid encodes an 8×8 tile of cells.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of tiles per side (each tile is 8×8 cells).
    int tiles_per_side = grid_dimensions >> 3; // grid_dimensions / 8

    // Define a block configuration. Using 16×16 threads per block to cover many tiles.
    dim3 block(16, 16);
    // Calculate grid dimensions to cover the entire tile grid.
    dim3 grid((tiles_per_side + block.x - 1) / block.x, (tiles_per_side + block.y - 1) / block.y);

    // Launch the kernel to execute one generation of Conway's Game of Life.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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
