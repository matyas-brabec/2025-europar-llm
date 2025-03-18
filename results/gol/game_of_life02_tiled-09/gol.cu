#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one generation update for Conway's Game of Life.
// The grid is stored in bit‐packed 8×8 “tiles” (each tile is a std::uint64_t with 64 cells).
// Each thread processes one tile. To compute the next state of cells in the center
// tile, the thread loads the 8 neighboring tiles (if available) to form an extended 10×10
// bit grid (a 1-cell border around the 8×8 tile). Then, for every cell in the center
// (rows 1..8 and columns 1..8 of the extended grid) it counts the live neighbors and applies
// the Game of Life rules. The result is re-packed into a std::uint64_t and written to output.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int tiles_per_dim)
{
    // Compute tile indices in tile-space.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_x >= tiles_per_dim || tile_y >= tiles_per_dim)
        return;

    int tile_idx = tile_y * tiles_per_dim + tile_x;

    // Load the current (center) tile and its 8 neighbors.
    // Each tile stores an 8×8 block of cells in row‐major order.
    std::uint64_t center = input[tile_idx];
    std::uint64_t top      = (tile_y > 0) ? input[(tile_y - 1) * tiles_per_dim + tile_x] : 0;
    std::uint64_t bottom   = (tile_y < tiles_per_dim - 1) ? input[(tile_y + 1) * tiles_per_dim + tile_x] : 0;
    std::uint64_t left     = (tile_x > 0) ? input[tile_y * tiles_per_dim + (tile_x - 1)] : 0;
    std::uint64_t right    = (tile_x < tiles_per_dim - 1) ? input[tile_y * tiles_per_dim + (tile_x + 1)] : 0;
    std::uint64_t top_left     = (tile_x > 0 && tile_y > 0) ? input[(tile_y - 1) * tiles_per_dim + (tile_x - 1)] : 0;
    std::uint64_t top_right    = (tile_x < tiles_per_dim - 1 && tile_y > 0) ? input[(tile_y - 1) * tiles_per_dim + (tile_x + 1)] : 0;
    std::uint64_t bottom_left  = (tile_x > 0 && tile_y < tiles_per_dim - 1) ? input[(tile_y + 1) * tiles_per_dim + (tile_x - 1)] : 0;
    std::uint64_t bottom_right = (tile_x < tiles_per_dim - 1 && tile_y < tiles_per_dim - 1) ? input[(tile_y + 1) * tiles_per_dim + (tile_x + 1)] : 0;

    // Build the "extended" grid: a 10-row array (each row contains 10 bits in its lower bits).
    // The extended grid has a 1-cell border around the 8×8 center tile.
    // Convention: extended row indices 1..8 correspond to the 8 rows of the center tile,
    // and extended column indices 1..8 correspond to the 8 columns.
    int ext[10] = {0};

    // Fill the top extended row (row 0) from the top neighbor tiles.
    // For row 0, we need:
    // - column 0: the bottom-right cell of top_left tile (if available),
    // - columns 1..8: row 7 (the bottom row) of the top tile,
    // - column 9: the bottom-left cell of top_right tile (if available).
    int top_center_row = (tile_y > 0) ? int((top >> (7 * 8)) & 0xffULL) : 0;
    int top_left_bit  = (tile_x > 0 && tile_y > 0) ? int((top_left >> (7 * 8 + 7)) & 1ULL) : 0;
    int top_right_bit = (tile_x < tiles_per_dim - 1 && tile_y > 0) ? int((top_right >> (7 * 8 + 0)) & 1ULL) : 0;
    ext[0] = (top_left_bit) | (top_center_row << 1) | (top_right_bit << 9);

    // Fill the extended rows 1 to 8 from the center tile and its left/right neighbors.
    // For each center row i (0 <= i < 8):
    // - extended row index = i+1;
    // - column 0: rightmost cell of left tile's row i (if available),
    // - columns 1..8: row i of the center tile,
    // - column 9: leftmost cell of right tile's row i (if available).
    for (int i = 0; i < 8; i++) {
        int center_row = int((center >> (i * 8)) & 0xffULL);
        int left_bit = (tile_x > 0) ? int((left >> (i * 8 + 7)) & 1ULL) : 0;
        int right_bit = (tile_x < tiles_per_dim - 1) ? int((right >> (i * 8 + 0)) & 1ULL) : 0;
        ext[i + 1] = (left_bit) | (center_row << 1) | (right_bit << 9);
    }

    // Fill the bottom extended row (row 9) from the bottom neighbor tiles.
    // For row 9, we need:
    // - column 0: the top-right cell of bottom_left tile (if available),
    // - columns 1..8: row 0 (the top row) of the bottom tile,
    // - column 9: the top-left cell of bottom_right tile (if available).
    int bottom_center_row = (tile_y < tiles_per_dim - 1) ? int((bottom >> (0 * 8)) & 0xffULL) : 0;
    int bottom_left_bit = (tile_x > 0 && tile_y < tiles_per_dim - 1) ? int((bottom_left >> (0 * 8 + 7)) & 1ULL) : 0;
    int bottom_right_bit = (tile_x < tiles_per_dim - 1 && tile_y < tiles_per_dim - 1) ? int((bottom_right >> (0 * 8 + 0)) & 1ULL) : 0;
    ext[9] = (bottom_left_bit) | (bottom_center_row << 1) | (bottom_right_bit << 9);

    // Compute the next-generation state for the center 8×8 tile.
    // Each cell's new state is determined by summing the 8 neighbors in the extended grid.
    std::uint64_t new_tile = 0;
    // Loop over the 8 rows (extended indices 1 to 8).
    for (int i = 1; i <= 8; i++) {
        int new_row = 0; // 8-bit value for the updated row.
        // Loop over the 8 columns (extended indices 1 to 8).
        for (int j = 1; j <= 8; j++) {
            // Sum the eight neighboring bits.
            int count = 0;
            count += (ext[i - 1] >> (j - 1)) & 1;
            count += (ext[i - 1] >> (j    )) & 1;
            count += (ext[i - 1] >> (j + 1)) & 1;
            count += (ext[i    ] >> (j - 1)) & 1;
            count += (ext[i    ] >> (j + 1)) & 1;
            count += (ext[i + 1] >> (j - 1)) & 1;
            count += (ext[i + 1] >> (j    )) & 1;
            count += (ext[i + 1] >> (j + 1)) & 1;

            int current = (ext[i] >> j) & 1;
            int next = 0;
            // Apply Game of Life rules:
            //  - Alive cell survives if it has 2 or 3 live neighbors.
            //  - Dead cell becomes alive if it has exactly 3 live neighbors.
            next = (current && (count == 2 || count == 3)) || ((!current) && (count == 3));
            new_row |= (next << (j - 1));
        }
        new_tile |= (std::uint64_t(new_row) << ((i - 1) * 8));
    }
    // Write the updated tile state back to the output grid.
    output[tile_idx] = new_tile;
}

// Host function that launches one step of Conway's Game of Life on the GPU.
// The grid is provided in a bit-packed format where each std::uint64_t encodes an 8×8 tile.
// Parameters:
//   input          - pointer to device memory containing the current grid.
//   output         - pointer to device memory for storing the next-generation grid.
//   grid_dimensions- the width (and height) of the square grid in cells (always a power of 2).
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 8×8 tiles per dimension.
    int tiles_per_dim = grid_dimensions / 8;

    // Define the block and grid dimensions for the kernel launch.
    // A 16×16 thread block is chosen; this value can be tuned to the target hardware.
    dim3 blockDim(16, 16);
    dim3 gridDim((tiles_per_dim + blockDim.x - 1) / blockDim.x,
                 (tiles_per_dim + blockDim.y - 1) / blockDim.y);

    // Launch the kernel. The caller is responsible for any host-device synchronization.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, tiles_per_dim);
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
