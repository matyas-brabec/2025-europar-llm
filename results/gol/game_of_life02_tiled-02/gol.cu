#include <cuda_runtime.h>
#include <cstdint>

// This device function extracts one row (8 bits) from a packed 8×8 tile stored in a 64‐bit integer.
// We assume that bit0 represents column 0 and bit7 represents column 7 of that row.
// The 64-bit tile is laid out with row 0 in the least-significant 8 bits, row 1 in the next 8 bits, etc.
__device__ inline uint8_t get_tile_row(std::uint64_t tile, int row)
{
    return (tile >> (row * 8)) & 0xFF;
}

// __global__ kernel that computes one step of Conway's Game of Life.
// The input grid is stored bit-packed: each std::uint64_t holds an 8×8 tile of cells.
// The overall grid is grid_dimensions × grid_dimensions cells, and therefore
// contains (grid_dimensions/8) × (grid_dimensions/8) tiles.
// Each thread processes one tile. To compute the next state of its 8×8 cells we must
// also inspect the eight neighboring tiles (or assume dead cells if at the boundary).
//
// We first build a 10×10 "expanded" tile in registers: the central 8×8 region corresponds
// to the current tile, and an extra border is obtained from neighbor tiles (if they exist),
// or zero if out-of-bound (all cells outside are dead). In the expanded region, each row is
// stored in an integer with its 10 least-significant bits representing the cells. Bit0 is column 0.
// The central 8 columns are stored in bit positions 1..8, which simplifies accessing neighbors.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of tiles per row (and per column)
    int tiles_per_row = grid_dimensions / 8;

    // Compute the tile coordinates for this thread
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Out-of-bound threads do nothing.
    if (tile_x >= tiles_per_row || tile_y >= tiles_per_row)
        return;

    // Compute the linear index of the current tile.
    int tile_index = tile_y * tiles_per_row + tile_x;

    // Load the center tile from the input grid.
    std::uint64_t center_tile = input[tile_index];

    // Load neighbor tiles. Use 0 for tiles outside the grid (all dead cells).
    std::uint64_t tile_left        = (tile_x > 0)                  ? input[tile_y * tiles_per_row + (tile_x - 1)] : 0;
    std::uint64_t tile_right       = (tile_x < tiles_per_row - 1)  ? input[tile_y * tiles_per_row + (tile_x + 1)] : 0;
    std::uint64_t tile_above       = (tile_y > 0)                  ? input[(tile_y - 1) * tiles_per_row + tile_x] : 0;
    std::uint64_t tile_below       = (tile_y < tiles_per_row - 1)  ? input[(tile_y + 1) * tiles_per_row + tile_x] : 0;
    std::uint64_t tile_above_left  = (tile_y > 0 && tile_x > 0)                  ? input[(tile_y - 1) * tiles_per_row + (tile_x - 1)] : 0;
    std::uint64_t tile_above_right = (tile_y > 0 && tile_x < tiles_per_row - 1)  ? input[(tile_y - 1) * tiles_per_row + (tile_x + 1)] : 0;
    std::uint64_t tile_below_left  = (tile_y < tiles_per_row - 1 && tile_x > 0)  ? input[(tile_y + 1) * tiles_per_row + (tile_x - 1)] : 0;
    std::uint64_t tile_below_right = (tile_y < tiles_per_row - 1 && tile_x < tiles_per_row - 1)  ? input[(tile_y + 1) * tiles_per_row + (tile_x + 1)] : 0;

    // We now build a 10×10 expanded grid in an array of 10 integers.
    // Each integer's 10 least-significant bits represent one row of the expanded region.
    // In the expanded region, the central tile's cells will be stored in bit positions 1..8.
    int exp[10];

    // Row 0 of the expanded region is the top border of the central tile.
    // We obtain the corresponding row from the row 7 (i.e., bottom row) of the tile above.
    // For the left and right border bits, we inspect the neighbor tiles above-left and above-right.
    if (tile_y > 0) {
        uint8_t above_center = get_tile_row(tile_above, 7);
        uint8_t above_left   = (tile_x > 0) ? get_tile_row(tile_above_left, 7) : 0;
        uint8_t above_right  = (tile_x < tiles_per_row - 1) ? get_tile_row(tile_above_right, 7) : 0;
        int left_bit   = (tile_x > 0)                 ? ((above_left >> 7) & 1) : 0; // take bit at column 7
        int center_bits= above_center;                        // full 8 bits for columns 0..7
        int right_bit  = (tile_x < tiles_per_row - 1) ? (above_right & 1) : 0;   // take bit at column 0
        // Place left_bit into bit0, center_bits into bits 1..8, and right_bit into bit9.
        exp[0] = (left_bit) | (center_bits << 1) | (right_bit << 9);
    }
    else {
        exp[0] = 0;
    }

    // Rows 1 .. 8 of the expanded region correspond to rows 0 .. 7 of the central tile.
    // For each such row, we also incorporate the left and right neighbor bits from the same row.
    for (int r = 0; r < 8; r++) {
        uint8_t center_row = get_tile_row(center_tile, r);
        int left_bit  = (tile_x > 0) ? ((get_tile_row(tile_left, r) >> 7) & 1) : 0;
        int right_bit = (tile_x < tiles_per_row - 1) ? (get_tile_row(tile_right, r) & 1) : 0;
        exp[r + 1] = left_bit | (center_row << 1) | (right_bit << 9);
    }

    // Row 9 of the expanded region is the bottom border.
    // We obtain it from row 0 of the tile below.
    if (tile_y < tiles_per_row - 1) {
        uint8_t below_center = get_tile_row(tile_below, 0);
        uint8_t below_left   = (tile_x > 0) ? get_tile_row(tile_below_left, 0) : 0;
        uint8_t below_right  = (tile_x < tiles_per_row - 1) ? get_tile_row(tile_below_right, 0) : 0;
        int left_bit   = (tile_x > 0) ? ((below_left >> 7) & 1) : 0;
        int center_bits= below_center;
        int right_bit  = (tile_x < tiles_per_row - 1) ? (below_right & 1) : 0;
        exp[9] = left_bit | (center_bits << 1) | (right_bit << 9);
    }
    else {
        exp[9] = 0;
    }

    // Now compute the next state for each cell in the central 8×8 region.
    // We loop over the central region in the expanded grid (rows 1..8 and columns 1..8).
    // For each cell, we count the 8 neighboring bits in the expanded array.
    // Then we apply the Game of Life rules:
    //   - A live cell (current==1) survives if it has 2 or 3 neighbors.
    //   - A dead cell (current==0) becomes live if it has exactly 3 neighbors.
    uint8_t new_rows[8] = {0};

    for (int r = 1; r <= 8; r++) {
        uint8_t new_row = 0;
        for (int c = 1; c <= 8; c++) {
            int count = 0;
            // Top row neighbors.
            count += (exp[r - 1] >> (c - 1)) & 1;
            count += (exp[r - 1] >> (c    )) & 1;
            count += (exp[r - 1] >> (c + 1)) & 1;
            // Same row, excluding the center cell.
            count += (exp[r] >> (c - 1)) & 1;
            count += (exp[r] >> (c + 1)) & 1;
            // Bottom row neighbors.
            count += (exp[r + 1] >> (c - 1)) & 1;
            count += (exp[r + 1] >> (c    )) & 1;
            count += (exp[r + 1] >> (c + 1)) & 1;
            
            int current = (exp[r] >> c) & 1;
            int new_state = (count == 3 || (current && count == 2)) ? 1 : 0;
            // Set new_state in new_row at bit position (c-1) (bit0 corresponds to column0).
            new_row |= (new_state << (c - 1));
        }
        new_rows[r - 1] = new_row;
    }

    // Pack the 8 new_rows (each 8 bits) back into a 64-bit integer.
    std::uint64_t result_tile = 0;
    for (int r = 0; r < 8; r++) {
        result_tile |= (static_cast<std::uint64_t>(new_rows[r]) << (r * 8));
    }

    // Write the updated tile to the output grid.
    output[tile_index] = result_tile;
}

// Host function that launches one step of Conway's Game of Life using the CUDA kernel.
// The grids (input and output) are bit-packed arrays of std::uint64_t; each element encodes an 8×8 tile.
// grid_dimensions gives the width and height of the square grid (in cells), which is always a power of 2.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of tiles along one dimension.
    int tiles_per_dim = grid_dimensions / 8;

    // Configure kernel launch parameters. We choose a 16×16 thread block.
    dim3 block(16, 16);
    dim3 grid((tiles_per_dim + block.x - 1) / block.x,
              (tiles_per_dim + block.y - 1) / block.y);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // Note: Any necessary host-device synchronization is handled by the caller.
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
