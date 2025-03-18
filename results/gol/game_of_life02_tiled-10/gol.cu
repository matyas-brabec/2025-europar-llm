// Conway’s Game of Life – CUDA Implementation
// This implementation assumes that the grid is bit‐packed into std::uint64_t values,
// where each 64-bit element represents an 8×8 tile of cells (row‐major order).
// Each bit corresponds to a cell (bit==1 means alive, 0 means dead).
// The grid dimensions (width and height in cells) are always a power of 2, at least 512,
// so the grid comprises (grid_dimensions/8)×(grid_dimensions/8) tiles.
// Each CUDA thread handles one 8×8 tile. In order to correctly compute the life rules,
// the kernel builds a local 10×10 array (with a one‐cell border around the 8×8 tile)
// by gathering neighbor bits from the eight adjoining tiles (if they exist; if not, they are 0).
//
// The Game of Life rules are applied on each cell:
//    - Any live cell with 2 or 3 live neighbors survives; otherwise, it dies.
//    - Any dead cell with exactly 3 live neighbors becomes alive; otherwise, it remains dead.
//
// The function run_game_of_life launches this kernel. Host/device synchronization
// (if needed) is assumed to be handled externally.

#include <cstdint>
#include <cuda_runtime.h>

// __global__ kernel function: Computes one Game of Life step for one 8×8 tile.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions) {
    // Compute the number of tiles along one dimension.
    int tile_count = grid_dimensions >> 3; // equivalent to grid_dimensions / 8

    // Each thread processes one tile; determine its tile coordinates.
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tile_count || tile_y >= tile_count)
        return;

    // Compute the linear tile index.
    int tile_idx = tile_y * tile_count + tile_x;
    
    // Load the current (central) tile from global memory.
    std::uint64_t center = input[tile_idx];

    // Create a local 10x10 array to hold the 8x8 tile and its one-cell border.
    // The interior [1..8][1..8] will hold the central tile's cells.
    // The borders are filled with cells from neighboring tiles (if available),
    // otherwise they remain zero (dead).
    unsigned char cells[10][10];

    // Initialize the padded array to 0.
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        #pragma unroll
        for (int j = 0; j < 10; j++) {
            cells[i][j] = 0;
        }
    }

    // Unpack the central tile's 8x8 cells.
    // The bit at position (r*8 + c) in 'center' corresponds to the cell at row r, col c.
    // We place these in cells[1..8][1..8].
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int bit_index = r * 8 + c;
            cells[r + 1][c + 1] = (center >> bit_index) & 1ULL;
        }
    }

    // Helper lambda to extract a bit from a tile given row and column indices.
    auto get_tile_bit = [](__uint64_t tile, int r, int c) -> unsigned char {
        return (tile >> (r * 8 + c)) & 1ULL;
    };

    // Load neighboring tiles if they exist. For cells on the border of the current tile,
    // the neighbor cells come from adjacent tiles. If a neighbor tile is missing, the border remains 0.
    // Note: The tile grid is laid out in a row-major fashion.

    // North neighbor (tile above, same column)
    if (tile_y > 0) {
        int north_idx = (tile_y - 1) * tile_count + tile_x;
        std::uint64_t tileN = input[north_idx];
        // Use the bottom row (row 7) of the north tile.
        for (int c = 0; c < 8; c++) {
            cells[0][c + 1] = get_tile_bit(tileN, 7, c);
        }
    }

    // South neighbor (tile below, same column)
    if (tile_y < tile_count - 1) {
        int south_idx = (tile_y + 1) * tile_count + tile_x;
        std::uint64_t tileS = input[south_idx];
        // Use the top row (row 0) of the south tile.
        for (int c = 0; c < 8; c++) {
            cells[9][c + 1] = get_tile_bit(tileS, 0, c);
        }
    }

    // West neighbor (tile left, same row)
    if (tile_x > 0) {
        int west_idx = tile_y * tile_count + (tile_x - 1);
        std::uint64_t tileW = input[west_idx];
        // Use the rightmost column (col 7) of the west tile.
        for (int r = 0; r < 8; r++) {
            cells[r + 1][0] = get_tile_bit(tileW, r, 7);
        }
    }

    // East neighbor (tile right, same row)
    if (tile_x < tile_count - 1) {
        int east_idx = tile_y * tile_count + (tile_x + 1);
        std::uint64_t tileE = input[east_idx];
        // Use the leftmost column (col 0) of the east tile.
        for (int r = 0; r < 8; r++) {
            cells[r + 1][9] = get_tile_bit(tileE, r, 0);
        }
    }

    // Northwest neighbor (tile above-left)
    if (tile_x > 0 && tile_y > 0) {
        int nw_idx = (tile_y - 1) * tile_count + (tile_x - 1);
        std::uint64_t tileNW = input[nw_idx];
        // Use bottom-right cell (row 7, col 7) of the NW tile.
        cells[0][0] = get_tile_bit(tileNW, 7, 7);
    }

    // Northeast neighbor (tile above-right)
    if (tile_x < tile_count - 1 && tile_y > 0) {
        int ne_idx = (tile_y - 1) * tile_count + (tile_x + 1);
        std::uint64_t tileNE = input[ne_idx];
        // Use bottom-left cell (row 7, col 0) of the NE tile.
        cells[0][9] = get_tile_bit(tileNE, 7, 0);
    }

    // Southwest neighbor (tile below-left)
    if (tile_x > 0 && tile_y < tile_count - 1) {
        int sw_idx = (tile_y + 1) * tile_count + (tile_x - 1);
        std::uint64_t tileSW = input[sw_idx];
        // Use top-right cell (row 0, col 7) of the SW tile.
        cells[9][0] = get_tile_bit(tileSW, 0, 7);
    }

    // Southeast neighbor (tile below-right)
    if (tile_x < tile_count - 1 && tile_y < tile_count - 1) {
        int se_idx = (tile_y + 1) * tile_count + (tile_x + 1);
        std::uint64_t tileSE = input[se_idx];
        // Use top-left cell (row 0, col 0) of the SE tile.
        cells[9][9] = get_tile_bit(tileSE, 0, 0);
    }

    // Now compute the next state for each cell in the central 8x8 region.
    // The new state is computed based on the sum of the eight neighbors.
    std::uint64_t new_tile = 0ULL;
    for (int i = 1; i <= 8; i++) {
        for (int j = 1; j <= 8; j++) {
            // Count live neighbors.
            int live_neighbors =
                cells[i - 1][j - 1] + cells[i - 1][j] + cells[i - 1][j + 1] +
                cells[i][j - 1]                   + cells[i][j + 1] +
                cells[i + 1][j - 1] + cells[i + 1][j] + cells[i + 1][j + 1];

            // Apply Conway's rules.
            if (cells[i][j]) { // Cell is currently alive.
                if (live_neighbors == 2 || live_neighbors == 3)
                    new_tile |= (1ULL << ((i - 1) * 8 + (j - 1)));
            } else { // Cell is currently dead.
                if (live_neighbors == 3)
                    new_tile |= (1ULL << ((i - 1) * 8 + (j - 1)));
            }
        }
    }

    // Write the updated tile back to global memory.
    output[tile_idx] = new_tile;
}

// Host function to launch the Game of Life kernel for one simulation step.
// 'input' and 'output' are device pointers to the bit-packed grids.
// 'grid_dimensions' is the width/height of the grid in cells.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Compute the number of tiles along each dimension.
    int tile_count = grid_dimensions / 8;

    // Choose a block size that provides good occupancy on modern NVIDIA GPUs.
    dim3 block(16, 16);
    // The grid dimensions (in tiles) are computed by rounding up.
    dim3 grid((tile_count + block.x - 1) / block.x,
              (tile_count + block.y - 1) / block.y);

    // Launch the kernel. Any necessary host-device synchronization is assumed to be handled externally.
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
