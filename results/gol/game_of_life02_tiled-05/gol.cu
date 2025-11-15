#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Representation:
 *  - The global grid is N x N cells (N is a power of 2, > 512).
 *  - Cells are grouped into 8x8 tiles.
 *  - Each tile is stored as a single std::uint64_t.
 *  - Within a tile, bit (row * 8 + col) corresponds to the cell at:
 *      global_row = tile_y * 8 + row
 *      global_col = tile_x * 8 + col
 *    where row, col are in [0,7]. Bit 0 is (row=0, col=0).
 *
 *  - input:  device pointer to the current generation (bit-packed).
 *  - output: device pointer to the next generation (bit-packed).
 *
 * Kernel strategy:
 *  - One thread updates one 8x8 tile.
 *  - For each tile, the thread loads the 3x3 neighborhood of tiles (up to 9 tiles)
 *    into registers (neighbors[0..8]). Tiles outside the global grid are treated
 *    as all-dead (0).
 *  - For each of the 64 cells in the central tile, the thread:
 *      * Determines the 8 neighbor positions (row/col, potentially wrapping to
 *        the 3x3 tile neighborhood).
 *      * Extracts neighbor bits from neighbors[] with bit shifts.
 *      * Counts live neighbors and applies Conway's rules.
 *      * Writes the resulting bit into a local 64-bit accumulator.
 *  - Finally, the accumulator is written to the corresponding output tile.
 *
 * Notes on performance:
 *  - This kernel uses only registers and global memory (no shared/texture memory).
 *  - The number of global memory loads is minimized to 9 uint64_t values per tile.
 *  - Control flow inside the inner loops is uniform across threads in a warp, so
 *    there is no warp divergence in the per-cell neighbor processing.
 */

constexpr int TILE_DIM = 8;  // Tile is 8x8 cells

// Kernel that computes one Game of Life step on the bit-packed grid.
__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int tiles_per_dim)
{
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_dim || tile_y >= tiles_per_dim) {
        return;
    }

    const int tile_index = tile_y * tiles_per_dim + tile_x;

    // Load 3x3 neighboring tiles into registers.
    //
    // Layout in neighbors[]:
    //   [0] [1] [2]   => (tile_y-1, tile_x-1 .. tile_x+1)
    //   [3] [4] [5]   => (tile_y  , tile_x-1 .. tile_x+1)
    //   [6] [7] [8]   => (tile_y+1, tile_x-1 .. tile_x+1)
    //
    // neighbors[4] is the current tile.
    std::uint64_t neighbors[9];

    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = tile_y + dy;
        const bool row_in_bounds = (ny >= 0) && (ny < tiles_per_dim);

        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = tile_x + dx;
            const int idx9 = (dy + 1) * 3 + (dx + 1);

            if (row_in_bounds && nx >= 0 && nx < tiles_per_dim) {
                neighbors[idx9] = input[ny * tiles_per_dim + nx];
            } else {
                // Outside the global grid => treat as all-dead cells.
                neighbors[idx9] = 0ull;
            }
        }
    }

    const std::uint64_t center_tile = neighbors[4];
    std::uint64_t result_tile = 0ull;

    // Process each of the 8x8 cells in the center tile.
    // Outer loop over tile rows.
#pragma unroll
    for (int local_row = 0; local_row < TILE_DIM; ++local_row) {
        // For this row, precompute row indices and mapping to neighbor tiles.

        const int r_mid = local_row;
        const int row_mid_idx = 1;  // index 1 corresponds to the center row in neighbors[]
        int r_top = local_row - 1;
        int row_top_idx = 1;
        if (r_top < 0) {
            r_top += TILE_DIM;  // wrap to row 7 of the tile above
            row_top_idx = 0;    // row 0 in neighbors[] (above)
        }
        int r_bottom = local_row + 1;
        int row_bottom_idx = 1;
        if (r_bottom >= TILE_DIM) {
            r_bottom -= TILE_DIM;  // wrap to row 0 of the tile below
            row_bottom_idx = 2;    // row 2 in neighbors[] (below)
        }

        const int row_top_base    = r_top    * TILE_DIM;
        const int row_mid_base    = r_mid    * TILE_DIM;
        const int row_bottom_base = r_bottom * TILE_DIM;

        // Inner loop over tile columns.
#pragma unroll
        for (int local_col = 0; local_col < TILE_DIM; ++local_col) {
            const int c_mid = local_col;
            const int col_mid_idx = 1;  // index 1 corresponds to center column in neighbors[]

            int c_left = local_col - 1;
            int col_left_idx = 1;
            if (c_left < 0) {
                c_left += TILE_DIM;  // wrap to col 7 of the tile to the left
                col_left_idx = 0;    // column 0 in neighbors[] (left)
            }

            int c_right = local_col + 1;
            int col_right_idx = 1;
            if (c_right >= TILE_DIM) {
                c_right -= TILE_DIM;  // wrap to col 0 of the tile to the right
                col_right_idx = 2;    // column 2 in neighbors[] (right)
            }

            unsigned int neighbor_count = 0;
            int bit_index;

            // Top-left neighbor (r_top, c_left)
            bit_index = row_top_base + c_left;
            neighbor_count += (neighbors[row_top_idx * 3 + col_left_idx] >> bit_index) & 1ull;

            // Top neighbor (r_top, c_mid)
            bit_index = row_top_base + c_mid;
            neighbor_count += (neighbors[row_top_idx * 3 + col_mid_idx] >> bit_index) & 1ull;

            // Top-right neighbor (r_top, c_right)
            bit_index = row_top_base + c_right;
            neighbor_count += (neighbors[row_top_idx * 3 + col_right_idx] >> bit_index) & 1ull;

            // Left neighbor (r_mid, c_left)
            bit_index = row_mid_base + c_left;
            neighbor_count += (neighbors[row_mid_idx * 3 + col_left_idx] >> bit_index) & 1ull;

            // Right neighbor (r_mid, c_right)
            bit_index = row_mid_base + c_right;
            neighbor_count += (neighbors[row_mid_idx * 3 + col_right_idx] >> bit_index) & 1ull;

            // Bottom-left neighbor (r_bottom, c_left)
            bit_index = row_bottom_base + c_left;
            neighbor_count += (neighbors[row_bottom_idx * 3 + col_left_idx] >> bit_index) & 1ull;

            // Bottom neighbor (r_bottom, c_mid)
            bit_index = row_bottom_base + c_mid;
            neighbor_count += (neighbors[row_bottom_idx * 3 + col_mid_idx] >> bit_index) & 1ull;

            // Bottom-right neighbor (r_bottom, c_right)
            bit_index = row_bottom_base + c_right;
            neighbor_count += (neighbors[row_bottom_idx * 3 + col_right_idx] >> bit_index) & 1ull;

            // Current cell state from center_tile
            const int center_bit_index = row_mid_base + c_mid;
            const unsigned int current_alive = (center_tile >> center_bit_index) & 1ull;

            // Conway's Game of Life rules:
            // - A live cell with 2 or 3 live neighbors survives.
            // - A dead cell with exactly 3 live neighbors becomes alive.
            // - Otherwise, the cell dies or remains dead.
            //
            // new_alive = (neighbor_count == 3) || (current_alive && neighbor_count == 2)
            const unsigned int new_alive =
                (neighbor_count == 3u) | (current_alive & (neighbor_count == 2u));

            result_tile |= (std::uint64_t(new_alive) << center_bit_index);
        }
    }

    output[tile_index] = result_tile;
}

/*
 * Host function to launch one Game of Life step on the GPU.
 *
 * Parameters:
 *  - input:  device pointer to current generation grid (bit-packed, uint64 per 8x8 tile).
 *  - output: device pointer to next generation grid (same layout).
 *  - grid_dimensions: width/height N of the square grid (N x N), N is a power of 2 > 512.
 *
 * Notes:
 *  - Input and output must be allocated with cudaMalloc by the caller.
 *  - No host-device synchronization is performed here; the caller is responsible
 *    for any required synchronization (e.g., cudaDeviceSynchronize()).
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 8x8 tiles along one dimension.
    const int tiles_per_dim = grid_dimensions / TILE_DIM;

    // Choose a reasonable block size. 16x16 threads => 256 threads per block.
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (tiles_per_dim + block_dim.x - 1) / block_dim.x,
        (tiles_per_dim + block_dim.y - 1) / block_dim.y
    );

    game_of_life_step_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_dim);
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
