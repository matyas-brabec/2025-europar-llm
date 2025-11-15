#include <cstdint>
#include <cuda_runtime.h>

/*
 * Bit layout and tiling:
 *
 * - The full grid is grid_dimensions x grid_dimensions cells.
 * - grid_dimensions is a power of two, >= 512, and divisible by 8.
 * - Cells are grouped into 8x8 tiles. Each tile is stored in one std::uint64_t.
 * - Tiles are laid out in row-major order:
 *       tile_index = tile_y * tiles_per_dim + tile_x,
 *   where tiles_per_dim = grid_dimensions / 8.
 *
 * - Within a tile (8x8 cells), bits are laid out row-major:
 *       bit_index = local_row * 8 + local_col,
 *   where local_row, local_col in [0, 7].
 *   bit 0 is the top-left cell of the tile, bit 7 is top-right, bits 8..15 are the second row, etc.
 *
 * Boundary handling:
 * - Any cell outside the global grid is considered dead (0).
 * - Tiles outside the grid are treated as 0 (all cells dead).
 */

/**
 * @brief Extracts an 8-bit row from a tile word and extends it to 10 bits
 *        by prepending the rightmost bit of the left tile row and appending
 *        the leftmost bit of the right tile row.
 *
 * @param center_word Tile word containing the 8-bit center row.
 * @param left_word   Tile word containing the 8-bit left neighbor row (or 0 if none).
 * @param right_word  Tile word containing the 8-bit right neighbor row (or 0 if none).
 * @param row_in_tile Row index within the tile [0,7].
 *
 * @return 10-bit row in a uint16_t:
 *         bit 0   = left tile's column 7,
 *         bits 1-8 = center tile's columns 0-7,
 *         bit 9   = right tile's column 0.
 */
static __device__ __forceinline__
std::uint16_t make_extended_row(std::uint64_t center_word,
                                std::uint64_t left_word,
                                std::uint64_t right_word,
                                int row_in_tile)
{
    // Extract 8-bit rows from each tile word.
    std::uint8_t center_row = static_cast<std::uint8_t>((center_word >> (row_in_tile * 8)) & 0xFFu);
    std::uint8_t left_row   = static_cast<std::uint8_t>((left_word   >> (row_in_tile * 8)) & 0xFFu);
    std::uint8_t right_row  = static_cast<std::uint8_t>((right_word  >> (row_in_tile * 8)) & 0xFFu);

    // Bits 1..8 = center_row bits 0..7.
    std::uint16_t extended = static_cast<std::uint16_t>(center_row) << 1;
    // Bit 0 = bit 7 of the left row (neighbor to the left of column 0).
    extended |= static_cast<std::uint16_t>((left_row >> 7) & 1u);
    // Bit 9 = bit 0 of the right row (neighbor to the right of column 7).
    extended |= static_cast<std::uint16_t>(right_row & 1u) << 9;

    return extended;
}

/**
 * @brief CUDA kernel implementing one step of Conway's Game of Life on a bit-packed grid.
 *
 * Each thread processes one 8x8 tile (one std::uint64_t).
 * Neighbor information is gathered from up to 9 surrounding tiles (3x3 tile neighborhood).
 * Within each tile, the kernel iterates over the 8x8 cells, computing neighbor counts and
 * the next state using bit operations.
 */
static __global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                           std::uint64_t* __restrict__ output,
                                           int grid_dim)
{
    constexpr int TILE_SIZE = 8;
    const int tiles_per_dim = grid_dim >> 3;  // grid_dim / 8

    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_dim || tile_y >= tiles_per_dim)
        return;

    const std::size_t idx_center = static_cast<std::size_t>(tile_y) * tiles_per_dim + tile_x;

    // Load the 3x3 neighborhood of tiles around (tile_y, tile_x).
    // Missing neighbors are represented as 0 (all cells dead).
    const std::uint64_t tile_c  = input[idx_center];
    const std::uint64_t tile_l  = (tile_x > 0) ? input[idx_center - 1] : 0ull;
    const std::uint64_t tile_r  = (tile_x + 1 < tiles_per_dim) ? input[idx_center + 1] : 0ull;

    const std::uint64_t tile_u  = (tile_y > 0) ? input[idx_center - tiles_per_dim] : 0ull;
    const std::uint64_t tile_d  = (tile_y + 1 < tiles_per_dim) ? input[idx_center + tiles_per_dim] : 0ull;

    const std::uint64_t tile_ul = (tile_x > 0 && tile_y > 0)
                                  ? input[idx_center - tiles_per_dim - 1] : 0ull;
    const std::uint64_t tile_ur = (tile_x + 1 < tiles_per_dim && tile_y > 0)
                                  ? input[idx_center - tiles_per_dim + 1] : 0ull;

    const std::uint64_t tile_dl = (tile_x > 0 && tile_y + 1 < tiles_per_dim)
                                  ? input[idx_center + tiles_per_dim - 1] : 0ull;
    const std::uint64_t tile_dr = (tile_x + 1 < tiles_per_dim && tile_y + 1 < tiles_per_dim)
                                  ? input[idx_center + tiles_per_dim + 1] : 0ull;

    std::uint64_t new_tile = 0ull;

    const int global_y_base = tile_y * TILE_SIZE;
    const int grid_dim_minus1 = grid_dim - 1;

    // Process each of the 8 rows in this tile.
#pragma unroll
    for (int local_row = 0; local_row < TILE_SIZE; ++local_row)
    {
        const int gy = global_y_base + local_row; // global Y coordinate of this row

        // Build the extended row for the current row (mid).
        const std::uint16_t row_mid = make_extended_row(tile_c, tile_l, tile_r, local_row);

        // Build extended row for the row above (row_up). If gy == 0, there is no above row.
        std::uint16_t row_up = 0;
        if (gy > 0)
        {
            if (local_row > 0)
            {
                // Above row is within the same tile.
                row_up = make_extended_row(tile_c, tile_l, tile_r, local_row - 1);
            }
            else
            {
                // local_row == 0: above row is the last row (row 7) of the tiles above.
                row_up = make_extended_row(tile_u, tile_ul, tile_ur, TILE_SIZE - 1);
            }
        }

        // Build extended row for the row below (row_down). If gy == grid_dim-1, no below row.
        std::uint16_t row_down = 0;
        if (gy < grid_dim_minus1)
        {
            if (local_row < TILE_SIZE - 1)
            {
                // Below row is within the same tile.
                row_down = make_extended_row(tile_c, tile_l, tile_r, local_row + 1);
            }
            else
            {
                // local_row == 7: below row is the first row (row 0) of the tiles below.
                row_down = make_extended_row(tile_d, tile_dl, tile_dr, 0);
            }
        }

        // Current row's 8 cells from the center tile.
        const std::uint8_t center_row_bits =
            static_cast<std::uint8_t>((tile_c >> (local_row * 8)) & 0xFFu);

        std::uint8_t new_row_bits = 0u;

        // Compute the next state for each column in this row.
#pragma unroll
        for (int col = 0; col < TILE_SIZE; ++col)
        {
            /*
             * For the cell at (gy, gx) where gx = tile_x * 8 + col,
             * the neighbors are:
             * - Above row: up-left, up, up-right      -> row_up bits [col..col+2]
             * - Same row: left, right                 -> row_mid bits [col] and [col+2]
             * - Below row: down-left, down, down-right-> row_down bits [col..col+2]
             *
             * row_mid bit layout for this row:
             *   bit (col)   = left neighbor
             *   bit (col+1) = self (center cell)
             *   bit (col+2) = right neighbor
             *
             * We pack these 8 neighbor bits into a byte and count them using __popc.
             */

            std::uint8_t neighborhood = 0u;

            // Upper 3 neighbors -> bits 0..2.
            neighborhood |= static_cast<std::uint8_t>((row_up >> col) & 0x7u);

            // Same-row left neighbor -> bit 3.
            neighborhood |= static_cast<std::uint8_t>(((row_mid >> col) & 0x1u) << 3);

            // Same-row right neighbor -> bit 4.
            neighborhood |= static_cast<std::uint8_t>(((row_mid >> (col + 2)) & 0x1u) << 4);

            // Lower 3 neighbors -> bits 5..7.
            neighborhood |= static_cast<std::uint8_t>(((row_down >> col) & 0x7u) << 5);

            const unsigned neighbor_count = __popc(static_cast<unsigned>(neighborhood));

            const unsigned cell_alive = (center_row_bits >> col) & 0x1u;

            // Conway's Game of Life rules:
            // - A live cell survives if it has 2 or 3 neighbors.
            // - A dead cell becomes alive if it has exactly 3 neighbors.
            const unsigned survive = (cell_alive & (neighbor_count == 2u));
            const unsigned born    = (neighbor_count == 3u);
            const unsigned next_alive = survive | born;

            new_row_bits |= static_cast<std::uint8_t>(next_alive << col);
        }

        // Store the updated 8-bit row back into the 64-bit tile word.
        new_tile |= static_cast<std::uint64_t>(new_row_bits) << (local_row * 8);
    }

    output[idx_center] = new_tile;
}

/**
 * @brief Host wrapper that launches the CUDA kernel for one step of Conway's Game of Life.
 *
 * @param input           Device pointer to the input grid (bit-packed, 8x8 tiles per std::uint64_t).
 * @param output          Device pointer to the output grid (same layout as input).
 * @param grid_dimensions Width and height of the square grid (power of two, multiple of 8).
 *
 * The function does not synchronize; the caller is responsible for any needed synchronization.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    constexpr int TILE_SIZE = 8;
    const int tiles_per_dim = grid_dimensions >> 3; // grid_dimensions / 8

    // Configure a 2D grid of thread blocks. Each thread processes one tile (one std::uint64_t).
    dim3 block_dim(16, 16);
    dim3 grid_dim((tiles_per_dim + block_dim.x - 1) / block_dim.x,
                  (tiles_per_dim + block_dim.y - 1) / block_dim.y);

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, grid_dimensions);
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
