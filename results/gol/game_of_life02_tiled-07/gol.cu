// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// 
// Layout assumptions:
// - The overall grid is square with `grid_dimensions` cells on each side.
//   `grid_dimensions` is a power of two, >= 512, and divisible by 8.
// - The grid is stored as 8×8 tiles, each encoded in one std::uint64_t.
// - Tiles are laid out in row-major order:
//     tile_index = tile_y * tiles_per_row + tile_x
//     where tiles_per_row = grid_dimensions / 8.
// - Within each 8×8 tile, cells are stored in row-major order, least-significant
//   bit first:
//     bit index = local_y * 8 + local_x
//     local_x, local_y in [0, 7].
//   So bit 0 is (x=0, y=0) of the tile, bit 63 is (x=7, y=7).
//
// Boundary rules:
// - Cells outside the global grid are treated as dead (no wrap-around).
//
// Parallelization strategy:
// - Each CUDA thread processes exactly one 8×8 tile (one std::uint64_t).
// - The thread reads the tile and its 8 neighboring tiles (if they exist),
//   all as 64-bit words from global memory.
// - For each of the 64 cells in the tile, the thread counts live neighbors
//   by reading bits from the appropriate rows of these 64-bit words and
//   applies the Game of Life rules to produce the next state.
// - No shared or texture memory is used; all data is kept in registers.

#include <cstdint>
#include <cuda_runtime.h>

namespace
{

// Kernel that computes one Game of Life step on a grid encoded as 8×8 tiles.
// `tiles_per_row` is the number of tiles along one side of the square grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_per_row)
{
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int tiles_per_col = tiles_per_row; // square grid

    if (tile_x >= tiles_per_row || tile_y >= tiles_per_col)
        return;

    const int idx = tile_y * tiles_per_row + tile_x;

    // Load center tile.
    const std::uint64_t center = input[idx];

    // Determine which neighbor tiles exist.
    const bool has_n = (tile_y > 0);
    const bool has_s = (tile_y + 1 < tiles_per_col);
    const bool has_w = (tile_x > 0);
    const bool has_e = (tile_x + 1 < tiles_per_row);

    // Load neighbor tiles where present; otherwise they remain 0 (dead).
    std::uint64_t north      = 0;
    std::uint64_t south      = 0;
    std::uint64_t west       = 0;
    std::uint64_t east       = 0;
    std::uint64_t north_west = 0;
    std::uint64_t north_east = 0;
    std::uint64_t south_west = 0;
    std::uint64_t south_east = 0;

    if (has_n)
    {
        north = input[idx - tiles_per_row];
        if (has_w) north_west = input[idx - tiles_per_row - 1];
        if (has_e) north_east = input[idx - tiles_per_row + 1];
    }
    if (has_s)
    {
        south = input[idx + tiles_per_row];
        if (has_w) south_west = input[idx + tiles_per_row - 1];
        if (has_e) south_east = input[idx + tiles_per_row + 1];
    }
    if (has_w)
    {
        west = input[idx - 1];
    }
    if (has_e)
    {
        east = input[idx + 1];
    }

    std::uint64_t out_word = 0;

    // Process the 8×8 cells in this tile row by row.
    // For each row we materialize the relevant 8-bit row slices from the
    // center and neighbor tiles into registers and then process 8 columns.
    #pragma unroll
    for (int y = 0; y < 8; ++y)
    {
        // Current row in the center tile.
        const std::uint8_t mid =
            static_cast<std::uint8_t>((center >> (y * 8)) & 0xFFu);

        // Row above current row (from center, north, or zero at top boundary).
        std::uint8_t up;
        if (y > 0)
        {
            up = static_cast<std::uint8_t>((center >> ((y - 1) * 8)) & 0xFFu);
        }
        else if (has_n)
        {
            up = static_cast<std::uint8_t>((north >> (7 * 8)) & 0xFFu);
        }
        else
        {
            up = 0;
        }

        // Row below current row (from center, south, or zero at bottom boundary).
        std::uint8_t down;
        if (y < 7)
        {
            down = static_cast<std::uint8_t>((center >> ((y + 1) * 8)) & 0xFFu);
        }
        else if (has_s)
        {
            down = static_cast<std::uint8_t>((south >> 0) & 0xFFu);
        }
        else
        {
            down = 0;
        }

        // Rows from left and right neighbor tiles, only needed for x == 0 or x == 7.
        std::uint8_t upL   = 0;
        std::uint8_t midL  = 0;
        std::uint8_t downL = 0;

        std::uint8_t upR   = 0;
        std::uint8_t midR  = 0;
        std::uint8_t downR = 0;

        if (has_w)
        {
            midL = static_cast<std::uint8_t>((west >> (y * 8)) & 0xFFu);

            if (y > 0)
            {
                upL = static_cast<std::uint8_t>((west >> ((y - 1) * 8)) & 0xFFu);
            }
            else if (has_n)
            {
                upL = static_cast<std::uint8_t>((north_west >> (7 * 8)) & 0xFFu);
            }

            if (y < 7)
            {
                downL = static_cast<std::uint8_t>((west >> ((y + 1) * 8)) & 0xFFu);
            }
            else if (has_s)
            {
                downL = static_cast<std::uint8_t>((south_west >> 0) & 0xFFu);
            }
        }

        if (has_e)
        {
            midR = static_cast<std::uint8_t>((east >> (y * 8)) & 0xFFu);

            if (y > 0)
            {
                upR = static_cast<std::uint8_t>((east >> ((y - 1) * 8)) & 0xFFu);
            }
            else if (has_n)
            {
                upR = static_cast<std::uint8_t>((north_east >> (7 * 8)) & 0xFFu);
            }

            if (y < 7)
            {
                downR = static_cast<std::uint8_t>((east >> ((y + 1) * 8)) & 0xFFu);
            }
            else if (has_s)
            {
                downR = static_cast<std::uint8_t>((south_east >> 0) & 0xFFu);
            }
        }

        // Process all 8 columns in this row.
        #pragma unroll
        for (int x = 0; x < 8; ++x)
        {
            const unsigned int mask_x    = 1u << x;
            const unsigned int left_mask = (x > 0) ? (mask_x >> 1) : 0u;
            const unsigned int right_mask= (x < 7) ? (mask_x << 1) : 0u;

            unsigned int neighbors = 0;

            // Neighbors in the row above.
            neighbors += (up   & left_mask)  ? 1u : 0u;
            neighbors += (up   & mask_x)     ? 1u : 0u;
            neighbors += (up   & right_mask) ? 1u : 0u;

            // Neighbors in the same row (excluding center cell).
            neighbors += (mid  & left_mask)  ? 1u : 0u;
            neighbors += (mid  & right_mask) ? 1u : 0u;

            // Neighbors in the row below.
            neighbors += (down & left_mask)  ? 1u : 0u;
            neighbors += (down & mask_x)     ? 1u : 0u;
            neighbors += (down & right_mask) ? 1u : 0u;

            // For x == 0, additional neighbors from the left tile
            // (up-left, left, down-left).
            if (x == 0 && has_w)
            {
                neighbors += (upL   & (1u << 7)) ? 1u : 0u;
                neighbors += (midL  & (1u << 7)) ? 1u : 0u;
                neighbors += (downL & (1u << 7)) ? 1u : 0u;
            }

            // For x == 7, additional neighbors from the right tile
            // (up-right, right, down-right).
            if (x == 7 && has_e)
            {
                neighbors += (upR   & 1u) ? 1u : 0u;
                neighbors += (midR  & 1u) ? 1u : 0u;
                neighbors += (downR & 1u) ? 1u : 0u;
            }

            const unsigned int alive = (mid & mask_x) ? 1u : 0u;

            // Game of Life rules:
            // - Alive cell survives with 2 or 3 neighbors.
            // - Dead cell becomes alive with exactly 3 neighbors.
            const unsigned int new_alive =
                (neighbors == 3u) | (alive & (neighbors == 2u));

            if (new_alive)
            {
                const int bit_index = y * 8 + x;
                out_word |= (std::uint64_t(1) << bit_index);
            }
        }
    }

    output[idx] = out_word;
}

} // anonymous namespace

// Host function that launches the Game of Life kernel for one simulation step.
// `input` and `output` are device pointers allocated with cudaMalloc.
// `grid_dimensions` is the width/height of the square grid in cells.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each tile is 8×8 cells.
    const int tiles_per_row = grid_dimensions >> 3; // divide by 8

    // 2D launch configuration: one thread per tile.
    const dim3 block_dim(16, 16, 1);
    const dim3 grid_dim(
        (tiles_per_row + block_dim.x - 1) / block_dim.x,
        (tiles_per_row + block_dim.y - 1) / block_dim.y,
        1);

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_row);
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
