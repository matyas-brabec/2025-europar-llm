#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Data layout:
 * - The grid is square with side length `grid_dimensions` (power of 2, >= 512).
 * - Each std::uint64_t encodes an 8x8 tile of cells.
 * - Tiles are laid out in row-major order:
 *       tile_index = tile_y * tiles_per_dim + tile_x
 *   where tiles_per_dim = grid_dimensions / 8.
 * - Within each 8x8 tile, bits are stored row-major:
 *       bit_index = local_y * 8 + local_x
 *   with (local_x, local_y) in [0,7] x [0,7], and bit 0 (LSB) corresponding
 *   to the top-left cell of the tile.
 *
 * Boundary conditions:
 * - Cells outside the global grid are considered dead (0).
 * - This is implemented by treating tiles outside the [0, tiles_per_dim) range
 *   as all zeros when reading neighbors.
 *
 * Kernel:
 * - One thread processes one 8x8 tile (one std::uint64_t).
 * - For each tile, the kernel reads the tile and its 8 neighboring tiles
 *   (3x3 block of tiles). Missing neighbors at boundaries are treated as zero.
 * - The 3x3 tiles are composed into a 10x10 logical neighborhood window
 *   around the central 8x8 tile using 10 16-bit row masks.
 * - For each of the 64 cells in the central 8x8 tile, neighbor counts are
 *   computed using bit masks and __popc, and the Game of Life rules are applied.
 * - The resulting 8x8 tile is written back as one std::uint64_t.
 */

static constexpr int TILE_SIZE = 8;  // 8x8 cells per tile

// Extract an 8-bit row (local_y in [0,7]) from a tile.
// Bits are returned in positions 0..7, with bit 0 being local_x = 0.
__device__ __forceinline__ std::uint8_t get_tile_row(std::uint64_t tile, int local_y) {
    return static_cast<std::uint8_t>((tile >> (local_y * TILE_SIZE)) & 0xFFu);
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_per_dim) {
    // Compute tile coordinates handled by this thread
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_dim || tile_y >= tiles_per_dim) {
        return;
    }

    const int idx = tile_y * tiles_per_dim + tile_x;

    // Determine existence of neighbor tiles
    const bool has_left  = (tile_x > 0);
    const bool has_right = (tile_x + 1 < tiles_per_dim);
    const bool has_up    = (tile_y > 0);
    const bool has_down  = (tile_y + 1 < tiles_per_dim);

    // Load central tile
    const std::uint64_t C  = input[idx];

    // Load neighbor tiles if they exist, otherwise treat as 0 (dead)
    const std::uint64_t N  = has_up    ? input[idx - tiles_per_dim] : 0ull;
    const std::uint64_t S  = has_down  ? input[idx + tiles_per_dim] : 0ull;
    const std::uint64_t W  = has_left  ? input[idx - 1]             : 0ull;
    const std::uint64_t E  = has_right ? input[idx + 1]             : 0ull;

    const std::uint64_t NW = (has_up    && has_left)  ? input[idx - tiles_per_dim - 1] : 0ull;
    const std::uint64_t NE = (has_up    && has_right) ? input[idx - tiles_per_dim + 1] : 0ull;
    const std::uint64_t SW = (has_down  && has_left)  ? input[idx + tiles_per_dim - 1] : 0ull;
    const std::uint64_t SE = (has_down  && has_right) ? input[idx + tiles_per_dim + 1] : 0ull;

    // Build a 10x10 neighborhood window around the central tile.
    // Represented as 10 rows of 10 bits each; each row is stored in a uint16_t.
    //
    // Row index mapping:
    //   rows[0]   : one row above central tile (from N / NW / NE)
    //   rows[1..8]: rows of central tile (from W / C / E)
    //   rows[9]   : one row below central tile (from S / SW / SE)
    //
    // Column mapping within each row (bit positions 0..9):
    //   bit 0     : one column to the left of central tile (from W / NW / SW)
    //   bits 1..8 : columns of central tile (local_x 0..7)
    //   bit 9     : one column to the right of central tile (from E / NE / SE)
    std::uint16_t rows[10];

    // Top neighbor row (rows[0]) from NW, N, NE (row index 7 of those tiles)
    const std::uint8_t rowNW7 = (has_up && has_left)  ? get_tile_row(NW, 7) : 0u;
    const std::uint8_t rowN7  =  has_up               ? get_tile_row(N,  7) : 0u;
    const std::uint8_t rowNE7 = (has_up && has_right) ? get_tile_row(NE, 7) : 0u;
    rows[0] = static_cast<std::uint16_t>(
                  ((static_cast<std::uint16_t>((rowNW7 >> 7) & 0x1u))      ) | // bit 0
                  ((static_cast<std::uint16_t>( rowN7        )     ) << 1) | // bits 1..8
                  ((static_cast<std::uint16_t>( rowNE7 & 0x1u) ) << 9));     // bit 9

    // Middle rows (rows[1..8]) from W, C, E
#pragma unroll
    for (int r = 0; r < TILE_SIZE; ++r) { // r: local_y in central tile
        const std::uint8_t rowW = has_left  ? get_tile_row(W, r) : 0u;
        const std::uint8_t rowC =              get_tile_row(C, r);
        const std::uint8_t rowE = has_right ? get_tile_row(E, r) : 0u;

        rows[r + 1] = static_cast<std::uint16_t>(
                          ((static_cast<std::uint16_t>((rowW >> 7) & 0x1u))      ) | // bit 0
                          ((static_cast<std::uint16_t>( rowC        )     ) << 1) | // bits 1..8
                          ((static_cast<std::uint16_t>( rowE & 0x1u) ) << 9));     // bit 9
    }

    // Bottom neighbor row (rows[9]) from SW, S, SE (row index 0 of those tiles)
    const std::uint8_t rowSW0 = (has_down && has_left)  ? get_tile_row(SW, 0) : 0u;
    const std::uint8_t rowS0  =  has_down               ? get_tile_row(S,  0) : 0u;
    const std::uint8_t rowSE0 = (has_down && has_right) ? get_tile_row(SE, 0) : 0u;
    rows[9] = static_cast<std::uint16_t>(
                  ((static_cast<std::uint16_t>((rowSW0 >> 7) & 0x1u))      ) | // bit 0
                  ((static_cast<std::uint16_t>( rowS0        )     ) << 1) | // bits 1..8
                  ((static_cast<std::uint16_t>( rowSE0 & 0x1u) ) << 9));     // bit 9

    // Compute next state for the central 8x8 tile
    std::uint64_t next_tile = 0ull;

#pragma unroll
    for (int cy = 0; cy < TILE_SIZE; ++cy) { // cy: local_y in [0,7]
        const std::uint16_t row_above = rows[cy];     // global row gy = cy + 0
        const std::uint16_t row_curr  = rows[cy + 1]; // global row gy = cy + 1
        const std::uint16_t row_below = rows[cy + 2]; // global row gy = cy + 2

#pragma unroll
        for (int cx = 0; cx < TILE_SIZE; ++cx) { // cx: local_x in [0,7]
            const int gx = cx + 1; // column index in the 10-bit row (1..8)

            // Extract neighbor bits in 3x3 neighborhood around (gx, gy) using masks:
            // From row_above and row_below: 3 bits (left, center, right) each
            // From row_curr: left and right bits only (exclude center)
            const unsigned int mask_shift = static_cast<unsigned int>(gx - 1);

            const unsigned int above_bits =
                static_cast<unsigned int>((row_above >> mask_shift) & 0x7u);
            const unsigned int below_bits =
                static_cast<unsigned int>((row_below >> mask_shift) & 0x7u);
            const unsigned int curr_lr_bits =
                static_cast<unsigned int>((row_curr  >> mask_shift) & 0x5u); // 0b101: left+right

            const int neighbors =
                __popc(above_bits) + __popc(below_bits) + __popc(curr_lr_bits);

            const bool alive =
                ((row_curr >> gx) & 0x1u) != 0; // center cell state

            // Game of Life rules:
            // - Alive stays alive with 2 or 3 neighbors
            // - Dead becomes alive with exactly 3 neighbors
            const bool next_alive =
                (neighbors == 3) || (alive && neighbors == 2);

            if (next_alive) {
                const int bit_index = cy * TILE_SIZE + cx;
                next_tile |= (0x1ull << bit_index);
            }
        }
    }

    // Store updated tile
    output[idx] = next_tile;
}

// Host wrapper: execute one step of Conway's Game of Life.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // grid_dimensions is guaranteed to be a power of 2 and >= 512,
    // hence divisible by TILE_SIZE (8).
    const int tiles_per_dim = grid_dimensions >> 3; // grid_dimensions / 8

    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (tiles_per_dim + block_dim.x - 1) / block_dim.x,
        (tiles_per_dim + block_dim.y - 1) / block_dim.y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_dim);
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
