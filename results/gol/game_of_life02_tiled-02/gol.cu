#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Layout assumptions:
 * - The grid is square with side length `grid_dimensions`, which is a power of 2 > 512.
 * - Each std::uint64_t encodes an 8×8 tile of cells in row-major order:
 *      bit index = y * 8 + x, where 0 <= x,y < 8
 *      bit 0 corresponds to (x=0,y=0), bit 1 to (1,0), ..., bit 63 to (7,7).
 * - Tiles themselves are laid out row-major:
 *      tile_x = 0..(grid_dimensions/8 - 1)
 *      tile_y = 0..(grid_dimensions/8 - 1)
 *      linear index = tile_y * tiles_per_row + tile_x
 *
 * - All cells outside the global grid are considered dead (0).
 *
 * This kernel assigns one 8×8 tile to each thread. Each thread:
 *   1. Loads the 3×3 neighborhood of tiles (9 uint64_t values) around its tile
 *      into registers; tiles outside the grid are treated as 0.
 *   2. Computes the next state for all 64 cells in its tile, writing a single
 *      uint64_t to the output.
 *
 * The per-cell neighbor counting is split into:
 *   - Interior cells (x,y in 1..6): all neighbors are within the center tile
 *     and can be handled using only the center word.
 *   - Edge cells (cells on the tile border): neighbors may cross tile
 *     boundaries and are handled via a small helper that indexes into the
 *     3×3 tile neighborhood.
 */

/**
 * @brief Fetch the state of a cell given local coordinates relative to the center tile.
 *
 * @param tiles  Array of 9 uint64_t values forming a 3×3 neighborhood of tiles around
 *               the current tile. Layout:
 *                 tiles[0] = top-left      (-1,-1)
 *                 tiles[1] = top           ( 0,-1)
 *                 tiles[2] = top-right     (+1,-1)
 *                 tiles[3] = left          (-1, 0)
 *                 tiles[4] = center        ( 0, 0)
 *                 tiles[5] = right         (+1, 0)
 *                 tiles[6] = bottom-left   (-1,+1)
 *                 tiles[7] = bottom        ( 0,+1)
 *                 tiles[8] = bottom-right  (+1,+1)
 *
 * @param local_x  X coordinate in range [-1,8] relative to the center tile,
 *                 where [0,7] lie within the center tile, -1 is one cell to
 *                 the left, and 8 is one cell to the right.
 * @param local_y  Y coordinate in range [-1,8] relative to the center tile,
 *                 with the same convention as local_x.
 *
 * The function maps (local_x, local_y) into one of the 3×3 tiles and a bit
 * index within that tile, then returns 0 or 1 depending on whether the cell
 * is dead or alive.
 *
 * Tiles outside the global grid have already been set to 0 by the caller, so
 * this function does not need to perform any explicit boundary checks.
 */
__device__ __forceinline__ int getBitFromTiles(const std::uint64_t tiles[9],
                                               int local_x,
                                               int local_y)
{
    // Determine which tile in the 3×3 neighborhood this coordinate falls into.
    // local_x ∈ [-1,8], local_y ∈ [-1,8].
    // Using arithmetic right shift on signed ints:
    //   -1 >> 3 == -1, 0..7 >> 3 == 0, 8 >> 3 == 1.
    int tile_dx = local_x >> 3;  // -1, 0, or 1
    int tile_dy = local_y >> 3;  // -1, 0, or 1

    // Normalize local coordinates into [0,7] within that tile.
    // For 2's complement, (-1 & 7) == 7, (0..7 & 7) == 0..7, (8 & 7) == 0.
    int x = local_x & 7;
    int y = local_y & 7;

    // Compute flat tile index in [0,8].
    int tileIdx = (tile_dy + 1) * 3 + (tile_dx + 1);

    std::uint64_t t = tiles[tileIdx];

    int bitIdx = (y << 3) + x;
    return static_cast<int>((t >> bitIdx) & 1ULL);
}

/**
 * @brief CUDA kernel that advances one step of Conway's Game of Life.
 *
 * @param input            Device pointer to the input grid (bit-packed).
 * @param output           Device pointer to the output grid (bit-packed).
 * @param grid_dimensions  Width/height of the square grid in cells.
 */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    const int tilesPerRow = grid_dimensions >> 3;  // grid_dimensions / 8

    // Tile coordinates in tile space.
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tilesPerRow || tile_y >= tilesPerRow)
        return;

    const int idx = tile_y * tilesPerRow + tile_x;

    // Load center tile.
    const std::uint64_t center = input[idx];

    // Load 3×3 neighborhood of tiles into registers.
    // Layout of tiles[] documented in getBitFromTiles().
    std::uint64_t tiles[9];

    tiles[4] = center;  // center tile

    // Top row of tiles (y-1).
    if (tile_y > 0)
    {
        const int idx_top = (tile_y - 1) * tilesPerRow + tile_x;

        tiles[1] = input[idx_top];  // top (0,-1)

        if (tile_x > 0)
            tiles[0] = input[idx_top - 1];  // top-left (-1,-1)
        else
            tiles[0] = 0;

        if (tile_x + 1 < tilesPerRow)
            tiles[2] = input[idx_top + 1];  // top-right (+1,-1)
        else
            tiles[2] = 0;
    }
    else
    {
        // Outside grid: treated as dead.
        tiles[0] = tiles[1] = tiles[2] = 0;
    }

    // Middle row: left and right tiles.
    if (tile_x > 0)
        tiles[3] = input[idx - 1];  // left (-1,0)
    else
        tiles[3] = 0;

    if (tile_x + 1 < tilesPerRow)
        tiles[5] = input[idx + 1];  // right (+1,0)
    else
        tiles[5] = 0;

    // Bottom row of tiles (y+1).
    if (tile_y + 1 < tilesPerRow)
    {
        const int idx_bottom = (tile_y + 1) * tilesPerRow + tile_x;

        tiles[7] = input[idx_bottom];  // bottom (0,+1)

        if (tile_x > 0)
            tiles[6] = input[idx_bottom - 1];  // bottom-left (-1,+1)
        else
            tiles[6] = 0;

        if (tile_x + 1 < tilesPerRow)
            tiles[8] = input[idx_bottom + 1];  // bottom-right (+1,+1)
        else
            tiles[8] = 0;
    }
    else
    {
        tiles[6] = tiles[7] = tiles[8] = 0;
    }

    // Compute next state for this tile.
    std::uint64_t nextTile = 0;

    // First handle interior cells: 1 <= x,y <= 6.
    // All neighbors for these cells lie within the center tile, so we can
    // use simple bit-index arithmetic on `center` only.
#pragma unroll
    for (int ly = 1; ly <= 6; ++ly)
    {
#pragma unroll
        for (int lx = 1; lx <= 6; ++lx)
        {
            const int bitIdx = (ly << 3) + lx;

            int neighborCount = 0;
            neighborCount += (int)((center >> (bitIdx - 9)) & 1ULL);  // NW
            neighborCount += (int)((center >> (bitIdx - 8)) & 1ULL);  // N
            neighborCount += (int)((center >> (bitIdx - 7)) & 1ULL);  // NE
            neighborCount += (int)((center >> (bitIdx - 1)) & 1ULL);  // W
            neighborCount += (int)((center >> (bitIdx + 1)) & 1ULL);  // E
            neighborCount += (int)((center >> (bitIdx + 7)) & 1ULL);  // SW
            neighborCount += (int)((center >> (bitIdx + 8)) & 1ULL);  // S
            neighborCount += (int)((center >> (bitIdx + 9)) & 1ULL);  // SE

            const int isAlive = (int)((center >> bitIdx) & 1ULL);

            // Game of Life rules:
            // - A live cell survives if it has 2 or 3 live neighbors.
            // - A dead cell becomes alive if it has exactly 3 live neighbors.
            const int newState = (neighborCount == 3) | (isAlive & (neighborCount == 2));

            nextTile |= (std::uint64_t)newState << bitIdx;
        }
    }

    // Handle edge cells (border of the 8×8 tile). These may have neighbors
    // in adjacent tiles, so we use getBitFromTiles() to read from the full
    // 3×3 tile neighborhood.
#pragma unroll
    for (int ly = 0; ly < 8; ++ly)
    {
#pragma unroll
        for (int lx = 0; lx < 8; ++lx)
        {
            // Skip interior cells, already processed above.
            if (lx > 0 && lx < 7 && ly > 0 && ly < 7)
                continue;

            int neighborCount = 0;

#pragma unroll
            for (int dy = -1; dy <= 1; ++dy)
            {
#pragma unroll
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    neighborCount += getBitFromTiles(tiles, lx + dx, ly + dy);
                }
            }

            const int bitIdx = (ly << 3) + lx;
            const int isAlive = (int)((center >> bitIdx) & 1ULL);
            const int newState = (neighborCount == 3) | (isAlive & (neighborCount == 2));

            nextTile |= (std::uint64_t)newState << bitIdx;
        }
    }

    // Write the updated tile to output.
    output[idx] = nextTile;
}

/**
 * @brief Host-side function to launch one step of the Game of Life.
 *
 * @param input            Device pointer to the input grid (bit-packed).
 * @param output           Device pointer to the output grid (bit-packed).
 * @param grid_dimensions  Width/height of the square grid in cells.
 *
 * Both input and output must be allocated with cudaMalloc and must contain
 * (grid_dimensions/8) * (grid_dimensions/8) elements of std::uint64_t.
 *
 * Any necessary synchronization is expected to be handled by the caller.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int tilesPerRow = grid_dimensions >> 3;  // grid_dimensions / 8

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (tilesPerRow + blockDim.x - 1) / blockDim.x,
        (tilesPerRow + blockDim.y - 1) / blockDim.y);

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
