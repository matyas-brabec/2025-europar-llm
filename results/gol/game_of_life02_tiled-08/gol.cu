// CUDA implementation of a single step of Conway's Game of Life
// on a bit-packed grid where each std::uint64_t encodes an 8x8 tile.
//
// Layout and assumptions:
//  - The overall grid is square of size N x N cells.
//  - N is a power of 2, N > 512, and N is divisible by 8.
//  - The grid is tile-packed: each 8x8 tile of cells is represented by one
//    64-bit word. Tiles are stored in row-major order:
//       tile_index = tile_y * (N / 8) + tile_x
//    Within a tile, cells are stored in row-major order as bits:
//       bit_index = local_y * 8 + local_x,  (0 <= local_y, local_x < 8)
//       bit value 1 = alive, 0 = dead.
//  - Cells outside the grid are implicitly dead.
//  - The input and output buffers are device pointers allocated with cudaMalloc.
//  - Input and output must not alias (i.e. input != output).
//
// Parallelization strategy:
//  - One CUDA thread handles one 8x8 tile.
//  - Each thread loads its tile and the 8 neighboring tiles (3x3 block) into
//    registers. Missing neighbors at the boundaries are treated as all-dead (0).
//  - For each of the 64 cells in the tile, the thread counts the 8 neighbors
//    by consulting these 9 preloaded 64-bit words, then applies the Game of
//    Life rules and sets the corresponding bit in the output tile.
//
// Memory usage:
//  - Global memory only; no shared or texture memory is used (per problem
//    statement). Each tile update performs 9 * 8B = 72B of reads and 8B of
//    writes, which is very reasonable on modern GPUs.
//
// Performance notes:
//  - Neighbor location is computed arithmetically using local coordinates and
//    relative offsets, avoiding global index recomputation.
//  - Neighbor tile selection uses simple branching on offsets in {-1,0,1}.
//    These offsets depend only on local cell coordinates and loop indices,
//    so all threads in a warp follow identical control flow (no divergence).
//  - The kernel is launched with a 2D grid of thread blocks over tiles, and
//    no host-side synchronization is performed inside run_game_of_life.

#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel: update one simulation step for a tiled, bit-packed Game of Life grid.
// tiles_per_row = grid_dimensions / 8
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_per_row)
{
    // Compute tile coordinates handled by this thread
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_row || tile_y >= tiles_per_row)
        return;

    const int idx = tile_y * tiles_per_row + tile_x;

    // Load center tile
    const std::uint64_t tC = input[idx];

    // Load neighbor tiles with boundary checks.
    // Tiles outside the grid are treated as all-dead (0).
    std::uint64_t tN  = 0;
    std::uint64_t tS  = 0;
    std::uint64_t tW  = 0;
    std::uint64_t tE  = 0;
    std::uint64_t tNW = 0;
    std::uint64_t tNE = 0;
    std::uint64_t tSW = 0;
    std::uint64_t tSE = 0;

    // Vertical neighbors
    if (tile_y > 0) {
        tN = input[idx - tiles_per_row];
    }
    if (tile_y + 1 < tiles_per_row) {
        tS = input[idx + tiles_per_row];
    }

    // Horizontal neighbors
    if (tile_x > 0) {
        tW = input[idx - 1];
    }
    if (tile_x + 1 < tiles_per_row) {
        tE = input[idx + 1];
    }

    // Diagonal neighbors
    if (tile_y > 0 && tile_x > 0) {
        tNW = input[idx - tiles_per_row - 1];
    }
    if (tile_y > 0 && tile_x + 1 < tiles_per_row) {
        tNE = input[idx - tiles_per_row + 1];
    }
    if (tile_y + 1 < tiles_per_row && tile_x > 0) {
        tSW = input[idx + tiles_per_row - 1];
    }
    if (tile_y + 1 < tiles_per_row && tile_x + 1 < tiles_per_row) {
        tSE = input[idx + tiles_per_row + 1];
    }

    std::uint64_t next_tile = 0ull;

    // Process the 8x8 cells inside this tile.
    // local_y, local_x are coordinates inside the tile (0..7).
    // Bit index inside tC is (local_y * 8 + local_x).
#pragma unroll
    for (int local_y = 0; local_y < 8; ++local_y) {
        const int row_base_bit = local_y * 8;

#pragma unroll
        for (int local_x = 0; local_x < 8; ++local_x) {
            unsigned int neighbor_count = 0;

            // For each of the 8 neighbors (relative offsets dy, dx in {-1,0,1} \ {(0,0)}),
            // determine which tile it lies in (offset of -1,0,1 in tile space),
            // and the local coordinates inside that tile, then fetch the bit.
#pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        // Skip the cell itself
                        continue;
                    }

                    // Compute neighbor cell coordinates relative to this tile.
                    // We map neighbor local coordinates (which may be in [-1,8])
                    // to (tile_offset, local_coord) using simple arithmetic:
                    //
                    //   n = coord + delta + 8  => n in [7,16]
                    //   tile_offset = (n >> 3) - 1  => -1,0,1
                    //   local_coord = n & 7        => 0..7
                    //
                    // This avoids branches and division/modulo.
                    const int ny = local_y + dy + 8;
                    const int nx = local_x + dx + 8;

                    const int tile_off_y = (ny >> 3) - 1;  // -1, 0, 1
                    const int tile_off_x = (nx >> 3) - 1;  // -1, 0, 1

                    const int neighbor_local_y = ny & 7;
                    const int neighbor_local_x = nx & 7;

                    // Select the appropriate 8x8 tile word for this neighbor.
                    std::uint64_t tile_word;
                    if (tile_off_y == -1) {
                        if (tile_off_x == -1) {
                            tile_word = tNW;
                        } else if (tile_off_x == 0) {
                            tile_word = tN;
                        } else { // tile_off_x == 1
                            tile_word = tNE;
                        }
                    } else if (tile_off_y == 0) {
                        if (tile_off_x == -1) {
                            tile_word = tW;
                        } else if (tile_off_x == 0) {
                            tile_word = tC;  // not actually used because (dx,dy) != (0,0)
                        } else { // tile_off_x == 1
                            tile_word = tE;
                        }
                    } else { // tile_off_y == 1
                        if (tile_off_x == -1) {
                            tile_word = tSW;
                        } else if (tile_off_x == 0) {
                            tile_word = tS;
                        } else { // tile_off_x == 1
                            tile_word = tSE;
                        }
                    }

                    const int neighbor_bit_index = (neighbor_local_y << 3) + neighbor_local_x;
                    neighbor_count += static_cast<unsigned int>(
                        (tile_word >> neighbor_bit_index) & 1ull
                    );
                }
            }

            // Current cell state
            const int self_bit_index = row_base_bit + local_x;
            const unsigned int self_alive = static_cast<unsigned int>(
                (tC >> self_bit_index) & 1ull
            );

            // Game of Life rules:
            // - Alive cell survives iff it has 2 or 3 alive neighbors.
            // - Dead cell becomes alive iff it has exactly 3 alive neighbors.
            const unsigned int will_live =
                (neighbor_count == 3u) |
                (self_alive & (neighbor_count == 2u));

            next_tile |= (static_cast<std::uint64_t>(will_live) << self_bit_index);
        }
    }

    // Store result tile
    output[idx] = next_tile;
}

// Host function: launch the CUDA kernel to execute one Game of Life step.
//
// Parameters:
//  - input:  device pointer to input tiles (bit-packed, one uint64_t per 8x8 tile).
//  - output: device pointer to output tiles (same layout).
//  - grid_dimensions: width/height of the square grid in cells.
//                     Must be a power of two and divisible by 8.
//
// Notes:
//  - This function does not perform any host-device synchronization;
//    the caller is responsible for synchronizing if needed.
//  - input and output must not alias.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 8x8 tiles per row/column.
    const int tiles_per_row = grid_dimensions >> 3;  // grid_dimensions / 8

    // Configure a 2D grid over tiles. A 16x16 block (256 threads) is a good
    // general choice for modern GPUs.
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        (tiles_per_row + block_dim.x - 1) / block_dim.x,
        (tiles_per_row + block_dim.y - 1) / block_dim.y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_row);
    // No cudaDeviceSynchronize() here; caller handles synchronization.
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
