// CUDA implementation of one step of Conway's Game of Life on a square grid.
//
// Representation:
// - The grid is grid_dimensions x grid_dimensions cells.
// - grid_dimensions is a power of two, >= 512, and a multiple of 8.
// - Cells are bit-packed into 64-bit words (std::uint64_t).
// - Each 64-bit word encodes an 8x8 tile of cells.
// - Within a tile, bit (row * 8 + col) corresponds to the cell at
//   local coordinates (row, col), where row and col are in [0,7] and
//   (0,0) is the top-left cell of the tile.
// - Tiles are stored in row-major order: tile index = tile_y * tiles_per_side + tile_x.
//   tiles_per_side = grid_dimensions / 8.
//
// Boundary conditions:
// - Cells outside the global grid are treated as dead (0).
//
// Algorithm (kernel):
// - One CUDA thread processes one 8x8 tile.
// - For each tile, the thread loads up to 9 neighboring tiles into registers:
//   center, north, south, west, east, and the four diagonals.
// - It builds a 10x10 "patch" of cells representing the center tile and its 1-cell halo.
//   This patch is filled entirely in registers (as unsigned char values 0 or 1).
// - For each of the 64 cells of the center tile, it sums the 8 neighbors in the patch,
//   applies the Game of Life rules, and packs the result back into a 64-bit word.
//
// Performance notes:
// - No shared or texture memory is used, as per problem statement.
// - Global memory accesses are coalesced for the main tile load/store.
// - Small loops are unrolled to encourage the compiler to optimize aggressively.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Kernel: one thread per 8x8 tile
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int tiles_per_side)
{
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_side || tile_y >= tiles_per_side) {
        return;
    }

    const int stride      = tiles_per_side;
    const int idx_center  = tile_y * stride + tile_x;

    // Load center tile
    const std::uint64_t center = input[idx_center];

    // Load neighbor tiles with boundary checks
    std::uint64_t north = 0, south = 0, west = 0, east = 0;
    std::uint64_t nw = 0, ne = 0, sw = 0, se = 0;

    if (tile_y > 0) {
        const int idx_n = idx_center - stride;
        north = input[idx_n];
        if (tile_x > 0) {
            nw = input[idx_n - 1];
        }
        if (tile_x + 1 < stride) {
            ne = input[idx_n + 1];
        }
    }

    if (tile_y + 1 < stride) {
        const int idx_s = idx_center + stride;
        south = input[idx_s];
        if (tile_x > 0) {
            sw = input[idx_s - 1];
        }
        if (tile_x + 1 < stride) {
            se = input[idx_s + 1];
        }
    }

    if (tile_x > 0) {
        west = input[idx_center - 1];
    }
    if (tile_x + 1 < stride) {
        east = input[idx_center + 1];
    }

    // 10x10 patch of the neighborhood around the 8x8 center tile.
    // patch[y][x], with y,x in [0,9].
    // The 8x8 center tile corresponds to patch[1..8][1..8].
    // Values are 0 (dead) or 1 (alive).
    unsigned char patch[10][10];

    // Initialize patch to all dead
#pragma unroll
    for (int y = 0; y < 10; ++y) {
#pragma unroll
        for (int x = 0; x < 10; ++x) {
            patch[y][x] = 0;
        }
    }

    // Fill center tile into patch[1..8][1..8]
#pragma unroll
    for (int row = 0; row < 8; ++row) {
        const std::uint8_t row_bits =
            static_cast<std::uint8_t>((center >> (row * 8)) & 0xFFu);

        patch[row + 1][1] =  row_bits        & 0x1u;
        patch[row + 1][2] = (row_bits >> 1) & 0x1u;
        patch[row + 1][3] = (row_bits >> 2) & 0x1u;
        patch[row + 1][4] = (row_bits >> 3) & 0x1u;
        patch[row + 1][5] = (row_bits >> 4) & 0x1u;
        patch[row + 1][6] = (row_bits >> 5) & 0x1u;
        patch[row + 1][7] = (row_bits >> 6) & 0x1u;
        patch[row + 1][8] = (row_bits >> 7) & 0x1u;
    }

    // Fill north halo row (patch[0][1..8]) from north tile row 7
    if (tile_y > 0) {
        const std::uint8_t row_bits =
            static_cast<std::uint8_t>((north >> (7 * 8)) & 0xFFu);

        patch[0][1] =  row_bits        & 0x1u;
        patch[0][2] = (row_bits >> 1) & 0x1u;
        patch[0][3] = (row_bits >> 2) & 0x1u;
        patch[0][4] = (row_bits >> 3) & 0x1u;
        patch[0][5] = (row_bits >> 4) & 0x1u;
        patch[0][6] = (row_bits >> 5) & 0x1u;
        patch[0][7] = (row_bits >> 6) & 0x1u;
        patch[0][8] = (row_bits >> 7) & 0x1u;
    }

    // Fill south halo row (patch[9][1..8]) from south tile row 0
    if (tile_y + 1 < stride) {
        const std::uint8_t row_bits =
            static_cast<std::uint8_t>(south & 0xFFu);

        patch[9][1] =  row_bits        & 0x1u;
        patch[9][2] = (row_bits >> 1) & 0x1u;
        patch[9][3] = (row_bits >> 2) & 0x1u;
        patch[9][4] = (row_bits >> 3) & 0x1u;
        patch[9][5] = (row_bits >> 4) & 0x1u;
        patch[9][6] = (row_bits >> 5) & 0x1u;
        patch[9][7] = (row_bits >> 6) & 0x1u;
        patch[9][8] = (row_bits >> 7) & 0x1u;
    }

    // Fill west halo column (patch[1..8][0]) from west tile column 7
    if (tile_x > 0) {
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const std::uint8_t row_bits =
                static_cast<std::uint8_t>((west >> (row * 8)) & 0xFFu);
            patch[row + 1][0] = (row_bits >> 7) & 0x1u;
        }
    }

    // Fill east halo column (patch[1..8][9]) from east tile column 0
    if (tile_x + 1 < stride) {
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const std::uint8_t row_bits =
                static_cast<std::uint8_t>((east >> (row * 8)) & 0xFFu);
            patch[row + 1][9] = row_bits & 0x1u;
        }
    }

    // Fill four halo corner cells from diagonal tiles
    if (tile_x > 0 && tile_y > 0) {
        // NW tile, row 7 col 7 -> bit 63
        patch[0][0] = static_cast<unsigned char>((nw >> 63) & 0x1u);
    }
    if (tile_x + 1 < stride && tile_y > 0) {
        // NE tile, row 7 col 0 -> bit 56
        patch[0][9] = static_cast<unsigned char>((ne >> 56) & 0x1u);
    }
    if (tile_x > 0 && tile_y + 1 < stride) {
        // SW tile, row 0 col 7 -> bit 7
        patch[9][0] = static_cast<unsigned char>((sw >> 7) & 0x1u);
    }
    if (tile_x + 1 < stride && tile_y + 1 < stride) {
        // SE tile, row 0 col 0 -> bit 0
        patch[9][9] = static_cast<unsigned char>(se & 0x1u);
    }

    // Compute next generation for the center 8x8 tile and pack into out_tile
    std::uint64_t out_tile = 0;

#pragma unroll
    for (int row = 0; row < 8; ++row) {
#pragma unroll
        for (int col = 0; col < 8; ++col) {
            // Sum the 8 neighbors around patch[row+1][col+1]
            int neighbors = 0;
            neighbors += patch[row    ][col    ];
            neighbors += patch[row    ][col + 1];
            neighbors += patch[row    ][col + 2];
            neighbors += patch[row + 1][col    ];
            neighbors += patch[row + 1][col + 2];
            neighbors += patch[row + 2][col    ];
            neighbors += patch[row + 2][col + 1];
            neighbors += patch[row + 2][col + 2];

            const unsigned char self = patch[row + 1][col + 1];

            // Game of Life rules (branchless):
            // - A live cell survives if it has 2 or 3 neighbors.
            // - A dead cell becomes live if it has exactly 3 neighbors.
            const unsigned char is_n3 = (neighbors == 3);
            const unsigned char is_n2 = (neighbors == 2);
            const unsigned char survive = self & is_n2;
            const unsigned char alive_next = static_cast<unsigned char>(is_n3 | survive);

            const int bit_index = row * 8 + col;
            out_tile |= (static_cast<std::uint64_t>(alive_next) << bit_index);
        }
    }

    output[idx_center] = out_tile;
}

} // anonymous namespace

// Host function: launch one Game of Life step on the GPU.
// - input  : device pointer to bit-packed input grid.
// - output : device pointer to bit-packed output grid (same layout).
// - grid_dimensions : width/height of the square grid (must be a multiple of 8).
//
// This function performs no host-side synchronization; the caller is
// responsible for any required cudaDeviceSynchronize() or stream management.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int tiles_per_side = grid_dimensions / 8;

    // Tunable block size; 16x16 gives 256 threads per block.
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (tiles_per_side + block_dim.x - 1) / block_dim.x,
        (tiles_per_side + block_dim.y - 1) / block_dim.y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_side);
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
