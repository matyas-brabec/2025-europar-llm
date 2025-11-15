// Conway's Game of Life - CUDA implementation optimized for modern NVIDIA GPUs (A100/H100).
//
// This file provides:
//   - A high-performance CUDA kernel that computes one generation of Conway's Game of Life
//     on a square grid of boolean cells stored in device memory.
//   - A host function `run_game_of_life` that configures and launches the kernel.
//
// Assumptions & design choices:
//
//   * Grid:
//       - Square grid of size N x N (N = grid_dimensions).
//       - N is a power of 2, strictly greater than 512 (so N >= 1024).
//       - Since 1024 = 32 * 32, every such N is divisible by 32, which allows us
//         to use 32x32 CUDA thread blocks without any partial blocks.
//
//   * Memory layout:
//       - Input and output grids are 1D arrays of `bool` in row-major order:
//           index = y * N + x, for 0 <= x, y < N.
//       - Memory for `input` and `output` is allocated on the device via `cudaMalloc`
//         by the caller.
//       - Outside-grid cells are treated as dead (0).
//
//   * Kernel optimization:
//       - Uses 2D thread blocks of size 32x32 to match warp size and leverage
//         coalesced memory accesses.
//       - Each block processes a 32x32 tile of the grid.
//       - Uses shared memory with a halo (1-cell border) to minimize global memory
//         traffic: each cell is loaded at most once from global memory per iteration.
//         Per 32x32 block, global loads:
//            - 32*32 = 1024 center cells
//            - 4 * 32 = 128 edge halo cells
//            - 4 corner halo cells
//           => 1156 global loads for 1024 cells (~1.13 loads/cell vs. naive 9 loads/cell).
//       - The neighbor sum is fully unrolled (no loops) for maximum performance.
//       - Thread blocks are configured so no thread falls outside the grid, avoiding
//         extra bounds checks in the main path; boundary conditions are handled only
//         when loading halos into shared memory.
//
//   * API:
//       void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
//       - Launches a single asynchronous kernel and returns immediately.
//       - No host-side synchronization or error checking is performed; the caller
//         is responsible for synchronization and error handling if desired.
//

#include <cuda_runtime.h>

// Tunable block dimensions. For N being a power of 2 >= 1024, N is always divisible by 32.
#ifndef GOL_BLOCK_DIM_X
#define GOL_BLOCK_DIM_X 32
#endif

#ifndef GOL_BLOCK_DIM_Y
#define GOL_BLOCK_DIM_Y 32
#endif

// CUDA kernel that computes one step of Conway's Game of Life.
// Input:
//   input  - pointer to device memory (N x N bool grid, row-major).
//   output - pointer to device memory (N x N bool grid, row-major).
//   N      - grid dimension (width and height), power of 2, N >= 1024.
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int N)
{
    // Shared memory tile with halo:
    //
    //   tile has dimensions (GOL_BLOCK_DIM_Y + 2) x (GOL_BLOCK_DIM_X + 2)
    //
    //   Layout:
    //       halo row/col indices:
    //         [0][1..GOL_BLOCK_DIM_X]      : top halo row
    //         [GOL_BLOCK_DIM_Y+1][1..X]    : bottom halo row
    //         [1..GOL_BLOCK_DIM_Y][0]      : left halo column
    //         [1..GOL_BLOCK_DIM_Y][X+1]    : right halo column
    //         corners: [0][0], [0][X+1], [Y+1][0], [Y+1][X+1]
    //
    //       interior cells (actual block data):
    //         [1..GOL_BLOCK_DIM_Y][1..GOL_BLOCK_DIM_X]
    //
    //   We store 0/1 as unsigned char to keep shared memory footprint small.
    //
    __shared__ unsigned char tile[GOL_BLOCK_DIM_Y + 2][GOL_BLOCK_DIM_X + 2];

    // Local thread coordinates within the block.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global coordinates of the cell this thread is responsible for.
    const int x = blockIdx.x * GOL_BLOCK_DIM_X + tx;
    const int y = blockIdx.y * GOL_BLOCK_DIM_Y + ty;

    // Row-major index into input/output arrays.
    const int idx = y * N + x;

    // ------------------------------------------------------------------------
    // 1. Load the block's interior cell into shared memory.
    //    Each thread loads one cell into tile[ty+1][tx+1].
    //    These loads are fully coalesced: for a fixed 'y', threads in a warp
    //    access consecutive 'x' values.
    // ------------------------------------------------------------------------
    tile[ty + 1][tx + 1] = static_cast<unsigned char>(input[idx]);

    // ------------------------------------------------------------------------
    // 2. Load halo cells.
    //
    //    Only threads on the block boundaries load halo cells.
    //    For boundary blocks at the grid edges, some halo cells are outside
    //    of the global grid; they are treated as dead (0).
    // ------------------------------------------------------------------------

    // Left halo: threads in first column (tx == 0) load tile[ty+1][0].
    if (tx == 0) {
        const int nx = x - 1;
        if (nx >= 0) {
            tile[ty + 1][0] = static_cast<unsigned char>(input[y * N + nx]);
        } else {
            tile[ty + 1][0] = 0;
        }
    }

    // Right halo: threads in last column (tx == GOL_BLOCK_DIM_X - 1) load tile[ty+1][GOL_BLOCK_DIM_X+1].
    if (tx == GOL_BLOCK_DIM_X - 1) {
        const int nx = x + 1;
        if (nx < N) {
            tile[ty + 1][GOL_BLOCK_DIM_X + 1] = static_cast<unsigned char>(input[y * N + nx]);
        } else {
            tile[ty + 1][GOL_BLOCK_DIM_X + 1] = 0;
        }
    }

    // Top halo: threads in first row (ty == 0) load tile[0][tx+1].
    if (ty == 0) {
        const int ny = y - 1;
        if (ny >= 0) {
            tile[0][tx + 1] = static_cast<unsigned char>(input[ny * N + x]);
        } else {
            tile[0][tx + 1] = 0;
        }
    }

    // Bottom halo: threads in last row (ty == GOL_BLOCK_DIM_Y - 1) load tile[GOL_BLOCK_DIM_Y+1][tx+1].
    if (ty == GOL_BLOCK_DIM_Y - 1) {
        const int ny = y + 1;
        if (ny < N) {
            tile[GOL_BLOCK_DIM_Y + 1][tx + 1] = static_cast<unsigned char>(input[ny * N + x]);
        } else {
            tile[GOL_BLOCK_DIM_Y + 1][tx + 1] = 0;
        }
    }

    // Corner halos: four threads (one per block corner) handle the four corners.

    // Top-left corner halo: tile[0][0]
    if (tx == 0 && ty == 0) {
        const int nx = x - 1;
        const int ny = y - 1;
        if (nx >= 0 && ny >= 0) {
            tile[0][0] = static_cast<unsigned char>(input[ny * N + nx]);
        } else {
            tile[0][0] = 0;
        }
    }

    // Top-right corner halo: tile[0][GOL_BLOCK_DIM_X+1]
    if (tx == GOL_BLOCK_DIM_X - 1 && ty == 0) {
        const int nx = x + 1;
        const int ny = y - 1;
        if (nx < N && ny >= 0) {
            tile[0][GOL_BLOCK_DIM_X + 1] = static_cast<unsigned char>(input[ny * N + nx]);
        } else {
            tile[0][GOL_BLOCK_DIM_X + 1] = 0;
        }
    }

    // Bottom-left corner halo: tile[GOL_BLOCK_DIM_Y+1][0]
    if (tx == 0 && ty == GOL_BLOCK_DIM_Y - 1) {
        const int nx = x - 1;
        const int ny = y + 1;
        if (nx >= 0 && ny < N) {
            tile[GOL_BLOCK_DIM_Y + 1][0] = static_cast<unsigned char>(input[ny * N + nx]);
        } else {
            tile[GOL_BLOCK_DIM_Y + 1][0] = 0;
        }
    }

    // Bottom-right corner halo: tile[GOL_BLOCK_DIM_Y+1][GOL_BLOCK_DIM_X+1]
    if (tx == GOL_BLOCK_DIM_X - 1 && ty == GOL_BLOCK_DIM_Y - 1) {
        const int nx = x + 1;
        const int ny = y + 1;
        if (nx < N && ny < N) {
            tile[GOL_BLOCK_DIM_Y + 1][GOL_BLOCK_DIM_X + 1] = static_cast<unsigned char>(input[ny * N + nx]);
        } else {
            tile[GOL_BLOCK_DIM_Y + 1][GOL_BLOCK_DIM_X + 1] = 0;
        }
    }

    // Ensure all threads have finished loading their tile and halo cells.
    __syncthreads();

    // ------------------------------------------------------------------------
    // 3. Compute the next state for this cell using the classic Game of Life rules.
    //
    //    We work entirely from shared memory tile[...][...] to avoid repeated
    //    global memory accesses. Neighbor sum is fully unrolled for speed.
    //
    //    Neighborhood mapping around tile[ty+1][tx+1]:
    //
    //        (ty,   tx)     (ty,   tx+1)     (ty,   tx+2)
    //        (ty+1, tx)     (  * center* )   (ty+1, tx+2)
    //        (ty+2, tx)     (ty+2, tx+1)     (ty+2, tx+2)
    //
    // ------------------------------------------------------------------------

    // Sum of the 8 neighbors (all in shared memory).
    unsigned int alive_neighbors =
        static_cast<unsigned int>(tile[ty    ][tx    ]) +
        static_cast<unsigned int>(tile[ty    ][tx + 1]) +
        static_cast<unsigned int>(tile[ty    ][tx + 2]) +
        static_cast<unsigned int>(tile[ty + 1][tx    ]) +
        static_cast<unsigned int>(tile[ty + 1][tx + 2]) +
        static_cast<unsigned int>(tile[ty + 2][tx    ]) +
        static_cast<unsigned int>(tile[ty + 2][tx + 1]) +
        static_cast<unsigned int>(tile[ty + 2][tx + 2]);

    const unsigned char current = tile[ty + 1][tx + 1];

    // Game of Life rules:
    //
    //   - Any alive cell with 2 or 3 neighbors survives.
    //   - Any dead cell with exactly 3 neighbors becomes alive.
    //   - All other cells die or remain dead.
    //
    // Using boolean logic:
    //   next = (alive_neighbors == 3) || (current == 1 && alive_neighbors == 2)
    //
    const bool next_state =
        (alive_neighbors == 3u) ||
        (current == 1u && alive_neighbors == 2u);

    // Write the next state back to global memory.
    output[idx] = next_state;
}

// Host function that launches the Game of Life kernel for one generation.
//
// Parameters:
//   input           - device pointer to current state grid (N x N, bool).
//   output          - device pointer to next state grid (N x N, bool).
//   grid_dimensions - N (width and height of the square grid), power of 2, > 512.
//
// Notes:
//   - This function performs a single asynchronous kernel launch.
//   - It does not perform any host-device synchronization or error checking;
//     the caller should handle that as needed for timing or correctness.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Given N is a power of 2 >= 1024 and block dims are 32x32, N is guaranteed
    // to be divisible by both GOL_BLOCK_DIM_X and GOL_BLOCK_DIM_Y.
    const dim3 block_dim(GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y);
    const dim3 grid_dim(N / GOL_BLOCK_DIM_X, N / GOL_BLOCK_DIM_Y);

    // Launch kernel (asynchronous).
    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, N);
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// this label is used to identify the memory layout of the input and output arrays by the framework
// MEMORY_LAYOUT: BOOLS
