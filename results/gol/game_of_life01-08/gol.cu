#include <cuda_runtime.h>

/*
 * High-performance CUDA implementation of one step of Conway's Game of Life.
 *
 * Assumptions:
 *  - Grid is square (N x N), N is a power of 2, N > 512.
 *  - Input and output are device pointers allocated with cudaMalloc.
 *  - Each cell is represented as a bool (false = dead, true = alive).
 *  - Cells outside the grid are considered dead (0) for neighbor computations.
 *
 * Performance considerations:
 *  - Each block computes a 2D tile of the grid.
 *  - The tile is loaded into shared memory with a 1-cell halo on all sides.
 *    This allows each cell's neighbors to be read from fast shared memory
 *    instead of repeatedly loading from global memory.
 *  - Shared memory dramatically reduces global memory traffic: each cell is
 *    loaded exactly once from global memory for a tile, but used by up to
 *    9 threads (itself + 8 neighbors).
 *  - Block dimensions are chosen to align with warp size for coalesced loads.
 */

#ifndef GOL_BLOCK_DIM_X
#define GOL_BLOCK_DIM_X 32  // Threads per block in X (must be multiple of warp size)
#endif

#ifndef GOL_BLOCK_DIM_Y
#define GOL_BLOCK_DIM_Y 16  // Threads per block in Y
#endif

// Kernel implementing one Game of Life step using shared-memory tiling.
template <int BLOCK_X, int BLOCK_Y>
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int N)
{
    // Shared memory tile with 1-cell halo in each direction:
    // tile[y][x], where:
    //   y in [0, BLOCK_Y+1], x in [0, BLOCK_X+1]
    // Entries [1..BLOCK_Y][1..BLOCK_X] correspond to the block's cells.
    // Halo rows/cols (0 and BLOCK_Y+1, 0 and BLOCK_X+1) store neighbors.
    //
    // We store cells as unsigned char (0 or 1) for cheap integer accumulation.
    __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Thread coordinates within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global coordinates of the cell this thread is responsible for
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Shared memory coordinates (offset by +1 due to halo)
    const int sx = tx + 1;
    const int sy = ty + 1;

    // Helper lambda to safely load a cell from global memory as 0/1.
    // Returns 0 if (gx, gy) is outside the grid.
    auto load_cell = [&](int gx, int gy) -> unsigned char {
        if (gx >= 0 && gx < N && gy >= 0 && gy < N) {
            // Read bool and convert to 0 or 1.
            return static_cast<unsigned char>(input[gy * N + gx]);
        }
        return 0;
    };

    // Load the central cell for this thread into shared memory.
    tile[sy][sx] = load_cell(x, y);

    // Load halo cells. Each halo element is loaded by one (or a few) threads.
    // Left halo column
    if (tx == 0) {
        tile[sy][0] = load_cell(x - 1, y);
    }
    // Right halo column
    if (tx == BLOCK_X - 1) {
        tile[sy][BLOCK_X + 1] = load_cell(x + 1, y);
    }
    // Top halo row
    if (ty == 0) {
        tile[0][sx] = load_cell(x, y - 1);
    }
    // Bottom halo row
    if (ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][sx] = load_cell(x, y + 1);
    }
    // Top-left corner halo
    if (tx == 0 && ty == 0) {
        tile[0][0] = load_cell(x - 1, y - 1);
    }
    // Top-right corner halo
    if (tx == BLOCK_X - 1 && ty == 0) {
        tile[0][BLOCK_X + 1] = load_cell(x + 1, y - 1);
    }
    // Bottom-left corner halo
    if (tx == 0 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][0] = load_cell(x - 1, y + 1);
    }
    // Bottom-right corner halo
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][BLOCK_X + 1] = load_cell(x + 1, y + 1);
    }

    // Ensure all threads have loaded their portion of the tile + halo
    __syncthreads();

    // Threads whose global coordinates are outside the grid don't produce output.
    if (x >= N || y >= N) {
        return;
    }

    // Compute the number of alive neighbors from shared memory.
    // Note: Each tile entry is 0 (dead) or 1 (alive), so summation is cheap.
    unsigned int neighbor_count = 0;
    neighbor_count += tile[sy - 1][sx - 1];
    neighbor_count += tile[sy - 1][sx    ];
    neighbor_count += tile[sy - 1][sx + 1];
    neighbor_count += tile[sy    ][sx - 1];
    neighbor_count += tile[sy    ][sx + 1];
    neighbor_count += tile[sy + 1][sx - 1];
    neighbor_count += tile[sy + 1][sx    ];
    neighbor_count += tile[sy + 1][sx + 1];

    const bool alive = (tile[sy][sx] != 0);

    // Apply Conway's Game of Life rules:
    // - Any alive cell with fewer than 2 alive neighbors dies.
    // - Any alive cell with 2 or 3 alive neighbors survives.
    // - Any alive cell with more than 3 alive neighbors dies.
    // - Any dead cell with exactly 3 alive neighbors becomes alive.
    bool next_state;
    if (alive) {
        next_state = (neighbor_count == 2 || neighbor_count == 3);
    } else {
        next_state = (neighbor_count == 3);
    }

    // Write the result back to global memory.
    output[y * N + x] = next_state;
}

/*
 * Host function to launch one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *  - input:  Device pointer to the input grid (N x N), bool per cell.
 *  - output: Device pointer to the output grid (N x N), bool per cell.
 *  - grid_dimensions: N, the width and height of the square grid.
 *
 * Notes:
 *  - This function does not perform any host-device synchronization.
 *    The caller is responsible for synchronization (e.g., cudaDeviceSynchronize).
 *  - input and output must already be allocated on the device with cudaMalloc.
 */
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Configure CUDA launch parameters.
    dim3 block_dim(GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y);
    dim3 grid_dim(
        (N + GOL_BLOCK_DIM_X - 1) / GOL_BLOCK_DIM_X,
        (N + GOL_BLOCK_DIM_Y - 1) / GOL_BLOCK_DIM_Y
    );

    // Launch the kernel. No synchronization here; caller handles it.
    game_of_life_kernel<GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y>
        <<<grid_dim, block_dim>>>(input, output, N);
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
