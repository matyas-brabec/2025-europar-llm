// Conway's Game of Life - CUDA implementation optimized for modern NVIDIA GPUs (A100/H100)
//
// This implementation assumes:
//   - The grid is square: N x N.
//   - N is a power of 2, greater than 512.
//   - N is small enough that the N x N grid fits into GPU memory.
//   - Input and output are linear device arrays allocated via cudaMalloc.
//   - Cells outside the grid are treated as dead (0).
//
// Each thread processes one cell. We use a 2D thread block and a 2D grid. To reduce global
// memory traffic, we load a tile of the grid into shared memory, including a 1-cell halo on
// all sides. Each block therefore uses a shared memory tile of size (BLOCK_Y+2) x (BLOCK_X+2).
//
// A typical block size of 32x16 threads is used for good occupancy and coalesced accesses.
// Since N is a power of two greater than 512, it is divisible by both 32 and 16, so the grid
// is exactly covered with no partial blocks. This allows us to avoid bounds checks for the
// main cell computing region (we still handle boundary cells logically via halos).
//
// input  : const bool* (device pointer) - current generation
// output : bool*       (device pointer) - next generation
// N      : int                     - grid dimension (N x N)
//
// Usage:
//   - Call run_game_of_life(input_dev, output_dev, N).
//   - Synchronization and error checking (cudaDeviceSynchronize / cudaGetLastError) are
//     expected to be handled by the caller.

#include <cuda_runtime.h>
#include <stdint.h>

// Tunable tile/block size. These are chosen to give good performance on modern GPUs.
// BLOCK_X should be a multiple of warp size (32) for coalesced global memory accesses.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 16;

// CUDA kernel for one step of Conway's Game of Life.
__global__
void game_of_life_kernel(const bool* __restrict__ input,
                         bool* __restrict__ output,
                         int grid_dim)
{
    // Shared memory tile with 1-cell halo on each side:
    //   - Interior: [1 .. BLOCK_Y] x [1 .. BLOCK_X]
    //   - Halo: rows 0 and BLOCK_Y+1, cols 0 and BLOCK_X+1
    __shared__ uint8_t tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BLOCK_X + tx;
    const int y  = blockIdx.y * BLOCK_Y + ty;

    // Compute 1D index into the linear grid array.
    // Use size_t for safety w.r.t. large grids.
    const size_t idx = static_cast<size_t>(y) * grid_dim + x;

    // Load the center cell into shared memory.
    // We rely on grid_dim being exactly divisible by BLOCK_X and BLOCK_Y
    // (guaranteed by the problem's constraints: N is a power of 2 > 512).
    tile[ty + 1][tx + 1] = static_cast<uint8_t>(input[idx]);

    // Load halo cells. Threads at block borders are responsible for halo loads.
    // Outside-of-grid cells are treated as dead (0).

    // Left halo (same row, x-1)
    if (tx == 0) {
        uint8_t val = 0;
        if (x > 0) {
            val = static_cast<uint8_t>(input[idx - 1]);
        }
        tile[ty + 1][0] = val;
    }

    // Right halo (same row, x+1)
    if (tx == BLOCK_X - 1) {
        uint8_t val = 0;
        if (x + 1 < grid_dim) {
            val = static_cast<uint8_t>(input[idx + 1]);
        }
        tile[ty + 1][BLOCK_X + 1] = val;
    }

    // Top halo (same column, y-1)
    if (ty == 0) {
        uint8_t val = 0;
        if (y > 0) {
            val = static_cast<uint8_t>(input[idx - static_cast<size_t>(grid_dim)]);
        }
        tile[0][tx + 1] = val;
    }

    // Bottom halo (same column, y+1)
    if (ty == BLOCK_Y - 1) {
        uint8_t val = 0;
        if (y + 1 < grid_dim) {
            val = static_cast<uint8_t>(input[idx + static_cast<size_t>(grid_dim)]);
        }
        tile[BLOCK_Y + 1][tx + 1] = val;
    }

    // Corner halos
    if (tx == 0 && ty == 0) {
        // Top-left (x-1, y-1)
        uint8_t val = 0;
        if (x > 0 && y > 0) {
            val = static_cast<uint8_t>(input[idx - static_cast<size_t>(grid_dim) - 1]);
        }
        tile[0][0] = val;
    }

    if (tx == BLOCK_X - 1 && ty == 0) {
        // Top-right (x+1, y-1)
        uint8_t val = 0;
        if (x + 1 < grid_dim && y > 0) {
            val = static_cast<uint8_t>(input[idx - static_cast<size_t>(grid_dim) + 1]);
        }
        tile[0][BLOCK_X + 1] = val;
    }

    if (tx == 0 && ty == BLOCK_Y - 1) {
        // Bottom-left (x-1, y+1)
        uint8_t val = 0;
        if (x > 0 && y + 1 < grid_dim) {
            val = static_cast<uint8_t>(input[idx + static_cast<size_t>(grid_dim) - 1]);
        }
        tile[BLOCK_Y + 1][0] = val;
    }

    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        // Bottom-right (x+1, y+1)
        uint8_t val = 0;
        if (x + 1 < grid_dim && y + 1 < grid_dim) {
            val = static_cast<uint8_t>(input[idx + static_cast<size_t>(grid_dim) + 1]);
        }
        tile[BLOCK_Y + 1][BLOCK_X + 1] = val;
    }

    // Ensure the entire tile (center + halos) is loaded before computing.
    __syncthreads();

    // Compute the sum of the 8 neighbors for this cell.
    // The current cell is at tile[ty+1][tx+1]; neighbors are around it.
    const int sum =
        tile[ty    ][tx    ] + tile[ty    ][tx + 1] + tile[ty    ][tx + 2] +
        tile[ty + 1][tx    ]                         + tile[ty + 1][tx + 2] +
        tile[ty + 2][tx    ] + tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];

    const uint8_t alive = tile[ty + 1][tx + 1];

    // Apply Game of Life rules:
    // - Any live cell with fewer than 2 live neighbors dies.
    // - Any live cell with 2 or 3 live neighbors lives on.
    // - Any live cell with more than 3 live neighbors dies.
    // - Any dead cell with exactly 3 live neighbors becomes a live cell.
    //
    // This can be written as:
    //   new_alive = (sum == 3) || (alive && sum == 2);
    const bool new_alive = (sum == 3) || (alive && sum == 2);

    // Store the result back to global memory.
    output[idx] = new_alive;
}

// Host-side function to run one step of Conway's Game of Life on the GPU.
// input  : device pointer to bool[N * N] containing the current generation.
// output : device pointer to bool[N * N] where the next generation will be stored.
// grid_dimensions : N (must be a power of 2, >512).
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Configure thread block and grid dimensions.
    // Given the problem constraints (N is a power of 2 > 512), we can assume
    // grid_dimensions is divisible by BLOCK_X and BLOCK_Y.
    const dim3 block_dim(BLOCK_X, BLOCK_Y);
    const dim3 grid_dim(grid_dimensions / BLOCK_X,
                        grid_dimensions / BLOCK_Y);

    // Launch the kernel. Synchronization and error checking are delegated
    // to the caller, as per the problem statement.
    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, grid_dimensions);
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
