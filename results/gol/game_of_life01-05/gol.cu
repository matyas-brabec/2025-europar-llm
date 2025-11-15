// Conway's Game of Life - High-performance CUDA implementation.
//
// This implementation targets modern NVIDIA data-center GPUs (e.g., A100/H100)
// and emphasizes global memory efficiency and parallelism.
//
// Key optimization ideas:
// 1. Each thread updates one cell.
// 2. Each block processes a rectangular tile of the grid.
// 3. A shared-memory tile with a 1-cell halo on all sides is used to cache
//    the block's cells and their immediate neighbors, so each cell's state
//    is read from global memory at most once per generation.
//    - Without this tiling, each cell would require 9 global loads (itself
//      + 8 neighbors). With tiling, the amortized loads are near 1 per cell.
// 4. The grid is guaranteed to be square, with size a power of 2 > 512,
//    which means it is divisible by our chosen block dimensions. Hence, no
//    per-thread bounds checks are required for the interior of the grid.
// 5. Boundary handling: cells outside the grid are treated as dead. The
//    shared-memory halo is explicitly filled with zeros for out-of-bounds
//    neighbors.
//
// The host function run_game_of_life launches a single kernel to advance the
// automaton by one generation. It assumes that `input` and `output` are
// device pointers allocated via cudaMalloc, and that any synchronization is
// handled by the caller.

#include <cuda_runtime.h>
#include <cstddef>

// Tunable block dimensions.
//
// BLOCK_DIM_X should be a multiple of 32 to give coalesced loads along X
// within a warp. BLOCK_DIM_Y trades off between occupancy and shared-memory
// footprint. (32 x 8 = 256 threads per block.)
constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 8;

// CUDA kernel that computes one generation of Conway's Game of Life.
//
// Parameters:
//   input  - device pointer to current state grid (bool per cell)
//   output - device pointer to next state grid (bool per cell)
//   n      - grid dimension (grid is n x n)
//
// The grid is partitioned into tiles of size BLOCK_DIM_Y x BLOCK_DIM_X.
// For each tile, we load an expanded (BLOCK_DIM_Y+2) x (BLOCK_DIM_X+2)
// region into shared memory to cover the 1-cell halo of neighbors.
template <int BX, int BY>
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int n)
{
    // Shared-memory tile:
    // - Interior [1..BY][1..BX] holds the block's cells.
    // - Border rows/columns at index 0 and BY+1 / BX+1 hold the halo.
    __shared__ bool tile[BY + 2][BX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gx = blockIdx.x * BX + tx;  // global x coordinate
    const int gy = blockIdx.y * BY + ty;  // global y coordinate

    // Use 64-bit indexing to be robust for very large grids
    const std::size_t n64  = static_cast<std::size_t>(n);
    const std::size_t gx64 = static_cast<std::size_t>(gx);
    const std::size_t gy64 = static_cast<std::size_t>(gy);
    const std::size_t idx  = gy64 * n64 + gx64;

    // 1. Load the central cell corresponding to this thread into shared memory.
    tile[ty + 1][tx + 1] = input[idx];

    // 2. Load halo cells. Each halo cell is loaded by exactly one thread.
    //    Out-of-bounds global coordinates are treated as dead (false).

    // Left halo for this row: (gx - 1, gy)
    if (tx == 0) {
        bool val = false;
        if (gx > 0) {
            val = input[idx - 1];
        }
        tile[ty + 1][0] = val;
    }

    // Right halo for this row: (gx + 1, gy)
    if (tx == BX - 1) {
        bool val = false;
        if (gx < n - 1) {
            val = input[idx + 1];
        }
        tile[ty + 1][BX + 1] = val;
    }

    // Top halo for this column: (gx, gy - 1)
    if (ty == 0) {
        bool val = false;
        if (gy > 0) {
            val = input[idx - n64];
        }
        tile[0][tx + 1] = val;
    }

    // Bottom halo for this column: (gx, gy + 1)
    if (ty == BY - 1) {
        bool val = false;
        if (gy < n - 1) {
            val = input[idx + n64];
        }
        tile[BY + 1][tx + 1] = val;
    }

    // Corner halos:

    // Top-left halo: (gx - 1, gy - 1)
    if (tx == 0 && ty == 0) {
        bool val = false;
        if (gx > 0 && gy > 0) {
            val = input[idx - n64 - 1];
        }
        tile[0][0] = val;
    }

    // Top-right halo: (gx + 1, gy - 1)
    if (tx == BX - 1 && ty == 0) {
        bool val = false;
        if (gx < n - 1 && gy > 0) {
            val = input[idx - n64 + 1];
        }
        tile[0][BX + 1] = val;
    }

    // Bottom-left halo: (gx - 1, gy + 1)
    if (tx == 0 && ty == BY - 1) {
        bool val = false;
        if (gx > 0 && gy < n - 1) {
            val = input[idx + n64 - 1];
        }
        tile[BY + 1][0] = val;
    }

    // Bottom-right halo: (gx + 1, gy + 1)
    if (tx == BX - 1 && ty == BY - 1) {
        bool val = false;
        if (gx < n - 1 && gy < n - 1) {
            val = input[idx + n64 + 1];
        }
        tile[BY + 1][BX + 1] = val;
    }

    // Make sure all shared-memory loads are visible before computing neighbors.
    __syncthreads();

    // 3. Compute the sum of the 8 neighbors for this cell.
    //
    // The current cell is at shared-memory coordinates (ty+1, tx+1).
    // Its neighbors in shared memory:
    //   (ty+0, tx+0) (ty+0, tx+1) (ty+0, tx+2)
    //   (ty+1, tx+0)             (ty+1, tx+2)
    //   (ty+2, tx+0) (ty+2, tx+1) (ty+2, tx+2)
    int neighbor_count = 0;

    neighbor_count += tile[ty + 0][tx + 0];
    neighbor_count += tile[ty + 0][tx + 1];
    neighbor_count += tile[ty + 0][tx + 2];

    neighbor_count += tile[ty + 1][tx + 0];
    neighbor_count += tile[ty + 1][tx + 2];

    neighbor_count += tile[ty + 2][tx + 0];
    neighbor_count += tile[ty + 2][tx + 1];
    neighbor_count += tile[ty + 2][tx + 2];

    const bool alive = tile[ty + 1][tx + 1];

    // 4. Apply Conway's Game of Life rules:
    //    - Any live cell with 2 or 3 live neighbors survives.
    //    - Any dead cell with exactly 3 live neighbors becomes a live cell.
    //    - All other live cells die, all other dead cells stay dead.
    bool next_state;
    if (alive) {
        next_state = (neighbor_count == 2 || neighbor_count == 3);
    } else {
        next_state = (neighbor_count == 3);
    }

    // 5. Store the updated state back to global memory.
    output[idx] = next_state;
}

// Host-side function to run one generation of Conway's Game of Life on the GPU.
//
// Parameters:
//   input           - device pointer to current state grid (bool per cell)
//   output          - device pointer to next state grid (bool per cell)
//   grid_dimensions - width/height of the square grid (n), a power of 2 > 512
//
// Assumptions:
//   - `input` and `output` are allocated via cudaMalloc with size n*n*sizeof(bool).
//   - `input` and `output` do not alias.
//   - The caller is responsible for any needed host-device synchronization.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int n = grid_dimensions;

    // Given that `n` is guaranteed to be a power of 2 and greater than 512,
    // it is divisible by BLOCK_DIM_X (32) and BLOCK_DIM_Y (8). Thus, we can
    // tile the domain exactly with no partial blocks.
    const dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 grid_dim(n / BLOCK_DIM_X, n / BLOCK_DIM_Y);

    // Launch the templated kernel specialized to our chosen block dimensions.
    game_of_life_kernel<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid_dim, block_dim>>>(input, output, n);

    // No synchronization here; the caller is responsible for it.
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
