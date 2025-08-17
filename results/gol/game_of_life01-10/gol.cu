#include <cuda_runtime.h>
#include <stdint.h>

/*
  High-performance single-step Conway's Game of Life on modern NVIDIA GPUs (A100/H100).

  Key design choices:
  - 2D tiling in shared memory with a 1-cell halo to minimize global memory traffic. Each tile is
    BLOCK_W x BLOCK_H threads, and we allocate shared memory of (BLOCK_W+2) x (BLOCK_H+2) to hold
    neighbors for all cells in the tile.
  - Shared memory is stored as 32-bit integers (0/1) to avoid shared memory bank conflicts that are
    common with 8-bit storage.
  - Grid dimensions are powers of two > 512, so we choose block dimensions that divide N exactly to
    avoid partial tiles and simplify boundary handling.
  - Boundary condition: cells outside the grid are treated as dead (0), implemented by conditional
    loads when populating shared memory halo regions.

  The core update rule:
    next = (neighbors == 3) || (current && neighbors == 2)
*/

#ifndef GOL_BLOCK_W
#define GOL_BLOCK_W 64  // Must be power of two that divides grid_dimensions. 64 divides any power-of-two >= 64.
#endif

#ifndef GOL_BLOCK_H
#define GOL_BLOCK_H 8   // Must be power of two that divides grid_dimensions. 8 divides any power-of-two >= 8.
#endif

// The kernel is specialized by block dimensions via macros to keep shared memory statically sized.
template<int BLOCK_W, int BLOCK_H>
__global__ __launch_bounds__(BLOCK_W * BLOCK_H, 2)
void game_of_life_kernel_tiled(const bool* __restrict__ input,
                               bool* __restrict__ output,
                               int N)
{
    // Thread coordinates within block and grid
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BLOCK_W + tx;
    const int y  = blockIdx.y * BLOCK_H + ty;

    // Shared memory tile with 1-cell halo. Use 32-bit to avoid bank conflicts.
    __shared__ uint32_t tile[BLOCK_H + 2][BLOCK_W + 2];

    // Precompute row offsets using 64-bit to avoid overflow for very large N
    const size_t Nsz = static_cast<size_t>(N);
    const size_t row     = static_cast<size_t>(y) * Nsz;
    const size_t row_abv = (y > 0)        ? static_cast<size_t>(y - 1) * Nsz : 0;
    const size_t row_bel = (y + 1 < N)    ? static_cast<size_t>(y + 1) * Nsz : 0;

    // Load center cell for this thread into shared memory (as 0/1)
    uint32_t center = 0;
    // Since N is assumed divisible by BLOCK_W and BLOCK_H and x,y are within bounds by construction,
    // the following conditionals are still retained for safety and clarity.
    if (x < N && y < N) {
        center = static_cast<uint32_t>(input[row + static_cast<size_t>(x)]);
    }
    tile[ty + 1][tx + 1] = center;

    // Load halo cells around the tile. Each condition maps to a subset of threads to cooperatively
    // load the halo rows/cols. Out-of-bounds neighbors are treated as 0.
    // Left halo
    if (tx == 0) {
        uint32_t v = 0;
        if (x > 0 && y < N) {
            v = static_cast<uint32_t>(input[row + static_cast<size_t>(x - 1)]);
        }
        tile[ty + 1][0] = v;
    }
    // Right halo
    if (tx == BLOCK_W - 1) {
        uint32_t v = 0;
        if (x + 1 < N && y < N) {
            v = static_cast<uint32_t>(input[row + static_cast<size_t>(x + 1)]);
        }
        tile[ty + 1][BLOCK_W + 1] = v;
    }
    // Top halo
    if (ty == 0) {
        uint32_t v = 0;
        if (y > 0 && x < N) {
            v = static_cast<uint32_t>(input[row_abv + static_cast<size_t>(x)]);
        }
        tile[0][tx + 1] = v;
    }
    // Bottom halo
    if (ty == BLOCK_H - 1) {
        uint32_t v = 0;
        if (y + 1 < N && x < N) {
            v = static_cast<uint32_t>(input[row_bel + static_cast<size_t>(x)]);
        }
        tile[BLOCK_H + 1][tx + 1] = v;
    }
    // Corner halos
    if (tx == 0 && ty == 0) {
        uint32_t v = 0;
        if (x > 0 && y > 0) {
            v = static_cast<uint32_t>(input[row_abv + static_cast<size_t>(x - 1)]);
        }
        tile[0][0] = v;
    }
    if (tx == BLOCK_W - 1 && ty == 0) {
        uint32_t v = 0;
        if (x + 1 < N && y > 0) {
            v = static_cast<uint32_t>(input[row_abv + static_cast<size_t>(x + 1)]);
        }
        tile[0][BLOCK_W + 1] = v;
    }
    if (tx == 0 && ty == BLOCK_H - 1) {
        uint32_t v = 0;
        if (x > 0 && y + 1 < N) {
            v = static_cast<uint32_t>(input[row_bel + static_cast<size_t>(x - 1)]);
        }
        tile[BLOCK_H + 1][0] = v;
    }
    if (tx == BLOCK_W - 1 && ty == BLOCK_H - 1) {
        uint32_t v = 0;
        if (x + 1 < N && y + 1 < N) {
            v = static_cast<uint32_t>(input[row_bel + static_cast<size_t>(x + 1)]);
        }
        tile[BLOCK_H + 1][BLOCK_W + 1] = v;
    }

    __syncthreads();

    // Compute neighbor sum from shared memory
    // Note: tile indices are shifted by +1 due to halo
    const int lx = tx + 1;
    const int ly = ty + 1;

    // Eight-neighbor sum
    uint32_t neighbors = 0;
    neighbors += tile[ly - 1][lx - 1];
    neighbors += tile[ly - 1][lx    ];
    neighbors += tile[ly - 1][lx + 1];
    neighbors += tile[ly    ][lx - 1];
    neighbors += tile[ly    ][lx + 1];
    neighbors += tile[ly + 1][lx - 1];
    neighbors += tile[ly + 1][lx    ];
    neighbors += tile[ly + 1][lx + 1];

    // Apply Game of Life rules
    // next = (neighbors == 3) || (center && neighbors == 2)
    uint32_t next = (neighbors == 3u) | (center & (neighbors == 2u));

    // Write result back to global memory
    if (x < N && y < N) {
        output[row + static_cast<size_t>(x)] = static_cast<bool>(next);
    }
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Assumptions:
    // - grid_dimensions is a power of 2 greater than 512 (e.g., 1024, 2048, 4096, ...)
    // - The input and output are already allocated on device via cudaMalloc
    // - No synchronization is required here; the caller will handle it.

    const int N = grid_dimensions;

    // Choose 2D block and grid configuration.
    // Because N is a power of two and >= 512, it is divisible by 64 and 8.
    dim3 block(GOL_BLOCK_W, GOL_BLOCK_H, 1);
    dim3 grid(N / GOL_BLOCK_W, N / GOL_BLOCK_H, 1);

    // Launch the tiled kernel specialized with the block dimensions
    game_of_life_kernel_tiled<GOL_BLOCK_W, GOL_BLOCK_H><<<grid, block>>>(input, output, N);
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
