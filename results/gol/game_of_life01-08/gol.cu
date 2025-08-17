#include <cuda_runtime.h>
#include <stdint.h>

/*
    High-performance CUDA implementation of one step of Conway's Game of Life.

    Key optimizations:
    - Tiled shared memory with a 1-cell halo (BLOCK_Y + 2) x (BLOCK_X + 2) to reuse neighbor data and minimize global memory traffic.
    - Coalesced global memory access: each warp loads contiguous bytes for the interior of the tile.
    - Boundary handling: cells outside the grid are treated as dead (0). Threads at the tile edges cooperatively load halo cells with proper bounds checks.
    - Branch-free rule application where possible: new_state = (neighbors == 3) | (alive & (neighbors == 2)).

    Notes:
    - The grid is square with dimensions N x N (N = grid_dimensions), power-of-two > 512, fitting in GPU memory.
    - Input and output are bool arrays in device memory. We reinterpret bool as uint8_t (1 byte) for efficient arithmetic.
    - This kernel avoids host-device synchronization; the caller is responsible for synchronization if needed.
    - BLOCK_X and BLOCK_Y are tuned for modern data center GPUs (A100/H100). 32x32 provides good balance of occupancy and memory reuse.
*/

#ifndef LIFEBLOCK_X
#define LIFEBLOCK_X 32
#endif

#ifndef LIFEBLOCK_Y
#define LIFEBLOCK_Y 32
#endif

// CUDA kernel: one step of Game of Life.
__global__ void game_of_life_kernel(const uint8_t* __restrict__ in,
                                    uint8_t* __restrict__ out,
                                    int N)
{
    // Shared memory tile with 1-cell halo on all sides.
    // Using uint8_t (1 byte) since input values are 0/1. The shared memory footprint is small (~ (BLOCK_Y+2)*(BLOCK_X+2) bytes).
    __shared__ uint8_t tile[LIFEBLOCK_Y + 2][LIFEBLOCK_X + 2];

    // Global coordinates for the cell this thread updates
    const int gx = blockIdx.x * LIFEBLOCK_X + threadIdx.x;
    const int gy = blockIdx.y * LIFEBLOCK_Y + threadIdx.y;

    // Local shared-memory coordinates (+1 offset for halo)
    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;

    // Lambda to load a global cell with zero-padding outside the grid.
    auto load_global = [&](int y, int x) -> uint8_t {
        // Treat out-of-bounds as dead (0).
        if ((unsigned)x >= (unsigned)N || (unsigned)y >= (unsigned)N) return 0;
        return in[y * N + x];
    };

    // Load the interior cell for this thread into shared memory.
    tile[sy][sx] = load_global(gy, gx);

    // Load halo cells: left and right columns
    if (threadIdx.x == 0) {
        tile[sy][0] = load_global(gy, gx - 1);
    }
    if (threadIdx.x == LIFEBLOCK_X - 1) {
        tile[sy][LIFEBLOCK_X + 1] = load_global(gy, gx + 1);
    }

    // Load halo cells: top and bottom rows
    if (threadIdx.y == 0) {
        tile[0][sx] = load_global(gy - 1, gx);
    }
    if (threadIdx.y == LIFEBLOCK_Y - 1) {
        tile[LIFEBLOCK_Y + 1][sx] = load_global(gy + 1, gx);
    }

    // Load halo corners (four threads will handle these)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0][0] = load_global(gy - 1, gx - 1);
    }
    if (threadIdx.x == LIFEBLOCK_X - 1 && threadIdx.y == 0) {
        tile[0][LIFEBLOCK_X + 1] = load_global(gy - 1, gx + 1);
    }
    if (threadIdx.x == 0 && threadIdx.y == LIFEBLOCK_Y - 1) {
        tile[LIFEBLOCK_Y + 1][0] = load_global(gy + 1, gx - 1);
    }
    if (threadIdx.x == LIFEBLOCK_X - 1 && threadIdx.y == LIFEBLOCK_Y - 1) {
        tile[LIFEBLOCK_Y + 1][LIFEBLOCK_X + 1] = load_global(gy + 1, gx + 1);
    }

    // Ensure the entire tile (including halo) is loaded before computation.
    __syncthreads();

    // Only compute and write if inside the grid.
    if ((unsigned)gx < (unsigned)N && (unsigned)gy < (unsigned)N) {
        // Accumulate neighbor count from 8 surrounding cells.
        // Values are 0/1, so sum is in [0..8].
        int neighbors = 0;

        // Top row
        neighbors += tile[sy - 1][sx - 1];
        neighbors += tile[sy - 1][sx    ];
        neighbors += tile[sy - 1][sx + 1];
        // Middle row (left and right)
        neighbors += tile[sy    ][sx - 1];
        neighbors += tile[sy    ][sx + 1];
        // Bottom row
        neighbors += tile[sy + 1][sx - 1];
        neighbors += tile[sy + 1][sx    ];
        neighbors += tile[sy + 1][sx + 1];

        const uint8_t alive = tile[sy][sx];

        // Conway's rules:
        // - A live cell survives with 2 or 3 neighbors.
        // - A dead cell becomes alive with exactly 3 neighbors.
        // Branch-minimized formulation:
        // new_state = (neighbors == 3) | (alive & (neighbors == 2));
        const uint8_t born     = (neighbors == 3);
        const uint8_t survive  = (neighbors == 2) & alive;
        const uint8_t newState = born | survive;

        out[gy * N + gx] = newState;
    }
}

/*
    Host helper to launch the kernel.

    Parameters:
    - input:  device pointer to N*N bool array (true=alive, false=dead)
    - output: device pointer to N*N bool array to store the next state
    - grid_dimensions: N, where the grid is N x N

    Behavior:
    - Reinterprets bool arrays as uint8_t (0 or 1) for arithmetic.
    - Launches a tiled kernel; no device synchronization performed here.
*/
/// @FIXED
/// extern "C" void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    if (grid_dimensions <= 0) return;

    const int N = grid_dimensions;

    // Configure execution parameters. For power-of-two N (>512), N is typically divisible by 32.
    dim3 block(LIFEBLOCK_X, LIFEBLOCK_Y, 1);
    dim3 grid((N + LIFEBLOCK_X - 1) / LIFEBLOCK_X,
              (N + LIFEBLOCK_Y - 1) / LIFEBLOCK_Y,
              1);

    // Reinterpret bool pointers as byte pointers (bool is 1 byte on CUDA).
    const uint8_t* in  = reinterpret_cast<const uint8_t*>(input);
    uint8_t* out       = reinterpret_cast<uint8_t*>(output);

    // Launch the kernel. No synchronization here; the caller can synchronize if needed.
    game_of_life_kernel<<<grid, block>>>(in, out, N);
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
