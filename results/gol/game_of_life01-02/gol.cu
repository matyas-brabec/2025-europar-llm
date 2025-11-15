// Conway's Game of Life - CUDA implementation optimized for modern NVIDIA GPUs (A100/H100)
//
// This implementation uses a 2D tiled stencil kernel with shared memory to
// minimize global memory traffic. Each thread computes one output cell.
// The input and output grids are in device memory (allocated via cudaMalloc)
// and are 1-byte per cell (bool). We reinterpret them as uint8_t for more
// predictable behavior and arithmetic.
//
// Key points:
// - 2D block of threads covers a tile of the grid.
// - Each block loads a tile of (BLOCK_DIM_Y + 2) x (BLOCK_DIM_X + 2) cells into
//   shared memory, including a 1-cell halo border on all sides.
// - Global memory loads for the halo cells handle the boundary condition
//   ("outside grid is dead") by loading 0 when indices are out of bounds.
// - Each thread then computes the next state for its corresponding cell using
//   the 8 neighbors in shared memory.
// - The rules are implemented in a branchless form for better warp efficiency.
// - Grid dimensions are assumed to be power-of-two and >= 512, but the kernel
//   also safely handles arbitrary sizes via boundary checks.
//
// The host function `run_game_of_life` launches one kernel step without
// performing any synchronization; the caller is responsible for synchronization
// and error checking if desired.

#include <cuda_runtime.h>
#include <stdint.h>

// Tuneable block dimensions. 32x32 tends to perform well on modern GPUs for
// this kind of regular stencil and works nicely with power-of-two grid sizes.
constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 32;

// CUDA kernel for one step of Conway's Game of Life.
// - input:  pointer to device memory, one byte per cell (0 or 1)
// - output: pointer to device memory, one byte per cell (0 or 1)
// - n:      grid dimension (width = height = n)
template <int BLOCK_X, int BLOCK_Y>
__global__ void game_of_life_kernel(const uint8_t* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    int n)
{
    // Shared memory tile with 1-cell halo on all sides:
    // tile height = BLOCK_Y + 2, width = BLOCK_X + 2
    __shared__ uint8_t tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global coordinates of the cell this thread is responsible for
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Load the shared memory tile, including halo.
    //
    // The tile's coordinate system:
    //   tile[0 .. BLOCK_Y+1][0 .. BLOCK_X+1]
    // where:
    //   - tile[1 .. BLOCK_Y][1 .. BLOCK_X] are the cells owned by this block
    //   - the halo is at indices 0 and BLOCK_Y+1 in Y, 0 and BLOCK_X+1 in X
    //
    // We map each tile coordinate (sx, sy) to global coordinates:
    //   global_x = blockIdx.x * BLOCK_X + (sx - 1)
    //   global_y = blockIdx.y * BLOCK_Y + (sy - 1)
    //
    // Threads cooperate to load the entire tile using strided loops so that
    // we can cover the (BLOCK_X+2) x (BLOCK_Y+2) region even when it is larger
    // than the thread block size.
    for (int sy = ty; sy < BLOCK_Y + 2; sy += BLOCK_Y) {
        const int global_y = blockIdx.y * BLOCK_Y + (sy - 1);
        const bool in_y = (global_y >= 0) && (global_y < n);

        for (int sx = tx; sx < BLOCK_X + 2; sx += BLOCK_X) {
            const int global_x = blockIdx.x * BLOCK_X + (sx - 1);
            const bool in_x = (global_x >= 0) && (global_x < n);

            uint8_t val = 0;
            if (in_x && in_y) {
                // Row-major indexing: index = y * n + x
                val = input[global_y * n + global_x];
            }

            tile[sy][sx] = val;
        }
    }

    __syncthreads();

    // Only threads whose global coordinates are within the grid compute/store
    if (x < n && y < n) {
        // Local coordinates in the shared memory tile for this thread's cell
        const int sx = tx + 1;
        const int sy = ty + 1;

        // Sum of the 8 neighbors around (sy, sx) in the shared memory tile.
        //
        // Explicitly enumerated for clarity and to help the compiler unroll.
        int neighbor_sum = 0;
        neighbor_sum += tile[sy - 1][sx - 1];
        neighbor_sum += tile[sy - 1][sx    ];
        neighbor_sum += tile[sy - 1][sx + 1];
        neighbor_sum += tile[sy    ][sx - 1];
        neighbor_sum += tile[sy    ][sx + 1];
        neighbor_sum += tile[sy + 1][sx - 1];
        neighbor_sum += tile[sy + 1][sx    ];
        neighbor_sum += tile[sy + 1][sx + 1];

        const uint8_t center = tile[sy][sx];

        // Branchless implementation of Conway's rules:
        // - If center == 1 (alive):
        //     survives if neighbor_sum is 2 or 3
        // - If center == 0 (dead):
        //     becomes alive if neighbor_sum is exactly 3
        //
        // This can be written as:
        //   new_state = (neighbor_sum == 3) || (center && neighbor_sum == 2)
        //
        // Using bitwise operations (booleans promoted to 0/1):
        const uint8_t is_3 = (neighbor_sum == 3);
        const uint8_t is_2 = (neighbor_sum == 2);
        const uint8_t new_state = static_cast<uint8_t>(is_3 | (center & is_2));

        output[y * n + x] = new_state;
    }
}

// Host function to run one step of Conway's Game of Life on the GPU.
//
// Parameters:
// - input:           device pointer to an n x n grid of bools (true/false).
// - output:          device pointer to an n x n grid of bools to store result.
// - grid_dimensions: n, the grid size (width = height = n).
//
// Both input and output must be allocated by cudaMalloc before calling this
// function. This function does not perform any synchronization or error
// checking; the caller is responsible for that if desired.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    if (grid_dimensions <= 0) {
        return;
    }

    // Reinterpret bool* as uint8_t* (bool is 1 byte on CUDA, but using uint8_t
    // makes arithmetic and shared memory usage more explicit and predictable).
    const uint8_t* d_input  = reinterpret_cast<const uint8_t*>(input);
    uint8_t*       d_output = reinterpret_cast<uint8_t*>(output);

    const int n = grid_dimensions;

    // Configure a 2D grid of 2D blocks.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Launch one kernel step. No synchronization here; the caller may call
    // cudaDeviceSynchronize() or use CUDA events as needed.
    game_of_life_kernel<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid, block>>>(d_input, d_output, n);
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
