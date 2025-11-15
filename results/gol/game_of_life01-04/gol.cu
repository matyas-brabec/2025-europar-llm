#include <cuda_runtime.h>

/*
 * High-performance CUDA implementation of one step of Conway's Game of Life.
 *
 * Design notes:
 * - Grid is square (N x N), N is a power of two > 512.
 * - Each cell is stored as a byte (`bool` on the device, but treated as `unsigned char`).
 * - The kernel uses 2D tiling with shared memory to minimize global memory traffic.
 *   Each block loads a tile of size (BLOCK_DIM_Y x BLOCK_DIM_X) plus a 1-cell halo
 *   on each side (i.e., (BLOCK_DIM_Y+2) x (BLOCK_DIM_X+2) into shared memory).
 * - Boundary conditions: cells outside the grid are treated as dead (0).
 * - Pointers are given as `bool*` but reinterpreted as `unsigned char*` to avoid
 *   potential issues with `bool` on the device and to make aliasing rules explicit.
 */

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 32
#endif

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 8
#endif

// Load helper that uses the read-only data path on capable architectures.
__device__ __forceinline__ unsigned char load_cell(const unsigned char* ptr)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

/*
 * CUDA kernel implementing one step of Conway's Game of Life.
 *
 * Parameters:
 *   input  - pointer to input grid on device, 1 byte per cell (0 or 1).
 *   output - pointer to output grid on device, 1 byte per cell (0 or 1).
 *   width  - grid width (also height, since the grid is square).
 */
__global__ void game_of_life_kernel(const unsigned char* __restrict__ input,
                                    unsigned char* __restrict__ output,
                                    int width)
{
    // Shared memory tile with 1-cell halo on all sides.
    // Layout: tile[0 .. BLOCK_DIM_Y+1][0 .. BLOCK_DIM_X+1]
    __shared__ unsigned char tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    const int global_x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int global_y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

    const int local_x  = threadIdx.x + 1; // +1 accounts for left halo
    const int local_y  = threadIdx.y + 1; // +1 accounts for top halo

    // Load center cell (corresponds to tile[local_y][local_x]).
    unsigned char center = 0;
    if (global_x < width && global_y < width) {
        const int idx = global_y * width + global_x;
        center = load_cell(input + idx);
    }
    tile[local_y][local_x] = center;

    // Load left halo cells: tile[local_y][0]
    if (threadIdx.x == 0) {
        unsigned char val = 0;
        const int nx = global_x - 1;
        if (nx >= 0 && global_y < width) {
            const int idx = global_y * width + nx;
            val = load_cell(input + idx);
        }
        tile[local_y][0] = val;
    }

    // Load right halo cells: tile[local_y][BLOCK_DIM_X + 1]
    if (threadIdx.x == BLOCK_DIM_X - 1) {
        unsigned char val = 0;
        const int nx = global_x + 1;
        if (nx < width && global_y < width) {
            const int idx = global_y * width + nx;
            val = load_cell(input + idx);
        }
        tile[local_y][BLOCK_DIM_X + 1] = val;
    }

    // Load top halo cells: tile[0][local_x]
    if (threadIdx.y == 0) {
        unsigned char val = 0;
        const int ny = global_y - 1;
        if (ny >= 0 && global_x < width) {
            const int idx = ny * width + global_x;
            val = load_cell(input + idx);
        }
        tile[0][local_x] = val;
    }

    // Load bottom halo cells: tile[BLOCK_DIM_Y + 1][local_x]
    if (threadIdx.y == BLOCK_DIM_Y - 1) {
        unsigned char val = 0;
        const int ny = global_y + 1;
        if (ny < width && global_x < width) {
            const int idx = ny * width + global_x;
            val = load_cell(input + idx);
        }
        tile[BLOCK_DIM_Y + 1][local_x] = val;
    }

    // Load corner halo cells.
    // Top-left corner: tile[0][0]
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned char val = 0;
        const int nx = global_x - 1;
        const int ny = global_y - 1;
        if (nx >= 0 && ny >= 0) {
            const int idx = ny * width + nx;
            val = load_cell(input + idx);
        }
        tile[0][0] = val;
    }

    // Top-right corner: tile[0][BLOCK_DIM_X + 1]
    if (threadIdx.x == BLOCK_DIM_X - 1 && threadIdx.y == 0) {
        unsigned char val = 0;
        const int nx = global_x + 1;
        const int ny = global_y - 1;
        if (nx < width && ny >= 0) {
            const int idx = ny * width + nx;
            val = load_cell(input + idx);
        }
        tile[0][BLOCK_DIM_X + 1] = val;
    }

    // Bottom-left corner: tile[BLOCK_DIM_Y + 1][0]
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_DIM_Y - 1) {
        unsigned char val = 0;
        const int nx = global_x - 1;
        const int ny = global_y + 1;
        if (nx >= 0 && ny < width) {
            const int idx = ny * width + nx;
            val = load_cell(input + idx);
        }
        tile[BLOCK_DIM_Y + 1][0] = val;
    }

    // Bottom-right corner: tile[BLOCK_DIM_Y + 1][BLOCK_DIM_X + 1]
    if (threadIdx.x == BLOCK_DIM_X - 1 && threadIdx.y == BLOCK_DIM_Y - 1) {
        unsigned char val = 0;
        const int nx = global_x + 1;
        const int ny = global_y + 1;
        if (nx < width && ny < width) {
            const int idx = ny * width + nx;
            val = load_cell(input + idx);
        }
        tile[BLOCK_DIM_Y + 1][BLOCK_DIM_X + 1] = val;
    }

    // Ensure the entire tile (including halo) is loaded before computation.
    __syncthreads();

    // Threads that map outside the grid don't produce an output cell.
    if (global_x >= width || global_y >= width) {
        return;
    }

    // Read neighbors from shared memory.
    // Indices relative to (local_y, local_x) correspond to:
    //  (-1,-1) (-1,0) (-1,+1)
    //  ( 0,-1)  cell  ( 0,+1)
    //  (+1,-1) (+1,0) (+1,+1)
    int sum = 0;
    sum += tile[local_y - 1][local_x - 1];
    sum += tile[local_y - 1][local_x    ];
    sum += tile[local_y - 1][local_x + 1];
    sum += tile[local_y    ][local_x - 1];
    sum += tile[local_y    ][local_x + 1];
    sum += tile[local_y + 1][local_x - 1];
    sum += tile[local_y + 1][local_x    ];
    sum += tile[local_y + 1][local_x + 1];

    const unsigned char alive = tile[local_y][local_x];

    // Apply Conway's Game of Life rules in a branchless manner:
    // - A live cell survives if it has 2 or 3 neighbors.
    // - A dead cell becomes alive if it has exactly 3 neighbors.
    // new_state = (sum == 3) || (alive && sum == 2)
    const unsigned char survive = (sum == 2) & alive;
    const unsigned char birth   = (sum == 3);
    const unsigned char new_state = survive | birth;

    const int out_idx = global_y * width + global_x;
    output[out_idx] = new_state;
}

/*
 * Host function that launches one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *   input           - device pointer to input grid (bool*, 1 byte per cell).
 *   output          - device pointer to output grid (bool*, 1 byte per cell).
 *   grid_dimensions - width and height of the square grid (power of two).
 *
 * Notes:
 *   - `input` and `output` must have been allocated by cudaMalloc with at least
 *     grid_dimensions * grid_dimensions bytes.
 *   - No host-device synchronization is performed here; the caller is responsible
 *     for synchronizing and for error checking after the kernel launch if desired.
 */
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int width = grid_dimensions;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (width + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Reinterpret bool* as unsigned char* to ensure well-defined byte-level access.
    const unsigned char* d_input  = reinterpret_cast<const unsigned char*>(input);
    unsigned char*       d_output = reinterpret_cast<unsigned char*>(output);

    game_of_life_kernel<<<gridDim, blockDim>>>(d_input, d_output, width);
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
