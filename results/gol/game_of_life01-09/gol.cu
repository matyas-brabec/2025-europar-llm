#include <cuda_runtime.h>

#define BLOCK_X 32
#define BLOCK_Y 16

// CUDA kernel that computes one step of Conway's Game of Life.
// The grid is stored as a flat array in row-major order.
// Each block operates on a tile of size BLOCK_Y x BLOCK_X, with a
// 1-cell halo on all sides in shared memory for neighbor access.
__global__ void game_of_life_kernel(const unsigned char* __restrict__ input,
                                    unsigned char* __restrict__ output,
                                    int width,
                                    int height)
{
    // Thread indices within block and corresponding global coordinates
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BLOCK_X + tx;
    const int y  = blockIdx.y * BLOCK_Y + ty;

    // Shared memory tile:
    // height: BLOCK_Y + 2, width: BLOCK_X + 2 (including 1-cell halo on all sides)
    const int tile_width = BLOCK_X + 2;
    __shared__ unsigned char tile[(BLOCK_Y + 2) * (BLOCK_X + 2)];

    // Index of this thread's own cell in the shared-memory tile
    const int s_idx = (ty + 1) * tile_width + (tx + 1);

    // Load center cell into shared memory (0 if outside grid)
    unsigned char center = 0;
    if (x < width && y < height) {
        const int g_idx = y * width + x;
        center = input[g_idx];
    }
    tile[s_idx] = center;

    // Load halo cells from global memory into shared memory.
    // Cells outside the grid are treated as dead (0).

    // Left halo (column 0 of tile)
    if (tx == 0) {
        unsigned char val = 0;
        const int gx = x - 1;
        if (gx >= 0 && y < height) {
            const int g_idx = y * width + gx;
            val = input[g_idx];
        }
        tile[(ty + 1) * tile_width + 0] = val;
    }

    // Right halo (column BLOCK_X + 1 of tile)
    if (tx == BLOCK_X - 1) {
        unsigned char val = 0;
        const int gx = x + 1;
        if (gx < width && y < height) {
            const int g_idx = y * width + gx;
            val = input[g_idx];
        }
        tile[(ty + 1) * tile_width + (BLOCK_X + 1)] = val;
    }

    // Top halo (row 0 of tile, excluding corners)
    if (ty == 0) {
        unsigned char val = 0;
        const int gy = y - 1;
        if (gy >= 0 && x < width) {
            const int g_idx = gy * width + x;
            val = input[g_idx];
        }
        tile[0 * tile_width + (tx + 1)] = val;
    }

    // Bottom halo (row BLOCK_Y + 1 of tile, excluding corners)
    if (ty == BLOCK_Y - 1) {
        unsigned char val = 0;
        const int gy = y + 1;
        if (gy < height && x < width) {
            const int g_idx = gy * width + x;
            val = input[g_idx];
        }
        tile[(BLOCK_Y + 1) * tile_width + (tx + 1)] = val;
    }

    // Corner halos

    // Top-left corner
    if (tx == 0 && ty == 0) {
        unsigned char val = 0;
        const int gx = x - 1;
        const int gy = y - 1;
        if (gx >= 0 && gy >= 0) {
            const int g_idx = gy * width + gx;
            val = input[g_idx];
        }
        tile[0 * tile_width + 0] = val;
    }

    // Top-right corner
    if (tx == BLOCK_X - 1 && ty == 0) {
        unsigned char val = 0;
        const int gx = x + 1;
        const int gy = y - 1;
        if (gx < width && gy >= 0) {
            const int g_idx = gy * width + gx;
            val = input[g_idx];
        }
        tile[0 * tile_width + (BLOCK_X + 1)] = val;
    }

    // Bottom-left corner
    if (tx == 0 && ty == BLOCK_Y - 1) {
        unsigned char val = 0;
        const int gx = x - 1;
        const int gy = y + 1;
        if (gx >= 0 && gy < height) {
            const int g_idx = gy * width + gx;
            val = input[g_idx];
        }
        tile[(BLOCK_Y + 1) * tile_width + 0] = val;
    }

    // Bottom-right corner
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        unsigned char val = 0;
        const int gx = x + 1;
        const int gy = y + 1;
        if (gx < width && gy < height) {
            const int g_idx = gy * width + gx;
            val = input[g_idx];
        }
        tile[(BLOCK_Y + 1) * tile_width + (BLOCK_X + 1)] = val;
    }

    // Make sure all halo and center loads are visible to all threads
    __syncthreads();

    // Threads that map outside the grid don't produce results
    if (x >= width || y >= height) {
        return;
    }

    // Compute number of alive neighbors from shared memory
    const int north = s_idx - tile_width;
    const int south = s_idx + tile_width;

    const int neighbors =
        tile[north - 1] + tile[north] + tile[north + 1] +
        tile[s_idx - 1]               + tile[s_idx + 1] +
        tile[south - 1] + tile[south] + tile[south + 1];

    const unsigned char alive = tile[s_idx];

    // Game of Life rules:
    // Alive cell survives with 2 or 3 neighbors.
    // Dead cell becomes alive with exactly 3 neighbors.
    // Branchless form:
    // new_state = (neighbors == 3) | (alive & (neighbors == 2));
    const unsigned char n_eq_3 = (neighbors == 3);
    const unsigned char n_eq_2 = (neighbors == 2);
    const unsigned char new_state = (unsigned char)(n_eq_3 | (alive & n_eq_2));

    const int g_idx = y * width + x;
    output[g_idx] = new_state;
}

// Host function that launches one Game of Life step on the GPU.
// - input:  device pointer to current state (bool[N*N])
// - output: device pointer to next state (bool[N*N])
// - grid_dimensions: side length N of the square grid
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int width  = grid_dimensions;
    const int height = grid_dimensions;

    // Interpret bool arrays as byte arrays. bool is 1 byte on CUDA devices,
    // and character types are allowed to alias any object representation.
    const unsigned char* d_in  = reinterpret_cast<const unsigned char*>(input);
    unsigned char*       d_out = reinterpret_cast<unsigned char*>(output);

    // Block configuration: 32x16 threads (512 threads per block).
    // X dimension is a full warp for optimal coalescing.
    dim3 block_dim(BLOCK_X, BLOCK_Y);
    dim3 grid_dim(
        (width  + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y
    );

    // Launch kernel on default stream; synchronization is handled by caller.
    game_of_life_kernel<<<grid_dim, block_dim>>>(d_in, d_out, width, height);
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
