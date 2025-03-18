#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define the block size for the kernel.
// This block size is chosen to maximize occupancy and allow efficient shared-memory tiling.
#define BLOCK_SIZE 32

// CUDA kernel that computes one Game of Life generation.
// Each thread computes the next state for one cell in the grid.
// Shared memory is used to load a tile of the input grid including a one‐cell halo.
__global__ void game_of_life_kernel(const bool* d_input, bool* d_output, int grid_dimensions)
{
    // Shared memory tile dimensions: BLOCK_SIZE x BLOCK_SIZE interior plus a 1-cell halo border on each side.
    // The array dimension is (BLOCK_SIZE + 2) x (BLOCK_SIZE + 2).
    __shared__ bool tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute thread (local) indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Compute corresponding global grid indices.
    int gx = blockIdx.x * BLOCK_SIZE + tx;
    int gy = blockIdx.y * BLOCK_SIZE + ty;

    // Load interior cell into shared memory.
    bool cell = false;
    if (gx < grid_dimensions && gy < grid_dimensions)
    {
        cell = d_input[gy * grid_dimensions + gx];
    }
    tile[ty + 1][tx + 1] = cell;

    // Load halo cells into shared memory.
    // Left halo: load cell at (gx-1, gy) into tile[ty+1][0].
    if (tx == 0)
    {
        int gxh = gx - 1;
        bool left_cell = false;
        if (gxh >= 0 && gy < grid_dimensions)
        {
            left_cell = d_input[gy * grid_dimensions + gxh];
        }
        tile[ty + 1][0] = left_cell;
    }
    // Right halo: load cell at (gx+1, gy) into tile[ty+1][BLOCK_SIZE+1].
    if (tx == BLOCK_SIZE - 1)
    {
        int gxh = gx + 1;
        bool right_cell = false;
        if (gxh < grid_dimensions && gy < grid_dimensions)
        {
            right_cell = d_input[gy * grid_dimensions + gxh];
        }
        tile[ty + 1][BLOCK_SIZE + 1] = right_cell;
    }
    // Top halo: load cell at (gx, gy-1) into tile[0][tx+1].
    if (ty == 0)
    {
        int gyh = gy - 1;
        bool top_cell = false;
        if (gyh >= 0 && gx < grid_dimensions)
        {
            top_cell = d_input[gyh * grid_dimensions + gx];
        }
        tile[0][tx + 1] = top_cell;
    }
    // Bottom halo: load cell at (gx, gy+1) into tile[BLOCK_SIZE+1][tx+1].
    if (ty == BLOCK_SIZE - 1)
    {
        int gyh = gy + 1;
        bool bottom_cell = false;
        if (gyh < grid_dimensions && gx < grid_dimensions)
        {
            bottom_cell = d_input[gyh * grid_dimensions + gx];
        }
        tile[BLOCK_SIZE + 1][tx + 1] = bottom_cell;
    }
    // Top-left corner: load cell at (gx-1, gy-1) into tile[0][0].
    if (tx == 0 && ty == 0)
    {
        int gxh = gx - 1;
        int gyh = gy - 1;
        bool top_left = false;
        if (gxh >= 0 && gyh >= 0)
        {
            top_left = d_input[gyh * grid_dimensions + gxh];
        }
        tile[0][0] = top_left;
    }
    // Top-right corner: load cell at (gx+1, gy-1) into tile[0][BLOCK_SIZE+1].
    if (tx == BLOCK_SIZE - 1 && ty == 0)
    {
        int gxh = gx + 1;
        int gyh = gy - 1;
        bool top_right = false;
        if (gxh < grid_dimensions && gyh >= 0)
        {
            top_right = d_input[gyh * grid_dimensions + gxh];
        }
        tile[0][BLOCK_SIZE + 1] = top_right;
    }
    // Bottom-left corner: load cell at (gx-1, gy+1) into tile[BLOCK_SIZE+1][0].
    if (tx == 0 && ty == BLOCK_SIZE - 1)
    {
        int gxh = gx - 1;
        int gyh = gy + 1;
        bool bottom_left = false;
        if (gxh >= 0 && gyh < grid_dimensions)
        {
            bottom_left = d_input[gyh * grid_dimensions + gxh];
        }
        tile[BLOCK_SIZE + 1][0] = bottom_left;
    }
    // Bottom-right corner: load cell at (gx+1, gy+1) into tile[BLOCK_SIZE+1][BLOCK_SIZE+1].
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1)
    {
        int gxh = gx + 1;
        int gyh = gy + 1;
        bool bottom_right = false;
        if (gxh < grid_dimensions && gyh < grid_dimensions)
        {
            bottom_right = d_input[gyh * grid_dimensions + gxh];
        }
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = bottom_right;
    }

    // Ensure all threads have finished loading shared data.
    __syncthreads();

    // Compute the next state only if the thread corresponds to a valid grid cell.
    if (gx < grid_dimensions && gy < grid_dimensions)
    {
        // Count the number of alive neighbors.
        // The indices in the shared memory tile are arranged as follows:
        //   tile[ty+1][tx+1] is the current cell.
        //   Its eight neighbors:
        //     tile[ty][tx]      (top-left)
        //     tile[ty][tx+1]    (top-center)
        //     tile[ty][tx+2]    (top-right)
        //     tile[ty+1][tx]    (left)
        //     tile[ty+1][tx+2]  (right)
        //     tile[ty+2][tx]    (bottom-left)
        //     tile[ty+2][tx+1]  (bottom-center)
        //     tile[ty+2][tx+2]  (bottom-right)
        int alive_neighbors = 0;
        alive_neighbors += tile[ty][tx];         // top-left
        alive_neighbors += tile[ty][tx + 1];       // top-center
        alive_neighbors += tile[ty][tx + 2];       // top-right
        alive_neighbors += tile[ty + 1][tx];       // left
        alive_neighbors += tile[ty + 1][tx + 2];   // right
        alive_neighbors += tile[ty + 2][tx];       // bottom-left
        alive_neighbors += tile[ty + 2][tx + 1];   // bottom-center
        alive_neighbors += tile[ty + 2][tx + 2];   // bottom-right

        // Apply Conway’s Game of Life rules:
        // - A dead cell with exactly 3 alive neighbors becomes alive.
        // - A live cell with 2 or 3 alive neighbors stays alive; otherwise it dies.
        bool current = tile[ty + 1][tx + 1];
        bool next_state = (alive_neighbors == 3) || (current && (alive_neighbors == 2));

        // Write the result to the output grid.
        d_output[gy * grid_dimensions + gx] = next_state;
    }
}

// Host-side function that launches one step of Conway’s Game of Life simulation.
// 'input' and 'output' are pointers to grids allocated in GPU memory.
// 'grid_dimensions' is the width/height of the square grid.
/// @FIXED: extern "C" void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Set up thread block dimensions.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // Compute grid dimensions so that each block processes a BLOCK_SIZE x BLOCK_SIZE tile.
    dim3 gridDim((grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE, (grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel.
    // Performance is achieved by efficient shared memory tiling and manual unrolling of neighbor sum.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);

    // No explicit synchronization is performed here as per the requirement;
    // any necessary host-device synchronization is assumed to be handled by the caller.
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
