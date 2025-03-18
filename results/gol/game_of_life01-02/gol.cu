#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define block dimension (tile size). We choose 32x32 threads per block for high occupancy on modern GPUs.
#define BLOCK_DIM 32

// CUDA kernel to perform one iteration of Conway's Game of Life.
// Uses shared memory tiling with a 1-cell halo to avoid redundant global memory loads.
__global__ void game_of_life_kernel(const bool *input, bool *output, int grid_dim) {
    // Calculate global coordinates of the cell handled by this thread.
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

    // Indices in the shared memory tile. We offset by 1 to leave room for the halo.
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;

    // Declare a shared memory tile with an extra border (halo) of 1 cell on each side.
    __shared__ bool tile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    // Load the central cell into shared memory if within bounds; otherwise, set to false.
    if (x < grid_dim && y < grid_dim)
        tile[s_y][s_x] = input[y * grid_dim + x];
    else
        tile[s_y][s_x] = false;

    // Load halo cells along the left and right borders.
    if (threadIdx.x == 0) {
        int x_left = x - 1;
        if (x_left >= 0 && y < grid_dim)
            tile[s_y][s_x - 1] = input[y * grid_dim + x_left];
        else
            tile[s_y][s_x - 1] = false;
    }
    if (threadIdx.x == BLOCK_DIM - 1) {
        int x_right = x + 1;
        if (x_right < grid_dim && y < grid_dim)
            tile[s_y][s_x + 1] = input[y * grid_dim + x_right];
        else
            tile[s_y][s_x + 1] = false;
    }

    // Load halo cells along the top and bottom borders.
    if (threadIdx.y == 0) {
        int y_top = y - 1;
        if (y_top >= 0 && x < grid_dim)
            tile[s_y - 1][s_x] = input[y_top * grid_dim + x];
        else
            tile[s_y - 1][s_x] = false;
    }
    if (threadIdx.y == BLOCK_DIM - 1) {
        int y_bottom = y + 1;
        if (y_bottom < grid_dim && x < grid_dim)
            tile[s_y + 1][s_x] = input[y_bottom * grid_dim + x];
        else
            tile[s_y + 1][s_x] = false;
    }

    // Load halo cells at the four corners.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int x_left = x - 1;
        int y_top = y - 1;
        if (x_left >= 0 && y_top >= 0)
            tile[s_y - 1][s_x - 1] = input[y_top * grid_dim + x_left];
        else
            tile[s_y - 1][s_x - 1] = false;
    }
    if (threadIdx.x == BLOCK_DIM - 1 && threadIdx.y == 0) {
        int x_right = x + 1;
        int y_top = y - 1;
        if (x_right < grid_dim && y_top >= 0)
            tile[s_y - 1][s_x + 1] = input[y_top * grid_dim + x_right];
        else
            tile[s_y - 1][s_x + 1] = false;
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_DIM - 1) {
        int x_left = x - 1;
        int y_bottom = y + 1;
        if (x_left >= 0 && y_bottom < grid_dim)
            tile[s_y + 1][s_x - 1] = input[y_bottom * grid_dim + x_left];
        else
            tile[s_y + 1][s_x - 1] = false;
    }
    if (threadIdx.x == BLOCK_DIM - 1 && threadIdx.y == BLOCK_DIM - 1) {
        int x_right = x + 1;
        int y_bottom = y + 1;
        if (x_right < grid_dim && y_bottom < grid_dim)
            tile[s_y + 1][s_x + 1] = input[y_bottom * grid_dim + x_right];
        else
            tile[s_y + 1][s_x + 1] = false;
    }

    // Synchronize to ensure all shared memory loads complete before computing.
    __syncthreads();

    // Perform computation only if the global thread index is within the grid bounds.
    if (x < grid_dim && y < grid_dim) {
        int live_neighbors = 0;
        // Sum the states of the eight neighboring cells in shared memory.
        live_neighbors += tile[s_y - 1][s_x - 1];
        live_neighbors += tile[s_y - 1][s_x];
        live_neighbors += tile[s_y - 1][s_x + 1];
        live_neighbors += tile[s_y][s_x - 1];
        live_neighbors += tile[s_y][s_x + 1];
        live_neighbors += tile[s_y + 1][s_x - 1];
        live_neighbors += tile[s_y + 1][s_x];
        live_neighbors += tile[s_y + 1][s_x + 1];

        // Retrieve current cell state.
        bool alive = tile[s_y][s_x];
        bool new_state = false;

        // Apply Conway's Game of Life rules.
        if (alive) {
            // Any live cell with two or three live neighbors survives.
            new_state = (live_neighbors == 2 || live_neighbors == 3);
        } else {
            // Any dead cell with exactly three live neighbors becomes a live cell.
            new_state = (live_neighbors == 3);
        }

        // Write the new state to global memory.
        output[y * grid_dim + x] = new_state;
    }
}

// Host function to run one simulation step of Conway's Game of Life.
// It launches the CUDA kernel to compute the next generation.
// The input and output pointers are assumed to be device pointers allocated with cudaMalloc.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Define block and grid dimensions.
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((grid_dimensions + BLOCK_DIM - 1) / BLOCK_DIM, (grid_dimensions + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);

    // No host-device synchronization here as it is assumed to be handled externally.
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
