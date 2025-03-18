// Includes necessary CUDA runtime header.
#include <cuda_runtime.h>

// Define the block size used for tiling the grid in shared memory.
// A block of 32x32 threads is chosen for optimal occupancy on modern GPUs.
#define BLOCK_SIZE 32

// CUDA kernel implementing one step of Conway's Game of Life.
// Each thread computes the next state of one cell by reading from a shared memory tile.
// Shared memory is used to reduce global memory accesses by loading a tile with a 1-cell halo border.
__global__ void game_of_life_kernel(const bool* input, bool* output, int grid_dim) {
    // Allocate shared memory tile with an extra border (halo) of 1 cell in each direction.
    // The tile size is (BLOCK_SIZE + 2) x (BLOCK_SIZE + 2).
    __shared__ bool tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute global x and y indices for the cell processed by this thread.
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Compute the indices within the shared memory tile.
    // The central cell is stored at position (threadIdx.y + 1, threadIdx.x + 1).
    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // Load the central cell from global memory into shared memory.
    // If the global index is out-of-bound, treat the cell as dead.
    if (x < grid_dim && y < grid_dim)
        tile[local_y][local_x] = input[y * grid_dim + x];
    else
        tile[local_y][local_x] = false;

    // Load halo cells from global memory into shared memory.
    // Left halo.
    if (threadIdx.x == 0) {
        int nx = x - 1;
        if (nx >= 0 && y < grid_dim)
            tile[local_y][0] = input[y * grid_dim + nx];
        else
            tile[local_y][0] = false;
    }
    // Right halo.
    if (threadIdx.x == BLOCK_SIZE - 1) {
        int nx = x + 1;
        if (nx < grid_dim && y < grid_dim)
            tile[local_y][BLOCK_SIZE + 1] = input[y * grid_dim + nx];
        else
            tile[local_y][BLOCK_SIZE + 1] = false;
    }
    // Top halo.
    if (threadIdx.y == 0) {
        int ny = y - 1;
        if (ny >= 0 && x < grid_dim)
            tile[0][local_x] = input[ny * grid_dim + x];
        else
            tile[0][local_x] = false;
    }
    // Bottom halo.
    if (threadIdx.y == BLOCK_SIZE - 1) {
        int ny = y + 1;
        if (ny < grid_dim && x < grid_dim)
            tile[BLOCK_SIZE + 1][local_x] = input[ny * grid_dim + x];
        else
            tile[BLOCK_SIZE + 1][local_x] = false;
    }
    // Top-left corner.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int nx = x - 1;
        int ny = y - 1;
        if (nx >= 0 && ny >= 0)
            tile[0][0] = input[ny * grid_dim + nx];
        else
            tile[0][0] = false;
    }
    // Top-right corner.
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        int nx = x + 1;
        int ny = y - 1;
        if (nx < grid_dim && ny >= 0)
            tile[0][BLOCK_SIZE + 1] = input[ny * grid_dim + nx];
        else
            tile[0][BLOCK_SIZE + 1] = false;
    }
    // Bottom-left corner.
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        int nx = x - 1;
        int ny = y + 1;
        if (nx >= 0 && ny < grid_dim)
            tile[BLOCK_SIZE + 1][0] = input[ny * grid_dim + nx];
        else
            tile[BLOCK_SIZE + 1][0] = false;
    }
    // Bottom-right corner.
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        int nx = x + 1;
        int ny = y + 1;
        if (nx < grid_dim && ny < grid_dim)
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = input[ny * grid_dim + nx];
        else
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = false;
    }

    // Ensure all shared memory loads complete before computation.
    __syncthreads();

    // Only compute if the thread's global indices are within grid bounds.
    if (x < grid_dim && y < grid_dim) {
        // Count alive neighbors using the shared memory tile.
        // Sum the eight neighboring cells (all cells in the 3x3 neighborhood except the center).
        int alive_neighbors = 0;
        alive_neighbors += tile[local_y - 1][local_x - 1]; // Top-left
        alive_neighbors += tile[local_y - 1][local_x];     // Top
        alive_neighbors += tile[local_y - 1][local_x + 1];   // Top-right
        alive_neighbors += tile[local_y][local_x - 1];       // Left
        alive_neighbors += tile[local_y][local_x + 1];       // Right
        alive_neighbors += tile[local_y + 1][local_x - 1];   // Bottom-left
        alive_neighbors += tile[local_y + 1][local_x];       // Bottom
        alive_neighbors += tile[local_y + 1][local_x + 1];   // Bottom-right

        // Apply Conway's Game of Life rules.
        bool current = tile[local_y][local_x];
        bool next_state = false;
        if (current) {
            // An alive cell survives if it has 2 or 3 alive neighbors.
            if (alive_neighbors == 2 || alive_neighbors == 3)
                next_state = true;
        } else {
            // A dead cell becomes alive if it has exactly 3 alive neighbors.
            if (alive_neighbors == 3)
                next_state = true;
        }

        // Write the computed state to the output grid in global memory.
        output[y * grid_dim + x] = next_state;
    }
}

// Host function to execute one step of Conway's Game of Life.
// This function sets up the kernel launch configuration and calls the kernel.
// The input and output grids are assumed to be allocated on the device using cudaMalloc.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Define block dimensions.
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Calculate grid dimensions, ensuring coverage of the entire grid.
    dim3 grid((grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch the CUDA kernel to compute the next generation.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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

