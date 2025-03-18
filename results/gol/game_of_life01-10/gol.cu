// Conway's Game of Life CUDA implementation.
// This code performs one simulation step of Conway's Game of Life.
// Each cell is represented as a bool (true = alive, false = dead).
// The grid is a square of size grid_dimensions x grid_dimensions.
// The grid is stored in device memory (allocated via cudaMalloc).
// The kernel uses shared memory for a tile plus a 1-cell halo to reduce global memory accesses.

#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // Block dimensions (32x32 threads per block).

// CUDA kernel to compute one generation of Conway's Game of Life.
// Each thread processes one cell. The kernel loads a tile of cells (plus halo)
// into shared memory to accelerate neighbor access.
__global__ void game_of_life_kernel(const bool* __restrict__ input, bool* __restrict__ output, int grid_dim) {
    // Compute global x and y coordinates for the thread.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory tile dimensions: (BLOCK_SIZE + 2) x (BLOCK_SIZE + 2).
    // The extra border (halo) holds neighbors of the tile.
    __shared__ bool tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute shared memory indices for the interior region.
    int smem_x = threadIdx.x + 1;
    int smem_y = threadIdx.y + 1;

    // Load the main (interior) cell from global memory into shared memory.
    tile[smem_y][smem_x] = (x < grid_dim && y < grid_dim) ? input[y * grid_dim + x] : false;

    // Load the halo cells from global memory.
    // Left halo: for threads at the left edge of the block.
    if (threadIdx.x == 0) {
        int left_x = x - 1;
        tile[smem_y][0] = (left_x >= 0 && y < grid_dim) ? input[y * grid_dim + left_x] : false;
    }
    // Right halo: for threads at the right edge.
    if (threadIdx.x == blockDim.x - 1) {
        int right_x = x + 1;
        tile[smem_y][BLOCK_SIZE + 1] = (right_x < grid_dim && y < grid_dim) ? input[y * grid_dim + right_x] : false;
    }
    // Top halo: for threads at the top edge.
    if (threadIdx.y == 0) {
        int top_y = y - 1;
        tile[0][smem_x] = (top_y >= 0 && x < grid_dim) ? input[top_y * grid_dim + x] : false;
    }
    // Bottom halo: for threads at the bottom edge.
    if (threadIdx.y == blockDim.y - 1) {
        int bottom_y = y + 1;
        tile[BLOCK_SIZE + 1][smem_x] = (bottom_y < grid_dim && x < grid_dim) ? input[bottom_y * grid_dim + x] : false;
    }
    // Top-left corner.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int left_x = x - 1;
        int top_y = y - 1;
        tile[0][0] = (left_x >= 0 && top_y >= 0) ? input[top_y * grid_dim + left_x] : false;
    }
    // Top-right corner.
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int right_x = x + 1;
        int top_y = y - 1;
        tile[0][BLOCK_SIZE + 1] = (right_x < grid_dim && top_y >= 0) ? input[top_y * grid_dim + right_x] : false;
    }
    // Bottom-left corner.
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int left_x = x - 1;
        int bottom_y = y + 1;
        tile[BLOCK_SIZE + 1][0] = (left_x >= 0 && bottom_y < grid_dim) ? input[bottom_y * grid_dim + left_x] : false;
    }
    // Bottom-right corner.
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int right_x = x + 1;
        int bottom_y = y + 1;
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = (right_x < grid_dim && bottom_y < grid_dim) ? input[bottom_y * grid_dim + right_x] : false;
    }

    // Synchronize to ensure all shared memory loads are completed.
    __syncthreads();

    // Only perform computation if the thread corresponds to a valid grid cell.
    if (x < grid_dim && y < grid_dim) {
        int live_neighbors = 0;
        // Count live neighbors using the shared memory tile.
        live_neighbors += tile[smem_y - 1][smem_x - 1] ? 1 : 0;
        live_neighbors += tile[smem_y - 1][smem_x    ] ? 1 : 0;
        live_neighbors += tile[smem_y - 1][smem_x + 1] ? 1 : 0;
        live_neighbors += tile[smem_y    ][smem_x - 1] ? 1 : 0;
        live_neighbors += tile[smem_y    ][smem_x + 1] ? 1 : 0;
        live_neighbors += tile[smem_y + 1][smem_x - 1] ? 1 : 0;
        live_neighbors += tile[smem_y + 1][smem_x    ] ? 1 : 0;
        live_neighbors += tile[smem_y + 1][smem_x + 1] ? 1 : 0;

        // Apply Conway's Game of Life rules.
        bool current_cell = tile[smem_y][smem_x];
        bool next_state;
        if (current_cell) {
            // Alive cell: survives if it has 2 or 3 live neighbors.
            next_state = (live_neighbors == 2 || live_neighbors == 3);
        } else {
            // Dead cell: becomes alive if it has exactly 3 live neighbors.
            next_state = (live_neighbors == 3);
        }
        // Write the computed next state to the output grid.
        output[y * grid_dim + x] = next_state;
    }
}

// Host function to execute one step of Conway's Game of Life.
// It configures and launches the CUDA kernel.
// The input and output pointers refer to device memory.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Configure CUDA thread block dimensions.
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Calculate grid dimensions to cover the entire board.
    dim3 grid((grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel. Shared memory is statically allocated.
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
