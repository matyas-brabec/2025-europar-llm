#include <cuda_runtime.h>

// CUDA kernel for one Game of Life step, using shared memory tiling for performance.
// Each thread loads its own cell as well as the appropriate halo cells into shared memory,
// then computes the number of live neighbors and applies the Game of Life rules.
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int grid_dim) {
    // Calculate global coordinates of the cell to process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Define shared memory tile dimensions.
    // The shared memory tile covers a (blockDim.x + 2) x (blockDim.y + 2) region.
    extern __shared__ bool tile[];
    const int tile_width = blockDim.x + 2;

    // Compute corresponding local coordinates inside the shared memory tile.
    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // Load the central cell into shared memory. Use false (dead) for out-of-bound cells.
    bool center = false;
    if (x < grid_dim && y < grid_dim)
        center = input[y * grid_dim + x];
    tile[local_y * tile_width + local_x] = center;

    // Load the left halo cell.
    if (threadIdx.x == 0) {
        bool value = false;
        int global_x = x - 1;
        if (global_x >= 0 && y < grid_dim)
            value = input[y * grid_dim + global_x];
        tile[local_y * tile_width + (local_x - 1)] = value;
    }
    // Load the right halo cell.
    if (threadIdx.x == blockDim.x - 1) {
        bool value = false;
        int global_x = x + 1;
        if (global_x < grid_dim && y < grid_dim)
            value = input[y * grid_dim + global_x];
        tile[local_y * tile_width + (local_x + 1)] = value;
    }
    // Load the top halo cell.
    if (threadIdx.y == 0) {
        bool value = false;
        int global_y = y - 1;
        if (global_y >= 0 && x < grid_dim)
            value = input[global_y * grid_dim + x];
        tile[(local_y - 1) * tile_width + local_x] = value;
    }
    // Load the bottom halo cell.
    if (threadIdx.y == blockDim.y - 1) {
        bool value = false;
        int global_y = y + 1;
        if (global_y < grid_dim && x < grid_dim)
            value = input[global_y * grid_dim + x];
        tile[(local_y + 1) * tile_width + local_x] = value;
    }
    // Load the top-left corner.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        bool value = false;
        int global_x = x - 1;
        int global_y = y - 1;
        if (global_x >= 0 && global_y >= 0)
            value = input[global_y * grid_dim + global_x];
        tile[(local_y - 1) * tile_width + (local_x - 1)] = value;
    }
    // Load the top-right corner.
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        bool value = false;
        int global_x = x + 1;
        int global_y = y - 1;
        if (global_x < grid_dim && global_y >= 0)
            value = input[global_y * grid_dim + global_x];
        tile[(local_y - 1) * tile_width + (local_x + 1)] = value;
    }
    // Load the bottom-left corner.
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        bool value = false;
        int global_x = x - 1;
        int global_y = y + 1;
        if (global_x >= 0 && global_y < grid_dim)
            value = input[global_y * grid_dim + global_x];
        tile[(local_y + 1) * tile_width + (local_x - 1)] = value;
    }
    // Load the bottom-right corner.
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        bool value = false;
        int global_x = x + 1;
        int global_y = y + 1;
        if (global_x < grid_dim && global_y < grid_dim)
            value = input[global_y * grid_dim + global_x];
        tile[(local_y + 1) * tile_width + (local_x + 1)] = value;
    }

    // Synchronize to ensure all shared memory loads complete.
    __syncthreads();

    // If the cell is out of the grid bounds, return without computation.
    if (x >= grid_dim || y >= grid_dim)
        return;

    // Count the live neighbors from the shared memory tile.
    int live_neighbors = 0;
    live_neighbors += tile[(local_y - 1) * tile_width + (local_x - 1)];
    live_neighbors += tile[(local_y - 1) * tile_width + (local_x    )];
    live_neighbors += tile[(local_y - 1) * tile_width + (local_x + 1)];
    live_neighbors += tile[(local_y    ) * tile_width + (local_x - 1)];
    live_neighbors += tile[(local_y    ) * tile_width + (local_x + 1)];
    live_neighbors += tile[(local_y + 1) * tile_width + (local_x - 1)];
    live_neighbors += tile[(local_y + 1) * tile_width + (local_x    )];
    live_neighbors += tile[(local_y + 1) * tile_width + (local_x + 1)];

    // Apply Conway's Game of Life rules.
    bool current = tile[local_y * tile_width + local_x];
    bool next_state = (current && (live_neighbors == 2 || live_neighbors == 3)) ||
                      (!current && live_neighbors == 3);

    // Write the computed state to the output grid.
    output[y * grid_dim + x] = next_state;
}

// Host function to perform one step of Conway's Game of Life on the GPU.
// It configures the kernel launch parameters and calls the kernel.
// The input and output grids must be allocated on the device (e.g., via cudaMalloc).
// Any required host-device synchronization is assumed to be handled by the caller.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Choose a block size tuned for modern GPUs (e.g., 32x32 threads per block).
    dim3 blockDim(32, 32);
    // Compute the grid size in blocks, rounding up as needed.
    dim3 gridDim((grid_dimensions + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Compute the amount of dynamic shared memory needed:
    // Each block uses a tile of size (blockDim.x + 2) x (blockDim.y + 2).
    size_t sharedMemSize = (blockDim.x + 2) * (blockDim.y + 2) * sizeof(bool);

    // Launch the kernel. The kernel execution is asynchronous.
    game_of_life_kernel<<<gridDim, blockDim, sharedMemSize>>>(input, output, grid_dimensions);
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
