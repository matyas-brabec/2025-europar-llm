#include <cuda_runtime.h>

// CUDA kernel for one simulation step of Conway’s Game of Life.
// This kernel uses a tiling strategy with shared memory to reduce global memory traffic.
// Each thread block processes a tile of the grid, loading a (blockDim.x+2) x (blockDim.y+2) region
// into shared memory (including a one-cell halo on all sides). After synchronizing, each thread
// computes the number of live (true) neighbors from the shared memory and updates the cell state
// according to the Game of Life rules.
__global__ void game_of_life_kernel(const bool* __restrict__ input, bool* __restrict__ output, int grid_dim)
{
    // Define shared memory tile dimensions.
    // The tile has an extra border (halo) of 1 cell on each side.
    extern __shared__ bool sdata[]; // size = (blockDim.x + 2) * (blockDim.y + 2)
    const int sWidth = blockDim.x + 2;  // Shared memory row width.

    // Compute global indices for the cell processed by this thread.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + tx;
    int global_y = blockIdx.y * blockDim.y + ty;

    // Compute indices into the shared memory tile.
    // Center cell corresponding to (global_x, global_y) is stored at (ty+1, tx+1).
    int local_x = tx + 1;
    int local_y = ty + 1;
    int local_index = local_y * sWidth + local_x;

    // Load central cell into shared memory.
    bool centerVal = false;
    if (global_x < grid_dim && global_y < grid_dim)
        centerVal = input[global_y * grid_dim + global_x];
    sdata[local_index] = centerVal;

    // Load halo cells from global memory into shared memory.
    // For cells that fall outside the grid, assume the cell is dead (false).

    // Left halo 
    if (tx == 0) {
        int gx = global_x - 1; // left neighbor in global coordinates
        bool leftVal = (gx >= 0 && global_y < grid_dim) ? input[global_y * grid_dim + gx] : false;
        sdata[local_y * sWidth + 0] = leftVal;
    }
    // Right halo
    if (tx == blockDim.x - 1) {
        int gx = global_x + 1; // right neighbor in global coordinates
        bool rightVal = (gx < grid_dim && global_y < grid_dim) ? input[global_y * grid_dim + gx] : false;
        sdata[local_y * sWidth + (local_x + 1)] = rightVal;
    }
    // Top halo
    if (ty == 0) {
        int gy = global_y - 1; // top neighbor in global coordinates
        bool topVal = (gy >= 0 && global_x < grid_dim) ? input[gy * grid_dim + global_x] : false;
        sdata[0 * sWidth + local_x] = topVal;
    }
    // Bottom halo
    if (ty == blockDim.y - 1) {
        int gy = global_y + 1; // bottom neighbor in global coordinates
        bool bottomVal = (gy < grid_dim && global_x < grid_dim) ? input[gy * grid_dim + global_x] : false;
        sdata[(local_y + 1) * sWidth + local_x] = bottomVal;
    }
    // Top-left corner
    if (tx == 0 && ty == 0) {
        int gx = global_x - 1;
        int gy = global_y - 1;
        bool tlVal = (gx >= 0 && gy >= 0) ? input[gy * grid_dim + gx] : false;
        sdata[0 * sWidth + 0] = tlVal;
    }
    // Top-right corner
    if (tx == blockDim.x - 1 && ty == 0) {
        int gx = global_x + 1;
        int gy = global_y - 1;
        bool trVal = (gx < grid_dim && gy >= 0) ? input[gy * grid_dim + gx] : false;
        sdata[0 * sWidth + (local_x + 1)] = trVal;
    }
    // Bottom-left corner
    if (tx == 0 && ty == blockDim.y - 1) {
        int gx = global_x - 1;
        int gy = global_y + 1;
        bool blVal = (gx >= 0 && gy < grid_dim) ? input[gy * grid_dim + gx] : false;
        sdata[(local_y + 1) * sWidth + 0] = blVal;
    }
    // Bottom-right corner
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int gx = global_x + 1;
        int gy = global_y + 1;
        bool brVal = (gx < grid_dim && gy < grid_dim) ? input[gy * grid_dim + gx] : false;
        sdata[(local_y + 1) * sWidth + (local_x + 1)] = brVal;
    }

    // Make sure all threads have loaded their data into shared memory.
    __syncthreads();

    // Only proceed if the thread corresponds to a valid global cell.
    if (global_x < grid_dim && global_y < grid_dim)
    {
        // Count the number of live neighbors.
        // The center cell is at (local_y, local_x) in shared memory.
        int live_neighbors = 0;
        live_neighbors += (int) sdata[(local_y - 1) * sWidth + (local_x - 1)];
        live_neighbors += (int) sdata[(local_y - 1) * sWidth + (local_x    )];
        live_neighbors += (int) sdata[(local_y - 1) * sWidth + (local_x + 1)];
        live_neighbors += (int) sdata[(local_y    ) * sWidth + (local_x - 1)];
        live_neighbors += (int) sdata[(local_y    ) * sWidth + (local_x + 1)];
        live_neighbors += (int) sdata[(local_y + 1) * sWidth + (local_x - 1)];
        live_neighbors += (int) sdata[(local_y + 1) * sWidth + (local_x    )];
        live_neighbors += (int) sdata[(local_y + 1) * sWidth + (local_x + 1)];

        // Apply the Game of Life rules.
        bool current = sdata[local_index];
        bool next_state;
        if (current)
        {
            // Survival: live cell with 2 or 3 neighbors survives.
            next_state = (live_neighbors == 2 || live_neighbors == 3);
        }
        else
        {
            // Reproduction: dead cell with exactly 3 live neighbors becomes alive.
            next_state = (live_neighbors == 3);
        }
        // Write the result to global memory.
        output[global_y * grid_dim + global_x] = next_state;
    }
}

// Host function that launches one step of Conway's Game of Life using the CUDA kernel.
// - input: Pointer to the input grid on device memory (each cell is true for alive, false for dead).
// - output: Pointer to the output grid on device memory.
// - grid_dimensions: The width and height of the square grid (always a power of 2).
// The function calculates the grid and block dimensions and allocates the necessary shared memory size.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Choose block dimensions. 32x32 threads per block is a good match for modern GPUs.
    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 32;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    // Calculate grid dimensions based on total grid size.
    dim3 grid((grid_dimensions + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (grid_dimensions + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Compute the size of shared memory required per block.
    // Each block loads a tile of size (BLOCK_DIM_X + 2) x (BLOCK_DIM_Y + 2).
    size_t shared_mem_size = (BLOCK_DIM_X + 2) * (BLOCK_DIM_Y + 2) * sizeof(bool);

    // Launch the kernel. The caller is responsible for any host-device synchronization.
    game_of_life_kernel<<<grid, block, shared_mem_size>>>(input, output, grid_dimensions);
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
