#include <cuda_runtime.h>

// Define tile dimensions for shared memory tiling.
// We choose a 16x16 thread block size for high occupancy and efficient shared memory use.
// The shared memory tile is (BLOCK_SIZE+2)x(BLOCK_SIZE+2) to include a 1-cell halo in every direction.
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel that computes one generation of Conway's Game of Life.
// Each thread loads its corresponding cell and its halo neighbors into shared memory,
// then computes the number of live neighbors and updates its cell state according to the rules.
__global__ void game_of_life_kernel(const bool* input, bool* output, int grid_dim)
{
    // Allocate shared memory tile.
    // The array dimensions include a 1-cell halo on every side.
    __shared__ bool s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];

    // Thread coordinates within the block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global row and column indices.
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_HEIGHT + ty;

    // Load the interior cell from global memory into shared memory.
    // If the thread is within the grid, read the value; otherwise, treat it as dead.
    bool cell = false;
    if (row < grid_dim && col < grid_dim)
    {
        cell = input[row * grid_dim + col];
    }
    s_tile[ty + 1][tx + 1] = cell;

    // Load halo cells from global memory.
    // For cells outside the grid boundaries, we set the value to false (dead).

    // Left halo: thread at left border of block loads cell to the left.
    if (tx == 0)
    {
        int left_col = col - 1;
        bool cell_left = false;
        if (left_col >= 0 && row < grid_dim)
            cell_left = input[row * grid_dim + left_col];
        s_tile[ty + 1][0] = cell_left;
    }
    // Right halo: thread at right border of block loads cell to the right.
    if (tx == TILE_WIDTH - 1)
    {
        int right_col = col + 1;
        bool cell_right = false;
        if (right_col < grid_dim && row < grid_dim)
            cell_right = input[row * grid_dim + right_col];
        s_tile[ty + 1][TILE_WIDTH + 1] = cell_right;
    }
    // Top halo: thread at top border of block loads cell above.
    if (ty == 0)
    {
        int top_row = row - 1;
        bool cell_top = false;
        if (top_row >= 0 && col < grid_dim)
            cell_top = input[top_row * grid_dim + col];
        s_tile[0][tx + 1] = cell_top;
    }
    // Bottom halo: thread at bottom border of block loads cell below.
    if (ty == TILE_HEIGHT - 1)
    {
        int bottom_row = row + 1;
        bool cell_bottom = false;
        if (bottom_row < grid_dim && col < grid_dim)
            cell_bottom = input[bottom_row * grid_dim + col];
        s_tile[TILE_HEIGHT + 1][tx + 1] = cell_bottom;
    }
    // Top-left corner: thread at top-left of block loads top-left halo.
    if (tx == 0 && ty == 0)
    {
        int top_row = row - 1;
        int left_col = col - 1;
        bool cell_top_left = false;
        if (top_row >= 0 && left_col >= 0)
            cell_top_left = input[top_row * grid_dim + left_col];
        s_tile[0][0] = cell_top_left;
    }
    // Top-right corner: thread at top-right of block loads top-right halo.
    if (tx == TILE_WIDTH - 1 && ty == 0)
    {
        int top_row = row - 1;
        int right_col = col + 1;
        bool cell_top_right = false;
        if (top_row >= 0 && right_col < grid_dim)
            cell_top_right = input[top_row * grid_dim + right_col];
        s_tile[0][TILE_WIDTH + 1] = cell_top_right;
    }
    // Bottom-left corner: thread at bottom-left of block loads bottom-left halo.
    if (tx == 0 && ty == TILE_HEIGHT - 1)
    {
        int bottom_row = row + 1;
        int left_col = col - 1;
        bool cell_bottom_left = false;
        if (bottom_row < grid_dim && left_col >= 0)
            cell_bottom_left = input[bottom_row * grid_dim + left_col];
        s_tile[TILE_HEIGHT + 1][0] = cell_bottom_left;
    }
    // Bottom-right corner: thread at bottom-right of block loads bottom-right halo.
    if (tx == TILE_WIDTH - 1 && ty == TILE_HEIGHT - 1)
    {
        int bottom_row = row + 1;
        int right_col = col + 1;
        bool cell_bottom_right = false;
        if (bottom_row < grid_dim && right_col < grid_dim)
            cell_bottom_right = input[bottom_row * grid_dim + right_col];
        s_tile[TILE_HEIGHT + 1][TILE_WIDTH + 1] = cell_bottom_right;
    }

    // Ensure all shared memory loads are complete before computation.
    __syncthreads();

    // Compute the number of live (true) neighbors.
    // The cell itself is at s_tile[ty+1][tx+1], so sum the other surrounding 8 cells.
    int live_neighbors = 0;
    live_neighbors += s_tile[ty    ][tx    ] ? 1 : 0; // top-left
    live_neighbors += s_tile[ty    ][tx + 1] ? 1 : 0; // top
    live_neighbors += s_tile[ty    ][tx + 2] ? 1 : 0; // top-right
    live_neighbors += s_tile[ty + 1][tx    ] ? 1 : 0; // left
    live_neighbors += s_tile[ty + 1][tx + 2] ? 1 : 0; // right
    live_neighbors += s_tile[ty + 2][tx    ] ? 1 : 0; // bottom-left
    live_neighbors += s_tile[ty + 2][tx + 1] ? 1 : 0; // bottom
    live_neighbors += s_tile[ty + 2][tx + 2] ? 1 : 0; // bottom-right

    // Apply the rules of Conway's Game of Life.
    bool current_state = s_tile[ty + 1][tx + 1];
    bool new_state = false;
    // Any live cell with 2 or 3 live neighbors survives; dead cell with exactly 3 live neighbors becomes live.
    if (current_state)
    {
        new_state = (live_neighbors == 2 || live_neighbors == 3);
    }
    else
    {
        new_state = (live_neighbors == 3);
    }

    // Write the computed state to the output grid.
    if (row < grid_dim && col < grid_dim)
    {
        output[row * grid_dim + col] = new_state;
    }
}

// Host function that executes one step of Conway's Game of Life.
// It configures the kernel launch parameters based on grid dimensions and launches the kernel.
// The input and output grids are assumed to be allocated in GPU memory (via cudaMalloc).
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Define CUDA block size (threads per block).
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    // Compute grid dimensions in terms of blocks.
    // As grid_dimensions is a power of 2 and the block size divides it evenly, no extra boundary check is needed.
    dim3 grid(grid_dimensions / TILE_WIDTH, grid_dimensions / TILE_HEIGHT);

    // Launch the kernel.
    // We assume that host-device synchronization is handled externally.
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
