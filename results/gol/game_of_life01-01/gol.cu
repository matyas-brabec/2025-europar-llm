#include <cuda_runtime.h>
// Define the block (tile) size for the kernel.
// We choose 32 because modern GPUs (A100, H100) perform well with 32x32 thread blocks.
#define BLOCK_SIZE 32

// The CUDA kernel that computes one step of Conway’s Game of Life.
// The implementation uses shared memory tiling to minimize global memory accesses.
// Each block loads a (BLOCK_SIZE x BLOCK_SIZE) tile from the global input grid
// plus a 1-cell wide halo on all sides. Each cell’s eight neighbors are then summed
// using the shared memory tile. The Game of Life rules are applied and the result
// is stored in the output grid.
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int grid_dim)
{
    // Allocate shared memory for a tile of cells, including a 1-cell halo on each side.
    // The shared tile dimensions are (BLOCK_SIZE+2) x (BLOCK_SIZE+2).
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute the global (x,y) index of the cell this thread is working on.
    int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // The local index inside the shared memory tile. Offset by 1 to account for the halo.
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Load the cell corresponding to (gx,gy) into the interior of the shared memory tile.
    // If the global index is out-of-bounds (should not happen for full blocks in a power-of-2 grid),
    // then load a dead cell (0).
    tile[ly][lx] = (gx < grid_dim && gy < grid_dim && input[gy * grid_dim + gx]) ? 1 : 0;

    // Load horizontal halo cells.
    // Left halo: load the cell immediately to the left of the current tile row.
    if (threadIdx.x == 0) {
        int gx_left = gx - 1;
        tile[ly][0] = (gx_left >= 0 && gx_left < grid_dim && gy < grid_dim && gy >= 0) ? (input[gy * grid_dim + gx_left] ? 1 : 0) : 0;
    }
    // Right halo: load the cell immediately to the right.
    if (threadIdx.x == BLOCK_SIZE - 1) {
        int gx_right = gx + 1;
        tile[ly][BLOCK_SIZE + 1] = (gx_right < grid_dim && gx_right >= 0 && gy < grid_dim && gy >= 0) ? (input[gy * grid_dim + gx_right] ? 1 : 0) : 0;
    }

    // Load vertical halo cells.
    // Top halo: load the cell immediately above.
    if (threadIdx.y == 0) {
        int gy_top = gy - 1;
        tile[0][lx] = (gy_top >= 0 && gy_top < grid_dim && gx < grid_dim && gx >= 0) ? (input[gy_top * grid_dim + gx] ? 1 : 0) : 0;
    }
    // Bottom halo: load the cell immediately below.
    if (threadIdx.y == BLOCK_SIZE - 1) {
        int gy_bottom = gy + 1;
        tile[BLOCK_SIZE + 1][lx] = (gy_bottom < grid_dim && gy_bottom >= 0 && gx < grid_dim && gx >= 0) ? (input[gy_bottom * grid_dim + gx] ? 1 : 0) : 0;
    }

    // Load the four corner halo cells.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int gx_tl = gx - 1;
        int gy_tl = gy - 1;
        tile[0][0] = (gx_tl >= 0 && gy_tl >= 0 && gx_tl < grid_dim && gy_tl < grid_dim) ? (input[gy_tl * grid_dim + gx_tl] ? 1 : 0) : 0;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        int gx_tr = gx + 1;
        int gy_tr = gy - 1;
        tile[0][BLOCK_SIZE + 1] = (gx_tr < grid_dim && gy_tr >= 0 && gx_tr >= 0 && gy_tr < grid_dim) ? (input[gy_tr * grid_dim + gx_tr] ? 1 : 0) : 0;
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        int gx_bl = gx - 1;
        int gy_bl = gy + 1;
        tile[BLOCK_SIZE + 1][0] = (gx_bl >= 0 && gy_bl < grid_dim && gy_bl >= 0 && gx_bl < grid_dim) ? (input[gy_bl * grid_dim + gx_bl] ? 1 : 0) : 0;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        int gx_br = gx + 1;
        int gy_br = gy + 1;
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = (gx_br < grid_dim && gy_br < grid_dim) ? (input[gy_br * grid_dim + gx_br] ? 1 : 0) : 0;
    }

    // Ensure all threads have loaded their part of the tile (including halo) before computing.
    __syncthreads();

    // Compute the sum of the eight neighbors around the cell at (lx, ly) using the shared tile.
    int neighbors = tile[ly - 1][lx - 1] + tile[ly - 1][lx] + tile[ly - 1][lx + 1] +
                    tile[ly][lx - 1]                   +          tile[ly][lx + 1] +
                    tile[ly + 1][lx - 1] + tile[ly + 1][lx] + tile[ly + 1][lx + 1];

    // Apply Conway's Game of Life rules.
    // If the cell is alive (non-zero) it survives with 2 or 3 live neighbors;
    // otherwise it dies. If the cell is dead, it becomes alive only with exactly 3 live neighbors.
    bool current = tile[ly][lx] != 0;
    bool next_state = false;
    if (current) {
        next_state = (neighbors == 2 || neighbors == 3);
    } else {
        next_state = (neighbors == 3);
    }

    // Write the computed next state to the global output grid.
    if (gx < grid_dim && gy < grid_dim) {
        output[gy * grid_dim + gx] = next_state;
    }
}

// Host function that runs one iteration (step) of Conway’s Game of Life.
// It sets up the CUDA kernel launch parameters and calls the game_of_life_kernel.
// The input and output pointers are assumed to be allocated in GPU memory (via cudaMalloc).
// This function only launches the kernel; any necessary host-device synchronization
// is assumed to be handled by the caller.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Calculate grid and block dimensions.
    // We use BLOCK_SIZE x BLOCK_SIZE threads per block.
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);

    // Note: Error checking and synchronization (if required) should be handled by the caller.
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
