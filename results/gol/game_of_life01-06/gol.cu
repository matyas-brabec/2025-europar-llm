// This CUDA implementation of Conway's Game of Life uses shared memory tiling
// for high performance on modern NVIDIA GPUs (H100/A100). Each thread block
// loads a TILE_DIM x TILE_DIM portion of the grid along with a 1-cell-wide halo
// region into shared memory. This minimizes global memory accesses during the
// neighbor computations for each cell. The grid outside the simulation domain
// is assumed to be permanently dead (false).
//
// The grid is square with dimensions that are a power of 2 (and at least 512).
// The block size is fixed at TILE_DIM x TILE_DIM (32x32), so the number of blocks 
// in each dimension is grid_dimensions / TILE_DIM. This choice yields good performance
// on modern GPUs.
//
// The kernel loads its own cell (center) and the appropriate halo cells.
// It then computes the sum of alive neighbor cells (from the shared memory tile)
// and applies Conway's Game of Life rules:
//   - An alive cell survives if it has 2 or 3 alive neighbors; otherwise it dies.
//   - A dead cell becomes alive if it has exactly 3 alive neighbors.
//
// The run_game_of_life function launches the kernel for one simulation step.
// All input and output pointers refer to device memory allocated via cudaMalloc.
// Host-device synchronization is assumed to be handled externally.
 
#include <cuda_runtime.h>
 
// Define the tile dimension (block width/height).
#define TILE_DIM 32
 
// CUDA kernel for one simulation step of Conway's Game of Life.
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int grid_dim) {
    // Allocate shared memory tile with halo (dimensions: (TILE_DIM+2) x (TILE_DIM+2)).
    __shared__ bool tile[TILE_DIM + 2][TILE_DIM + 2];
 
    // Compute global indices for the current cell.
    int global_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int global_y = blockIdx.y * TILE_DIM + threadIdx.y;
 
    // Compute indices in the shared memory tile.
    // The cell that corresponds to (global_x, global_y) is loaded into tile[threadIdx.y+1][threadIdx.x+1].
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
 
    // Load the center cell.
    tile[ly][lx] = (global_x < grid_dim && global_y < grid_dim) ?
                   input[global_y * grid_dim + global_x] : false;
 
    // Load halo elements along the borders.
    // Top halo row.
    if (threadIdx.y == 0) {
        int gy = global_y - 1;
        tile[0][lx] = (gy >= 0 && global_x < grid_dim) ?
                      input[gy * grid_dim + global_x] : false;
    }
    // Bottom halo row.
    if (threadIdx.y == TILE_DIM - 1) {
        int gy = global_y + 1;
        tile[TILE_DIM + 1][lx] = (gy < grid_dim && global_x < grid_dim) ?
                                 input[gy * grid_dim + global_x] : false;
    }
    // Left halo column.
    if (threadIdx.x == 0) {
        int gx = global_x - 1;
        tile[ly][0] = (gx >= 0 && global_y < grid_dim) ?
                      input[global_y * grid_dim + gx] : false;
    }
    // Right halo column.
    if (threadIdx.x == TILE_DIM - 1) {
        int gx = global_x + 1;
        tile[ly][TILE_DIM + 1] = (gx < grid_dim && global_y < grid_dim) ?
                                 input[global_y * grid_dim + gx] : false;
    }
    // Load corner elements.
    if (threadIdx.x == 0 && threadIdx.y == 0) {  // Top-left corner.
        int gx = global_x - 1;
        int gy = global_y - 1;
        tile[0][0] = (gx >= 0 && gy >= 0) ?
                     input[gy * grid_dim + gx] : false;
    }
    if (threadIdx.x == TILE_DIM - 1 && threadIdx.y == 0) {  // Top-right corner.
        int gx = global_x + 1;
        int gy = global_y - 1;
        tile[0][TILE_DIM + 1] = (gx < grid_dim && gy >= 0) ?
                                input[gy * grid_dim + gx] : false;
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE_DIM - 1) {  // Bottom-left corner.
        int gx = global_x - 1;
        int gy = global_y + 1;
        tile[TILE_DIM + 1][0] = (gx >= 0 && gy < grid_dim) ?
                                input[gy * grid_dim + gx] : false;
    }
    if (threadIdx.x == TILE_DIM - 1 && threadIdx.y == TILE_DIM - 1) {  // Bottom-right corner.
        int gx = global_x + 1;
        int gy = global_y + 1;
        tile[TILE_DIM + 1][TILE_DIM + 1] = (gx < grid_dim && gy < grid_dim) ?
                                           input[gy * grid_dim + gx] : false;
    }
 
    // Ensure that all threads have loaded their shared memory cells.
    __syncthreads();
 
    // Proceed only if the global index is within the grid.
    if (global_x < grid_dim && global_y < grid_dim) {
        // Compute the sum of the 8 neighbors using the shared memory tile.
        // The center cell is at tile[ly][lx].
        int neighbor_count = 0;
        neighbor_count += tile[ly - 1][lx - 1]; // Top-left.
        neighbor_count += tile[ly - 1][lx];     // Top.
        neighbor_count += tile[ly - 1][lx + 1]; // Top-right.
        neighbor_count += tile[ly][lx - 1];       // Left.
        neighbor_count += tile[ly][lx + 1];       // Right.
        neighbor_count += tile[ly + 1][lx - 1];   // Bottom-left.
        neighbor_count += tile[ly + 1][lx];       // Bottom.
        neighbor_count += tile[ly + 1][lx + 1];   // Bottom-right.
 
        // Apply Conway's Game of Life rules.
        bool alive = tile[ly][lx];
        bool next_state;
        if (alive) {
            // Alive cell survives with 2 or 3 neighbors.
            next_state = (neighbor_count == 2 || neighbor_count == 3);
        } else {
            // Dead cell becomes alive with exactly 3 neighbors.
            next_state = (neighbor_count == 3);
        }
 
        // Write the computed next state to the output grid.
        output[global_y * grid_dim + global_x] = next_state;
    }
}
 
// Host function that launches one simulation step of Conway's Game of Life.
// The input and output parameters point to device memory allocated by cudaMalloc.
// grid_dimensions is the width (and height) of the square grid.
/// @FIXED: extern "C" void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Define the block dimensions as TILE_DIM x TILE_DIM.
    dim3 block(TILE_DIM, TILE_DIM);
    // Calculate grid dimensions such that each block covers a TILE_DIM x TILE_DIM region.
    dim3 grid(grid_dimensions / TILE_DIM, grid_dimensions / TILE_DIM);
 
    // Launch the kernel. Note: The kernel call is asynchronous.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // Host-device synchronization is assumed to be handled by the caller.
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
