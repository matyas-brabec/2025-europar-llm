// CUDA implementation of a single-step Conway's Game of Life update.
// 
// - The grid is a square of size N x N, where N is a power of 2 and > 512.
// - Cells are stored as bools in device memory, with true = alive, false = dead.
// - All cells outside the grid are treated as dead (zero-padding boundary conditions).
// - This implementation focuses on performance on modern NVIDIA data-center GPUs
//   (A100, H100) and uses a shared-memory tiled kernel to minimize global memory
//   traffic and to provide coalesced memory accesses.
//
// API:
//   void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
//
// The input and output pointers are assumed to be device pointers allocated with cudaMalloc.
// The function executes one Game of Life step entirely on the GPU. Any host-side
// synchronization is the caller's responsibility.

#include <cuda_runtime.h>

// Tile dimensions for the kernel. These are chosen for:
// - blockDim.x = 32: perfect warp alignment and fully coalesced row loads
// - blockDim.y = 32: 1024 threads per block (max), good occupancy on A100/H100
// Shared memory per block is very small (34x34 bytes).
static constexpr int BLOCK_DIM_X = 32;
static constexpr int BLOCK_DIM_Y = 32;

// Kernel implementing a single Game of Life step.
// 
// We work on an unsigned char representation (0 or 1) instead of bool in the kernel,
// because:
// - unsigned char is guaranteed to be 1 byte and behaves well with arithmetic.
// - We can safely alias the original bool arrays as unsigned char arrays on the host.
//
// The kernel uses a 2D shared-memory tile of size (BLOCK_DIM_Y+2) x (BLOCK_DIM_X+2),
// where the extra 1-cell border in each direction holds the halo cells necessary
// to compute neighbor sums for the interior BLOCK_DIM_Y x BLOCK_DIM_X tile.
__global__ void game_of_life_kernel(const unsigned char* __restrict__ input,
                                    unsigned char* __restrict__ output,
                                    int N)
{
    // Shared memory tile including 1-cell halo on all sides.
    // tile[1..BLOCK_DIM_Y][1..BLOCK_DIM_X]      -> interior cells for this block
    // tile[0][*], tile[BLOCK_DIM_Y+1][*]       -> top/bottom halo rows
    // tile[*][0], tile[*][BLOCK_DIM_X+1]       -> left/right halo columns
    // Corners tile[0][0], tile[0][BLOCK_DIM_X+1], tile[BLOCK_DIM_Y+1][0],
    //         tile[BLOCK_DIM_Y+1][BLOCK_DIM_X+1] -> halo corners.
    __shared__ unsigned char tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * BLOCK_DIM_X + tx;  // global x index
    const int gy = blockIdx.y * BLOCK_DIM_Y + ty;  // global y index

    const int local_x = tx + 1;  // local tile x index (offset by 1 for halo)
    const int local_y = ty + 1;  // local tile y index (offset by 1 for halo)

    // Convenience flags for bounds checks.
    const bool in_bounds_x = (gx < N);
    const bool in_bounds_y = (gy < N);
    const bool in_bounds   = in_bounds_x && in_bounds_y;

    // Load the cell corresponding to this thread into the central part of the tile.
    unsigned char center = 0;
    if (in_bounds) {
        center = input[gy * N + gx];
    }
    tile[local_y][local_x] = center;

    // Load left halo cell for this row (gx-1, gy) into tile[local_y][0].
    // Only one thread per row (tx == 0) performs this load.
    if (tx == 0) {
        const int ngx = gx - 1;
        unsigned char val = 0;
        if (ngx >= 0 && in_bounds_y) {
            val = input[gy * N + ngx];
        }
        tile[local_y][0] = val;
    }

    // Load right halo cell for this row (gx+1, gy) into tile[local_y][BLOCK_DIM_X+1].
    if (tx == BLOCK_DIM_X - 1) {
        const int ngx = gx + 1;
        unsigned char val = 0;
        if (ngx < N && in_bounds_y) {
            val = input[gy * N + ngx];
        }
        tile[local_y][BLOCK_DIM_X + 1] = val;
    }

    // Load top halo cell for this column (gx, gy-1) into tile[0][local_x].
    if (ty == 0) {
        const int ngy = gy - 1;
        unsigned char val = 0;
        if (ngy >= 0 && in_bounds_x) {
            val = input[ngy * N + gx];
        }
        tile[0][local_x] = val;
    }

    // Load bottom halo cell for this column (gx, gy+1) into tile[BLOCK_DIM_Y+1][local_x].
    if (ty == BLOCK_DIM_Y - 1) {
        const int ngy = gy + 1;
        unsigned char val = 0;
        if (ngy < N && in_bounds_x) {
            val = input[ngy * N + gx];
        }
        tile[BLOCK_DIM_Y + 1][local_x] = val;
    }

    // Load top-left halo corner (gx-1, gy-1) into tile[0][0].
    if (tx == 0 && ty == 0) {
        const int ngx = gx - 1;
        const int ngy = gy - 1;
        unsigned char val = 0;
        if (ngx >= 0 && ngy >= 0) {
            val = input[ngy * N + ngx];
        }
        tile[0][0] = val;
    }

    // Load top-right halo corner (gx+1, gy-1) into tile[0][BLOCK_DIM_X+1].
    if (tx == BLOCK_DIM_X - 1 && ty == 0) {
        const int ngx = gx + 1;
        const int ngy = gy - 1;
        unsigned char val = 0;
        if (ngx < N && ngy >= 0) {
            val = input[ngy * N + ngx];
        }
        tile[0][BLOCK_DIM_X + 1] = val;
    }

    // Load bottom-left halo corner (gx-1, gy+1) into tile[BLOCK_DIM_Y+1][0].
    if (tx == 0 && ty == BLOCK_DIM_Y - 1) {
        const int ngx = gx - 1;
        const int ngy = gy + 1;
        unsigned char val = 0;
        if (ngx >= 0 && ngy < N) {
            val = input[ngy * N + ngx];
        }
        tile[BLOCK_DIM_Y + 1][0] = val;
    }

    // Load bottom-right halo corner (gx+1, gy+1) into tile[BLOCK_DIM_Y+1][BLOCK_DIM_X+1].
    if (tx == BLOCK_DIM_X - 1 && ty == BLOCK_DIM_Y - 1) {
        const int ngx = gx + 1;
        const int ngy = gy + 1;
        unsigned char val = 0;
        if (ngx < N && ngy < N) {
            val = input[ngy * N + ngx];
        }
        tile[BLOCK_DIM_Y + 1][BLOCK_DIM_X + 1] = val;
    }

    // Ensure all shared memory loads are visible to all threads in the block.
    __syncthreads();

    // Threads that map outside the grid do nothing further (no write).
    if (!in_bounds) {
        return;
    }

    // Compute the sum of the 8 neighbors from the shared memory tile.
    // Access pattern:
    //   (local_x, local_y) is the current cell.
    //   We look at all 8 surrounding positions.
    const int neighbor_sum =
        tile[local_y - 1][local_x - 1] +
        tile[local_y - 1][local_x    ] +
        tile[local_y - 1][local_x + 1] +
        tile[local_y    ][local_x - 1] +
        tile[local_y    ][local_x + 1] +
        tile[local_y + 1][local_x - 1] +
        tile[local_y + 1][local_x    ] +
        tile[local_y + 1][local_x + 1];

    const unsigned char alive = tile[local_y][local_x];

    // Apply Conway's Game of Life rules:
    // - Any alive cell with fewer than 2 alive neighbors dies (underpopulation).
    // - Any alive cell with 2 or 3 alive neighbors lives on.
    // - Any alive cell with more than 3 alive neighbors dies (overpopulation).
    // - Any dead cell with exactly 3 live neighbors becomes alive (reproduction).
    unsigned char next_state = 0;
    if (alive) {
        next_state = (neighbor_sum == 2 || neighbor_sum == 3) ? 1u : 0u;
    } else {
        next_state = (neighbor_sum == 3) ? 1u : 0u;
    }

    // Write result back to global memory.
    output[gy * N + gx] = next_state;
}

// Host-side wrapper that launches the kernel for a single Game of Life step.
//
// Parameters:
//   input  - device pointer to N x N grid of bool values (true = alive, false = dead)
//   output - device pointer to N x N grid of bool values (receives next state)
//   grid_dimensions - N (width and height of the square grid)
//
// Notes:
//   - input and output must be allocated with cudaMalloc and be at least N*N*sizeof(bool) bytes.
//   - This function does not perform any synchronization; the caller is responsible for
//     CUDA error checking and synchronization as needed.
//   - For performance, we reinterpret the bool* as unsigned char* (1 byte per cell) and
//     operate in that domain in the kernel. unsigned char is allowed to alias any type.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Alias bool arrays as unsigned char arrays for efficient arithmetic.
    const unsigned char* in_u8  = reinterpret_cast<const unsigned char*>(input);
    unsigned char*       out_u8 = reinterpret_cast<unsigned char*>(output);

    // Configure a 2D grid of 32x32-thread blocks that covers the entire N x N domain.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(
        (N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );

    // Launch the kernel. No explicit cudaDeviceSynchronize() here; the caller can
    // synchronize and perform error checking if desired.
    game_of_life_kernel<<<grid, block>>>(in_u8, out_u8, N);
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
