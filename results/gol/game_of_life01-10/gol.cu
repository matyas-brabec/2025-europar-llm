// Conway's Game of Life - CUDA implementation optimized for modern NVIDIA GPUs (A100/H100).
//
// - Grid is square: N x N, where N is a power of 2, N > 512.
// - Cells are stored as bool (1 byte per cell) in row-major order.
// - Cells outside the grid are treated as dead (0).
// - This implementation uses a 2D block of 32x32 threads and a shared-memory tile
//   with a 1-cell halo (34x34 bytes). The grid size is assumed to be a multiple
//   of 32 due to N being a power of 2 >= 1024.
// - Each thread computes one cell's next state using data from shared memory.
//
// Interface:
//   void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
//
// Both `input` and `output` are device pointers allocated via cudaMalloc.

#include <cuda_runtime.h>
#include <stdint.h>

// Tile/block dimension. 32 is a good choice for modern NVIDIA GPUs:
// - It matches the warp size along one dimension for fully coalesced loads.
// - N is guaranteed to be a power of 2 >= 1024, so N % 32 == 0.
constexpr int BLOCK_DIM = 32;

// CUDA kernel for one step of Conway's Game of Life.
// Each block processes a BLOCK_DIM x BLOCK_DIM tile of the grid.
// Shared memory holds an expanded tile of size (BLOCK_DIM + 2) x (BLOCK_DIM + 2)
// to include a one-cell halo on all sides.
template <int BDIM>
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int N)
{
    // Shared-memory tile with a 1-cell halo around the block's cells.
    // Indexing: tile[local_y][local_x], where local indices go from 0..BDIM+1
    // and the "real" cells are at 1..BDIM along each dimension.
    __shared__ uint8_t tile[BDIM + 2][BDIM + 2];

    // Global coordinates of the cell handled by this thread.
    const int gx = blockIdx.x * BDIM + threadIdx.x;
    const int gy = blockIdx.y * BDIM + threadIdx.y;

    // Local coordinates in the shared-memory tile (shifted by +1 for halo).
    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;

    // ------------------------------------------------------------------------
    // Load the central block cells into shared memory.
    // Due to the assumption N % BDIM == 0, gx and gy are always valid indices.
    // ------------------------------------------------------------------------
    const int idx = gy * N + gx;
    tile[sy][sx] = static_cast<uint8_t>(input[idx]);

    // ------------------------------------------------------------------------
    // Load halo cells:
    // - Left/right columns if thread is at x==0 or x==BDIM-1.
    // - Top/bottom rows if thread is at y==0 or y==BDIM-1.
    // - Four corner halo cells if thread is at a block corner.
    //
    // Cells outside the grid are treated as 0 (dead).
    // ------------------------------------------------------------------------

    // Left halo (sx = 0)
    if (threadIdx.x == 0) {
        const int nx = gx - 1;
        uint8_t val = 0;
        if (nx >= 0) {
            val = static_cast<uint8_t>(input[gy * N + nx]);
        }
        tile[sy][0] = val;
    }

    // Right halo (sx = BDIM + 1)
    if (threadIdx.x == BDIM - 1) {
        const int nx = gx + 1;
        uint8_t val = 0;
        if (nx < N) {
            val = static_cast<uint8_t>(input[gy * N + nx]);
        }
        tile[sy][BDIM + 1] = val;
    }

    // Top halo (sy = 0)
    if (threadIdx.y == 0) {
        const int ny = gy - 1;
        uint8_t val = 0;
        if (ny >= 0) {
            val = static_cast<uint8_t>(input[ny * N + gx]);
        }
        tile[0][sx] = val;
    }

    // Bottom halo (sy = BDIM + 1)
    if (threadIdx.y == BDIM - 1) {
        const int ny = gy + 1;
        uint8_t val = 0;
        if (ny < N) {
            val = static_cast<uint8_t>(input[ny * N + gx]);
        }
        tile[BDIM + 1][sx] = val;
    }

    // Top-left corner halo (0,0)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const int nx = gx - 1;
        const int ny = gy - 1;
        uint8_t val = 0;
        if (nx >= 0 && ny >= 0) {
            val = static_cast<uint8_t>(input[ny * N + nx]);
        }
        tile[0][0] = val;
    }

    // Top-right corner halo (0, BDIM + 1)
    if (threadIdx.x == BDIM - 1 && threadIdx.y == 0) {
        const int nx = gx + 1;
        const int ny = gy - 1;
        uint8_t val = 0;
        if (nx < N && ny >= 0) {
            val = static_cast<uint8_t>(input[ny * N + nx]);
        }
        tile[0][BDIM + 1] = val;
    }

    // Bottom-left corner halo (BDIM + 1, 0)
    if (threadIdx.x == 0 && threadIdx.y == BDIM - 1) {
        const int nx = gx - 1;
        const int ny = gy + 1;
        uint8_t val = 0;
        if (nx >= 0 && ny < N) {
            val = static_cast<uint8_t>(input[ny * N + nx]);
        }
        tile[BDIM + 1][0] = val;
    }

    // Bottom-right corner halo (BDIM + 1, BDIM + 1)
    if (threadIdx.x == BDIM - 1 && threadIdx.y == BDIM - 1) {
        const int nx = gx + 1;
        const int ny = gy + 1;
        uint8_t val = 0;
        if (nx < N && ny < N) {
            val = static_cast<uint8_t>(input[ny * N + nx]);
        }
        tile[BDIM + 1][BDIM + 1] = val;
    }

    // Ensure all shared-memory loads are visible to all threads in the block.
    __syncthreads();

    // ------------------------------------------------------------------------
    // Compute the number of alive neighbors using the shared-memory tile.
    // The halo ensures all neighbor accesses are in-bounds of `tile`.
    // ------------------------------------------------------------------------
    int neighbors =
        tile[sy - 1][sx - 1] + tile[sy - 1][sx] + tile[sy - 1][sx + 1] +
        tile[sy][sx - 1]                        + tile[sy][sx + 1] +
        tile[sy + 1][sx - 1] + tile[sy + 1][sx] + tile[sy + 1][sx + 1];

    const uint8_t alive = tile[sy][sx];

    // Conway's Game of Life rules:
    // - Any alive cell with fewer than 2 alive neighbors dies.
    // - Any alive cell with 2 or 3 alive neighbors survives.
    // - Any alive cell with more than 3 alive neighbors dies.
    // - Any dead cell with exactly 3 alive neighbors becomes alive.
    bool next_state;
    if (alive) {
        next_state = (neighbors == 2 || neighbors == 3);
    } else {
        next_state = (neighbors == 3);
    }

    // Write the next state back to global memory.
    output[idx] = next_state;
}

// Host function that launches the CUDA kernel to perform one Game of Life step.
//
// Parameters:
//   input           - device pointer to input grid (N x N), bool per cell.
//   output          - device pointer to output grid (N x N), bool per cell.
//   grid_dimensions - grid width/height N (power of 2, > 512).
//
// Notes:
//   - Assumes `input` and `output` are already allocated on the GPU via cudaMalloc.
//   - Does not perform any host-device synchronization; caller is responsible.
//   - Relies on the condition that N is a multiple of BLOCK_DIM (32), which is
//     guaranteed here because N is a power of 2 and N >= 1024.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Configure a 2D grid of 32x32-thread blocks.
    // Because N is a power of 2 >= 1024, N % BLOCK_DIM == 0 and the grid
    // covers the domain exactly without partial blocks.
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(N / BLOCK_DIM, N / BLOCK_DIM);

    // Launch kernel. No explicit synchronization here.
    game_of_life_kernel<BLOCK_DIM><<<gridDim, blockDim>>>(input, output, N);
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
