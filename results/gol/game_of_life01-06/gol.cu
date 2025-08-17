#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA implementation of one step of Conway's Game of Life optimized for modern NVIDIA data center GPUs.

  Key performance choices:
  - 2D tiling with shared memory and a 1-cell halo border to reuse neighbor data across threads.
  - Shared memory is stored as 32-bit values to eliminate bank conflicts common with 8-bit shared memory access.
  - Coalesced global reads for the interior tile and top/bottom halo rows; left/right halo reads are few and acceptable.
  - Branching for boundary handling ensures cells outside the grid are treated as dead without invalid memory accesses.

  Assumptions:
  - The grid is square of size N x N where N is a power of 2 and N > 512.
  - input and output are device pointers allocated with cudaMalloc.
  - Each cell is a bool: false (0) or true (1).
  - Caller handles any stream synchronization, if needed.
*/

#ifndef GOL_BLOCK_X
#define GOL_BLOCK_X 32
#endif

#ifndef GOL_BLOCK_Y
#define GOL_BLOCK_Y 16
#endif

// CUDA kernel computing one generation of Conway's Game of Life.
// Input and output are treated as byte arrays (0 or 1), though provided as bool*.
__global__ void game_of_life_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width)
{
    // Shared memory tile sized to block + 2-cell halo in each dimension.
    // Store as 32-bit to avoid shared memory bank conflicts on Ampere/Hopper.
    constexpr int BLOCK_X = GOL_BLOCK_X;
    constexpr int BLOCK_Y = GOL_BLOCK_Y;
    constexpr int SMEM_PITCH = BLOCK_X + 2;
    constexpr int SMEM_HEIGHT = BLOCK_Y + 2;
    __shared__ uint32_t tile[SMEM_PITCH * SMEM_HEIGHT];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_X;
    const int by = blockIdx.y * BLOCK_Y;

    const int gx = bx + tx;
    const int gy = by + ty;

    // Local coordinates inside the shared memory tile (offset by +1 for halo).
    const int lx = tx + 1;
    const int ly = ty + 1;
    const int s_idx = ly * SMEM_PITCH + lx;

    // Utility lambdas for guarded global reads that clamp outside-grid to 0.
    auto load_cell = [&](int x, int y) -> uint32_t {
        if (x >= 0 && x < width && y >= 0 && y < width) {
            // Casting bool* to unsigned char* is safe (0 or 1) and faster for memory ops.
            return static_cast<uint32_t>(input[static_cast<size_t>(y) * width + x]);
        }
        return 0u;
    };

    // Load center of tile (coalesced).
    tile[s_idx] = load_cell(gx, gy);

    // Load halo rows (top and bottom) - coalesced across x.
    if (ty == 0) {
        tile[0 * SMEM_PITCH + lx] = load_cell(gx, gy - 1);
    }
    if (ty == BLOCK_Y - 1) {
        tile[(SMEM_HEIGHT - 1) * SMEM_PITCH + lx] = load_cell(gx, gy + 1);
    }

    // Load halo columns (left and right).
    if (tx == 0) {
        tile[ly * SMEM_PITCH + 0] = load_cell(gx - 1, gy);
    }
    if (tx == BLOCK_X - 1) {
        tile[ly * SMEM_PITCH + (SMEM_PITCH - 1)] = load_cell(gx + 1, gy);
    }

    // Load halo corners (4 cells).
    if (tx == 0 && ty == 0) {
        tile[0] = load_cell(bx - 1 + 0, by - 1 + 0);
    }
    if (tx == BLOCK_X - 1 && ty == 0) {
        tile[0 + (SMEM_PITCH - 1)] = load_cell(bx + BLOCK_X + 0, by - 1 + 0);
    }
    if (tx == 0 && ty == BLOCK_Y - 1) {
        tile[(SMEM_HEIGHT - 1) * SMEM_PITCH + 0] = load_cell(bx - 1 + 0, by + BLOCK_Y + 0);
    }
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        tile[(SMEM_HEIGHT - 1) * SMEM_PITCH + (SMEM_PITCH - 1)] =
            load_cell(bx + BLOCK_X + 0, by + BLOCK_Y + 0);
    }

    __syncthreads();

    // Only compute and store if within the grid.
    if (gx < width && gy < width) {
        const int up = s_idx - SMEM_PITCH;
        const int dn = s_idx + SMEM_PITCH;

        // Sum 8 neighbors from shared memory. Using 32-bit values for zero bank conflicts.
        uint32_t neighbor_sum =
            tile[up - 1] + tile[up] + tile[up + 1] +
            tile[s_idx - 1]            + tile[s_idx + 1] +
            tile[dn - 1] + tile[dn] + tile[dn + 1];

        uint32_t current = tile[s_idx];

        // Apply Conway's rules:
        // - Alive with 2 or 3 neighbors survives.
        // - Dead with exactly 3 neighbors becomes alive.
        // - Else dead.
        // Boolean arithmetic yields 0/1, kept as uint32_t for efficient ops.
        uint32_t next = (neighbor_sum == 3u) | (current & (neighbor_sum == 2u));

        output[static_cast<size_t>(gy) * width + gx] = static_cast<unsigned char>(next);
    }
}

// Host API function that executes one simulation step.
// input:  pointer to device memory (bool array).
// output: pointer to device memory (bool array) for the next generation.
// grid_dimensions: width/height (square grid).
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    if (grid_dimensions <= 0) return;

    constexpr int BLOCK_X = GOL_BLOCK_X;
    constexpr int BLOCK_Y = GOL_BLOCK_Y;

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid(
        (grid_dimensions + BLOCK_X - 1) / BLOCK_X,
        (grid_dimensions + BLOCK_Y - 1) / BLOCK_Y,
        1);

    // Reinterpret bool* as unsigned char* for direct byte-level operations.
    const unsigned char* in_ptr = reinterpret_cast<const unsigned char*>(input);
    unsigned char* out_ptr = reinterpret_cast<unsigned char*>(output);

    game_of_life_kernel<<<grid, block>>>(in_ptr, out_ptr, grid_dimensions);
}