#include <cuda_runtime.h>
#include <stdint.h>

// High-performance CUDA implementation of one step of Conway's Game of Life.
// Assumptions and design notes:
// - Input/output are device pointers allocated with cudaMalloc.
// - Grid is square (N x N), N is a power of two > 512, large enough to benefit from tiling.
// - Outside-grid cells are considered dead (zero-padding for boundary handling).
// - The kernel uses shared memory tiling with a 1-cell halo to minimize global memory traffic.
// - Shared memory uses 32-bit elements to avoid bank conflicts on modern GPUs (32 banks, 4 bytes/bank).
// - The kernel applies separate code paths for interior vs boundary blocks to minimize boundary checks.
// - Reads from input are cached via __ldg (read-only data cache) when appropriate.

#ifndef GOL_BLOCK_X
#define GOL_BLOCK_X 32  // X dimension of thread block; 32 aligns with warp size for coalesced loads
#endif

#ifndef GOL_BLOCK_Y
#define GOL_BLOCK_Y 16  // Y dimension of thread block; 16 provides 512 threads/block for good occupancy
#endif

using u8  = unsigned char;
using u32 = unsigned int;

// Device kernel: computes one generation of Game of Life using shared-memory tiling.
// Template parameters define the block geometry for compile-time shared memory sizing.
template<int BX, int BY>
__launch_bounds__(BX * BY, 2)
__global__ void gol_step_kernel(const u8* __restrict__ in, u8* __restrict__ out, int N)
{
    // Shared memory tile with 1-cell halo on all sides.
    // Using 32-bit elements avoids shared memory bank conflicts for row-wise access patterns.
    __shared__ u32 tile[BY + 2][BX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BX + tx;
    const int y  = blockIdx.y * BY + ty;

    // Determine if this is a block fully in the interior (no global boundary interactions).
    // For interior blocks, we can avoid bounds checks when loading halos.
    const bool interior_block =
        (blockIdx.x > 0) && (blockIdx.x < gridDim.x - 1) &&
        (blockIdx.y > 0) && (blockIdx.y < gridDim.y - 1);

    // Central cell load
    u32 center = 0;
    if (interior_block) {
        center = __ldg(&in[y * N + x]);
    } else {
        if (x < N && y < N) center = __ldg(&in[y * N + x]);
        else center = 0;
    }
    tile[ty + 1][tx + 1] = center;

    // Halo loads
    if (interior_block) {
        // Left and right halo columns
        if (tx == 0)             tile[ty + 1][0]         = __ldg(&in[y * N + (x - 1)]);
        if (tx == BX - 1)        tile[ty + 1][BX + 1]    = __ldg(&in[y * N + (x + 1)]);
        // Top and bottom halo rows
        if (ty == 0)             tile[0][tx + 1]         = __ldg(&in[(y - 1) * N + x]);
        if (ty == BY - 1)        tile[BY + 1][tx + 1]    = __ldg(&in[(y + 1) * N + x]);
        // Corner halos
        if (tx == 0 && ty == 0)                          tile[0][0]               = __ldg(&in[(y - 1) * N + (x - 1)]);
        if (tx == BX - 1 && ty == 0)                     tile[0][BX + 1]          = __ldg(&in[(y - 1) * N + (x + 1)]);
        if (tx == 0 && ty == BY - 1)                     tile[BY + 1][0]          = __ldg(&in[(y + 1) * N + (x - 1)]);
        if (tx == BX - 1 && ty == BY - 1)                tile[BY + 1][BX + 1]     = __ldg(&in[(y + 1) * N + (x + 1)]);
    } else {
        // Boundary-safe halo loads with zero-padding outside the grid.
        // Left halo
        if (tx == 0) {
            u32 v = 0;
            if (x > 0 && y < N) v = __ldg(&in[y * N + (x - 1)]);
            tile[ty + 1][0] = v;
        }
        // Right halo
        if (tx == BX - 1) {
            u32 v = 0;
            if ((x + 1) < N && y < N) v = __ldg(&in[y * N + (x + 1)]);
            tile[ty + 1][BX + 1] = v;
        }
        // Top halo
        if (ty == 0) {
            u32 v = 0;
            if (y > 0 && x < N) v = __ldg(&in[(y - 1) * N + x]);
            tile[0][tx + 1] = v;
        }
        // Bottom halo
        if (ty == BY - 1) {
            u32 v = 0;
            if ((y + 1) < N && x < N) v = __ldg(&in[(y + 1) * N + x]);
            tile[BY + 1][tx + 1] = v;
        }
        // Corner halos
        if (tx == 0 && ty == 0) {
            u32 v = 0;
            if (x > 0 && y > 0) v = __ldg(&in[(y - 1) * N + (x - 1)]);
            tile[0][0] = v;
        }
        if (tx == BX - 1 && ty == 0) {
            u32 v = 0;
            if ((x + 1) < N && y > 0) v = __ldg(&in[(y - 1) * N + (x + 1)]);
            tile[0][BX + 1] = v;
        }
        if (tx == 0 && ty == BY - 1) {
            u32 v = 0;
            if (x > 0 && (y + 1) < N) v = __ldg(&in[(y + 1) * N + (x - 1)]);
            tile[BY + 1][0] = v;
        }
        if (tx == BX - 1 && ty == BY - 1) {
            u32 v = 0;
            if ((x + 1) < N && (y + 1) < N) v = __ldg(&in[(y + 1) * N + (x + 1)]);
            tile[BY + 1][BX + 1] = v;
        }
    }

    __syncthreads();

    // Compute next state if within bounds.
    if (x < N && y < N) {
        // Row pointers to reduce address arithmetic.
        u32* r0 = &tile[ty][0];
        u32* r1 = &tile[ty + 1][0];
        u32* r2 = &tile[ty + 2][0];

        // Neighbor sum (8 neighbors)
        const int nx = tx + 1; // center column index in shared tile row
        int neighbors =
            r0[nx - 1] + r0[nx] + r0[nx + 1] +
            r1[nx - 1]           + r1[nx + 1] +
            r2[nx - 1] + r2[nx] + r2[nx + 1];

        const u32 alive = r1[nx];

        // Branchless rule application:
        // next = (neighbors == 3) | (alive & (neighbors == 2))
        const u32 next = ((neighbors == 3) | (alive & (neighbors == 2)));

        out[y * N + x] = static_cast<u8>(next);
    }
}

// Host function: launches the CUDA kernel to compute one Game of Life step.
// - input: device pointer to N x N grid of bool (false=dead, true=alive)
// - output: device pointer to N x N grid of bool (next state)
// - grid_dimensions: N (width/height), power of 2 > 512
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Reinterpret bool pointers as bytes for efficient memory access.
    // On CUDA, bool is 1 byte; storing 0/1 is compatible with bool semantics.
    const u8* in  = reinterpret_cast<const u8*>(input);
    u8* out       = reinterpret_cast<u8*>(output);

    // Configure a 2D grid of thread blocks covering the entire N x N domain.
    dim3 block(GOL_BLOCK_X, GOL_BLOCK_Y, 1);
    dim3 grid((N + GOL_BLOCK_X - 1) / GOL_BLOCK_X,
              (N + GOL_BLOCK_Y - 1) / GOL_BLOCK_Y,
              1);

    // Launch kernel. No explicit stream or synchronization here; caller handles synchronization.
    gol_step_kernel<GOL_BLOCK_X, GOL_BLOCK_Y><<<grid, block>>>(in, out, N);
}