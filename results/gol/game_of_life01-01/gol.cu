// CUDA implementation of Conway's Game of Life optimized for modern NVIDIA GPUs (A100/H100).
// Key optimizations:
// - 2D tiling with shared memory and a 1-cell halo to minimize global memory traffic.
// - Coalesced global memory accesses for the interior of the tile and for top/bottom halos.
// - Minimal conditional logic (boundary checks only on the halo loads).
// - Block dimensions chosen to align with warp size and typical SM scheduling.
//
// Assumptions from the problem statement:
// - input and output are device pointers allocated with cudaMalloc.
// - Grid is square with dimensions N x N, where N is a power of 2 (> 512). Thus, N is divisible by 32 and 16.
// - Cells outside the grid are considered dead (0).
// - run_game_of_life launches a single step (one generation) of the simulation.
// - Caller handles synchronization; this function only launches the kernel.

#include <cuda_runtime.h>
#include <stdint.h>

// Tuned block dimensions:
// - X dimension is 32 to match a warp for maximal coalescing on row-wise loads.
// - Y dimension is 16 to provide 512 threads/block and good occupancy on A100/H100.
#ifndef GOL_BLOCK_DIM_X
#define GOL_BLOCK_DIM_X 32
#endif

#ifndef GOL_BLOCK_DIM_Y
#define GOL_BLOCK_DIM_Y 16
#endif

// Kernel implements one Game of Life step using shared-memory tiling.
// Pointers are marked __restrict__ to aid compiler alias analysis.
// The input pointer is treated as read-only; the compiler will often route through the read-only cache.
__launch_bounds__(GOL_BLOCK_DIM_X * GOL_BLOCK_DIM_Y, 2)
__global__ void game_of_life_kernel_shared(const bool* __restrict__ input,
                                           bool* __restrict__ output,
                                           int N)
{
    // Shared memory tile with 1-cell halo on all sides.
    // Using unsigned char for compact per-cell storage (bools map to 0/1).
    __shared__ unsigned char tile[GOL_BLOCK_DIM_Y + 2][GOL_BLOCK_DIM_X + 2];

    // Local thread indices and global coordinates.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gx = blockIdx.x * GOL_BLOCK_DIM_X + tx;
    const int gy = blockIdx.y * GOL_BLOCK_DIM_Y + ty;

    const int lx = tx + 1; // +1 accounts for halo
    const int ly = ty + 1;

    // Cast input/output to byte-addressable views for efficient loads/stores.
    const unsigned char* __restrict__ in  = reinterpret_cast<const unsigned char*>(input);
    unsigned char* __restrict__ out = reinterpret_cast<unsigned char*>(output);

    // The following loads are structured for coalescing:
    // - Interior tile load: all threads load their corresponding cell (fully coalesced along x).
    // - Top/bottom halos: loaded by threads with ty==0 or ty==BLOCK_Y-1 (coalesced along x).
    // - Left/right halos: loaded by threads with tx==0 or tx==BLOCK_X-1 (strided in y, but only 2*BLOCK_Y loads).

    // Load interior cell into shared memory (always in-bounds because N is divisible by block dims).
    // Still keep a check for robustness if used with arbitrary sizes.
    unsigned char center = 0;
    if (gx < N && gy < N) {
        // Use __ldg to prefer read-only caching where beneficial on supported architectures.
        center = __ldg(in + (size_t)gy * N + gx);
    }
    tile[ly][lx] = center;

    // Left halo (x-1)
    if (tx == 0) {
        unsigned char val = 0;
        if (gx > 0 && gy < N) {
            val = __ldg(in + (size_t)gy * N + (gx - 1));
        }
        tile[ly][0] = val;
    }

    // Right halo (x+1)
    if (tx == (GOL_BLOCK_DIM_X - 1)) {
        unsigned char val = 0;
        if ((gx + 1) < N && gy < N) {
            val = __ldg(in + (size_t)gy * N + (gx + 1));
        }
        tile[ly][GOL_BLOCK_DIM_X + 1] = val;
    }

    // Top halo (y-1)
    if (ty == 0) {
        unsigned char val = 0;
        if (gy > 0 && gx < N) {
            val = __ldg(in + (size_t)(gy - 1) * N + gx);
        }
        tile[0][lx] = val;
    }

    // Bottom halo (y+1)
    if (ty == (GOL_BLOCK_DIM_Y - 1)) {
        unsigned char val = 0;
        if ((gy + 1) < N && gx < N) {
            val = __ldg(in + (size_t)(gy + 1) * N + gx);
        }
        tile[GOL_BLOCK_DIM_Y + 1][lx] = val;
    }

    // Corner halos:
    if (tx == 0 && ty == 0) {
        unsigned char val = 0;
        if (gx > 0 && gy > 0) {
            val = __ldg(in + (size_t)(gy - 1) * N + (gx - 1));
        }
        tile[0][0] = val;
    }

    if (tx == (GOL_BLOCK_DIM_X - 1) && ty == 0) {
        unsigned char val = 0;
        if ((gx + 1) < N && gy > 0) {
            val = __ldg(in + (size_t)(gy - 1) * N + (gx + 1));
        }
        tile[0][GOL_BLOCK_DIM_X + 1] = val;
    }

    if (tx == 0 && ty == (GOL_BLOCK_DIM_Y - 1)) {
        unsigned char val = 0;
        if (gx > 0 && (gy + 1) < N) {
            val = __ldg(in + (size_t)(gy + 1) * N + (gx - 1));
        }
        tile[GOL_BLOCK_DIM_Y + 1][0] = val;
    }

    if (tx == (GOL_BLOCK_DIM_X - 1) && ty == (GOL_BLOCK_DIM_Y - 1)) {
        unsigned char val = 0;
        if ((gx + 1) < N && (gy + 1) < N) {
            val = __ldg(in + (size_t)(gy + 1) * N + (gx + 1));
        }
        tile[GOL_BLOCK_DIM_Y + 1][GOL_BLOCK_DIM_X + 1] = val;
    }

    __syncthreads();

    // Compute next state if within bounds.
    if (gx < N && gy < N) {
        // Neighbor sum calculation using shared memory. We do not include the center cell.
        // Access pattern uses the halo to avoid conditionals for edges.
        const int lx0 = lx - 1, lx1 = lx, lx2 = lx + 1;
        const int ly0 = ly - 1, ly2 = ly + 1;

        // Sum 8 neighbors: three from row above, two from current row (left/right), three from row below.
        int nsum = 0;
        nsum += tile[ly0][lx0] + tile[ly0][lx1] + tile[ly0][lx2];
        nsum += tile[ly][lx0]                 +                tile[ly][lx2];
        nsum += tile[ly2][lx0] + tile[ly2][lx1] + tile[ly2][lx2];

        const unsigned char cur = tile[ly][lx];
        const unsigned char next = (nsum == 3) | (cur & (nsum == 2));

        out[(size_t)gy * N + gx] = next;
    }
}

// Host function that launches one step of Conway's Game of Life.
// input:  device pointer to N*N bools representing the current grid.
// output: device pointer to N*N bools to receive the next grid.
// N:      grid dimensions (width == height), power of 2 > 512.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Grid configuration. Because N is a power of 2 and > 512, it is divisible by these block dims.
    dim3 block(GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y);
    dim3 grid((N + GOL_BLOCK_DIM_X - 1) / GOL_BLOCK_DIM_X,
              (N + GOL_BLOCK_DIM_Y - 1) / GOL_BLOCK_DIM_Y);

    // Launch the kernel. No explicit synchronization here; caller handles it.
    game_of_life_kernel_shared<<<grid, block>>>(input, output, N);
}