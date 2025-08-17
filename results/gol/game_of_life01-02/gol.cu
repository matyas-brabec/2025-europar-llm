#include <cuda_runtime.h>
#include <stdint.h>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life.

  Key performance choices:
  - Tiled shared-memory stencil with 1-cell halo in each direction minimizes redundant global loads.
  - 32x32 thread blocks (1024 threads) map naturally to 32-lane warps along x for coalesced memory access.
  - Interior blocks (not at the domain boundaries) avoid bounds checks entirely; boundary blocks apply clamped loads (outside = 0).
  - Branch-free update rule using bitwise logic: next = (sum == 3) | (alive & (sum == 2)).

  Notes:
  - Grid is square, size is a power-of-two and >= 512, which is divisible by 32. This makes every block fully occupied.
  - All outside-of-grid cells are treated as dead (0).
  - Input and output are bool arrays allocated with cudaMalloc. We operate on them as bytes on device.
*/

template <int BX, int BY>
__device__ __forceinline__ unsigned char ld_cell_nocheck(const bool* __restrict__ in, int width, int x, int y) {
    // Fast path: caller ensures x,y are in-bounds.
    return static_cast<unsigned char>(in[y * width + x]);
}

template <int BX, int BY>
__device__ __forceinline__ unsigned char ld_cell_clamped(const bool* __restrict__ in, int width, int height, int x, int y) {
    // Slow path: return 0 if out-of-bounds, otherwise return cell value as 0/1.
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) return 0;
    return static_cast<unsigned char>(in[y * width + x]);
}

template <int BX, int BY>
__launch_bounds__(BX * BY, 2)
__global__ void game_of_life_kernel_tiled(const bool* __restrict__ in, bool* __restrict__ out, int width, int height) {
    // Tile size includes a 1-cell halo on all sides.
    constexpr int TILE_W = BX + 2;
    constexpr int TILE_H = BY + 2;

    // Use a 1D shared memory buffer with row-major layout for fast index math.
    __shared__ unsigned char tile[TILE_W * TILE_H];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int x = bx * BX + tx;
    const int y = by * BY + ty;

    // Shared memory index of the "center" element corresponding to (x,y).
    const int s_idx = (ty + 1) * TILE_W + (tx + 1);

    // Determine if this is a strictly interior block (i.e., all neighbors exist in-bounds).
    const bool interiorBlock = (bx > 0 && bx < gridDim.x - 1 && by > 0 && by < gridDim.y - 1);

    if (interiorBlock) {
        // Center cell
        tile[s_idx] = ld_cell_nocheck<BX, BY>(in, width, x, y);

        // Halo in X
        if (tx == 0) {
            tile[(ty + 1) * TILE_W + 0] = ld_cell_nocheck<BX, BY>(in, width, x - 1, y);
        }
        if (tx == BX - 1) {
            tile[(ty + 1) * TILE_W + (BX + 1)] = ld_cell_nocheck<BX, BY>(in, width, x + 1, y);
        }

        // Halo in Y
        if (ty == 0) {
            tile[0 * TILE_W + (tx + 1)] = ld_cell_nocheck<BX, BY>(in, width, x, y - 1);
        }
        if (ty == BY - 1) {
            tile[(BY + 1) * TILE_W + (tx + 1)] = ld_cell_nocheck<BX, BY>(in, width, x, y + 1);
        }

        // Corner halos
        if (tx == 0 && ty == 0) {
            tile[0] = ld_cell_nocheck<BX, BY>(in, width, x - 1, y - 1);
        }
        if (tx == BX - 1 && ty == 0) {
            tile[0 * TILE_W + (BX + 1)] = ld_cell_nocheck<BX, BY>(in, width, x + 1, y - 1);
        }
        if (tx == 0 && ty == BY - 1) {
            tile[(BY + 1) * TILE_W + 0] = ld_cell_nocheck<BX, BY>(in, width, x - 1, y + 1);
        }
        if (tx == BX - 1 && ty == BY - 1) {
            tile[(BY + 1) * TILE_W + (BX + 1)] = ld_cell_nocheck<BX, BY>(in, width, x + 1, y + 1);
        }
    } else {
        // Boundary blocks: clamp out-of-bounds to 0.
        unsigned char c = 0;
        if ((unsigned)x < (unsigned)width && (unsigned)y < (unsigned)height) {
            c = static_cast<unsigned char>(in[y * width + x]);
        }
        tile[s_idx] = c;

        if (tx == 0) {
            tile[(ty + 1) * TILE_W + 0] = ((x > 0) && ((unsigned)y < (unsigned)height)) ? static_cast<unsigned char>(in[y * width + (x - 1)]) : 0;
        }
        if (tx == BX - 1) {
            tile[(ty + 1) * TILE_W + (BX + 1)] = ((x + 1 < width) && ((unsigned)y < (unsigned)height)) ? static_cast<unsigned char>(in[y * width + (x + 1)]) : 0;
        }
        if (ty == 0) {
            tile[0 * TILE_W + (tx + 1)] = ((y > 0) && ((unsigned)x < (unsigned)width)) ? static_cast<unsigned char>(in[(y - 1) * width + x]) : 0;
        }
        if (ty == BY - 1) {
            tile[(BY + 1) * TILE_W + (tx + 1)] = ((y + 1 < height) && ((unsigned)x < (unsigned)width)) ? static_cast<unsigned char>(in[(y + 1) * width + x]) : 0;
        }

        if (tx == 0 && ty == 0) {
            tile[0] = (x > 0 && y > 0) ? static_cast<unsigned char>(in[(y - 1) * width + (x - 1)]) : 0;
        }
        if (tx == BX - 1 && ty == 0) {
            tile[0 * TILE_W + (BX + 1)] = ((x + 1 < width) && (y > 0)) ? static_cast<unsigned char>(in[(y - 1) * width + (x + 1)]) : 0;
        }
        if (tx == 0 && ty == BY - 1) {
            tile[(BY + 1) * TILE_W + 0] = ((x > 0) && (y + 1 < height)) ? static_cast<unsigned char>(in[(y + 1) * width + (x - 1)]) : 0;
        }
        if (tx == BX - 1 && ty == BY - 1) {
            tile[(BY + 1) * TILE_W + (BX + 1)] = ((x + 1 < width) && (y + 1 < height)) ? static_cast<unsigned char>(in[(y + 1) * width + (x + 1)]) : 0;
        }
    }

    __syncthreads();

    // Safety guard for non-multiple-of-block sizes (not needed for the power-of-two inputs but keeps the kernel robust).
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) {
        return;
    }

    // Sum the 8 neighbors from the shared-memory tile.
    const int idx = s_idx;
    int sum = 0;
    // Top row neighbors
    sum += tile[idx - TILE_W - 1];
    sum += tile[idx - TILE_W];
    sum += tile[idx - TILE_W + 1];
    // Left and right neighbors
    sum += tile[idx - 1];
    sum += tile[idx + 1];
    // Bottom row neighbors
    sum += tile[idx + TILE_W - 1];
    sum += tile[idx + TILE_W];
    sum += tile[idx + TILE_W + 1];

    // Alive state and next state (branch-free rule application).
    const unsigned char alive = tile[idx];
    const int is2 = (sum == 2);
    const int is3 = (sum == 3);
    const unsigned char next = static_cast<unsigned char>(is3 | (alive & is2));

    out[y * width + x] = (next != 0);
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Launch parameters tuned for modern data center GPUs (A100/H100).
    // 32x32 threads per block gives excellent memory coalescing and utilization.
    constexpr int BX = 32;
    constexpr int BY = 32;

    const int width = grid_dimensions;
    const int height = grid_dimensions;

    dim3 block(BX, BY, 1);
    dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY, 1);

    game_of_life_kernel_tiled<BX, BY><<<grid, block>>>(input, output, width, height);
}