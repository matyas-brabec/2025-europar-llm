#include <cuda_runtime.h>
#include <stdint.h>

// CUDA implementation of one step of Conway's Game of Life.
// - Input and output are device pointers allocated with cudaMalloc.
// - Grid is square: N x N cells, stored row-major as bytes (bools on device are 1 byte).
// - Cells outside the grid are treated as dead (0).
// - Kernel uses shared-memory tiling with a 1-cell halo to minimize global memory traffic.
// - Tile size is 32x32, leveraging warp-friendly dimensions and ensuring divisibility given N is a power of 2 > 512.

// We rely on bool being 1 byte to safely alias it as unsigned char for efficient loads/stores.
static_assert(sizeof(bool) == 1, "Device bool is expected to be 1 byte.");

// Tunable tile dimensions; chosen for high throughput and coalesced memory access.
#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 32
#endif

// Kernel: computes one Game of Life step on a 2D grid.
// - input/output: reinterpret_cast from bool* to unsigned char* at call site.
// - N: grid dimension (width == height == N).
// - Shared memory tile is (TILE_Y+2) x (TILE_X+2) to hold a halo border.
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void life_step_kernel(const unsigned char* __restrict__ input,
                      unsigned char* __restrict__ output,
                      int N)
{
    // Shared memory tile; halo of 1 cell on each side.
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    // Global coordinates of the cell this thread is responsible for.
    const int gx = blockIdx.x * TILE_W + threadIdx.x;
    const int gy = blockIdx.y * TILE_H + threadIdx.y;

    // Local coordinates inside the shared tile (offset by halo).
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;

    // Load the center cell into shared memory, with out-of-bounds treated as dead (0).
    unsigned char center = 0;
    if (gx < N && gy < N) {
        center = input[gy * N + gx];
    }
    tile[ly][lx] = center;

    // Load halo cells. We use boundary checks to avoid out-of-bounds accesses.
    // Top halo
    if (threadIdx.y == 0) {
        int yy = gy - 1;
        unsigned char val = (yy >= 0 && gx < N) ? input[yy * N + gx] : 0;
        tile[0][lx] = val;
    }
    // Bottom halo
    if (threadIdx.y == TILE_H - 1) {
        int yy = gy + 1;
        unsigned char val = (yy < N && gx < N) ? input[yy * N + gx] : 0;
        tile[TILE_H + 1][lx] = val;
    }
    // Left halo
    if (threadIdx.x == 0) {
        int xx = gx - 1;
        unsigned char val = (xx >= 0 && gy < N) ? input[gy * N + xx] : 0;
        tile[ly][0] = val;
    }
    // Right halo
    if (threadIdx.x == TILE_W - 1) {
        int xx = gx + 1;
        unsigned char val = (xx < N && gy < N) ? input[gy * N + xx] : 0;
        tile[ly][TILE_W + 1] = val;
    }
    // Corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int xx = gx - 1, yy = gy - 1;
        unsigned char val = (xx >= 0 && yy >= 0) ? input[yy * N + xx] : 0;
        tile[0][0] = val;
    }
    if (threadIdx.x == TILE_W - 1 && threadIdx.y == 0) {
        int xx = gx + 1, yy = gy - 1;
        unsigned char val = (xx < N && yy >= 0) ? input[yy * N + xx] : 0;
        tile[0][TILE_W + 1] = val;
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE_H - 1) {
        int xx = gx - 1, yy = gy + 1;
        unsigned char val = (xx >= 0 && yy < N) ? input[yy * N + xx] : 0;
        tile[TILE_H + 1][0] = val;
    }
    if (threadIdx.x == TILE_W - 1 && threadIdx.y == TILE_H - 1) {
        int xx = gx + 1, yy = gy + 1;
        unsigned char val = (xx < N && yy < N) ? input[yy * N + xx] : 0;
        tile[TILE_H + 1][TILE_W + 1] = val;
    }

    __syncthreads();

    // Compute next state if within bounds; outside cells are ignored (they were treated as dead).
    if (gx < N && gy < N) {
        // Sum 8 neighbors from shared memory around (ly, lx).
        // Access pattern is arranged to be simple and fast.
        int sum =
            tile[ly - 1][lx - 1] + tile[ly - 1][lx] + tile[ly - 1][lx + 1] +
            tile[ly][lx - 1]                           + tile[ly][lx + 1] +
            tile[ly + 1][lx - 1] + tile[ly + 1][lx] + tile[ly + 1][lx + 1];

        unsigned char alive = tile[ly][lx];

        // Game of Life rule (branchless):
        // next = (sum == 3) || (alive && sum == 2)
        unsigned char next = (unsigned char)((sum == 3) | (alive & (sum == 2)));

        output[gy * N + gx] = next;
    }
}

// Host API: executes one step of Conway's Game of Life.
// - input: device pointer to N x N bool grid (true=alive, false=dead).
// - output: device pointer to N x N bool grid (next state).
// - grid_dimensions: N (power of 2, > 512).
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Alias bool storage as bytes for efficient bulk operations.
    const unsigned char* in = reinterpret_cast<const unsigned char*>(input);
    unsigned char* out = reinterpret_cast<unsigned char*>(output);

    const int N = grid_dimensions;

    // Tile configuration. Given N is a power of 2 greater than 512, it is divisible by 32.
    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y, 1);

    // Launch kernel. No synchronization here; caller handles it as stated.
    life_step_kernel<TILE_X, TILE_Y><<<grid, block>>>(in, out, N);
}