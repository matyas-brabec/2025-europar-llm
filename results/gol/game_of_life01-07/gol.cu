#include <cuda_runtime.h>
#include <stdint.h>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life.

  Key optimizations:
  - Tiling with shared memory and a 1-cell halo reduces global memory traffic from 8 loads per cell to ~1 load per cell.
  - Block size of 32x32 threads aligns with warp size and the problem guarantee (grid dimension is a power of 2 > 512),
    ensuring the grid is exactly covered by blocks with no partial edge blocks. This avoids bounds checks on interior loads/stores.
  - Memory is treated as 8-bit (unsigned char) for compactness and efficient shared memory usage.
  - All out-of-grid neighbors are treated as dead (0) by explicit boundary checks when loading halo cells.

  Assumptions (per problem statement):
  - The grid is square with size N x N where N is a power of 2 and N > 512.
  - Input and output arrays are device pointers allocated via cudaMalloc.
  - Caller handles any necessary synchronization.
*/

template<int BLOCK_X, int BLOCK_Y>
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int width)
{
    // Reinterpret to 8-bit pointers for efficient arithmetic and shared memory use.
    const unsigned char* __restrict__ in  = reinterpret_cast<const unsigned char*>(input);
    unsigned char* __restrict__ out = reinterpret_cast<unsigned char*>(output);

    // Thread's global coordinates
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    // Shared memory tile with 1-cell halo on all sides.
    // Layout: (BLOCK_Y + 2) rows x (BLOCK_X + 2) columns
    __shared__ unsigned char tile[(BLOCK_Y + 2) * (BLOCK_X + 2)];
    const int tile_pitch = BLOCK_X + 2;     // stride between shared-memory rows

    // Thread's coordinates within the shared-memory tile (offset by +1 for halo).
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int s_index = ly * tile_pitch + lx;

    // Center cell: always in-bounds because width is a power of two > 512 and BLOCK dims divide width.
    tile[s_index] = __ldg(&in[y * width + x]);

    // Load halo cells cooperatively. Out-of-grid neighbors are treated as dead (0).
    // Top halo
    if (threadIdx.y == 0) {
        tile[0 * tile_pitch + lx] = (y > 0) ? __ldg(&in[(y - 1) * width + x]) : 0;
    }
    // Bottom halo
    if (threadIdx.y == BLOCK_Y - 1) {
        tile[(BLOCK_Y + 1) * tile_pitch + lx] = (y < width - 1) ? __ldg(&in[(y + 1) * width + x]) : 0;
    }
    // Left halo
    if (threadIdx.x == 0) {
        tile[ly * tile_pitch + 0] = (x > 0) ? __ldg(&in[y * width + (x - 1)]) : 0;
    }
    // Right halo
    if (threadIdx.x == BLOCK_X - 1) {
        tile[ly * tile_pitch + (BLOCK_X + 1)] = (x < width - 1) ? __ldg(&in[y * width + (x + 1)]) : 0;
    }

    // Corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0 * tile_pitch + 0] = (x > 0 && y > 0) ? __ldg(&in[(y - 1) * width + (x - 1)]) : 0;
    }
    if (threadIdx.x == BLOCK_X - 1 && threadIdx.y == 0) {
        tile[0 * tile_pitch + (BLOCK_X + 1)] = (x < width - 1 && y > 0) ? __ldg(&in[(y - 1) * width + (x + 1)]) : 0;
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_Y - 1) {
        tile[(BLOCK_Y + 1) * tile_pitch + 0] = (x > 0 && y < width - 1) ? __ldg(&in[(y + 1) * width + (x - 1)]) : 0;
    }
    if (threadIdx.x == BLOCK_X - 1 && threadIdx.y == BLOCK_Y - 1) {
        tile[(BLOCK_Y + 1) * tile_pitch + (BLOCK_X + 1)] =
            (x < width - 1 && y < width - 1) ? __ldg(&in[(y + 1) * width + (x + 1)]) : 0;
    }

    __syncthreads();

    // Sum the 8 neighbors from the shared-memory tile.
    const int up_row    = (ly - 1) * tile_pitch;
    const int mid_row   = ly * tile_pitch;
    const int down_row  = (ly + 1) * tile_pitch;
    const int left_col  = lx - 1;
    const int right_col = lx + 1;

    int neighbors = 0;
    neighbors += tile[up_row + left_col];
    neighbors += tile[up_row + lx];
    neighbors += tile[up_row + right_col];
    neighbors += tile[mid_row + left_col];
    neighbors += tile[mid_row + right_col];
    neighbors += tile[down_row + left_col];
    neighbors += tile[down_row + lx];
    neighbors += tile[down_row + right_col];

    // Apply Conway's rules.
    const unsigned char current = tile[s_index];
    const unsigned char next = (current ? (neighbors == 2 || neighbors == 3) : (neighbors == 3)) ? 1 : 0;

    // Store result.
    out[y * width + x] = next;
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Using 32x32 blocks:
    // - 1024 threads per block, good occupancy on A100/H100.
    // - Since grid_dimensions is a power of two > 512, it is divisible by 32.
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 32;

    const int width = grid_dimensions;

    // Grid dimensions exactly cover the domain with no partial blocks.
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(width / BLOCK_X, width / BLOCK_Y);

    // Launch kernel. No synchronization here; the caller handles it.
    game_of_life_kernel<BLOCK_X, BLOCK_Y><<<grid, block>>>(input, output, width);
}