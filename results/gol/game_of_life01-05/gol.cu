#include <cuda_runtime.h>
#include <stdint.h>

/*
High-performance CUDA implementation of one step of Conway's Game of Life.

Key optimization choices:
- Tiled shared-memory staging with a 1-cell halo on all sides to minimize global memory traffic.
- Shared memory stores use 32-bit elements (0/1) to avoid bank conflicts that occur with 8-bit elements.
- Coalesced global loads for the main tile and top/bottom halo rows; only a small number of strided loads for left/right halo columns.
- Branch-free rule evaluation for alive/dead transitions to reduce warp divergence.
- 2D block size chosen as 32x16 (512 threads) to align with warp size and provide good occupancy on A100/H100.

Assumptions:
- Grid is square with dimension N, power of two, >= 512.
- All out-of-bounds cells are treated as dead (0). Halo loads clamp to zero when outside.
- Input and output pointers are device pointers allocated by cudaMalloc.
- Caller handles any necessary synchronization.
*/

#ifndef GOL_BLOCK_X
#define GOL_BLOCK_X 32
#endif

#ifndef GOL_BLOCK_Y
#define GOL_BLOCK_Y 16
#endif

// Convert a device bool to 0/1 as uint32_t, force-inlined for performance.
static __device__ __forceinline__ uint32_t bool_to_u32(bool b) {
    // Using ternary avoids potential pitfalls of casting bool to integer types in device code.
    return b ? 1u : 0u;
}

template<int BLOCK_X, int BLOCK_Y>
__global__ void game_of_life_kernel(const bool* __restrict__ input,
                                    bool* __restrict__ output,
                                    int N)
{
    // Shared tile with 1-cell halo around the block tile.
    // Using 32-bit elements avoids shared memory bank conflicts for row-wise neighbor accesses.
    __shared__ uint32_t tile[(BLOCK_Y + 2) * (BLOCK_X + 2)];

    const int pitch = BLOCK_X + 2;

    // Local thread coordinates within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global coordinates this thread is responsible for
    const int gx = blockIdx.x * BLOCK_X + tx;
    const int gy = blockIdx.y * BLOCK_Y + ty;

    // Convenience lambda to compute 1D index into shared tile
    auto s_index = [pitch](int y, int x) { return y * pitch + x; };

    // Load the central tile cell into shared memory at (ty+1, tx+1).
    uint32_t center = 0u;
    if (gx < N && gy < N) {
        center = bool_to_u32(input[gy * N + gx]);
    }
    tile[s_index(ty + 1, tx + 1)] = center;

    // Top halo row: threads in the first row of the block load the row above
    if (ty == 0) {
        const int gy_top = gy - 1;
        uint32_t val = 0u;
        if (gx < N && gy_top >= 0) {
            val = bool_to_u32(input[gy_top * N + gx]);
        }
        tile[s_index(0, tx + 1)] = val;
    }

    // Bottom halo row: threads in the last row of the block load the row below
    if (ty == BLOCK_Y - 1) {
        const int gy_bot = gy + 1;
        uint32_t val = 0u;
        if (gx < N && gy_bot < N) {
            val = bool_to_u32(input[gy_bot * N + gx]);
        }
        tile[s_index(BLOCK_Y + 1, tx + 1)] = val;
    }

    // Left halo column: threads in the first column of the block load the column to the left
    if (tx == 0) {
        const int gx_left = gx - 1;
        uint32_t val = 0u;
        if (gx_left >= 0 && gy < N) {
            val = bool_to_u32(input[gy * N + gx_left]);
        }
        tile[s_index(ty + 1, 0)] = val;
    }

    // Right halo column: threads in the last column of the block load the column to the right
    if (tx == BLOCK_X - 1) {
        const int gx_right = gx + 1;
        uint32_t val = 0u;
        if (gx_right < N && gy < N) {
            val = bool_to_u32(input[gy * N + gx_right]);
        }
        tile[s_index(ty + 1, BLOCK_X + 1)] = val;
    }

    // Four corners of the halo, handled by the four corner threads
    if (tx == 0 && ty == 0) {
        uint32_t val = 0u;
        if (gx > 0 && gy > 0) {
            val = bool_to_u32(input[(gy - 1) * N + (gx - 1)]);
        }
        tile[s_index(0, 0)] = val;
    }
    if (tx == BLOCK_X - 1 && ty == 0) {
        uint32_t val = 0u;
        if (gx + 1 < N && gy > 0) {
            val = bool_to_u32(input[(gy - 1) * N + (gx + 1)]);
        }
        tile[s_index(0, BLOCK_X + 1)] = val;
    }
    if (tx == 0 && ty == BLOCK_Y - 1) {
        uint32_t val = 0u;
        if (gx > 0 && gy + 1 < N) {
            val = bool_to_u32(input[(gy + 1) * N + (gx - 1)]);
        }
        tile[s_index(BLOCK_Y + 1, 0)] = val;
    }
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        uint32_t val = 0u;
        if (gx + 1 < N && gy + 1 < N) {
            val = bool_to_u32(input[(gy + 1) * N + (gx + 1)]);
        }
        tile[s_index(BLOCK_Y + 1, BLOCK_X + 1)] = val;
    }

    // Ensure the tile (including halo) is fully populated before computing neighbors
    __syncthreads();

    // Only threads that map to valid grid cells write results
    if (gx < N && gy < N) {
        // Shared tile index for the current thread's cell
        const int si = s_index(ty + 1, tx + 1);

        // Sum the 8 neighbors from shared memory
        // Using 32-bit adds; tile entries are 0 or 1.
        uint32_t sum =
            tile[si - pitch - 1] + tile[si - pitch] + tile[si - pitch + 1] +
            tile[si - 1]                      +          tile[si + 1] +
            tile[si + pitch - 1] + tile[si + pitch] + tile[si + pitch + 1];

        // Current state (0 or 1)
        const uint32_t cur = tile[si];

        // Branch-free rule evaluation:
        // next = (sum == 3) || (cur == 1 && sum == 2)
        const bool next = (sum == 3u) | ((cur == 1u) & (sum == 2u));

        // Write next state to global memory; coalesced across the x dimension.
        output[gy * N + gx] = next;
    }
}

// Host API: executes one step of Conway's Game of Life.
// input  - device pointer to N*N bools (alive=true, dead=false)
// output - device pointer to N*N bools to receive next state
// grid_dimensions - N (width/height), power of two, >= 512
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Block and grid configuration; chosen to match warp size in X and provide high occupancy.
    constexpr int BX = GOL_BLOCK_X;
    constexpr int BY = GOL_BLOCK_Y;

    dim3 block(BX, BY, 1);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY, 1);

    // Launch the kernel. Caller is responsible for any synchronization.
    game_of_life_kernel<BX, BY><<<grid, block>>>(input, output, N);
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
