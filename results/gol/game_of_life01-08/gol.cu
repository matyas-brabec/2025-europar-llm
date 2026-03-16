#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// High-performance single-step Conway's Game of Life for square power-of-two grids.
//
// Design choices tuned for modern datacenter GPUs (A100/H100 class):
// - 32x16 thread blocks:
//   * blockDim.x == 32 means each warp owns exactly one tile row.
//   * this lets horizontal neighbors come from warp shuffles instead of shared memory.
// - Shared memory stores 32-bit values, not bytes:
//   * the input/output grid is byte-sized (bool), but 8-bit shared-memory accesses create
//     severe bank conflicts. Staging cells as uint32_t avoids that.
// - Interior blocks use a fast path with no per-load bounds checks.
//   Only the outermost ring of blocks executes boundary-safe zero-padding logic.
// - Linear indexing uses 64 bits because n*n can exceed 2^31 even when both grids still fit
//   in GPU memory on large devices.

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 16;
constexpr int BLOCK_X_LOG2 = 5;
constexpr int BLOCK_Y_LOG2 = 4;
constexpr int TILE_W = BLOCK_X + 2;
constexpr int TILE_H = BLOCK_Y + 2;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

static_assert(BLOCK_X == 32, "This kernel assumes one warp per block row.");
static_assert(sizeof(bool) == 1, "CUDA bool is expected to be byte-sized.");

__device__ __forceinline__ uint32_t load_cell(const uint8_t* ptr) {
    // Input bools are represented as bytes in device memory; reading through uint8_t is valid
    // and keeps the hot path simple.
    return static_cast<uint32_t>(*ptr);
}

// 32x16 = 512 threads/block. launch_bounds encourages the compiler to stay within a register
// footprint that still allows 4 such blocks/SM on A100/H100-class GPUs.
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y, 4)
void game_of_life_kernel(const uint8_t* __restrict__ input,
                         bool* __restrict__ output,
                         int n) {
    __shared__ uint32_t tile[TILE_H][TILE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);

    const int gx = (bx << BLOCK_X_LOG2) + tx;
    const int gy = (by << BLOCK_Y_LOG2) + ty;

    const ptrdiff_t pitch = static_cast<ptrdiff_t>(n);
    const size_t idx = static_cast<size_t>(gy) * static_cast<size_t>(pitch) + static_cast<size_t>(gx);
    const uint8_t* const in_ptr = input + idx;

    // Keep the current cell in a register for the entire kernel; neighbors in y come from shared
    // memory, neighbors in x come from warp shuffles.
    const uint32_t self = load_cell(in_ptr);
    tile[ty + 1][tx + 1] = self;

    // Uniform (block-wide) branch: only the outermost ring of blocks needs boundary handling.
    const bool interior_block =
        (bx > 0) &
        (by > 0) &
        (bx + 1 < static_cast<int>(gridDim.x)) &
        (by + 1 < static_cast<int>(gridDim.y));

    if (interior_block) {
        // Fast path: all halo loads are in-bounds.
        if (tx == 0) {
            tile[ty + 1][0] = load_cell(in_ptr - 1);
        }
        if (tx == BLOCK_X - 1) {
            tile[ty + 1][BLOCK_X + 1] = load_cell(in_ptr + 1);
        }
        if (ty == 0) {
            tile[0][tx + 1] = load_cell(in_ptr - pitch);
        }
        if (ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][tx + 1] = load_cell(in_ptr + pitch);
        }
        if (tx == 0 && ty == 0) {
            tile[0][0] = load_cell(in_ptr - pitch - 1);
        }
        if (tx == BLOCK_X - 1 && ty == 0) {
            tile[0][BLOCK_X + 1] = load_cell(in_ptr - pitch + 1);
        }
        if (tx == 0 && ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][0] = load_cell(in_ptr + pitch - 1);
        }
        if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][BLOCK_X + 1] = load_cell(in_ptr + pitch + 1);
        }
    } else {
        // Boundary-safe path: outside-the-grid cells are treated as dead (0).
        if (tx == 0) {
            tile[ty + 1][0] = (gx > 0) ? load_cell(in_ptr - 1) : 0u;
        }
        if (tx == BLOCK_X - 1) {
            tile[ty + 1][BLOCK_X + 1] = (gx + 1 < n) ? load_cell(in_ptr + 1) : 0u;
        }
        if (ty == 0) {
            tile[0][tx + 1] = (gy > 0) ? load_cell(in_ptr - pitch) : 0u;
        }
        if (ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][tx + 1] = (gy + 1 < n) ? load_cell(in_ptr + pitch) : 0u;
        }
        if (tx == 0 && ty == 0) {
            tile[0][0] = (gx > 0 && gy > 0) ? load_cell(in_ptr - pitch - 1) : 0u;
        }
        if (tx == BLOCK_X - 1 && ty == 0) {
            tile[0][BLOCK_X + 1] = (gx + 1 < n && gy > 0) ? load_cell(in_ptr - pitch + 1) : 0u;
        }
        if (tx == 0 && ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][0] = (gx > 0 && gy + 1 < n) ? load_cell(in_ptr + pitch - 1) : 0u;
        }
        if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][BLOCK_X + 1] =
                (gx + 1 < n && gy + 1 < n) ? load_cell(in_ptr + pitch + 1) : 0u;
        }
    }

    __syncthreads();

    // After staging:
    // - north/south center-column values come from shared memory
    // - left/right neighbors come from warp shuffles
    // - the current cell stays in the 'self' register
    const uint32_t north = tile[ty][tx + 1];
    const uint32_t south = tile[ty + 2][tx + 1];

    uint32_t north_l = __shfl_up_sync(FULL_MASK, north, 1);
    uint32_t north_r = __shfl_down_sync(FULL_MASK, north, 1);
    uint32_t self_l  = __shfl_up_sync(FULL_MASK, self, 1);
    uint32_t self_r  = __shfl_down_sync(FULL_MASK, self, 1);
    uint32_t south_l = __shfl_up_sync(FULL_MASK, south, 1);
    uint32_t south_r = __shfl_down_sync(FULL_MASK, south, 1);

    // Warp edges pull horizontal halo values from shared memory.
    if (tx == 0) {
        north_l = tile[ty][0];
        self_l  = tile[ty + 1][0];
        south_l = tile[ty + 2][0];
    }
    if (tx == BLOCK_X - 1) {
        north_r = tile[ty][BLOCK_X + 1];
        self_r  = tile[ty + 1][BLOCK_X + 1];
        south_r = tile[ty + 2][BLOCK_X + 1];
    }

    const int neighbor_count = static_cast<int>(
        north_l + north + north_r +
        self_l            + self_r +
        south_l + south + south_r);

    // Branch-free Game of Life rule evaluation:
    // next = (neighbors == 3) || (self && neighbors == 2)
    const uint32_t next =
        static_cast<uint32_t>(neighbor_count == 3) |
        (static_cast<uint32_t>(neighbor_count == 2) & self);

    output[idx] = static_cast<bool>(next);
}

}  // namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // The problem guarantees a power-of-two square board with dimension > 512.
    // Therefore these 32x16 tiles divide the board exactly: no partial blocks are needed.
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned int>(grid_dimensions >> BLOCK_X_LOG2),
                    static_cast<unsigned int>(grid_dimensions >> BLOCK_Y_LOG2));

    const auto* const input_bytes = reinterpret_cast<const uint8_t*>(input);

    // Intentionally no synchronization and no error-checking here; the caller owns both.
    game_of_life_kernel<<<grid, block>>>(input_bytes, output, grid_dimensions);
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
