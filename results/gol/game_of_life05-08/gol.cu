#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace detail {

// Broadword / SWAR implementation for Conway's Game of Life.
//
// - One CUDA thread owns one 64-bit packed word, so no atomics are needed.
// - Each thread advances 64 cells at once with bitwise logic.
// - The 0th and 63rd bits are handled by injecting edge bits from the adjacent
//   left/right words, so there is no per-bit scalar slow path.
// - A tiny boundary kernel peels the outer perimeter away from the hot interior
//   kernel. This keeps the interior loads branch-free while still honoring the
//   "outside the grid is dead" rule.
// - Shared/texture memory are intentionally not used here; for this access
//   pattern, the hardware caches on modern A100/H100-class GPUs are sufficient.

using u64   = std::uint64_t;
using usize = std::size_t;

constexpr int kThreadsPerBlock = 256;
constexpr int kWordShift       = 6;   // 64 cells per packed word.
constexpr int kEdgeBit         = 63;  // Cross-word injection bit for left/right shifts.

// 3:2 compressor / carry-save adder.
// For three 1-bit planes a,b,c it produces:
//   value = sum + 2 * carry
// lane-wise and independently for all 64 bits.
// This boolean pattern maps well to modern NVIDIA ternary-logic hardware.
__device__ __forceinline__ void csa3(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (c & axb);
}

// Build the "left-neighbor" bitboard for the 64 cells stored in center_word.
// Bit i of the result becomes cell (i-1) from center_word, except result bit 0,
// which receives bit 63 from left_word. This is exactly the special handling
// needed for the 0th bit across the three words to the left (upper/current/lower).
__device__ __forceinline__ u64 align_left_neighbors(u64 left_word, u64 center_word) {
    return (center_word << 1) | (left_word >> kEdgeBit);
}

// Symmetric operation for the right neighbors: result bit 63 receives bit 0
// from right_word, which handles the 63rd-bit cross-word case.
__device__ __forceinline__ u64 align_right_neighbors(u64 center_word, u64 right_word) {
    return (center_word >> 1) | (right_word << kEdgeBit);
}

// Advance one 64-cell word.
// uL/uC/uR = words from the row above
// mL/mC/mR = words from the current row
// bL/bC/bR = words from the row below
//
// The eight neighbor planes are reduced with a carry-save tree into:
//   ones = neighbor_count bit 0
//   twos = neighbor_count bit 1
//   ge4  = neighbor_count >= 4
//
// The Game-of-Life rule
//   next = (count == 3) | (alive & count == 2)
// simplifies to
//   next = twos & !ge4 & (ones | alive)
__device__ __forceinline__ u64 evolve_word(
    u64 uL, u64 uC, u64 uR,
    u64 mL, u64 mC, u64 mR,
    u64 bL, u64 bC, u64 bR) {

    u64 s1, c1;
    {
        const u64 top_left  = align_left_neighbors(uL, uC);
        const u64 top       = uC;
        const u64 top_right = align_right_neighbors(uC, uR);
        csa3(top_left, top, top_right, s1, c1);
    }

    u64 s2, c2;
    {
        const u64 left        = align_left_neighbors(mL, mC);
        const u64 right       = align_right_neighbors(mC, mR);
        const u64 bottom_left = align_left_neighbors(bL, bC);
        csa3(left, right, bottom_left, s2, c2);
    }

    u64 s3, c3;
    csa3(s1, s2, bC, s3, c3);

    const u64 bottom_right = align_right_neighbors(bC, bR);
    const u64 ones         = s3 ^ bottom_right;
    const u64 c4           = s3 & bottom_right;

    u64 s4, c5;
    csa3(c1, c2, c3, s4, c5);

    const u64 twos = s4 ^ c4;
    const u64 ge4  = c5 | (s4 & c4);

    return twos & ~ge4 & (ones | mC);
}

// Exact tiling of the full word lattice with a 1D block grid.
// We keep a 2D thread block so warps still traverse contiguous words within a row,
// but flatten the grid to 1D to avoid CUDA's 65535 limit on grid.y.
// Because both the board dimensions and the block geometry are powers of two,
// blockIdx.x can be split back into (tile_x, tile_y) with a mask and a shift.
template <int BLOCK_X_SHIFT, int BLOCK_Y_SHIFT>
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_interior_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int words_per_row,
    int row_shift,
    int grid_dimensions,
    int tiles_x_mask,
    int tiles_x_shift) {

    constexpr int BLOCK_X = 1 << BLOCK_X_SHIFT;
    constexpr int BLOCK_Y = 1 << BLOCK_Y_SHIFT;
    (void)BLOCK_X;
    (void)BLOCK_Y;

    const int tile   = static_cast<int>(blockIdx.x);
    const int tile_x = tile & tiles_x_mask;
    const int tile_y = tile >> tiles_x_shift;

    const int x = (tile_x << BLOCK_X_SHIFT) + static_cast<int>(threadIdx.x);
    const int y = (tile_y << BLOCK_Y_SHIFT) + static_cast<int>(threadIdx.y);

    const int last_word = words_per_row - 1;
    const int last_row  = grid_dimensions - 1;

    // The boundary is handled by a tiny second kernel so the hot path below can
    // assume that all eight neighboring words exist.
    if (x == 0 || x == last_word || y == 0 || y == last_row) {
        return;
    }

    const usize pitch = static_cast<usize>(words_per_row);
    const usize base  = (static_cast<usize>(y) << row_shift) + static_cast<usize>(x);

    const u64* mid  = input + base;
    const u64* up   = mid - pitch;
    const u64* down = mid + pitch;

    const u64 uL = up[-1];
    const u64 uC = up[0];
    const u64 uR = up[1];

    const u64 mL = mid[-1];
    const u64 mC = mid[0];
    const u64 mR = mid[1];

    const u64 bL = down[-1];
    const u64 bC = down[0];
    const u64 bR = down[1];

    output[base] = evolve_word(uL, uC, uR, mL, mC, mR, bL, bC, bR);
}

// Separate boundary pass.
// Grouping boundary words as top, bottom, left, right keeps most warps within a
// single boundary class, which minimizes divergence inside this already-small kernel.
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_boundary_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int words_per_row,
    int row_shift,
    int grid_dimensions) {

    const int tid = static_cast<int>(blockIdx.x) * kThreadsPerBlock + static_cast<int>(threadIdx.x);

    const int side_rows      = grid_dimensions - 2;
    const int top_end        = words_per_row;
    const int bottom_end     = top_end + words_per_row;
    const int left_end       = bottom_end + side_rows;
    const int boundary_words = left_end + side_rows;

    if (tid >= boundary_words) {
        return;
    }

    const int last_word = words_per_row - 1;
    const int last_row  = grid_dimensions - 1;

    int x;
    int y;

    if (tid < top_end) {
        y = 0;
        x = tid;
    } else if (tid < bottom_end) {
        y = last_row;
        x = tid - top_end;
    } else if (tid < left_end) {
        y = 1 + (tid - bottom_end);
        x = 0;
    } else {
        y = 1 + (tid - left_end);
        x = last_word;
    }

    const bool has_up    = (y != 0);
    const bool has_down  = (y != last_row);
    const bool has_left  = (x != 0);
    const bool has_right = (x != last_word);

    const usize pitch = static_cast<usize>(words_per_row);
    const usize base  = (static_cast<usize>(y) << row_shift) + static_cast<usize>(x);

    const u64* mid = input + base;

    u64 uL = 0, uC = 0, uR = 0;
    u64 mL = 0, mC = mid[0], mR = 0;
    u64 bL = 0, bC = 0, bR = 0;

    if (has_up) {
        const u64* up = mid - pitch;
        uC = up[0];
        if (has_left)  uL = up[-1];
        if (has_right) uR = up[1];
    }

    if (has_left)  mL = mid[-1];
    if (has_right) mR = mid[1];

    if (has_down) {
        const u64* down = mid + pitch;
        bC = down[0];
        if (has_left)  bL = down[-1];
        if (has_right) bR = down[1];
    }

    output[base] = evolve_word(uL, uC, uR, mL, mC, mR, bL, bC, bR);
}

// Host-side helper: input is guaranteed to be a power of two.
inline int log2_power_of_two(unsigned value) {
    int shift = 0;
    while ((1u << shift) < value) {
        ++shift;
    }
    return shift;
}

template <int BLOCK_X_SHIFT, int BLOCK_Y_SHIFT>
inline void launch_interior_kernel(
    const u64* input,
    u64* output,
    int words_per_row,
    int row_shift,
    int grid_dimensions) {

    constexpr int BLOCK_X = 1 << BLOCK_X_SHIFT;
    constexpr int BLOCK_Y = 1 << BLOCK_Y_SHIFT;
    static_assert(BLOCK_X * BLOCK_Y == kThreadsPerBlock, "interior block must have 256 threads");

    const int tiles_x       = words_per_row >> BLOCK_X_SHIFT;
    const int tiles_y       = grid_dimensions >> BLOCK_Y_SHIFT;
    const int tiles_x_mask  = tiles_x - 1;
    const int tiles_x_shift = log2_power_of_two(static_cast<unsigned>(tiles_x));

    const unsigned int blocks = static_cast<unsigned int>(tiles_x) * static_cast<unsigned int>(tiles_y);
    const dim3 grid(blocks, 1, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    game_of_life_interior_kernel<BLOCK_X_SHIFT, BLOCK_Y_SHIFT>
        <<<grid, block>>>(input, output, words_per_row, row_shift, grid_dimensions, tiles_x_mask, tiles_x_shift);
}

}  // namespace detail

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Out-of-place update: input and output are assumed to be different device buffers.
    // The caller is responsible for any synchronization/error handling around this launch.
    const int words_per_row = grid_dimensions >> detail::kWordShift;
    const int row_shift     = detail::log2_power_of_two(static_cast<unsigned>(words_per_row));

    // Keep 256 threads/block but adapt the x dimension to the row width so that,
    // whenever possible, each warp owns a contiguous run of words within a row:
    //   16 words/row  -> 16x16
    //   32 words/row  -> 32x8
    //   64+ words/row -> 64x4
    if (words_per_row >= 64) {
        detail::launch_interior_kernel<6, 2>(input, output, words_per_row, row_shift, grid_dimensions);
    } else if (words_per_row >= 32) {
        detail::launch_interior_kernel<5, 3>(input, output, words_per_row, row_shift, grid_dimensions);
    } else {
        detail::launch_interior_kernel<4, 4>(input, output, words_per_row, row_shift, grid_dimensions);
    }

    // Perimeter in words, with corners counted only once:
    //   top row + bottom row + left side (without corners) + right side (without corners)
    const int boundary_words = 2 * (words_per_row + grid_dimensions - 2);
    if (boundary_words > 0) {
        const unsigned int boundary_blocks =
            static_cast<unsigned int>((boundary_words + detail::kThreadsPerBlock - 1) / detail::kThreadsPerBlock);
        detail::game_of_life_boundary_kernel
            <<<boundary_blocks, detail::kThreadsPerBlock>>>(input, output, words_per_row, row_shift, grid_dimensions);
    }
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// This label is used by the framework to identify the memory layout of the input and output arrays.
// MEMORY_LAYOUT: ROWS
