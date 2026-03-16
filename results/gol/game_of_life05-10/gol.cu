#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/*
  High-throughput Conway's Game of Life for a square, bit-packed grid.

  Layout:
    - Row-major.
    - Each 64-bit word stores 64 consecutive cells from one row.
    - Bit 0 is the leftmost cell in that 64-cell run; bit 63 is the rightmost.

  Performance strategy:
    - One thread updates one 64-cell word.
    - The hot kernel covers only interior rows. blockDim.x is fixed to 32 so one warp spans
      32 consecutive words from a single row.
    - Each lane loads only the center word from the row above/current/below.
      Horizontal neighbor words are obtained with warp shuffles; only the first and last
      active lanes fetch a halo word from global memory.
    - The 64 cells inside a word are updated simultaneously with bit-sliced logic. We never
      iterate over cells and we never need atomics.
    - A tiny second kernel handles the first and last rows, keeping the main kernel free of
      top/bottom boundary checks.

  Boundary rule:
    - Cells outside the grid are dead. Missing rows/columns are therefore represented by zero words.
*/

namespace {

using u64 = std::uint64_t;

constexpr int kInteriorBlockX = 32;   // One warp spans one horizontal run of words.
constexpr int kInteriorBlockY = 8;    // 8 warps/block -> 256 threads/block.
constexpr int kBoundaryBlock  = 256;
constexpr int kMaxGridY       = 65535;  // Architectural limit for 2D CUDA launches.

static_assert(kInteriorBlockX == 32, "The interior kernel relies on one warp spanning one row segment.");
static_assert(sizeof(u64) == 8, "This implementation requires 64-bit words.");

// CUDA shuffle intrinsics have guaranteed overloads for unsigned long long.
// Casting keeps this helper portable even if std::uint64_t is typedef'd to another 64-bit unsigned type.
__device__ __forceinline__ u64 shfl_up_u64(unsigned mask, u64 value, unsigned delta) {
    return static_cast<u64>(__shfl_up_sync(mask, static_cast<unsigned long long>(value), delta));
}

__device__ __forceinline__ u64 shfl_down_u64(unsigned mask, u64 value, unsigned delta) {
    return static_cast<u64>(__shfl_down_sync(mask, static_cast<unsigned long long>(value), delta));
}

// Align the x-1 neighbors of all 64 cells in one operation.
// For bit i>0 the source is bit i-1 of center_word.
// For bit 0 the source is bit 63 of left_word.
// If the current word is on the global left edge, the caller passes left_word = 0.
__device__ __forceinline__ u64 neighbors_from_left(u64 center_word, u64 left_word) {
    return (center_word << 1) | (left_word >> 63);
}

// Align the x+1 neighbors of all 64 cells in one operation.
// For bit i<63 the source is bit i+1 of center_word.
// For bit 63 the source is bit 0 of right_word.
// If the current word is on the global right edge, the caller passes right_word = 0.
__device__ __forceinline__ u64 neighbors_from_right(u64 center_word, u64 right_word) {
    return (center_word >> 1) | (right_word << 63);
}

// Compute the next-state word for 64 cells at once from the surrounding 3x3 block of words.
// The current-center word (mc) is *not* included in the neighbor count; it is only used in the final rule.
__device__ __forceinline__ u64 life_word(
    u64 ul, u64 uc, u64 ur,
    u64 ml, u64 mc, u64 mr,
    u64 dl, u64 dc, u64 dr)
{
    // Align all eight neighbor directions to the bit positions of the current word.
    const u64 up_left    = neighbors_from_left (uc, ul);
    const u64 up_center  = uc;
    const u64 up_right   = neighbors_from_right(uc, ur);

    const u64 mid_left   = neighbors_from_left (mc, ml);
    const u64 mid_right  = neighbors_from_right(mc, mr);

    const u64 dn_left    = neighbors_from_left (dc, dl);
    const u64 dn_center  = dc;
    const u64 dn_right   = neighbors_from_right(dc, dr);

    // Horizontal row sums, computed bit-sliced:
    //   top_count    = top_lo + 2*top_hi    for (up_left + up_center + up_right), range 0..3
    //   middle_count = mid_lo + 2*mid_hi    for (mid_left + mid_right),            range 0..2
    //   bottom_count = bot_lo + 2*bot_hi    for (dn_left + dn_center + dn_right),  range 0..3
    u64 x = up_left ^ up_center;
    const u64 top_lo = x ^ up_right;
    const u64 top_hi = (up_left & up_center) | (up_right & x);

    const u64 mid_lo = mid_left ^ mid_right;
    const u64 mid_hi = mid_left & mid_right;

    x = dn_left ^ dn_center;
    const u64 bot_lo = x ^ dn_right;
    const u64 bot_hi = (dn_left & dn_center) | (dn_right & x);

    // Add the three row-sum low bits:
    //   total_neighbors = count_bit0 + 2*(top_hi + mid_hi + bot_hi + carry_2)
    x = top_lo ^ mid_lo;
    const u64 count_bit0 = x ^ bot_lo;
    const u64 carry_2    = (top_lo & mid_lo) | (bot_lo & x);

    // We do not need the full 4-bit neighbor count.
    // The Life rule only cares about counts 2 and 3, which means:
    //   top_hi + mid_hi + bot_hi + carry_2 must equal exactly 1.
    //
    // Pairwise reduction of those four "2's-place" bits:
    //   pair_a_lo/pair_a_hi encode top_hi + mid_hi
    //   pair_b_lo/pair_b_hi encode bot_hi + carry_2
    //
    // count_bit1 is the low bit of that 4-way sum.
    // ge4 is true when that 4-way sum is >= 2, which means the full neighbor count is >= 4.
    const u64 pair_a_lo = top_hi ^ mid_hi;
    const u64 pair_a_hi = top_hi & mid_hi;
    const u64 pair_b_lo = bot_hi ^ carry_2;
    const u64 pair_b_hi = bot_hi & carry_2;

    const u64 count_bit1 = pair_a_lo ^ pair_b_lo;
    const u64 ge4        = pair_a_hi | pair_b_hi | (pair_a_lo & pair_b_lo);

    // count == 2 or 3  <=>  count_bit1 == 1 and ge4 == 0
    const u64 count_is_2_or_3 = count_bit1 & ~ge4;

    // Final Life rule:
    //   next = (count == 3) | (alive & (count == 2))
    //
    // For counts 2/3, count_bit0 distinguishes them:
    //   count == 2 -> count_bit0 = 0, so survival requires mc = 1
    //   count == 3 -> count_bit0 = 1, so birth/survival is unconditional
    return count_is_2_or_3 & (count_bit0 | mc);
}

// Hot path: rows 1..N-2 only.
// The top and bottom neighbor rows always exist here, so the kernel has no row-boundary checks.
// Horizontal boundary columns are still handled by zero halo words on the first/last word of each row.
__global__ __launch_bounds__(kInteriorBlockX * kInteriorBlockY)
void game_of_life_interior_rows_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int grid_dimensions,
    int words_per_row)
{
    const int last_row = grid_dimensions - 1;
    const int row_start = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (row_start >= last_row) {
        return;
    }

    // blockDim.x is exactly one warp, so threadIdx.x is also the lane id.
    const int lane = threadIdx.x;
    const int col  = blockIdx.x * blockDim.x + lane;
    if (col >= words_per_row) {
        return;
    }

    // Only the high lanes of the final x-block can be inactive, so the active mask is always
    // a contiguous prefix. Therefore only lane 0 and the last active lane need explicit halo loads.
    const unsigned active_mask   = __activemask();
    const int last_active_lane   = 31 - __clz(active_mask);
    const bool fetch_left_halo   = (lane == 0);
    const bool fetch_right_halo  = (lane == last_active_lane);

    const int last_col           = words_per_row - 1;
    const bool has_left_col      = (col != 0);
    const bool has_right_col     = (col != last_col);

    const std::size_t stride_words = static_cast<std::size_t>(words_per_row);
    const int row_step             = blockDim.y * gridDim.y;
    const std::size_t base_step    = static_cast<std::size_t>(row_step) * stride_words;

    std::size_t base =
        static_cast<std::size_t>(row_start) * stride_words +
        static_cast<std::size_t>(col);

    for (int row = row_start; row < last_row; row += row_step, base += base_step) {
        const u64* above = input + (base - stride_words);
        const u64* mid   = input +  base;
        const u64* below = input + (base + stride_words);

        // Center-column words for the three relevant rows.
        const u64 uc = above[0];
        const u64 mc = mid[0];
        const u64 dc = below[0];

        // Most horizontal neighbors come from adjacent lanes instead of global memory.
        u64 ul = shfl_up_u64  (active_mask, uc, 1);
        u64 ml = shfl_up_u64  (active_mask, mc, 1);
        u64 dl = shfl_up_u64  (active_mask, dc, 1);

        u64 ur = shfl_down_u64(active_mask, uc, 1);
        u64 mr = shfl_down_u64(active_mask, mc, 1);
        u64 dr = shfl_down_u64(active_mask, dc, 1);

        // Only the warp-edge lanes need explicit horizontal halo loads.
        if (fetch_left_halo) {
            ul = 0;
            ml = 0;
            dl = 0;
            if (has_left_col) {
                ul = above[-1];
                ml = mid[-1];
                dl = below[-1];
            }
        }

        if (fetch_right_halo) {
            ur = 0;
            mr = 0;
            dr = 0;
            if (has_right_col) {
                ur = above[1];
                mr = mid[1];
                dr = below[1];
            }
        }

        output[base] = life_word(ul, uc, ur, ml, mc, mr, dl, dc, dr);
    }
}

// Tiny boundary-only kernel for row 0 and row N-1.
// These rows are a negligible fraction of the work, so simple direct global loads are preferable
// to complicating the main interior kernel.
__global__ __launch_bounds__(kBoundaryBlock)
void game_of_life_boundary_rows_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int grid_dimensions,
    int words_per_row)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= words_per_row) {
        return;
    }

    const int last_col = words_per_row - 1;
    const bool has_left_col  = (col != 0);
    const bool has_right_col = (col != last_col);

    const std::size_t stride_words = static_cast<std::size_t>(words_per_row);
    const int last_row             = grid_dimensions - 1;
    const std::size_t bottom_base  = static_cast<std::size_t>(last_row) * stride_words;

    const u64* top    = input;
    const u64* second = input + stride_words;
    const u64* penult = input + (bottom_base - stride_words);
    const u64* bottom = input + bottom_base;

    // Top row: above-grid words are dead (zero).
    u64 top_ml = 0;
    u64 top_mr = 0;
    u64 top_dl = 0;
    u64 top_dr = 0;

    if (has_left_col) {
        top_ml = top[col - 1];
        top_dl = second[col - 1];
    }
    if (has_right_col) {
        top_mr = top[col + 1];
        top_dr = second[col + 1];
    }

    output[col] = life_word(
        0, 0, 0,
        top_ml, top[col], top_mr,
        top_dl, second[col], top_dr);

    // Bottom row: below-grid words are dead (zero).
    u64 bot_ul = 0;
    u64 bot_ur = 0;
    u64 bot_ml = 0;
    u64 bot_mr = 0;

    if (has_left_col) {
        bot_ul = penult[col - 1];
        bot_ml = bottom[col - 1];
    }
    if (has_right_col) {
        bot_ur = penult[col + 1];
        bot_mr = bottom[col + 1];
    }

    output[bottom_base + static_cast<std::size_t>(col)] = life_word(
        bot_ul, penult[col], bot_ur,
        bot_ml, bottom[col], bot_mr,
        0, 0, 0);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller manages synchronization; this function only enqueues kernels.
    // The usual double-buffered usage is assumed: input and output should not alias.

    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions is a power of two, so this is exact.
    const int interior_rows = grid_dimensions - 2;

    const dim3 interior_block(kInteriorBlockX, kInteriorBlockY);
    const int interior_grid_x = (words_per_row + kInteriorBlockX - 1) / kInteriorBlockX;
    const int raw_grid_y      = (interior_rows + kInteriorBlockY - 1) / kInteriorBlockY;
    const int interior_grid_y = (raw_grid_y < kMaxGridY) ? raw_grid_y : kMaxGridY;

    game_of_life_interior_rows_kernel<<<dim3(interior_grid_x, interior_grid_y), interior_block>>>(
        input, output, grid_dimensions, words_per_row);

    const int boundary_grid_x = (words_per_row + kBoundaryBlock - 1) / kBoundaryBlock;
    game_of_life_boundary_rows_kernel<<<boundary_grid_x, kBoundaryBlock>>>(
        input, output, grid_dimensions, words_per_row);
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
