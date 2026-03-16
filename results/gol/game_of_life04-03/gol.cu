#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kCellsPerWordShift = 6;  // 2^6 == 64 cells per packed word.
constexpr int kThreadsPerBlock   = 256;

// The public API uses std::uint64_t exactly as requested. Internally we use
// unsigned long long for the bitwise/shuffle math because CUDA explicitly
// guarantees shuffle support for that 64-bit type.
using storage_t = std::uint64_t;
using word_t    = unsigned long long;

__device__ __forceinline__ word_t xor3(const word_t a, const word_t b, const word_t c) {
    return a ^ b ^ c;
}

__device__ __forceinline__ word_t maj3(const word_t a, const word_t b, const word_t c) {
    return (a & b) | (a & c) | (b & c);
}

// Host-side setup cost is not performance-critical for this task, so a tiny
// portable helper loop is fine here.
inline int log2_pow2(unsigned int x) {
    int shift = 0;
    while (x > 1u) {
        x >>= 1u;
        ++shift;
    }
    return shift;
}

/*
 * Each thread updates one 64-bit packed word (64 cells).
 *
 * A straightforward "one thread per word" implementation still loops over the
 * 64 bit positions and uses masked popcount operations for each cell. The prompt
 * mentions popcount as an improvement over naive masking, but on modern NVIDIA
 * GPUs it is still slower than treating the 64 bits of a word as 64 SIMD lanes
 * and updating all of them at once.
 *
 * This kernel therefore:
 *   1) Loads the current word plus the vertical neighbors (row above/below).
 *   2) Uses warp shuffles to obtain the left/right words for those three rows.
 *      Only subgroup-boundary threads need fallback global loads. This removes
 *      the six extra global loads that a naive per-thread halo load would do.
 *   3) Builds the eight aligned neighbor bitboards (NW, N, NE, W, E, SW, S, SE).
 *   4) Adds those eight 1-bit bitboards with a carry-save / bit-sliced boolean
 *      network, producing the 1's, 2's, and 4's place of the neighbor count for
 *      all 64 cells in parallel.
 *   5) Applies the Life rule with pure bitwise logic.
 *
 * BLOCK_X is either:
 *   - 32 (kernel shape 32x8): one warp == one 32-word row segment.
 *   - 16 (kernel shape 16x16): width-16 shuffles split each warp into two
 *     independent 16-lane groups, which exactly match the smallest legal board
 *     width (1024 cells => 16 packed words per row).
 *
 * A 1D grid of 2D blocks is used so very tall boards do not hit grid.y limits.
 * Because the board width is a power-of-two and BLOCK_X is a power-of-two,
 * blocks_per_row is also a power-of-two. The kernel reconstructs
 * (block_row, block_col) from blockIdx.x with a shift and a mask instead of
 * division and modulo.
 */
template <int BLOCK_X, int BLOCK_Y>
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(const storage_t* __restrict__ input,
                         storage_t* __restrict__ output,
                         int grid_dimensions,
                         int words_per_row,
                         int blocks_per_row_shift,
                         int blocks_per_row_mask)
{
    static_assert(BLOCK_X * BLOCK_Y == kThreadsPerBlock, "Kernel tuned for 256-thread blocks.");
    static_assert(BLOCK_X == 16 || BLOCK_X == 32, "Unsupported block width.");

    const unsigned int block_linear = blockIdx.x;
    const int block_row = static_cast<int>(block_linear >> blocks_per_row_shift);
    const int block_col = static_cast<int>(block_linear & static_cast<unsigned int>(blocks_per_row_mask));

    const int row = block_row * BLOCK_Y + static_cast<int>(threadIdx.y);
    const int col = block_col * BLOCK_X + static_cast<int>(threadIdx.x);

    const std::size_t stride = static_cast<std::size_t>(words_per_row);
    const std::size_t idx = static_cast<std::size_t>(row) * stride + static_cast<std::size_t>(col);

    const int last_row = grid_dimensions - 1;
    const bool has_up   = (row != 0);
    const bool has_down = (row != last_row);

    const std::size_t idx_up   = idx - stride;
    const std::size_t idx_down = idx + stride;

    // Common-case memory traffic per thread is only three loads: current row,
    // row above, and row below.
    const word_t center = static_cast<word_t>(input[idx]);
    const word_t up     = has_up   ? static_cast<word_t>(input[idx_up])   : 0ULL;
    const word_t down   = has_down ? static_cast<word_t>(input[idx_down]) : 0ULL;

    // Exchange the three loaded words horizontally inside the subgroup.
    // width == BLOCK_X keeps the 16-wide special case confined to independent
    // half-warps and the 32-wide case as a full warp.
    const unsigned int shuffle_mask = 0xffffffffu;
    word_t left_center  = __shfl_up_sync  (shuffle_mask, center, 1, BLOCK_X);
    word_t right_center = __shfl_down_sync(shuffle_mask, center, 1, BLOCK_X);
    word_t left_up      = __shfl_up_sync  (shuffle_mask, up,     1, BLOCK_X);
    word_t right_up     = __shfl_down_sync(shuffle_mask, up,     1, BLOCK_X);
    word_t left_down    = __shfl_up_sync  (shuffle_mask, down,   1, BLOCK_X);
    word_t right_down   = __shfl_down_sync(shuffle_mask, down,   1, BLOCK_X);

    const int lane = static_cast<int>(threadIdx.x);

    if (BLOCK_X == 16) {
        // This specialization is only launched when words_per_row == 16, so each
        // 16-lane subgroup spans the full row and its horizontal halo is outside
        // the board.
        if (lane == 0) {
            left_center = 0ULL;
            left_up     = 0ULL;
            left_down   = 0ULL;
        }
        if (lane == 15) {
            right_center = 0ULL;
            right_up     = 0ULL;
            right_down   = 0ULL;
        }
    } else {
        // For the 32x8 kernel, subgroup boundaries may lie inside a row. Only
        // the boundary threads need fallback global loads to cross from one
        // 32-word segment to the next.
        const int last_col = words_per_row - 1;
        const bool has_left  = (col != 0);
        const bool has_right = (col != last_col);

        if (lane == 0) {
            left_center = has_left ? static_cast<word_t>(input[idx - 1]) : 0ULL;
            left_up     = (has_up   && has_left) ? static_cast<word_t>(input[idx_up   - 1]) : 0ULL;
            left_down   = (has_down && has_left) ? static_cast<word_t>(input[idx_down - 1]) : 0ULL;
        }
        if (lane == 31) {
            right_center = has_right ? static_cast<word_t>(input[idx + 1]) : 0ULL;
            right_up     = (has_up   && has_right) ? static_cast<word_t>(input[idx_up   + 1]) : 0ULL;
            right_down   = (has_down && has_right) ? static_cast<word_t>(input[idx_down + 1]) : 0ULL;
        }
    }

    // Align the eight neighbor directions so bit i corresponds to the i-th cell
    // in 'center'. The 0th and 63rd bits automatically pick up their halo bits
    // from the left/right words handled above.
    const word_t nw   = (up     << 1) | (left_up      >> 63);
    const word_t ne   = (up     >> 1) | (right_up     << 63);
    const word_t west = (center << 1) | (left_center  >> 63);
    const word_t east = (center >> 1) | (right_center << 63);
    const word_t sw   = (down   << 1) | (left_down    >> 63);
    const word_t se   = (down   >> 1) | (right_down   << 63);

    // Add the eight 1-bit neighbor bitboards in parallel across all 64 lanes.
    // We keep only the 1's, 2's, and 4's place of the count. Count==8 would
    // require the 8's place, but it has the 2's place clear, so it is already
    // rejected by the final Life rule expression.
    const word_t p0 = nw ^ up;
    const word_t q0 = nw & up;
    const word_t p1 = ne ^ west;
    const word_t q1 = ne & west;

    const word_t s0 = p0 ^ p1;
    const word_t k0 = p0 & p1;
    const word_t t0 = xor3(q0, q1, k0);
    const word_t f0 = maj3(q0, q1, k0);

    const word_t p2 = east ^ sw;
    const word_t q2 = east & sw;
    const word_t p3 = down ^ se;
    const word_t q3 = down & se;

    const word_t s1 = p2 ^ p3;
    const word_t k1 = p2 & p3;
    const word_t t1 = xor3(q2, q3, k1);
    const word_t f1 = maj3(q2, q3, k1);

    // cnt1/cnt2/cnt4 are the 1's, 2's, and 4's place of the neighbor count.
    const word_t cnt1 = s0 ^ s1;
    const word_t k2   = s0 & s1;
    const word_t cnt2 = xor3(t0, t1, k2);
    const word_t k3   = maj3(t0, t1, k2);
    const word_t cnt4 = xor3(f0, f1, k3);

    // Conway's Game of Life:
    //   next = (count == 3) | (alive & count == 2)
    //
    // In terms of the count bitplanes:
    //   count in {2,3} <=> cnt2 == 1 and cnt4 == 0
    //   among those, count == 3 iff cnt1 == 1
    //
    // Therefore:
    //   next = (cnt2 & ~cnt4) & (cnt1 | center)
    const word_t next = (cnt2 & ~cnt4) & (cnt1 | center);
    output[idx] = static_cast<storage_t>(next);
}

template <int BLOCK_X, int BLOCK_Y>
inline void launch_game_of_life_kernel(const storage_t* input,
                                       storage_t* output,
                                       int grid_dimensions,
                                       int words_per_row)
{
    static_assert(BLOCK_X * BLOCK_Y == kThreadsPerBlock, "Kernel tuned for 256-thread blocks.");

    // Given the problem constraints:
    //   - grid_dimensions is a power-of-two
    //   - words_per_row = grid_dimensions / 64 is therefore a power-of-two
    //   - BLOCK_X and BLOCK_Y are powers-of-two
    // Hence blocks_per_row and block_rows are exact powers-of-two as well.
    const int blocks_per_row = words_per_row / BLOCK_X;
    const int block_rows = grid_dimensions / BLOCK_Y;
    const int blocks_per_row_shift = log2_pow2(static_cast<unsigned int>(blocks_per_row));
    const int blocks_per_row_mask = blocks_per_row - 1;

    const unsigned int total_blocks = static_cast<unsigned int>(
        static_cast<std::uint64_t>(blocks_per_row) *
        static_cast<std::uint64_t>(block_rows));

    const dim3 grid(total_blocks, 1, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    game_of_life_kernel<BLOCK_X, BLOCK_Y>
        <<<grid, block>>>(input,
                          output,
                          grid_dimensions,
                          words_per_row,
                          blocks_per_row_shift,
                          blocks_per_row_mask);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // One 64-bit word stores 64 consecutive cells from a single row.
    const int words_per_row = grid_dimensions >> kCellsPerWordShift;

    // No synchronization or error handling here by design: the caller owns
    // synchronization, and only kernel performance matters for this task.
    if (words_per_row == 16) {
        // Smallest legal board: 1024x1024 cells => 16 packed words per row.
        // 16x16 keeps the block fully utilized and uses width-16 shuffles.
        launch_game_of_life_kernel<16, 16>(input, output, grid_dimensions, words_per_row);
    } else {
        // All larger legal boards have at least 32 packed words per row.
        // 32x8 makes one warp map cleanly to one 32-word row segment.
        launch_game_of_life_kernel<32, 8>(input, output, grid_dimensions, words_per_row);
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
