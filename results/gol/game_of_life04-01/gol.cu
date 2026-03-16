#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace
{
using u64 = std::uint64_t;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;
constexpr int kCellsPerWord = 64;

/*
This kernel is optimized for the problem's actual performance-critical structure:

- The board is already bit-packed, so one thread updates one 64-bit word = 64 cells.
- A per-bit implementation can be improved with __popc/__popcll, but it still performs
  64 neighborhood counts per thread. On modern NVIDIA data-center GPUs, it is faster to
  keep the data packed and evaluate all 64 cells in parallel with bit-sliced logic.
- Shared memory and texture memory are intentionally avoided. The access pattern is regular,
  adjacent words are naturally exchanged with warp/half-warp shuffles, and the remaining
  loads are straightforward global reads.

The adder tree below computes the 8-neighbor count for all 64 cells at once:
  north trio  -> s0 + 2*c0
  south trio  -> s1 + 2*c1
  west+east   -> s2 + 2*c2

Then:
  total_neighbors = s0+s1+s2 + 2*(c0+c1+c2)
                  = ones + 2*(twos_a + sum_twos + 2*carry_twos)

Life needs total_neighbors == 2 or 3.
That means the parenthesized term must be exactly 1, and the low bit (ones) determines
whether it is 2 (ones=0) or 3 (ones=1). The final rule therefore becomes:
  next = exact_two_or_three & (ones | current)
where exact_two_or_three = ((twos_a ^ sum_twos) & ~carry_twos).
*/

// Carry-save adder over 64 independent 1-bit lanes.
// For each bit position i:
//   a_i + b_i + c_i = sum_i + 2 * carry_i
__device__ __forceinline__ void csa64(const u64 a, const u64 b, const u64 c, u64& sum, u64& carry)
{
    const u64 ab = a ^ b;
    sum = ab ^ c;
    carry = (a & b) | (ab & c);
}

// Finish the Life rule once the neighborhood has been reduced to the three partial counters:
//   north = s0 + 2*c0
//   south = s1 + 2*c1
//   horiz = s2 + 2*c2
__device__ __forceinline__ u64 finish_word(const u64 current,
                                           const u64 s0, const u64 c0,
                                           const u64 s1, const u64 c1,
                                           const u64 s2, const u64 c2)
{
    u64 ones, twos_a;
    csa64(s0, s1, s2, ones, twos_a);

    u64 sum_twos, carry_twos;
    csa64(c0, c1, c2, sum_twos, carry_twos);

    const u64 exact_two_or_three = (twos_a ^ sum_twos) & ~carry_twos;
    return exact_two_or_three & (ones | current);
}

/*
Interior path:
- No boundary checks.
- Adjacent packed words are obtained from neighboring lanes with __shfl*_sync.
- BLOCK_X is both the block width and the shuffle subgroup width:
    * 32 -> one full warp per row segment
    * 16 -> one half-warp per row segment
  This keeps left/right exchanges within a single row segment.
*/
template <int BLOCK_X>
__device__ __forceinline__ u64 evolve_shfl_interior(const u64* __restrict__ input,
                                                    std::size_t idx,
                                                    int words_per_row)
{
    static_assert(BLOCK_X == 16 || BLOCK_X == 32, "BLOCK_X must be 16 or 32");

    const u64* __restrict__ p = input + idx;
    const int stride = words_per_row;
    const int lane_in_row = static_cast<int>(threadIdx.x);

    const u64 current = p[0];

    u64 s2, c2;
    {
        u64 left_word  = __shfl_up_sync(kFullWarpMask, current, 1, BLOCK_X);
        u64 right_word = __shfl_down_sync(kFullWarpMask, current, 1, BLOCK_X);

        if (lane_in_row == 0) {
            left_word = p[-1];
        }
        if (lane_in_row == BLOCK_X - 1) {
            right_word = p[1];
        }

        const u64 west = (current << 1) | (left_word >> 63);
        const u64 east = (current >> 1) | (right_word << 63);

        s2 = west ^ east;
        c2 = west & east;
    }

    u64 s0, c0;
    {
        const u64 up = p[-stride];

        u64 up_left_word  = __shfl_up_sync(kFullWarpMask, up, 1, BLOCK_X);
        u64 up_right_word = __shfl_down_sync(kFullWarpMask, up, 1, BLOCK_X);

        if (lane_in_row == 0) {
            up_left_word = p[-stride - 1];
        }
        if (lane_in_row == BLOCK_X - 1) {
            up_right_word = p[-stride + 1];
        }

        const u64 north_west = (up << 1) | (up_left_word >> 63);
        const u64 north_east = (up >> 1) | (up_right_word << 63);

        csa64(north_west, up, north_east, s0, c0);
    }

    u64 s1, c1;
    {
        const u64 down = p[stride];

        u64 down_left_word  = __shfl_up_sync(kFullWarpMask, down, 1, BLOCK_X);
        u64 down_right_word = __shfl_down_sync(kFullWarpMask, down, 1, BLOCK_X);

        if (lane_in_row == 0) {
            down_left_word = p[stride - 1];
        }
        if (lane_in_row == BLOCK_X - 1) {
            down_right_word = p[stride + 1];
        }

        const u64 south_west = (down << 1) | (down_left_word >> 63);
        const u64 south_east = (down >> 1) | (down_right_word << 63);

        csa64(south_west, down, south_east, s1, c1);
    }

    return finish_word(current, s0, c0, s1, c1, s2, c2);
}

/*
Boundary path:
- Same shuffle-based structure as the interior path.
- Missing words outside the board are replaced with zero, implementing the required
  "outside the grid is dead" semantics.
- Only boundary blocks take this path, so a few extra predicates are acceptable.
*/
template <int BLOCK_X>
__device__ __forceinline__ u64 evolve_shfl_boundary(const u64* __restrict__ input,
                                                    std::size_t idx,
                                                    int row,
                                                    int col,
                                                    int grid_dimensions,
                                                    int words_per_row)
{
    static_assert(BLOCK_X == 16 || BLOCK_X == 32, "BLOCK_X must be 16 or 32");

    const u64* __restrict__ p = input + idx;
    const int stride = words_per_row;
    const int lane_in_row = static_cast<int>(threadIdx.x);

    const int last_row  = grid_dimensions - 1;
    const int last_word = words_per_row - 1;

    const bool has_up   = (row != 0);
    const bool has_down = (row != last_row);

    const u64 current = p[0];

    u64 s2, c2;
    {
        u64 left_word  = __shfl_up_sync(kFullWarpMask, current, 1, BLOCK_X);
        u64 right_word = __shfl_down_sync(kFullWarpMask, current, 1, BLOCK_X);

        if (lane_in_row == 0) {
            left_word = (col != 0) ? p[-1] : u64{0};
        }
        if (lane_in_row == BLOCK_X - 1) {
            right_word = (col != last_word) ? p[1] : u64{0};
        }

        const u64 west = (current << 1) | (left_word >> 63);
        const u64 east = (current >> 1) | (right_word << 63);

        s2 = west ^ east;
        c2 = west & east;
    }

    u64 s0, c0;
    {
        const u64 up = has_up ? p[-stride] : u64{0};

        u64 up_left_word  = __shfl_up_sync(kFullWarpMask, up, 1, BLOCK_X);
        u64 up_right_word = __shfl_down_sync(kFullWarpMask, up, 1, BLOCK_X);

        if (lane_in_row == 0) {
            up_left_word = (has_up && (col != 0)) ? p[-stride - 1] : u64{0};
        }
        if (lane_in_row == BLOCK_X - 1) {
            up_right_word = (has_up && (col != last_word)) ? p[-stride + 1] : u64{0};
        }

        const u64 north_west = (up << 1) | (up_left_word >> 63);
        const u64 north_east = (up >> 1) | (up_right_word << 63);

        csa64(north_west, up, north_east, s0, c0);
    }

    u64 s1, c1;
    {
        const u64 down = has_down ? p[stride] : u64{0};

        u64 down_left_word  = __shfl_up_sync(kFullWarpMask, down, 1, BLOCK_X);
        u64 down_right_word = __shfl_down_sync(kFullWarpMask, down, 1, BLOCK_X);

        if (lane_in_row == 0) {
            down_left_word = (has_down && (col != 0)) ? p[stride - 1] : u64{0};
        }
        if (lane_in_row == BLOCK_X - 1) {
            down_right_word = (has_down && (col != last_word)) ? p[stride + 1] : u64{0};
        }

        const u64 south_west = (down << 1) | (down_left_word >> 63);
        const u64 south_east = (down >> 1) | (down_right_word << 63);

        csa64(south_west, down, south_east, s1, c1);
    }

    return finish_word(current, s0, c0, s1, c1, s2, c2);
}

/*
The kernel is launched as a 1D grid of 2D blocks.

Why 1D blocks-in-grid instead of a conventional 2D grid?
- It avoids the legacy 2D-grid Y-dimension limit for very large boards.
- The board width in packed words is a power of two, and BLOCK_X is also a power of two,
  so blocks_per_row is a power of two.
- That lets the kernel recover (block_row, block_col) from blockIdx.x using one shift
  and one mask instead of division/modulo.
*/
template <int BLOCK_X, int BLOCK_Y>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         int grid_dimensions,
                         int words_per_row,
                         int blocks_per_row,
                         int blocks_per_row_log2,
                         int block_rows)
{
    static_assert(BLOCK_X * BLOCK_Y == 256, "Block configuration must contain exactly 256 threads");
    static_assert((BLOCK_X & (BLOCK_X - 1)) == 0, "BLOCK_X must be a power of two");

    const unsigned int block_linear = blockIdx.x;
    const int block_col = static_cast<int>(block_linear & static_cast<unsigned int>(blocks_per_row - 1));
    const int block_row = static_cast<int>(block_linear >> blocks_per_row_log2);

    const int col = block_col * BLOCK_X + static_cast<int>(threadIdx.x);
    const int row = block_row * BLOCK_Y + static_cast<int>(threadIdx.y);

    const std::size_t idx =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row) +
        static_cast<std::size_t>(col);

    const bool block_interior =
        (block_col > 0) && (block_col + 1 < blocks_per_row) &&
        (block_row > 0) && (block_row + 1 < block_rows);

    if (block_interior) {
        output[idx] = evolve_shfl_interior<BLOCK_X>(input, idx, words_per_row);
    } else {
        output[idx] = evolve_shfl_boundary<BLOCK_X>(input, idx, row, col, grid_dimensions, words_per_row);
    }
}

// Host-side utility; this is not on the measured path.
inline int log2_power_of_two(unsigned int x)
{
    int s = 0;
    while ((1u << s) < x) {
        ++s;
    }
    return s;
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The caller is expected to provide distinct ping-pong buffers and handle synchronization.
    // The launch itself is intentionally asynchronous.

    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Two tuned shapes:
    //   32x8  -> one full warp updates one row segment of 32 packed words
    //   16x16 -> one half-warp updates one row segment of 16 packed words
    //
    // Because the board dimensions are powers of two and > 512, these tile the board exactly.
    if (words_per_row >= 32) {
        constexpr int BLOCK_X = 32;
        constexpr int BLOCK_Y = 8;

        const int blocks_per_row = words_per_row / BLOCK_X;
        const int block_rows = grid_dimensions / BLOCK_Y;
        const int blocks_per_row_log2 = log2_power_of_two(static_cast<unsigned int>(blocks_per_row));
        const unsigned int total_blocks =
            static_cast<unsigned int>(static_cast<std::size_t>(blocks_per_row) *
                                      static_cast<std::size_t>(block_rows));

        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(total_blocks);

        game_of_life_kernel<BLOCK_X, BLOCK_Y><<<grid, block>>>(
            input,
            output,
            grid_dimensions,
            words_per_row,
            blocks_per_row,
            blocks_per_row_log2,
            block_rows);
    } else {
        constexpr int BLOCK_X = 16;
        constexpr int BLOCK_Y = 16;

        const int blocks_per_row = words_per_row / BLOCK_X;
        const int block_rows = grid_dimensions / BLOCK_Y;
        const int blocks_per_row_log2 = log2_power_of_two(static_cast<unsigned int>(blocks_per_row));
        const unsigned int total_blocks =
            static_cast<unsigned int>(static_cast<std::size_t>(blocks_per_row) *
                                      static_cast<std::size_t>(block_rows));

        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(total_blocks);

        game_of_life_kernel<BLOCK_X, BLOCK_Y><<<grid, block>>>(
            input,
            output,
            grid_dimensions,
            words_per_row,
            blocks_per_row,
            blocks_per_row_log2,
            block_rows);
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
