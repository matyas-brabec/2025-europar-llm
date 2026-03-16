#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Fast bit-parallel Conway's Game of Life for a square bit-packed grid.
//
// Encoding assumption used by this kernel:
//   * within each 64-bit word, bit 0 is the leftmost cell in that 64-cell span
//   * bit 63 is the rightmost cell in that 64-cell span
//
// That matches the required special handling:
//   * bit 0 needs the three words to the left  (upper/current/lower rows)
//   * bit 63 needs the three words to the right (upper/current/lower rows)
//
// Performance notes:
//   * one CUDA thread updates one 64-bit word => 64 cells/thread, no atomics
//   * horizontal neighbor reuse is handled with warp/subwarp shuffles
//   * only the first/last lane of each row segment performs extra global loads
//   * the 8 neighbor bitboards are reduced with carry-save adders, so all
//     64 neighbor counts are computed simultaneously with bitwise logic
//   * shared memory is intentionally not used; shuffles cover horizontal reuse
//     and the hardware caches are sufficient for vertical reuse on A100/H100

namespace {

using u64 = std::uint64_t;
using usize = std::size_t;

constexpr unsigned int FULL_MASK = 0xFFFFFFFFu;

// Both launch configurations keep the block size at 256 threads.
// BLOCK_X also defines the logical shuffle width.
constexpr int BLOCK_X_32 = 32;
constexpr int BLOCK_Y_32 = 8;

constexpr int BLOCK_X_16 = 16;
constexpr int BLOCK_Y_16 = 16;

constexpr int TUNED_BLOCK_THREADS = 256;
constexpr int MIN_BLOCKS_PER_SM = 2;

// Host-side helper: input is guaranteed to be a power of two by the problem.
inline unsigned int log2_pow2_host(unsigned int value) {
    unsigned int shift = 0;
    while (value > 1u) {
        value >>= 1u;
        ++shift;
    }
    return shift;
}

// 0x96 = XOR3(a,b,c)
// 0xE8 = MAJORITY(a,b,c) = carry bit of a 3-input full adder
__device__ __forceinline__ u64 xor3_u64(u64 a, u64 b, u64 c) {
#if defined(__CUDA_ARCH__)
    u64 r;
    asm("lop3.b64 %0, %1, %2, %3, 0x96;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
    return r;
#else
    return a ^ b ^ c;
#endif
}

__device__ __forceinline__ u64 maj3_u64(u64 a, u64 b, u64 c) {
#if defined(__CUDA_ARCH__)
    u64 r;
    asm("lop3.b64 %0, %1, %2, %3, 0xE8;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
    return r;
#else
    return (a & b) | (a & c) | (b & c);
#endif
}

__device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    sum = xor3_u64(a, b, c);
    carry = maj3_u64(a, b, c);
}

__device__ __forceinline__ void ha(u64 a, u64 b, u64& sum, u64& carry) {
    sum = a ^ b;
    carry = a & b;
}

template <int BLOCK_X, int BLOCK_Y>
__global__ __launch_bounds__(TUNED_BLOCK_THREADS, MIN_BLOCKS_PER_SM)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         int words_per_row,
                         int rows,
                         unsigned int tile_x_mask,
                         unsigned int tile_x_shift) {
    static_assert(BLOCK_X == 16 || BLOCK_X == 32, "BLOCK_X must be 16 or 32.");
    static_assert(BLOCK_X * BLOCK_Y == TUNED_BLOCK_THREADS,
                  "Kernel tuning assumes 256-thread blocks.");

    // The block grid is intentionally 1D to avoid the 65535 limit on grid.y.
    // Because the number of x-tiles per row is always a power of two, we can
    // decode blockIdx.x into (tile_x, block_row) with a mask and a shift.
    const unsigned int linear_block = blockIdx.x;
    const int tile_x = static_cast<int>(linear_block & tile_x_mask);
    const int block_row = static_cast<int>(linear_block >> tile_x_shift);

    // BLOCK_X is also the shuffle width:
    //   * BLOCK_X == 32: one warp == one row segment
    //   * BLOCK_X == 16: one warp == two independent 16-lane row segments
    //                    and the width parameter keeps them isolated
    const int lane = static_cast<int>(threadIdx.x);
    const int x = tile_x * BLOCK_X + lane;
    const int y = block_row * BLOCK_Y + static_cast<int>(threadIdx.y);

    // Exact tiling is guaranteed by the problem constraints and the launch
    // configuration chosen in run_game_of_life, so no x/y bounds checks are
    // needed inside the hot kernel.
    const bool has_left  = (x != 0);
    const bool has_right = (x != (words_per_row - 1));
    const bool has_up    = (y != 0);
    const bool has_down  = (y != (rows - 1));

    const usize row_stride = static_cast<usize>(words_per_row);
    const usize base = static_cast<usize>(y) * row_stride + static_cast<usize>(x);
    const usize up_base = base - row_stride;
    const usize down_base = base + row_stride;

    // Current word and vertical neighbors. Off-grid rows are injected as zero.
    const u64 mid  = input[base];
    const u64 up   = has_up   ? input[up_base]   : 0ull;
    const u64 down = has_down ? input[down_base] : 0ull;

    // Horizontal neighbors normally come from shuffles inside the row segment.
    // The first/last lane of the segment handles cross-segment / row-edge
    // cases with explicit loads. These are exactly the "three words to the
    // left" / "three words to the right" mentioned in the prompt.
    u64 up_left   = __shfl_up_sync(FULL_MASK, up,   1, BLOCK_X);
    u64 mid_left  = __shfl_up_sync(FULL_MASK, mid,  1, BLOCK_X);
    u64 down_left = __shfl_up_sync(FULL_MASK, down, 1, BLOCK_X);
    if (lane == 0) {
        mid_left  = has_left ? input[base - 1] : 0ull;
        up_left   = (has_left && has_up)   ? input[up_base - 1]   : 0ull;
        down_left = (has_left && has_down) ? input[down_base - 1] : 0ull;
    }

    u64 up_right   = __shfl_down_sync(FULL_MASK, up,   1, BLOCK_X);
    u64 mid_right  = __shfl_down_sync(FULL_MASK, mid,  1, BLOCK_X);
    u64 down_right = __shfl_down_sync(FULL_MASK, down, 1, BLOCK_X);
    if (lane == BLOCK_X - 1) {
        mid_right  = has_right ? input[base + 1] : 0ull;
        up_right   = (has_right && has_up)   ? input[up_base + 1]   : 0ull;
        down_right = (has_right && has_down) ? input[down_base + 1] : 0ull;
    }

    // Build aligned neighbor bitboards.
    // For example, bit k of "west" contains the state of the cell immediately
    // left of the current cell at bit k.
    //
    // The eight neighbors are grouped as:
    //   top row    : NW, N, NE
    //   middle row : W, E
    //   bottom row : SW, S, SE
    //
    // Then we use a carry-save adder tree to compute the 4 bitplanes of the
    // neighbor count (0..8) for all 64 cells in parallel.
    u64 s_top, c_top;
    {
        const u64 nw = (up << 1) | (up_left >> 63);
        const u64 ne = (up >> 1) | (up_right << 63);
        csa(nw, up, ne, s_top, c_top);
    }

    u64 s_mid, c_mid;
    {
        const u64 w = (mid << 1) | (mid_left >> 63);
        const u64 e = (mid >> 1) | (mid_right << 63);
        ha(w, e, s_mid, c_mid);
    }

    u64 s_bot, c_bot;
    {
        const u64 sw = (down << 1) | (down_left >> 63);
        const u64 se = (down >> 1) | (down_right << 63);
        csa(sw, down, se, s_bot, c_bot);
    }

    u64 bit0, carry_ones;
    csa(s_top, s_mid, s_bot, bit0, carry_ones);

    u64 twos_sum, fours_carry;
    csa(c_top, c_mid, c_bot, twos_sum, fours_carry);

    u64 bit1, carry_twos;
    ha(twos_sum, carry_ones, bit1, carry_twos);

    u64 bit2, bit3;
    ha(fours_carry, carry_twos, bit2, bit3);

    // count == 2 or 3  <=>  bit1 == 1 and bit2 == bit3 == 0
    // next = (count == 3) | (alive & count == 2)
    //      = live23 & (bit0 | alive)
    const u64 live23 = bit1 & ~(bit2 | bit3);
    output[base] = live23 & (bit0 | mid);
}

template <int BLOCK_X, int BLOCK_Y>
inline void launch_game_of_life_kernel(const u64* input,
                                       u64* output,
                                       int words_per_row,
                                       int rows) {
    static_assert(BLOCK_X == 16 || BLOCK_X == 32, "BLOCK_X must be 16 or 32.");
    static_assert(BLOCK_X * BLOCK_Y == TUNED_BLOCK_THREADS,
                  "Kernel tuning assumes 256-thread blocks.");

    // Exact tiling guaranteed by the problem:
    //   * words_per_row is a power of two
    //   * rows is a power of two
    //   * BLOCK_X and BLOCK_Y are powers of two that exactly divide the chosen
    //     valid cases
    const unsigned int tiles_x =
        static_cast<unsigned int>(words_per_row / BLOCK_X);
    const unsigned int block_rows =
        static_cast<unsigned int>(rows / BLOCK_Y);

    // tiles_x is a power of two, so we can decode linear block indices with
    // a mask and a shift instead of division/modulo inside the kernel.
    const unsigned int tile_x_mask = tiles_x - 1u;
    const unsigned int tile_x_shift = log2_pow2_host(tiles_x);

    // Row-major block order: x-tiles vary fastest.
    const unsigned int total_blocks = tiles_x * block_rows;

    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid(total_blocks, 1, 1);

    game_of_life_kernel<BLOCK_X, BLOCK_Y>
        <<<grid, block>>>(input, output, words_per_row, rows,
                          tile_x_mask, tile_x_shift);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The caller is responsible for synchronization and for ping-ponging
    // input/output buffers across steps. No host-device sync is performed here.
    if (grid_dimensions <= 0) {
        return;
    }

    // Direct packed-word processing is already the fast path, so no unpack /
    // repack transformation is needed.
    const int words_per_row = grid_dimensions >> 6;

    // Problem constraints:
    //   * grid_dimensions is a power of two
    //   * grid_dimensions > 512
    //
    // Therefore:
    //   * words_per_row is also a power of two
    //   * the only valid case with fewer than 32 words per row is 1024x1024,
    //     which has exactly 16 words per row
    //
    // We use a 16-lane subwarp specialization for that case so no lanes are
    // wasted. All larger valid cases use full 32-lane row segments.
    if (words_per_row == BLOCK_X_16) {
        launch_game_of_life_kernel<BLOCK_X_16, BLOCK_Y_16>(
            input, output, words_per_row, grid_dimensions);
    } else {
        launch_game_of_life_kernel<BLOCK_X_32, BLOCK_Y_32>(
            input, output, words_per_row, grid_dimensions);
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
