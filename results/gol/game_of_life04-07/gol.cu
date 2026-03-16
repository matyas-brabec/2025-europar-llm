#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Optimized Conway's Game of Life step for bit-packed rows.
// - One CUDA thread owns one 64-bit word, i.e. 64 cells.
// - No shared memory is used: for this stencil on modern A100/H100-class GPUs,
//   warp shuffles are enough to eliminate most redundant left/right loads while
//   vertical reuse is handled well by the hardware caches.
// - A per-cell implementation with tiny masks and __popc is still much slower
//   than summing all 64 lanes in parallel with bit-sliced logic.

namespace {

using word_t = unsigned long long;

constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;
constexpr unsigned int kRowsPerBlock = 8u;  // 32x8 -> 256 threads, 16x8 -> 128 threads.
static_assert((kRowsPerBlock & (kRowsPerBlock - 1u)) == 0u,
              "kRowsPerBlock must remain a power of two.");

// Host-side integer log2 for a known power-of-two value.
// Host overhead is not performance-critical here; keep it simple and portable.
static inline unsigned int ilog2_pow2(unsigned int v) {
    unsigned int r = 0u;
    while (v > 1u) {
        v >>= 1u;
        ++r;
    }
    return r;
}

// Bitwise majority of three 64-bit masks.
// The compiler maps this form efficiently to ternary logic (LOP3) on Ampere/Hopper.
__device__ __forceinline__ word_t majority3(word_t a, word_t b, word_t c) {
    return (a & b) | (c & (a ^ b));
}

template <int TILE_W>
__global__ void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    unsigned int words_per_row,
    unsigned int last_row,
    unsigned int row_tiles_log2,
    unsigned int row_tiles_mask,
    unsigned int block_x_last)
{
    static_assert(TILE_W == 16 || TILE_W == 32,
                  "The launch helper only instantiates 16-wide or 32-wide tiles.");

    // grid_dimensions and words_per_row are powers of two.
    // We linearize the logical 2D block grid into grid.x because grid.y is too small
    // for the largest legal inputs. row tiles are made the fast-varying dimension so
    // consecutive block indices tend to walk vertically, which improves cache reuse
    // because adjacent row tiles share input rows.
    const unsigned int block_linear = blockIdx.x;
    const unsigned int block_y = block_linear & row_tiles_mask;
    const unsigned int block_x = block_linear >> row_tiles_log2;

    const unsigned int row = block_y * kRowsPerBlock + threadIdx.y;
    const unsigned int word = block_x * TILE_W + threadIdx.x;

    const std::size_t idx = static_cast<std::size_t>(row) * words_per_row + word;

    const bool has_up = (row != 0u);
    const bool has_down = (row != last_row);

    // Arithmetic on size_t is well-defined even when the result is not used because
    // unsigned wrap-around is defined. Loads remain predicated by has_up/has_down.
    const std::size_t up_idx = idx - words_per_row;
    const std::size_t down_idx = idx + words_per_row;

    // Center words from the row above, current row, and row below.
    const word_t mc = static_cast<word_t>(input[idx]);
    const word_t tc = has_up ? static_cast<word_t>(input[up_idx]) : 0ull;
    const word_t bc = has_down ? static_cast<word_t>(input[down_idx]) : 0ull;

    // Horizontal neighbors are obtained from warp shuffles whenever possible.
    // For TILE_W == 32, one warp == one row chunk.
    // For TILE_W == 16, the shuffle width partitions each warp into two independent
    // 16-lane sub-warps, which exactly matches the only legal narrow-row case.
    word_t ml = __shfl_up_sync(kFullWarpMask, mc, 1, TILE_W);
    word_t mr = __shfl_down_sync(kFullWarpMask, mc, 1, TILE_W);
    word_t tl = __shfl_up_sync(kFullWarpMask, tc, 1, TILE_W);
    word_t tr = __shfl_down_sync(kFullWarpMask, tc, 1, TILE_W);
    word_t bl = __shfl_up_sync(kFullWarpMask, bc, 1, TILE_W);
    word_t br = __shfl_down_sync(kFullWarpMask, bc, 1, TILE_W);

    // Threads at the sub-warp boundaries need cross-chunk loads (or zero at the true
    // grid boundary). Only two lanes per sub-warp take these paths.
    const bool has_left_chunk = (block_x != 0u);
    const bool has_right_chunk = (block_x != block_x_last);

    if (threadIdx.x == 0u) {
        ml = has_left_chunk ? static_cast<word_t>(input[idx - 1]) : 0ull;
        tl = (has_left_chunk && has_up) ? static_cast<word_t>(input[up_idx - 1]) : 0ull;
        bl = (has_left_chunk && has_down) ? static_cast<word_t>(input[down_idx - 1]) : 0ull;
    }

    if (threadIdx.x == static_cast<unsigned int>(TILE_W - 1)) {
        mr = has_right_chunk ? static_cast<word_t>(input[idx + 1]) : 0ull;
        tr = (has_right_chunk && has_up) ? static_cast<word_t>(input[up_idx + 1]) : 0ull;
        br = (has_right_chunk && has_down) ? static_cast<word_t>(input[down_idx + 1]) : 0ull;
    }

    // Bit layout convention:
    //   bit 0  = left-most cell inside this 64-cell word
    //   bit 63 = right-most cell inside this 64-cell word
    // This matches the user's boundary note that bit 0 reaches into the word to the
    // left and bit 63 reaches into the word to the right.
    //
    // Therefore:
    //   west-aligned  neighbors use << 1 and carry in bit 63 from the previous word
    //   east-aligned  neighbors use >> 1 and carry in bit 0  from the next word

    // Horizontal 3-cell / 2-cell row sums in bit-sliced form:
    // upper row: (u1:u0) = up_w + up_c + up_e   in [0,3]
    // middle row: (m1:m0) = mid_w + mid_e       in [0,2]
    // lower row: (d1:d0) = dn_w + dn_c + dn_e   in [0,3]
    word_t u0, u1;
    {
        const word_t up_w = (tc << 1) | (tl >> 63);
        const word_t up_e = (tc >> 1) | (tr << 63);
        u0 = up_w ^ tc ^ up_e;
        u1 = majority3(up_w, tc, up_e);
    }

    word_t m0, m1;
    {
        const word_t mid_w = (mc << 1) | (ml >> 63);
        const word_t mid_e = (mc >> 1) | (mr << 63);
        m0 = mid_w ^ mid_e;
        m1 = mid_w & mid_e;
    }

    word_t d0, d1;
    {
        const word_t dn_w = (bc << 1) | (bl >> 63);
        const word_t dn_e = (bc >> 1) | (br << 63);
        d0 = dn_w ^ bc ^ dn_e;
        d1 = majority3(dn_w, bc, dn_e);
    }

    // Add the three partial row counts.
    // We only materialize the low 3 bits of the 0..8 neighbor count:
    //   count = c0 + 2*c1 + 4*c2 (+ 8*c3, ignored)
    //
    // That is sufficient because the Life rule can be written as:
    //   next = (count == 3) | (alive & count == 2)
    //        = c1 & ~c2 & (alive | c0)
    //
    // The only missed case when c3 would matter is count == 8, but that already has
    // c1 == 0, so it is automatically rejected.
    const word_t t0 = u0 ^ m0;
    const word_t carry0 = u0 & m0;
    const word_t t1 = u1 ^ m1 ^ carry0;
    const word_t t2 = majority3(u1, m1, carry0);

    const word_t c0 = t0 ^ d0;
    const word_t carry1 = t0 & d0;
    const word_t c1 = t1 ^ d1 ^ carry1;
    const word_t c2 = t2 ^ majority3(t1, d1, carry1);

    const word_t next = c1 & ~c2 & (mc | c0);
    output[idx] = static_cast<std::uint64_t>(next);
}

template <int TILE_W>
inline void launch_game_of_life_kernel(
    const std::uint64_t* input,
    std::uint64_t* output,
    unsigned int grid_dim,
    unsigned int words_per_row)
{
    static_assert(TILE_W == 16 || TILE_W == 32, "Unsupported tile width.");

    const unsigned int blocks_x = words_per_row / TILE_W;
    const unsigned int row_tiles = grid_dim / kRowsPerBlock;

    // All of these are powers of two:
    //   grid_dim         by contract
    //   words_per_row    = grid_dim / 64
    //   TILE_W           = 16 or 32
    //   kRowsPerBlock    = 8
    //
    // So blocks_x and row_tiles are powers of two as well.
    const unsigned int row_tiles_log2 = ilog2_pow2(row_tiles);
    const unsigned int row_tiles_mask = row_tiles - 1u;
    const unsigned int block_x_last = blocks_x - 1u;
    const unsigned int total_blocks = blocks_x * row_tiles;

    const dim3 block(TILE_W, kRowsPerBlock, 1u);
    game_of_life_kernel<TILE_W><<<total_blocks, block>>>(
        input,
        output,
        words_per_row,
        grid_dim - 1u,
        row_tiles_log2,
        row_tiles_mask,
        block_x_last);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller owns synchronization. Keep this function fully asynchronous:
    // no host/device sync and no error-querying here.
    const unsigned int grid_dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = grid_dim >> 6;

    // grid_dimensions is a power of two > 512, so words_per_row is a power of two.
    // The only legal case smaller than 32 words/row is exactly 16.
    if (words_per_row == 16u) {
        launch_game_of_life_kernel<16>(input, output, grid_dim, words_per_row);
    } else {
        launch_game_of_life_kernel<32>(input, output, grid_dim, words_per_row);
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
