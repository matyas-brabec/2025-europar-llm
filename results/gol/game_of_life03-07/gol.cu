#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// Optimized CUDA implementation of one Conway's Game of Life step for the exact problem statement.
//
// Design choices focused purely on throughput on modern data-center GPUs:
//
// 1) One thread updates one 64-bit word (64 cells) exactly as requested.
//    The input/output format is already the desired packed format, so no unpack/repack is needed.
//
// 2) A 1D grid-stride launch is used instead of a 2D grid.
//    This avoids grid.y limits for very large boards while keeping addressing cheap because
//    words_per_row is a power of two:
//        x = word_index & (words_per_row - 1)
//        y = word_index >> log2(words_per_row)
//
// 3) Shared memory is intentionally not used.
//    Horizontal neighbor reuse is handled with warp shuffles, which are cheaper and simpler here.
//    Every thread always loads only its vertically aligned words (up/current/down). The horizontal
//    words are obtained from neighboring lanes; only segment-edge lanes fall back to extra global
//    loads when a row is wider than one shuffle segment.
//
// 4) Neighbor counts for all 64 cells in a word are computed in parallel with a bit-sliced adder
//    tree. No per-bit loops, no atomics, and no unpacking.
//
// 5) The problem constraints are exploited directly:
//    - grid_dimensions is a power of two and > 512
//    - therefore words_per_row is also a power of two and >= 16
//    - total_words is a power of two and, with the chosen block size, has no launch tail
//
// The block-count cap is kept as a power of two on purpose. Under the stated constraints that keeps
// the grid-stride length as a power of two as well, so all threads execute the same number of loop
// iterations and every warp is fully active for every iteration.

using u64 = std::uint64_t;

constexpr unsigned int kCellsPerWordShift = 6u;  // 64 cells / word.
constexpr int          kBlockSize         = 256; // 8 warps per block.
constexpr unsigned int kMaxBlocks         = 2048u;
constexpr unsigned int kFullWarpMask      = 0xFFFFFFFFu;

static_assert(sizeof(u64) == 8, "u64 must be 64 bits");
static_assert((kBlockSize & (kBlockSize - 1)) == 0, "kBlockSize must be a power of two");
static_assert((kBlockSize & 31) == 0, "kBlockSize must be a multiple of the warp size");
static_assert((kMaxBlocks & (kMaxBlocks - 1u)) == 0u, "kMaxBlocks must be a power of two");

__device__ __forceinline__ u64 majority64(const u64 a, const u64 b, const u64 c) {
    // Carry bit of a 3-input full adder.
    // This boolean form maps well to modern NVIDIA LOP3 instructions.
    return (a & b) | (c & (a ^ b));
}

__device__ __forceinline__ u64 evolve_word(
    const u64 center,
    const u64 up_left,   const u64 up,   const u64 up_right,
    const u64 mid_left,                    const u64 mid_right,
    const u64 down_left, const u64 down, const u64 down_right)
{
    // The prompt's boundary note implies the bit layout inside a word is:
    //   bit 0  = leftmost cell of the 64-cell run
    //   bit 63 = rightmost cell
    //
    // With that convention:
    //   west-aligned(row, left_word) = (row << 1) | (left_word  >> 63)
    //   east-aligned(row, right_word)= (row >> 1) | (right_word << 63)
    //
    // We sum the eight neighbor bitboards in two 4-neighbor groups:
    //   A = NW + N + NE + W
    //   B = E + SW + S + SE
    // Then A + B gives the 3-bit neighbor count for all 64 cells at once.
    //
    // The final rule
    //     next = (count == 3) | (alive & (count == 2))
    // can be written as
    //     next = (~ge4) & bit1 & (bit0 | alive)
    // where:
    //   bit0 = count bit 0
    //   bit1 = count bit 1
    //   ge4  = 1 for counts 4..8
    //
    // The scopes below intentionally keep live ranges short to help register allocation.

    u64 a0, a1, a2;
    {
        const u64 nw = (up << 1) | (up_left >> 63);
        const u64 n  = up;
        const u64 ne = (up >> 1) | (up_right << 63);
        const u64 w  = (center << 1) | (mid_left >> 63);

        const u64 p0 = nw ^ n;
        const u64 p1 = nw & n;
        const u64 q0 = ne ^ w;
        const u64 q1 = ne & w;

        const u64 carry0 = p0 & q0;
        const u64 t1     = p1 ^ q1;

        a0 = p0 ^ q0;
        a1 = t1 ^ carry0;
        a2 = majority64(p1, q1, carry0);
    }

    u64 b0, b1, b2;
    {
        const u64 e  = (center >> 1) | (mid_right << 63);
        const u64 sw = (down << 1) | (down_left >> 63);
        const u64 s  = down;
        const u64 se = (down >> 1) | (down_right << 63);

        const u64 p0 = e ^ sw;
        const u64 p1 = e & sw;
        const u64 q0 = s ^ se;
        const u64 q1 = s & se;

        const u64 carry0 = p0 & q0;
        const u64 t1     = p1 ^ q1;

        b0 = p0 ^ q0;
        b1 = t1 ^ carry0;
        b2 = majority64(p1, q1, carry0);
    }

    const u64 bit0   = a0 ^ b0;
    const u64 carry0 = a0 & b0;
    const u64 t1     = a1 ^ b1;
    const u64 bit1   = t1 ^ carry0;
    const u64 carry1 = (a1 & b1) | (t1 & carry0);
    const u64 ge4    = a2 | b2 | carry1;

    return ~ge4 & bit1 & (bit0 | center);
}

template <int ShuffleWidth, bool CrossWarpX>
__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         std::size_t total_words,
                         unsigned int words_per_row,
                         unsigned int word_mask,
                         unsigned int word_shift,
                         unsigned int rows)
{
    static_assert(ShuffleWidth == 16 || ShuffleWidth == 32,
                  "ShuffleWidth must match the only two row-segment sizes used here");

    // CrossWarpX == false:
    //   The shuffle segment is the full row.
    //   This happens for 1024x1024 boards (16 words/row -> two 16-lane segments per warp)
    //   and 2048x2048 boards (32 words/row -> one 32-lane segment per warp).
    //
    // CrossWarpX == true:
    //   The row is wider than one 32-word warp segment.
    //   Only the first/last lane of each segment must fetch the missing horizontal word
    //   from global memory; all interior lanes get it from shuffles.

    const std::size_t row_stride  = static_cast<std::size_t>(words_per_row);
    const std::size_t grid_stride = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(kBlockSize);
    const unsigned int seg_lane   = static_cast<unsigned int>(threadIdx.x) & static_cast<unsigned int>(ShuffleWidth - 1);
    const unsigned int seg_last   = static_cast<unsigned int>(ShuffleWidth - 1);
    const unsigned int last_row   = rows - 1u;

    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kBlockSize) +
                           static_cast<std::size_t>(threadIdx.x);
         idx < total_words;
         idx += grid_stride)
    {
        const unsigned int y        = static_cast<unsigned int>(idx >> word_shift);
        const bool has_up           = (y != 0u);
        const bool has_down         = (y != last_row);

        // Every thread always loads only the vertically aligned words.
        const u64 center = input[idx];
        const u64 up     = has_up   ? input[idx - row_stride] : u64{0};
        const u64 down   = has_down ? input[idx + row_stride] : u64{0};

        // Horizontal neighbors come from the warp whenever possible.
        // ShuffleWidth is 16 for the 1024x1024 case and 32 otherwise.
        u64 mid_left   = __shfl_up_sync  (kFullWarpMask, center, 1, ShuffleWidth);
        u64 mid_right  = __shfl_down_sync(kFullWarpMask, center, 1, ShuffleWidth);
        u64 up_left    = __shfl_up_sync  (kFullWarpMask, up,     1, ShuffleWidth);
        u64 up_right   = __shfl_down_sync(kFullWarpMask, up,     1, ShuffleWidth);
        u64 down_left  = __shfl_up_sync  (kFullWarpMask, down,   1, ShuffleWidth);
        u64 down_right = __shfl_down_sync(kFullWarpMask, down,   1, ShuffleWidth);

        if (CrossWarpX) {
            // Only segment-edge lanes need special handling.
            if (seg_lane == 0u) {
                const bool has_left = ((static_cast<unsigned int>(idx) & word_mask) != 0u);

                mid_left  = has_left              ? input[idx - 1]                : u64{0};
                up_left   = (has_left && has_up)  ? input[idx - row_stride - 1]   : u64{0};
                down_left = (has_left && has_down)? input[idx + row_stride - 1]   : u64{0};
            } else if (seg_lane == seg_last) {
                const bool has_right = ((static_cast<unsigned int>(idx) & word_mask) != word_mask);

                mid_right  = has_right               ? input[idx + 1]               : u64{0};
                up_right   = (has_right && has_up)   ? input[idx - row_stride + 1]  : u64{0};
                down_right = (has_right && has_down) ? input[idx + row_stride + 1]  : u64{0};
            }
        } else {
            // Here each shuffle segment is an entire row, so segment boundaries are physical
            // grid boundaries and the missing words are simply dead.
            if (seg_lane == 0u) {
                mid_left  = u64{0};
                up_left   = u64{0};
                down_left = u64{0};
            } else if (seg_lane == seg_last) {
                mid_right  = u64{0};
                up_right   = u64{0};
                down_right = u64{0};
            }
        }

        output[idx] = evolve_word(center,
                                  up_left, up, up_right,
                                  mid_left, mid_right,
                                  down_left, down, down_right);
    }
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees valid cudaMalloc'ed device pointers.
    // Double-buffering is assumed: input and output must not alias.
    // No synchronization or error checking is inserted here because the caller explicitly owns that,
    // and avoiding extra host-side API traffic matters in tight multi-step simulations.

    if (grid_dimensions <= 0) {
        return;
    }

    const unsigned int rows          = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = rows >> kCellsPerWordShift;
    const unsigned int word_mask     = words_per_row - 1u;

    // Portable log2(words_per_row) for a power-of-two value.
    unsigned int word_shift = 0u;
    for (unsigned int t = words_per_row; t > 1u; t >>= 1u) {
        ++word_shift;
    }

    const std::size_t total_words =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(words_per_row);

    // Under the stated constraints:
    //   rows          = 2^n, n >= 10
    //   words_per_row = 2^(n-6)
    //   total_words   = 2^(2n-6)
    // so total_words is always a multiple of kBlockSize (256). That means there is no tail.
    const std::size_t blocks_needed = total_words / static_cast<std::size_t>(kBlockSize);
    const unsigned int blocks = static_cast<unsigned int>(
        (blocks_needed < static_cast<std::size_t>(kMaxBlocks))
            ? blocks_needed
            : static_cast<std::size_t>(kMaxBlocks));

    // Three launch variants cover every legal board size:
    //   16 words/row  -> 1024x1024 board, two independent 16-lane row segments per warp
    //   32 words/row  -> 2048x2048 board, one full row per warp
    //   >32 words/row -> wider rows, segment-edge lanes fetch cross-warp horizontal neighbors
    if (words_per_row == 16u) {
        game_of_life_kernel<16, false><<<blocks, kBlockSize>>>(
            input, output, total_words, words_per_row, word_mask, word_shift, rows);
    } else if (words_per_row == 32u) {
        game_of_life_kernel<32, false><<<blocks, kBlockSize>>>(
            input, output, total_words, words_per_row, word_mask, word_shift, rows);
    } else {
        game_of_life_kernel<32, true><<<blocks, kBlockSize>>>(
            input, output, total_words, words_per_row, word_mask, word_shift, rows);
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
