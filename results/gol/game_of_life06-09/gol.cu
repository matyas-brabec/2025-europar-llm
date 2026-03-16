#include <cstdint>
#include <cuda_runtime.h>

namespace {

using u64 = std::uint64_t;

// The problem guarantees power-of-two grids > 512, so:
//   words_per_row = grid_dimensions / 64 is either 16 (only for 1024x1024)
//   or a multiple of 32.
// We exploit that with exact x-dimensions and no partial-warps in x.
//
// A fixed, power-of-two y-cap keeps the row-stride loop regular without paying
// host-side occupancy-query overhead on every call.
constexpr unsigned kSmallBlockX = 16u;
constexpr unsigned kLargeBlockX = 32u;
constexpr unsigned kBlockY      = 8u;
constexpr unsigned kMaxGridY    = 1024u;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert((kBlockY & (kBlockY - 1u)) == 0u, "kBlockY must stay a power of two.");
static_assert((kMaxGridY & (kMaxGridY - 1u)) == 0u, "kMaxGridY must stay a power of two.");

// Majority for three 1-bit inputs, applied bit-slice-wise across 64 cells.
// On modern NVIDIA GPUs this maps efficiently to a single LOP3.
static __device__ __forceinline__ u64 majority3(const u64 a, const u64 b, const u64 c) {
    return (a & b) | (a & c) | (b & c);
}

// Bit packing convention used here:
//   bit k in a word is cell x = word_base + k.
// Therefore:
//   left neighbor  -> shift left, borrowing bit 63 from the word to the left
//   right neighbor -> shift right, borrowing bit 0  from the word to the right
static __device__ __forceinline__ u64 align_left(const u64 center, const u64 left_word) {
    return (center << 1) | (left_word >> 63);
}

static __device__ __forceinline__ u64 align_right(const u64 center, const u64 right_word) {
    return (center >> 1) | (right_word << 63);
}

// Compute one Game-of-Life step for 64 cells packed in one word.
// Inputs are the raw neighbor words around the current word.
// Missing words outside the grid are passed in as zero, matching the required
// dead-outside boundary condition.
static __device__ __forceinline__ u64 next_word(
    const u64 current_word,
    const u64 left_word,  const u64 right_word,
    const u64 above_word, const u64 above_left_word, const u64 above_right_word,
    const u64 below_word, const u64 below_left_word, const u64 below_right_word)
{
    // Align all eight neighbors to the current bit positions.
    const u64 tl = align_left(above_word, above_left_word);
    const u64 tc = above_word;
    const u64 tr = align_right(above_word, above_right_word);

    const u64 ml = align_left(current_word, left_word);
    const u64 mr = align_right(current_word, right_word);

    const u64 bl = align_left(below_word, below_left_word);
    const u64 bc = below_word;
    const u64 br = align_right(below_word, below_right_word);

    // Bit-sliced carry-save addition:
    //
    // Top row count    = t0 + 2*t1, where
    //   t0 = tl ^ tc ^ tr
    //   t1 = majority(tl, tc, tr)
    //
    // Middle row count = m0 + 2*m1, where
    //   m0 = ml ^ mr
    //   m1 = ml & mr
    //
    // Bottom row count = b0 + 2*b1, where
    //   b0 = bl ^ bc ^ br
    //   b1 = majority(bl, bc, br)
    //
    // This processes all 64 cell positions in parallel with plain bitwise ops.
    const u64 t0 = tl ^ tc ^ tr;
    const u64 t1 = majority3(tl, tc, tr);

    const u64 m0 = ml ^ mr;
    const u64 m1 = ml & mr;

    const u64 b0 = bl ^ bc ^ br;
    const u64 b1 = majority3(bl, bc, br);

    // Combine the three group-counts. The full 4-bit count would be:
    //   bit0 = bit0
    //   bit1 = c0 ^ s1
    //   bit2 = c1 ^ (c0 & s1)
    //   bit3 = c1 & (c0 & s1)
    //
    // Conway's rule only needs to distinguish counts {2, 3} from everything
    // else, so we materialize:
    //   bit0  = count bit 0
    //   bit1  = count bit 1
    //   ge4   = (count >= 4)
    const u64 bit0 = t0 ^ m0 ^ b0;
    const u64 c0   = majority3(t0, m0, b0);

    const u64 s1   = t1 ^ m1 ^ b1;
    const u64 c1   = majority3(t1, m1, b1);

    const u64 bit1 = c0 ^ s1;
    const u64 ge4  = c1 | (c0 & s1);

    // Game of Life:
    //   next = (count == 3) | (alive && count == 2)
    //        = (~ge4) & bit1 & (bit0 | alive)
    return (~ge4) & bit1 & (bit0 | current_word);
}

// Kernel body shared by the 16-wide and 32-wide specializations.
// kBlockX is both the block width and the horizontal shuffle width:
//
//   kBlockX == 32:
//     one warp = one 32-word horizontal segment
//
//   kBlockX == 16:
//     one warp = two independent 16-lane segments
//     (used only for the smallest legal grid: 1024x1024 -> 16 words/row)
//
// Horizontal neighbor words within a segment are exchanged with warp shuffles;
// only the segment-edge lanes issue extra global loads across segment boundaries.
// Vertical reuse is left to the hardware caches; shared memory adds overhead for
// this access pattern and is unnecessary here.
template <unsigned kBlockX>
static __device__ __forceinline__ void game_of_life_kernel_body(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const unsigned grid_dimensions)
{
    static_assert(kBlockX == kSmallBlockX || kBlockX == kLargeBlockX,
                  "Unsupported horizontal tile width.");

    const u64 row_words  = static_cast<u64>(grid_dimensions >> 6);
    const u64 total_rows = static_cast<u64>(grid_dimensions);

    const bool first_segment = (blockIdx.x == 0);
    const bool last_segment  = (blockIdx.x + 1 == gridDim.x);

    const bool at_segment_left  = (threadIdx.x == 0);
    const bool at_segment_right = (threadIdx.x == (kBlockX - 1u));

    const bool has_left         = !first_segment || !at_segment_left;
    const bool has_right        = !last_segment  || !at_segment_right;
    const bool use_left_global  = !first_segment && at_segment_left;
    const bool use_right_global = !last_segment  && at_segment_right;

    const u64 col = static_cast<u64>(blockIdx.x) * kBlockX + threadIdx.x;

    // Because grid_dimensions, kBlockY, and kMaxGridY are powers of two, this
    // row-stride is also a power of two. Combined with the problem's power-of-two
    // grid size, that keeps loop trip counts uniform.
    u64 row = static_cast<u64>(blockIdx.y) * kBlockY + threadIdx.y;
    const u64 row_step = static_cast<u64>(gridDim.y) * kBlockY;

    u64 idx = row * row_words + col;
    const u64 idx_step = row_step * row_words;

    for (; row < total_rows; row += row_step, idx += idx_step) {
        const bool has_top    = (row != 0);
        const bool has_bottom = (row + 1u < total_rows);

        const u64 idx_above = idx - row_words;
        const u64 idx_below = idx + row_words;

        const u64 current_word = input[idx];
        const u64 above_word   = has_top    ? input[idx_above] : u64{0};
        const u64 below_word   = has_bottom ? input[idx_below] : u64{0};

        // Shuffles are exact and always safe here because the launch geometry is
        // chosen so that x is exact (no partial x-blocks), and kBlockX matches the
        // logical horizontal segment width.
        const u64 current_left_shfl = __shfl_up_sync  (kFullWarpMask, current_word, 1, kBlockX);
        const u64 current_right_shfl= __shfl_down_sync(kFullWarpMask, current_word, 1, kBlockX);
        const u64 above_left_shfl   = __shfl_up_sync  (kFullWarpMask, above_word,   1, kBlockX);
        const u64 above_right_shfl  = __shfl_down_sync(kFullWarpMask, above_word,   1, kBlockX);
        const u64 below_left_shfl   = __shfl_up_sync  (kFullWarpMask, below_word,   1, kBlockX);
        const u64 below_right_shfl  = __shfl_down_sync(kFullWarpMask, below_word,   1, kBlockX);

        u64 left_word = 0;
        u64 right_word = 0;
        if (has_left) {
            left_word = use_left_global ? input[idx - 1] : current_left_shfl;
        }
        if (has_right) {
            right_word = use_right_global ? input[idx + 1] : current_right_shfl;
        }

        u64 above_left_word = 0;
        u64 above_right_word = 0;
        if (has_top) {
            if (has_left) {
                above_left_word = use_left_global ? input[idx_above - 1] : above_left_shfl;
            }
            if (has_right) {
                above_right_word = use_right_global ? input[idx_above + 1] : above_right_shfl;
            }
        }

        u64 below_left_word = 0;
        u64 below_right_word = 0;
        if (has_bottom) {
            if (has_left) {
                below_left_word = use_left_global ? input[idx_below - 1] : below_left_shfl;
            }
            if (has_right) {
                below_right_word = use_right_global ? input[idx_below + 1] : below_right_shfl;
            }
        }

        output[idx] = next_word(
            current_word,
            left_word, right_word,
            above_word, above_left_word, above_right_word,
            below_word, below_left_word, below_right_word);
    }
}

// Thin wrappers provide exact launch bounds for the two only useful x-widths.
__launch_bounds__(128)
__global__ void game_of_life_kernel_16(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const unsigned grid_dimensions)
{
    game_of_life_kernel_body<kSmallBlockX>(input, output, grid_dimensions);
}

__launch_bounds__(256)
__global__ void game_of_life_kernel_32(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const unsigned grid_dimensions)
{
    game_of_life_kernel_body<kLargeBlockX>(input, output, grid_dimensions);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The caller handles synchronization. For performance, this function only
    // enqueues the kernel and does not add host-side synchronization or extra
    // runtime tuning overhead.

    const unsigned dim = static_cast<unsigned>(grid_dimensions);
    const unsigned words_per_row = dim >> 6;
    const unsigned row_tiles = dim >> 3;  // exact because kBlockY == 8 and dim is a power of two
    const unsigned grid_y = (row_tiles < kMaxGridY) ? row_tiles : kMaxGridY;

    // Exact x launch geometry:
    //   1024x1024 -> 16 words/row -> 16-wide specialization
    //   larger    -> multiple of 32 words/row -> 32-wide specialization
    if (words_per_row == kSmallBlockX) {
        game_of_life_kernel_16<<<dim3(1u, grid_y), dim3(kSmallBlockX, kBlockY)>>>(
            input, output, dim);
    } else {
        game_of_life_kernel_32<<<dim3(words_per_row >> 5, grid_y), dim3(kLargeBlockX, kBlockY)>>>(
            input, output, dim);
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
