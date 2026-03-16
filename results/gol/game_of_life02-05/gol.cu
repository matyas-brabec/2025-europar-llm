#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// This implementation intentionally avoids shared/texture memory.
// The useful horizontal reuse is cheaper via warp shuffles, while the remaining
// vertical accesses are regular row-stride loads that cache well on modern GPUs.
//
// One thread updates one 64-bit word = 64 cells.  The kernel is fully bit-sliced:
// every bit position in a 64-bit register is an independent cell lane.

using word_t = std::uint64_t;

constexpr int kBlockThreads = 256;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert(sizeof(word_t) == 8, "word_t must be 64-bit.");
static_assert(kBlockThreads % 32 == 0, "Block size must be warp-aligned.");
static_assert(kBlockThreads % 16 == 0, "Block size must support 16-lane tiles.");

// Extract the word-edge bits that must cross 64-bit word boundaries.
__device__ __forceinline__ unsigned word_msb(const word_t v) {
    return static_cast<unsigned>(v >> 63);
}

__device__ __forceinline__ unsigned word_lsb(const word_t v) {
    return static_cast<unsigned>(v) & 1u;
}

// Bit-sliced 3-input full adder.
// For every bit position i:
//   a_i + b_i + c_i = sum_i + 2 * carry_i
//
// Modern NVIDIA compilers generally map these boolean expressions to very few
// LOP3 instructions.
__device__ __forceinline__ void full_add(
    const word_t a,
    const word_t b,
    const word_t c,
    word_t& sum,
    word_t& carry)
{
    const word_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// Each thread needs exactly two carried-in boundary bits per row:
// - previous word's MSB for the left-shifted (west-neighbor) view
// - next word's LSB for the right-shifted (east-neighbor) view
//
// We pack the three rows' edge bits into one 3-bit integer so the whole
// 3-row exchange costs only two shuffles per thread, not six.
template <int GROUP_WIDTH>
__device__ __forceinline__ void exchange_packed_edge_bits(
    const unsigned packed_msbs,
    const unsigned packed_lsbs,
    const bool left_group_edge,
    const bool right_group_edge,
    const unsigned left_boundary_packed,
    const unsigned right_boundary_packed,
    unsigned& packed_from_left,
    unsigned& packed_from_right)
{
    packed_from_left  = __shfl_up_sync(kFullWarpMask, packed_msbs, 1, GROUP_WIDTH);
    packed_from_right = __shfl_down_sync(kFullWarpMask, packed_lsbs, 1, GROUP_WIDTH);

    // Only group-edge lanes need explicit boundary values.
    if (left_group_edge) {
        packed_from_left = left_boundary_packed;
    }
    if (right_group_edge) {
        packed_from_right = right_boundary_packed;
    }
}

// Build the 2-bit population count for one row of the 3x3 window:
//   west + center + east = sum + 2*carry
//
// Out-of-bounds cells are dead, so missing carry-in bits are zero.
__device__ __forceinline__ void count_row(
    const word_t center,
    const unsigned left_carry,
    const unsigned right_carry,
    word_t& sum,
    word_t& carry)
{
    const word_t west = (center << 1) | static_cast<word_t>(left_carry);
    const word_t east = (center >> 1) | (static_cast<word_t>(right_carry) << 63);
    full_add(west, center, east, sum, carry);
}

// We sum the full 3x3 population, center included.  Conway's rule then becomes:
//
//   next = (population == 3) | (center & (population == 4))
//
// That is cheaper than first forming the 8-neighbor count and then handling
// survival as a separate path.
__device__ __forceinline__ word_t apply_life_rule(
    const word_t center,
    const word_t low0,
    const word_t low1,
    const word_t high0,
    const word_t high1)
{
    const word_t middle_bit = low1 ^ high0;
    const word_t total_eq_3 = low0 & middle_bit & ~high1;
    const word_t total_eq_4 = ~low0 & ~middle_bit & (high1 ^ low1);
    return total_eq_3 | (center & total_eq_4);
}

template <int GROUP_WIDTH>
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(
    const word_t* __restrict__ input,
    word_t* __restrict__ output,
    int grid_dimensions,
    int words_per_row_shift)
{
    static_assert(GROUP_WIDTH == 16 || GROUP_WIDTH == 32,
                  "Only 16- or 32-lane shuffle tiles are supported.");

    // Under the stated problem constraints:
    // - total word count is an exact multiple of kBlockThreads
    // - words_per_row is a power of two
    // - GROUP_WIDTH exactly tiles each row (16 for 1024x1024, otherwise 32)
    //
    // Therefore every launched thread is valid and no tail predicate is needed.
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * kBlockThreads + threadIdx.x;

    const std::size_t row_stride = std::size_t{1} << words_per_row_shift;  // 64-bit words per row
    const std::size_t row_mask   = row_stride - 1;
    const std::size_t grid_dim   = static_cast<std::size_t>(grid_dimensions);

    const std::size_t row       = idx >> words_per_row_shift;
    const std::size_t row_local = idx & row_mask;

    const bool has_top        = (row != 0);
    const bool has_bottom     = (row + 1 != grid_dim);
    const bool has_left_word  = (row_local != 0);
    const bool has_right_word = (row_local != row_mask);

    const unsigned lane_in_group = threadIdx.x & (GROUP_WIDTH - 1);
    const bool left_group_edge   = (lane_in_group == 0u);
    const bool right_group_edge  = (lane_in_group == static_cast<unsigned>(GROUP_WIDTH - 1));

    const std::size_t top_idx = idx - row_stride;
    const std::size_t bot_idx = idx + row_stride;

    // Vertical neighbors are explicit row-stride loads.
    // Horizontal reuse is handled by exchanging just the required edge bits.
    const word_t mid_c = input[idx];
    const word_t top_c = has_top    ? input[top_idx] : word_t{0};
    const word_t bot_c = has_bottom ? input[bot_idx] : word_t{0};

    // Packed boundary bits for the group-edge lanes:
    // bit 0 = top row, bit 1 = middle row, bit 2 = bottom row.
    unsigned left_boundary_packed = 0u;
    if (left_group_edge && has_left_word) {
        left_boundary_packed = word_msb(input[idx - 1]) << 1;
        if (has_top) {
            left_boundary_packed |= word_msb(input[top_idx - 1]);
        }
        if (has_bottom) {
            left_boundary_packed |= word_msb(input[bot_idx - 1]) << 2;
        }
    }

    unsigned right_boundary_packed = 0u;
    if (right_group_edge && has_right_word) {
        right_boundary_packed = word_lsb(input[idx + 1]) << 1;
        if (has_top) {
            right_boundary_packed |= word_lsb(input[top_idx + 1]);
        }
        if (has_bottom) {
            right_boundary_packed |= word_lsb(input[bot_idx + 1]) << 2;
        }
    }

    const unsigned packed_msbs =
        word_msb(top_c) |
        (word_msb(mid_c) << 1) |
        (word_msb(bot_c) << 2);

    const unsigned packed_lsbs =
        word_lsb(top_c) |
        (word_lsb(mid_c) << 1) |
        (word_lsb(bot_c) << 2);

    unsigned packed_from_left;
    unsigned packed_from_right;
    exchange_packed_edge_bits<GROUP_WIDTH>(
        packed_msbs,
        packed_lsbs,
        left_group_edge,
        right_group_edge,
        left_boundary_packed,
        right_boundary_packed,
        packed_from_left,
        packed_from_right);

    const unsigned top_left_in  =  packed_from_left        & 1u;
    const unsigned mid_left_in  = (packed_from_left  >> 1) & 1u;
    const unsigned bot_left_in  =  packed_from_left  >> 2;

    const unsigned top_right_in =  packed_from_right       & 1u;
    const unsigned mid_right_in = (packed_from_right >> 1) & 1u;
    const unsigned bot_right_in =  packed_from_right >> 2;

    // 2-bit row sums for the three rows of the 3x3 window.
    word_t top0, top1;
    word_t mid0, mid1;
    word_t bot0, bot1;
    count_row(top_c, top_left_in, top_right_in, top0, top1);
    count_row(mid_c, mid_left_in, mid_right_in, mid0, mid1);
    count_row(bot_c, bot_left_in, bot_right_in, bot0, bot1);

    // Sum the three 2-bit row counts into a 4-bit 3x3 population count.
    word_t low0, low1;
    word_t high0, high1;
    full_add(top0, mid0, bot0, low0, low1);
    full_add(top1, mid1, bot1, high0, high1);

    output[idx] = apply_life_rule(mid_c, low0, low1, high0, high1);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The documented preconditions are used as optimization assumptions:
    // - grid_dimensions is a power of two
    // - grid_dimensions > 512
    // - input/output are device allocations from cudaMalloc
    // - input and output do not overlap (out-of-place update)
    //
    // Because words_per_row is also a power of two, the kernel can recover
    // row/column positions with bit shifts and masks instead of division/modulo.
    const int words_per_row = grid_dimensions >> 6;

    int words_per_row_shift = 0;
    for (int t = words_per_row; t > 1; t >>= 1) {
        ++words_per_row_shift;
    }

    // One thread updates one 64-bit word.  total_words is always a power of two
    // and, with grid_dimensions > 512, an exact multiple of kBlockThreads.
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) << words_per_row_shift;
    const unsigned int blocks =
        static_cast<unsigned int>(total_words / kBlockThreads);

    // Only two tile widths can occur here:
    // - 16 lanes when grid_dimensions == 1024  (16 words per row)
    // - 32 lanes for all larger grids
    //
    // The largest legal tile minimizes boundary traffic while guaranteeing that
    // shuffle groups never cross a row boundary.
    if (words_per_row == 16) {
        game_of_life_kernel<16><<<blocks, kBlockThreads>>>(
            input,
            output,
            grid_dimensions,
            words_per_row_shift);
    } else {
        game_of_life_kernel<32><<<blocks, kBlockThreads>>>(
            input,
            output,
            grid_dimensions,
            words_per_row_shift);
    }

    // The caller owns synchronization and any desired launch-error handling.
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
