#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/*
  One-step Conway's Game of Life on a square, bit-packed grid.

  Performance-oriented design:
  - One CUDA thread updates exactly one 64-bit word, i.e. 64 cells at once.
    That makes the update naturally race-free and eliminates atomics.
  - The grid is row-major and each row contains grid_dimensions / 64 words.
  - Outside-the-grid cells are treated as dead by substituting zero for missing
    neighbor words/rows.
  - Most horizontal/diagonal neighbor words are obtained with warp shuffles from
    threads that already loaded them as their own center/north/south words.
    Only lanes on warp/subwarp boundaries fall back to extra global loads.
  - Eight 1-bit neighbor bitboards are summed with a carry-save adder tree.
    This computes the neighbor counts for all 64 cells in parallel using simple
    bitwise logic.
*/

namespace {

using word_t = std::uint64_t;

constexpr int kBlockSize  = 256;
constexpr int kBlockShift = 8;

static_assert((kBlockSize & (kBlockSize - 1)) == 0, "kBlockSize must be a power of two.");
static_assert((1 << kBlockShift) == kBlockSize,     "kBlockShift must match kBlockSize.");
static_assert((kBlockSize % 32) == 0,               "kBlockSize must be warp-aligned.");

/*
  Carry-save adder for three 1-bit bitboards a, b, c.

  For each bit position independently:
    sum_bit   = a ^ b ^ c
    carry_bit = majority(a, b, c)

  Interpreted numerically, that is:
    a + b + c = sum + 2 * carry
*/
__device__ __forceinline__ void csa(word_t a, word_t b, word_t c, word_t& sum, word_t& carry) {
    const word_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (u & c);
}

/*
  kShuffleWidth is 16 only for the 1024x1024 case (16 packed words per row),
  otherwise it is 32.

  Why two specializations are enough:
  - grid_dimensions is a power of two > 512
  - words_per_row = grid_dimensions / 64 is also a power of two
  - therefore words_per_row is either 16 or a multiple of 32
  - when words_per_row == 16, each warp covers two rows in word-space, so shuffles
    must be restricted to 16-lane subwarps
  - otherwise, rows are 32-word aligned and full-warp shuffles never cross rows
*/
template <int kShuffleWidth>
__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(
    const word_t* __restrict__ input,
    word_t* __restrict__ output,
    std::size_t words_per_row,
    std::size_t last_row_start,
    unsigned int word_mask)
{
    static_assert(kShuffleWidth == 16 || kShuffleWidth == 32, "Unexpected shuffle width.");

    constexpr unsigned int kFullMask    = 0xFFFFFFFFu;
    constexpr unsigned int kSegmentMask = static_cast<unsigned int>(kShuffleWidth - 1);

    // Because the launch uses exactly total_words / kBlockSize blocks and
    // total_words is a power of two >= 2^14, every block and every warp is full.
    // That lets us use full-mask shuffles without an in-kernel bounds check.
    const std::size_t idx =
        (static_cast<std::size_t>(blockIdx.x) << kBlockShift) +
        static_cast<std::size_t>(threadIdx.x);

    // x-coordinate in word-space. This is just a bitmask because words_per_row is a power of two.
    const unsigned int xw = static_cast<unsigned int>(idx) & word_mask;

    const bool has_left  = (xw != 0u);
    const bool has_right = (xw != word_mask);
    const bool has_up    = (idx >= words_per_row);
    const bool has_down  = (idx < last_row_start);

    // The three words every thread always needs for itself.
    const word_t current = input[idx];

    word_t north = 0;
    if (has_up) {
        north = input[idx - words_per_row];
    }

    word_t south = 0;
    if (has_down) {
        south = input[idx + words_per_row];
    }

    // Use shuffles to obtain left/right words for current, north, and south rows.
    // Only lanes on the edge of a warp/subwarp segment need to fall back to global loads.
    word_t current_left  = __shfl_up_sync  (kFullMask, current, 1, kShuffleWidth);
    word_t current_right = __shfl_down_sync(kFullMask, current, 1, kShuffleWidth);
    word_t north_left    = __shfl_up_sync  (kFullMask, north,   1, kShuffleWidth);
    word_t north_right   = __shfl_down_sync(kFullMask, north,   1, kShuffleWidth);
    word_t south_left    = __shfl_up_sync  (kFullMask, south,   1, kShuffleWidth);
    word_t south_right   = __shfl_down_sync(kFullMask, south,   1, kShuffleWidth);

    const unsigned int lane_in_segment = static_cast<unsigned int>(threadIdx.x) & kSegmentMask;

    if (lane_in_segment == 0u) {
        current_left = 0;
        north_left   = 0;
        south_left   = 0;

        if (has_left) {
            current_left = input[idx - 1];
            if (has_up) {
                north_left = input[idx - words_per_row - 1];
            }
            if (has_down) {
                south_left = input[idx + words_per_row - 1];
            }
        }
    } else if (lane_in_segment == kSegmentMask) {
        current_right = 0;
        north_right   = 0;
        south_right   = 0;

        if (has_right) {
            current_right = input[idx + 1];
            if (has_up) {
                north_right = input[idx - words_per_row + 1];
            }
            if (has_down) {
                south_right = input[idx + words_per_row + 1];
            }
        }
    }

    /*
      Align each neighbor direction so that bit i in the resulting word is the
      state of the corresponding neighbor of cell i.

      Example:
        west = (current << 1) | (current_left >> 63)

      That means:
        - target bit i gets source bit i-1 from the current word
        - target bit 0 gets source bit 63 from the word to the left

      This is exactly the required special handling for bit 0 / bit 63.
    */
    const word_t west      = (current << 1) | (current_left >> 63);
    const word_t east      = (current >> 1) | (current_right << 63);
    const word_t northwest = (north   << 1) | (north_left   >> 63);
    const word_t northeast = (north   >> 1) | (north_right  << 63);
    const word_t southwest = (south   << 1) | (south_left   >> 63);
    const word_t southeast = (south   >> 1) | (south_right  << 63);

    /*
      Sum the eight 1-bit neighbor bitboards with a CSA tree:

        group 0: northwest + north + northeast -> ones0 + 2*twos0
        group 1: west + east + southwest       -> ones1 + 2*twos1
        group 2: south + southeast             -> ones2 + 2*twos2

      Then combine the low and high bits of those partial sums:
        (ones0, ones1, ones2) -> count_bit0 + 2*carry2_from_ones
        (twos0, twos1, twos2) -> extra_twos + 2*extra_fours

      Finally:
        count = count_bit0
              + 2*(carry2_from_ones + extra_twos)
              + 4*extra_fours

      So:
        count_bit1 = carry2_from_ones ^ extra_twos
        count_bit2 = extra_fours ^ (carry2_from_ones & extra_twos)

      count_bit3 is not needed:
        - counts 4..7 already have count_bit2 = 1
        - count 8 has count_bit1 = 0
        Therefore the Life rule can be evaluated using only count_bit0/1/2.
    */
    word_t ones0, twos0;
    word_t ones1, twos1;
    csa(northwest, north, northeast, ones0, twos0);
    csa(west, east, southwest,        ones1, twos1);

    const word_t ones2 = south ^ southeast;
    const word_t twos2 = south & southeast;

    word_t count_bit0, carry2_from_ones;
    word_t extra_twos, extra_fours;
    csa(ones0, ones1, ones2, count_bit0,        carry2_from_ones);
    csa(twos0, twos1, twos2, extra_twos,        extra_fours);

    const word_t count_bit1 = carry2_from_ones ^ extra_twos;
    const word_t count_bit2 = extra_fours ^ (carry2_from_ones & extra_twos);

    // Game of Life rule in bit-sliced form:
    //   next = (count == 3) | (current & (count == 2))
    //
    // With the bitboards above:
    //   count == 2 or 3  <=>  count_bit1 & ~count_bit2
    //   count == 3        <=>  count_bit0 == 1
    //
    // Therefore:
    //   next = (count_bit1 & ~count_bit2) & (count_bit0 | current)
    output[idx] = (count_bit1 & ~count_bit2) & (count_bit0 | current);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    /*
      The problem guarantees:
      - grid_dimensions is a power of two
      - grid_dimensions > 512

      Therefore:
      - words_per_row = grid_dimensions / 64 is also a power of two
      - total_words   = grid_dimensions * words_per_row is a power of two
      - with grid_dimensions >= 1024, total_words >= 2^14 and is divisible by 2^8 = 256

      So the kernel launch can cover the grid exactly with 256-thread blocks and
      does not need an in-kernel bounds guard.
    */
    const std::size_t words_per_row =
        static_cast<std::size_t>(grid_dimensions) >> 6;
    const std::size_t total_words =
        words_per_row * static_cast<std::size_t>(grid_dimensions);
    const std::size_t last_row_start = total_words - words_per_row;
    const unsigned int word_mask =
        static_cast<unsigned int>(words_per_row - 1);
    const unsigned int blocks =
        static_cast<unsigned int>(total_words >> kBlockShift);

    // The caller manages synchronization; this function intentionally only enqueues the kernel.
    if (words_per_row == 16u) {
        game_of_life_kernel<16><<<blocks, kBlockSize>>>(
            input, output, words_per_row, last_row_start, word_mask);
    } else {
        game_of_life_kernel<32><<<blocks, kBlockSize>>>(
            input, output, words_per_row, last_row_start, word_mask);
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
