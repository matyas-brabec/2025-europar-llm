#include <cstdint>
#include <cuda_runtime.h>

/*
 * High-performance CUDA implementation of one step of Conway's Game of Life
 * on a bit-packed grid.
 *
 * - Each cell is one bit.
 * - Each std::uint64_t word encodes 64 consecutive cells in the same row.
 * - Each CUDA thread processes exactly one 64-bit word (64 cells).
 *
 * The kernel:
 *   1. Loads up to 9 relevant words from global memory:
 *        - Current row: left, center, right
 *        - Row above:  left, center, right
 *        - Row below:  left, center, right
 *      Missing neighbors at borders are treated as all-dead (zeros).
 *
 *   2. For each of the three rows (above, current, below), it constructs
 *      horizontally shifted versions that already incorporate cross-word
 *      neighbors. For example, for the current row:
 *
 *         curLeft  (W neighbors)  : (cur << 1) with bit 0 filled from left word
 *         cur      (center)       : unchanged
 *         curRight (E neighbors)  : (cur >> 1) with bit 63 filled from right word
 *
 *      Doing this once per row avoids special handling of bit 0 and bit 63
 *      inside the main logic; all 64 bits can then be processed uniformly.
 *
 *   3. Constructs eight 64-bit "neighbor direction" bitboards:
 *        n0: NW, n1: N, n2: NE,
 *        n3: W,        n4: E,
 *        n5: SW, n6: S, n7: SE
 *
 *      Bit i in these words is 1 if the corresponding neighbor in that
 *      direction is alive for cell i in the current word.
 *
 *   4. Computes the neighbor count for all 64 cells in parallel using a
 *      carry-save adder (CSA) tree. The neighbor count (0..8) for each cell
 *      is represented in four bitplanes:
 *
 *         ones   : bit 0 of the count
 *         twos   : bit 1 of the count
 *         fours  : bit 2 of the count
 *         eights : bit 3 of the count
 *
 *      This is equivalent to a per-cell popcount over {n0..n7}, but it is done
 *      for 64 cells at once using only bitwise operations, which is much more
 *      efficient than 64 scalar popcounts per word.
 *
 *   5. Applies the Game of Life rules using these bitplanes:
 *
 *        - For a live cell, it survives if neighbor_count == 2 or 3.
 *        - For a dead cell, it becomes alive if neighbor_count == 3.
 *
 *      Given the bitplane representation, the masks for "count == 2" and
 *      "count == 3" are:
 *
 *        eq2 = (~eights & ~fours &  twos & ~ones)
 *        eq3 = (~eights & ~fours &  twos &  ones)
 *
 *      The next state is:
 *
 *        next = (cur & (eq2 | eq3)) | (~cur & eq3)
 *
 * The host function run_game_of_life() simply configures and launches the
 * kernel for one simulation step. Synchronization is left to the caller.
 */

namespace {

/* Carry-Save Adder: adds three 64-bit bitboards "a", "b", "c" bitwise.
 *
 * For each bit position:
 *   a + b + c = l + 2*h
 *
 * where:
 *   l (low) is the "sum" bit (LSB of the result),
 *   h (high) is the "carry" bit (MSB of the result).
 *
 * There is no carry propagation between bit positions, which is exactly
 * what we want for per-cell independent counts.
 */
__device__ __forceinline__
void csa(std::uint64_t &h, std::uint64_t &l,
         std::uint64_t a,  std::uint64_t b, std::uint64_t c)
{
    std::uint64_t u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

/* Compute the per-cell neighbor count (0..8) from eight 64-bit neighbor
 * direction bitboards (n0..n7).
 *
 * The result is returned in four bitplanes (ones, twos, fours, eights)
 * such that, for each cell:
 *
 *   neighbor_count = ones + 2*twos + 4*fours + 8*eights
 *
 * Given there are at most 8 neighbors, the eights plane is only set when
 * the count is exactly 8.
 */
__device__ __forceinline__
void compute_neighbor_bitplanes(std::uint64_t n0, std::uint64_t n1,
                                std::uint64_t n2, std::uint64_t n3,
                                std::uint64_t n4, std::uint64_t n5,
                                std::uint64_t n6, std::uint64_t n7,
                                std::uint64_t &ones,
                                std::uint64_t &twos,
                                std::uint64_t &fours,
                                std::uint64_t &eights)
{
    // Stage 1: compress 8 inputs into 6 using CSAs
    std::uint64_t c01, s01;
    std::uint64_t c23, s23;
    std::uint64_t c45, s45;
    csa(c01, s01, n0, n1, n2);   // n0 + n1 + n2 = s01 + 2*c01
    csa(c23, s23, n3, n4, n5);   // n3 + n4 + n5 = s23 + 2*c23
    csa(c45, s45, n6, n7, 0ull); // n6 + n7 + 0  = s45 + 2*c45

    // Stage 2: compress sums and carries separately
    std::uint64_t c67, s67;
    std::uint64_t c89, s89;
    csa(c67, s67, s01, s23, s45); // s01 + s23 + s45 = s67 + 2*c67
    csa(c89, s89, c01, c23, c45); // c01 + c23 + c45 = s89 + 2*c89

    // Stage 3: finalize bitplanes
    // Total sum S:
    //   S = s67 + 2*(c67 + s89) + 4*c89
    //
    // Let T = c67 + s89.
    //   T = t0 + 2*carry_t0
    //
    // Then:
    //   S = s67 + 2*t0 + 4*(carry_t0 + c89)
    //
    // Let U = carry_t0 + c89.
    //   U = t1 + 2*carry_t1
    //
    // So:
    //   S = ones + 2*twos + 4*fours + 8*eights
    // with:
    //   ones   = s67
    //   twos   = t0
    //   fours  = t1
    //   eights = carry_t1

    ones = s67;

    std::uint64_t t0        = c67 ^ s89;
    std::uint64_t carry_t0  = c67 & s89;
    twos = t0;

    std::uint64_t t1        = carry_t0 ^ c89;
    std::uint64_t carry_t1  = carry_t0 & c89;
    fours  = t1;
    eights = carry_t1;
}

/* CUDA kernel: perform one Game of Life step on a bit-packed square grid.
 *
 * Parameters:
 *   input         - device pointer to input grid (bit-packed)
 *   output        - device pointer to output grid (bit-packed)
 *   grid_dim      - width/height of the square grid in cells (power of 2)
 *   words_per_row - grid_dim / 64
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row)
{
    std::size_t global_word_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dim);

    if (global_word_idx >= total_words) {
        return;
    }

    // Compute row and column (word index) within that row
    int row       = static_cast<int>(global_word_idx / words_per_row);
    int word_col  = static_cast<int>(global_word_idx % words_per_row);

    const bool has_left   = (word_col > 0);
    const bool has_right  = (word_col + 1 < words_per_row);
    const bool has_above  = (row > 0);
    const bool has_below  = (row + 1 < grid_dim);

    const std::size_t row_base = static_cast<std::size_t>(row) * words_per_row;

    // Current row: left, center, right words
    std::uint64_t cur_left_word   = has_left  ? input[row_base + word_col - 1] : 0ull;
    std::uint64_t cur_center_word =            input[row_base + word_col];
    std::uint64_t cur_right_word  = has_right ? input[row_base + word_col + 1] : 0ull;

    // Above row: left, center, right words (if exists)
    std::uint64_t above_left_word   = 0ull;
    std::uint64_t above_center_word = 0ull;
    std::uint64_t above_right_word  = 0ull;
    if (has_above) {
        const std::size_t above_base = static_cast<std::size_t>(row - 1) * words_per_row;
        if (has_left) {
            above_left_word = input[above_base + word_col - 1];
        }
        above_center_word = input[above_base + word_col];
        if (has_right) {
            above_right_word = input[above_base + word_col + 1];
        }
    }

    // Below row: left, center, right words (if exists)
    std::uint64_t below_left_word   = 0ull;
    std::uint64_t below_center_word = 0ull;
    std::uint64_t below_right_word  = 0ull;
    if (has_below) {
        const std::size_t below_base = static_cast<std::size_t>(row + 1) * words_per_row;
        if (has_left) {
            below_left_word = input[below_base + word_col - 1];
        }
        below_center_word = input[below_base + word_col];
        if (has_right) {
            below_right_word = input[below_base + word_col + 1];
        }
    }

    // Build horizontally shifted versions of each row that already include
    // cross-word neighbors. After this step, all 64 bits can be processed
    // uniformly without special casing bit 0 / bit 63 for left/right neighbors.

    // Current row
    std::uint64_t cur_row   = cur_center_word;
    std::uint64_t cur_left  = cur_row << 1;
    std::uint64_t cur_right = cur_row >> 1;
    if (has_left) {
        cur_left |= (cur_left_word >> 63);           // fill bit 0 from left word's MSB
    }
    if (has_right) {
        cur_right |= (cur_right_word & 1ull) << 63;  // fill bit 63 from right word's LSB
    }

    // Above row (if any)
    std::uint64_t above_row   = 0ull;
    std::uint64_t above_left  = 0ull;
    std::uint64_t above_right = 0ull;
    if (has_above) {
        above_row  = above_center_word;
        above_left = above_row << 1;
        above_right = above_row >> 1;
        if (has_left) {
            above_left |= (above_left_word >> 63);
        }
        if (has_right) {
            above_right |= (above_right_word & 1ull) << 63;
        }
    }

    // Below row (if any)
    std::uint64_t below_row   = 0ull;
    std::uint64_t below_left  = 0ull;
    std::uint64_t below_right = 0ull;
    if (has_below) {
        below_row  = below_center_word;
        below_left = below_row << 1;
        below_right = below_row >> 1;
        if (has_left) {
            below_left |= (below_left_word >> 63);
        }
        if (has_right) {
            below_right |= (below_right_word & 1ull) << 63;
        }
    }

    // Construct neighbor direction bitboards
    const std::uint64_t n0 = above_left;   // NW
    const std::uint64_t n1 = above_row;    // N
    const std::uint64_t n2 = above_right;  // NE
    const std::uint64_t n3 = cur_left;     // W
    const std::uint64_t n4 = cur_right;    // E
    const std::uint64_t n5 = below_left;   // SW
    const std::uint64_t n6 = below_row;    // S
    const std::uint64_t n7 = below_right;  // SE

    // Compute neighbor count bitplanes for all 64 cells in this word
    std::uint64_t ones, twos, fours, eights;
    compute_neighbor_bitplanes(n0, n1, n2, n3, n4, n5, n6, n7,
                               ones, twos, fours, eights);

    // Derive masks for "neighbor count == 2" and "neighbor count == 3".
    // Counts (0..8) are represented as:
    //   count = ones + 2*twos + 4*fours + 8*eights
    //
    // Since count never exceeds 8, the patterns we care about are:
    //   count == 2: 0010  (twos=1, others=0)
    //   count == 3: 0011  (twos=1, ones=1, others=0)

    const std::uint64_t not_eights = ~eights;
    const std::uint64_t not_fours  = ~fours;
    const std::uint64_t not_ones   = ~ones;

    const std::uint64_t eq2 =
        not_eights & not_fours & twos & not_ones;
    const std::uint64_t eq3 =
        not_eights & not_fours & twos & ones;

    // Apply Game of Life rules
    const std::uint64_t cur = cur_row;

    const std::uint64_t survive = cur & (eq2 | eq3); // alive and (2 or 3 neighbors)
    const std::uint64_t born    = (~cur) & eq3;      // dead and exactly 3 neighbors

    const std::uint64_t next = survive | born;

    output[global_word_idx] = next;
}

} // namespace

/* Host wrapper that launches one simulation step on the GPU.
 *
 * Parameters:
 *   input           - device pointer to input grid (bit-packed)
 *   output          - device pointer to output grid (bit-packed)
 *   grid_dimensions - width/height of the square grid in cells
 *
 * Notes:
 *   - grid_dimensions is guaranteed to be a power of 2, > 512.
 *   - Each word encodes 64 cells horizontally, so words_per_row = grid_dimensions / 64.
 *   - The caller is responsible for any necessary host-device synchronization.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row =
        grid_dimensions >> 6; // grid_dimensions / 64, safe since it's a power of 2 >= 512
    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) *
        static_cast<std::size_t>(grid_dimensions);

    const int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    game_of_life_kernel<<<blocks, threads_per_block>>>(
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
