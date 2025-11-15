#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
 * High-performance CUDA implementation of one step of Conway's Game of Life
 * on a bit-packed grid.
 *
 * Representation:
 *   - The grid is N x N cells, N is a power of two (N >= 512).
 *   - Each 64-bit word (std::uint64_t) encodes 64 consecutive cells of a row.
 *   - Bit 0 is the least significant bit; bit 63 is the most significant.
 *   - The grid is stored in row-major order: row 0 words, then row 1 words, etc.
 *
 * Constraints / Strategy:
 *   - Each CUDA thread processes exactly one 64-bit word (64 cells).
 *   - No atomics are needed; each word is written by a single thread.
 *   - Neighbor cells outside the grid boundaries are considered dead (0).
 *   - Only global memory is used; shared/texture memory is unnecessary.
 *
 * Algorithm (bit-parallel, no per-cell loops):
 *
 *   For each word representing the "center" 64 cells:
 *     1. Load up to 9 words: the 3x3 block of words around the center word
 *        (above-left, above, above-right, center-left, center, center-right,
 *         below-left, below, below-right). Missing neighbors at boundaries
 *         are treated as zero.
 *
 *     2. For each of the three rows (above, center, below), compute three
 *        64-bit masks representing neighbors horizontally:
 *
 *           left  = (center_word <<  1) | (left_word >> 63)
 *           mid   =  center_word
 *           right = (center_word >>  1) | (right_word << 63)
 *
 *        For any missing left/right neighbor words, that word is 0, so bits
 *        crossing the word boundary at bit 0 and bit 63 are correctly set
 *        to 0 (cells outside the grid are dead).
 *
 *        For each bit position j in the word, the corresponding bits in
 *        (left, mid, right) encode the three cells in that row at positions
 *        (j-1, j, j+1).
 *
 *     3. For each row, compute the per-cell sum of the three bits (left, mid,
 *        right) using boolean logic in parallel for all 64 cells:
 *
 *           row_ones = left ^ mid ^ right        // 1s bit of sum (0..3)
 *           row_twos = (left & mid) | (left & right) | (mid & right)
 *                  // 2s bit of sum (0..3)
 *
 *        The sum per cell is: row_sum = row_ones + 2 * row_twos (0..3).
 *
 *     4. Combine the three row sums (above, center, below) into a single
 *        per-cell count of all 9 positions (3x3 neighborhood including the
 *        center cell) using bit-sliced addition:
 *
 *        - First, add row A and row B:
 *
 *            // add two 2-bit numbers (0..3) encoded as (ones, twos)
 *            ab_ones = A_ones ^ B_ones          // bit 1
 *            carry1  = A_ones & B_ones          // carry from 1s to 2s place
 *
 *            ab_s2   = A_twos ^ B_twos
 *            carry2  = A_twos & B_twos
 *
 *            ab_twos  = ab_s2 ^ carry1          // bit 2 (2s place)
 *            ab_fours = carry2 | (ab_s2 & carry1) // bit 4 (4s place)
 *
 *          Now AB sum per cell is encoded as:
 *             AB = ab_ones + 2 * ab_twos + 4 * ab_fours  (0..6)
 *
 *        - Then add row C (0..3, encoded as C_ones, C_twos) to AB:
 *
 *            // ones place
 *            s1    = ab_ones ^ C_ones
 *            c1_2  = ab_ones & C_ones            // carry from 1s to 2s
 *
 *            // twos place: add ab_twos, C_twos, c1_2 (three 1-bit inputs)
 *            t1    = ab_twos
 *            t2    = C_twos
 *            t3    = c1_2
 *
 *            s2    = t1 ^ t2 ^ t3                // 2s bit
 *            c2_4  = (t1 & t2) | (t1 & t3) | (t2 & t3)  // carry to 4s
 *
 *            // fours place: add ab_fours and c2_4
 *            s4    = ab_fours ^ c2_4             // 4s bit
 *            s8    = ab_fours & c2_4             // 8s bit
 *
 *          Now total per-cell count of the 9 cells (including the center)
 *          is encoded as:
 *             total9 = s1 + 2 * s2 + 4 * s4 + 8 * s8   (0..9)
 *
 *     5. Game of Life rules depend on the count of the 8 neighbors, not
 *        including the center cell. Because total9 includes the center cell
 *        (when alive), neighbor_count = total9 - center_bit.
 *
 *        Instead of subtracting explicitly, we directly interpret total9:
 *
 *          Let center_bit = 1 if cell is alive, else 0.
 *
 *          neighbors = total9 - center_bit
 *
 *          Cell is alive in next generation if:
 *            - neighbors == 3 (birth or survival), or
 *            - neighbors == 2 and center_bit == 1 (survival).
 *
 *          In terms of total9:
 *            neighbors == 3 and center_bit == 0 -> total9 == 3
 *            neighbors == 2 and center_bit == 1 -> total9 == 3
 *            neighbors == 3 and center_bit == 1 -> total9 == 4
 *
 *          So:
 *            eq3 = (total9 == 3)
 *            eq4 = (total9 == 4)
 *
 *          Next state mask:
 *            next = eq3 | (center & eq4)
 *
 *        We test equality with 3 and 4 using the bitplanes:
 *          total9 == 3 -> 0011b  => s1=1, s2=1, s4=0, s8=0
 *          total9 == 4 -> 0100b  => s1=0, s2=0, s4=1, s8=0
 *
 *          eq3 =  s1 &  s2 & ~s4 & ~s8
 *          eq4 = ~s1 & ~s2 &  s4 & ~s8
 */

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    const int words_per_row = grid_dim >> 6; // grid_dim / 64, grid_dim is power of 2
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             static_cast<std::size_t>(threadIdx.x);
    if (tid >= total_words) {
        return;
    }

    const int row = static_cast<int>(tid / words_per_row);
    const int col = static_cast<int>(tid - static_cast<std::size_t>(row) * words_per_row);

    // Base pointers for this row and its neighbors
    const std::uint64_t* row_ptr      = input + static_cast<std::size_t>(row) * words_per_row;

    const bool has_row_above = (row > 0);
    const bool has_row_below = (row + 1 < grid_dim);
    const bool has_col_left  = (col > 0);
    const bool has_col_right = (col + 1 < words_per_row);

    const std::uint64_t* row_above_ptr = has_row_above
        ? (input + static_cast<std::size_t>(row - 1) * words_per_row)
        : nullptr;
    const std::uint64_t* row_below_ptr = has_row_below
        ? (input + static_cast<std::size_t>(row + 1) * words_per_row)
        : nullptr;

    // Load center word
    const std::uint64_t c11 = row_ptr[col];

    // Initialize neighbors to 0; missing neighbors remain 0 (outside grid => dead)
    std::uint64_t c00 = 0, c01 = 0, c02 = 0;
    std::uint64_t c10 = 0,        c12 = 0;
    std::uint64_t c20 = 0, c21 = 0, c22 = 0;

    // Load neighbors in the same row (left, right)
    if (has_col_left) {
        c10 = row_ptr[col - 1];
    }
    if (has_col_right) {
        c12 = row_ptr[col + 1];
    }

    // Load neighbors in the row above
    if (has_row_above) {
        c01 = row_above_ptr[col];
        if (has_col_left) {
            c00 = row_above_ptr[col - 1];
        }
        if (has_col_right) {
            c02 = row_above_ptr[col + 1];
        }
    }

    // Load neighbors in the row below
    if (has_row_below) {
        c21 = row_below_ptr[col];
        if (has_col_left) {
            c20 = row_below_ptr[col - 1];
        }
        if (has_col_right) {
            c22 = row_below_ptr[col + 1];
        }
    }

    // Build horizontally shifted neighbor masks for each of the three rows.
    //
    // For each mask, bit j corresponds to the cell in the given row at
    // column j + offset, using bits from left/center/right words with
    // cross-word carries at bit 0 and bit 63.
    //
    // Above row
    const std::uint64_t above_left   = (c01 << 1) | (c00 >> 63);
    const std::uint64_t above_mid    =  c01;
    const std::uint64_t above_right  = (c01 >> 1) | (c02 << 63);

    // Current row
    const std::uint64_t curr_left    = (c11 << 1) | (c10 >> 63);
    const std::uint64_t curr_mid     =  c11;
    const std::uint64_t curr_right   = (c11 >> 1) | (c12 << 63);

    // Below row
    const std::uint64_t below_left   = (c21 << 1) | (c20 >> 63);
    const std::uint64_t below_mid    =  c21;
    const std::uint64_t below_right  = (c21 >> 1) | (c22 << 63);

    // Step 1: per-row sums of the three bits (left, mid, right) => values 0..3
    // encoded as (row_ones, row_twos).
    //
    // row_ones:  1s bit of the sum
    // row_twos:  2s bit of the sum

    // Above row
    const std::uint64_t rowA_ones =
        above_left ^ above_mid ^ above_right;
    const std::uint64_t rowA_twos =
        (above_left & above_mid) | (above_left & above_right) | (above_mid & above_right);

    // Current row
    const std::uint64_t rowB_ones =
        curr_left ^ curr_mid ^ curr_right;
    const std::uint64_t rowB_twos =
        (curr_left & curr_mid) | (curr_left & curr_right) | (curr_mid & curr_right);

    // Below row
    const std::uint64_t rowC_ones =
        below_left ^ below_mid ^ below_right;
    const std::uint64_t rowC_twos =
        (below_left & below_mid) | (below_left & below_right) | (below_mid & below_right);

    // Step 2: add rowA and rowB (values 0..3 each) to produce AB (0..6),
    // encoded in three bitplanes: ab_ones (1s), ab_twos (2s), ab_fours (4s).

    // Add ones bits
    const std::uint64_t ab_ones = rowA_ones ^ rowB_ones;
    const std::uint64_t ab_c1   = rowA_ones & rowB_ones; // carry to 2s place

    // Add twos bits plus carry from ones
    const std::uint64_t ab_s2   = rowA_twos ^ rowB_twos;
    const std::uint64_t ab_c2   = rowA_twos & rowB_twos;

    const std::uint64_t ab_twos  = ab_s2 ^ ab_c1;                 // 2s bit
    const std::uint64_t ab_fours = ab_c2 | (ab_s2 & ab_c1);       // 4s bit

    // Step 3: add rowC (0..3) to AB (0..6) to get total9 (0..9), encoded as:
    //   s1 (1s), s2 (2s), s4 (4s), s8 (8s).

    // Ones place: add rowC_ones to ab_ones
    const std::uint64_t s1   = ab_ones ^ rowC_ones;
    const std::uint64_t c1_2 = ab_ones & rowC_ones;    // carry to 2s place

    // Twos place: add ab_twos, rowC_twos, and carry from ones (c1_2).
    const std::uint64_t t1   = ab_twos;
    const std::uint64_t t2   = rowC_twos;
    const std::uint64_t t3   = c1_2;

    const std::uint64_t s2 =
        t1 ^ t2 ^ t3;  // 2s bit of total9
    const std::uint64_t c2_4 =
        (t1 & t2) | (t1 & t3) | (t2 & t3);  // carry to 4s place

    // Fours place: add ab_fours and carry from twos (c2_4)
    const std::uint64_t s4 = ab_fours ^ c2_4;  // 4s bit of total9
    const std::uint64_t s8 = ab_fours & c2_4;  // 8s bit of total9

    // Step 4: Determine cells where total9 == 3 or 4.
    //
    // total9 == 3 -> 0011b => s1=1, s2=1, s4=0, s8=0
    // total9 == 4 -> 0100b => s1=0, s2=0, s4=1, s8=0
    //
    // eq3 =  s1 &  s2 & ~s4 & ~s8
    // eq4 = ~s1 & ~s2 &  s4 & ~s8

    const std::uint64_t not_s1 = ~s1;
    const std::uint64_t not_s2 = ~s2;
    const std::uint64_t not_s4 = ~s4;
    const std::uint64_t not_s8 = ~s8;

    const std::uint64_t eq3 = s1 & s2 & not_s4 & not_s8;
    const std::uint64_t eq4 = not_s1 & not_s2 & s4 & not_s8;

    // Center cell bits (current state)
    const std::uint64_t center = curr_mid;

    // Step 5: Apply Game of Life rules:
    //
    // next = eq3 | (center & eq4)
    //
    // Explanation:
    //   - eq3 is true when total9 == 3:
    //       * If center == 0: neighbors == 3 => birth.
    //       * If center == 1: neighbors == 2 => survival.
    //   - eq4 is true when total9 == 4:
    //       * If center == 1: neighbors == 3 => survival.
    //       * If center == 0: neighbors == 4 => death.
    //
    //   So combining:
    //      next = eq3 | (center & eq4)
    //
    const std::uint64_t next = eq3 | (center & eq4);

    // Write result back to global memory
    output[tid] = next;
}

/*
 * Host-side wrapper to launch one Game of Life step on the GPU.
 *
 * Parameters:
 *   - input:  Device pointer to input grid (bit-packed, std::uint64_t words).
 *   - output: Device pointer to output grid (same layout as input).
 *   - grid_dimensions: Width and height of the square grid in cells (N),
 *                      guaranteed to be a power of 2 and N >= 512.
 *
 * Requirements:
 *   - input and output must be allocated with cudaMalloc.
 *   - No host-device synchronization is performed here; the caller is
 *     responsible for synchronization if needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Configure a 1D grid of threads, each handling one 64-bit word.
    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions);
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
