#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single-step update of Conway's Game of Life
 * on a bit-packed grid.
 *
 * Data layout:
 * - The grid is square with dimensions N x N, where N is a power of 2 (> 512).
 * - Each std::uint64_t word stores 64 consecutive cells in the same row.
 *   Bit 0 (LSB) is the leftmost cell of the 64-cell block, bit 63 the rightmost.
 * - The grid is stored in row-major order: row 0 words, then row 1 words, etc.
 *
 * Execution strategy:
 * - Each CUDA thread processes exactly one 64-bit word (64 cells).
 * - Threads do not interact; no atomics are needed.
 * - For the word at (row, col), the thread loads up to 9 words:
 *     (row-1, col-1), (row-1, col), (row-1, col+1)
 *     (row,   col-1), (row,   col), (row,   col+1)
 *     (row+1, col-1), (row+1, col), (row+1, col+1)
 *   (where out-of-bounds accesses are treated as zero / absent neighbors).
 *
 * Neighborhood computation:
 * - From the three rows (above, current, below) and their adjacent words,
 *   we construct eight 64-bit "direction" bit-planes:
 *     nw, n, ne, w, e, sw, s, se
 *   Each bit position j (0..63) in these words represents the neighbor in
 *   that direction for the cell at bit j in the current word.
 * - Horizontal neighbors (W/E) and diagonal neighbors (NW/NE/SW/SE) are formed
 *   by shifting the row words left/right by 1 bit and patching the bit 0 / 63
 *   using the left/right neighbor words (to avoid cross-bit contamination).
 * - All cells outside the global N x N grid are treated as dead (0).
 *
 * Neighbor counting:
 * - For each bit position (cell), we need the count of 8 neighbors (0..8).
 * - We compute this count for all 64 cells in parallel using bit-sliced
 *   arithmetic:
 *     - Maintain four 64-bit accumulators: ones, twos, fours, eights.
 *       For each bit position j these store the 4-bit binary representation
 *       of the neighbor count at that position:
 *         count[j] = ones[j]*1 + twos[j]*2 + fours[j]*4 + eights[j]*8
 *     - Add each of the eight neighbor bit-planes (nw, n, ne, w, e, sw, s, se)
 *       into the accumulators using ripple-carry binary addition expressed
 *       entirely in bitwise operations. This ensures that each bit position
 *       (cell) is updated independently (no cross-cell carry propagation).
 *
 * Applying the Game of Life rules:
 * - Let center be the current 64-bit word of cell states.
 * - From the accumulators we derive two 64-bit masks:
 *     eq2: bits where neighbor count == 2
 *     eq3: bits where neighbor count == 3
 * - The update rule per cell is:
 *     - If the cell is alive and has 2 or 3 neighbors, it survives.
 *     - If the cell is dead and has exactly 3 neighbors, it becomes alive.
 *   In bitwise form for entire word:
 *     next = eq3 | (eq2 & center)
 *   (eq3 covers both births (dead->alive) and survival of cells with 3 neighbors.
 *    eq2 & center covers survival of cells with exactly 2 neighbors.)
 */

/* 
 * Bit-sliced addition of a single 1-bit "plane" v into the multi-bit
 * neighbor count represented by (ones, twos, fours, eights).
 *
 * Each argument is a 64-bit word. For each bit position j, we treat:
 *   count[j] = ones[j] + 2*twos[j] + 4*fours[j] + 8*eights[j]
 * v[j] is 0 or 1 and we want:
 *   count'[j] = count[j] + v[j]
 *
 * We implement a ripple-carry adder in bit-sliced form:
 * - First add v to ones, generating carry0 into twos.
 * - Add carry0 to twos, generating carry1 into fours.
 * - Add carry1 to fours, generating carry2 into eights.
 * - Add carry2 to eights (no further carry needed since max count is 8).
 */
__device__ __forceinline__
void add_bitplane(std::uint64_t v,
                  std::uint64_t &ones,
                  std::uint64_t &twos,
                  std::uint64_t &fours,
                  std::uint64_t &eights)
{
    // Add v into the least significant bit (ones)
    std::uint64_t carry0 = ones & v;
    ones ^= v;

    // Propagate carry into the 2's bit (twos)
    std::uint64_t carry1 = twos & carry0;
    twos ^= carry0;

    // Propagate carry into the 4's bit (fours)
    std::uint64_t carry2 = fours & carry1;
    fours ^= carry1;

    // Propagate carry into the 8's bit (eights)
    eights ^= carry2;
}

/*
 * Kernel that computes one Game of Life step on a bit-packed grid.
 *
 * Parameters:
 *   input      - pointer to input grid (N x N cells bit-packed into uint64_t)
 *   output     - pointer to output grid (same layout as input)
 *   grid_dim   - N, width and height of the square grid
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim)
{
    const int words_per_row = grid_dim >> 6;  // N / 64
    const int total_words   = words_per_row * grid_dim;

    const int word_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (word_idx >= total_words)
        return;

    const int row = word_idx / words_per_row;
    const int col = word_idx - row * words_per_row;

    // Load center word for current row and its horizontal neighbors.
    // For boundary columns, left/right words are treated as zero.
    const std::uint64_t center = input[word_idx];
    std::uint64_t left  = 0;
    std::uint64_t right = 0;

    if (col > 0) {
        left = input[word_idx - 1];
    }
    if (col + 1 < words_per_row) {
        right = input[word_idx + 1];
    }

    // Load words from the row above (if any)
    std::uint64_t above    = 0;
    std::uint64_t above_l  = 0;
    std::uint64_t above_r  = 0;
    if (row > 0) {
        const int above_idx = word_idx - words_per_row;
        above = input[above_idx];
        if (col > 0) {
            above_l = input[above_idx - 1];
        }
        if (col + 1 < words_per_row) {
            above_r = input[above_idx + 1];
        }
    }

    // Load words from the row below (if any)
    std::uint64_t below    = 0;
    std::uint64_t below_l  = 0;
    std::uint64_t below_r  = 0;
    if (row + 1 < grid_dim) {
        const int below_idx = word_idx + words_per_row;
        below = input[below_idx];
        if (col > 0) {
            below_l = input[below_idx - 1];
        }
        if (col + 1 < words_per_row) {
            below_r = input[below_idx + 1];
        }
    }

    // Construct eight neighbor direction bit-planes.
    //
    // Bit mapping convention:
    //   - Bit i (0..63) corresponds to column (col*64 + i).
    //   - For the cell at bit i:
    //       west neighbor is bit (i - 1),
    //       east neighbor is bit (i + 1).
    //
    // Within a word, we obtain these by shifting by ±1. For bit 0 and bit 63,
    // neighbors may cross word boundaries, which we fix using left/right words.
    //
    // 'above', 'center', and 'below' correspond to N, C, S rows, respectively.
    std::uint64_t n  = above;
    std::uint64_t s  = below;
    std::uint64_t w  = center << 1;
    std::uint64_t e  = center >> 1;

    std::uint64_t nw = above << 1;
    std::uint64_t ne = above >> 1;
    std::uint64_t sw = below << 1;
    std::uint64_t se = below >> 1;

    // Patch cross-word neighbors for bit 0 and bit 63.
    if (col > 0) {
        const std::uint64_t left_msb_a = above_l >> 63;
        const std::uint64_t left_msb_c = left     >> 63;
        const std::uint64_t left_msb_b = below_l  >> 63;

        nw |= left_msb_a;  // NW neighbor for bit 0 from above_l bit 63
        w  |= left_msb_c;  // W  neighbor for bit 0 from left     bit 63
        sw |= left_msb_b;  // SW neighbor for bit 0 from below_l  bit 63
    }

    if (col + 1 < words_per_row) {
        const std::uint64_t right_lsb_a = above_r & 1u;
        const std::uint64_t right_lsb_c = right   & 1u;
        const std::uint64_t right_lsb_b = below_r & 1u;

        ne |= right_lsb_a << 63;  // NE neighbor for bit 63 from above_r bit 0
        e  |= right_lsb_c << 63;  // E  neighbor for bit 63 from right   bit 0
        se |= right_lsb_b << 63;  // SE neighbor for bit 63 from below_r bit 0
    }

    // Bit-sliced neighbor count accumulators.
    std::uint64_t ones   = 0;
    std::uint64_t twos   = 0;
    std::uint64_t fours  = 0;
    std::uint64_t eights = 0;

    // Add all eight neighbor direction planes.
    add_bitplane(n,  ones, twos, fours, eights);
    add_bitplane(s,  ones, twos, fours, eights);
    add_bitplane(e,  ones, twos, fours, eights);
    add_bitplane(w,  ones, twos, fours, eights);
    add_bitplane(ne, ones, twos, fours, eights);
    add_bitplane(nw, ones, twos, fours, eights);
    add_bitplane(se, ones, twos, fours, eights);
    add_bitplane(sw, ones, twos, fours, eights);

    // At this point, for each bit position j, the neighbor count is:
    //   neighbors[j] = ones[j] + 2*twos[j] + 4*fours[j] + 8*eights[j]
    // We only need to distinguish exact counts of 2 and 3.
    const std::uint64_t not_ones   = ~ones;
    const std::uint64_t not_fours  = ~fours;
    const std::uint64_t not_eights = ~eights;

    // Mask where neighbor count == 3 (ones=1, twos=1, fours=0, eights=0)
    const std::uint64_t eq3 = ones & twos & not_fours & not_eights;

    // Mask where neighbor count == 2 (ones=0, twos=1, fours=0, eights=0)
    const std::uint64_t eq2 = not_ones & twos & not_fours & not_eights;

    // Game of Life update:
    // new_cell = (neighbors == 3) OR (center == 1 AND neighbors == 2)
    const std::uint64_t next = eq3 | (eq2 & center);

    output[word_idx] = next;
}

/*
 * Host wrapper that launches one Game of Life step on the GPU.
 *
 * Parameters:
 *   input            - device pointer to input grid (bit-packed)
 *   output           - device pointer to output grid (bit-packed)
 *   grid_dimensions  - N, width and height of the square grid
 *
 * Notes:
 * - input and output must already be allocated on the device via cudaMalloc.
 * - This function does not perform any stream or device synchronization;
 *   the caller is responsible for synchronization as needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    // Use a reasonably large block size to maximize occupancy while keeping
    // register usage under control on modern data-center GPUs (A100/H100).
    const int block_size = 256;
    const int grid_size  = (total_words + block_size - 1) / block_size;

    game_of_life_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
