#include <cstdint>
#include <cuda_runtime.h>

/*
 * Bit-sliced helper: add a 1-bit neighbor mask to a 4-bit per-cell counter.
 *
 * Each of c0, c1, c2, c3 is a 64-bit word that represents one bit-plane
 * of the per-cell neighbor count:
 *   count = c0 * 1 + c1 * 2 + c2 * 4 + c3 * 8   (bitwise, per cell)
 *
 * The input "neighbor" is a 64-bit mask where each bit is either 0 or 1,
 * representing one of the 8 possible neighbor directions.
 *
 * The update implements a ripple-carry adder across the bit-planes using
 * only bitwise operations, so each bit position (cell) is updated
 * independently with no cross-cell carries.
 */
__device__ __forceinline__
void add_neighbor_mask(std::uint64_t neighbor,
                       std::uint64_t &c0,
                       std::uint64_t &c1,
                       std::uint64_t &c2,
                       std::uint64_t &c3)
{
    // Add "neighbor" to the 4-bit counter (c3 c2 c1 c0)
    std::uint64_t carry  = c0 & neighbor;
    c0 ^= neighbor;

    std::uint64_t carry2 = c1 & carry;
    c1 ^= carry;

    std::uint64_t carry3 = c2 & carry2;
    c2 ^= carry2;

    c3 ^= carry3;
}

/*
 * CUDA kernel: one thread processes one 64-bit word (64 cells).
 *
 * Grid layout:
 *   - The full grid is grid_dim x grid_dim cells.
 *   - Each row is packed into grid_dim/64 std::uint64_t words.
 *   - word index = row * words_per_row + col_word
 *
 * Bit layout within a word:
 *   - Bit 0  = leftmost cell in the word
 *   - Bit 63 = rightmost cell in the word
 *
 * Boundary conditions:
 *   - Cells outside the grid are treated as dead (0).
 *   - For the 0th bit, neighbors to the left / diagonally left are in
 *     the words immediately to the left (if any).
 *   - For the 63rd bit, neighbors to the right / diagonally right are in
 *     the words immediately to the right (if any).
 *
 * This kernel:
 *   1. Loads the 3x3 neighborhood of words around the current word.
 *   2. Builds 8 neighbor masks via shifts and cross-word bit injection.
 *   3. Accumulates the 8 neighbor masks into a 4-bit per-cell neighbor
 *      count using bit-sliced addition.
 *   4. Computes masks for "neighbor count is 2 or 3" and "neighbor
 *      count is exactly 3".
 *   5. Applies Game of Life rules to produce the next 64-bit word.
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim)
{
    const int words_per_row = grid_dim >> 6;            // grid_dim / 64; power of two
    const int total_words   = grid_dim * words_per_row;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Because words_per_row is a power of two, we can get the column-in-row
    // index with a simple bitmask instead of an integer modulus.
    const int col_mask = words_per_row - 1;
    const int col      = idx & col_mask;

    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_above = (idx >= words_per_row);
    const bool has_below = (idx + words_per_row < total_words);

    // Load the 3x3 neighborhood of words around (row, col_word).
    const std::uint64_t center = input[idx];

    const std::uint64_t left   = has_left  ? input[idx - 1]               : 0ULL;
    const std::uint64_t right  = has_right ? input[idx + 1]               : 0ULL;
    const std::uint64_t above  = has_above ? input[idx - words_per_row]   : 0ULL;
    const std::uint64_t below  = has_below ? input[idx + words_per_row]   : 0ULL;

    const std::uint64_t above_left  = (has_above && has_left)
                                      ? input[idx - words_per_row - 1]   : 0ULL;
    const std::uint64_t above_right = (has_above && has_right)
                                      ? input[idx - words_per_row + 1]   : 0ULL;
    const std::uint64_t below_left  = (has_below && has_left)
                                      ? input[idx + words_per_row - 1]   : 0ULL;
    const std::uint64_t below_right = (has_below && has_right)
                                      ? input[idx + words_per_row + 1]   : 0ULL;

    // Build 8 neighbor masks. For each direction, the bit at position j in
    // the resulting mask corresponds to the state of that neighbor of cell j.
    //
    // Horizontal neighbors:
    //   - Left neighbors (dx = -1, dy = 0):
    //       nL_j = center_{j-1} for j > 0, and left_{63} for j = 0.
    //       Implemented as (center << 1) plus left >> 63 injection into bit 0.
    //   - Right neighbors (dx = +1, dy = 0):
    //       nR_j = center_{j+1} for j < 63, and right_{0} for j = 63.
    //       Implemented as (center >> 1) plus right << 63 injection into bit 63.
    const std::uint64_t nL = (center << 1) | (left  >> 63);
    const std::uint64_t nR = (center >> 1) | (right << 63);

    // Vertical neighbors (dx = 0, dy = -1 / +1):
    const std::uint64_t nU = above;
    const std::uint64_t nD = below;

    // Diagonal neighbors:
    //   - Up-left (dx = -1, dy = -1):
    //       nUL_j = above_{j-1} for j > 0, and above_left_{63} for j = 0.
    //   - Up-right (dx = +1, dy = -1):
    //       nUR_j = above_{j+1} for j < 63, and above_right_{0} for j = 63.
    //   - Down-left (dx = -1, dy = +1):
    //       nDL_j = below_{j-1} for j > 0, and below_left_{63} for j = 0.
    //   - Down-right (dx = +1, dy = +1):
    //       nDR_j = below_{j+1} for j < 63, and below_right_{0} for j = 63.
    const std::uint64_t nUL = (above << 1) | (above_left  >> 63);
    const std::uint64_t nUR = (above >> 1) | (above_right << 63);
    const std::uint64_t nDL = (below << 1) | (below_left  >> 63);
    const std::uint64_t nDR = (below >> 1) | (below_right << 63);

    // Accumulate the 8 neighbor masks into a 4-bit counter per cell.
    // c0, c1, c2, c3 are 64-bit words forming bit-slices of the neighbor count.
    std::uint64_t c0 = 0ULL;
    std::uint64_t c1 = 0ULL;
    std::uint64_t c2 = 0ULL;
    std::uint64_t c3 = 0ULL;

    add_neighbor_mask(nL,  c0, c1, c2, c3);
    add_neighbor_mask(nR,  c0, c1, c2, c3);
    add_neighbor_mask(nU,  c0, c1, c2, c3);
    add_neighbor_mask(nD,  c0, c1, c2, c3);
    add_neighbor_mask(nUL, c0, c1, c2, c3);
    add_neighbor_mask(nUR, c0, c1, c2, c3);
    add_neighbor_mask(nDL, c0, c1, c2, c3);
    add_neighbor_mask(nDR, c0, c1, c2, c3);

    // We now have neighbor counts in binary:
    //   count = c0 + 2*c1 + 4*c2 + 8*c3.
    //
    // For Game of Life, we only need:
    //   - cells with exactly 2 neighbors (for survival),
    //   - cells with exactly 3 neighbors (for survival + birth),
    // and we don't care about the exact values for other counts.
    //
    // "neighbor count is 2 or 3" => binary 0010 or 0011:
    //   c3 = 0, c2 = 0, c1 = 1, c0 = 0 or 1.
    const std::uint64_t eq23 = (~c3) & (~c2) & c1;
    // "neighbor count is exactly 3" => binary 0011:
    //   same as eq23 plus c0 = 1.
    const std::uint64_t eq3  = eq23 & c0;

    const std::uint64_t alive = center;

    // Game of Life rules in bitwise form:
    //   - A live cell survives if it has 2 or 3 neighbors.
    //   - A dead cell becomes alive if it has exactly 3 neighbors.
    //
    // next = (alive & (count == 2 or 3)) | (!alive & (count == 3))
    const std::uint64_t next = (alive & eq23) | (~alive & eq3);

    output[idx] = next;
}

/*
 * Host wrapper: launch one generation step of Conway's Game of Life.
 *
 * Parameters:
 *   - input:  pointer to device memory holding the current grid, bit-packed.
 *   - output: pointer to device memory where the next grid state is written.
 *   - grid_dimensions: width and height of the square grid (power of 2).
 *
 * The grid is grid_dimensions x grid_dimensions cells, laid out in row-major
 * order and bit-packed so that each std::uint64_t holds 64 consecutive cells
 * within a row. No host-device synchronization is performed here; the caller
 * is responsible for synchronizing as needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;             // grid_dimensions / 64
    const int total_words   = grid_dimensions * words_per_row;

    const int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

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
