#include <cstdint>
#include <cuda_runtime.h>

/*
 * Bit-parallel neighbor counter for Conway's Game of Life.
 *
 * For each bit position (lane), we maintain four bitboards:
 *   - ones   : 1s place (2^0)
 *   - twos   : 2s place (2^1)
 *   - fours  : 4s place (2^2)
 *   - eights : 8s place (2^3)
 *
 * Each call adds a 0/1 value per lane (from bitboard b) into this
 * per-lane binary counter using full-adder logic, without any carries
 * propagating between bit positions.
 *
 * After adding all 8 neighbor bitboards, the per-lane neighbor count
 * (0..8) is encoded in (ones, twos, fours, eights) as a 4-bit binary number.
 */
__device__ __forceinline__
void add_neighbor_bitboard(std::uint64_t b,
                           std::uint64_t &ones,
                           std::uint64_t &twos,
                           std::uint64_t &fours,
                           std::uint64_t &eights)
{
    std::uint64_t new_ones  = ones ^ b;       // sum bit (1s) of ones + b
    std::uint64_t carry1    = ones & b;       // carry to 2s place

    std::uint64_t new_twos  = twos ^ carry1;  // sum bit (2s) of twos + carry1
    std::uint64_t carry2    = twos & carry1;  // carry to 4s place

    std::uint64_t new_fours = fours ^ carry2; // sum bit (4s) of fours + carry2
    std::uint64_t carry3    = fours & carry2; // carry to 8s place

    eights |= carry3;                         // accumulate possible 8s bit

    ones = new_ones;
    twos = new_twos;
    fours = new_fours;
}

/*
 * Kernel: one thread processes one 64-bit word (64 cells) of the grid.
 *
 * Grid layout:
 *   - The logical grid is grid_dimensions x grid_dimensions cells.
 *   - Each row is packed into (grid_dimensions / 64) 64-bit words.
 *   - Words are stored in row-major order.
 *
 * The kernel:
 *   - Loads the word corresponding to the current thread (center).
 *   - Loads its neighboring words (left/right, above/below, and diagonals).
 *   - Constructs 8 bitboards, each representing one of the 8 neighbor
 *     directions for all 64 bits in the word.
 *   - Uses a bit-parallel binary counter to compute neighbor count per bit.
 *   - Applies Conway's Game of Life rules:
 *       * A live cell survives with 2 or 3 neighbors.
 *       * A dead cell becomes live with exactly 3 neighbors.
 *   - Writes the next generation word to the output.
 *
 * Boundary handling:
 *   - Out-of-grid cells are treated as dead (0).
 *   - This is handled by substituting 0 for any neighbor words that would
 *     fall outside the grid (top/bottom rows, leftmost/rightmost columns).
 *
 * Indexing:
 *   - words_per_row = grid_dimensions / 64, guaranteed to be a power of two.
 *   - total_words   = words_per_row * grid_dimensions.
 *   - Thread index idx in [0, total_words) processes input[idx].
 *   - row = idx >> log2_words_per_row
 *   - col = idx & (words_per_row - 1)
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int words_per_row,
                         int grid_dimensions,
                         int log2_words_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = words_per_row * grid_dimensions;
    if (idx >= total_words) {
        return;
    }

    // Decode 2D coordinates from linear word index, using the fact that
    // words_per_row is a power of two.
    int row = idx >> log2_words_per_row;
    int col = idx & (words_per_row - 1);

    int row_offset = row * words_per_row;
    const std::uint64_t* row_ptr = input + row_offset;

    // Current row words: left, center, right.
    std::uint64_t center_word = row_ptr[col];
    std::uint64_t left_word   = (col > 0) ? row_ptr[col - 1] : 0ull;
    std::uint64_t right_word  = (col + 1 < words_per_row) ? row_ptr[col + 1] : 0ull;

    // Pointers to rows above and below (nullptr if outside grid).
    const std::uint64_t* row_above_ptr =
        (row > 0) ? (row_ptr - words_per_row) : nullptr;
    const std::uint64_t* row_below_ptr =
        (row + 1 < grid_dimensions) ? (row_ptr + words_per_row) : nullptr;

    // Neighboring words from the row above: left, center, right.
    std::uint64_t above_left_word   = 0ull;
    std::uint64_t above_center_word = 0ull;
    std::uint64_t above_right_word  = 0ull;
    if (row_above_ptr != nullptr) {
        above_center_word = row_above_ptr[col];
        above_left_word   = (col > 0) ? row_above_ptr[col - 1] : 0ull;
        above_right_word  = (col + 1 < words_per_row) ? row_above_ptr[col + 1] : 0ull;
    }

    // Neighboring words from the row below: left, center, right.
    std::uint64_t below_left_word   = 0ull;
    std::uint64_t below_center_word = 0ull;
    std::uint64_t below_right_word  = 0ull;
    if (row_below_ptr != nullptr) {
        below_center_word = row_below_ptr[col];
        below_left_word   = (col > 0) ? row_below_ptr[col - 1] : 0ull;
        below_right_word  = (col + 1 < words_per_row) ? row_below_ptr[col + 1] : 0ull;
    }

    // Construct bitboards for neighbors in the current row (left/right).
    //
    // For each bit position i (0..63):
    //   neighbors_left[i]  = cell(row, col_word, i-1)
    //   neighbors_right[i] = cell(row, col_word, i+1)
    //
    // The cross-word neighbors for bit 0 and 63 are handled via left_word
    // and right_word, respectively.
    std::uint64_t neighbors_left  = (center_word << 1) | (left_word >> 63);
    std::uint64_t neighbors_right = (center_word >> 1) | (right_word << 63);

    // Construct bitboards for neighbors from the row above:
    //
    // For each bit position i:
    //   neighbors_above_left[i]  = cell(row-1, col_word, i-1)
    //   neighbors_above[i]       = cell(row-1, col_word, i)
    //   neighbors_above_right[i] = cell(row-1, col_word, i+1)
    //
    // Cross-word neighbors (for bit 0 and 63) use above_left_word and
    // above_right_word. If row == 0, all these are zero.
    std::uint64_t neighbors_above_left  = 0ull;
    std::uint64_t neighbors_above       = 0ull;
    std::uint64_t neighbors_above_right = 0ull;
    if (row_above_ptr != nullptr) {
        neighbors_above_left  = (above_center_word << 1) | (above_left_word >> 63);
        neighbors_above       =  above_center_word;
        neighbors_above_right = (above_center_word >> 1) | (above_right_word << 63);
    }

    // Construct bitboards for neighbors from the row below (analogous to above).
    std::uint64_t neighbors_below_left  = 0ull;
    std::uint64_t neighbors_below       = 0ull;
    std::uint64_t neighbors_below_right = 0ull;
    if (row_below_ptr != nullptr) {
        neighbors_below_left  = (below_center_word << 1) | (below_left_word >> 63);
        neighbors_below       =  below_center_word;
        neighbors_below_right = (below_center_word >> 1) | (below_right_word << 63);
    }

    // Bit-parallel neighbor counting.
    //
    // We sum the 8 neighbor bitboards:
    //   neighbors_above_left, neighbors_above, neighbors_above_right,
    //   neighbors_left, neighbors_right,
    //   neighbors_below_left, neighbors_below, neighbors_below_right
    //
    // into a 4-bit per-cell counter stored in ones, twos, fours, eights.
    std::uint64_t ones   = 0ull;
    std::uint64_t twos   = 0ull;
    std::uint64_t fours  = 0ull;
    std::uint64_t eights = 0ull;

    add_neighbor_bitboard(neighbors_above_left,  ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_above,       ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_above_right, ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_left,        ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_right,       ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_below_left,  ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_below,       ones, twos, fours, eights);
    add_neighbor_bitboard(neighbors_below_right, ones, twos, fours, eights);

    // Compute masks for "exactly 2 neighbors" and "exactly 3 neighbors".
    //
    // After accumulation, each bit position's neighbor count N (0..8) is:
    //   N = ones*1 + twos*2 + fours*4 + eights*8
    //
    // We need:
    //   eq2: N == 2  -> ones=0, twos=1, fours=0, eights=0
    //   eq3: N == 3  -> ones=1, twos=1, fours=0, eights=0
    //
    // Thus:
    //   not_fours_or_eights = ~(fours | eights)
    //   eq3 = ones & twos & not_fours_or_eights
    //   eq2 = (~ones) & twos & not_fours_or_eights
    std::uint64_t not_fours_or_eights = ~(fours | eights);

    std::uint64_t eq3 =  ones &  twos & not_fours_or_eights;
    std::uint64_t eq2 = (~ones) & twos & not_fours_or_eights;

    // Apply Conway's Game of Life rules:
    //
    // - A dead cell with exactly 3 neighbors becomes alive.
    // - A live cell with 2 or 3 neighbors survives.
    //
    // Let C be the current state bitboard (center_word).
    // New state bitboard:
    //   new = eq3 | (eq2 & C)
    //
    // Explanation:
    //   - eq3 covers both births (dead->alive) and survival with 3 neighbors.
    //   - eq2 & C covers survival with 2 neighbors (only for live cells).
    std::uint64_t new_state = eq3 | (eq2 & center_word);

    output[idx] = new_state;
}

/*
 * Host function: run one step of Conway's Game of Life on the GPU.
 *
 * Arguments:
 *   - input           : device pointer to input grid, bit-packed
 *                       (std::uint64_t, 64 cells per word).
 *   - output          : device pointer to output grid (same layout).
 *   - grid_dimensions : width and height of the square grid (cells),
 *                       guaranteed to be a power of two and > 512.
 *
 * Assumptions:
 *   - input and output are allocated via cudaMalloc and large enough to hold
 *     grid_dimensions * grid_dimensions cells in bit-packed form.
 *   - No host-device synchronization is performed here; the caller is
 *     responsible for synchronization if needed.
 *
 * Implementation details:
 *   - words_per_row = grid_dimensions / 64, which is also a power of two.
 *   - We precompute log2_words_per_row on the host so that the device kernel
 *     can decode (row, col) from the linear index using shifts and bitmasks,
 *     avoiding integer division/modulo in the kernel.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row (grid_dimensions is a power of two).
    int words_per_row = grid_dimensions >> 6;  // divide by 64

    // Total number of 64-bit words in the grid.
    int total_words = words_per_row * grid_dimensions;

    // Compute log2(words_per_row) on the host (words_per_row is a power of two).
    int log2_words_per_row = 0;
    int tmp = words_per_row;
    while (tmp > 1) {
        tmp >>= 1;
        ++log2_words_per_row;
    }

    // Launch configuration: 1D grid, 1D blocks.
    // 256 threads per block is a good compromise for modern NVIDIA GPUs.
    int block_size = 256;
    int grid_size  = (total_words + block_size - 1) / block_size;

    game_of_life_kernel<<<grid_size, block_size>>>(
        input,
        output,
        words_per_row,
        grid_dimensions,
        log2_words_per_row
    );
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
