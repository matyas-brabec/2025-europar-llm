#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single-step update for Conway's Game of Life
 * on a square grid, bit-packed into 64-bit words (one bit per cell).
 *
 * Representation:
 *  - Grid is grid_dim x grid_dim cells, grid_dim is a power of two (> 512).
 *  - Each row is stored as consecutive std::uint64_t words.
 *  - Bit i (0 <= i < 64) of word w in row r represents cell at (x = 64*w + i, y = r).
 *  - A bit value of 1 means "alive", 0 means "dead".
 *
 * Boundary conditions:
 *  - All cells outside the grid are considered dead.
 *
 * Parallelization strategy:
 *  - Each CUDA thread processes one 64-bit word: 64 cells in a row.
 *  - For each word at position (row, w), the thread reads at most 9 words:
 *    the current word and its neighbors in the 3x3 block of words
 *    (left, center, right in current, above, and below rows).
 *
 * Neighbor counting (bit-parallel):
 *  - For each word, we construct 8 bitboards, each encoding one of the 8
 *    neighbor directions for all 64 cells simultaneously:
 *      NW, N, NE, W, E, SW, S, SE.
 *  - Horizontal adjacency across 64-bit word boundaries is handled by combining
 *    the left/right neighbor words when shifting.
 *  - We then add these 8 one-bit-per-cell bitboards using a per-bit
 *    ripple-carry adder implemented with bitwise operations. This yields the
 *    neighbor count modulo 8 for each cell in three bitboards:
 *      ones  -> bit 0 (1)
 *      twos  -> bit 1 (2)
 *      fours -> bit 2 (4)
 *    Counts are correct modulo 8; since we only need to detect counts 2 and 3,
 *    modulo arithmetic is sufficient (8 neighbors -> 0 mod 8, never misidentified
 *    as 2 or 3).
 *
 * Game of Life rule:
 *  - Let center be the bitboard of current cell states for this 64-cell segment.
 *  - From (ones, twos, fours) we build two masks:
 *      eq2: cells with exactly 2 neighbors (binary 010)
 *      eq3: cells with exactly 3 neighbors (binary 011)
 *  - New state per bit:
 *      new = (neighbors == 3) || (alive && neighbors == 2)
 *           = eq3 | (center & eq2)
 */

/* Device helper: add a bitboard of 0/1 values into a 3-bit per-cell accumulator.
 *
 * Parameters:
 *   n      - input bitboard (each bit is 0 or 1, representing a neighbor presence)
 *   ones   - accumulator LSBs (bit 0 of neighbor count per cell)
 *   twos   - accumulator middle bits (bit 1 of neighbor count per cell)
 *   fours  - accumulator MSBs (bit 2 of neighbor count per cell)
 *
 * After the call, (fours, twos, ones) represent the count modulo 8 of all
 * neighbors added so far at each bit position.
 */
__device__ __forceinline__
void add_bitboard(std::uint64_t n,
                  std::uint64_t &ones,
                  std::uint64_t &twos,
                  std::uint64_t &fours)
{
    // Ripple-carry binary addition of 1-bit n into the 3-bit value (fours, twos, ones)
    // per bit position. All operations are bitwise and therefore operate on 64
    // independent circuits in parallel.

    // Add n to ones (LSB).
    std::uint64_t carry0 = ones & n;
    ones ^= n;

    // Propagate carry into twos (next bit).
    std::uint64_t carry1 = twos & carry0;
    twos ^= carry0;

    // Propagate carry into fours (third bit). We ignore overflow beyond fours (mod 8).
    fours ^= carry1;
}

/* CUDA kernel: compute one Game of Life step on a bit-packed grid. */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row)
{
    // 2D thread coordinates mapped to (word index within row, row index).
    int w   = blockIdx.x * blockDim.x + threadIdx.x; // word index in row
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (w >= words_per_row || row >= grid_dim)
        return;

    // Precompute row base index into the 1D input/output arrays.
    const int row_base = row * words_per_row;

    // Flags for boundary handling.
    const bool has_left_word  = (w > 0);
    const bool has_right_word = (w + 1 < words_per_row);
    const bool has_above      = (row > 0);
    const bool has_below      = (row + 1 < grid_dim);

    // Load current row words: center, left, right.
    const std::uint64_t cC = input[row_base + w];
    const std::uint64_t cL = has_left_word  ? input[row_base + (w - 1)] : 0ull;
    const std::uint64_t cR = has_right_word ? input[row_base + (w + 1)] : 0ull;

    // Load above row words if present.
    std::uint64_t aL = 0ull, aC = 0ull, aR = 0ull;
    if (has_above) {
        const int above_base = row_base - words_per_row;
        aC = input[above_base + w];
        aL = has_left_word  ? input[above_base + (w - 1)] : 0ull;
        aR = has_right_word ? input[above_base + (w + 1)] : 0ull;
    }

    // Load below row words if present.
    std::uint64_t bL = 0ull, bC = 0ull, bR = 0ull;
    if (has_below) {
        const int below_base = row_base + words_per_row;
        bC = input[below_base + w];
        bL = has_left_word  ? input[below_base + (w - 1)] : 0ull;
        bR = has_right_word ? input[below_base + (w + 1)] : 0ull;
    }

    // Construct neighbor bitboards for above row (NW, N, NE).
    // Horizontal neighbors across word boundaries are handled via shifts that
    // include bits from left/right words.
    std::uint64_t above_W = 0ull;
    std::uint64_t above_C = 0ull;
    std::uint64_t above_E = 0ull;
    if (has_above) {
        above_W = (aC << 1) | (aL >> 63);  // cells at (row-1, x-1)
        above_C = aC;                      // cells at (row-1, x)
        above_E = (aC >> 1) | (aR << 63);  // cells at (row-1, x+1)
    }

    // Construct neighbor bitboards for below row (SW, S, SE).
    std::uint64_t below_W = 0ull;
    std::uint64_t below_C = 0ull;
    std::uint64_t below_E = 0ull;
    if (has_below) {
        below_W = (bC << 1) | (bL >> 63);  // cells at (row+1, x-1)
        below_C = bC;                      // cells at (row+1, x)
        below_E = (bC >> 1) | (bR << 63);  // cells at (row+1, x+1)
    }

    // Construct neighbor bitboards for current row (W, E).
    const std::uint64_t curr_W = (cC << 1) | (cL >> 63);  // cells at (row, x-1)
    const std::uint64_t curr_E = (cC >> 1) | (cR << 63);  // cells at (row, x+1)

    // Accumulate neighbor counts (modulo 8) into (ones, twos, fours).
    std::uint64_t ones  = 0ull;
    std::uint64_t twos  = 0ull;
    std::uint64_t fours = 0ull;

    add_bitboard(above_W, ones, twos, fours); // NW
    add_bitboard(above_C, ones, twos, fours); // N
    add_bitboard(above_E, ones, twos, fours); // NE
    add_bitboard(curr_W,  ones, twos, fours); // W
    add_bitboard(curr_E,  ones, twos, fours); // E
    add_bitboard(below_W, ones, twos, fours); // SW
    add_bitboard(below_C, ones, twos, fours); // S
    add_bitboard(below_E, ones, twos, fours); // SE

    // Current cell states for this word.
    const std::uint64_t center = cC;

    // Compute masks for exactly 2 and exactly 3 neighbors.
    // Count bits per cell: (fours, twos, ones) representing 0..7 modulo 8.
    //  - neighbors == 2 -> 010
    //  - neighbors == 3 -> 011
    const std::uint64_t not_fours = ~fours;
    const std::uint64_t eq2 = not_fours &  twos & ~ones;
    const std::uint64_t eq3 = not_fours &  twos &  ones;

    // Apply Game of Life rule:
    // new = (neighbors == 3) || (alive && neighbors == 2)
    const std::uint64_t next = eq3 | (center & eq2);

    // Store updated word.
    output[row_base + w] = next;
}

/*
 * Host wrapper: execute one step of Conway's Game of Life on a bit-packed grid.
 *
 * Parameters:
 *   input           - device pointer to input grid (bit-packed std::uint64_t words)
 *   output          - device pointer to output grid (same layout as input)
 *   grid_dimensions - width and height of the square grid (power of 2, > 512)
 *
 * Notes:
 *   - input and output must be allocated with cudaMalloc and have at least
 *     (grid_dimensions * grid_dimensions / 64) 64-bit elements.
 *   - This function launches the CUDA kernel but does not perform any
 *     synchronization; the caller is responsible for cudaDeviceSynchronize()
 *     or using CUDA events as needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row (grid_dimensions is a power of two, so
    // division by 64 is exact).
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Configure CUDA launch parameters.
    // Using a 2D grid where:
    //   - x-dimension covers words within a row,
    //   - y-dimension covers rows.
    const dim3 blockDim(32, 8); // 256 threads per block
    const dim3 gridDim(
        (words_per_row   + blockDim.x - 1) / blockDim.x,
        (grid_dimensions + blockDim.y - 1) / blockDim.y
    );

    // Launch kernel. Caller handles error checking and synchronization.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output,
                                               grid_dimensions,
                                               words_per_row);
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
