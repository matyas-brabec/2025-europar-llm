// Conway's Game of Life - CUDA implementation.
//
// Each CUDA thread processes one 64-bit word, which encodes 64 consecutive
// cells in a row of the 2D grid, one bit per cell (LSB = leftmost).
//
// Grid layout:
//   - Grid is square with dimensions grid_dimensions x grid_dimensions.
//   - grid_dimensions is a power of 2, >= 512, and a multiple of 64.
//   - Each row has (grid_dimensions / 64) 64-bit words.
//   - Cells are stored in row-major order in a linear array.
//
// Neighbor handling:
//   - Cells outside the grid are treated as dead (0).
//   - To compute neighbors for the 64 cells in a word, each thread loads up to
//     9 words from global memory:
//       * 3 from the row above: left, center, right
//       * 3 from the current row: left, center, right
//       * 3 from the row below: left, center, right
//     Missing neighbors at edges are treated as 0.
//   - Horizontal neighbors (left/right) are constructed with bit shifts, using
//     bits from adjacent words to handle the 0th and 63rd bits correctly.
//
// Neighbor counting:
//   - For each 64-bit word, we build 8 "neighbor bitboards", one for each
//     neighbor direction (N, NE, E, SE, S, SW, W, NW). Each bitboard has a 1
//     at positions where that neighbor exists and is alive.
//   - We then count neighbors per cell using a bit-sliced binary counter:
//       ones, twos, fours : 64-bit masks
//     representing the neighbor count in binary:
//       count = ones + 2*twos + 4*fours  (mod 8)
//   - We add the 8 neighbor bitboards into (ones, twos, fours) using simple
//     bit operations that emulate binary addition, without per-cell loops.
//   - Because the true neighbor count is in [0, 8], the low three bits
//     (mod 8) uniquely determine whether the count is 2 or 3:
//       2 -> 010  (ones=0, twos=1, fours=0)
//       3 -> 011  (ones=1, twos=1, fours=0)
//     Higher counts (4-8) have fours=1 or different patterns, so they are
//     distinguishable from 2 and 3.
//
// Game of Life rule application:
//   - alive_next = (neighbors == 3) OR (alive_current AND neighbors == 2)
//   - Using the bit-sliced counters:
//       neighbor_eq_2 = twos & ~ones & ~fours
//       neighbor_eq_3 = twos &  ones & ~fours
//       next_word     = (neighbor_eq_3) | (neighbor_eq_2 & current_word)

#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel: compute one Game of Life step on a bit-packed grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = grid_dimensions * words_per_row;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) {
        return;
    }

    const int row = idx / words_per_row;
    const int col = idx - row * words_per_row;

    // Current cell word (64 cells).
    const std::uint64_t center = input[idx];

    // Neighbor words; initialize to 0 for out-of-bounds regions.
    std::uint64_t left        = 0;
    std::uint64_t right       = 0;
    std::uint64_t above       = 0;
    std::uint64_t above_left  = 0;
    std::uint64_t above_right = 0;
    std::uint64_t below       = 0;
    std::uint64_t below_left  = 0;
    std::uint64_t below_right = 0;

    // Load left and right neighbors in the current row (if they exist).
    if (col > 0) {
        left = input[idx - 1];
    }
    if (col + 1 < words_per_row) {
        right = input[idx + 1];
    }

    // Load neighbors from the row above.
    if (row > 0) {
        const int above_idx = idx - words_per_row;
        above = input[above_idx];

        if (col > 0) {
            above_left = input[above_idx - 1];
        }
        if (col + 1 < words_per_row) {
            above_right = input[above_idx + 1];
        }
    }

    // Load neighbors from the row below.
    if (row + 1 < grid_dimensions) {
        const int below_idx = idx + words_per_row;
        below = input[below_idx];

        if (col > 0) {
            below_left = input[below_idx - 1];
        }
        if (col + 1 < words_per_row) {
            below_right = input[below_idx + 1];
        }
    }

    // Construct horizontally shifted neighbor boards for each of the three rows.
    //
    // For a row with (left_word, center_word, right_word):
    //   left_shift  [bit i] = cell at column i-1 (or 0 at left edge)
    //   right_shift [bit i] = cell at column i+1 (or 0 at right edge)
    //
    // left_shift is computed as:
    //   (center_word << 1) | (left_word >> 63)  if there is a left word
    //   (center_word << 1)                       otherwise
    //
    // right_shift is computed as:
    //   (center_word >> 1) | (right_word << 63) if there is a right word
    //   (center_word >> 1)                      otherwise

    const std::uint64_t above_left_shift =
        (above << 1) | ((col > 0) ? (above_left >> 63) : 0ULL);
    const std::uint64_t above_right_shift =
        (above >> 1) | ((col + 1 < words_per_row) ? (above_right << 63) : 0ULL);

    const std::uint64_t center_left_shift =
        (center << 1) | ((col > 0) ? (left >> 63) : 0ULL);
    const std::uint64_t center_right_shift =
        (center >> 1) | ((col + 1 < words_per_row) ? (right << 63) : 0ULL);

    const std::uint64_t below_left_shift =
        (below << 1) | ((col > 0) ? (below_left >> 63) : 0ULL);
    const std::uint64_t below_right_shift =
        (below >> 1) | ((col + 1 < words_per_row) ? (below_right << 63) : 0ULL);

    // Eight neighbor bitboards (one for each direction).
    const std::uint64_t n0 = above_left_shift;   // NW
    const std::uint64_t n1 = above;             // N
    const std::uint64_t n2 = above_right_shift; // NE
    const std::uint64_t n3 = center_left_shift; // W
    const std::uint64_t n4 = center_right_shift;// E
    const std::uint64_t n5 = below_left_shift;  // SW
    const std::uint64_t n6 = below;             // S
    const std::uint64_t n7 = below_right_shift; // SE

    // Bit-sliced neighbor count:
    //   ones  - bit 0 of count for each cell
    //   twos  - bit 1 of count for each cell
    //   fours - bit 2 of count for each cell
    //
    // Each ADD_NEIGHBOR(n) call adds the 1-bit board 'n' into this 3-bit counter
    // without carries between bit positions. The logic corresponds to:
    //
    //   temp = ones + n;
    //   ones = temp (mod 2);
    //   carry_to_twos = temp >> 1;  // where both ones and n were 1
    //
    //   temp2 = twos + carry_to_twos;
    //   twos = temp2 (mod 2);
    //   carry_to_fours = temp2 >> 1;
    //
    //   fours |= carry_to_fours;
    //
    // Implemented using bitwise operators across all 64 bits in parallel.
    std::uint64_t ones  = 0;
    std::uint64_t twos  = 0;
    std::uint64_t fours = 0;

#define ADD_NEIGHBOR(n)                               \
    do {                                              \
        std::uint64_t carry  = ones & (n);            \
        ones ^= (n);                                  \
        std::uint64_t carry2 = twos & carry;          \
        twos ^= carry;                                \
        fours |= carry2;                              \
    } while (0)

    ADD_NEIGHBOR(n0);
    ADD_NEIGHBOR(n1);
    ADD_NEIGHBOR(n2);
    ADD_NEIGHBOR(n3);
    ADD_NEIGHBOR(n4);
    ADD_NEIGHBOR(n5);
    ADD_NEIGHBOR(n6);
    ADD_NEIGHBOR(n7);

#undef ADD_NEIGHBOR

    // Determine where the neighbor count equals 2 or 3.
    //
    // Using the 3-bit representation (ones, twos, fours) = (b0, b1, b2):
    //   count == 2  <=> (b2, b1, b0) == (0, 1, 0)
    //   count == 3  <=> (b2, b1, b0) == (0, 1, 1)
    //
    // So:
    //   neighbor_eq_2 =  twos & ~ones & ~fours
    //   neighbor_eq_3 =  twos &  ones & ~fours
    const std::uint64_t neighbor_eq_2 = twos & ~ones & ~fours;
    const std::uint64_t neighbor_eq_3 = twos &  ones & ~fours;

    // Apply Game of Life rule:
    //   next = (neighbors == 3) OR (alive & neighbors == 2)
    const std::uint64_t survive_mask = neighbor_eq_2 & center;
    const std::uint64_t born_mask    = neighbor_eq_3;

    const std::uint64_t next_word = survive_mask | born_mask;

    output[idx] = next_word;
}

// Host function: launch one step of Game of Life on the GPU.
//
// Parameters:
//   input           - device pointer to bit-packed input grid
//   output          - device pointer to bit-packed output grid
//   grid_dimensions - width and height of the square grid
//
// Notes:
//   - input and output must be allocated with cudaMalloc.
//   - No synchronization is performed here; the caller is responsible for any
//     cudaDeviceSynchronize() or error checking after the kernel launch.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = grid_dimensions * words_per_row;

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
