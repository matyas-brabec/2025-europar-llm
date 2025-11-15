// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
//
// Each cell is encoded as a single bit inside a 64-bit word (std::uint64_t).
// - Bit value 1: live cell
// - Bit value 0: dead cell
//
// Each CUDA thread processes exactly one 64-bit word (= 64 cells).
//
// The main goals are:
//   * Avoid per-cell loops (which are expensive) by using bit-parallel arithmetic.
//   * Use only global memory (no shared/texture memory), as per the problem statement.
//   * Correctly handle non-periodic boundaries (cells outside the grid are dead).
//
// High-level algorithm (bit-sliced neighbor counting):
//
// 1. For a word at (row, col_word), we load up to 9 words:
//      [north-left]  [north]  [north-right]
//      [west]        [center] [east]
//      [south-left]  [south]  [south-right]
//
//    For boundary words, some of these are zero (outside grid).
//
// 2. From these 9 words we derive 8 bitboards, each representing one neighbor direction
//    for all 64 cells in the word simultaneously:
//       N, S, E, W, NE, NW, SE, SW
//
//    Horizontal and diagonal neighbors require special handling of bit 0 and bit 63:
//       - For bit 0, we also need bits from the words to the left (same row, row-1, row+1).
//       - For bit 63, we also need bits from the words to the right.
//    This is done by combining shifts with cross-word bit injection.
//    Example for east neighbors in the same row:
//       east = (center >> 1) | ((right_word & 1ULL) << 63);
//
// 3. To compute neighbor counts without per-cell loops, we maintain four 64-bit bitplanes
//    (c0, c1, c2, c3) that form a 4-bit counter per cell (0..8 possible neighbors).
//    Each bit position i across these bitplanes encodes the neighbor count for cell i.
//    We add each of the 8 neighbor bitboards to this 4-bit counter using bit-sliced
//    ripple-carry addition implemented with AND/XOR, without any cross-cell carries.
//
//    Adding a 1-bit bitboard "b" to the 4-bit counter (c3 c2 c1 c0) is:
//
//       carry0 = c0 & b;   c0 ^= b;
//       carry1 = c1 & carry0;   c1 ^= carry0;
//       carry2 = c2 & carry1;   c2 ^= carry1;
//       c3 ^= carry2;
//
//    All operations are bitwise, so each bit position is independent.
//
// 4. After processing all 8 neighbor directions, the 4-bit count at each bit position
//    is in (c3 c2 c1 c0). We need only to know where the count is exactly 2 or 3:
//
//       count == 2:  c3=0, c2=0, c1=1, c0=0
//       count == 3:  c3=0, c2=0, c1=1, c0=1
//
//    So we compute two bitboards eq2 and eq3 with boolean logic on the bitplanes.
//
// 5. The Game of Life transition rule per cell is:
//
//       next = (count == 3) OR (alive AND count == 2)
//
//    So for the 64 cells in the word:
//
//       next_word = eq3 | (center & eq2)
//
// 6. The kernel writes next_word to the output array.
//
// The function `run_game_of_life` configures the kernel launch so that each thread
// processes a single 64-bit word, and executes one step of the simulation.

#include <cstdint>
#include <cuda_runtime.h>

// Bit-sliced increment of a 4-bit counter (c3 c2 c1 c0) by a 1-bit bitboard "neighbor".
// Each bit position represents an independent 4-bit integer in [0, 8].
// This function performs a ripple-carry add for each bit position in parallel.
//
// The representation is:
//   count = (c3 << 3) | (c2 << 2) | (c1 << 1) | c0
//
// All arguments are 64-bit bitboards; bitwise operations act independently per cell.
__device__ __forceinline__
void add_bitboard(std::uint64_t neighbor,
                  std::uint64_t &c0,
                  std::uint64_t &c1,
                  std::uint64_t &c2,
                  std::uint64_t &c3)
{
    std::uint64_t carry0 = c0 & neighbor;
    c0 ^= neighbor;

    std::uint64_t carry1 = c1 & carry0;
    c1 ^= carry0;

    std::uint64_t carry2 = c2 & carry1;
    c2 ^= carry1;

    c3 ^= carry2;
}

// Kernel: one thread processes one 64-bit word (64 cells).
// "words_per_row" is grid_dimensions / 64.
// "rows" is grid_dimensions (square grid).
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int words_per_row,
                         int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = words_per_row * rows;
    if (idx >= total_words) {
        return;
    }

    // Compute 2D position (row, column-word)
    int row = idx / words_per_row;
    int col = idx - row * words_per_row;

    const bool top    = (row == 0);
    const bool bottom = (row == rows - 1);
    const bool left   = (col == 0);
    const bool right  = (col == words_per_row - 1);

    // Load center word
    std::uint64_t center = input[idx];

    // Load vertical neighbors (same column)
    std::uint64_t north = top    ? 0ULL : input[idx - words_per_row];
    std::uint64_t south = bottom ? 0ULL : input[idx + words_per_row];

    // Load horizontal neighbors (same row)
    std::uint64_t west_word  = left  ? 0ULL : input[idx - 1];
    std::uint64_t east_word  = right ? 0ULL : input[idx + 1];

    // Load diagonal neighbor words for rows above and below
    std::uint64_t north_west = (top || left)    ? 0ULL : input[idx - words_per_row - 1];
    std::uint64_t north_east = (top || right)   ? 0ULL : input[idx - words_per_row + 1];
    std::uint64_t south_west = (bottom || left) ? 0ULL : input[idx + words_per_row - 1];
    std::uint64_t south_east = (bottom || right)? 0ULL : input[idx + words_per_row + 1];

    // Build neighbor bitboards for all 64 cells in this word.
    //
    // Vertical neighbors:
    //   N: north, S: south
    std::uint64_t N = north;
    std::uint64_t S = south;

    // Horizontal neighbors in the same row:
    //
    // For each bit position i:
    //   E[i] = cell at (i+1) in the same row  (or from east_word if i == 63)
    //   W[i] = cell at (i-1) in the same row  (or from west_word if i == 0)
    //
    // We treat bits outside the grid as 0 by setting the neighbor words to 0 at edges.
    std::uint64_t E = (center >> 1) | ((east_word  & 1ULL) << 63);
    std::uint64_t W = (center << 1) | (west_word >> 63);

    // Diagonal neighbors:
    //
    // North-East:
    //   NE[i] = north[i+1] for i=0..62, and north_east[0] for i=63.
    // North-West:
    //   NW[i] = north[i-1] for i=1..63, and north_west[63] for i=0.
    // South-East and South-West similarly for the south row.
    std::uint64_t NE = (north >> 1) | ((north_east & 1ULL) << 63);
    std::uint64_t NW = (north << 1) | (north_west >> 63);
    std::uint64_t SE = (south >> 1) | ((south_east & 1ULL) << 63);
    std::uint64_t SW = (south << 1) | (south_west >> 63);

    // 4-bit per-cell neighbor counters (initialized to zero for all 64 cells).
    std::uint64_t c0 = 0ULL;
    std::uint64_t c1 = 0ULL;
    std::uint64_t c2 = 0ULL;
    std::uint64_t c3 = 0ULL;

    // Accumulate counts from all 8 neighbor directions.
    add_bitboard(N,  c0, c1, c2, c3);
    add_bitboard(S,  c0, c1, c2, c3);
    add_bitboard(E,  c0, c1, c2, c3);
    add_bitboard(W,  c0, c1, c2, c3);
    add_bitboard(NE, c0, c1, c2, c3);
    add_bitboard(NW, c0, c1, c2, c3);
    add_bitboard(SE, c0, c1, c2, c3);
    add_bitboard(SW, c0, c1, c2, c3);

    // Determine where neighbor count == 2 and == 3 using the bitplanes.
    //
    // count (binary) = c3 c2 c1 c0
    //   count == 2: 0 0 1 0
    //   count == 3: 0 0 1 1
    //
    // We compute these as bitboards eq2 and eq3.
    std::uint64_t not_c0 = ~c0;
    std::uint64_t not_c1 = ~c1;
    std::uint64_t not_c2 = ~c2;
    std::uint64_t not_c3 = ~c3;

    (void)not_c1; // not_c1 is not needed for eq2/eq3 but computed for clarity; avoid unused warning

    std::uint64_t eq2 = not_c3 & not_c2 & c1      & not_c0; // 0010
    std::uint64_t eq3 = not_c3 & not_c2 & c1      & c0;     // 0011

    // Apply Game of Life rules:
    //   next = (count == 3) OR (alive AND count == 2)
    std::uint64_t survive = center & eq2;
    std::uint64_t birth   = eq3;
    std::uint64_t next    = survive | birth;

    // Write result
    output[idx] = next;
}

// Host function: executes one step of Conway's Game of Life on the GPU.
// - input / output: device pointers (allocated with cudaMalloc).
// - grid_dimensions: width and height of the square grid (power of 2, > 512).
//
// The grid is bit-packed: each std::uint64_t encodes 64 cells in a row.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int rows = grid_dimensions;

    const int total_words = words_per_row * rows;

    // Choose a reasonable block size for modern GPUs.
    const int blockSize = 256;
    const int gridSize  = (total_words + blockSize - 1) / blockSize;

    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, words_per_row, rows);
    // Host-device synchronization (if desired) is handled by the caller.
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
