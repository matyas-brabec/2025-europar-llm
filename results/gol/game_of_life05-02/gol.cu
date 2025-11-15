// Conway's Game of Life CUDA implementation with bit-packed grid.
// Each 64-bit word encodes 64 horizontal cells (1 bit per cell).
// Each CUDA thread processes one 64-bit word, updating 64 cells in parallel
// using bit-sliced arithmetic to compute neighbor counts.
//
// The algorithm works as follows:
// 1. For the current word and its 8 neighboring words (3x3 block of words),
//    construct three 64-bit masks per row: left, center, right.
//    These masks are aligned so that for any bit position i in the current word:
//      - left_row[i]   is the cell at column (i-1) in that row,
//      - center_row[i] is the cell at column i,
//      - right_row[i]  is the cell at column (i+1).
//    This is done via shifts within the word and by pulling edge bits from
//    neighbor words (word to the left / right), so bit 0 and bit 63 are
//    correctly handled across word boundaries.
//
// 2. For each of the three rows (above, current, below), compute the horizontal
//    sum of the three masks at each bit position (left + center + right).
//    This is a sum of three 1-bit values per cell, so the result is in {0,1,2,3}.
//    We represent this sum using two 64-bit "bitplanes":
//      row_lo: least significant bit of the sum
//      row_hi: most significant bit of the sum
//    For three 1-bit inputs L, C, R:
//
//      row_lo = L ^ C ^ R
//      row_hi = (L & C) | (L & R) | (C & R)
//              // row_hi == 1 where at least two of L,C,R are 1.
//
//    For the row above, this gives counts of {NW, N, NE} per cell.
//    For the current row, counts of {W, self, E}.
//    For the row below, counts of {SW, S, SE}.
//
// 3. Vertically sum the three two-bit horizontal sums (above, current, below)
//    to get the total number of live cells in the 3x3 neighborhood INCLUDING
//    the center cell (self). This total S is in {0,1,...,9} and is represented
//    with four bitplanes (vs0..vs3) as a 4-bit binary number.
//
//    Let (A1,A0) be the above row sum bits, (B1,B0) the current row,
//    and (C1,C0) the below row. Then:
//
//      // First compute T = A + B, where A,B in [0,3], so T in [0,6] (3 bits).
//      T0 = A0 ^ B0                       // bit 0
//      carry0 = A0 & B0                   // carry from bit 0 to bit 1
//
//      T1 = A1 ^ B1 ^ carry0              // bit 1
//      T2 = (A1 & B1) | (A1 & carry0) | (B1 & carry0)  // bit 2
//
//      // Now compute S = T + C, where T in [0,6], C in [0,3], S in [0,9] (4 bits).
//      S0 = T0 ^ C0
//      carry1 = T0 & C0
//
//      S1 = T1 ^ C1 ^ carry1
//      carry2 = (T1 & C1) | (T1 & carry1) | (C1 & carry1)
//
//      S2 = T2 ^ carry2
//      S3 = T2 & carry2
//
//    The four bitplanes S3..S0 form the binary total S for each cell.
//
//    Note: For any cell, S = (# of live neighbors) + (self ? 1 : 0).
//
// 4. Game of Life rule in terms of S (total, including self):
//      neighbors = S - self
//      next is alive if:
//
//        (neighbors == 3)                              // birth
//        OR (self == 1 && neighbors == 2)             // survival
//
//    Translating to S and self:
//
//      neighbors == 3:
//        if self == 0 -> S == 3
//        if self == 1 -> S == 4
//
//      self == 1 && neighbors == 2:
//        S == 3
//
//    So:
//      - if S == 3, the cell is alive next step regardless of self.
//      - if S == 4, the cell is alive next step only if self == 1.
//      - all other S -> cell is dead.
//
//    Thus we only need to detect S == 3 and S == 4:
//
//      S == 3 -> bits (S3,S2,S1,S0) = (0,0,1,1)
//      S == 4 -> bits (S3,S2,S1,S0) = (0,1,0,0)
//
//    For bitplanes vs3..vs0:
//
//      eq3 = vs0 & vs1 & ~vs2 & ~vs3
//      eq4 = ~vs0 & ~vs1 &  vs2 & ~vs3
//
//      self = current row center bitboard
//
//      next = eq3 | (eq4 & self)
//
// 5. Boundary handling:
//    - All cells outside the grid are considered dead.
//    - For words at the left/right edges, we treat missing neighbor words as 0.
//      Bit 0 and bit 63 within each 64-bit word correctly pull neighbor bits
//      from the left/right words where they exist, otherwise 0.
//    - For the top/bottom rows, words in rows above/below are treated as 0.
//
// 6. Each CUDA thread processes one 64-bit word from the input and writes the
//    corresponding 64-bit word in the output. No atomics are needed.

#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel: one thread processes one 64-bit word (64 cells).
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int words_per_row,
                                    int grid_dim)
{
    const std::size_t global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dim);

    if (global_idx >= total_words) {
        return;
    }

    const int row = static_cast<int>(global_idx / words_per_row);
    const int col = static_cast<int>(global_idx - static_cast<std::size_t>(row) * words_per_row);

    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim);

    const std::uint64_t ZERO = 0ull;

    // Current row words: left, center, right
    const std::uint64_t center = input[global_idx];

    const std::uint64_t left  = has_left  ? input[global_idx - 1] : ZERO;
    const std::uint64_t right = has_right ? input[global_idx + 1] : ZERO;

    // Row above words: left, center, right
    std::uint64_t above_left  = ZERO;
    std::uint64_t above       = ZERO;
    std::uint64_t above_right = ZERO;

    if (has_up) {
        const std::size_t idx_above = global_idx - static_cast<std::size_t>(words_per_row);
        above = input[idx_above];
        if (has_left) {
            above_left = input[idx_above - 1];
        }
        if (has_right) {
            above_right = input[idx_above + 1];
        }
    }

    // Row below words: left, center, right
    std::uint64_t below_left  = ZERO;
    std::uint64_t below       = ZERO;
    std::uint64_t below_right = ZERO;

    if (has_down) {
        const std::size_t idx_below = global_idx + static_cast<std::size_t>(words_per_row);
        below = input[idx_below];
        if (has_left) {
            below_left = input[idx_below - 1];
        }
        if (has_right) {
            below_right = input[idx_below + 1];
        }
    }

    // Construct left/center/right masks for each row, aligned with the current word bits.
    //
    // For any row bitboard X and neighbor word to the left XL and right XR:
    //   left_row  = (X << 1) | (XL >> 63)  // cells at column-1
    //   center_row= X                     // cells at column
    //   right_row = (X >> 1) | (XR << 63) // cells at column+1
    //
    // This way, for any bit position i in the current word:
    //   left_row[i]   corresponds to cell (row, col+i-1),
    //   center_row[i] corresponds to cell (row, col+i),
    //   right_row[i]  corresponds to cell (row, col+i+1).

    // Above row (r-1)
    const std::uint64_t aC = above;
    const std::uint64_t aL = (aC << 1) | (above_left >> 63);
    const std::uint64_t aR = (aC >> 1) | (above_right << 63);

    // Current row (r)
    const std::uint64_t cC = center;
    const std::uint64_t cL = (cC << 1) | (left >> 63);
    const std::uint64_t cR = (cC >> 1) | (right << 63);

    // Below row (r+1)
    const std::uint64_t bC = below;
    const std::uint64_t bL = (bC << 1) | (below_left >> 63);
    const std::uint64_t bR = (bC >> 1) | (below_right << 63);

    // Horizontal sums for each row: sum of {left, center, right} at each bit position.
    // Result is in [0,3], represented as 2 bitplanes: lo (bit0), hi (bit1).

    // Above row horizontal sum
    const std::uint64_t hA_lo = aL ^ aC ^ aR;
    const std::uint64_t hA_hi = (aL & aC) | (aL & aR) | (aC & aR);

    // Current row horizontal sum (includes self)
    const std::uint64_t hC_lo = cL ^ cC ^ cR;
    const std::uint64_t hC_hi = (cL & cC) | (cL & cR) | (cC & cR);

    // Below row horizontal sum
    const std::uint64_t hB_lo = bL ^ bC ^ bR;
    const std::uint64_t hB_hi = (bL & bC) | (bL & bR) | (bC & bR);

    // Vertical sum: S = hAbove + hCurrent + hBelow.
    // Each hX is a 2-bit number in [0,3]. S is in [0,9] and is represented by 4 bitplanes (vs0..vs3).

    const std::uint64_t A0 = hA_lo;
    const std::uint64_t A1 = hA_hi;
    const std::uint64_t B0 = hC_lo;
    const std::uint64_t B1 = hC_hi;
    const std::uint64_t C0 = hB_lo;
    const std::uint64_t C1 = hB_hi;

    // T = A + B  (A,B in [0,3], T in [0,6])
    const std::uint64_t T0    = A0 ^ B0;
    const std::uint64_t carry0 = A0 & B0;

    const std::uint64_t T1 = A1 ^ B1 ^ carry0;
    const std::uint64_t T2 =
        (A1 & B1) | (A1 & carry0) | (B1 & carry0);

    // S = T + C  (T in [0,6], C in [0,3], S in [0,9])
    const std::uint64_t S0    = T0 ^ C0;
    const std::uint64_t carry1 = T0 & C0;

    const std::uint64_t S1 = T1 ^ C1 ^ carry1;
    const std::uint64_t carry2 =
        (T1 & C1) | (T1 & carry1) | (C1 & carry1);

    const std::uint64_t S2 = T2 ^ carry2;
    const std::uint64_t S3 = T2 & carry2;

    // S is now represented at each bit position by (S3,S2,S1,S0) as a 4-bit integer in [0,9],
    // where S = (#neighbors) + (self ? 1 : 0).

    // Detect S == 3 (0011) and S == 4 (0100) per bit.

    const std::uint64_t eq3 =  S0 &  S1 & ~S2 & ~S3;  // S == 3
    const std::uint64_t eq4 = ~S0 & ~S1 &  S2 & ~S3;  // S == 4

    // Current cell bitboard ("self" for each bit)
    const std::uint64_t self = cC;

    // Game of Life rule in terms of S and self:
    //   next_alive = (S == 3) || (self && S == 4)

    const std::uint64_t next = eq3 | (eq4 & self);

    output[global_idx] = next;
}

// Host-side entry point: run one Game of Life step on the GPU.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is the width and height in cells.
    // words_per_row is number of 64-bit words per row.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Choose a reasonable block size; 256 threads per block works well on modern GPUs.
    const int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (total_words + static_cast<std::size_t>(threads_per_block) - 1) /
        static_cast<std::size_t>(threads_per_block));

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output,
                                                       words_per_row,
                                                       grid_dimensions);
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
