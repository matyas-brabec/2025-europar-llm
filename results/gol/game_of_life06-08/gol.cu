// Conway's Game of Life, bit-parallel CUDA implementation.
//
// Each cell is a single bit. The grid is square (N x N), where N is a power of 2
// and N >= 512. The grid is bit-packed into 64-bit words, so each std::uint64_t
// represents 64 consecutive cells in a row.
//
// Layout:
//   - Rows are stored consecutively in row-major order.
//   - For grid_dimensions = N, the number of 64-bit words per row is W = N / 64.
//   - Word index for (row, word_col) is: idx = row * W + word_col.
//
// Each CUDA thread processes one 64-bit word (i.e., 64 cells).
//
// Neighbor counting strategy:
//
// For every word (representing 64 cells in a row) we need to count, for each bit,
// the number of alive neighbors (0..8) around that cell. We use bitwise full-adder
// logic to process 64 cells in parallel.
//
// 1. Horizontal pass (within each of the three relevant rows: top, middle, bottom)
//    ---------------------------------------------------------------------------
//    For a given row, we build three 64-bit bitfields for each word:
//      L = left neighbors  (cells at column-1)
//      C = center cells    (cells at column)
//      R = right neighbors (cells at column+1)
//
//    The values of L and R are built using bit shifts plus cross-word carry for
//    bit 0 and bit 63 (handled via words to the left/right):
//
//      L = (C << 1) | (left_word >> 63)
//      R = (C >> 1) | ((right_word & 1) << 63)
//
//    For each bit position, L, C, R are three one-bit inputs representing the
//    presence of a live cell in the left, center, and right positions of that
//    row's 3-cell neighborhood.
//
//    We then use a full adder to compute the count of live cells among these three:
//
//      h0 = L ^ C ^ R                     (LSB, 1's bit of the count, range 0..1)
//      h1 = majority(L, C, R)             (MSB, 2's bit of the count, range 0..1)
//
//    Where majority(a,b,c) = (a & b) | (a & c) | (b & c).
//
//    Thus, for each row:
//      0 neighbors -> h1h0 = 00 -> 0
//      1 neighbor  -> h1h0 = 01 -> 1
//      2 neighbors -> h1h0 = 10 -> 2
//      3 neighbors -> h1h0 = 11 -> 3
//
//    We compute (t0,t1) for the top row, (m0,m1) for the middle row, and (b0,b1)
//    for the bottom row, where each *0 is the LSB (1's bit), and each *1 is the
//    MSB (2's bit).
//
// 2. Vertical pass (combine the three rows)
//    --------------------------------------
//    For each bit position, the total number of live cells in the 3x3 neighborhood
//    INCLUDING the center cell is:
//
//      n' = t + m + b
//
//    with t, m, b in 0..3:
//      t = t0 + 2*t1
//      m = m0 + 2*m1
//      b = b0 + 2*b1
//
//    So:
//      n' = (t0 + m0 + b0) + 2 * (t1 + m1 + b1)
//
//    Define:
//      A = t0 + m0 + b0   (0..3)  -> encoded in bits A1:A0
//      B = t1 + m1 + b1   (0..3)  -> encoded in bits B1:B0
//
//    Because each of {t0,m0,b0} and {t1,m1,b1} are single-bit values, we can
//    compute A and B with another full-adder style operation:
//
//      A0 = t0 ^ m0 ^ b0                    (LSB of A)
//      A1 = majority(t0, m0, b0)            (MSB of A)
//
//      B0 = t1 ^ m1 ^ b1                    (LSB of B)
//      B1 = majority(t1, m1, b1)            (MSB of B)
//
//    Then for each bit lane:
//      A = 2*A1 + A0 in {0,1,2,3}
//      B = 2*B1 + B0 in {0,1,2,3}
//      n' = A + 2*B in {0,...,9}
//
//    Note n' counts the 8 neighbors PLUS the center cell itself. This is because:
//      - t counts (north-west, north, north-east)
//      - b counts (south-west, south, south-east)
//      - m counts (west, center, east)
//
//      => t + m + b = (#neighbors) + center.
//
// 3. Game of Life rules in terms of n'
//    ---------------------------------
//    Let:
//      c = current cell bit (0 or 1)
//      n = number of live neighbors (0..8)
//      n' = n + c                       (since defined as above)
//
//    Original rules:
//      If c == 1:
//        - survives if n == 2 or n == 3
//      If c == 0:
//        - becomes alive if n == 3
//
//    In terms of n':
//      If c == 1 => n' = n + 1:
//        - survives if n' == 3 or n' == 4
//      If c == 0 => n' = n:
//        - becomes alive if n' == 3
//
//    Therefore, the new cell value is:
//      new = (n' == 3) || (c && (n' == 4))
//
//    We only need equality tests for n' == 3 and n' == 4.
//
// 4. Detect n' == 3 and n' == 4 from A,B
//    ------------------------------------
//    n' = A + 2*B, with A, B in 0..3 as above.
//
//    Enumerating all combinations yields:
//
//      n' == 3  iff (B=0 & A=3) or (B=1 & A=1)
//      n' == 4  iff (B=1 & A=2) or (B=2 & A=0)
//
//    with binary encodings:
//      A = A1 A0, B = B1 B0 (two bits each).
//
//    After simplification:
//
//      eq3 (n' == 3)  = ~B1 & A0 & (B0 ^ A1)
//
//      eq4 (n' == 4)  = (~B1 &  B0 &  A1 & ~A0)      // (B=1 & A=2)
//                     | ( B1 & ~B0 & ~A1 & ~A0)      // (B=2 & A=0)
//
//    Then the next state bit is:
//
//      next = eq3 | (c & eq4)
//
// 5. Boundary handling
//    ------------------
//    - For rows above the top boundary and below the bottom boundary, all values
//      are treated as zero (dead).
//    - For columns left of the left boundary and right of the right boundary,
//      neighbors are treated as zero (dead).
//    - This is implemented by conditional loads of neighbor words:
//        * If row == 0   -> top row words are 0.
//        * If row == N-1 -> bottom row words are 0.
//        * If col == 0   -> left neighbor words are 0.
//        * If col == W-1 -> right neighbor words are 0.
//
// 6. Mapping to CUDA
//    ----------------
//    - Each thread processes exactly one 64-bit word.
//    - We compute (row, col) from the linear index, load the 3x3 window of words,
//      compute neighbor counts as described, and write the resulting 64-bit word.
//    - No shared or texture memory is used; only global memory and registers.
//    - Pointers are marked __restrict__ to help the compiler optimize.
//
//    This approach uses only simple bitwise operations and a few branches for
//    boundary handling, while processing 64 cells per thread in parallel.
//
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>

// Helper: majority of three 64-bit bitfields, bitwise.
// For each bit position, majority(a,b,c) == 1 iff at least two of {a,b,c} are 1.
static __device__ __forceinline__ std::uint64_t majority3(std::uint64_t a,
                                                          std::uint64_t b,
                                                          std::uint64_t c)
{
    return (a & b) | (a & c) | (b & c);
}

// CUDA kernel: one Game of Life step on a bit-packed square grid.
// - input, output: device pointers to grids, bit-packed as described.
// - grid_dim:     width/height of the square grid (N).
// - words_per_row: N / 64.
// - total_words:  total number of 64-bit words: words_per_row * N.
static __global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                           std::uint64_t* __restrict__ output,
                                           int grid_dim,
                                           int words_per_row,
                                           int total_words)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_words)
        return;

    // Compute row and column in "word space".
    int row = tid / words_per_row;
    int col = tid - row * words_per_row;

    // ------------------------------------------------------------------------
    // Load center row (middle) words: center, left, right.
    // ------------------------------------------------------------------------
    const std::uint64_t midC = input[tid];

    std::uint64_t midLw = 0;
    std::uint64_t midRw = 0;

    if (col > 0) {
        midLw = input[tid - 1];
    }
    if (col + 1 < words_per_row) {
        midRw = input[tid + 1];
    }

    // ------------------------------------------------------------------------
    // Load top row words (if any): center, left, right.
    // ------------------------------------------------------------------------
    std::uint64_t topC = 0;
    std::uint64_t topLw = 0;
    std::uint64_t topRw = 0;

    if (row > 0) {
        int top_base = tid - words_per_row;  // same column, row-1
        topC = input[top_base];

        if (col > 0) {
            topLw = input[top_base - 1];
        }
        if (col + 1 < words_per_row) {
            topRw = input[top_base + 1];
        }
    }

    // ------------------------------------------------------------------------
    // Load bottom row words (if any): center, left, right.
    // ------------------------------------------------------------------------
    std::uint64_t botC = 0;
    std::uint64_t botLw = 0;
    std::uint64_t botRw = 0;

    if (row + 1 < grid_dim) {
        int bot_base = tid + words_per_row;  // same column, row+1
        botC = input[bot_base];

        if (col > 0) {
            botLw = input[bot_base - 1];
        }
        if (col + 1 < words_per_row) {
            botRw = input[bot_base + 1];
        }
    }

    // ------------------------------------------------------------------------
    // Horizontal pass: for each of the three rows, compute h0 (LSB) and h1 (MSB)
    // for the 3-cell neighborhood (left, center, right) using full adder logic.
    // ------------------------------------------------------------------------

    // Middle row horizontal neighborhood.
    const std::uint64_t midL = (midC << 1) | (midLw >> 63);
    const std::uint64_t midR = (midC >> 1) | ((midRw & 1ULL) << 63);
    const std::uint64_t m0   = midL ^ midC ^ midR;
    const std::uint64_t m1   = majority3(midL, midC, midR);

    // Top row horizontal neighborhood (if any).
    std::uint64_t t0 = 0;
    std::uint64_t t1 = 0;
    if (row > 0) {
        const std::uint64_t topL = (topC << 1) | (topLw >> 63);
        const std::uint64_t topR = (topC >> 1) | ((topRw & 1ULL) << 63);
        t0 = topL ^ topC ^ topR;
        t1 = majority3(topL, topC, topR);
    }

    // Bottom row horizontal neighborhood (if any).
    std::uint64_t b0 = 0;
    std::uint64_t b1 = 0;
    if (row + 1 < grid_dim) {
        const std::uint64_t botL = (botC << 1) | (botLw >> 63);
        const std::uint64_t botR = (botC >> 1) | ((botRw & 1ULL) << 63);
        b0 = botL ^ botC ^ botR;
        b1 = majority3(botL, botC, botR);
    }

    // ------------------------------------------------------------------------
    // Vertical pass: sum the per-row 3-cell counts across the three rows.
    //
    // For each bit lane:
    //   t = t0 + 2*t1, m = m0 + 2*m1, b = b0 + 2*b1
    //   n' = t + m + b = A + 2*B, where:
    //
    //   A = t0 + m0 + b0 in {0..3}
    //   B = t1 + m1 + b1 in {0..3}
    //
    // A and B are computed again with full-adder style logic.
    // ------------------------------------------------------------------------
    const std::uint64_t A0 = t0 ^ m0 ^ b0;
    const std::uint64_t A1 = majority3(t0, m0, b0);

    const std::uint64_t B0 = t1 ^ m1 ^ b1;
    const std::uint64_t B1 = majority3(t1, m1, b1);

    // ------------------------------------------------------------------------
    // Detect n' == 3 and n' == 4 for each bit lane.
    //
    // eq3: n' == 3  -> ~B1 & A0 & (B0 ^ A1)
    // eq4: n' == 4  ->
    //         (~B1 &  B0 &  A1 & ~A0)   // B=1, A=2
    //       | ( B1 & ~B0 & ~A1 & ~A0)   // B=2, A=0
    // ------------------------------------------------------------------------
    const std::uint64_t notB1 = ~B1;
    const std::uint64_t notB0 = ~B0;
    const std::uint64_t notA0 = ~A0;
    const std::uint64_t notA1 = ~A1;

    const std::uint64_t eq3 = notB1 & A0 & (B0 ^ A1);

    const std::uint64_t eq4_part1 = notB1 & B0 & A1 & notA0;
    const std::uint64_t eq4_part2 = B1 & notB0 & notA1 & notA0;
    const std::uint64_t eq4 = eq4_part1 | eq4_part2;

    // ------------------------------------------------------------------------
    // Apply Game of Life rule:
    //
    //   c   = current cell bit (midC)
    //   n'  = neighbors + c
    //
    //   next = (n' == 3) || (c && n' == 4)
    //
    // Implemented as:
    //   next = eq3 | (c & eq4)
    // ------------------------------------------------------------------------
    const std::uint64_t c = midC;
    const std::uint64_t next = eq3 | (c & eq4);

    output[tid] = next;
}

// Host function: run one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   input           - device pointer to bit-packed input grid.
//   output          - device pointer to bit-packed output grid.
//   grid_dimensions - width and height of the square grid (N), power of 2.
//
// Assumptions:
//   - input and output were allocated with cudaMalloc and are large enough.
//   - No host-side synchronization is done here; the caller handles it.
//   - grid_dimensions is divisible by 64 (as per problem statement).
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;            // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    // Configure CUDA launch.
    // Use a reasonably large block size for good occupancy.
    const int block_size = 256;
    const int grid_size  = (total_words + block_size - 1) / block_size;

    game_of_life_kernel<<<grid_size, block_size>>>(input,
                                                   output,
                                                   grid_dimensions,
                                                   words_per_row,
                                                   total_words);
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
