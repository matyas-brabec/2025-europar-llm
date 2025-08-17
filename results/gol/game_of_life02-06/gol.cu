#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Encoding:
  - Each 64-bit word encodes 64 consecutive cells of a row, LSB = leftmost bit in the word.
  - Cells outside the grid are considered dead (0).

  Algorithm overview (bit-sliced, branch-free per bit):
  - Each thread processes one 64-bit word at (row, colWord).
  - For three vertically adjacent rows (top, middle, bottom), we construct horizontal neighbor masks:
      leftShift(rowWord)   -> neighbors to the left
      rowWord              -> vertical neighbors (top/bottom only)
      rightShift(rowWord)  -> neighbors to the right
    using adjacent words to handle cross-word bit propagation at 64-bit boundaries.
  - For the top and bottom rows, we compute the horizontal sums (left + center + right) as a 2-bit number per bit position:
      sum0 = XOR of (left, center, right)  -> LSB per bit-lane
      sum1 = majority(left, center, right) -> MSB per bit-lane (1 iff at least two of them are 1)
  - For the middle row, we exclude the center cell (as it's not a neighbor), so the horizontal sum is just (left + right):
      sum0 = left ^ right
      sum1 = left & right
  - We then vertically add the three 2-bit numbers (top, mid, bottom) per bit-lane to obtain the 4-bit neighbor count (0..8):
      Let T=(t1:t0), M=(m1:m0), B=(b1:b0).
      s0  = t0 ^ m0 ^ b0
      c01 = majority(t0, m0, b0)                        // carry into bit1 from LSB addition
      sum1 = t1 ^ m1 ^ b1
      c1_from_inputs = majority(t1, m1, b1)             // carry contribution from MSB inputs
      s1  = sum1 ^ c01
      c1_from_c01 = sum1 & c01                          // carry produced when adding c01 to sum1
      s2  = c1_from_inputs ^ c1_from_c01
      s3  = c1_from_inputs & c1_from_c01
    Thus, the neighbor count per bit-lane is (s3 s2 s1 s0).
  - Next-state rule:
      next = (neighbors == 3) | (alive & (neighbors == 2))
    With bit-slices:
      eq2 = ~s3 & ~s2 &  s1 & ~s0
      eq3 = ~s3 & ~s2 &  s1 &  s0
      next = eq3 | (eq2 & middleRowCenterWord)

  Notes:
  - No shared or texture memory is used, as requested.
  - All loads are coalesced when mapping threads along the word-columns within the same row.
  - Branching is only used for boundary handling (first/last row/column), impacting a tiny fraction of threads.

*/

static __forceinline__ __device__ std::uint64_t majority3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    // Bitwise majority of three operands: 1 if at least two inputs are 1 in that bit-lane.
    return (a & b) | (a & c) | (b & c);
}

static __forceinline__ __device__ std::uint64_t shift_left_with_carry(std::uint64_t center, std::uint64_t left_word) {
    // Shift cells left by 1 within the row, sourcing the incoming LSB from the MSB of the left neighbor word.
    // If at the leftmost word, left_word must be 0, yielding zero carry-in.
    return (center << 1) | (left_word >> 63);
}

static __forceinline__ __device__ std::uint64_t shift_right_with_carry(std::uint64_t center, std::uint64_t right_word) {
    // Shift cells right by 1 within the row, sourcing the incoming MSB from the LSB of the right neighbor word.
    // If at the rightmost word, right_word must be 0, yielding zero carry-in.
    return (center >> 1) | (right_word << 63);
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int grid_dim,
                                         int words_per_row)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // word column index
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (col >= words_per_row || row >= grid_dim) return;

    const int row_base = row * words_per_row;

    // Determine boundary availability
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim);

    // Load center row words
    const std::uint64_t mC = input[row_base + col];
    const std::uint64_t mL = has_left  ? input[row_base + col - 1] : 0ull;
    const std::uint64_t mR = has_right ? input[row_base + col + 1] : 0ull;

    // Load upper row words (or zeros at top boundary)
    const int up_base = row_base - words_per_row;
    const std::uint64_t uC = has_up ? input[up_base + col] : 0ull;
    const std::uint64_t uL = (has_up && has_left)  ? input[up_base + col - 1] : 0ull;
    const std::uint64_t uR = (has_up && has_right) ? input[up_base + col + 1] : 0ull;

    // Load lower row words (or zeros at bottom boundary)
    const int dn_base = row_base + words_per_row;
    const std::uint64_t dC = has_down ? input[dn_base + col] : 0ull;
    const std::uint64_t dL = (has_down && has_left)  ? input[dn_base + col - 1] : 0ull;
    const std::uint64_t dR = (has_down && has_right) ? input[dn_base + col + 1] : 0ull;

    // Construct horizontal neighbor masks for each of the three rows.
    // Top row: include left, center, right (all are neighbors)
    const std::uint64_t topL = shift_left_with_carry(uC, uL);
    const std::uint64_t topC = uC;
    const std::uint64_t topR = shift_right_with_carry(uC, uR);

    // Middle row: exclude center (only left and right)
    const std::uint64_t midL = shift_left_with_carry(mC, mL);
    const std::uint64_t midR = shift_right_with_carry(mC, mR);

    // Bottom row: include left, center, right (all are neighbors)
    const std::uint64_t botL = shift_left_with_carry(dC, dL);
    const std::uint64_t botC = dC;
    const std::uint64_t botR = shift_right_with_carry(dC, dR);

    // Horizontal sums per row (bit-sliced):
    // For top row: sum of three 1-bit inputs -> 2-bit number (t1:t0)
    const std::uint64_t t0 = topL ^ topC ^ topR;
    const std::uint64_t t1 = majority3(topL, topC, topR);

    // For middle row: sum of two 1-bit inputs -> 2-bit number (m1:m0)
    const std::uint64_t m0 = midL ^ midR;
    const std::uint64_t m1 = midL & midR;

    // For bottom row: sum of three 1-bit inputs -> 2-bit number (b1:b0)
    const std::uint64_t b0 = botL ^ botC ^ botR;
    const std::uint64_t b1 = majority3(botL, botC, botR);

    // Vertical addition of T=(t1:t0), M=(m1:m0), B=(b1:b0)
    // Compute the 4-bit neighbor count per bit-lane: (s3 s2 s1 s0)

    // Bit 0:
    const std::uint64_t s0 = t0 ^ m0 ^ b0;

    // Carry into bit 1 from bit 0 addition (majority of t0,m0,b0)
    const std::uint64_t c01 = majority3(t0, m0, b0);

    // Sum of bit-1 components before adding c01
    const std::uint64_t sum1 = t1 ^ m1 ^ b1;
    const std::uint64_t c1_from_inputs = majority3(t1, m1, b1); // carry towards bit 2 from t1+m1+b1

    // Bit 1:
    const std::uint64_t s1 = sum1 ^ c01;

    // Additional carry to bit 2 when adding c01 to sum1
    const std::uint64_t c1_from_c01 = sum1 & c01;

    // Bit 2 and Bit 3:
    const std::uint64_t s2 = c1_from_inputs ^ c1_from_c01;
    const std::uint64_t s3 = c1_from_inputs & c1_from_c01;

    // Compute (neighbors == 2) and (neighbors == 3) masks:
    const std::uint64_t nz = ~(s3 | s2);          // ~s3 & ~s2
    const std::uint64_t base = nz & s1;           // ~s3 & ~s2 & s1
    const std::uint64_t eq3 = base & s0;          // == 3
    const std::uint64_t eq2 = base & (~s0);       // == 2

    // Apply Game of Life rule:
    // next = (neighbors == 3) | (alive & neighbors == 2)
    const std::uint64_t next = eq3 | (eq2 & mC);

    output[row_base + col] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is the grid width and height (square), and a power of two > 512.
    // Each row holds words_per_row 64-bit words.
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // Choose a block geometry that favors coalesced accesses along X (columns of words)
    // and covers multiple rows per block in Y to keep grid.y within limits.
    const int block_x = 128;
    const int block_y = 4;

    dim3 block(block_x, block_y, 1);
    dim3 grid((words_per_row + block_x - 1) / block_x,
              (grid_dimensions + block_y - 1) / block_y,
              1);

    game_of_life_step_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
