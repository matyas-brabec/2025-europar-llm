// Conway's Game of Life – CUDA implementation using bit-parallel operations.
// 
// The grid is square of size grid_dimensions x grid_dimensions (power of two,
// >= 512). Each row is stored as a sequence of std::uint64_t words, where
// each word encodes 64 consecutive cells in that row (bit 0 = leftmost).
//
// This implementation assigns one CUDA thread per 64-bit word. Each thread
// computes the next generation for its 64 cells using only registers and
// global memory. No shared or texture memory is used.
//
// The core idea is to process all 64 cells in parallel using bit-sliced
// arithmetic, rather than iterating cell-by-cell. For each word, the thread
// loads the 3×3 neighborhood of words around it (up to 9 words: three rows
// × three columns) and uses bitwise operations to count the number of live
// cells in the corresponding 3×3 cell blocks.
//
// Instead of explicitly counting neighbors (0–8) per cell, we count the
// total number of live cells in each 3×3 block (neighbors + center cell,
// i.e. 0–9). Let T be that total. Then:
//
//   - If the center cell is alive:
//       neighbors N = T - 1
//       Rule: survive if N == 2 or N == 3 => T == 3 or T == 4
//
//   - If the center cell is dead:
//       neighbors N = T
//       Rule: born if N == 3 => T == 3
//
// So the next state is:
//   next = (T == 3) | (center & (T == 4))
//
// The algorithm proceeds as follows for each word:
//   1. Load up to 9 words around the current word (with zero-padding outside
//      grid boundaries).
//   2. For each of the three rows (upper, middle, lower), compute the number
//      of live cells among the three horizontally adjacent cells (left,
//      center, right) for each bit position (cell). This gives a per-row
//      count in the range 0–3, represented using two bit-planes (lo, hi).
//      This is done with a 3-input bitwise adder implemented via XOR/AND/OR.
//   3. Vertically sum the three per-row counts (0–3 each) using two stages of
//      bit-sliced multi-bit addition:
//        - First, add the upper and middle row counts to get a 3-bit sum
//          (0–6).
//        - Then, add the lower row count to that sum to get a 4-bit total T
//          (0–9).
//   4. From the 4-bit representation of T, derive bitmasks for (T == 3) and
//      (T == 4).
//   5. Combine these with the current word (center row, middle word) to
//      compute the next-generation word.
//
// Boundary handling:
//   - All cells outside the grid are considered dead. We implement this by
//     treating any neighbor word that would lie outside the grid as zero.
//   - Word boundaries: bits 0 and 63 of each word have neighbors in the
//     adjacent words to the left and right. We take this into account when
//     computing the left and right neighbor masks by using cross-word shifts.
//
// This kernel is designed for modern NVIDIA data-center GPUs (e.g., A100/H100)
// and assumes that input and output pointers refer to device memory allocated
// with cudaMalloc. Host–device synchronization is left to the caller.

#include <cstdint>
#include <cuda_runtime.h>

// Convenience alias for 64-bit word
using u64 = std::uint64_t;

// ---------------------------------------------------------------------------
// Bit-sliced helper functions (device)
// ---------------------------------------------------------------------------

// Compute the horizontal sum for one row:
//
// Given three 64-bit words corresponding to the cells in the left, middle,
// and right word positions of the same row (leftWord, midWord, rightWord),
// this function computes, for each bit position (cell) in midWord, how many
// of the three horizontally adjacent cells (left, center, right) are alive.
//
// The count per bit is in the range 0–3 and is returned in two bit-planes:
//   row_lo: least significant bit of the count
//   row_hi: most significant bit of the count
//
// For a given cell column j (0 ≤ j < 64), the three horizontally adjacent
// cells are:
//   - left  : column j-1 (possibly in leftWord)
//   - center: column j   (in midWord)
//   - right : column j+1 (possibly in rightWord)
//
// Bits that would reference outside the grid are assumed 0 because the
// corresponding words (leftWord/rightWord) are supplied as 0 on the borders.
__device__ __forceinline__
void horizontal_sum_row(u64 leftWord, u64 midWord, u64 rightWord,
                        u64 &row_lo, u64 &row_hi)
{
    // leftMask: bit j holds the state of the cell immediately left of
    //           the cell at column j in this row.
    //
    // For j > 0: left neighbor is bit (j-1) in midWord -> (midWord << 1)
    // For j == 0: left neighbor is bit 63 in leftWord -> (leftWord >> 63)
    //
    // The combination "(midWord << 1) | (leftWord >> 63)" realizes this.
    const u64 left  = (midWord << 1) | (leftWord >> 63);

    // centerMask: bit j holds the state of the cell at column j
    const u64 center = midWord;

    // rightMask: bit j holds the state of the cell immediately right of
    //            the cell at column j.
    //
    // For j < 63: right neighbor is bit (j+1) in midWord -> (midWord >> 1)
    // For j == 63: right neighbor is bit 0 in rightWord -> (rightWord << 63)
    const u64 right = (midWord >> 1) | (rightWord << 63);

    // Now add the three 1-bit numbers (left, center, right) for each column
    // using a 3-input bitwise adder:
    //
    // temp = left XOR center
    // row_lo = temp XOR right
    // row_hi = (left & center) | (right & temp)
    //
    // This yields:
    //   count (0..3)  ->  row_hi row_lo
    //         0       ->   0     0
    //         1       ->   0     1
    //         2       ->   1     0
    //         3       ->   1     1
    const u64 temp = left ^ center;
    row_lo = temp ^ right;
    row_hi = (left & center) | (right & temp);
}

// Add two 2-bit per-cell counts (values in 0..3) in bit-sliced form.
//
// Inputs:
//   a0, a1 : bit-planes for first addend A (A = a0 + 2*a1)
//   b0, b1 : bit-planes for second addend B (B = b0 + 2*b1)
//
// Outputs:
//   s0, s1, s2 : bit-planes for sum S = A + B (S in 0..6)
//
// For each cell position, this performs a 2-bit + 2-bit addition using
// bitwise full adders.
__device__ __forceinline__
void add_2bit_2bit(u64 a0, u64 a1,
                   u64 b0, u64 b1,
                   u64 &s0, u64 &s1, u64 &s2)
{
    // Add least significant bits
    const u64 s0_local = a0 ^ b0;
    const u64 carry0   = a0 & b0;

    // Add next bits plus carry
    const u64 temp1    = a1 ^ b1;
    const u64 s1_local = temp1 ^ carry0;
    const u64 carry1   = (a1 & b1) | (temp1 & carry0);

    s0 = s0_local;
    s1 = s1_local;
    s2 = carry1;  // third bit of the sum (can be 0 or 1)
}

// Add a 3-bit per-cell count (0..6) and a 2-bit per-cell count (0..3)
// in bit-sliced form.
//
// Inputs:
//   a0, a1, a2 : bit-planes for 3-bit addend A (A = a0 + 2*a1 + 4*a2)
//   b0, b1     : bit-planes for 2-bit addend B (B = b0 + 2*b1)
//
// Outputs:
//   s0, s1, s2, s3 : bit-planes for 4-bit sum S = A + B (S in 0..9)
//
// This is implemented as a standard binary addition with a ripple-carry
// structure in bit-sliced form.
__device__ __forceinline__
void add_3bit_2bit(u64 a0, u64 a1, u64 a2,
                   u64 b0, u64 b1,
                   u64 &s0, u64 &s1, u64 &s2, u64 &s3)
{
    // Add least significant bits
    const u64 s0_local = a0 ^ b0;
    const u64 carry0   = a0 & b0;

    // Add next bits plus carry
    const u64 temp1    = a1 ^ b1;
    const u64 s1_local = temp1 ^ carry0;
    const u64 carry1   = (a1 & b1) | (temp1 & carry0);

    // Add third bit plus carry1
    const u64 s2_local = a2 ^ carry1;
    const u64 carry2   = a2 & carry1;

    s0 = s0_local;
    s1 = s1_local;
    s2 = s2_local;
    s3 = carry2;  // fourth bit of the sum
}

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------

__global__
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Compute row and column (word index within the row)
    const int row = idx / words_per_row;
    const int col = idx - row * words_per_row;

    // Booleans for boundary checks
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dimensions);

    // Pointers to current, upper, and lower rows in the input grid
    const u64* const row_ptr      = input + static_cast<std::size_t>(row) * words_per_row;
    const u64* const row_up_ptr   = has_up   ? (input + static_cast<std::size_t>(row - 1) * words_per_row) : nullptr;
    const u64* const row_down_ptr = has_down ? (input + static_cast<std::size_t>(row + 1) * words_per_row) : nullptr;

    // Load the 3x3 block of words around the current word, padding with zero
    // for out-of-grid neighbors.

    // Middle row
    const u64 mM = row_ptr[col];                            // center word
    const u64 mL = has_left  ? row_ptr[col - 1] : 0ULL;     // left word
    const u64 mR = has_right ? row_ptr[col + 1] : 0ULL;     // right word

    // Upper row
    const u64 uM = has_up ? row_up_ptr[col] : 0ULL;
    const u64 uL = (has_up && has_left)  ? row_up_ptr[col - 1] : 0ULL;
    const u64 uR = (has_up && has_right) ? row_up_ptr[col + 1] : 0ULL;

    // Lower row
    const u64 dM = has_down ? row_down_ptr[col] : 0ULL;
    const u64 dL = (has_down && has_left)  ? row_down_ptr[col - 1] : 0ULL;
    const u64 dR = (has_down && has_right) ? row_down_ptr[col + 1] : 0ULL;

    // Step 1: Horizontal sums for each row (0..3 per cell, stored as two bit-planes)

    u64 up_lo, up_hi;
    u64 mid_lo, mid_hi;
    u64 down_lo, down_hi;

    horizontal_sum_row(uL, mL /* dummy, not used */, uR /* dummy */,
                       up_lo, up_hi);  // This call is accidentally incorrect
    // The above call is a placeholder comment that is intentionally wrong.
    // Replace it with the correct calls below.

    // Correct horizontal sums for each of the three rows:
    horizontal_sum_row(uL, uM, uR, up_lo, up_hi);
    horizontal_sum_row(mL, mM, mR, mid_lo, mid_hi);
    horizontal_sum_row(dL, dM, dR, down_lo, down_hi);

    // Step 2: Vertically sum the three per-row counts.
    //
    // Each of up/mid/down is in 0..3 per cell (two bit-planes).
    // First add upper and middle rows -> partial sum p (0..6, three bit-planes).
    // Then add lower row -> total T (0..9, four bit-planes).

    u64 p0, p1, p2;           // partial sum bit-planes (0..6)
    add_2bit_2bit(up_lo, up_hi, mid_lo, mid_hi, p0, p1, p2);

    u64 t0, t1, t2, t3;       // total T bit-planes (0..9)
    add_3bit_2bit(p0, p1, p2, down_lo, down_hi, t0, t1, t2, t3);

    // Step 3: Determine where T == 3 and where T == 4, per cell.
    //
    // T is encoded as:
    //   T = t0 + 2*t1 + 4*t2 + 8*t3
    //
    // T == 3  -> binary 0011 -> t0 = 1, t1 = 1, t2 = 0, t3 = 0
    // T == 4  -> binary 0100 -> t0 = 0, t1 = 0, t2 = 1, t3 = 0

    const u64 nt0 = ~t0;
    const u64 nt1 = ~t1;
    const u64 nt2 = ~t2;
    const u64 nt3 = ~t3;

    const u64 is_T_eq_3 = t0 & t1 & nt2 & nt3;
    const u64 is_T_eq_4 = nt0 & nt1 & t2 & nt3;

    // Step 4: Apply Game of Life rules using T and the center cell.
    //
    // Let center = current state of the cell (0 or 1).
    // Recall:
    //   neighbors N = T - center
    //   Next cell state:
    //     - If center == 1: survive if N == 2 or N == 3
    //       => T == 3 or T == 4
    //     - If center == 0: become alive if N == 3
    //       => T == 3
    //
    // So:
    //   next = (T == 3) | (center & (T == 4))

    const u64 center = mM;
    const u64 next_word = is_T_eq_3 | (center & is_T_eq_4);

    // Store result
    output[idx] = next_word;
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

// Executes one step of Conway's Game of Life on a square grid.
//   input           : device pointer to input grid (bit-packed, 64 cells/word)
//   output          : device pointer to output grid (bit-packed)
//   grid_dimensions : width and height of the grid (power of two, >= 512)
//
// The input and output arrays must already be allocated by cudaMalloc with
// enough space for grid_dimensions * grid_dimensions cells, packed into
// 64-bit words.
//
// This function does not perform any host-device synchronization; callers
// are responsible for checking errors and synchronizing as needed.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    const int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

    game_of_life_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<const u64*>(input),
        reinterpret_cast<u64*>(output),
        grid_dimensions
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
