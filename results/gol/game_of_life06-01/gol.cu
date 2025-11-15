// Conway's Game of Life - CUDA implementation with bit-packed cells.
//
// Each cell is represented by one bit in a 64-bit word (std::uint64_t).
// The grid is N x N cells (N is a power of two), stored row-major.
// One CUDA thread processes one 64-bit word (64 cells) and computes
// the next-generation state using purely bitwise operations.
//
// Neighbor counting is done in a fully bit-parallel way:
// - First, for each of the three relevant rows (above, current, below),
//   we construct three bit-vectors corresponding to the left, center, and
//   right neighbors for every cell in the current word.
//   Cross-word dependencies at bit 0 and bit 63 are handled by explicitly
//   loading the left and right words and combining via shifts.
//
// - For each row we sum these three bit-vectors using full adder logic:
//      sum = a ^ b ^ c
//      carry = (a & b) | (a & c) | (b & c)
//   This yields a 2-bit per-cell neighbor count (0..3) for that row.
//
// - We then add the three row-wise counts (top, middle, bottom) using
//   cascaded bit-sliced adders (again only XOR/AND/OR), resulting in a
//   4-bit per-cell neighbor count (0..8).
//
// - Finally, we compute masks for "neighbor count == 2" and "== 3" and
//   apply Conway's rules:
//
//      alive' = (alive & (neighbors == 2 || neighbors == 3))
//             | (~alive & (neighbors == 3))
//
// All arithmetic on counts is done with bit-sliced adders so there is no
// carry propagation between bit positions; each of the 64 cells in the word
// is processed independently but in parallel within each 64-bit operation.
//
// No shared memory or textures are used; performance relies on regular,
// coalesced global memory accesses and bit-level parallelism.

#include <cstdint>
#include <cstddef>

////////////////////////////////////////////////////////////////////////////////
// Bit-sliced helper functions (device-only, inlined)
////////////////////////////////////////////////////////////////////////////////

// Full adder for three 1-bit inputs in bit-sliced form.
// For each bit position i:
//   sum[i]   = a[i] ^ b[i] ^ c[i]
//   carry[i] = majority(a[i], b[i], c[i])
__device__ __forceinline__
void full_adder3(std::uint64_t a,
                 std::uint64_t b,
                 std::uint64_t c,
                 std::uint64_t &sum,
                 std::uint64_t &carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c); // majority function
}

// Add two 2-bit per-cell values A and B (bit-sliced).
// A = (a1,a0), B = (b1,b0), each component is a 64-bit word containing one bit per cell.
// Result C = A + B is 3 bits per cell: (c2,c1,c0).
//
// This is standard binary addition applied independently to each bit position:
//   - Add LSBs a0 and b0.
//   - Add MSBs a1, b1 and the carry from the LSB addition.
__device__ __forceinline__
void add_2bit_vectors(std::uint64_t a1, std::uint64_t a0,
                      std::uint64_t b1, std::uint64_t b0,
                      std::uint64_t &c2, std::uint64_t &c1, std::uint64_t &c0)
{
    // LSB addition
    std::uint64_t sum0   = a0 ^ b0;
    std::uint64_t carry0 = a0 & b0;

    // MSB addition (without carry-in)
    std::uint64_t sum1_tmp   = a1 ^ b1;
    std::uint64_t carry1_tmp = a1 & b1;

    // Incorporate carry from LSB into MSB
    std::uint64_t sum1   = sum1_tmp ^ carry0;
    std::uint64_t carry1 = carry1_tmp | (sum1_tmp & carry0);

    c0 = sum0;
    c1 = sum1;
    c2 = carry1;
}

// Add a 3-bit per-cell value A and a 2-bit per-cell value B.
// A = (a2,a1,a0), B = (b1,b0).
// Result R = A + B is 4 bits per cell: (r3,r2,r1,r0).
__device__ __forceinline__
void add_3bit_and_2bit(std::uint64_t a2, std::uint64_t a1, std::uint64_t a0,
                       std::uint64_t b1, std::uint64_t b0,
                       std::uint64_t &r3, std::uint64_t &r2,
                       std::uint64_t &r1, std::uint64_t &r0)
{
    // LSB addition: a0 + b0
    std::uint64_t s0 = a0 ^ b0;
    std::uint64_t c0 = a0 & b0;

    // Next bit: a1 + b1 + c0
    std::uint64_t sum1_tmp   = a1 ^ b1;
    std::uint64_t carry1_tmp = a1 & b1;

    std::uint64_t s1 = sum1_tmp ^ c0;
    std::uint64_t c1 = carry1_tmp | (sum1_tmp & c0);

    // Next bit: a2 + c1
    std::uint64_t s2 = a2 ^ c1;
    std::uint64_t c2 = a2 & c1;

    r0 = s0;
    r1 = s1;
    r2 = s2;
    r3 = c2;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel
////////////////////////////////////////////////////////////////////////////////

__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dimensions,
                         int words_per_row)
{
    const int total_words = grid_dimensions * words_per_row;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Compute row and word index within the row for this thread.
    const int row      = idx / words_per_row;
    const int word_col = idx - row * words_per_row;

    // Convenience lambda to safely load a word or return 0 if out of bounds.
    auto load_word = [input, words_per_row, grid_dimensions] __device__
                     (int r, int c) -> std::uint64_t {
        if (r < 0 || r >= grid_dimensions || c < 0 || c >= words_per_row)
            return 0ull;
        return input[static_cast<std::size_t>(r) * words_per_row + c];
    };

    // Load the 3x3 block of 64-bit words surrounding the current word:
    //
    //   up_left    up_center    up_right
    //   mid_left   mid_center   mid_right
    //   dn_left    dn_center    dn_right
    //
    // Words outside the grid are treated as 0 (dead cells).
    const int row_up   = row - 1;
    const int row_dn   = row + 1;
    const int col_left = word_col - 1;
    const int col_right= word_col + 1;

    const std::uint64_t up_center  = load_word(row_up, word_col);
    const std::uint64_t up_left    = load_word(row_up, col_left);
    const std::uint64_t up_right   = load_word(row_up, col_right);

    const std::uint64_t mid_center = load_word(row,    word_col);
    const std::uint64_t mid_left   = load_word(row,    col_left);
    const std::uint64_t mid_right  = load_word(row,    col_right);

    const std::uint64_t dn_center  = load_word(row_dn, word_col);
    const std::uint64_t dn_left    = load_word(row_dn, col_left);
    const std::uint64_t dn_right   = load_word(row_dn, col_right);

    // Construct bit-vectors for neighbor cells relative to each bit position
    // in the current word (word_col, row). For each row (up, mid, down) we
    // build three vectors: left, center, right, where
    //
    //   row_left  : cell to the left  (x-1, same row)
    //   row_center: cell directly above/below/same (x, same row)
    //   row_right : cell to the right (x+1, same row)
    //
    // For bit 0 and bit 63 we must cross word boundaries:
    //   - bit 0's left neighbor is bit 63 of the word to the left
    //   - bit 63's right neighbor is bit 0 of the word to the right
    //
    // These are handled by combining shifts of the current word with shifts
    // of the adjacent words.

    // Top row neighbors
    const std::uint64_t upL = (up_center << 1) | (up_left  >> 63);  // (x-1, y-1)
    const std::uint64_t upC =  up_center;                           // (x  , y-1)
    const std::uint64_t upR = (up_center >> 1) | (up_right << 63);  // (x+1, y-1)

    // Middle row (same y) neighbors (left and right only; center is not a neighbor)
    const std::uint64_t midL = (mid_center << 1) | (mid_left  >> 63); // (x-1, y)
    const std::uint64_t midC =  mid_center;                            // current cell state
    const std::uint64_t midR = (mid_center >> 1) | (mid_right << 63); // (x+1, y)

    // Bottom row neighbors
    const std::uint64_t dnL = (dn_center << 1) | (dn_left  >> 63);   // (x-1, y+1)
    const std::uint64_t dnC =  dn_center;                            // (x  , y+1)
    const std::uint64_t dnR = (dn_center >> 1) | (dn_right << 63);   // (x+1, y+1)

    // Row-wise neighbor counts using full adder logic.
    //
    // For the top and bottom rows, there are three neighbors horizontally:
    //    top:    upL, upC, upR
    //    bottom: dnL, dnC, dnR
    // For the middle row, there are only two neighbors horizontally:
    //    mid:    midL, midR
    // We can treat the middle row as a 3-input adder with the third input = 0.
    //
    // Each row's count per cell is in [0,3] and encoded in two bitplanes:
    //   row_lo (LSB), row_hi (MSB).

    std::uint64_t top_lo, top_hi;
    full_adder3(upL, upC, upR, top_lo, top_hi);  // top row: 3 neighbors

    const std::uint64_t mid_lo = midL ^ midR;    // mid row: 2 neighbors + 0
    const std::uint64_t mid_hi = midL & midR;

    std::uint64_t bot_lo, bot_hi;
    full_adder3(dnL, dnC, dnR, bot_lo, bot_hi);  // bottom row: 3 neighbors

    // Sum the three row-wise counts (top, mid, bottom) using cascaded
    // bit-sliced adders:
    //
    //   Step 1: P = top + mid  -> 3 bits (p2,p1,p0)
    //   Step 2: N = P + bottom -> 4 bits (n3,n2,n1,n0), 0 <= neighbors <= 8

    std::uint64_t p2, p1, p0;
    add_2bit_vectors(top_hi, top_lo, mid_hi, mid_lo, p2, p1, p0);

    std::uint64_t n3, n2, n1, n0;
    add_3bit_and_2bit(p2, p1, p0, bot_hi, bot_lo, n3, n2, n1, n0);

    // Now apply Conway's rules based on the neighbor count.
    //
    // neighbor == 2  <=>  n3=0, n2=0, n1=1, n0=0
    // neighbor == 3  <=>  n3=0, n2=0, n1=1, n0=1
    //
    // We derive bitmasks eq2 and eq3 for these cases, then compute:
    //
    //   alive_next = (alive & (eq2 | eq3)) | (~alive & eq3)

    const std::uint64_t n3_zero = ~n3;
    const std::uint64_t n2_zero = ~n2;

    const std::uint64_t eq2 = n3_zero & n2_zero &  n1 & ~n0;
    const std::uint64_t eq3 = n3_zero & n2_zero &  n1 &  n0;

    const std::uint64_t alive   = midC;
    const std::uint64_t survive = alive & (eq2 | eq3);
    const std::uint64_t born    = (~alive) & eq3;

    const std::uint64_t next = survive | born;

    output[idx] = next;
}

////////////////////////////////////////////////////////////////////////////////
// Host wrapper
////////////////////////////////////////////////////////////////////////////////

// Executes one Game of Life step on the GPU.
//
// Parameters:
//   input           - device pointer to input grid (bit-packed, N x N cells).
//   output          - device pointer to output grid (bit-packed, N x N cells).
//   grid_dimensions - number of cells along one dimension (N).
//
// Notes:
//   - Each std::uint64_t encodes 64 consecutive cells within a row.
//   - The grid is assumed square (N x N) and N is a power of two > 512.
//   - input and output must already be allocated in device memory.
//   - This function launches the kernel asynchronously; synchronization,
//     error checking, and multi-step iteration are handled by the caller.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) *
        static_cast<std::size_t>(words_per_row);

    const int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    game_of_life_kernel<<<blocks, threads_per_block>>>(
        input, output, grid_dimensions, words_per_row);
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
