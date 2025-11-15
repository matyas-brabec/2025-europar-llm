// Conway's Game of Life, bit-packed and optimized for modern NVIDIA GPUs.
//
// Each cell is a single bit (1 = alive, 0 = dead). The grid is square,
// with dimensions that are powers of two (> 512), and every row consists
// of grid_dimensions / 64 `std::uint64_t` words. Each CUDA thread processes
// one 64-bit word (64 cells) and computes the next generation using only
// register operations and global memory.
//
// Neighbor counts are computed in parallel across all 64 bits by treating
// the eight neighbor bitmasks as inputs to a small bit-sliced adder network
// built from full adders (XOR for sum, majority for carry). This yields a
// 4-bit per-cell neighbor count (0..8) without cross-bit carry propagation,
// so 64 independent cells are updated simultaneously using bitwise logic.

#include <cstdint>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Bit-sliced full adder for three 64-bit words.
//
// For each bit position i, this computes:
//   sum[i]   = a[i] XOR b[i] XOR c[i]
//   carry[i] = majority(a[i], b[i], c[i])
//            = (a[i] & b[i]) | (a[i] & c[i]) | (b[i] & c[i])
//
// This behaves like a full adder for three 1-bit inputs replicated across
// 64 lanes in parallel.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void full_add_3(std::uint64_t a,
                                           std::uint64_t b,
                                           std::uint64_t c,
                                           std::uint64_t &sum,
                                           std::uint64_t &carry)
{
    // Compute XOR of a and b once and reuse.
    const std::uint64_t axb = a ^ b;
    sum   = axb ^ c;
    // Majority function: (a & b) | (c & (a ^ b))
    carry = (a & b) | (c & axb);
}

// -----------------------------------------------------------------------------
// Compute 4-bit neighbor counts for 64 cells in parallel.
//
// Inputs n0..n7 are bitmasks for the eight neighbor positions:
//
//   n0: top-left       (TL)
//   n1: top            (T)
//   n2: top-right      (TR)
//   n3: left           (L)
//   n4: right          (R)
//   n5: bottom-left    (BL)
//   n6: bottom         (B)
//   n7: bottom-right   (BR)
//
// For each bit position, we count how many of these eight masks have a 1,
// producing a 4-bit count in binary (0..8):
//
//   bit0: least significant bit (1's place)
//   bit1: 2's place
//   bit2: 4's place
//   bit3: 8's place
//
// This is done using a small tree of full adders that never propagates carry
// between bit positions; all 64 cells are processed independently.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void neighbor_count_8(std::uint64_t n0,
                                                 std::uint64_t n1,
                                                 std::uint64_t n2,
                                                 std::uint64_t n3,
                                                 std::uint64_t n4,
                                                 std::uint64_t n5,
                                                 std::uint64_t n6,
                                                 std::uint64_t n7,
                                                 std::uint64_t &bit0,
                                                 std::uint64_t &bit1,
                                                 std::uint64_t &bit2,
                                                 std::uint64_t &bit3)
{
    // Stage 1: Compress eight neighbor bits into three partial sums (columns).
    //
    // We group neighbors by their column relative to the current cell:
    //
    //   Left column:   TL (n0), L (n3),  BL (n5)
    //   Center column: T  (n1), B (n6),  0
    //   Right column:  TR (n2), R (n4), BR (n7)
    //
    // Each group of three bits is reduced using a full adder, giving:
    //   sX: 1's bit of the local count (0..3)
    //   cX: 2's bit of the local count
    std::uint64_t sA, cA;
    std::uint64_t sB, cB;
    std::uint64_t sC, cC;

    // Left column: n0 + n3 + n5
    full_add_3(n0, n3, n5, sA, cA);
    // Center column: n1 + n6 + 0
    full_add_3(n1, n6, 0ull, sB, cB);
    // Right column: n2 + n4 + n7
    full_add_3(n2, n4, n7, sC, cC);

    // Stage 2: Sum the three partial column counts.
    //
    // Each column count is sX + 2*cX, so:
    //   S = sA + sB + sC      (0..3)
    //   C = cA + cB + cC      (0..3)
    //
    // We again use full adders to compute these 2-bit results:
    //   S = sS + 2*cS
    //   C = sT + 2*cT
    std::uint64_t sS, cS;
    std::uint64_t sT, cT;

    full_add_3(sA, sB, sC, sS, cS);  // S = sA + sB + sC
    full_add_3(cA, cB, cC, sT, cT);  // C = cA + cB + cC

    // Stage 3: Combine S and C into the total neighbor count.
    //
    // Total count = S + 2*C
    //             = (sS + 2*cS) + 2*(sT + 2*cT)
    //
    // We treat S as a 4-bit number with bits [sS, cS, 0, 0]
    // and 2*C as a 4-bit number with bits [0, sT, cT, 0], then perform
    // a 4-bit add (LSB to MSB) using full adders.
    //
    // This yields the final 4-bit count in bit0..bit3.
    bit0 = sS;  // LSB of the total count (no carry-in at bit 0).
    std::uint64_t carry = 0ull;

    // Bit 1: add S bit1 (cS), (2*C) bit1 (sT), and carry from bit 0 (0).
    full_add_3(cS, sT, carry, bit1, carry);

    // Bit 2: add S bit2 (0), (2*C) bit2 (cT), and carry from bit 1.
    full_add_3(0ull, cT, carry, bit2, carry);

    // Bit 3: add S bit3 (0), (2*C) bit3 (0), and carry from bit 2.
    // The sum here is the 8's bit; any further carry would correspond to
    // counts >= 16, which are impossible (maximum is 8 neighbors).
    bit3 = carry;
}

// -----------------------------------------------------------------------------
// CUDA kernel: one thread per 64-cell word.
//
// `input` and `output` are bit-packed grids; each row has `words_per_row`
// 64-bit words. The grid is `grid_dim_cells` by `grid_dim_cells` cells.
//
// For each word (thread), we:
//   1. Load the current word and its 8 surrounding neighbor words (with
//      boundary checks; out-of-range words are treated as all zeros).
//   2. Construct eight neighbor bitmasks (TL, T, TR, L, R, BL, B, BR),
//      correctly handling bit 0 and bit 63 via cross-word shifts.
//   3. Use `neighbor_count_8` to compute the neighbor count (0..8) per bit.
//   4. Apply Game of Life rules to derive the next state word.
// -----------------------------------------------------------------------------
__global__ void game_of_life_kernel(const std::uint64_t * __restrict__ input,
                                    std::uint64_t       * __restrict__ output,
                                    int grid_dim_cells,
                                    int words_per_row,
                                    int log2_words_per_row)
{
    const int total_words = grid_dim_cells * words_per_row;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_words)
        return;

    // Convert global word index into (row, col_in_row).
    const int row = gid >> log2_words_per_row;          // row index in [0, grid_dim_cells)
    const int col = gid & (words_per_row - 1);          // word index within the row

    const bool has_north = (row > 0);
    const bool has_south = (row + 1 < grid_dim_cells);
    const bool has_west  = (col > 0);
    const bool has_east  = (col + 1 < words_per_row);

    // Pointers to the current row and its vertical neighbors.
    const std::uint64_t *row_ptr = input + static_cast<std::size_t>(row) * words_per_row;

    std::uint64_t center = row_ptr[col];
    std::uint64_t west   = has_west ? row_ptr[col - 1] : 0ull;
    std::uint64_t east   = has_east ? row_ptr[col + 1] : 0ull;

    std::uint64_t north      = 0ull;
    std::uint64_t north_west = 0ull;
    std::uint64_t north_east = 0ull;

    if (has_north) {
        const std::uint64_t *north_row = row_ptr - words_per_row;
        north = north_row[col];
        if (has_west) {
            north_west = north_row[col - 1];
        }
        if (has_east) {
            north_east = north_row[col + 1];
        }
    }

    std::uint64_t south      = 0ull;
    std::uint64_t south_west = 0ull;
    std::uint64_t south_east = 0ull;

    if (has_south) {
        const std::uint64_t *south_row = row_ptr + words_per_row;
        south = south_row[col];
        if (has_west) {
            south_west = south_row[col - 1];
        }
        if (has_east) {
            south_east = south_row[col + 1];
        }
    }

    // Construct neighbor bitmasks. Each mask is aligned so that bit i
    // corresponds to the neighbor of the cell in bit i of `center`.
    //
    // For horizontal neighbors and diagonals we need to bridge across
    // 64-bit word boundaries. For example, the left neighbor of bit 0
    // comes from bit 63 of the word to the left (if it exists).
    //
    // Bit numbering convention within a word:
    //   bit 0   -> leftmost cell in the 64-cell block
    //   bit 63  -> rightmost cell in the block
    //
    // Neighbors:
    //   TL (top-left):    above row, column-1
    //   T  (top):         above row, same column
    //   TR (top-right):   above row, column+1
    //   L  (left):        same row, column-1
    //   R  (right):       same row, column+1
    //   BL (bottom-left): below row, column-1
    //   B  (bottom):      below row, same column
    //   BR (bottom-right):below row, column+1
    //
    // Diagonal masks are formed by shifting the center word of the row and
    // inserting the edge bits from the neighboring word.
    const std::uint64_t top_left     = (north << 1) | (north_west >> 63);
    const std::uint64_t top          = north;
    const std::uint64_t top_right    = (north >> 1) | (north_east << 63);

    const std::uint64_t left         = (center << 1) | (west >> 63);
    const std::uint64_t right        = (center >> 1) | (east << 63);

    const std::uint64_t bottom_left  = (south << 1) | (south_west >> 63);
    const std::uint64_t bottom       = south;
    const std::uint64_t bottom_right = (south >> 1) | (south_east << 63);

    // Compute 4-bit neighbor count for each cell in the 64-bit word.
    std::uint64_t cnt0, cnt1, cnt2, cnt3;
    neighbor_count_8(top_left, top, top_right,
                     left, right,
                     bottom_left, bottom, bottom_right,
                     cnt0, cnt1, cnt2, cnt3);

    // Apply Conway's Game of Life rules:
    //
    //   - Any live cell with two or three neighbors survives.
    //   - Any dead cell with exactly three neighbors becomes alive.
    //   - All other cells die or remain dead.
    //
    // We need masks for neighbor_count == 2 and neighbor_count == 3.
    //
    // The 4-bit count per cell is:
    //   cnt3 cnt2 cnt1 cnt0  (from MSB to LSB)
    //
    // 2 (0010): cnt3=0, cnt2=0, cnt1=1, cnt0=0
    // 3 (0011): cnt3=0, cnt2=0, cnt1=1, cnt0=1
    const std::uint64_t not_cnt3 = ~cnt3;
    const std::uint64_t not_cnt2 = ~cnt2;
    const std::uint64_t not_cnt1 = ~cnt1;
    const std::uint64_t not_cnt0 = ~cnt0;

    const std::uint64_t eq3 = not_cnt3 & not_cnt2 & cnt1      & cnt0;
    const std::uint64_t eq2 = not_cnt3 & not_cnt2 & cnt1      & not_cnt0;

    // Next state: born if exactly 3 neighbors, survive if alive and exactly 2.
    const std::uint64_t next = eq3 | (center & eq2);

    output[gid] = next;
}

// -----------------------------------------------------------------------------
// Host function: run one Game of Life step on the GPU.
//
// `input`  - device pointer to current grid state (bit-packed).
// `output` - device pointer to next grid state (bit-packed).
// `grid_dimensions` - side length of the square grid in cells.
//                     It is guaranteed to be a power of two > 512.
//
// Assumptions:
//   - `input` and `output` are allocated with cudaMalloc and are large enough
//     to hold grid_dimensions * grid_dimensions cells, bit-packed as
//     std::uint64_t (64 cells per word).
//   - The caller performs any required synchronization after this call.
// -----------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t*       output,
                      int                  grid_dimensions)
{
    // Number of 64-bit words per row.
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // log2(words_per_row), used inside the kernel to compute (row, col)
    // using shift and mask instead of expensive integer division/modulo.
    int log2_words_per_row = 0;
    int tmp = words_per_row;
    while ((tmp >>= 1) != 0) {
        ++log2_words_per_row;
    }

    const int total_words = grid_dimensions * words_per_row;

    // Launch configuration: one thread per 64-bit word.
    // 256 threads per block is a good general-purpose choice for A100/H100.
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = (total_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    game_of_life_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
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
