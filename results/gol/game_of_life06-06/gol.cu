#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Grid layout:
 *   - The grid is square: grid_dimensions x grid_dimensions cells.
 *   - Each 64-bit word encodes 64 consecutive cells in a row.
 *   - Words are stored in row-major order.
 *   - Bit i (0 <= i < 64) of word w corresponds to column (w * 64 + i) in that row.
 *
 * Each CUDA thread processes one 64-bit word (i.e., 64 cells) and computes their next
 * generation states using bit-parallel logic with full adders.
 *
 * Neighbor counting:
 *   - For each word, we construct 8 64-bit masks corresponding to the 8 neighbor
 *     directions (N, S, E, W, NE, NW, SE, SW), aligned so that bit k of each mask
 *     is the neighbor of bit k in the current word.
 *   - Horizontal neighbors (W/E) and diagonal neighbors (NW/NE/SW/SE) are built
 *     using bit shifts and cross-word contributions from the left/right words.
 *   - Out-of-bounds neighbors (outside the grid) are treated as dead by using zero.
 *
 *   - We then add these eight 1-bit-per-cell masks using a tree of 3-input full adders:
 *
 *       Given 3 one-bit inputs a, b, c:
 *         sum  = a XOR b XOR c          (least significant bit of their sum)
 *         carry= majority(a,b,c)        (second bit of their sum)
 *
 *       For 8 neighbor masks b0..b7:
 *         1) Group into three triplets:
 *              G0 = (b0, b1, b2)
 *              G1 = (b3, b4, b5)
 *              G2 = (b6, b7, 0)
 *            Each group yields a 2-bit sum (s_i, t_i), representing 0..3.
 *
 *         2) Sum the three group results:
 *              A = s0 + s1 + s2   (via full adder)
 *              B = t0 + t1 + t2   (via full adder)
 *
 *            Write:
 *              A = As + 2*At
 *              B = Bs + 2*Bt
 *
 *            Total neighbor count:
 *              count = A + 2*B
 *                    = As + 2*At + 2*Bs + 4*Bt
 *                    = As + 2*(At + Bs) + 4*Bt
 *
 *         3) Let M = At + Bs (via full adder):
 *              M = Ms + 2*Mt
 *
 *            Then:
 *              count = As + 2*Ms + 4*(Mt + Bt)
 *
 *         4) Interpreting count modulo 8 (we don't need the bit for 8 neighbors):
 *              bit0 (LSB) = As
 *              bit1       = Ms
 *              bit2       = Mt XOR Bt   (since 4*(Mt+Bt) contributes to bit2, with
 *                                         overflow to 8 ignored)
 *
 *       So we obtain three bit-planes (c0,c1,c2) giving the neighbor count modulo 8.
 *       Since there are at most 8 neighbors, modulo-8 encoding is identical to
 *       the true count for values 0..7; 8 maps to 0, which does not equal 2 or 3,
 *       so tests for "exactly 2" and "exactly 3" remain correct.
 *
 * Game of Life rules:
 *   - Let alive be the mask of currently alive cells in the word.
 *   - From (c0,c1,c2), define:
 *       eq2 = (count == 2) = ~c2 &  c1 & ~c0
 *       eq3 = (count == 3) = ~c2 &  c1 &  c0
 *
 *   - Next state:
 *       survive = alive & (eq2 | eq3)
 *       birth   = ~alive & eq3
 *       next    = survive | birth
 */

/* 3-input full adder for 64-bit bitplanes.
 * For each bit position, it adds a, b, c (each 0 or 1) and produces:
 *   sum   = (a + b + c) & 1
 *   carry = ((a + b + c) >> 1) & 1
 * Using:
 *   sum   = a XOR b XOR c
 *   carry = (a & b) | (a & c) | (b & c)   // majority function
 */
__device__ __forceinline__
void full_adder3(std::uint64_t a,
                 std::uint64_t b,
                 std::uint64_t c,
                 std::uint64_t &sum,
                 std::uint64_t &carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

/* Sum eight 1-bit-per-cell neighbor masks using a tree of full adders.
 * Inputs:
 *   b0..b7 : eight 64-bit masks for neighbor directions
 * Outputs:
 *   c0, c1, c2 : bitplanes of neighbor counts (modulo 8)
 *                count = c0 + 2*c1 + 4*c2
 */
__device__ __forceinline__
void neighbor_count_8way(std::uint64_t b0, std::uint64_t b1,
                         std::uint64_t b2, std::uint64_t b3,
                         std::uint64_t b4, std::uint64_t b5,
                         std::uint64_t b6, std::uint64_t b7,
                         std::uint64_t &c0, std::uint64_t &c1, std::uint64_t &c2)
{
    // First level: three groups of three bits (last group padded with 0)
    std::uint64_t s0, t0;
    std::uint64_t s1, t1;
    std::uint64_t s2, t2;

    full_adder3(b0, b1, b2, s0, t0);   // group 0
    full_adder3(b3, b4, b5, s1, t1);   // group 1
    full_adder3(b6, b7, 0,  s2, t2);   // group 2 (padded)

    // Second level: sum the s* bits -> A, and the t* bits -> B
    std::uint64_t As, At;
    std::uint64_t Bs, Bt;

    full_adder3(s0, t0, s1, As, At);    // This is arbitrary; to strictly follow
                                        // the derivation, we should do:
                                        // full_adder3(s0, s1, s2, As, At);
                                        // full_adder3(t0, t1, t2, Bs, Bt);
                                        // but the mix of s and t would be wrong.
    // Correct: follow the derivation strictly
    full_adder3(s0, s1, s2, As, At);    // A = s0 + s1 + s2
    full_adder3(t0, t1, t2, Bs, Bt);    // B = t0 + t1 + t2

    // Third level: M = At + Bs
    std::uint64_t Ms, Mt;
    full_adder3(At, Bs, 0, Ms, Mt);

    // Final neighbor count bits (modulo 8):
    //   bit0 = As
    //   bit1 = Ms
    //   bit2 = Mt XOR Bt
    c0 = As;
    c1 = Ms;
    c2 = Mt ^ Bt;
}

/* Corrected neighbor_count_8way: the above function accidentally mixed s and t
 * in an intermediate (dead) computation before the correct one. To avoid any
 * confusion, we redefine it cleanly here and rely on this version only.
 */
__device__ __forceinline__
void neighbor_count_8way_correct(std::uint64_t b0, std::uint64_t b1,
                                 std::uint64_t b2, std::uint64_t b3,
                                 std::uint64_t b4, std::uint64_t b5,
                                 std::uint64_t b6, std::uint64_t b7,
                                 std::uint64_t &c0, std::uint64_t &c1, std::uint64_t &c2)
{
    std::uint64_t s0, t0;
    std::uint64_t s1, t1;
    std::uint64_t s2, t2;

    // Group 0: b0, b1, b2
    full_adder3(b0, b1, b2, s0, t0);

    // Group 1: b3, b4, b5
    full_adder3(b3, b4, b5, s1, t1);

    // Group 2: b6, b7, 0
    full_adder3(b6, b7, 0,  s2, t2);

    // Sum s0 + s1 + s2 => A = As + 2*At
    std::uint64_t As, At;
    full_adder3(s0, s1, s2, As, At);

    // Sum t0 + t1 + t2 => B = Bs + 2*Bt
    std::uint64_t Bs, Bt;
    full_adder3(t0, t1, t2, Bs, Bt);

    // M = At + Bs => M = Ms + 2*Mt
    std::uint64_t Ms, Mt;
    full_adder3(At, Bs, 0, Ms, Mt);

    // count = As + 2*Ms + 4*(Mt + Bt)  (modulo 8)
    c0 = As;
    c1 = Ms;
    c2 = Mt ^ Bt;
}

/*
 * CUDA kernel: compute one Game of Life step on a bit-packed grid.
 *
 * Parameters:
 *   input          : device pointer to input grid (bit-packed uint64_t words)
 *   output         : device pointer to output grid (same layout as input)
 *   grid_dim       : width/height of square grid (power of 2)
 *   words_per_row  : number of 64-bit words per row (grid_dim / 64)
 *   row_shift      : log2(words_per_row), since words_per_row is a power of 2
 *   total_words    : total number of 64-bit words (words_per_row * grid_dim)
 *
 * Each thread processes one word at index 'idx'.
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row,
                         int row_shift,
                         std::size_t total_words)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      static_cast<std::size_t>(threadIdx.x);
    if (idx >= total_words)
        return;

    // Compute row and column in word space:
    //   row = idx / words_per_row
    //   col = idx % words_per_row
    // Since words_per_row is power-of-two, we can use shifts/masks.
    int row = static_cast<int>(idx >> row_shift);
    int col = static_cast<int>(idx & static_cast<std::size_t>(words_per_row - 1));

    const bool has_north = (row > 0);
    const bool has_south = (row + 1 < grid_dim);
    const bool has_west  = (col > 0);
    const bool has_east  = (col + 1 < words_per_row);

    // Load the central word and its immediate left/right neighbors in the same row.
    const std::uint64_t self   = input[idx];
    const std::uint64_t selfL  = has_west ? input[idx - 1] : 0ull;
    const std::uint64_t selfR  = has_east ? input[idx + 1] : 0ull;

    // Load words from the row above (north) and below (south), including left/right.
    std::uint64_t northC = 0ull, northL = 0ull, northR = 0ull;
    std::uint64_t southC = 0ull, southL = 0ull, southR = 0ull;

    if (has_north) {
        std::size_t baseN = idx - static_cast<std::size_t>(words_per_row);
        northC = input[baseN];
        northL = has_west ? input[baseN - 1] : 0ull;
        northR = has_east ? input[baseN + 1] : 0ull;
    }

    if (has_south) {
        std::size_t baseS = idx + static_cast<std::size_t>(words_per_row);
        southC = input[baseS];
        southL = has_west ? input[baseS - 1] : 0ull;
        southR = has_east ? input[baseS + 1] : 0ull;
    }

    // Construct aligned neighbor bitmasks for the 8 directions.
    //
    // For each central bit position k in [0,63]:
    //   W[k] = cell at (row, col_bit-1)
    //   E[k] = cell at (row, col_bit+1)
    //   N[k] = cell at (row-1, col_bit)
    //   S[k] = cell at (row+1, col_bit)
    //   NW[k]= cell at (row-1, col_bit-1)
    //   NE[k]= cell at (row-1, col_bit+1)
    //   SW[k]= cell at (row+1, col_bit-1)
    //   SE[k]= cell at (row+1, col_bit+1)
    //
    // Horizontal neighbors need cross-word contributions for bit 0 and bit 63.
    const std::uint64_t W  = (self << 1) | (selfL >> 63);
    const std::uint64_t E  = (self >> 1) | (selfR << 63);
    const std::uint64_t N  = northC;
    const std::uint64_t S  = southC;

    const std::uint64_t NW = (northC << 1) | (northL >> 63);
    const std::uint64_t NE = (northC >> 1) | (northR << 63);
    const std::uint64_t SW = (southC << 1) | (southL >> 63);
    const std::uint64_t SE = (southC >> 1) | (southR << 63);

    // Count neighbors using bit-parallel full-adder logic.
    std::uint64_t c0, c1, c2;
    neighbor_count_8way_correct(W, E, N, S, NW, NE, SW, SE, c0, c1, c2);

    // Compute masks for "neighbor count == 2" and "neighbor count == 3".
    const std::uint64_t n1 = c0; // bit 0 of count
    const std::uint64_t n2 = c1; // bit 1 of count
    const std::uint64_t n4 = c2; // bit 2 of count

    const std::uint64_t eq2 = (~n4) &  n2  & (~n1); // 010b
    const std::uint64_t eq3 = (~n4) &  n2  &  n1;   // 011b

    // Apply Game of Life rules.
    const std::uint64_t alive   = self;
    const std::uint64_t survive = alive & (eq2 | eq3); // alive with 2 or 3 neighbors
    const std::uint64_t birth   = (~alive) & eq3;      // dead with exactly 3 neighbors
    const std::uint64_t next    = survive | birth;

    // Store result.
    output[idx] = next;
}

/*
 * Host function to launch one Game of Life step.
 *
 * Parameters:
 *   input           : device pointer to input bit-packed grid (cudaMalloc'ed)
 *   output          : device pointer to output bit-packed grid (cudaMalloc'ed)
 *   grid_dimensions : width/height of the square grid; power of 2 (> 512)
 *
 * Notes:
 *   - This function performs only device-side work; no host-device synchronization
 *     is done here. The caller is responsible for synchronization if needed.
 *   - Shared or texture memory is not used, as per the problem constraints.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each 64-bit word encodes 64 cells.
    constexpr int BITS_PER_WORD = 64;

    // Number of 64-bit words per row (grid_dimensions is a power of 2, >= 512).
    const int words_per_row = grid_dimensions / BITS_PER_WORD;

    // Compute log2(words_per_row) for efficient row/column computation in the kernel.
    int row_shift = 0;
    int tmp = words_per_row;
    while (tmp > 1) {
        tmp >>= 1;
        ++row_shift;
    }

    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) *
        static_cast<std::size_t>(grid_dimensions);

    // Choose a reasonable block size; 256 threads per block is a good default.
    constexpr int BLOCK_SIZE = 256;
    const std::size_t num_blocks =
        (total_words + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    game_of_life_kernel<<<grid_dim, block_dim>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
        row_shift,
        total_words
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
