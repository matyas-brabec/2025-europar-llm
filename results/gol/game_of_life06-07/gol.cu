#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// Bring commonly used fixed-width types into the global namespace for brevity.
using std::uint64_t;
using std::size_t;

/*
 * 3-input carry-save adder (full adder) operating on 64 lanes in parallel.
 *
 * For each bit position i:
 *   a_i + b_i + c_i = (sum_i) + 2 * (carry_i)
 *
 * sum   = a XOR b XOR c
 * carry = majority(a, b, c) = (a & b) | (a & c) | (b & c)
 *
 * This is the fundamental primitive used to accumulate neighbor counts
 * for Conway's Game of Life in a bit-parallel fashion.
 */
static __device__ __forceinline__ void csa3(uint64_t a, uint64_t b, uint64_t c,
                                            uint64_t &sum, uint64_t &carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

/*
 * CUDA kernel: compute one Game of Life step on a bit-packed grid.
 *
 * - The grid is square of dimension grid_dim x grid_dim cells.
 * - Each 64-bit word encodes 64 consecutive cells in a row (bit 0 = column 0).
 * - words_per_row = grid_dim / 64.
 * - One CUDA thread processes exactly one 64-bit word (i.e., 64 cells).
 *
 * Neighbor counting:
 *   For each word we load the 3x3 block of neighboring words (up to 9 words):
 *
 *        nL   nC   nR        row-1 (north)
 *        cL   C    cR        row
 *        sL   sC   sR        row+1 (south)
 *
 *   For each of the three rows we construct three shifted words:
 *     - Row above:  upL (NW), upC (N),  upR (NE)
 *     - Row center: midL (W),  0,       midR (E)  (center cell excluded)
 *     - Row below:  dnL (SW), dnC (S),  dnR (SE)
 *
 *   These eight directions form the 8 neighbors for every bit in the center word.
 *
 *   We then:
 *     1. Use csa3 (full adder) per row to add its three direction words:
 *          top_count  = upL + upC + upR
 *          mid_count  = midL + 0   + midR
 *          bot_count  = dnL + dnC + dnR
 *        Each count in [0,3] is represented as two bitplanes (sum, carry):
 *          row_sum, row_carry  with: row_count = row_sum + 2 * row_carry
 *
 *     2. Add the three row sums (top_sum, mid_sum, bot_sum) with csa3 to
 *        obtain:
 *          S_sum, S_carry
 *        representing:
 *          S_part = top_sum + mid_sum + bot_sum
 *
 *     3. Similarly add the three row carries (top_carry, mid_carry, bot_carry)
 *        with csa3 to get:
 *          C_sum, C_carry
 *        representing:
 *          C_part = top_carry + mid_carry + bot_carry
 *
 *        Note: each row_carry already has weight 2, so C_part contributes
 *              2 * C_part to the neighbor count.
 *
 *     4. Combining everything, for each bit position:
 *
 *          neighbor_count =
 *              S_sum                      (1's place)
 *            + 2 * (S_carry + C_sum)      (2's place)
 *            + 4 * C_carry                (4's and 8's place)
 *
 *        We compute the bit-sliced representation (n0, n1, n2, n3) such that:
 *
 *          neighbor_count = n0 + 2*n1 + 4*n2 + 8*n3
 *
 *        Derivation:
 *          Let T = S_carry + C_sum, T in {0,1,2}:
 *            t0 = S_carry XOR C_sum  (T % 2)
 *            t1 = S_carry &   C_sum  (T == 2)
 *
 *          Let Q = t1 + C_carry, Q in {0,1,2}:
 *            q0 = t1 XOR C_carry     (Q % 2)
 *            q1 = t1 &   C_carry     (Q == 2)
 *
 *          Then:
 *            neighbor_count = S_sum + 2*t0 + 4*q0 + 8*q1
 *
 *          So:
 *            n0 = S_sum
 *            n1 = t0
 *            n2 = q0
 *            n3 = q1
 *
 *     5. Apply Game of Life rules using these bitplanes:
 *          - eq2: neighbor_count == 2  => n3=0, n2=0, n1=1, n0=0
 *          - eq3: neighbor_count == 3  => n3=0, n2=0, n1=1, n0=1
 *
 *          eq2_mask = !n3 & !n2 &  n1 & !n0
 *          eq3_mask = !n3 & !n2 &  n1 &  n0
 *
 *        Next state:
 *          - A dead cell becomes alive if eq3.
 *          - A live cell survives if it has 2 or 3 neighbors.
 *
 *        Bitwise:
 *           next = eq3 | (C & eq2);
 *
 * Boundary handling:
 *   Cells outside the grid are treated as dead (0). For words on the borders
 *   of the grid, the corresponding neighbor words are set to 0 by checking
 *   row/column bounds before loading.
 */
static __global__ void game_of_life_step_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__       output,
    int                          grid_dim,
    int                          words_per_row,
    size_t                       total_words)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                 static_cast<size_t>(threadIdx.x);

    if (idx >= total_words) {
        return;
    }

    const size_t wpr = static_cast<size_t>(words_per_row);

    // Derive 2D position (row, col) of this word in the word grid.
    size_t row = idx / wpr;
    size_t col = idx - row * wpr;
    size_t row_base = row * wpr;

    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < static_cast<size_t>(grid_dim));
    const bool has_left   = (col > 0);
    const bool has_right  = (col + 1 < wpr);

    // Center word (current row, current column).
    const uint64_t C = input[idx];

    // Neighbor words (initialize to 0 for out-of-bounds).
    uint64_t nC = 0, nL = 0, nR = 0;  // Row above
    uint64_t sC = 0, sL = 0, sR = 0;  // Row below
    uint64_t cL = 0, cR = 0;          // Same row, left/right

    if (has_top) {
        const size_t north_base = row_base - wpr;
        nC = input[north_base + col];
        if (has_left)  nL = input[north_base + (col - 1)];
        if (has_right) nR = input[north_base + (col + 1)];
    }

    if (has_bottom) {
        const size_t south_base = row_base + wpr;
        sC = input[south_base + col];
        if (has_left)  sL = input[south_base + (col - 1)];
        if (has_right) sR = input[south_base + (col + 1)];
    }

    if (has_left)  cL = input[row_base + (col - 1)];
    if (has_right) cR = input[row_base + (col + 1)];

    // Construct direction words for each row:
    // Row above: NW, N, NE
    const uint64_t upL = (nC << 1) | (nL >> 63);  // NW
    const uint64_t upC = nC;                      // N
    const uint64_t upR = (nC >> 1) | (nR << 63);  // NE

    // Row center: W, 0 (no self), E
    const uint64_t midL = (C << 1) | (cL >> 63);  // W
    const uint64_t midC = 0ull;                   // Center cell not counted
    const uint64_t midR = (C >> 1) | (cR << 63);  // E

    // Row below: SW, S, SE
    const uint64_t dnL = (sC << 1) | (sL >> 63);  // SW
    const uint64_t dnC = sC;                      // S
    const uint64_t dnR = (sC >> 1) | (sR << 63);  // SE

    // Row-wise sums using 3-input adders.
    uint64_t top_s, top_c;
    uint64_t mid_s, mid_c;
    uint64_t bot_s, bot_c;

    csa3(upL,  upC,  upR,  top_s, top_c);  // top row: NW + N + NE
    csa3(midL, midC, midR, mid_s, mid_c);  // middle row: W + 0 + E
    csa3(dnL,  dnC,  dnR,  bot_s, bot_c);  // bottom row: SW + S + SE

    // Combine the three row sums (sum bitplanes).
    uint64_t S_sum, S_carry;
    csa3(top_s, mid_s, bot_s, S_sum, S_carry);

    // Combine the three row carries (carry bitplanes).
    uint64_t C_sum, C_carry;
    csa3(top_c, mid_c, bot_c, C_sum, C_carry);

    // Compute intermediate totals:
    // T = S_carry + C_sum,  where:
    //   t0 = T % 2,  t1 = (T == 2)
    const uint64_t t0 = S_carry ^ C_sum;
    const uint64_t t1 = S_carry & C_sum;

    // Q = t1 + C_carry, where:
    //   q0 = Q % 2 (4's bit of neighbor_count)
    //   q1 = (Q == 2) (8's bit of neighbor_count)
    const uint64_t q0 = t1 ^ C_carry;
    const uint64_t q1 = t1 & C_carry;

    // Neighbor count bitplanes:
    const uint64_t n0 = S_sum;  // 1's bit
    const uint64_t n1 = t0;     // 2's bit
    const uint64_t n2 = q0;     // 4's bit
    const uint64_t n3 = q1;     // 8's bit (only non-zero when count == 8)

    // Masks for neighbors == 2 and neighbors == 3.
    const uint64_t not_n0 = ~n0;
    const uint64_t not_n2 = ~n2;
    const uint64_t not_n3 = ~n3;

    const uint64_t eq2 = not_n3 & not_n2 &  n1 & not_n0;  // count == 2
    const uint64_t eq3 = not_n3 & not_n2 &  n1 &  n0;     // count == 3

    // Game of Life transition:
    //   next_alive = (neighbor_count == 3) OR (alive & neighbor_count == 2)
    const uint64_t next = eq3 | (C & eq2);

    output[idx] = next;
}

/*
 * Host function: run one step of Conway's Game of Life on a bit-packed grid.
 *
 * Parameters:
 *   - input:  device pointer to current grid (bit-packed, 64 cells per uint64_t).
 *   - output: device pointer to destination grid for next state.
 *   - grid_dimensions: width and height of square grid in cells (power of 2).
 *
 * The input and output arrays must already be allocated with cudaMalloc.
 * This function launches the CUDA kernel but does not perform any
 * host-device synchronization; the caller is responsible for that.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t*       output,
                      int                  grid_dimensions)
{
    const int    words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const size_t total_words   =
        static_cast<size_t>(grid_dimensions) *
        static_cast<size_t>(words_per_row);

    // Choose a reasonable block size for modern data center GPUs.
    const int block_dim = 256;
    const int grid_dim =
        static_cast<int>((total_words + block_dim - 1) / block_dim);

    // Launch kernel: one thread per 64-bit word.
    game_of_life_step_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output),
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
