#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance, branch-minimized CUDA implementation of a single step of Conway's Game of Life
  on a bit-packed grid (1 bit per cell, 64 cells per 64-bit word). Each CUDA thread processes one
  64-bit word (i.e., 64 cells in a single row), computing the next generation using only global memory
  and register-resident bitwise logic. No shared/texture memory is used, as it adds unnecessary complexity
  without improving performance for this particular workload on modern data center GPUs.

  Coordinate system and packing:
  - The grid is N x N (N is a power of two > 512). The grid is bit-packed by rows.
  - words_per_row = N / 64. Each uint64_t encodes 64 consecutive cells in the same row.
  - Bit 0 (LSB) of a word corresponds to the leftmost cell in that 64-cell segment. Bit 63 is the rightmost.
  - All cells outside the grid are considered dead (zero-padding at the boundaries).

  Neighborhood handling:
  - To compute the 8-neighbor count for each bit, we use three rows: North (row-1), Current (row), South (row+1).
  - Horizontal neighbors are handled via 1-bit shifts with cross-word carry from adjacent words.
    For example, shifting a row-left (<< 1) fills bit 0 with bit 63 of the left neighbor word; shifting right (>> 1)
    fills bit 63 with bit 0 of the right neighbor word. At row boundaries (first/last word), these carries are zero.

  Bit-parallel neighbor counting:
  - For each of the three rows, we form horizontal masks:
      RowAbove: left (TL), center (TC), right (TR)
      RowSame:  left (ML), right (MR)            [center is excluded to not count the cell itself]
      RowBelow: left (BL), center (BC), right (BR)
  - Rowwise sums are computed using bitwise full-adders:
      For three contributions (L,C,R): 
        lo = L ^ C ^ R
        hi = (L & C) | (L & R) | (C & R)     // represents '2' in the count for that row
      For two contributions (L,R):
        lo = L ^ R
        hi = L & R                           // represents '2' in the count for that row
  - Vertical accumulation:
      Let N_lo/N_hi be the lo/hi for the north row, M_lo/M_hi for the same row (excluding center), S_lo/S_hi for south.
      Total neighbor count per bit is:
        count = (N_lo + M_lo + S_lo) + 2 * (N_hi + M_hi + S_hi + carry_lo)
      where carry_lo is the carry (i.e., '2') from adding the three 'lo' terms.
  - We compute the lower 4 bits of 'count' without per-bit cross-carries using boolean logic:
      c0 = (N_lo ^ M_lo ^ S_lo)                                // bit 0 (1's place)
      carry_lo = (N_lo & M_lo) | (N_lo & S_lo) | (M_lo & S_lo) // carry from LO sum, weight=2
      Define A=N_hi, B=M_hi, C=S_hi, D=carry_lo (each is weight=2)
      For X = A + B + C + D (all weight-2 contributors, but we compute it as a count of 0..4):
        s1 = A ^ B ^ C
        c1 = (A & B) | (A & C) | (B & C)     // at least two among (A,B,C)
        s2 = s1 ^ D
        t2 = s1 & D
        X0 = s2                               // LSB of X  (X bit0)
        X1 = c1 ^ t2                          // second bit of X (X bit1)
        X2 = c1 & t2                          // third bit of X (X bit2) -> only set when X==4
      Since count = c0 + 2*X, the bits of 'count' are:
        bit0 = c0
        bit1 = X0
        bit2 = X1
        bit3 = X2
  - Game of Life rules:
      - Alive survives if neighbor count is 2 or 3.
      - Dead becomes alive if neighbor count is 3.
    Using the bitplanes:
      is_ge4 = bit2 | bit3
      is_two = (bit1 & ~bit0) & ~is_ge4
      is_three = (bit1 &  bit0) & ~is_ge4
      next = (alive & (is_two | is_three)) | (~alive & is_three)
    This simplifies to:
      low_mask = ~(bit2 | bit3)        // neighbor count < 4
      survivors = alive & bit1 & low_mask
      births    = (~alive) & bit1 & c0 & low_mask
      next      = survivors | births
*/

static __device__ __forceinline__ void sum3_u64(uint64_t a, uint64_t b, uint64_t c, uint64_t &lo, uint64_t &hi) {
    // Sum three 1-bit-per-position numbers (a + b + c) yielding:
    // lo: least significant bit of the sum (mod 2)
    // hi: carry bit (2's place), set when at least two inputs are 1 for that bit position.
    lo = a ^ b ^ c;
    hi = (a & b) | (a & c) | (b & c);
}

static __device__ __forceinline__ uint64_t shift_left_with_carry(uint64_t x, uint64_t left_word) {
    // Shift left by 1, injecting bit 63 of left_word into bit 0.
    // If there is no valid left_word, pass left_word = 0 to zero the carry.
    return (x << 1) | (left_word >> 63);
}

static __device__ __forceinline__ uint64_t shift_right_with_carry(uint64_t x, uint64_t right_word) {
    // Shift right by 1, injecting bit 0 of right_word into bit 63.
    // If there is no valid right_word, pass right_word = 0 to zero the carry.
    return (x >> 1) | ((right_word & 1ULL) << 63);
}

__global__ void life_step_kernel(const std::uint64_t* __restrict__ input,
                                 std::uint64_t* __restrict__ output,
                                 int grid_dim,          // N
                                 int words_per_row) {   // N / 64
    const std::size_t total_words = static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    // Grid-stride loop to cover arbitrary grid sizes efficiently.
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_words; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / words_per_row);
        const int col = static_cast<int>(idx % words_per_row);

        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < words_per_row);
        const bool has_north = (row > 0);
        const bool has_south = (row + 1 < grid_dim);

        // Load center word for current row
        const std::uint64_t C = input[idx];

        // Load left and right neighbor words for current row (0 at boundaries)
        const std::uint64_t CL = has_left  ? input[idx - 1] : 0ULL;
        const std::uint64_t CR = has_right ? input[idx + 1] : 0ULL;

        // Load north row words (center, left, right). 0 if outside the grid.
        const std::uint64_t N  = has_north ? input[idx - words_per_row] : 0ULL;
        const std::uint64_t NL = (has_north && has_left)  ? input[idx - words_per_row - 1] : 0ULL;
        const std::uint64_t NR = (has_north && has_right) ? input[idx - words_per_row + 1] : 0ULL;

        // Load south row words (center, left, right). 0 if outside the grid.
        const std::uint64_t S  = has_south ? input[idx + words_per_row] : 0ULL;
        const std::uint64_t SL = (has_south && has_left)  ? input[idx + words_per_row - 1] : 0ULL;
        const std::uint64_t SR = (has_south && has_right) ? input[idx + words_per_row + 1] : 0ULL;

        // Horizontal neighbor masks with cross-word carry handling.
        // North row (top-left, top-center, top-right)
        const std::uint64_t TL = shift_left_with_carry(N, NL);
        const std::uint64_t TC = N;
        const std::uint64_t TR = shift_right_with_carry(N, NR);

        // Middle row (left and right only, center excluded)
        const std::uint64_t ML = shift_left_with_carry(C, CL);
        const std::uint64_t MR = shift_right_with_carry(C, CR);

        // South row (bottom-left, bottom-center, bottom-right)
        const std::uint64_t BL = shift_left_with_carry(S, SL);
        const std::uint64_t BC = S;
        const std::uint64_t BR = shift_right_with_carry(S, SR);

        // Row-wise 3-input and 2-input additions (bitwise, per-bit full adders).
        std::uint64_t N_lo, N_hi;
        sum3_u64(TL, TC, TR, N_lo, N_hi);

        const std::uint64_t M_lo = ML ^ MR;
        const std::uint64_t M_hi = ML & MR;

        std::uint64_t S_lo, S_hi;
        sum3_u64(BL, BC, BR, S_lo, S_hi);

        // Sum of the three row 'lo' terms => c0 and carry_lo (weight=2).
        const std::uint64_t c0 = N_lo ^ M_lo ^ S_lo;
        const std::uint64_t carry_lo = (N_lo & M_lo) | (N_lo & S_lo) | (M_lo & S_lo);

        // Accumulate all weight-2 contributors: A=N_hi, B=M_hi, C=S_hi, D=carry_lo.
        // Compute X = A + B + C + D in binary (per bit position) without cross-bit carries:
        const std::uint64_t A = N_hi;
        const std::uint64_t B = M_hi;
        const std::uint64_t Cw = S_hi; // rename to avoid shadowing 'C' (center word)
        const std::uint64_t D = carry_lo;

        const std::uint64_t s1 = A ^ B ^ Cw;
        const std::uint64_t c1 = (A & B) | (A & Cw) | (B & Cw);

        const std::uint64_t s2 = s1 ^ D;  // X0 (LSB of X)
        const std::uint64_t t2 = s1 & D;  // carry from adding D to s1

        // Bits of total neighbor count:
        const std::uint64_t bit0 = c0;            // 1's
        const std::uint64_t bit1 = s2;            // 2's (from 2 * X)
        const std::uint64_t bit2 = c1 ^ t2;       // 4's
        const std::uint64_t bit3 = c1 & t2;       // 8's

        // Apply Game of Life rules:
        // - survive if alive and neighbors == 2 or 3
        // - born if dead and neighbors == 3
        const std::uint64_t alive = C;
        const std::uint64_t ge4_mask = bit2 | bit3;         // count >= 4
        const std::uint64_t low_mask = ~ge4_mask;           // count < 4
        const std::uint64_t survivors = alive & bit1 & low_mask;          // neighbors 2 or 3
        const std::uint64_t births    = (~alive) & bit1 & bit0 & low_mask; // neighbors == 3
        const std::uint64_t next      = survivors | births;

        // Store the result
        output[idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions (N) is guaranteed to be a power of two > 512.
    // words_per_row must be N / 64 (since each 64-bit word encodes 64 cells).
    const int words_per_row = grid_dimensions >> 6; // N / 64

    // Total number of 64-bit words in the grid
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Launch configuration:
    // - Use a reasonable block size that provides good occupancy on modern GPUs.
    // - Use a grid-stride loop in the kernel, so we can cap grid size to avoid over-large grids.
    const int block_size = 256;
    // Cap grid size to a large, conventional limit; kernel uses grid-stride to cover the rest.
    int max_blocks = 65535;
    int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);
    if (grid_size > max_blocks) grid_size = max_blocks;
    if (grid_size < 1) grid_size = 1;

    life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row);
}