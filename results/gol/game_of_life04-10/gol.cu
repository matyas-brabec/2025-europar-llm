#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life
  on a bit-packed grid (1 bit per cell, 64 cells per 64-bit word).

  Design:
  - Each thread processes exactly one 64-bit word (64 horizontally adjacent cells).
  - The grid is square; width == height == grid_dimensions (power of two, >512).
  - The grid is stored row-major, with each row containing words_per_row words,
    where words_per_row = grid_dimensions / 64.
  - All out-of-bounds cells are considered dead (0). No wrap-around.

  Approach:
  - For the thread's target word, we load the center word from the current row,
    its left/right neighboring words (for cross-word horizontal neighbor bits),
    and the words from the rows above and below (including their left/right neighbors).
    This totals up to 9 64-bit loads per thread.
  - We then form eight 64-bit "neighbor bitboards" corresponding to the 8 neighbor
    directions (left/center/right for the row above, left/right for the current row,
    left/center/right for the row below). Horizontal shifts across word boundaries
    are handled by pulling in the MSB/LSB from neighboring words.
  - Using a carry-save adder (CSA) tree, we sum these eight 1-bit-wide bitboards
    to produce per-bit neighbor counts in bit-sliced form (bitplanes for 1,2,4,8).
    From these bitplanes we compute masks for neighbor count == 2 and == 3.
  - The next-state bitboard is: born (count == 3) OR (survive: current_alive AND count == 2).

  Notes:
  - This bit-sliced CSA method avoids per-bit loops and is typically faster than
    popcount-based methods when computing per-cell neighbor counts in parallel.
  - No shared or texture memory is used; global loads are coalesced.
*/

using u64 = std::uint64_t;

// Carry-Save Adder for three 64-bit bitboards.
// Produces two 64-bit outputs: sum (bitwise sum mod 2) and carry (bitwise carry)
// such that numeric sum = sum + 2*carry for each bit position.
__device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    // sum = a ^ b ^ c
    // carry bit set when at least two of {a,b,c} are 1:
    // carry = (a & b) | (a & c) | (b & c)
    // A slightly cheaper equivalent: carry = (a & b) | (c & (a ^ b))
    u64 axb = a ^ b;
    sum = axb ^ c;
    carry = (a & b) | (c & axb);
}

// Shift left by 1 bit across words: inject bit 63 of 'left' into bit 0 of result.
__device__ __forceinline__ u64 shl_with_carry(u64 x, u64 left) {
    return (x << 1) | (left >> 63);
}

// Shift right by 1 bit across words: inject bit 0 of 'right' into bit 63 of result.
__device__ __forceinline__ u64 shr_with_carry(u64 x, u64 right) {
    return (x >> 1) | (right << 63);
}

__global__ void game_of_life_kernel(const u64* __restrict__ in,
                                    u64* __restrict__ out,
                                    int height,
                                    int words_per_row,
                                    int word_shift) {
    // Map thread to 64-bit word index
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total_words = (size_t)height * (size_t)words_per_row;
    if (tid >= total_words) return;

    // Decode row and column within the word-grid
    int col = static_cast<int>(tid & (size_t)(words_per_row - 1)); // words_per_row is power of two
    int row = static_cast<int>(tid >> word_shift);

    // Boundary flags
    bool has_left  = (col > 0);
    bool has_right = (col + 1) < words_per_row;
    bool has_up    = (row > 0);
    bool has_down  = (row + 1) < height;

    // Indices for neighbor rows
    size_t idx_cur  = tid;
    size_t idx_up   = tid - (size_t)words_per_row; // valid only if has_up
    size_t idx_down = tid + (size_t)words_per_row; // valid only if has_down

    // Load center and horizontal neighbors from current row
    u64 Bc = in[idx_cur];
    u64 Bl = has_left  ? in[idx_cur - 1] : 0ull;
    u64 Br = has_right ? in[idx_cur + 1] : 0ull;

    // Load words from the row above (and its horizontal neighbors), if any
    u64 Au  = has_up ? in[idx_up] : 0ull;
    u64 AuL = (has_up && has_left)  ? in[idx_up - 1] : 0ull;
    u64 AuR = (has_up && has_right) ? in[idx_up + 1] : 0ull;

    // Load words from the row below (and its horizontal neighbors), if any
    u64 Cu  = has_down ? in[idx_down] : 0ull;
    u64 CuL = (has_down && has_left)  ? in[idx_down - 1] : 0ull;
    u64 CuR = (has_down && has_right) ? in[idx_down + 1] : 0ull;

    // Build the 8 neighbor bitboards for this word:
    // - From the row above: left, center, right
    // - From the current row: left, right (exclude center itself)
    // - From the row below: left, center, right
    u64 A_l = shl_with_carry(Au, AuL);
    u64 A_c = Au;
    u64 A_r = shr_with_carry(Au, AuR);

    u64 B_l = shl_with_carry(Bc, Bl);
    u64 B_r = shr_with_carry(Bc, Br);

    u64 C_l = shl_with_carry(Cu, CuL);
    u64 C_c = Cu;
    u64 C_r = shr_with_carry(Cu, CuR);

    // Sum the 8 neighbor bitboards using a CSA tree to obtain bit-sliced counts.
    // Group into three CSAs:
    //   g1: A_l, A_c, A_r
    //   g2: B_l, B_r, C_l
    //   g3: C_c, C_r, 0
    u64 s1, c1;
    u64 s2, c2;
    u64 s3, c3;
    csa(A_l, A_c, A_r, s1, c1);
    csa(B_l, B_r, C_l, s2, c2);
    csa(C_c, C_r, 0ull, s3, c3);

    // Sum the "sum" outputs (s1, s2, s3) -> P + 2*Q
    u64 P, Q;
    csa(s1, s2, s3, P, Q);

    // Sum the "carry" outputs (c1, c2, c3) -> P2 + 2*Q2
    u64 P2, Q2;
    csa(c1, c2, c3, P2, Q2);

    // Total neighbor count per bit = P + 2*(Q + P2) + 4*Q2
    // Bitplanes:
    //   b0 (ones)  = P
    //   b1 (twos)  = (Q xor P2)
    //   b2 (fours) = (Q & P2) xor Q2
    //   b3 (eights)= (Q & P2) & Q2
    u64 b0 = P;
    u64 b1 = Q ^ P2;
    u64 t4 = Q & P2;      // intermediate for 4's place from adding Q + P2
    u64 b2 = t4 ^ Q2;
    u64 b3 = t4 & Q2;

    // Masks for neighbor count == 3 and == 2.
    // 3 (0011): b0=1, b1=1, b2=0, b3=0
    // 2 (0010): b0=0, b1=1, b2=0, b3=0
    u64 not_b2 = ~b2;
    u64 not_b3 = ~b3;
    u64 eq3 = b0 & b1 & not_b2 & not_b3;
    u64 eq2 = (~b0) & b1 & not_b2 & not_b3;

    // Apply Game of Life rules:
    // next = born | survive = (neighbors == 3) | (alive & neighbors == 2)
    u64 next = eq3 | (Bc & eq2);

    out[idx_cur] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // words_per_row is grid_dimensions / 64 (power of two by problem statement)
    int words_per_row = grid_dimensions >> 6;

    // Precompute log2(words_per_row) on host (since it's a power of two)
    int word_shift = 0;
    {
        int tmp = words_per_row;
        while ((1 << word_shift) != tmp) ++word_shift;
    }

    // Total number of 64-bit words
    size_t total_words = static_cast<size_t>(grid_dimensions) * static_cast<size_t>(words_per_row);

    // Choose a block size that generally provides high occupancy
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    // Launch kernel (no synchronization here; caller is responsible)
    game_of_life_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row, word_shift);
}