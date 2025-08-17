#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Data layout and threading:
  - The grid is a square of size N x N, where N is a power of 2 and >= 512.
  - Each row is bit-packed into 64-bit words (uint64_t): 1 bit per cell (LSB = leftmost cell in the 64-cell chunk).
  - A thread processes exactly one 64-bit word (i.e., 64 cells). This avoids atomics and maintains coalesced accesses.

  Neighbor computation:
  - For a given word W0 at row r and column-word c, the eight neighbors of each bit are in:
      - Row r-1: columns c-1 (AL), c (A0), c+1 (AR)
      - Row r  : columns c-1 (WL) [left neighbor only], c+1 (WR) [right neighbor only]
      - Row r+1: columns c-1 (BL), c (B0), c+1 (BR)
  - To align neighbor contributions to the current word's bit positions:
      - left-shift (<< 1) aligns "left neighbors" (col-1) to the current bit position b (pulling in bit63 from the left word).
      - right-shift (>> 1) aligns "right neighbors" (col+1) to the current bit position b (pulling in bit0 from the right word).
      - center (no shift) aligns vertical neighbors directly.
  - Boundary handling: Words outside the grid are treated as zero. For bit 0 and bit 63 within a word, we incorporate spill-in bits
    from the adjacent words (AL/BL/WL for bit 0, and AR/BR/WR for bit 63). At the outer grid boundaries, these spill-ins are zero.

  Bit-parallel neighbor count via carry-save adders (CSA):
  - We compute the per-bit neighbor sum (range 0..8) for 64 cells in parallel using a CSA tree:
      Given eight 64-bit masks x0..x7, representing the eight neighbor contributions:
        CSA returns (sum, carry) for three inputs, where:
          sum = a ^ b ^ c
          carry = majority(a,b,c) = (a & b) | (a & c) | (b & c)
      Steps:
        (sA,cA) = CSA(x0,x1,x2)
        (sB,cB) = CSA(x3,x4,x5)
        (sC,cC) = CSA(x6,x7,0)

        (sD,cD) = CSA(sA,sB,sC)
        (sE,cE) = CSA(cA,cB,cC)

      The final 3-bit neighbor count per bit-lane is:
        n0 (LSB, 1's place) = sD
        n1 (2's place)      = cD ^ sE
        n2 (4's place)      = cE ^ (cD & sE)
      The 8's place (n3) is implicitly:
        n3 = cE & (cD & sE)
      However, we do not need n3 to determine equals-2 or equals-3.

  Next-state rule (Conway's Life):
    next = (neighbors == 3) OR (alive & (neighbors == 2))
    Using the bit-planes:
      eq3 = (~n2) & n1 & n0
      eq2 = (~n2) & n1 & (~n0)
      next_word = eq3 | (current_word & eq2)
*/

static __device__ __forceinline__ void csa(const std::uint64_t a,
                                           const std::uint64_t b,
                                           const std::uint64_t c,
                                           std::uint64_t &sum,
                                           std::uint64_t &carry)
{
    // Efficient carry-save adder for bitwise parallel addition of three inputs over 64 lanes.
    // sum = a ^ b ^ c; carry = majority(a,b,c).
    const std::uint64_t u = a ^ b;
    sum = u ^ c;
    carry = (a & b) | (u & c);
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int width_words,
                                    int height,
                                    int width_words_log2)
{
    const std::size_t total_words = static_cast<std::size_t>(width_words) * static_cast<std::size_t>(height);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    const std::size_t ww = static_cast<std::size_t>(width_words);
    const std::size_t ww_mask = ww - 1;

    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += stride)
    {
        // Compute row (i) and column-word (j) indices exploiting that width_words is power-of-two.
        const std::size_t j = idx & ww_mask;
        const std::size_t i = idx >> width_words_log2;

        // Load center word (current row).
        const std::uint64_t W0 = in[idx];

        // Determine boundaries.
        const bool has_left_word   = (j > 0);
        const bool has_right_word  = (j + 1 < ww);
        const bool has_above_row   = (i > 0);
        const bool has_below_row   = (i + 1 < static_cast<std::size_t>(height));

        // Neighbor words in the same row (only used for left/right horizontal neighbors).
        const std::uint64_t WL = has_left_word  ? in[idx - 1] : 0ULL;
        const std::uint64_t WR = has_right_word ? in[idx + 1] : 0ULL;

        // Neighbor words in the above row: AL, A0, AR
        std::uint64_t A0 = 0ULL, AL = 0ULL, AR = 0ULL;
        if (has_above_row) {
            const std::size_t aidx = idx - ww;
            A0 = in[aidx];
            AL = has_left_word  ? in[aidx - 1] : 0ULL;
            AR = has_right_word ? in[aidx + 1] : 0ULL;
        }

        // Neighbor words in the below row: BL, B0, BR
        std::uint64_t B0 = 0ULL, BL = 0ULL, BR = 0ULL;
        if (has_below_row) {
            const std::size_t bidx = idx + ww;
            B0 = in[bidx];
            BL = has_left_word  ? in[bidx - 1] : 0ULL;
            BR = has_right_word ? in[bidx + 1] : 0ULL;
        }

        // Horizontally align neighbor contributions to the current word's bit positions.
        // Note:
        // - For left shifts, spill-in the MSB (bit63) from the left neighbor word.
        // - For right shifts, spill-in the LSB (bit0) from the right neighbor word.
        // - Center terms (A0, B0) are unshifted; for the current row we exclude the center (W0).
        const std::uint64_t aL = (A0 << 1) | (AL >> 63);
        const std::uint64_t aC = A0;
        const std::uint64_t aR = (A0 >> 1) | (AR << 63);

        const std::uint64_t wL = (W0 << 1) | (WL >> 63);
        const std::uint64_t wR = (W0 >> 1) | (WR << 63);

        const std::uint64_t bL = (B0 << 1) | (BL >> 63);
        const std::uint64_t bC = B0;
        const std::uint64_t bR = (B0 >> 1) | (BR << 63);

        // Carry-save adder tree to sum the eight neighbor bit-vectors:
        // Group triples via CSA and then combine sums and carries to derive the 3-bit count (n2 n1 n0).
        std::uint64_t sA, cA; csa(aL, aC, aR, sA, cA);
        std::uint64_t sB, cB; csa(wL, wR, bL, sB, cB);
        std::uint64_t sC, cC; csa(bC, bR, 0ULL, sC, cC);

        std::uint64_t sD, cD; csa(sA, sB, sC, sD, cD); // sD: LSB of neighbor count
        std::uint64_t sE, cE; csa(cA, cB, cC, sE, cE);

        // Neighbor count bit-planes:
        // n0 = sD
        // n1 = cD ^ sE
        // n2 = cE ^ (cD & sE)
        const std::uint64_t n0 = sD;
        const std::uint64_t n1 = cD ^ sE;
        const std::uint64_t n2 = cE ^ (cD & sE);

        // Evaluate Life rule:
        // eq3 = (~n2) & n1 & n0  -> exactly 3 neighbors
        // eq2 = (~n2) & n1 & ~n0 -> exactly 2 neighbors
        const std::uint64_t not_n2 = ~n2;
        const std::uint64_t eq3 = not_n2 & n1 & n0;
        const std::uint64_t eq2 = not_n2 & n1 & (~n0);

        // Next state: alive if eq3 OR (currently alive AND eq2).
        const std::uint64_t next = eq3 | (W0 & eq2);

        out[idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is a power of two and >= 512 by problem statement.
    // width_words = grid_dimensions / 64 is also a power of two.
    const int width_words = grid_dimensions >> 6; // divide by 64
    const int height = grid_dimensions;

    // Compute log2(width_words) for fast div/mod in the kernel using bit ops.
    int ww_log2 = 0;
    {
        int w = width_words;
        // Since width_words is a power of two, count trailing zeros yields log2.
        // Portable fallback loop (fast since log2(width_words) <= 26 for practical sizes).
        while ((1 << ww_log2) != w) { ++ww_log2; }
    }

    const std::size_t total_words = static_cast<std::size_t>(width_words) * static_cast<std::size_t>(height);

    // Configure a 1D launch; each thread processes multiple words via grid-stride loop.
    // 256 threads per block is a good balance for modern NVIDIA GPUs (A100/H100).
    const int threads_per_block = 256;
    int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);
    if (blocks <= 0) blocks = 1;

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, width_words, height, ww_log2);
}