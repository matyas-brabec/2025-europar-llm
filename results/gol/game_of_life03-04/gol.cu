#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of a single Conway's Game of Life step for a square grid.

  - Grid is bit-packed: each 64-bit word encodes 64 consecutive cells within a row (bit 0 = cell 0 of the word).
  - Each CUDA thread processes exactly one 64-bit word (64 cells).
  - Outside-of-grid cells are treated as dead (0) — no wrap-around.

  Algorithm overview:
  - For each word, load up to 9 words from global memory (3x3 neighborhood at the word level: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right).
  - Construct eight 64-bit bitboards of neighbors (N, NE, E, SE, S, SW, W, NW) aligned to the current word's bit positions.
      - Horizontal neighbors use cross-word shifts with carry-in/out across 64-bit boundaries:
          W  = (center << 1) | (left >> 63)
          E  = (center >> 1) | (right << 63)
        Special cases at row edges: missing words are treated as zero.
  - Compute neighbor counts per bit (0..8) using a carry-save adder (CSA) tree on the eight bitboards.
      - The CSA tree yields three 64-bit masks representing the neighbor count bits:
          bit0: least significant bit of neighbor count (1's place)
          bit1: second bit (2's place)
          bit2: third bit (4's place)
        We do not need higher bits to apply Life rules (only equality tests for 2 or 3).
  - Apply Life rules in bit-parallel form:
      eq2 = (~bit2) & bit1 & (~bit0)
      eq3 = (~bit2) & bit1 & ( bit0)
      next = eq3 | (eq2 & current)
  - Write the result to the output buffer.

  Performance notes:
  - Global memory access is coalesced when threads process contiguous words.
  - No shared or texture memory is used as it was found unnecessary for this bit-packed approach.
  - Uses __ldg() for read-only cache on supported architectures; pointers are marked __restrict__ for better compiler optimization.
*/

#ifndef __CUDA_ARCH__
#define __ldg(p) (*(p))
#endif

// Carry-Save Adder: sums three 64-bit bitboards (per-bit), producing
// sum (bit0 of the sum) and carry (bit1 of the sum) without cross-bit carries.
// For inputs a,b,c (each bit is 0 or 1), the outputs are:
//   sum   = a ^ b ^ c
//   carry = majority(a,b,c) but encoded as "carry" bits of weight 2.
static __device__ __forceinline__ void csa(const std::uint64_t a,
                                           const std::uint64_t b,
                                           const std::uint64_t c,
                                           std::uint64_t &sum,
                                           std::uint64_t &carry)
{
    const std::uint64_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (u & c);
}

// Horizontal neighbor shifts across 64-bit word boundaries.
//
// For a word encoding 64 cells left-to-right as bits 0..63:
// - West neighbor of bit j is original bit j-1. For j=0, it comes from the MSB (bit63) of the left word.
// - East neighbor of bit j is original bit j+1. For j=63, it comes from the LSB (bit0) of the right word.
static __device__ __forceinline__ std::uint64_t shift_west(const std::uint64_t center, const std::uint64_t left)
{
    return (center << 1) | (left >> 63);
}
static __device__ __forceinline__ std::uint64_t shift_east(const std::uint64_t center, const std::uint64_t right)
{
    return (center >> 1) | (right << 63);
}

static __global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                       std::uint64_t* __restrict__ output,
                                       int grid_dim_cells,
                                       int words_per_row,
                                       int lg_words_per_row)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_words = static_cast<std::size_t>(grid_dim_cells) * static_cast<std::size_t>(words_per_row);
    if (idx >= total_words) return;

    // Compute row/column in the word grid. words_per_row is a power of two.
    const std::size_t row = idx >> lg_words_per_row;
    const std::size_t col = idx & static_cast<std::size_t>(words_per_row - 1);

    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < static_cast<std::size_t>(words_per_row));
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < static_cast<std::size_t>(grid_dim_cells));

    const std::size_t up_idx   = idx - static_cast<std::size_t>(words_per_row);
    const std::size_t down_idx = idx + static_cast<std::size_t>(words_per_row);

    // Load the 3x3 neighborhood at the word level, zeroing out-of-grid neighbors.
    const std::uint64_t mC = __ldg(input + idx);
    const std::uint64_t mL = has_left  ? __ldg(input + idx - 1) : 0ull;
    const std::uint64_t mR = has_right ? __ldg(input + idx + 1) : 0ull;

    const std::uint64_t tC = has_up    ? __ldg(input + up_idx) : 0ull;
    const std::uint64_t tL = (has_up && has_left)  ? __ldg(input + up_idx - 1) : 0ull;
    const std::uint64_t tR = (has_up && has_right) ? __ldg(input + up_idx + 1) : 0ull;

    const std::uint64_t bC = has_down  ? __ldg(input + down_idx) : 0ull;
    const std::uint64_t bL = (has_down && has_left)  ? __ldg(input + down_idx - 1) : 0ull;
    const std::uint64_t bR = (has_down && has_right) ? __ldg(input + down_idx + 1) : 0ull;

    // Construct the eight neighbor bitboards aligned to the current word's bit positions.
    // Horizontal neighbors within each of the three rows (top/middle/bottom) require carry across 64-bit words.
    const std::uint64_t N  = tC;
    const std::uint64_t S  = bC;
    const std::uint64_t W  = shift_west(mC, mL);
    const std::uint64_t E  = shift_east(mC, mR);
    const std::uint64_t NW = shift_west(tC, tL);
    const std::uint64_t NE = shift_east(tC, tR);
    const std::uint64_t SW = shift_west(bC, bL);
    const std::uint64_t SE = shift_east(bC, bR);

    // Carry-save adder tree to sum eight 1-bit inputs per bit position.
    // First level: group into triplets (last group uses a zero).
    std::uint64_t sA, cA; csa(NW, N,  NE, sA, cA);   // A: NW + N + NE
    std::uint64_t sB, cB; csa(W,  E,  SW, sB, cB);   // B: W  + E + SW
    std::uint64_t sC, cC; csa(S,  SE, 0ull, sC, cC); // C: S  + SE + 0

    // Second level: sum the "sum" outputs (weight 1), producing sD (bit0) and cD (weight 2).
    std::uint64_t sD, cD; csa(sA, sB, sC, sD, cD);   // sD -> bit0 of neighbor count

    // Third level: combine all "carry" (weight 2) outputs.
    std::uint64_t sE, cE; csa(cA, cB, cC, sE, cE);   // sE: weight 2, cE: weight 4
    std::uint64_t sF, cF; csa(cD, sE, 0ull, sF, cF); // sF -> bit1 (weight 2), cF: weight 4

    // Fourth level: combine all weight-4 carries to get bit2 (weight 4).
    std::uint64_t sG, cG; csa(cE, cF, 0ull, sG, cG); // sG -> bit2 (weight 4), cG: weight 8

    // Neighbor count bits (per bit position across the 64-bit word):
    const std::uint64_t bit0 = sD; // 1's place
    const std::uint64_t bit1 = sF; // 2's place
    const std::uint64_t bit2 = sG; // 4's place
    // const std::uint64_t bit3 = cG; // 8's place (not needed for rules)

    // Apply life rules:
    // eq2: exactly 2 neighbors => (~bit2) & bit1 & (~bit0)
    // eq3: exactly 3 neighbors => (~bit2) & bit1 & ( bit0)
    // Next state: eq3 (births and survives) OR (eq2 & currently alive)
    const std::uint64_t nb2 = (~bit2) & bit1 & (~bit0);
    const std::uint64_t nb3 = (~bit2) & bit1 & ( bit0);
    const std::uint64_t next = nb3 | (nb2 & mC);

    output[idx] = next;
}

static inline int ilog2_pow2_int(int x)
{
    // Computes log2(x) for x > 0 and a power-of-two.
    int l = 0;
    while (x > 1) { x >>= 1; ++l; }
    return l;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is number of cells per side (power of two, > 512).
    // words_per_row is also a power of two as grid_dimensions is divisible by 64.
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    const int lg_words_per_row = ilog2_pow2_int(words_per_row);

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((total_words + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gol_step_kernel<<<grid_size, BLOCK_SIZE>>>(input, output, grid_dimensions, words_per_row, lg_words_per_row);
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
