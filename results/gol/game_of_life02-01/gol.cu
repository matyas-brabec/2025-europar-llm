#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Representation:
  - The grid is square with side length N (power of two, N >= 512).
  - Each row is bit-packed into 64-bit words; one bit per cell (1 = alive, 0 = dead).
  - Each 64-bit word encodes 64 consecutive cells within the same row.
  - All cells outside the finite grid are considered dead (zero-padding at borders).

  Parallelization strategy:
  - Each CUDA thread processes one 64-bit word (i.e., 64 cells within a row).
  - The kernel reads up to 9 words (3x3 neighborhood in word space): for above, current, and below rows, and left/center/right words in each of those rows.
  - Horizontal neighbors within a row are handled by shifting with carry from adjacent words (to correctly span word boundaries without wrapping).
  - Neighbor counts are computed per-bit using bit-sliced arithmetic (XOR/AND/OR), which keeps 64 parallel "lanes" independent and avoids cross-bit carries.
  - We never use shared or texture memory per the requirement; global loads are coalesced and benefit from the GPU caches.

  Life rule:
  - A cell becomes alive in the next generation if it has exactly 3 neighbors, or if it is already alive and has exactly 2 neighbors.
  - We compute the 8-neighbor count (excluding the center cell) and derive eq2 (count == 2) and eq3 (count == 3) bitmasks, then combine with the current cell bitmask.

  Notes on efficiency:
  - Division/modulo to compute (row, col) from a linear index is avoided using bit operations because words_per_row is a power of two: row = idx >> words_shift; col = idx & (words_per_row - 1).
  - The kernel uses only integer bitwise ops and a small fixed number of global memory reads.
*/

static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t center, std::uint64_t left_word) {
    // Shift left by 1 within the virtual infinite row, with carry-in from the previous word's MSB.
    return (center << 1) | (left_word >> 63);
}

static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t center, std::uint64_t right_word) {
    // Shift right by 1 within the virtual infinite row, with carry-in from the next word's LSB.
    return (center >> 1) | (right_word << 63);
}

static __device__ __forceinline__ void sum2_bits(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
    // Bit-sliced addition of two 1-bit bitboards: lo = (a + b) LSB, hi = (a + b) carry.
    lo = a ^ b;
    hi = a & b;
}

static __device__ __forceinline__ void sum3_bits(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t& lo, std::uint64_t& hi) {
    // Bit-sliced addition of three 1-bit bitboards (values 0..3), result in two planes: lo (LSB), hi (MSB).
    // Equivalent to a + b + c with no carries between bit lanes.
    std::uint64_t t = a ^ b;
    std::uint64_t ab = a & b;
    lo = t ^ c;
    std::uint64_t tc = t & c;
    hi = ab | tc;
}

static __device__ __forceinline__ void add_2x2_to_3bit(std::uint64_t a0, std::uint64_t a1,
                                                      std::uint64_t b0, std::uint64_t b1,
                                                      std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2) {
    // Add two 2-bit numbers (a1:a0) + (b1:b0) per bit lane, producing a 3-bit result (s2:s1:s0).
    std::uint64_t t0 = a0 ^ b0;
    std::uint64_t c0 = a0 & b0;  // carry from bit 0 to bit 1
    std::uint64_t t1 = a1 ^ b1;
    std::uint64_t c1 = a1 & b1;  // carry from bit 1 to bit 2
    s0 = t0;
    std::uint64_t s1_nc = t1 ^ c0;
    std::uint64_t c1b = (t1 & c0) | c1;
    s1 = s1_nc;
    s2 = c1b;
}

static __device__ __forceinline__ void add_3bit_and_2bit(std::uint64_t a0, std::uint64_t a1, std::uint64_t a2,
                                                         std::uint64_t b0, std::uint64_t b1,
                                                         std::uint64_t& r0, std::uint64_t& r1, std::uint64_t& r2, std::uint64_t& r3) {
    // Add a 3-bit number (a2:a1:a0) and a 2-bit number (b1:b0), producing a 4-bit result (r3:r2:r1:r0).
    // Ripple addition within each bit lane.
    std::uint64_t s0 = a0 ^ b0;
    std::uint64_t c0 = a0 & b0;

    std::uint64_t t1 = a1 ^ b1;
    std::uint64_t c1a = a1 & b1;
    std::uint64_t s1 = t1 ^ c0;
    std::uint64_t c1b = t1 & c0;
    std::uint64_t c1 = c1a | c1b;

    std::uint64_t s2 = a2 ^ c1;
    std::uint64_t c2 = a2 & c1;

    r0 = s0;
    r1 = s1;
    r2 = s2;
    r3 = c2;
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int n, int words_per_row, int words_shift, std::size_t total_words)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    // Compute (row, col) from linear index using power-of-two properties.
    std::size_t row = idx >> words_shift;
    int col = static_cast<int>(idx & static_cast<std::size_t>(words_per_row - 1));

    const int last_col = words_per_row - 1;
    const int last_row = n - 1;

    // Load the 3x3 word neighborhood (UL, UC, UR / ML, MC, MR / DL, DC, DR) with zero-padding at borders.
    // Above row
    std::uint64_t UL = (row > 0 && col > 0)                ? in[(row - 1) * (std::size_t)words_per_row + (col - 1)] : 0ULL;
    std::uint64_t UC = (row > 0)                           ? in[(row - 1) * (std::size_t)words_per_row + col]       : 0ULL;
    std::uint64_t UR = (row > 0 && col < last_col)         ? in[(row - 1) * (std::size_t)words_per_row + (col + 1)] : 0ULL;

    // Middle row
    std::uint64_t ML = (col > 0)                           ? in[row * (std::size_t)words_per_row + (col - 1)]       : 0ULL;
    std::uint64_t MC =                                       in[row * (std::size_t)words_per_row + col];
    std::uint64_t MR = (col < last_col)                    ? in[row * (std::size_t)words_per_row + (col + 1)]       : 0ULL;

    // Below row
    std::uint64_t DL = (row < static_cast<std::size_t>(last_row) && col > 0)
                                                         ? in[(row + 1) * (std::size_t)words_per_row + (col - 1)] : 0ULL;
    std::uint64_t DC = (row < static_cast<std::size_t>(last_row))
                                                         ? in[(row + 1) * (std::size_t)words_per_row + col]       : 0ULL;
    std::uint64_t DR = (row < static_cast<std::size_t>(last_row) && col < last_col)
                                                         ? in[(row + 1) * (std::size_t)words_per_row + (col + 1)] : 0ULL;

    // Compute horizontally shifted neighbor bitboards (aligned so that each bit position corresponds to the center cell).
    // Top row: three neighbors above (NW, N, NE)
    std::uint64_t U_west  = shl1_with_carry(UC, UL);
    std::uint64_t U_center= UC;
    std::uint64_t U_east  = shr1_with_carry(UC, UR);

    // Middle row: only left and right neighbors (exclude center)
    std::uint64_t M_west  = shl1_with_carry(MC, ML);
    std::uint64_t M_east  = shr1_with_carry(MC, MR);

    // Bottom row: three neighbors below (SW, S, SE)
    std::uint64_t D_west  = shl1_with_carry(DC, DL);
    std::uint64_t D_center= DC;
    std::uint64_t D_east  = shr1_with_carry(DC, DR);

    // Sum horizontally within each row.
    // U_sum = U_west + U_center + U_east  (2-bit result per lane)
    std::uint64_t U_lo, U_hi;
    sum3_bits(U_west, U_center, U_east, U_lo, U_hi);

    // D_sum = D_west + D_center + D_east (2-bit result per lane)
    std::uint64_t D_lo, D_hi;
    sum3_bits(D_west, D_center, D_east, D_lo, D_hi);

    // LR_sum = M_west + M_east (2-bit result per lane), excludes center cell.
    std::uint64_t LR_lo, LR_hi;
    sum2_bits(M_west, M_east, LR_lo, LR_hi);

    // Vertical accumulation: (U_lo,U_hi) + (D_lo,D_hi) -> 3-bit partial sum
    std::uint64_t S0, S1, S2; // 3-bit partial sum planes
    add_2x2_to_3bit(U_lo, U_hi, D_lo, D_hi, S0, S1, S2);

    // Add LR_sum (2-bit) to the 3-bit partial sum -> full 4-bit neighbor count (0..8)
    std::uint64_t R0, R1, R2, R3;
    add_3bit_and_2bit(S0, S1, S2, LR_lo, LR_hi, R0, R1, R2, R3);

    // Compute eq2 (neighbor count == 2) and eq3 (neighbor count == 3) per bit lane.
    std::uint64_t notR3 = ~R3;
    std::uint64_t notR2 = ~R2;
    std::uint64_t notR0 = ~R0;
    std::uint64_t eq2 = notR3 & notR2 & R1 & notR0; // 0b0010
    std::uint64_t eq3 = notR3 & notR2 & R1 & R0;    // 0b0011

    // Next state: alive if (neighbors == 3) or (alive and neighbors == 2).
    std::uint64_t next = eq3 | (MC & eq2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Precompute launch parameters and derived constants.
    // words_per_row is guaranteed to be a power of two since grid_dimensions is a power of two (and divisible by 64).
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    int words_shift = 0; // log2(words_per_row)
    {
        // Compute log2(words_per_row) cheaply on host (once per call).
        int w = words_per_row;
        while ((1 << words_shift) < w) ++words_shift;
        // At this point, (1 << words_shift) == words_per_row for power-of-two w.
    }

    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Choose a launch configuration tuned for large grids on modern NVIDIA data center GPUs.
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = static_cast<int>((total_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    game_of_life_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input, output, grid_dimensions, words_per_row, words_shift, total_words
    );
    // Any synchronization or error checking is intentionally omitted; the caller handles synchronization.
}