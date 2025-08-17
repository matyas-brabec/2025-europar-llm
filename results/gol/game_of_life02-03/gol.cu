#include <cuda_runtime.h>
#include <cstdint>

// This implementation computes one step of Conway's Game of Life on a bit-packed grid.
// Each 64-bit word encodes 64 cells from the same row (LSB is the left-most cell of the word).
// The kernel is optimized to:
// - Keep memory access simple and coalesced (no shared/texture memory per instructions).
// - Use bit-parallel arithmetic (carry-save style) to compute neighbor counts for all 64 cells
//   in a word simultaneously.
//
// Boundary handling: Cells outside the grid are treated as dead.
// The grid size is a power of two, >512. The bit-packed layout ensures each row contains
// grid_dimensions/64 words. We launch a 1D grid of threads over all words to avoid 2D grid
// dimension limits; row and column word indices are derived via bit-manipulation.
//
// Key idea for neighbor counting:
// - Build 8 direction bitmasks (UL, U, UR, L, R, LL, D, LR), using cross-word "shifts with carry".
// - Use bit-sliced addition via boolean algebra (CSAs) to compute per-bit neighbor counts,
//   finally producing bit-planes b0,b1,b2,b3 (LSB..MSB of the count).
// - Next state is: (neighbors == 3) OR (alive AND neighbors == 2).

// Combine left shift by 1 with carry-in from left (previous) word's MSB.
static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t x, std::uint64_t left_word) {
    return (x << 1) | (left_word >> 63);
}

// Combine right shift by 1 with carry-in from right (next) word's LSB.
static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t x, std::uint64_t right_word) {
    return (x >> 1) | (right_word << 63);
}

static __device__ __forceinline__ void sum3_bits(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                std::uint64_t& sum, std::uint64_t& carry) {
    // Per-bit sum of three 1-bit operands: sum (LSB), carry (MSB) -> represents a + b + c = sum + 2*carry
    sum   = a ^ b ^ c;
    carry = (a & b) | (b & c) | (a & c);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int grid_dim, int words_per_row,
                                unsigned int row_shift, std::size_t row_mask,
                                std::size_t total_words)
{
    // Global linear index of the 64-cell word processed by this thread
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    // Derive row and word-in-row using power-of-two properties
    // idx = row * words_per_row + cw
    std::size_t cw  = idx & row_mask;            // column word index
    std::size_t row = idx >> row_shift;          // row index

    // Base indices for current, previous, and next rows in the word-addressed arrays
    std::size_t base_cur = idx & ~row_mask; // idx - cw
    bool has_up   = (row > 0);
    bool has_down = (row + 1 < static_cast<std::size_t>(grid_dim));
    bool has_left_word  = (cw > 0);
    bool has_right_word = (cw + 1 < static_cast<std::size_t>(words_per_row));

    std::size_t base_up   = has_up   ? (base_cur - static_cast<std::size_t>(words_per_row)) : 0;
    std::size_t base_down = has_down ? (base_cur + static_cast<std::size_t>(words_per_row)) : 0;

    // Load the 3x3 words around (row, cw) needed for cross-word shifts
    // Use read-only cache loads; boundary words are treated as zero.
    std::uint64_t upL   = (has_up   && has_left_word ) ? __ldg(&input[base_up   + (cw - 1)]) : 0ull;
    std::uint64_t up    = (has_up                      ) ? __ldg(&input[base_up   +  cw     ]) : 0ull;
    std::uint64_t upR   = (has_up   && has_right_word)  ? __ldg(&input[base_up   + (cw + 1)]) : 0ull;

    std::uint64_t midL  = (has_left_word ) ? __ldg(&input[base_cur  + (cw - 1)]) : 0ull;
    std::uint64_t mid   = __ldg(&input[base_cur  +  cw     ]);
    std::uint64_t midR  = (has_right_word) ? __ldg(&input[base_cur  + (cw + 1)]) : 0ull;

    std::uint64_t dnL   = (has_down && has_left_word ) ? __ldg(&input[base_down + (cw - 1)]) : 0ull;
    std::uint64_t dn    = (has_down                 )  ? __ldg(&input[base_down +  cw     ]) : 0ull;
    std::uint64_t dnR   = (has_down && has_right_word) ? __ldg(&input[base_down + (cw + 1)]) : 0ull;

    // Build directional neighbor bitmasks for this word
    // Upper row neighbors
    std::uint64_t UL = shl1_with_carry(up,   upL);
    std::uint64_t U  = up;
    std::uint64_t UR = shr1_with_carry(up,   upR);

    // Middle row horizontal neighbors (exclude the center cell)
    std::uint64_t L  = shl1_with_carry(mid,  midL);
    std::uint64_t R  = shr1_with_carry(mid,  midR);

    // Lower row neighbors
    std::uint64_t LL = shl1_with_carry(dn,   dnL);
    std::uint64_t D  = dn;
    std::uint64_t LR = shr1_with_carry(dn,   dnR);

    // Sum neighbors in three groups: top (UL,U,UR), bottom (LL,D,LR), middle (L,R)
    std::uint64_t s_top, c_top;
    std::uint64_t s_bot, c_bot;
    sum3_bits(UL, U, UR, s_top, c_top);
    sum3_bits(LL, D, LR, s_bot, c_bot);

    // Sum of L and R (two operands)
    std::uint64_t s_mid = L ^ R;
    std::uint64_t c_mid = L & R;

    // Accumulate bit-plane 0 (b0) and initial carries to bit-plane 1
    std::uint64_t t0   = s_top ^ s_bot;   // partial sum of s_top + s_bot
    std::uint64_t t1a  = s_top & s_bot;   // carry to bit1 from s_top + s_bot
    std::uint64_t b0   = t0 ^ s_mid;      // final bit0 after adding s_mid
    std::uint64_t t1b  = t0 & s_mid;      // additional carry to bit1

    // Accumulate bit-plane 1 (b1) from five sources: t1a, t1b, c_top, c_bot, c_mid.
    // First CSA: t1a + t1b + c_top
    std::uint64_t y1   = t1a ^ t1b ^ c_top;
    std::uint64_t c2a  = (t1a & t1b) | (t1b & c_top) | (t1a & c_top); // carries to bit2

    // Second: c_bot + c_mid
    std::uint64_t y2   = c_bot ^ c_mid;
    std::uint64_t c2b  = c_bot & c_mid;   // carries to bit2

    // Final add for bit1 plane
    std::uint64_t b1   = y1 ^ y2;
    std::uint64_t c2c  = y1 & y2;         // carries to bit2

    // Accumulate bit-plane 2 (b2) and potential bit3 from three carry sources
    std::uint64_t z1 = c2a, z2 = c2b, z3 = c2c;
    std::uint64_t b2   = z1 ^ z2 ^ z3;
    std::uint64_t b3   = (z1 & z2) | (z2 & z3) | (z1 & z3);

    // Apply Life rules:
    // next = (neighbors == 3) | (alive & neighbors == 2)
    std::uint64_t not_b2 = ~b2;
    std::uint64_t not_b3 = ~b3;
    std::uint64_t eq3    = not_b3 & not_b2 & b1 & b0;         // exactly 3 neighbors
    std::uint64_t eq2    = not_b3 & not_b2 & b1 & (~b0);      // exactly 2 neighbors

    std::uint64_t next   = eq3 | (mid & eq2);

    output[idx] = next;
}

static inline unsigned int compute_log2_pow2(unsigned int x) {
    // x is a power of two > 0
    unsigned int r = 0;
    while ((x >> r) > 1u) ++r;
    return r;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the square dimension (power of two, >512)
    const int N = grid_dimensions;
    const int words_per_row = N >> 6; // N / 64
    const std::size_t total_words = static_cast<std::size_t>(N) * static_cast<std::size_t>(words_per_row);

    // Precompute mapping parameters for 1D thread indexing
    const unsigned int row_shift = compute_log2_pow2(static_cast<unsigned int>(words_per_row));
    const std::size_t row_mask = static_cast<std::size_t>(words_per_row - 1);

    // Launch configuration: 256 threads per block is a good default for A100/H100
    constexpr int THREADS = 256;
    const unsigned int blocks = static_cast<unsigned int>((total_words + THREADS - 1) / THREADS);

    gol_step_kernel<<<blocks, THREADS>>>(input, output, N, words_per_row, row_shift, row_mask, total_words);
}