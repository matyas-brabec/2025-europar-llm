#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Encoding:
  - Each 64-bit word encodes 64 horizontal cells in a row: bit k (0..63) is the cell at column (64*word_idx + k).
  - A bit value of 1 denotes a live cell, 0 a dead cell.
  - Grid is square of size N x N, with N a power of two > 512. N is divisible by 64, so words_per_row = N/64 is also a power of two.
  - All cells outside the grid are considered dead (no wrap-around).

  Parallelization:
  - Each CUDA thread processes exactly one 64-bit word (64 cells).
  - Threads read up to 9 words around their own (for the 3x3 neighborhood) and compute the next 64 cells in parallel using bitwise operations.

  Neighbor counting:
  - For each bit position, there are 8 neighbors: N, NE, E, SE, S, SW, W, NW.
  - We form 8 bit-vectors (each a uint64_t), one per neighbor direction, aligned so that bit position j corresponds to the neighbor of cell j.
  - Horizontal and diagonal neighbors require cross-word bit injection for bit positions 0 and 63. This is done via shifts and using adjacent words' boundary bits.
  - We then sum the 8 one-bit values per position using a carry-save adder (CSA) tree:
      * First reduce the three top-row neighbors (NW,N,NE) and the three bottom-row neighbors (SW,S,SE) separately via full adders.
      * Reduce the two middle-row neighbors (W,E) via a half adder.
      * Combine the partial sums to obtain the low two bits of the neighbor count (ones and twos) and a flag indicating whether the count is >=4.
    This gives us exactly what we need to apply Conway's rules:
      - eq3_mask: positions with exactly 3 neighbors -> (~fours) & ones & twos
      - eq2_mask: positions with exactly 2 neighbors -> (~fours) & (~ones) & twos
    The next state per bit is: next = eq3_mask | (alive & eq2_mask).
*/

static __forceinline__ __device__ void csa3(const std::uint64_t a,
                                            const std::uint64_t b,
                                            const std::uint64_t c,
                                            std::uint64_t &sum,
                                            std::uint64_t &carry)
{
    // Full adder for three 1-bit inputs per bit-position:
    // sum   = a ^ b ^ c                  (weight 1)
    // carry = majority(a,b,c)            (weight 2)
    const std::uint64_t ab = a ^ b;
    sum   = ab ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

__global__ void gol_kernel_bitpacked(const std::uint64_t* __restrict__ in,
                                     std::uint64_t* __restrict__ out,
                                     int words_per_row,
                                     int rows,
                                     unsigned int log2_words_per_row,
                                     unsigned int words_mask,
                                     std::uint64_t total_words)
{
    const std::uint64_t tid = blockIdx.x * std::uint64_t(blockDim.x) + threadIdx.x;
    if (tid >= total_words) return;

    // Compute word index within row (x) and row index (y) using power-of-two properties
    const unsigned int x = static_cast<unsigned int>(tid) & words_mask;
    const unsigned int y = static_cast<unsigned int>(tid >> log2_words_per_row);

    // Determine row boundary predicates
    const bool have_left  = (x > 0);
    const bool have_right = (x + 1u < static_cast<unsigned int>(words_per_row));
    const bool have_above = (y > 0);
    const bool have_below = (y + 1u < static_cast<unsigned int>(rows));

    const std::uint64_t center = in[tid];

    // Neighbor word loads with out-of-bounds clamped to zero
    std::uint64_t left   = have_left  ? in[tid - 1] : 0ull;
    std::uint64_t right  = have_right ? in[tid + 1] : 0ull;

    const std::uint64_t idx_above = tid - static_cast<std::uint64_t>(words_per_row);
    const std::uint64_t idx_below = tid + static_cast<std::uint64_t>(words_per_row);

    std::uint64_t above_c = have_above ? in[idx_above] : 0ull;
    std::uint64_t above_l = (have_above && have_left)  ? in[idx_above - 1] : 0ull;
    std::uint64_t above_r = (have_above && have_right) ? in[idx_above + 1] : 0ull;

    std::uint64_t below_c = have_below ? in[idx_below] : 0ull;
    std::uint64_t below_l = (have_below && have_left)  ? in[idx_below - 1] : 0ull;
    std::uint64_t below_r = (have_below && have_right) ? in[idx_below + 1] : 0ull;

    // Construct aligned neighbor bit-vectors:
    // For each direction D, D's bit j equals the neighbor of center bit j in that direction.
    // - N,S use the centered words directly.
    // - E,W and diagonals require shifts with cross-word bit injection:
    //   E  = (center >> 1) | (right  << 63)   // inject right bit0 into bit63
    //   W  = (center << 1) | (left   >> 63)   // inject left  bit63 into bit0
    //   NE = (above_c >> 1) | (above_r << 63)
    //   NW = (above_c << 1) | (above_l >> 63)
    //   SE = (below_c >> 1) | (below_r << 63)
    //   SW = (below_c << 1) | (below_l >> 63)
    const std::uint64_t N  = above_c;
    const std::uint64_t S  = below_c;
    const std::uint64_t E  = (center >> 1) | (right  << 63);
    const std::uint64_t W  = (center << 1) | (left   >> 63);
    const std::uint64_t NE = (above_c >> 1) | (above_r << 63);
    const std::uint64_t NW = (above_c << 1) | (above_l >> 63);
    const std::uint64_t SE = (below_c >> 1) | (below_r << 63);
    const std::uint64_t SW = (below_c << 1) | (below_l >> 63);

    // Carry-save adder tree:
    // Sum top triple (NW,N,NE) -> t1 (ones), t2 (twos)
    std::uint64_t t1, t2;
    csa3(NW, N, NE, t1, t2);

    // Sum bottom triple (SW,S,SE) -> b1 (ones), b2 (twos)
    std::uint64_t b1, b2;
    csa3(SW, S, SE, b1, b2);

    // Sum middle pair (W,E) -> h1 (ones), h2 (twos) as a half-adder
    const std::uint64_t h1 = W ^ E;
    const std::uint64_t h2 = W & E;

    // Accumulate the ones (t1 + b1 + h1) and twos (t2 + b2 + h2) groups
    std::uint64_t ones, c_ones;   // ones: bit0 of total neighbor count; c_ones carries into twos
    csa3(t1, b1, h1, ones, c_ones);

    std::uint64_t twos, c_twos;   // twos: bit1 (weight 2) partial; c_twos carries into fours
    csa3(t2, b2, h2, twos, c_twos);

    // Combine the two weight-2 contributions (twos + c_ones).
    // twos_final: bit1 (weight 2) of total neighbor count
    // fours_any : boolean flag for ">= 4" neighbors (weight 4 or higher present)
    const std::uint64_t twos_final = twos ^ c_ones;          // remaining weight-2 units
    std::uint64_t fours_any = (twos & c_ones) | c_twos;      // any weight-4 generated

    // Apply rules:
    // eq3: exactly 3 neighbors -> (~fours_any) & ones & twos_final
    // eq2: exactly 2 neighbors -> (~fours_any) & (~ones) & twos_final
    const std::uint64_t not_fours = ~fours_any;
    const std::uint64_t eq3 = not_fours & ones & twos_final;
    const std::uint64_t eq2 = not_fours & (~ones) & twos_final;

    // Next state: born with 3 neighbors or survive if alive with 2 neighbors
    const std::uint64_t next = eq3 | (center & eq2);

    out[tid] = next;
}

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is a power of two > 512 and divisible by 64.
    const int rows = grid_dimensions;
    const int words_per_row = grid_dimensions >> 6; // N / 64
    const std::uint64_t total_words = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(words_per_row);

    // Compute log2(words_per_row) and mask for fast x/y extraction from linear index
    unsigned int log2_words = 0;
    {
        unsigned int w = static_cast<unsigned int>(words_per_row);
        // Since words_per_row is a power of two, count trailing zeros equals log2.
        while ((1u << log2_words) < w) ++log2_words;
    }
    const unsigned int words_mask = static_cast<unsigned int>(words_per_row - 1);

    // Launch configuration: one thread per 64-bit word
    const int block_size = 256; // tuned for modern data center GPUs (A100/H100)
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    gol_kernel_bitpacked<<<grid_size, block_size>>>(
        input, output, words_per_row, rows, log2_words, words_mask, total_words
    );
}