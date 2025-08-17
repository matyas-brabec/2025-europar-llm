#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance, bit-parallel CUDA implementation of one Conway's Game of Life step.

  Key points:
  - Grid is bit-packed: each 64-bit word encodes 64 cells in a row (bit 0 = leftmost bit in the word).
  - Each CUDA thread processes exactly one 64-bit word (no atomics needed).
  - Outside-of-grid cells are treated as dead (zero), so boundary threads inject zeros for out-of-range neighbors.
  - We avoid shared/texture memory as it does not help this workload notably on modern GPUs (A100/H100).
  - We compute each word's 8-neighbor count per bit using bit-sliced adders (carry-save at the bitboard level),
    which is extremely efficient and avoids per-bit loops or branching.
  - The 0th and 63rd bits within a 64-bit word require contributions from adjacent words (left/right), which is
    handled by including cross-word carry-in bits on shifts:
      - Left neighbors use: (cur << 1) | (left >> 63)
      - Right neighbors use: (cur >> 1) | ((right & 1ull) << 63)
    and similarly for above/below rows.

  Bit-sliced neighbor sum overview:
  - We first form the 8 neighbor bitboards aligned to the central word's bit positions:
      up-left, up, up-right,
      left,      right,
      down-left, down, down-right.
  - We then reduce horizontally per row (3 for up, 2 for middle, 3 for down) to 2-bit "row counts":
      row_count = row_lsb + 2 * row_msb  (per bit position)
  - Combine the three rows' counts:
      total = (u0 + m0 + d0) + 2 * (u1 + m1 + d1)
    Using carry-save adders, we produce the bit-planes b0, b1, b2, b3 representing the total neighbor count
    (0..8) per bit:
      total = b0 + 2*b1 + 4*b2 + 8*b3
    Only counts 0..8 can occur, so b3 is only ever set for count == 8.
  - Finally apply rules:
      next = (count == 3) | (alive & (count == 2))

  Note on performance:
  - This bit-sliced method minimizes instruction count and memory I/O. It often outperforms popcount-based
    per-cell methods. Using __popc/__popcll can speed certain approaches, but the bit-slice adder strategy
    here tends to perform extremely well for bit-packed Life on modern GPUs.
*/

static __device__ __forceinline__
void add3_bitwise(uint64_t a, uint64_t b, uint64_t c, uint64_t &sum, uint64_t &carry) {
    // Computes per-bit sum (sum: bit0 plane) and carry (carry: bit1 plane) of a + b + c,
    // without cross-bit carries (bit-sliced addition).
    // sum = a ^ b ^ c
    // carry = (a&b) | (c&(a^b))
    uint64_t t = a ^ b;
    sum = t ^ c;
    carry = (a & b) | (c & t);
}

static __device__ __forceinline__
void add2_bitwise(uint64_t a, uint64_t b, uint64_t &sum, uint64_t &carry) {
    // Computes per-bit sum (sum: bit0 plane) and carry (carry: bit1 plane) of a + b,
    // without cross-bit carries (bit-sliced addition).
    // sum = a ^ b
    // carry = a & b
    sum = a ^ b;
    carry = a & b;
}

__global__ void life_kernel_wordwise(const std::uint64_t* __restrict__ in,
                                     std::uint64_t* __restrict__ out,
                                     int words_per_row,
                                     int rows)
{
    // Global word index
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total_words = (size_t)words_per_row * (size_t)rows;
    if (idx >= total_words) return;

    // 2D location in word grid
    int row = (int)(idx / words_per_row);
    int col = (int)(idx - (size_t)row * (size_t)words_per_row);

    // Boundary flags
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < rows);
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    const size_t row_off = (size_t)row * (size_t)words_per_row;

    // Load current row words
    const std::uint64_t cur   = in[row_off + col];
    const std::uint64_t left  = has_left  ? in[row_off + (col - 1)] : 0ull;
    const std::uint64_t right = has_right ? in[row_off + (col + 1)] : 0ull;

    // Load above row neighborhood words (0 if out of bounds)
    const size_t up_off = has_up ? ((size_t)(row - 1) * (size_t)words_per_row) : 0;
    const std::uint64_t upC  = has_up ? in[up_off + col] : 0ull;
    const std::uint64_t upLw = (has_up && has_left)  ? in[up_off + (col - 1)] : 0ull;
    const std::uint64_t upRw = (has_up && has_right) ? in[up_off + (col + 1)] : 0ull;

    // Load below row neighborhood words (0 if out of bounds)
    const size_t dn_off = has_down ? ((size_t)(row + 1) * (size_t)words_per_row) : 0;
    const std::uint64_t dnC  = has_down ? in[dn_off + col] : 0ull;
    const std::uint64_t dnLw = (has_down && has_left)  ? in[dn_off + (col - 1)] : 0ull;
    const std::uint64_t dnRw = (has_down && has_right) ? in[dn_off + (col + 1)] : 0ull;

    // Build the eight neighbor bitboards, aligned to this word's bit positions.
    // Cross-word spills for bit 0 and bit 63 are handled by incorporating a single bit from neighbor words.
    // Above row neighbors:
    const std::uint64_t n_upL = (upC << 1) | (upLw >> 63);
    const std::uint64_t n_upC = upC;
    const std::uint64_t n_upR = (upC >> 1) | ((upRw & 1ull) << 63);
    // Current row neighbors (left/right only, the center cell itself is not a neighbor):
    const std::uint64_t n_midL = (cur << 1) | (left >> 63);
    const std::uint64_t n_midR = (cur >> 1) | ((right & 1ull) << 63);
    // Below row neighbors:
    const std::uint64_t n_dnL = (dnC << 1) | (dnLw >> 63);
    const std::uint64_t n_dnC = dnC;
    const std::uint64_t n_dnR = (dnC >> 1) | ((dnRw & 1ull) << 63);

    // Reduce each row's three/two neighbor bitboards into a 2-bit count: row_count = row0 + 2*row1

    // Top row: add3 of (upL, upC, upR) -> (u0, u1)
    std::uint64_t u0, u1;
    add3_bitwise(n_upL, n_upC, n_upR, u0, u1);

    // Middle row: add2 of (midL, midR) -> (m0, m1)
    std::uint64_t m0, m1;
    add2_bitwise(n_midL, n_midR, m0, m1);

    // Bottom row: add3 of (dnL, dnC, dnR) -> (d0, d1)
    std::uint64_t d0, d1;
    add3_bitwise(n_dnL, n_dnC, n_dnR, d0, d1);

    // Combine the three rows:
    // total = (u0 + m0 + d0) + 2*(u1 + m1 + d1)
    // First sum the low (LSB) planes across rows -> s0 (bit0), c0 (bit1)
    std::uint64_t s0, c0;
    add3_bitwise(u0, m0, d0, s0, c0);

    // Then sum the high (MSB) planes across rows -> s1 (bit0), c1 (bit1)
    std::uint64_t s1, c1;
    add3_bitwise(u1, m1, d1, s1, c1);

    // Now construct final neighbor count bit-planes b0..b3:
    // low bit (1s):
    const std::uint64_t b0 = s0;
    // 2s bit: from c0 (carry of low planes) XOR s1 (sum of high planes)
    const std::uint64_t b1 = c0 ^ s1;
    // 4s contributions come from:
    //   - c1 (carry of high planes) => one '4'
    //   - (c0 & s1) (two '2's added together) => another '4'
    const std::uint64_t four_a = c1;
    const std::uint64_t four_b = c0 & s1;
    const std::uint64_t b2 = four_a ^ four_b;   // 4s bit
    const std::uint64_t b3 = four_a & four_b;   // 8s bit (only set when count == 8)

    // Apply Game of Life rules:
    // - Birth if count == 3
    // - Survival if alive and count == 2
    const std::uint64_t alive = cur;
    const std::uint64_t not_b2 = ~b2;
    const std::uint64_t not_b3 = ~b3;

    const std::uint64_t eq3 = not_b3 & not_b2 & b1 & b0;         // 0b0011
    const std::uint64_t eq2 = not_b3 & not_b2 & b1 & (~b0);      // 0b0010

    const std::uint64_t next = eq3 | (alive & eq2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions: number of cells per row/column (square), power of 2, >512.
    // words_per_row: number of 64-bit words per row (grid_dimensions / 64).
    const int words_per_row = grid_dimensions >> 6;  // /64
    const size_t total_words = (size_t)words_per_row * (size_t)grid_dimensions;

    // Launch configuration: simple 1D grid over words.
    // 256 threads per block is a good default on A100/H100 for this memory/ALU mix.
    constexpr int threads_per_block = 256;
    const int blocks = (int)((total_words + threads_per_block - 1) / threads_per_block);

    life_kernel_wordwise<<<blocks, threads_per_block>>>(input, output, words_per_row, grid_dimensions);

    // No synchronization here; caller is responsible for synchronization if needed.
}