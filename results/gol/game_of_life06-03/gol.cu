#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
  CUDA Conway's Game of Life (bit-packed, 64 cells per word)

  Key points:
  - Each CUDA thread processes exactly one 64-bit word (64 cells in a row).
  - The grid is N x N, N is a power of two, N > 512. Words per row = N / 64 (also a power of two).
  - Bit-packed layout: the i-th bit of a word represents one cell (1=alive, 0=dead).
  - Cells outside the grid are considered dead (zero), so boundary conditions are zero-padded.
  - To compute the 8-neighbor counts efficiently, we use carry-save adder (CSA) logic on 64-bit lanes to
    add bitmasks in parallel across all 64 cells (SIMD within a register).
  - For neighbor alignment across word boundaries, we combine bit shifts with neighbor words:
      - Left neighbor across word boundary uses the MSB of the left word for the 0th bit.
      - Right neighbor across word boundary uses the LSB of the right word for the 63rd bit.
  - We only need to test for (count == 3) or (count == 2) for live cell survival/birth.
    The neighbor sum of 8 fits in 4 bits; we compute only the lower 3 bits via CSA (mod 8).
    Sum==8 (1000b) has low 3 bits 000, which does not alias 2 (010) or 3 (011), so ignoring the 8's bit is safe.
*/

static inline int ilog2_pow2_host(unsigned int x) {
    // Host-side integer log2 for power-of-two x (used to avoid integer division in kernel).
    // This loop runs log2(x) iterations; for typical grids this is tiny (e.g., widthWords >= 8 -> <= 12 iterations).
    int s = 0;
    while ((x & 1u) == 0u) { x >>= 1; ++s; }
    return s;
}

__device__ __forceinline__ void csa_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                        std::uint64_t &sum, std::uint64_t &carry) {
    // Carry-Save Adder for three 64-bit operands at once, bitwise across all 64 lanes.
    // sum  = a XOR b XOR c             -> LSB of the per-bit sum
    // carry= majority(a,b,c)           -> 1 if at least two inputs are 1 at that bit position
    // Optimized majority: (a & b) | (c & (a ^ b))
    const std::uint64_t axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (c & axb);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int widthWords, int height, int widthWordsShift)
{
    const std::size_t totalWords = static_cast<std::size_t>(widthWords) * static_cast<std::size_t>(height);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalWords) return;

    // Map linear index -> (row, word-column). widthWords is a power of two, so we use bit ops:
    const int y  = static_cast<int>(idx >> widthWordsShift);
    const int xw = static_cast<int>(idx & static_cast<std::size_t>(widthWords - 1));

    const std::size_t rowBase = static_cast<std::size_t>(y) * static_cast<std::size_t>(widthWords);

    // Determine neighbor row bases, guarded by bounds
    const bool hasUp   = (y > 0);
    const bool hasDown = (y + 1 < height);

    const std::size_t rowUpBase   = hasUp   ? (rowBase - static_cast<std::size_t>(widthWords)) : 0;
    const std::size_t rowDownBase = hasDown ? (rowBase + static_cast<std::size_t>(widthWords)) : 0;

    // Load center word and its 8 neighbor words (missing neighbors are treated as zero).
    // We avoid out-of-bounds by guarding each load.
    const bool hasLeft  = (xw > 0);
    const bool hasRight = (xw + 1 < widthWords);

    const std::uint64_t cw  = in[rowBase + xw];
    const std::uint64_t wl  = hasLeft  ? in[rowBase + (xw - 1)] : 0ull;
    const std::uint64_t wr  = hasRight ? in[rowBase + (xw + 1)] : 0ull;

    const std::uint64_t up_c = hasUp   ? in[rowUpBase   + xw]               : 0ull;
    const std::uint64_t up_l = (hasUp   && hasLeft)  ? in[rowUpBase   + (xw - 1)] : 0ull;
    const std::uint64_t up_r = (hasUp   && hasRight) ? in[rowUpBase   + (xw + 1)] : 0ull;

    const std::uint64_t dn_c = hasDown ? in[rowDownBase + xw]               : 0ull;
    const std::uint64_t dn_l = (hasDown && hasLeft)  ? in[rowDownBase + (xw - 1)] : 0ull;
    const std::uint64_t dn_r = (hasDown && hasRight) ? in[rowDownBase + (xw + 1)] : 0ull;

    // Construct the eight neighbor bitmasks aligned to the current word's bit positions:
    // - Horizontal (same row): left and right neighbors with cross-word carries.
    // - Vertical: up and down (no shift).
    // - Diagonals: UL, UR (from up row), DL, DR (from down row), with cross-word carries.
    const std::uint64_t hL  = (cw << 1) | (wl >> 63);
    const std::uint64_t hR  = (cw >> 1) | (wr << 63);

    const std::uint64_t vU  = up_c;
    const std::uint64_t vD  = dn_c;

    const std::uint64_t dUL = (up_c << 1) | (up_l >> 63);
    const std::uint64_t dUR = (up_c >> 1) | (up_r << 63);
    const std::uint64_t dDL = (dn_c << 1) | (dn_l >> 63);
    const std::uint64_t dDR = (dn_c >> 1) | (dn_r << 63);

    // Carry-save adder tree to compute the lower 3 bits (mod 8) of the 8-neighbor popcount per bit.
    // Step 1: Reduce 8 inputs -> three partial sums using CSAs (units and twos planes).
    std::uint64_t s1, c1, s2, c2, s3, c3;
    csa_u64(hL,  hR,  vU,  s1, c1);          // s1: units, c1: twos
    csa_u64(vD,  dUL, dUR, s2, c2);          // s2: units, c2: twos
    csa_u64(dDL, dDR, 0ull, s3, c3);         // s3: units, c3: twos

    // Step 2: Sum units plane (s1 + s2 + s3) -> b0 (LSB of neighbor count), carry_b0 contributes to twos plane
    std::uint64_t b0, carry_b0;
    csa_u64(s1, s2, s3, b0, carry_b0);       // b0: bit 0 (units), carry_b0: to twos plane

    // Step 3: Sum twos plane (c1 + c2 + c3 + carry_b0)
    // First three via CSA -> s1_twos (still twos plane), c1_twos (to fours plane)
    std::uint64_t s1_twos, c1_twos;
    csa_u64(c1, c2, c3, s1_twos, c1_twos);

    // Add the remaining carry_b0 to the twos plane using a half-adder:
    const std::uint64_t b1       = s1_twos ^ carry_b0;   // bit 1 (twos)
    const std::uint64_t carry_b1 = s1_twos & carry_b0;   // carry to fours plane

    // Step 4: Sum fours plane (c1_twos + carry_b1) to get bit 2 (fours). Any carry out would be bit 3 (eights), which we can ignore.
    const std::uint64_t b2 = c1_twos ^ carry_b1;         // bit 2 (fours)
    // const std::uint64_t b3 = c1_twos & carry_b1;      // bit 3 (eights) - ignored

    // Apply Game of Life rules:
    // - Birth if exactly 3 neighbors: eq3 = (~b2) & b1 & b0
    // - Survival for alive cells if exactly 2 neighbors: eq2 = (~b2) & b1 & (~b0)
    // Next state: next = eq3 | (current & eq2)
    const std::uint64_t not_b2 = ~b2;
    const std::uint64_t eq3    = not_b2 & b1 &  b0;
    const std::uint64_t eq2    = not_b2 & b1 & ~b0;
    const std::uint64_t next   = eq3 | (cw & eq2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions = N (cells per side), a power of two, > 512
    // Words per row (64 cells per word):
    const int widthWords = grid_dimensions >> 6; // divide by 64
    const int height     = grid_dimensions;

    // Precompute shift to replace division/modulo by bit ops in the kernel.
    const int widthWordsShift = ilog2_pow2_host(static_cast<unsigned int>(widthWords));

    // Launch configuration: one thread per 64-bit word
    const std::size_t totalWords = static_cast<std::size_t>(widthWords) * static_cast<std::size_t>(height);
    const int blockSize = 256;
    const int gridSize  = static_cast<int>((totalWords + blockSize - 1) / blockSize);

    gol_step_kernel<<<gridSize, blockSize>>>(input, output, widthWords, height, widthWordsShift);
    // Caller is responsible for synchronization and error checking if desired.
}