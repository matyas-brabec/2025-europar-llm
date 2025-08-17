#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance one-step Conway's Game of Life using bit-parallel arithmetic.

  Representation:
  - The grid is square with dimensions N x N (N is a power of two, >512).
  - Each row is stored in words_per_row = N / 64 contiguous 64-bit words.
  - Each std::uint64_t encodes 64 consecutive cells of the same row: bit 0 is the leftmost of the 64-cells chunk.
  - Outside-the-grid cells are treated as dead (0).

  Kernel overview:
  - Each thread processes one 64-bit word (64 cells).
  - The algorithm uses bit-slicing (SWAR) and Boolean adders to compute the per-cell neighbor counts across the 3x3 neighborhood.
  - We avoid shared/texture memory; loads are coalesced and L2 cached.

  Neighbor count computation:
  - For each of the three relevant rows (above A, current C, below B) we form three bitsets:
      left-shifted (xL), center (xC), right-shifted (xR),
    where shifts include cross-word carry from adjacent 64-bit words.
  - The total number of neighbors N for each bit position is:
      N = (AL + CL + BL) + (AR + CR + BR) + (A + B)      // Note: no C in the center column (exclude self)
    Each term AL,CL,BL,AR,CR,BR,A,B is a bitmask (0/1 per bit). The sums are done with bitwise adders.

  Bitwise adders:
  - add2(a,b) -> returns (lsb, carry) representing a + b in binary (0..2), where:
      lsb = a XOR b
      carry = a AND b
  - add3(a,b,c) -> returns (lsb, carry) representing a + b + c in binary (0..3), where:
      lsb = a XOR b XOR c
      carry = (a&b) | ((a^b)&c)

  Building the neighbor count bit-planes:
  - Compute L = AL+CL+BL with add3 -> (L0, L1) representing values 0..3 in binary (1s and 2s planes).
  - Compute R = AR+CR+BR with add3 -> (R0, R1)
  - Compute V = A+B with add2 -> (V0, V1)
  - Sum the 1s planes: add3(L0, R0, V0) -> ones (n1) and carry_to_twos
  - Sum the 2s-plane terms (L1, R1, V1, carry_to_twos), yielding:
      - twos (n2) and 4s carries f0,f1,f2 via chained add2
      - Sum f0,f1,f2 with add3 -> fours (n4) and eights (n8)
  - Now neighbor count N is represented by bit-planes (n1, n2, n4, n8) corresponding to binary weights 1,2,4,8.

  Game of Life rule:
  - birth if N == 3
  - survive if current cell is alive and N == 2 or N == 3
  - Using bit-planes:
      eq3 = (n2 & n1) & ~(n4 | n8)      // 0011
      eq2 = (n2 & ~n1) & ~(n4 | n8)     // 0010
      next = eq3 | (C & eq2)

  Launch:
  - The kernel uses a grid-stride loop; run_game_of_life chooses a launch that scales well on A100/H100.
  - words_per_row is a power of two; we pass its log2 and mask to avoid integer division/modulo.
*/

static __device__ __forceinline__ void add2_u64(std::uint64_t a, std::uint64_t b,
                                                std::uint64_t &lsb, std::uint64_t &carry)
{
    lsb   = a ^ b;     // sum bit (1s plane)
    carry = a & b;     // carry bit (2s plane)
}

static __device__ __forceinline__ void add3_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                std::uint64_t &lsb, std::uint64_t &carry)
{
    // Full adder at bit-slice granularity: sum= a^b^c, carry = (a&b) | ((a^b)&c)
    std::uint64_t t = a ^ b;
    lsb   = t ^ c;
    carry = (a & b) | (t & c);
}

static __device__ __forceinline__ std::uint64_t shift_left_with_carry(std::uint64_t w, std::uint64_t wl)
{
    // Left neighbor alignment: for each bit position i, we want the bit at i-1 (left of cell).
    // Bring in the MSB of the left word into bit 0.
    return (w << 1) | (wl >> 63);
}

static __device__ __forceinline__ std::uint64_t shift_right_with_carry(std::uint64_t w, std::uint64_t wr)
{
    // Right neighbor alignment: for each bit position i, we want the bit at i+1 (right of cell).
    // Bring in the LSB of the right word into bit 63.
    return (w >> 1) | (wr << 63);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int N,               // grid height (rows)
                                int words_per_row,   // grid width in 64-bit words
                                int log2_wpr,        // log2(words_per_row), words_per_row is power of two
                                int wpr_mask)        // words_per_row - 1
{
    const std::size_t total_words = static_cast<std::size_t>(N) * static_cast<std::size_t>(words_per_row);
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        // Compute (row, col_word) without division/modulo using power-of-two properties
        const std::size_t row = idx >> log2_wpr;
        const int col = static_cast<int>(idx & static_cast<std::size_t>(wpr_mask));

        const bool has_up    = row > 0;
        const bool has_down  = (row + 1) < static_cast<std::size_t>(N);
        const bool has_left  = col > 0;
        const bool has_right = (col + 1) < words_per_row;

        const std::size_t row_base = row * static_cast<std::size_t>(words_per_row);

        // Load current row words
        const std::uint64_t WL = has_left  ? in[row_base + (col - 1)] : 0ull;
        const std::uint64_t W  = in[row_base + col];
        const std::uint64_t WR = has_right ? in[row_base + (col + 1)] : 0ull;

        // Load above row words
        const std::size_t above_base = has_up ? (row_base - static_cast<std::size_t>(words_per_row)) : 0u;
        const std::uint64_t AL = (has_up && has_left)  ? in[above_base + (col - 1)] : 0ull;
        const std::uint64_t A  = has_up                ? in[above_base + col]       : 0ull;
        const std::uint64_t AR = (has_up && has_right) ? in[above_base + (col + 1)] : 0ull;

        // Load below row words
        const std::size_t below_base = has_down ? (row_base + static_cast<std::size_t>(words_per_row)) : 0u;
        const std::uint64_t BL = (has_down && has_left)  ? in[below_base + (col - 1)] : 0ull;
        const std::uint64_t B  = has_down                ? in[below_base + col]       : 0ull;
        const std::uint64_t BR = (has_down && has_right) ? in[below_base + (col + 1)] : 0ull;

        // Build horizontally aligned neighbor contributions with cross-word carries
        const std::uint64_t ALs = shift_left_with_carry(A, AL);
        const std::uint64_t CLs = shift_left_with_carry(W, WL);
        const std::uint64_t BLs = shift_left_with_carry(B, BL);

        const std::uint64_t ARs = shift_right_with_carry(A, AR);
        const std::uint64_t CRs = shift_right_with_carry(W, WR);
        const std::uint64_t BRs = shift_right_with_carry(B, BR);

        // Vertical (center column) contributions exclude the current row W (self)
        const std::uint64_t AC = A;
        const std::uint64_t BC = B;

        // Sum left column contributions: ALs + CLs + BLs -> (L0, L1) bit-planes of 0..3
        std::uint64_t L0, L1;
        add3_u64(ALs, CLs, BLs, L0, L1);

        // Sum right column contributions: ARs + CRs + BRs -> (R0, R1)
        std::uint64_t R0, R1;
        add3_u64(ARs, CRs, BRs, R0, R1);

        // Sum vertical neighbors (above+below): AC + BC -> (V0, V1)
        std::uint64_t V0, V1;
        add2_u64(AC, BC, V0, V1);

        // Sum 1s-plane bits from left/right/vertical: produces ones bit-plane (n1) and carry to 2s-plane
        std::uint64_t n1, carry_to_twos;
        add3_u64(L0, R0, V0, n1, carry_to_twos);

        // Sum 2s-plane inputs: L1 + R1 + V1 + carry_to_twos
        // Use chained add2 to accumulate and collect carries to 4s-plane
        std::uint64_t t_lsb, f0, t_lsb2, f1, n2, f2;

        // Step 1: add L1 + R1 -> t_lsb (2s bit), f0 (one 4s carry)
        add2_u64(L1, R1, t_lsb, f0);
        // Step 2: add t_lsb + V1 -> t_lsb2 (2s), f1 (another 4s carry)
        add2_u64(t_lsb, V1, t_lsb2, f1);
        // Step 3: add t_lsb2 + carry_to_twos -> n2 (final 2s), f2 (another 4s carry)
        add2_u64(t_lsb2, carry_to_twos, n2, f2);

        // Sum 4s carries f0 + f1 + f2 -> n4 (4s bit-plane) and carry to 8s (n8)
        std::uint64_t n4, n8;
        add3_u64(f0, f1, f2, n4, n8);

        // Determine exactly 2 and exactly 3 neighbors:
        const std::uint64_t ge4 = n4 | n8;
        const std::uint64_t eq3 = (n2 & n1) & ~ge4;     // 0011
        const std::uint64_t eq2 = (n2 & ~n1) & ~ge4;    // 0010

        // Apply Game of Life rules:
        // survive: current alive & eq2; birth: eq3 (regardless of current)
        const std::uint64_t next = eq3 | (W & eq2);

        out[row_base + col] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is the number of cells per row/column (power of two).
    // words_per_row is also a power of two: grid_dimensions / 64.
    const int words_per_row = grid_dimensions >> 6;
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Compute log2(words_per_row) and mask for fast index arithmetic in the kernel
    int log2_wpr = 0;
    {
        int x = words_per_row;
        // Since words_per_row is a power of two, count trailing zeros gives log2.
        // Portable fallback loop:
        while ((1 << log2_wpr) < x) ++log2_wpr;
    }
    const int wpr_mask = words_per_row - 1;

    // Choose a launch configuration suitable for A100/H100.
    // Use a grid-stride loop inside the kernel to handle very large grids.
    constexpr int threads_per_block = 256;
    // Cap blocks to a reasonable number; kernel uses grid-stride to cover the rest.
    int max_blocks = 65535;
    int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);
    if (blocks > max_blocks) blocks = max_blocks;

    gol_step_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions, words_per_row, log2_wpr, wpr_mask);
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
