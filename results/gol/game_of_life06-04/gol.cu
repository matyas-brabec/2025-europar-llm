#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance, bit-parallel implementation of one step of Conway's Game of Life on the GPU.

  - Each CUDA thread processes exactly one 64-bit word (64 cells in the same row).
  - The grid is bit-packed row-wise: each std::uint64_t encodes 64 consecutive cells within a row.
  - Neighbor counting is done using a carry-save adder (full-adder) reduction network to sum 8 neighbor bit-vectors in parallel.
  - Boundary handling: cells outside the grid are considered dead (0). Word-edge bit neighbors (bit 0 and bit 63) pull from adjacent 64-bit words on the left/right when available; otherwise treated as 0.
  - No shared or texture memory is used; global loads are coalesced for the primary row and likely acceptable for adjacent rows.

  Next state rule per bit (for 8-neighbor counts):
    next = (neighbors == 3) | (alive & (neighbors == 2))
*/

static __forceinline__ __device__ void csa3(const std::uint64_t a,
                                            const std::uint64_t b,
                                            const std::uint64_t c,
                                            std::uint64_t &sum,
                                            std::uint64_t &carry) {
    // Bitwise full adder for three input bit-vectors:
    // sum   = a XOR b XOR c      (bit-plane of weight W)
    // carry = majority(a,b,c)    (bit-plane of weight 2W)
    sum   = a ^ b ^ c;
    carry = (a & b) | (b & c) | (c & a);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int words_per_row,
                                int grid_dim_cells)
{
    const std::size_t total_words = static_cast<std::size_t>(grid_dim_cells) * static_cast<std::size_t>(words_per_row);
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;

    // Map 1D thread id to (row, word_col)
    const int row = static_cast<int>(tid / words_per_row);
    const int word_col = static_cast<int>(tid - static_cast<std::size_t>(row) * words_per_row);

    // Boundary flags
    const bool has_left  = (word_col > 0);
    const bool has_right = (word_col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim_cells);

    // Load center word and neighbors as needed. Out-of-bounds treated as 0.
    const std::size_t idx      = tid;
    const std::size_t up_base  = has_up   ? (idx - static_cast<std::size_t>(words_per_row)) : 0;
    const std::size_t dn_base  = has_down ? (idx + static_cast<std::size_t>(words_per_row)) : 0;

    const std::uint64_t C  = input[idx];
    const std::uint64_t CL = has_left  ? input[idx - 1] : 0ull;
    const std::uint64_t CR = has_right ? input[idx + 1] : 0ull;

    const std::uint64_t U  = has_up    ? input[up_base] : 0ull;
    const std::uint64_t UL = (has_up && has_left)  ? input[up_base - 1] : 0ull;
    const std::uint64_t UR = (has_up && has_right) ? input[up_base + 1] : 0ull;

    const std::uint64_t D  = has_down  ? input[dn_base] : 0ull;
    const std::uint64_t DL = (has_down && has_left)  ? input[dn_base - 1] : 0ull;
    const std::uint64_t DR = (has_down && has_right) ? input[dn_base + 1] : 0ull;

    // Construct the eight neighbor bit-vectors for the current 64-bit word.
    // Horizontal neighbors (same row):
    // - For the "left" neighbor map: shift left by 1 (bit j gets neighbor from original bit j-1).
    //   Inject bit-63 from the left word into bit-0 to handle the j=0 edge.
    // - For the "right" neighbor map: shift right by 1 (bit j gets neighbor from original bit j+1).
    //   Inject bit-0 from the right word into bit-63 to handle the j=63 edge.
    const std::uint64_t W_left  = (C << 1) | (has_left  ? (CL >> 63)                 : 0ull);
    const std::uint64_t W_right = (C >> 1) | (has_right ? ((CR & 0x1ull) << 63)      : 0ull);

    // Vertical and diagonal neighbors constructed similarly, using adjacent row words and cross-word injections.
    const std::uint64_t N_center = U; // direct vertical neighbor (row-1, same column)
    const std::uint64_t S_center = D; // direct vertical neighbor (row+1, same column)

    const std::uint64_t N_left  = has_up   ? ((U << 1) | (has_left  ? (UL >> 63)            : 0ull)) : 0ull;                   // NW
    const std::uint64_t N_right = has_up   ? ((U >> 1) | (has_right ? ((UR & 0x1ull) << 63) : 0ull)) : 0ull;                   // NE
    const std::uint64_t S_left  = has_down ? ((D << 1) | (has_left  ? (DL >> 63)            : 0ull)) : 0ull;                   // SW
    const std::uint64_t S_right = has_down ? ((D >> 1) | (has_right ? ((DR & 0x1ull) << 63) : 0ull)) : 0ull;                   // SE

    // Neighbor bit-vectors: n0..n7 (order does not matter)
    const std::uint64_t n0 = N_left;
    const std::uint64_t n1 = N_center;
    const std::uint64_t n2 = N_right;
    const std::uint64_t n3 = W_left;
    const std::uint64_t n4 = W_right;
    const std::uint64_t n5 = S_left;
    const std::uint64_t n6 = S_center;
    const std::uint64_t n7 = S_right;

    // Carry-save adder reduction tree to sum eight 1-bit inputs per bit position (0..63).
    // Each stage reduces three inputs into two bit-planes: sum (weight W) and carry (weight 2W).
    // After reductions, we end up with bit-planes for weights 1, 2, 4, and 8 corresponding to the neighbor count per bit.
    std::uint64_t s01, c01; csa3(n0, n1, n2, s01, c01);            // weights: 1 and 2
    std::uint64_t s23, c23; csa3(n3, n4, n5, s23, c23);            // weights: 1 and 2
    std::uint64_t s67, c67;                                        // reduce last two (n6, n7, 0)
    s67 = (n6 ^ n7);
    c67 = (n6 & n7);                                               // weights: 1 and 2

    // Combine the three 'sum-of-ones' groups (weights remain 1 and 2):
    std::uint64_t sA, cA; csa3(s01, s23, s67, sA, cA);             // sA: weight 1, cA: weight 2

    // Combine the three 'carry-of-twos' groups (inputs have weight 2, outputs sB=2, cB=4):
    std::uint64_t sB, cB; csa3(c01, c23, c67, sB, cB);             // sB: weight 2, cB: weight 4

    // Merge all weight-2 contributions (cA and sB):
    std::uint64_t s2, c4a; csa3(cA, sB, 0ull, s2, c4a);            // s2: weight 2, c4a: weight 4

    // Merge all weight-4 contributions (c4a and cB):
    std::uint64_t s4, c8;  csa3(c4a, cB, 0ull, s4, c8);            // s4: weight 4, c8: weight 8

    // Bit-planes now represent per-bit neighbor counts in binary:
    //   ones   = sA (1's place)
    //   twos   = s2 (2's place)
    //   fours  = s4 (4's place)
    //   eights = c8 (8's place; only set when neighbor count == 8)

    // Compute equality masks for counts == 2 and == 3:
    const std::uint64_t not_ones   = ~sA;
    const std::uint64_t not_twos   = ~s2;
    const std::uint64_t not_fours  = ~s4;
    const std::uint64_t not_eights = ~c8;

    // eq2: 0010b => ~ones & twos & ~fours & ~eights
    const std::uint64_t eq2 = (not_ones) & (s2) & (not_fours) & (not_eights);

    // eq3: 0011b => ones & twos & ~fours & ~eights
    const std::uint64_t eq3 = (sA) & (s2) & (not_fours) & (not_eights);

    // Apply Game of Life rules: survive with 2 neighbors, born with 3 neighbors
    const std::uint64_t next = eq3 | (C & eq2);

    output[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is the width and height in cells; each 64-bit word packs 64 horizontal cells.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64 (always divisible since it's a power of two)
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Launch one thread per 64-bit word.
    // A moderate block size balances occupancy and latency hiding on modern NVIDIA GPUs.
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((total_words + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gol_step_kernel<<<grid_size, BLOCK_SIZE>>>(input, output, words_per_row, grid_dimensions);
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
