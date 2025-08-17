#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  CUDA implementation of a single Conway's Game of Life step on a bit-packed grid.

  - The grid is square with dimensions N x N (N is a power of two, > 512).
  - Each 64-bit word (std::uint64_t) encodes 64 consecutive cells in a row, LSB=bit 0.
  - A single CUDA thread processes exactly one 64-bit word (i.e., 64 cells).
  - We avoid shared/texture memory. All loads are directly from global memory.
  - Boundaries are zero-padded (cells outside the grid are dead).

  Core idea for performance:
  - Build eight 64-bit "neighbor direction" bitboards (NW,N,NE,W,E,SW,S,SE) for each 64-bit
    word by using shifts and by injecting the neighboring words' edge bits where needed.
  - Use a bit-sliced parallel counter (carry-save adder network) to count the number of
    alive neighbors per bit position without cross-bit carries (i.e., without mixing
    adjacent cells). This avoids per-bit loops and leverages 64-way parallelism per thread.
  - From the per-bit 3/4-bit counts (represented as bitplanes), compute next state:
    next = (count == 3) | (alive & (count == 2)).

  Special handling for bit 0 and bit 63:
  - For left shifts, the "new" bit 0 is taken from the MSB (bit 63) of the left neighbor word.
  - For right shifts, the "new" bit 63 is taken from the LSB (bit 0) of the right neighbor word.
  - If a neighbor word does not exist due to grid boundaries, zeros are injected.

  Note: Although __popc/__popcll can be useful, the bit-sliced CSA approach here
  is generally faster for this problem on modern NVIDIA GPUs because it avoids
  per-bit work and processes 64 cells with a small, fixed set of bitwise ops.
*/

namespace gol_detail {

using u64 = std::uint64_t;

// Bit-sliced carry-save adder for three 64-bit "bitboards" at the same significance.
// For each bit lane i: sum_i, carry_i = a_i + b_i + c_i.
// - sum (s) carries the LSB (weight 1) for each lane.
// - carry (c) carries the carry-out bit (weight 2) for each lane.
// Important: carry is NOT shifted; it is the next higher significance bitplane for the same lane.
__device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& s, u64& carry) {
    // sum bit: XOR of all three
    s = a ^ b ^ c;
    // carry bit: majority(a,b,c) == (a&b) | (a&c) | (b&c)
    carry = (a & b) | (a & c) | (b & c);
}

// Load helper with boundary check for neighbor words.
// If the requested (row, col) is outside the valid range, returns 0 (dead cells).
__device__ __forceinline__ u64 load_word_or_zero(const u64* __restrict__ in,
                                                 int n_rows, int words_per_row,
                                                 int row, int col) {
    if (row < 0 || row >= n_rows || col < 0 || col >= words_per_row) return 0ULL;
    return in[static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row) + static_cast<std::size_t>(col)];
}

// Compute the eight neighbor-direction bitboards for a given word at (row, col).
// This handles injecting the edge bits from left/right neighbor words and zero-padding
// at grid boundaries.
// Inputs:
//   c  = current word
//   l  = left word (same row), or 0 if col==0
//   r  = right word (same row), or 0 if col==last
//   cu = word in the row above (same column), or 0 if row==0
//   lu = word in the row above and to the left, or 0 if row==0 or col==0
//   ru = word in the row above and to the right, or 0 if row==0 or col==last
//   cd = word in the row below (same column), or 0 if row==last
//   ld = word in the row below and to the left, or 0 if row==last or col==0
//   rd = word in the row below and to the right, or 0 if row==last or col==last
// Outputs (neighbor directions):
//   NW,N,NE,W,E,SW,S,SE as bitboards relative to the current word's cells.
__device__ __forceinline__ void build_neighbor_bitboards(u64 c, u64 l, u64 r,
                                                         u64 cu, u64 lu, u64 ru,
                                                         u64 cd, u64 ld, u64 rd,
                                                         u64& NW, u64& N, u64& NE,
                                                         u64& W,  u64& E,
                                                         u64& SW, u64& S,  u64& SE) {
    // Horizontal neighbors in the current row (W and E):
    // - W: left shift current word by 1 and inject MSB from left neighbor's bit63 into bit0.
    // - E: right shift current word by 1 and inject LSB from right neighbor's bit0 into bit63.
    // At boundaries, l or r is zero, so injected bits become zero automatically.
    W = (c << 1) | (l >> 63);
    E = (c >> 1) | (r << 63);

    // Neighbors from the row above (N, NW, NE):
    N  = cu;
    NW = (cu << 1) | (lu >> 63);
    NE = (cu >> 1) | (ru << 63);

    // Neighbors from the row below (S, SW, SE):
    S  = cd;
    SW = (cd << 1) | (ld >> 63);
    SE = (cd >> 1) | (rd << 63);
}

} // namespace gol_detail

// Kernel: one thread processes one 64-bit word (i.e., 64 cells).
__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int grid_dim, int words_per_row) {
    using namespace gol_detail;

    const std::size_t total_words = static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    // Grid-stride loop to cover arbitrary sizes while keeping launch params reasonable.
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        // 2D coordinates of the word in the grid
        int row = static_cast<int>(idx / static_cast<std::size_t>(words_per_row));
        int col = static_cast<int>(idx % static_cast<std::size_t>(words_per_row));

        // Load center and 8 neighbor words (zero-padded at boundaries).
        // We avoid branches within neighbor computations by pre-zeroing invalid neighbors.
        const int up    = row - 1;
        const int down  = row + 1;
        const int left  = col - 1;
        const int right = col + 1;

        u64 c  = load_word_or_zero(input, grid_dim, words_per_row, row,  col);
        u64 l  = load_word_or_zero(input, grid_dim, words_per_row, row,  left);
        u64 r  = load_word_or_zero(input, grid_dim, words_per_row, row,  right);

        u64 cu = load_word_or_zero(input, grid_dim, words_per_row, up,   col);
        u64 lu = load_word_or_zero(input, grid_dim, words_per_row, up,   left);
        u64 ru = load_word_or_zero(input, grid_dim, words_per_row, up,   right);

        u64 cd = load_word_or_zero(input, grid_dim, words_per_row, down, col);
        u64 ld = load_word_or_zero(input, grid_dim, words_per_row, down, left);
        u64 rd = load_word_or_zero(input, grid_dim, words_per_row, down, right);

        // Build the eight neighbor direction bitboards for this word.
        u64 NW, N, NE, W, E, SW, S, SE;
        build_neighbor_bitboards(c, l, r, cu, lu, ru, cd, ld, rd,
                                 NW, N, NE, W, E, SW, S, SE);

        // Bit-sliced parallel counting of neighbors per bit using a CSA tree.
        // Inputs (all at weight 1): NW, N, NE, W, E, SW, S, SE (8 bitboards).
        // Stage 1: Reduce 8 inputs at weight 1 into:
        //  - three sums at weight 1: s01, s23, s45
        //  - three carries at weight 2: c01, c23, c45
        u64 s01, c01;
        u64 s23, c23;
        u64 s45, c45;
        csa(NW, N,  NE, s01, c01);      // -> s01 (x1), c01 (x2)
        csa(W,  E,  SW, s23, c23);      // -> s23 (x1), c23 (x2)
        csa(S,  SE, 0ULL, s45, c45);    // -> s45 (x1), c45 (x2). Using 0 as the third operand.

        // Stage 2: Combine weight-1 sums: s01 + s23 + s45 -> s1 (x1) and carry_s1 (x2).
        u64 s1, carry_s1;
        csa(s01, s23, s45, s1, carry_s1);  // carry_s1 at weight 2

        // Stage 3: Combine weight-2 carries: c01 + c23 + c45 -> s2a (x2), c2a (x4).
        u64 s2a, c2a;
        csa(c01, c23, c45, s2a, c2a);  // c2a at weight 4

        // Stage 4: Add carry_s1 (x2) to s2a (x2): -> s2 (x2), c2b (x4).
        u64 s2, c2b;
        csa(s2a, carry_s1, 0ULL, s2, c2b); // c2b at weight 4

        // Stage 5: Combine the two weight-4 planes: c2a + c2b -> s4 (x4), c4 (x8).
        u64 s4, c4;
        csa(c2a, c2b, 0ULL, s4, c4);   // c4 at weight 8 (only set when exactly 8 neighbors)

        // Now the neighbor count per bit is: count = s1*1 + s2*2 + s4*4 + c4*8
        // We only need to detect count == 2 and count == 3:
        //  - eq2: s1==0, s2==1, s4==0, c4==0
        //  - eq3: s1==1, s2==1, s4==0, c4==0
        u64 not_s4 = ~s4;
        u64 not_c4 = ~c4;
        u64 eq3 =  s1 &  s2 & not_s4 & not_c4;
        u64 eq2 = (~s1) & s2 & not_s4 & not_c4;

        // Apply Life rules: next = (eq3) | (alive & eq2)
        u64 next = eq3 | (c & eq2);

        // Store result
        output[idx] = next;
    }
}

// Host entry point: executes one step of Game of Life using the CUDA kernel.
// input  - device pointer to bit-packed input grid (N*N cells -> N*(N/64) words).
// output - device pointer to bit-packed output grid (same size as input).
// grid_dimensions - N (width/height), power of two > 512.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    if (grid_dimensions <= 0) return;

    const int words_per_row = grid_dimensions >> 6; // N / 64
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    if (total_words == 0) return;

    // Choose launch configuration.
    // We use a grid-stride loop, so the exact grid size is flexible.
    // Use a moderate block size to balance register usage and occupancy.
    constexpr int threads_per_block = 256;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    // Aim for several blocks per SM to hide latency, but cap by total work.
    // 8 blocks per SM is a reasonable default; adjust if desired.
    std::size_t suggested_blocks = static_cast<std::size_t>(sm_count > 0 ? sm_count * 8 : 256);
    std::size_t max_blocks = (total_words + threads_per_block - 1) / threads_per_block;
    if (suggested_blocks > max_blocks) suggested_blocks = max_blocks;

    int blocks = static_cast<int>(suggested_blocks > 0 ? suggested_blocks : 1);

    game_of_life_step_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions, words_per_row);
}