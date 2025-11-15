// Conway's Game of Life CUDA implementation operating directly on a bit-packed grid.
// Each std::uint64_t word encodes 64 horizontally consecutive cells in a row.
//
// The kernel processes the grid one 64-bit word per thread. For each word,
// the 8 neighbor directions are gathered into 8 bitboards, and a fully
// bit-parallel population count over those 8 bitboards is performed.
// This produces 4 bitplanes (N0..N3) encoding the per-cell neighbor count
// in binary for all 64 cells in the word simultaneously.
//
// Only global memory is used (no shared/texture memory) as per the problem
// constraints. The algorithm is designed to be compute-efficient and avoid
// per-cell loops, making full use of 64-bit logical operations on modern GPUs.

#include <cstdint>
#include <cuda_runtime.h>

// Bitwise 3-input adder used in a bit-sliced manner.
// For each bit position i, computes:
//   sum_i   = a_i ^ b_i ^ c_i        (LSB of a_i + b_i + c_i)
//   carry_i = majority(a_i, b_i, c_i) (MSB of a_i + b_i + c_i)
// Both 'sum' and 'carry' are 64-bit masks containing 64 independent bitwise sums.
__device__ __forceinline__ void bitwise_add3(std::uint64_t a,
                                             std::uint64_t b,
                                             std::uint64_t c,
                                             std::uint64_t &sum,
                                             std::uint64_t &carry)
{
    sum   = a ^ b ^ c;
    // Majority function: 1 if at least two of a,b,c are 1 at that bit position.
    carry = (a & b) | (a & c) | (b & c);
}

// Kernel computing a single Game of Life step on a square grid.
// - input/output: bit-packed grid, one uint64_t per 64 horizontal cells.
// - grid_dim: grid width and height (square), power of two > 512, and divisible by 64.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    const int words_per_row = grid_dim >> 6;  // grid_dim / 64
    const int total_words   = words_per_row * grid_dim;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Map linear word index to (row, column-word).
    const int row = idx / words_per_row;
    const int col = idx - row * words_per_row;

    const bool has_above = (row > 0);
    const bool has_below = (row + 1 < grid_dim);
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    const std::uint64_t* cur_row = input + static_cast<std::size_t>(row) * words_per_row;

    // Load current row words: left, center, right.
    std::uint64_t cur_left   = 0;
    std::uint64_t cur_center = cur_row[col];
    std::uint64_t cur_right  = 0;
    if (has_left)  cur_left  = cur_row[col - 1];
    if (has_right) cur_right = cur_row[col + 1];

    // Load above row words (if any).
    std::uint64_t above_left   = 0;
    std::uint64_t above_center = 0;
    std::uint64_t above_right  = 0;
    if (has_above) {
        const std::uint64_t* above_row = input + static_cast<std::size_t>(row - 1) * words_per_row;
        if (has_left)  above_left   = above_row[col - 1];
        above_center = above_row[col];
        if (has_right) above_right  = above_row[col + 1];
    }

    // Load below row words (if any).
    std::uint64_t below_left   = 0;
    std::uint64_t below_center = 0;
    std::uint64_t below_right  = 0;
    if (has_below) {
        const std::uint64_t* below_row = input + static_cast<std::size_t>(row + 1) * words_per_row;
        if (has_left)  below_left   = below_row[col - 1];
        below_center = below_row[col];
        if (has_right) below_right  = below_row[col + 1];
    }

    // Compute masks for neighbors in each of the 8 directions.
    //
    // Horizontal neighbors in the same row:
    // - same_left  bit i  = state of cell (row, x=i-1)  if inside row, or
    //                       (row, previous word bit 63) for i=0, otherwise 0 at row boundary.
    // - same_right bit i  = state of cell (row, x=i+1)  or from next word bit 0 at i=63.
    const std::uint64_t same_left  = (cur_center << 1) | (cur_left  >> 63);
    const std::uint64_t same_right = (cur_center >> 1) | (cur_right << 63);

    // Diagonal neighbors from the row above.
    const std::uint64_t above_left_shift  = (above_center << 1) | (above_left  >> 63); // NW
    const std::uint64_t above_right_shift = (above_center >> 1) | (above_right << 63); // NE

    // Diagonal neighbors from the row below.
    const std::uint64_t below_left_shift  = (below_center << 1) | (below_left  >> 63); // SW
    const std::uint64_t below_right_shift = (below_center >> 1) | (below_right << 63); // SE

    // Neighbor bitboards:
    // n0: W, n1: E, n2: N, n3: S, n4: NW, n5: NE, n6: SW, n7: SE.
    const std::uint64_t n0 = same_left;         // West
    const std::uint64_t n1 = same_right;        // East
    const std::uint64_t n2 = above_center;      // North
    const std::uint64_t n3 = below_center;      // South
    const std::uint64_t n4 = above_left_shift;  // North-West
    const std::uint64_t n5 = above_right_shift; // North-East
    const std::uint64_t n6 = below_left_shift;  // South-West
    const std::uint64_t n7 = below_right_shift; // South-East

    // Bit-parallel population count of the 8 neighbor bitboards.
    //
    // We compute for each bit position i the integer:
    //   neighbors_i = n0_i + n1_i + ... + n7_i,  (0..8)
    //
    // The result is held in 4 bitplanes N0..N3 such that:
    //   neighbors_i = N0_i + 2*N1_i + 4*N2_i + 8*N3_i.
    //
    // The logic below implements a small combinational adder tree using 3-input adders.

    // First, sum neighbors in three groups of three (last group has a zero input):
    // group A: n0 + n1 + n2
    // group B: n3 + n4 + n5
    // group C: n6 + n7 + 0
    std::uint64_t sA0, cA0;
    std::uint64_t sB0, cB0;
    std::uint64_t sC0, cC0;

    bitwise_add3(n0, n1, n2, sA0, cA0); // sA0: LSB, cA0: MSB of (n0+n1+n2)
    bitwise_add3(n3, n4, n5, sB0, cB0); // sB0: LSB, cB0: MSB of (n3+n4+n5)
    bitwise_add3(n6, n7, 0,  sC0, cC0); // sC0: LSB, cC0: MSB of (n6+n7)

    // Now combine the three partial sums:
    // (sA0 + sB0 + sC0) gives 1's place (and a carry representing 2's place).
    std::uint64_t l0, carryL;
    bitwise_add3(sA0, sB0, sC0, l0, carryL); // l0: bit0 (N0), carryL: intermediate for bit1

    // (cA0 + cB0 + cC0) are the carries from the previous stage (each representing 2).
    // We treat them as bits to be added, producing c0 (LSB) and c1 (MSB) of their sum.
    std::uint64_t c0, c1;
    bitwise_add3(cA0, cB0, cC0, c0, c1);

    // neighbors = (sA0 + sB0 + sC0) + 2*(cA0 + cB0 + cC0)
    //           =  l0 + 2*carryL + 2*c0 + 4*c1
    //
    // Let S1 = carryL + c0 = l1 + 2*h1 (two-bit sum per lane).
    const std::uint64_t l1 = carryL ^ c0;  // LSB of S1
    const std::uint64_t h1 = carryL & c0;  // MSB of S1

    // neighbors = l0 + 2*l1 + 4*(h1 + c1)
    // Let T = h1 + c1 = t0 + 2*t1 (two-bit sum per lane).
    const std::uint64_t t0 = h1 ^ c1; // LSB of T
    const std::uint64_t t1 = h1 & c1; // MSB of T

    // Final bitplanes representing the neighbor count:
    const std::uint64_t N0 = l0; // bit 0 (1's place)
    const std::uint64_t N1 = l1; // bit 1 (2's place)
    const std::uint64_t N2 = t0; // bit 2 (4's place)
    const std::uint64_t N3 = t1; // bit 3 (8's place)

    // Apply Game of Life rules:
    // next_cell_alive = (neighbors == 3) || (neighbors == 2 && current_cell_alive)
    //
    // neighbors == 2  -> binary 0010:  N3=0, N2=0, N1=1, N0=0
    // neighbors == 3  -> binary 0011:  N3=0, N2=0, N1=1, N0=1
    //
    // We first compute a mask 'base' for cells having neighbor count 2 or 3:
    //   base = ~N3 & ~N2 & N1
    // Then:
    //   eq3 = base & N0       (exactly 3 neighbors)
    //   eq2 = base & ~N0      (exactly 2 neighbors)
    //
    // Final next-state bitboard:
    //   next = eq3 | (eq2 & cur_center)

    const std::uint64_t notN3 = ~N3;
    const std::uint64_t notN2 = ~N2;
    const std::uint64_t tmp   = notN3 & notN2;
    const std::uint64_t base  = tmp & N1;

    const std::uint64_t eq3   = base & N0;
    const std::uint64_t eq2   = base & ~N0;

    const std::uint64_t stay_alive = eq2 & cur_center; // alive cells with exactly 2 neighbors
    const std::uint64_t born       = eq3;             // cells with exactly 3 neighbors
    const std::uint64_t next       = stay_alive | born;

    output[idx] = next;
}

// Host function to run a single Game of Life step on the GPU.
// - input:  pointer to device memory, bit-packed grid (std::uint64_t*).
// - output: pointer to device memory, bit-packed grid for next state.
// - grid_dimensions: width/height of the square grid (power of two, >512).
//
// The function configures and launches the CUDA kernel; it does not perform
// any host-device synchronization, which is left to the caller.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    const int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions);
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
