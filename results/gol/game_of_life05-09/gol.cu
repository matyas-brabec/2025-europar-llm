#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  - The grid is square with side length grid_dimensions (power of two > 512).
  - Each 64-bit word encodes 64 consecutive cells in a row (bit 0 = leftmost in the word).
  - Each CUDA thread processes exactly one 64-bit word (no atomics).
  - All cells outside the grid are dead (zero), which is enforced when handling boundary words/rows.
  - We compute the next generation using bit-parallel (SIMD within a register) operations.

  Core idea:
    - Construct the 8 neighbor bitboards (NW, N, NE, W, E, SW, S, SE) for the 64 cells packed in one word.
    - Sum these 8 one-bit-per-cell inputs using a carry-save adder (CSA) tree to obtain the per-cell neighbor count.
    - Apply the Life rule: next = (neighbors == 3) | (alive & (neighbors == 2)).

  The CSA-based reduction avoids cross-bit carries (each of the 64 lanes is independent) and processes all 64 cells at once.
  Horizontal neighbor contributions that cross 64-bit word boundaries are handled by explicitly loading left/right neighbor words
  and injecting their edge bits into the shifted masks:
    - West contribution: (row << 1) | (left_word >> 63)
    - East contribution: (row >> 1) | (right_word << 63)

  This correctly handles the 0th and 63rd bit special cases without branching for interior threads.
*/

using u64 = std::uint64_t;

// 3:2 Carry-Save Adder: reduces three input bitboards (bitwise) into a sum and a carry bitboard.
// For each bit position i, (a_i + b_i + c_i) = sum_i + 2 * carry_i.
static __forceinline__ __device__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// Compute one Life step for a single 64-bit word per thread.
static __global__ void game_of_life_kernel(const u64* __restrict__ in,
                                           u64* __restrict__ out,
                                           int words_per_row,
                                           int rows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // word index within the row
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x >= words_per_row || y >= rows) return;

    // Gather the 3x3 neighborhood at the granularity of 64-bit words.
    // Boundary conditions: any word outside the grid is treated as zero.
    const bool has_left  = (x > 0);
    const bool has_right = (x + 1 < words_per_row);
    const bool has_up    = (y > 0);
    const bool has_down  = (y + 1 < rows);

    const int idx      = y * words_per_row + x;
    const int idx_up   = (y - 1) * words_per_row + x;
    const int idx_down = (y + 1) * words_per_row + x;

    // Center row words
    const u64 ML = has_left  ? in[idx - 1] : 0ull;
    const u64 MC = in[idx];
    const u64 MR = has_right ? in[idx + 1] : 0ull;

    // Upper row words
    const u64 UL = (has_up && has_left)  ? in[idx_up - 1] : 0ull;
    const u64 UC =  has_up               ? in[idx_up]     : 0ull;
    const u64 UR = (has_up && has_right) ? in[idx_up + 1] : 0ull;

    // Lower row words
    const u64 DL = (has_down && has_left)  ? in[idx_down - 1] : 0ull;
    const u64 DC =  has_down               ? in[idx_down]     : 0ull;
    const u64 DR = (has_down && has_right) ? in[idx_down + 1] : 0ull;

    // Build the 8 neighbor bitboards. For each row, compute west (<<1 with carry-in from left word)
    // and east (>>1 with carry-in from right word) contributions.
    // This correctly accounts for bit 0 (needs left word's bit 63) and bit 63 (needs right word's bit 0).
    const u64 NW = (UC << 1) | (UL >> 63);      // Up row, west-shifted with left carry
    const u64 N  = UC;                          // Up row, centered
    const u64 NE = (UC >> 1) | (UR << 63);      // Up row, east-shifted with right carry

    const u64 W  = (MC << 1) | (ML >> 63);      // Mid row, west
    const u64 E  = (MC >> 1) | (MR << 63);      // Mid row, east

    const u64 SW = (DC << 1) | (DL >> 63);      // Down row, west
    const u64 S  = DC;                          // Down row, centered
    const u64 SE = (DC >> 1) | (DR << 63);      // Down row, east

    // Reduce 8 one-bit inputs per cell into a 4-bit neighbor count using a CSA tree.
    // Stage 1: group into threes (with one pair) to get partial sums and carries.
    u64 s0, c0; csa(NW, N,  NE, s0, c0);       // Top neighbors
    u64 s1, c1; csa(W,  E,  SW, s1, c1);       // Left/right + SW
    u64 s2, c2; csa(S,  SE, 0ull, s2, c2);     // Bottom neighbors (pair + zero)

    // Stage 2: sum the three "sum" bitboards, and the three "carry" bitboards.
    u64 s3, c3; csa(s0, s1, s2, s3, c3);       // s3: bit0 of neighbor count
    u64 s4, c4; csa(c0, c1, c2, s4, c4);       // carries from previous stage

    // Final bitplanes of the neighbor count per cell:
    // - bit0 = s3
    // - bit1 = c3 ^ s4
    // - bit2 = (c3 & s4) ^ c4
    // - bit3 = (c3 & s4) & c4
    const u64 nb0 = s3;
    const u64 nb1 = c3 ^ s4;
    const u64 nb2 = (c3 & s4) ^ c4;
    const u64 nb3 = (c3 & s4) & c4;

    // Apply Conway's rules in a vectorized manner:
    // next = (neighbors == 3) | (alive & (neighbors == 2))
    // neighbors == 2: 0b0010  => ~nb0 & nb1 & ~nb2 & ~nb3
    // neighbors == 3: 0b0011  =>  nb0 & nb1 & ~nb2 & ~nb3
    const u64 alive = MC;
    const u64 eq2   = (~nb0) & nb1 & (~nb2) & (~nb3);
    const u64 eq3   =  nb0  & nb1 & (~nb2) & (~nb3);
    const u64 next  = eq3 | (alive & eq2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the number of cells per side; words_per_row is the number of 64-bit words per row.
    const int rows = grid_dimensions;
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // Kernel launch configuration:
    // - Each thread processes one 64-bit word.
    // - We launch a 2D grid: X traverses words across the row, Y traverses rows.
    // Using 256 threads per block along X for good occupancy and coalesced memory access.
    const int block_x = 256;
    const int block_y = 1;
    dim3 block(block_x, block_y, 1);

    dim3 grid((words_per_row + block_x - 1) / block_x,
              (rows + block_y - 1) / block_y,
              1);

    game_of_life_kernel<<<grid, block>>>(input, output, words_per_row, rows);
}