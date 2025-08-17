#include <cstdint>
#include <cuda_runtime.h>

/*
  CUDA Conway's Game of Life - bit-parallel implementation

  - The grid is square of size N x N, N is a power of two >= 512.
  - Each 64-bit word encodes 64 horizontal cells (LSB = leftmost bit in the word).
  - Each CUDA thread processes one 64-bit word, i.e., 64 cells at once, to avoid atomics.

  Neighbor counting:
  - Build eight 64-bit masks (UL, U, UR, L, R, DL, D, DR) representing the 8-neighborhood.
  - Horizontal neighbor masks (L, R) are computed with 1-bit shifts plus cross-word carries from the left/right words.
  - Diagonals are formed from rows above/below via shifts with carries from adjacent words on those rows.
  - Outside-grid neighbors are treated as zero (dead).

  Full-adder / carry-save adder logic:
  - Use carry-save adders (CSA) to sum the eight 1-bit neighbor planes without carry propagation across bit positions.
  - Compress 8 inputs into bit-planes for 1, 2, 4, and 8 counts (ones, twos, fours, eights).
  - From these, compute masks for "exactly 2 neighbors" and "exactly 3 neighbors":
      eq3 = (~eights) & (~fours) & twos & ones
      eq2 = (~eights) & (~fours) & twos & (~ones)
  - Next state:
      next = eq3 | (eq2 & current)
*/

static __device__ __forceinline__ void csa_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                               std::uint64_t &sum, std::uint64_t &carry) {
    // Bitwise full adder for three operands:
    // sum   = a ^ b ^ c
    // carry = majority(a,b,c) = (a&b) | (b&c) | (a&c)
    sum   = a ^ b ^ c;
    carry = (a & b) | (b & c) | (a & c);
}

static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t x, std::uint64_t carry_in_bit) {
    // Shift left by 1, inject LSB from carry_in_bit (must be 0 or 1)
    return (x << 1) | (carry_in_bit & 1ull);
}

static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t x, std::uint64_t carry_in_bit) {
    // Shift right by 1, inject MSB from carry_in_bit (must be 0 or 1)
    return (x >> 1) | ((carry_in_bit & 1ull) << 63);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int grid_dim) {
    // 2D thread indexing: x = word-column within a row, y = row index
    const int words_per_row = grid_dim >> 6; // grid_dim / 64

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= words_per_row || row >= grid_dim) return;

    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(words_per_row);
    const size_t idx     = row_off + col;

    // Load current word
    const std::uint64_t W = in[idx];

    // Load horizontal neighbor words on the same row for cross-word carries
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    const std::uint64_t Wl = has_left  ? in[row_off + (col - 1)] : 0ull;
    const std::uint64_t Wr = has_right ? in[row_off + (col + 1)] : 0ull;

    // Load vertical rows: U (row-1) and D (row+1)
    const bool has_up   = (row > 0);
    const bool has_down = (row + 1 < grid_dim);

    const size_t up_off   = has_up   ? (static_cast<size_t>(row - 1) * static_cast<size_t>(words_per_row)) : 0ull;
    const size_t down_off = has_down ? (static_cast<size_t>(row + 1) * static_cast<size_t>(words_per_row)) : 0ull;

    const std::uint64_t U  = has_up   ? in[up_off   + col] : 0ull;
    const std::uint64_t D  = has_down ? in[down_off + col] : 0ull;

    // Load adjacent words on up/down rows for diagonal carries
    const std::uint64_t U_l = (has_up   && has_left)  ? in[up_off   + (col - 1)] : 0ull;
    const std::uint64_t U_r = (has_up   && has_right) ? in[up_off   + (col + 1)] : 0ull;
    const std::uint64_t D_l = (has_down && has_left)  ? in[down_off + (col - 1)] : 0ull;
    const std::uint64_t D_r = (has_down && has_right) ? in[down_off + (col + 1)] : 0ull;

    // Cross-word carry bits for 1-bit shifts
    const std::uint64_t carry_L   = Wl >> 63;       // to fill bit0 when shifting left
    const std::uint64_t carry_R   = Wr & 1ull;      // to fill bit63 when shifting right
    const std::uint64_t carry_UL  = U_l >> 63;
    const std::uint64_t carry_UR  = U_r & 1ull;
    const std::uint64_t carry_DL  = D_l >> 63;
    const std::uint64_t carry_DR  = D_r & 1ull;

    // Build the eight neighbor bitmasks
    const std::uint64_t L  = shl1_with_carry(W, carry_L);
    const std::uint64_t R  = shr1_with_carry(W, carry_R);
    const std::uint64_t UL = shl1_with_carry(U, carry_UL);
    const std::uint64_t UR = shr1_with_carry(U, carry_UR);
    const std::uint64_t DL = shl1_with_carry(D, carry_DL);
    const std::uint64_t DR = shr1_with_carry(D, carry_DR);
    const std::uint64_t UU = U;
    const std::uint64_t DD = D;

    // Sum the eight neighbor planes using carry-save adders.
    // First layer: three CSAs to compress into sums of weight1 and carries of weight2
    std::uint64_t s01, c01; csa_u64(UL, UU, UR, s01, c01); // weight1 sum, weight2 carry
    std::uint64_t s02, c02; csa_u64(L,  R,  DD, s02, c02); // note: DD is 'D' vertical neighbor
    std::uint64_t s03, c03; csa_u64(DL, DR, 0ull, s03, c03);

    // Second layer: combine the three weight1 sums -> s1 (weight1), k1 (weight2)
    std::uint64_t s1, k1; csa_u64(s01, s02, s03, s1, k1);

    // Combine all weight2 terms: c01, c02, c03, k1 -> produce s2 (weight2) and s4/eight via carries
    std::uint64_t b1, d1; csa_u64(c01, c02, c03, b1, d1);  // b1 weight2, d1 weight4
    const std::uint64_t s2          = b1 ^ k1;             // weight2 sum
    const std::uint64_t carry4_extra = b1 & k1;            // each 1 here represents an extra +4
    const std::uint64_t s4_pre      = d1 ^ carry4_extra;   // weight4 sum bits
    const std::uint64_t s8          = d1 & carry4_extra;   // weight8 bits (only when 8 neighbors)

    // Evaluate exactly-2 and exactly-3 neighbor masks from the bit-planes
    const std::uint64_t not_s4  = ~s4_pre;
    const std::uint64_t not_s8  = ~s8;
    const std::uint64_t eq3 = (not_s8 & not_s4) & s2 & s1;
    const std::uint64_t eq2 = (not_s8 & not_s4) & s2 & (~s1);

    // Next state: alive iff (neighbors == 3) or (neighbors == 2 and currently alive)
    out[idx] = eq3 | (eq2 & W);
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Choose a 2D launch configuration to avoid expensive division/modulo per thread.
    // Each thread processes one 64-bit word. X dimension spans words within a row, Y dimension spans rows.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Configure block dimensions aiming for good occupancy and coalesced access.
    // Use up to 1024 threads per block; prefer wide X for contiguous global loads.
    int block_x = 256;
    if (words_per_row < block_x) block_x = words_per_row > 0 ? words_per_row : 1;
    int block_y = 1024 / block_x; // keep total threads per block <= 1024
    if (block_y < 1) block_y = 1;

    dim3 block(block_x, block_y, 1);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y,
              1);

    gol_step_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // No cudaDeviceSynchronize() here; caller is responsible for synchronization if needed.
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
