#include <cuda_runtime.h>
#include <cstdint>

/*
Conway's Game of Life - CUDA implementation optimized for bit-packed grids.

Key points:
- Each 64-bit word encodes 64 horizontal cells (one bit per cell).
- Each CUDA thread processes one 64-bit word (64 cells) to avoid atomics.
- The kernel computes the next generation by counting 8-neighborhoods per bit-lane using bit-sliced arithmetic (no cross-bit carries).
- Horizontal neighbors across word boundaries are handled via shifts with carry-in from adjacent words (left/right).
- Boundary handling assumes cells outside the grid are dead; out-of-range words are treated as zero.

Bit-sliced neighbor counting:
- We compute eight 64-bit masks (one for each neighbor direction), then add them using a bit-plane ripple-carry adder (sum1, sum2, sum4, sum8).
- This yields the per-bit neighbor count (0..8) encoded across four bit-planes.
- Apply Life rules: next = (count == 3) | (alive & (count == 2)).
*/

static __device__ __forceinline__ std::uint64_t shift_left_with_carry(std::uint64_t curr, std::uint64_t prev) {
    // Shift left by 1; bring in prev's MSB (bit 63) into bit 0.
    return (curr << 1) | (prev >> 63);
}

static __device__ __forceinline__ std::uint64_t shift_right_with_carry(std::uint64_t curr, std::uint64_t next) {
    // Shift right by 1; bring in next's LSB (bit 0) into bit 63.
    return (curr >> 1) | (next << 63);
}

static __device__ __forceinline__ void add_bitplane_increment(std::uint64_t x,
                                                             std::uint64_t &s1,
                                                             std::uint64_t &s2,
                                                             std::uint64_t &s4,
                                                             std::uint64_t &s8) {
    // Ripple-carry addition of a 1-bit value 'x' into a 4-bit bit-sliced accumulator (s1,s2,s4,s8).
    // Each sK holds the K's-bit plane across 64 lanes (one bit per cell).
    std::uint64_t t = s1 ^ x;
    std::uint64_t c = s1 & x;
    s1 = t;

    t = s2 ^ c;
    c = s2 & c;
    s2 = t;

    t = s4 ^ c;
    c = s4 & c;
    s4 = t;

    // Since the maximum neighbor count is 8, there can be at most one carry into s8 per lane.
    s8 ^= c;
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ in,
                                         std::uint64_t* __restrict__ out,
                                         int grid_dim_cells,
                                         int words_per_row) {
    const int word_x = blockIdx.x * blockDim.x + threadIdx.x; // word column
    const int word_y = blockIdx.y * blockDim.y + threadIdx.y; // row index (in cells/words)

    if (word_x >= words_per_row || word_y >= grid_dim_cells) return;

    const int idx = word_y * words_per_row + word_x;

    // Load current row words (center row)
    const bool has_left  = (word_x > 0);
    const bool has_right = (word_x + 1 < words_per_row);
    const bool has_up    = (word_y > 0);
    const bool has_down  = (word_y + 1 < grid_dim_cells);

    // Current row
    const std::uint64_t c_prev = has_left  ? in[idx - 1] : 0ull;
    const std::uint64_t c_curr = in[idx];
    const std::uint64_t c_next = has_right ? in[idx + 1] : 0ull;

    // North row
    const int up_base = idx - words_per_row;
    const std::uint64_t n_prev = (has_up && has_left)  ? in[up_base - 1] : 0ull;
    const std::uint64_t n_curr =  has_up               ? in[up_base]     : 0ull;
    const std::uint64_t n_next = (has_up && has_right) ? in[up_base + 1] : 0ull;

    // South row
    const int dn_base = idx + words_per_row;
    const std::uint64_t s_prev = (has_down && has_left)  ? in[dn_base - 1] : 0ull;
    const std::uint64_t s_curr =  has_down               ? in[dn_base]     : 0ull;
    const std::uint64_t s_next = (has_down && has_right) ? in[dn_base + 1] : 0ull;

    // Build the 8 neighbor masks:
    // North row: left, center, right
    const std::uint64_t M0 = shift_left_with_carry(n_curr, n_prev);  // NW
    const std::uint64_t M1 = n_curr;                                 // N
    const std::uint64_t M2 = shift_right_with_carry(n_curr, n_next); // NE

    // Current row: left, right (center cell itself is not a neighbor)
    const std::uint64_t M3 = shift_left_with_carry(c_curr, c_prev);  // W
    const std::uint64_t M4 = shift_right_with_carry(c_curr, c_next); // E

    // South row: left, center, right
    const std::uint64_t M5 = shift_left_with_carry(s_curr, s_prev);  // SW
    const std::uint64_t M6 = s_curr;                                 // S
    const std::uint64_t M7 = shift_right_with_carry(s_curr, s_next); // SE

    // Accumulate neighbor counts across 64 lanes using a bit-sliced ripple-carry adder.
    std::uint64_t s1 = 0ull, s2 = 0ull, s4 = 0ull, s8 = 0ull;

    add_bitplane_increment(M0, s1, s2, s4, s8);
    add_bitplane_increment(M1, s1, s2, s4, s8);
    add_bitplane_increment(M2, s1, s2, s4, s8);
    add_bitplane_increment(M3, s1, s2, s4, s8);
    add_bitplane_increment(M4, s1, s2, s4, s8);
    add_bitplane_increment(M5, s1, s2, s4, s8);
    add_bitplane_increment(M6, s1, s2, s4, s8);
    add_bitplane_increment(M7, s1, s2, s4, s8);

    // Apply Game of Life rules using the bit-sliced neighbor count:
    // count == 3  => birth
    // count == 2  => survival if currently alive
    // s1 = bit0, s2 = bit1, s4 = bit2, s8 = bit3 (neighbor count)
    const std::uint64_t alive = c_curr;

    const std::uint64_t lt4_mask = ~s4 & ~s8;             // counts 0..3 only
    const std::uint64_t eq3_mask = lt4_mask & s2 &  s1;   // 0b0011
    const std::uint64_t eq2_mask = lt4_mask & s2 & ~s1;   // 0b0010

    const std::uint64_t next = eq3_mask | (alive & eq2_mask);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions: grid width/height in cells (power of 2, > 512).
    // Each row has words_per_row = grid_dimensions / 64 words.
    const int words_per_row = grid_dimensions >> 6;

    // Choose a 2D block for good occupancy and coalescing.
    // 32x8 = 256 threads per block; aligns with warp sizes and allows wide, coalesced x-dimension.
    dim3 block(32, 8, 1);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y,
              1);

    game_of_life_step_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
}