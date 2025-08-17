#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance, bit-parallel CUDA implementation of a single Conway's Game of Life step.

  Key points:
  - Bit-packed input: each std::uint64_t encodes 64 consecutive cells from the same row (bit 0 -> column 0).
  - Each CUDA thread processes exactly one 64-bit word (i.e., 64 cells) from the grid, avoiding atomics.
  - Outside-grid cells are treated as dead (0). Boundary threads zero out missing neighbor words.
  - Memory access: no shared/texture memory is used; coalesced global loads are sufficient on modern GPUs.
  - The kernel leverages bit-sliced addition with boolean operations to count neighbors in parallel for all 64 cells
    without propagating cross-bit carries, then applies the GoL rules to produce the next state mask.
  - The use of __popc/__popcll is generally helpful when counting bits in words, but for per-bit neighbor counts
    across 64 positions simultaneously, the bit-sliced approach below avoids serializing 64 independent popcounts.

  Algorithm overview (per 64-bit word):
  1) Load the 9 relevant words that form the 3x3 neighborhood at the word level:
     - For the current row: left (l), center (c), right (r)
     - For the upper row:   lu, cu, ru
     - For the lower row:   ld, cd, rd
     Outside the grid: use zeros.
  2) Build eight neighbor bitmasks using 1-bit funnel shifts across words:
     - Horizontal neighbors within the same row:
         ml = left neighbor bits  = (c << 1) | (l >> 63)
         mr = right neighbor bits = (c >> 1) | (r << 63)
     - Upper row neighbors:
         ul = (cu << 1) | (lu >> 63)
         uc = cu
         ur = (cu >> 1) | (ru << 63)
     - Lower row neighbors:
         dl = (cd << 1) | (ld >> 63)
         dc = cd
         dr = (cd >> 1) | (rd << 63)
     The special handling of bit 0 and bit 63 happens naturally by pulling in bit 63 from the left word for left-shifts
     and bit 0 from the right word for right-shifts.
  3) Accumulate neighbor counts (0..8) using bit-sliced addition (no cross-bit carries):
     For each row, sum its three (or two for the middle row) 1-bit inputs into two bitplanes:
       - ones (LSB of the 0..3 sum for that row)
       - twos (2's bit of the 0..3 sum for that row)
     Then combine the three rows:
       - Sum the ones from all rows -> produces ones_total (bit0) and a carry (co1) into the twos plane.
       - Sum the twos from all rows, then add co1 -> produces bit1 (twos) and a carry into the bit2 (fours) plane.
     Finally we have three bitplanes bit0, bit1, bit2 representing the total neighbor count per cell in binary.
  4) Apply GoL rules:
       - A cell is born if neighbor count == 3:    eq3 = (~bit2) & bit1 & bit0
       - A live cell survives if count == 2:       eq2 = (~bit2) & bit1 & ~bit0
       - Next = eq3 | (eq2 & current_state)
*/

static __device__ __forceinline__ std::uint64_t shl1_merge(const std::uint64_t c, const std::uint64_t l) {
    // Left neighbor mask: shift current word left by 1 and bring in bit 63 from the left word.
    return (c << 1) | (l >> 63);
}

static __device__ __forceinline__ std::uint64_t shr1_merge(const std::uint64_t c, const std::uint64_t r) {
    // Right neighbor mask: shift current word right by 1 and bring in bit 0 from the right word into bit 63.
    return (c >> 1) | (r << 63);
}

static __device__ __forceinline__ void sum3_1bit_masks(const std::uint64_t a,
                                                       const std::uint64_t b,
                                                       const std::uint64_t c,
                                                       std::uint64_t& ones,     // LSB of a+b+c (bitwise)
                                                       std::uint64_t& twos) {   // 2's bit of a+b+c (bitwise)
    // Compute ones = a ^ b ^ c
    // Compute twos = (a&b) | ((a^b)&c) == majority(a,b,c)
    const std::uint64_t t = a ^ b;
    ones = t ^ c;
    twos = (a & b) | (t & c);
}

static __global__ void life_step_kernel(const std::uint64_t* __restrict__ input,
                                        std::uint64_t* __restrict__ output,
                                        int words_per_row,
                                        int rows) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(rows);
    if (tid >= total_words) return;

    const int row = static_cast<int>(tid / words_per_row);
    const int col = static_cast<int>(tid - static_cast<std::size_t>(row) * words_per_row);

    const std::uint64_t* row_ptr = input + static_cast<std::size_t>(row) * words_per_row;

    // Load center row words; guard edges for left/right neighbors.
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < rows);

    // Center row words
    const std::uint64_t c  = row_ptr[col];
    const std::uint64_t l  = has_left  ? row_ptr[col - 1] : 0ull;
    const std::uint64_t r  = has_right ? row_ptr[col + 1] : 0ull;

    // Upper row words
    std::uint64_t lu = 0ull, cu = 0ull, ru = 0ull;
    if (has_up) {
        const std::uint64_t* up_ptr = row_ptr - words_per_row;
        cu = up_ptr[col];
        if (has_left)  lu = up_ptr[col - 1];
        if (has_right) ru = up_ptr[col + 1];
    }

    // Lower row words
    std::uint64_t ld = 0ull, cd = 0ull, rd = 0ull;
    if (has_down) {
        const std::uint64_t* down_ptr = row_ptr + words_per_row;
        cd = down_ptr[col];
        if (has_left)  ld = down_ptr[col - 1];
        if (has_right) rd = down_ptr[col + 1];
    }

    // Build neighbor masks using 1-bit funnel shifts across 64-bit words.
    const std::uint64_t ml = shl1_merge(c,  l);
    const std::uint64_t mr = shr1_merge(c,  r);

    const std::uint64_t ul = shl1_merge(cu, lu);
    const std::uint64_t uc = cu;
    const std::uint64_t ur = shr1_merge(cu, ru);

    const std::uint64_t dl = shl1_merge(cd, ld);
    const std::uint64_t dc = cd;
    const std::uint64_t dr = shr1_merge(cd, rd);

    // Sum neighbors within each row:
    // - Upper and lower rows: 3 inputs each -> (ones, twos)
    // - Middle row: 2 inputs   -> (ones, twos)
    std::uint64_t u1, u2;
    sum3_1bit_masks(ul, uc, ur, u1, u2);

    std::uint64_t d1, d2;
    sum3_1bit_masks(dl, dc, dr, d1, d2);

    const std::uint64_t m1 = ml ^ mr;  // ones of (ml + mr)
    const std::uint64_t m2 = ml & mr;  // twos of (ml + mr)

    // Sum ones across rows (u1 + m1 + d1) to produce:
    // - bit0 (ones) of total
    // - co1: carry into the twos plane (i.e., at least two of {u1,m1,d1} are set)
    const std::uint64_t t_um_ones = u1 ^ m1;
    const std::uint64_t bit0 = t_um_ones ^ d1;
    const std::uint64_t co1  = (u1 & m1) | (t_um_ones & d1);  // majority(u1, m1, d1)

    // Sum twos across rows (u2 + m2 + d2) -> (twos_lsb, twos_carry)
    const std::uint64_t t_um_twos   = u2 ^ m2;
    const std::uint64_t twos_lsb    = t_um_twos ^ d2;                        // parity among u2,m2,d2
    const std::uint64_t twos_carry  = (u2 & m2) | (t_um_twos & d2);          // majority(u2, m2, d2)

    // Add co1 (carry from ones) into the twos plane to form:
    // - bit1 (twos) of total: twos_lsb ^ co1
    // - bit2 (fours) of total: twos_carry | (twos_lsb & co1)
    const std::uint64_t bit1 = twos_lsb ^ co1;
    const std::uint64_t bit2 = twos_carry | (twos_lsb & co1);

    // Game of Life rules:
    // - Birth: neighbor count == 3 -> binary 011 -> (~bit2) & bit1 & bit0
    // - Survival (for alive cells): count == 2 -> binary 010 -> (~bit2) & bit1 & ~bit0
    const std::uint64_t not_bit2   = ~bit2;
    const std::uint64_t eq3_mask   = not_bit2 & bit1 & bit0;
    const std::uint64_t eq2_mask   = not_bit2 & bit1 & ~bit0;

    const std::uint64_t next = eq3_mask | (eq2_mask & c);

    output[tid] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is a power of two (> 512), so words_per_row = grid_dimensions / 64 is an integer.
    const int words_per_row = grid_dimensions >> 6;  // divide by 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Launch configuration: one thread per 64-bit word.
    // A block size of 256 to 512 is typically a good balance on modern GPUs.
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    life_step_kernel<<<grid_size, block_size>>>(input, output, words_per_row, grid_dimensions);
    // Synchronization (if any) is handled by the caller as per the problem statement.
}