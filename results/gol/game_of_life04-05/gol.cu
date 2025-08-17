#include <cuda_runtime.h>
#include <cstdint>

// This implementation uses a "broadword" (bit-sliced) approach where each thread processes one 64-bit word,
// i.e., 64 cells in a row. It avoids per-bit loops and atomics. For each word, the thread loads the 9 words
// from the 3x3 neighborhood (3 from the row above, 3 from the current row, 3 from the row below), with zero
// padding beyond grid boundaries. It then constructs eight neighbor masks (UL, U, UR, L, R, DL, D, DR) using
// cross-word shifts with carries to handle bit 0 and bit 63. The eight neighbor masks are summed using a
// carry-save adder (CSA) tree to produce three bitplanes representing the neighbor count bits (1, 2, and 4).
// Finally, the next generation is computed as: new = (count == 3) | (alive & (count == 2)).
// No shared/texture memory is used; global loads are coalesced for interior threads.

static __device__ __forceinline__ uint64_t shift_left_with_carry(uint64_t center, uint64_t left_word) {
    // For each bit i (0..63), produce center[i-1]; for i==0, bring in left_word[63].
    return (center << 1) | (left_word >> 63);
}

static __device__ __forceinline__ uint64_t shift_right_with_carry(uint64_t center, uint64_t right_word) {
    // For each bit i (0..63), produce center[i+1]; for i==63, bring in right_word[0].
    return (center >> 1) | ((right_word & 1ull) << 63);
}

static __device__ __forceinline__ void csa(uint64_t a, uint64_t b, uint64_t c, uint64_t& s, uint64_t& carry) {
    // Carry-Save Adder for three 64-bit bitmasks:
    // s = a ^ b ^ c (sum bit, weight 1)
    // carry = (a&b) | (c&(a^b)) (carry bit, weight 2)
    uint64_t ab = a ^ b;
    s = ab ^ c;
    carry = (a & b) | (c & ab);
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int grid_dim, int words_per_row) {
    const size_t total_words = static_cast<size_t>(grid_dim) * static_cast<size_t>(words_per_row);
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;

    // Map linear index to (y, x_word)
    const int y = static_cast<int>(tid / words_per_row);
    const int x = static_cast<int>(tid - static_cast<size_t>(y) * words_per_row);

    const bool has_left  = (x > 0);
    const bool has_right = (x + 1 < words_per_row);
    const bool has_up    = (y > 0);
    const bool has_down  = (y + 1 < grid_dim);

    const size_t row_base     = static_cast<size_t>(y) * words_per_row;
    const size_t row_up_base  = has_up   ? static_cast<size_t>(y - 1) * words_per_row : 0;
    const size_t row_dn_base  = has_down ? static_cast<size_t>(y + 1) * words_per_row : 0;

    // Load 3x3 neighborhood words with zero padding beyond boundaries
    uint64_t upL = 0, upC = 0, upR = 0;
    uint64_t mdL = 0, mdC = 0, mdR = 0;
    uint64_t dnL = 0, dnC = 0, dnR = 0;

    // Current row
    mdC = input[row_base + x];
    if (has_left)  mdL = input[row_base + (x - 1)];
    if (has_right) mdR = input[row_base + (x + 1)];

    // Row above
    if (has_up) {
        upC = input[row_up_base + x];
        if (has_left)  upL = input[row_up_base + (x - 1)];
        if (has_right) upR = input[row_up_base + (x + 1)];
    }

    // Row below
    if (has_down) {
        dnC = input[row_dn_base + x];
        if (has_left)  dnL = input[row_dn_base + (x - 1)];
        if (has_right) dnR = input[row_dn_base + (x + 1)];
    }

    // Construct the eight neighbor direction masks using cross-word shifts with carries.
    // Note that the center cell (mdC) itself is NOT included in these neighbor masks.
    const uint64_t UL = shift_left_with_carry(upC, upL);
    const uint64_t U  = upC;
    const uint64_t UR = shift_right_with_carry(upC, upR);

    const uint64_t L  = shift_left_with_carry(mdC, mdL);
    const uint64_t R  = shift_right_with_carry(mdC, mdR);

    const uint64_t DL = shift_left_with_carry(dnC, dnL);
    const uint64_t D  = dnC;
    const uint64_t DR = shift_right_with_carry(dnC, dnR);

    // Sum the 8 neighbor masks using a CSA reduction tree to obtain three bitplanes:
    // ones (weight 1), twos (weight 2), and fours (weight 4).
    uint64_t s1, c1;
    csa(UL, U, UR, s1, c1);

    uint64_t s2, c2;
    csa(L, R, DL, s2, c2);

    uint64_t s3, c3;
    csa(D, DR, 0ull, s3, c3);

    // Combine partial sums
    uint64_t s4, c4;
    csa(s1, s2, s3, s4, c4);     // s4: ones bitplane, c4: carries (weight 2)

    uint64_t c5, c6;
    csa(c1, c2, c3, c5, c6);     // c5: twos (partial), c6: fours

    // Final bitplanes:
    // ones = s4 (weight 1)
    // twos = c4 XOR c5 (weight 2)
    // fours = c6 OR (c4 & c5) (weight 4)  -> captures carry from adding c4 + c5
    const uint64_t ones  = s4;
    const uint64_t twos  = c4 ^ c5;
    const uint64_t fours = c6 | (c4 & c5);

    // Apply Game of Life rules:
    // - A cell becomes alive if neighbor count == 3
    // - A cell stays alive if it is currently alive and neighbor count == 2
    // count == 3 => fours == 0, twos == 1, ones == 1
    // count == 2 => fours == 0, twos == 1, ones == 0
    const uint64_t alive = mdC;
    const uint64_t eq3 = (~fours) & twos &  ones;
    const uint64_t eq2 = (~fours) & twos & ~ones;
    const uint64_t next = eq3 | (alive & eq2);

    output[row_base + x] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the side length of the square grid (power of 2, > 512).
    // Each row has words_per_row 64-bit words (bit-packed 64 cells per word).
    const int words_per_row = grid_dimensions >> 6;  // divide by 64
    const size_t total_words = static_cast<size_t>(grid_dimensions) * static_cast<size_t>(words_per_row);

    // Launch 1D grid where each thread processes one 64-bit word.
    const int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    game_of_life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row);
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
