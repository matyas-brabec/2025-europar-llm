#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// Each 64-bit word encodes 64 consecutive cells in a row (bit i corresponds to cell x = word_index*64 + i).
// Each CUDA thread processes one 64-bit word (64 cells), using bitwise full-adder logic to count neighbors
// for all 64 cells in parallel. Boundary conditions are "dead beyond the edge" (no wrap-around).
//
// Key ideas:
// - Load up to 9 words: the current word (C), its left/right neighbors (L/R),
//   and the three words above (TL/T/TR) and below (BL/B/BR).
// - Construct 8 bitfields representing the 8 neighbor directions for each of the 64 cells by shifting and OR'ing.
//   For example, left neighbors for the current row are computed as (C << 1) | (L >> 63), which handles the
//   0th-bit special case by pulling bit 63 from the left word.
// - Use carry-save adder (CSA) logic to add the 8 neighbor bitfields efficiently:
//   - Sum the top triple (TL/T/TR) and bottom triple (BL/B/BR) using full-adders (sum = XOR, carry = majority).
//   - Sum the middle pair (left/right) using a half-adder.
//   - Combine the three row sums to obtain the final neighbor count per bit, represented by four bitplanes
//     (n1, n2, n4, n8), corresponding to binary weights 1, 2, 4, and 8.
// - Apply Game of Life rules using these bitplanes without converting to integers:
//   next = (count == 3) | (alive & (count == 2))

// Majority function for three 64-bit bitfields: returns 1 where at least two inputs are 1.
static __device__ __forceinline__ std::uint64_t majority3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    return (a & b) | (b & c) | (a & c);
}

// 3-input bitwise full-adder for 64 parallel 1-bit lanes.
// Outputs:
// - s: sum bit (LSB) of a + b + c (XOR of inputs).
// - c: carry bit (represents the 2's place; 1 where at least two inputs are 1).
static __device__ __forceinline__ void add3(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                            std::uint64_t& s, std::uint64_t& carry) {
    s = a ^ b ^ c;
    carry = majority3(a, b, c);
}

// Kernel that computes one timestep of Game of Life.
// Each thread processes a single 64-bit word (64 cells) from the input grid.
static __global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                                std::uint64_t* __restrict__ output,
                                                int grid_dim) {
    const int words_per_row = grid_dim >> 6; // grid_dim is multiple of 64 by assumption.
    const std::size_t total_words = static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    const int row = static_cast<int>(idx / words_per_row);
    const int widx = static_cast<int>(idx - static_cast<std::size_t>(row) * words_per_row);

    // Determine boundary flags.
    const bool has_top = (row > 0);
    const bool has_bottom = (row + 1 < grid_dim);
    const bool has_left = (widx > 0);
    const bool has_right = (widx + 1 < words_per_row);

    // Load the 3x3 neighborhood words. OOB loads are treated as zeros.
    std::uint64_t C = input[idx];

    std::uint64_t L = 0, R = 0, T = 0, B = 0, TL = 0, TR = 0, BL = 0, BR = 0;

    if (has_left) {
        L = input[idx - 1];
        if (has_top)    TL = input[idx - 1 - words_per_row];
        if (has_bottom) BL = input[idx - 1 + words_per_row];
    }
    if (has_right) {
        R = input[idx + 1];
        if (has_top)    TR = input[idx + 1 - words_per_row];
        if (has_bottom) BR = input[idx + 1 + words_per_row];
    }
    if (has_top) {
        T = input[idx - words_per_row];
    }
    if (has_bottom) {
        B = input[idx + words_per_row];
    }

    // Construct the 8 neighbor bitfields for the 64 cells in this word.
    // Horizontal neighbors (same row):
    const std::uint64_t nL  = (C << 1) | (L >> 63); // left neighbors (handles bit 0 via left word)
    const std::uint64_t nR  = (C >> 1) | (R << 63); // right neighbors (handles bit 63 via right word)

    // Top row neighbors:
    const std::uint64_t nT  = T;                                  // vertical top
    const std::uint64_t nTL = (T << 1) | (TL >> 63);              // top-left diagonal
    const std::uint64_t nTR = (T >> 1) | (TR << 63);              // top-right diagonal

    // Bottom row neighbors:
    const std::uint64_t nB  = B;                                  // vertical bottom
    const std::uint64_t nBL = (B << 1) | (BL >> 63);              // bottom-left diagonal
    const std::uint64_t nBR = (B >> 1) | (BR << 63);              // bottom-right diagonal

    // Use carry-save adder logic to sum the 8 neighbor bitfields per bit position.
    // Step 1: Sum top triple (nTL + nT + nTR) and bottom triple (nBL + nB + nBR) using full adders.
    std::uint64_t s_top, c_top;
    add3(nTL, nT, nTR, s_top, c_top); // s_top: 1's place; c_top: 2's place for top triple

    std::uint64_t s_bot, c_bot;
    add3(nBL, nB, nBR, s_bot, c_bot); // s_bot: 1's place; c_bot: 2's place for bottom triple

    // Step 2: Sum middle pair (nL + nR) using a half-adder.
    const std::uint64_t s_mid = nL ^ nR; // 1's place
    const std::uint64_t c_mid = nL & nR; // 2's place

    // Step 3: Combine top and bottom triples.
    // Add the 1's places: s_top + s_bot => produces ones_tb and a carry into the 2's place.
    const std::uint64_t ones_tb = s_top ^ s_bot;         // LSB after adding s_top and s_bot
    const std::uint64_t carry_to_twos_tb = s_top & s_bot; // carry from LSB into 2's place

    // Now add the 2's places: c_top + c_bot + carry_to_twos_tb.
    const std::uint64_t twos_tb = c_top ^ c_bot ^ carry_to_twos_tb; // resulting 2's bitplane after TB
    const std::uint64_t fours_tb = majority3(c_top, c_bot, carry_to_twos_tb); // carry into 4's place

    // Step 4: Add the middle pair (s_mid, c_mid) into the TB sum.
    // Add LSBs: ones_tb + s_mid
    const std::uint64_t ones = ones_tb ^ s_mid;            // final 1's bitplane
    const std::uint64_t carry_to_twos2 = ones_tb & s_mid;  // carry into 2's place from LSB addition

    // Add 2's places: twos_tb + c_mid + carry_to_twos2
    const std::uint64_t twos = twos_tb ^ c_mid ^ carry_to_twos2; // final 2's bitplane (n2)
    const std::uint64_t carry4_extra = majority3(twos_tb, c_mid, carry_to_twos2); // carry into 4's place

    // Combine 4's place contributions. If both are 1, that produces an 8's bit (max possible).
    const std::uint64_t n4 = fours_tb ^ carry4_extra;      // final 4's bitplane
    const std::uint64_t n8 = fours_tb & carry4_extra;      // final 8's bitplane (only when count == 8)

    // Apply Game of Life rules:
    // next = (count == 3) | (alive & (count == 2))
    // Where:
    // - count == 3 -> n8=0, n4=0, n2=1, n1=1
    // - count == 2 -> n8=0, n4=0, n2=1, n1=0
    const std::uint64_t not_n4 = ~n4;
    const std::uint64_t not_n8 = ~n8;

    const std::uint64_t eq3 = not_n8 & not_n4 & twos & ones;        // exactly 3 neighbors
    const std::uint64_t eq2 = not_n8 & not_n4 & twos & (~ones);     // exactly 2 neighbors

    const std::uint64_t next = eq3 | (eq2 & C);

    output[idx] = next;
}

// Host function to launch one step of the Game of Life.
// - input: device pointer to bit-packed input grid (N x N), 64 cells per 64-bit word.
// - output: device pointer to bit-packed output grid (same layout).
// - grid_dimensions: N (width and height), a power-of-two > 512.
//
// Note: Caller is responsible for any synchronization after the launch.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6; // N / 64
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Choose a reasonable block size for modern data center GPUs (A100/H100).
    // 256 is a good balance for bitwise-heavy kernels with moderate register pressure.
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    game_of_life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
