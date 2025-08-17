#include <cstdint>
#include <cuda_runtime.h>

// Conway's Game of Life CUDA implementation with bit-packed rows (64 cells per 64-bit word).
// - Each CUDA thread computes the next state for one 64-bit word at a time.
// - Bits are aligned so that bit i in a word corresponds to the i-th cell in that 64-cell chunk of the row.
// - Neighbor counts are computed with bit-parallel logic using only integer operations (no shared/texture memory).
// - Out-of-bounds cells (outside the grid) are treated as dead, handled by zero-padding when loading neighbor words.

// Helper aliases
using u64 = std::uint64_t;

// Shift helpers that perform a logical shift by 1 while bringing in the cross-word bit from the adjacent word.
// These align neighbor contributions into the current 64-bit word's bit positions.
static __forceinline__ __device__ u64 shift_left1_aligned(u64 x, u64 prev) {
    // Logical left shift; fill bit 0 with previous word's MSB (bit 63).
    return (x << 1) | (prev >> 63);
}

static __forceinline__ __device__ u64 shift_right1_aligned(u64 x, u64 next) {
    // Logical right shift; fill bit 63 with next word's LSB (bit 0).
    return (x >> 1) | (next << 63);
}

// Sum of three 1-bit-per-position bitboards:
// Produces per-bit "ones" (LSB of the sum) and "twos" (2's bit of the sum) for values in {0..3}.
// For inputs a,b,c, per bit the sum s = a + b + c = ones + 2*twos.
static __forceinline__ __device__ void sum3(u64 a, u64 b, u64 c, u64 &ones, u64 &twos) {
    u64 ab_xor = a ^ b;
    ones = ab_xor ^ c;
    // Majority-of-three for the 2's place:
    // twos = (a & b) | (a & c) | (b & c)
    u64 ab = a & b;
    u64 ac = a & c;
    u64 bc = b & c;
    twos = (ab | ac) | bc;
}

// Kernel that computes one generation step for a bit-packed square grid.
static __global__ void life_kernel(const u64* __restrict__ input,
                                   u64* __restrict__ output,
                                   int words_per_row,
                                   int height,
                                   std::size_t total_words)
{
    const std::size_t stride = blockDim.x * (std::size_t)gridDim.x;
    for (std::size_t idx = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; idx < total_words; idx += stride) {
        // Map linear index to (row, col_word)
        int row = static_cast<int>(idx / (std::size_t)words_per_row);
        int col = static_cast<int>(idx - (std::size_t)row * (std::size_t)words_per_row);

        // Boundary flags
        const bool have_left  = (col > 0);
        const bool have_right = (col + 1 < words_per_row);
        const bool have_up    = (row > 0);
        const bool have_down  = (row + 1 < height);

        // Load current row words
        const u64 C      = input[idx];
        const u64 C_prev = have_left  ? input[idx - 1] : 0ull;
        const u64 C_next = have_right ? input[idx + 1] : 0ull;

        // Load upper row words (or zero if out-of-bounds)
        u64 U = 0ull, U_prev = 0ull, U_next = 0ull;
        if (have_up) {
            const std::size_t up_idx = idx - (std::size_t)words_per_row;
            U      = input[up_idx];
            U_prev = have_left  ? input[up_idx - 1] : 0ull;
            U_next = have_right ? input[up_idx + 1] : 0ull;
        }

        // Load lower row words (or zero if out-of-bounds)
        u64 D = 0ull, D_prev = 0ull, D_next = 0ull;
        if (have_down) {
            const std::size_t dn_idx = idx + (std::size_t)words_per_row;
            D      = input[dn_idx];
            D_prev = have_left  ? input[dn_idx - 1] : 0ull;
            D_next = have_right ? input[dn_idx + 1] : 0ull;
        }

        // Build neighbor masks aligned to the current word:
        // Upper row: up-left, up, up-right
        const u64 UL = shift_left1_aligned(U, U_prev);
        const u64 UC = U;
        const u64 UR = shift_right1_aligned(U, U_next);

        // Current row: left, right (note: the center cell itself is NOT a neighbor)
        const u64 CL = shift_left1_aligned(C, C_prev);
        const u64 CR = shift_right1_aligned(C, C_next);

        // Lower row: down-left, down, down-right
        const u64 DL = shift_left1_aligned(D, D_prev);
        const u64 DC = D;
        const u64 DR = shift_right1_aligned(D, D_next);

        // Sum contributions per row using bitwise 3-input adders.
        // Upper row (three inputs)
        u64 ones_u, twos_u;
        sum3(UL, UC, UR, ones_u, twos_u);

        // Middle row (two inputs -> ones = XOR, twos = AND)
        const u64 ones_m = CL ^ CR;
        const u64 twos_m = CL & CR;

        // Lower row (three inputs)
        u64 ones_d, twos_d;
        sum3(DL, DC, DR, ones_d, twos_d);

        // Combine the three row partial sums to get total neighbor count bits:
        // Let S = (ones_u + ones_m + ones_d) + 2*(twos_u + twos_m + twos_d)
        // Compute S's binary representation per bit: S1 (1's), S2 (2's), S4 (4's), S8 (8's).
        // First, sum the ones parts.
        const u64 ones_sum_ones = ones_u ^ ones_m ^ ones_d; // contributes to S1
        const u64 ones_sum_twos = (ones_u & ones_m) | (ones_u & ones_d) | (ones_m & ones_d); // contributes to S2

        // Then, sum the twos parts (note: each twos_* contributes weight 2).
        const u64 twos_sum_lsb   = twos_u ^ twos_m ^ twos_d; // contributes to S2
        const u64 twos_sum_carry = (twos_u & twos_m) | (twos_u & twos_d) | (twos_m & twos_d); // contributes to S4

        // Combine S2 contributions and propagate carry into S4.
        const u64 S1 = ones_sum_ones;
        const u64 S2 = ones_sum_twos ^ twos_sum_lsb;
        const u64 carry_to_S4 = ones_sum_twos & twos_sum_lsb;

        const u64 S4 = twos_sum_carry ^ carry_to_S4;
        const u64 S8 = twos_sum_carry & carry_to_S4;

        // Apply Game of Life rules:
        // next = (neighbors == 3) | (alive & neighbors == 2)
        // Translate: eq3 = S8==0 & S4==0 & S2==1 & S1==1
        //            eq2 = S8==0 & S4==0 & S2==1 & S1==0
        const u64 no_high = ~(S4 | S8);
        const u64 eq3 = no_high & S2 & S1;
        const u64 eq2 = no_high & S2 & ~S1;

        const u64 next = eq3 | (C & eq2);

        // Store next state
        output[idx] = next;
    }
}

// Host function to launch one Game of Life step.
// - input/output: device pointers (allocated with cudaMalloc) to bit-packed grids.
// - grid_dimensions: width = height of the square grid in cells; must be a power of two; > 512.
// - Each 64-bit word packs 64 consecutive cells in a row (LSB = leftmost within the word).
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Words per row is width/64 (grid_dimensions is a power-of-two > 512, so divisible by 64).
    const int words_per_row = grid_dimensions >> 6;
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Launch configuration: 1D grid-stride loop for simplicity and scalability.
    // Use a moderate block size to balance occupancy and register pressure.
    constexpr int THREADS_PER_BLOCK = 256;
    // Cap blocks to avoid excessive grid dimensions while allowing grid-stride coverage.
    int blocks = static_cast<int>((total_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (blocks > 65535) blocks = 65535;

    life_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, words_per_row, grid_dimensions, total_words);
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
