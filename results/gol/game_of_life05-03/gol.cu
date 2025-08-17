#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// Each uint64_t encodes 64 contiguous cells of a row (1 bit per cell, LSB = leftmost).
// Each CUDA thread processes one uint64_t word, computing the next generation
// for its 64 cells by counting neighbors via bit-parallel operations.
//
// Implementation details:
// - The grid is square with dimensions that are powers of two and >= 512,
//   therefore divisible by 64. This guarantees an integer number of uint64_t per row.
// - Borders are zero-padded (cells outside the grid are treated as dead).
// - For per-thread work, we avoid atomics by having each thread handle exactly one word.
// - We compute all 64 cells in a word in parallel using bitwise operations.
// - Neighbor counts for all 64 bits are obtained using carry-save adders (CSAs)
//   applied to the eight neighbor bitboards simultaneously. This yields four bitplanes
//   (1's, 2's, 4's, 8's) per bit position, enabling exact comparisons for "==2" and "==3".
// - Horizontal neighbor propagation across 64-bit word boundaries uses cross-word
//   carry-in/out via shifts with injected bits from adjacent words (LSB/MSB).
//
// The kernel does not use shared or texture memory; global loads are coalesced,
// and arithmetic is done with plain 64-bit bitwise ops.

static __device__ __forceinline__ uint64_t shl1_with_carry(uint64_t x, uint64_t left_word) {
    // Shift-left by 1, inserting the MSB of the left neighbor word into bit 0.
    // For the first word in a row, left_word must be 0 to enforce zero-padding.
    return (x << 1) | (left_word >> 63);
}

static __device__ __forceinline__ uint64_t shr1_with_carry(uint64_t x, uint64_t right_word) {
    // Shift-right by 1, inserting the LSB of the right neighbor word into bit 63.
    // For the last word in a row, right_word must be 0 to enforce zero-padding.
    return (x >> 1) | (right_word << 63);
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int words_per_row,
                                    int nrows)
{
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(nrows);

    for (std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
         gid < total_words;
         gid += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        const int row = static_cast<int>(gid / words_per_row);
        const int col = static_cast<int>(gid % words_per_row);

        // Load the 3x3 neighborhood of words surrounding the target word.
        // Zero out-of-bounds neighbors to enforce dead cells outside the grid.
        const std::uint64_t C  = __ldg(in + gid);

        std::uint64_t Ww = 0, Ew = 0;
        if (col > 0)                  Ww = __ldg(in + (gid - 1));
        if (col + 1 < words_per_row)  Ew = __ldg(in + (gid + 1));

        std::uint64_t Nw = 0, N = 0, Ne = 0;
        if (row > 0) {
            const std::size_t baseN = gid - static_cast<std::size_t>(words_per_row);
            N = __ldg(in + baseN);
            if (col > 0)                 Nw = __ldg(in + (baseN - 1));
            if (col + 1 < words_per_row) Ne = __ldg(in + (baseN + 1));
        }

        std::uint64_t Sw = 0, S = 0, Se = 0;
        if (row + 1 < nrows) {
            const std::size_t baseS = gid + static_cast<std::size_t>(words_per_row);
            S = __ldg(in + baseS);
            if (col > 0)                 Sw = __ldg(in + (baseS - 1));
            if (col + 1 < words_per_row) Se = __ldg(in + (baseS + 1));
        }

        // Compute horizontally shifted neighbor bitboards with correct cross-word carry-in/out.
        const std::uint64_t N_left   = shl1_with_carry(N,  Nw);
        const std::uint64_t N_right  = shr1_with_carry(N,  Ne);
        const std::uint64_t C_left   = shl1_with_carry(C,  Ww);
        const std::uint64_t C_right  = shr1_with_carry(C,  Ew);
        const std::uint64_t S_left   = shl1_with_carry(S,  Sw);
        const std::uint64_t S_right  = shr1_with_carry(S,  Se);

        // Eight neighbor bitboards aligned with the 64 target cells.
        const std::uint64_t b0 = N_left;   // NW
        const std::uint64_t b1 = N;        // N
        const std::uint64_t b2 = N_right;  // NE
        const std::uint64_t b3 = C_left;   // W
        const std::uint64_t b4 = C_right;  // E
        const std::uint64_t b5 = S_left;   // SW
        const std::uint64_t b6 = S;        // S
        const std::uint64_t b7 = S_right;  // SE

        // Sum the eight 1-bit neighbor bitboards using carry-save adders (CSA).
        // Each CSA: sum = a ^ b ^ c, carry = (a & b) | (a & c) | (b & c).
        // Layer 1: three groups (3,3,2) -> sums (s1,s2,s3) and carries (c1,c2,c3).
        const std::uint64_t s1 = b0 ^ b1 ^ b2;
        const std::uint64_t c1 = (b0 & b1) | (b0 & b2) | (b1 & b2);

        const std::uint64_t s2 = b3 ^ b4 ^ b5;
        const std::uint64_t c2 = (b3 & b4) | (b3 & b5) | (b4 & b5);

        const std::uint64_t s3 = b6 ^ b7;      // third operand zero
        const std::uint64_t c3 = (b6 & b7);    // third operand zero

        // Layer 2: sum the partial sums and partial carries.
        const std::uint64_t s4 = s1 ^ s2 ^ s3;                                       // 1's bitplane
        const std::uint64_t c4 = (s1 & s2) | (s1 & s3) | (s2 & s3);                  // weight-2
        const std::uint64_t s5 = c1 ^ c2 ^ c3;                                       // weight-2
        const std::uint64_t c5 = (c1 & c2) | (c1 & c3) | (c2 & c3);                  // weight-4

        // Combine weight-2 planes (s5, c4) and propagate to weight-4; include c5 (weight-4).
        const std::uint64_t ones  = s4;                  // 1's bit of neighbor count
        const std::uint64_t twos  = s5 ^ c4;             // 2's bit of neighbor count
        const std::uint64_t carry4_from_2 = s5 & c4;     // carry to 4's place from combining 2's
        const std::uint64_t fours = c5 ^ carry4_from_2;  // 4's bit of neighbor count
        const std::uint64_t eights = c5 & carry4_from_2; // 8's bit (only when count == 8)

        // Detect counts equal to 2 or 3 using the bitplanes.
        // count == 3: ones=1, twos=1, fours=0, eights=0
        // count == 2: ones=0, twos=1, fours=0, eights=0
        const std::uint64_t no_high = ~(fours | eights);
        const std::uint64_t eq3 = ones & twos & no_high;
        const std::uint64_t eq2 = (~ones) & twos & no_high;

        // Game of Life next state:
        // - Births: eq3
        // - Survivals: current & eq2 (alive with exactly 2 neighbors)
        const std::uint64_t next = eq3 | (C & eq2);

        out[gid] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is a power of 2 and >= 512, hence divisible by 64.
    const int words_per_row = grid_dimensions / 64;
    const int nrows = grid_dimensions;
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(nrows);

    // Launch configuration: use a reasonable block size and enough blocks,
    // relying on a grid-stride loop for scalability.
    const int threads_per_block = 256;
    unsigned int blocks = static_cast<unsigned int>((total_words + threads_per_block - 1) / threads_per_block);
    if (blocks == 0) blocks = 1;
    // Cap blocks to a safe 1D grid limit for broad device compatibility.
    if (blocks > 65535u) blocks = 65535u;

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, words_per_row, nrows);
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
