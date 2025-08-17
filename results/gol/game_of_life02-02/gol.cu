#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance, bit-parallel Conway's Game of Life step for a square grid.

  Encoding and assumptions:
  - The grid is stored in row-major order, bit-packed: each 64-bit word holds 64 consecutive cells within the same row.
  - Bit i (0..63) within a word corresponds to column (word_index * 64 + i). We treat bit 0 as the leftmost column in that 64-cell block.
  - New cells outside the grid are considered dead (0). We explicitly handle boundary rows and word boundaries.
  - Grid dimensions are a power of two, > 512, and small enough to fit in GPU memory. Width == height == grid_dimensions.

  Algorithm overview:
  - Each thread computes the next generation for one or more 64-bit words using a grid-stride loop.
  - For each 64-bit word (center block), load its neighbors from the same row and from the row above/below.
  - Construct the eight neighbor bitboards aligned to the center cells using 64-bit shifts with cross-word carry from adjacent words.
  - Use bit-parallel binary counters (three bitplanes s1, s2, s4) to accumulate the per-cell neighbor count modulo 8.
    This is sufficient to derive "exactly 2" and "exactly 3" masks without computing the full count (values 0..8).
  - Apply Game of Life rules using bitwise operations:
      next = (neighbors == 3) | (alive & (neighbors == 2))
  - Store the result in the output array.

  Notes on performance:
  - No shared or texture memory, as requested. All operations are register and global-memory based.
  - Loads are coalesced when threads process consecutive words.
  - The neighbor-count accumulation uses only AND/XOR operations with no branches in the hot path, except for boundary handling.
*/

static __device__ __forceinline__ std::uint64_t shift_left_with_neighbor(std::uint64_t v, std::uint64_t v_left) {
    // Shift left by 1 across 64-bit word boundaries.
    // Bit 0 receives the MSB of the left neighbor word.
    return (v << 1) | (v_left >> 63);
}

static __device__ __forceinline__ std::uint64_t shift_right_with_neighbor(std::uint64_t v, std::uint64_t v_right) {
    // Shift right by 1 across 64-bit word boundaries.
    // Bit 63 receives the LSB of the right neighbor word.
    return (v >> 1) | (v_right << 63);
}

static __device__ __forceinline__ void add_bitboard_to_counters(std::uint64_t bits,
                                                                std::uint64_t& s1,
                                                                std::uint64_t& s2,
                                                                std::uint64_t& s4) {
    // Increment the bitwise 3-bit counters (s1:1's place, s2:2's place, s4:4's place) by a 1-bit-per-cell 'bits' vector.
    // This performs per-bit binary addition, discarding overflow beyond 7 (s8), which is fine since we only need ==2 or ==3.
    std::uint64_t c1 = s1 & bits;
    s1 ^= bits;
    std::uint64_t c2 = s2 & c1;
    s2 ^= c1;
    s4 ^= c2;
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    for (std::size_t global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         global_idx < total_words;
         global_idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {

        // Compute row and word-in-row indices
        const int row = static_cast<int>(global_idx / words_per_row);
        const int col_word = static_cast<int>(global_idx - static_cast<std::size_t>(row) * words_per_row);

        // Determine boundary conditions
        const bool has_top = (row > 0);
        const bool has_bottom = (row + 1 < grid_dimensions);
        const bool has_left = (col_word > 0);
        const bool has_right = (col_word + 1 < words_per_row);

        // Load current row words
        const std::uint64_t mC = input[global_idx];
        const std::uint64_t mL = has_left  ? input[global_idx - 1] : 0ull;
        const std::uint64_t mR = has_right ? input[global_idx + 1] : 0ull;

        // Load top row words
        const std::size_t top_base = static_cast<std::size_t>(has_top ? (global_idx - words_per_row) : global_idx);
        const std::uint64_t tC = has_top ? input[top_base] : 0ull;
        const std::uint64_t tL = (has_top && has_left)  ? input[top_base - 1] : 0ull;
        const std::uint64_t tR = (has_top && has_right) ? input[top_base + 1] : 0ull;

        // Load bottom row words
        const std::size_t bot_base = static_cast<std::size_t>(has_bottom ? (global_idx + words_per_row) : global_idx);
        const std::uint64_t bC = has_bottom ? input[bot_base] : 0ull;
        const std::uint64_t bL = (has_bottom && has_left)  ? input[bot_base - 1] : 0ull;
        const std::uint64_t bR = (has_bottom && has_right) ? input[bot_base + 1] : 0ull;

        // Build neighbor bitboards aligned to the center cells.
        // Horizontal neighbors in the same row:
        const std::uint64_t nML = shift_left_with_neighbor(mC, mL);   // middle-left
        const std::uint64_t nMR = shift_right_with_neighbor(mC, mR);  // middle-right

        // Top row neighbors:
        const std::uint64_t nTC = tC;                                  // top
        const std::uint64_t nTL = shift_left_with_neighbor(tC, tL);    // top-left
        const std::uint64_t nTR = shift_right_with_neighbor(tC, tR);   // top-right

        // Bottom row neighbors:
        const std::uint64_t nBC = bC;                                  // bottom
        const std::uint64_t nBL = shift_left_with_neighbor(bC, bL);    // bottom-left
        const std::uint64_t nBR = shift_right_with_neighbor(bC, bR);   // bottom-right

        // Accumulate neighbor counts (mod 8) into three bitplanes s1 (1's), s2 (2's), s4 (4's).
        std::uint64_t s1 = 0ull, s2 = 0ull, s4 = 0ull;

        // Unrolled accumulation for the eight neighbors
        add_bitboard_to_counters(nTL, s1, s2, s4);
        add_bitboard_to_counters(nTC, s1, s2, s4);
        add_bitboard_to_counters(nTR, s1, s2, s4);
        add_bitboard_to_counters(nML, s1, s2, s4);
        add_bitboard_to_counters(nMR, s1, s2, s4);
        add_bitboard_to_counters(nBL, s1, s2, s4);
        add_bitboard_to_counters(nBC, s1, s2, s4);
        add_bitboard_to_counters(nBR, s1, s2, s4);

        // Derive masks for exactly-2 and exactly-3 neighbors using the bitplanes:
        // count == 3  => s4=0, s2=1, s1=1
        // count == 2  => s4=0, s2=1, s1=0
        const std::uint64_t not_s4 = ~s4;
        const std::uint64_t eq3 = (s1 & s2) & not_s4;
        const std::uint64_t eq2 = ((~s1) & s2) & not_s4;

        // Apply Game of Life rules: alive with 2 or 3 neighbors survives; dead with 3 neighbors becomes alive.
        const std::uint64_t next = eq3 | (mC & eq2);

        // Store the next state
        output[global_idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Compute total number of 64-bit words and launch configuration.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Choose a reasonable launch configuration for data-center GPUs (A100/H100).
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);
    if (grid_size <= 0) grid_size = 1;

    gol_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
