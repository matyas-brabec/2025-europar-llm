// Conway's Game of Life CUDA implementation with bit-packed grid.
//
// Each std::uint64_t encodes 64 cells (bits) of a single row, with bit value
// 1 meaning "alive" and 0 meaning "dead". The grid is square, with side length
// grid_dimensions (number of cells), and grid_dimensions is guaranteed to be
// a power of two greater than 512. Therefore, grid_dimensions is also a
// multiple of 64 and there are no partially filled 64‑bit words.
//
// Layout:
//   - The grid is stored row-major in an array of std::uint64_t.
//   - words_per_row = grid_dimensions / 64.
//   - Total words = grid_dimensions * words_per_row.
//   - Word index w corresponds to row = w / words_per_row, col = w % words_per_row.
//   - Within each 64‑bit word, bit index b (0..63) corresponds to a cell in that row.
//   - Neighboring cells may reside in the same word or in the immediate left/right
//     word in the same row, and/or in the 3 corresponding words (left, center,
//     right) in the rows above/below.
//
// Each CUDA thread processes exactly one 64‑bit word (64 cells). This avoids any
// need for atomics and lets us aggressively bit-pack data.
//
// Algorithm per thread:
//   1. Compute (row, col) for the word and load up to 9 relevant words:
//        - center:        (row,   col)
//        - left:          (row,   col-1)
//        - right:         (row,   col+1)
//        - above:         (row-1, col)
//        - above_left:    (row-1, col-1)
//        - above_right:   (row-1, col+1)
//        - below:         (row+1, col)
//        - below_left:    (row+1, col-1)
//        - below_right:   (row+1, col+1)
//      Out-of-bounds neighbors are treated as zero (dead).
//
//   2. Build eight 64‑bit masks, each corresponding to one neighbor direction:
//        up_left_mask, up_mask, up_right_mask,
//        left_mask,             right_mask,
//        down_left_mask, down_mask, down_right_mask.
//      For each bit position i (0..63) in the current word, the i-th bit of
//      these masks encodes whether the corresponding neighbor cell is alive.
//      These masks are constructed using shifts and cross-word bit injection
//      so they correctly handle bit 0 and bit 63, including diagonal neighbors.
//
//      Example for left neighbors within the same row:
//        - For bit i>0, left neighbor is bit (i-1) in the same word.
//        - For bit i=0, left neighbor is bit 63 in the word to the left.
//        Implementation:
//          left_mask = (center << 1) | (left >> 63) if left exists.
//          (Without left word, left >> 63 is treated as 0.)
//
//   3. For each of the 64 bits in this word:
//        - Read the 8 neighbor bits from the 8 masks at this bit position,
//          pack them into an 8-bit integer, and use __popc() to count how many
//          neighbors are alive.
//        - Apply Game of Life rules to decide the next state for that bit.
//        - Write the result into the output word.
//
// This approach:
//   - Performs only 9 global loads per thread (for the 9 words).
//   - Uses simple bitwise operations plus a small loop with __popc().
//   - Handles grid boundaries correctly (outside grid is always dead).
//
// No shared or texture memory is used, as requested.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// CUDA kernel: compute one Game of Life step on a bit-packed grid.
// - input:  bit-packed input grid (device pointer).
// - output: bit-packed output grid (device pointer).
// - grid_dim: number of cells per side of the square grid.
// - words_per_row: number of 64-bit words per row (grid_dim / 64).
// - total_words: total number of 64-bit words in the grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row,
                                    std::size_t total_words)
{
    std::size_t global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (global_idx >= total_words) {
        return;
    }

    // Compute this word's row and column in the grid of words.
    int row = static_cast<int>(global_idx / words_per_row);
    int col = static_cast<int>(global_idx % words_per_row);

    const std::uint64_t center = input[global_idx];

    // Determine existence of neighbor words.
    const bool has_left   = (col > 0);
    const bool has_right  = (col < words_per_row - 1);
    const bool has_above  = (row > 0);
    const bool has_below  = (row < grid_dim - 1);

    // Initialize neighbor words to zero (dead) by default.
    std::uint64_t left         = 0;
    std::uint64_t right        = 0;
    std::uint64_t above        = 0;
    std::uint64_t below        = 0;
    std::uint64_t above_left   = 0;
    std::uint64_t above_right  = 0;
    std::uint64_t below_left   = 0;
    std::uint64_t below_right  = 0;

    // Load horizontally adjacent words in the same row.
    if (has_left) {
        left = input[global_idx - 1];
    }
    if (has_right) {
        right = input[global_idx + 1];
    }

    // Load words from the row above, if it exists.
    if (has_above) {
        std::size_t above_idx = global_idx - static_cast<std::size_t>(words_per_row);
        above = input[above_idx];
        if (has_left) {
            above_left = input[above_idx - 1];
        }
        if (has_right) {
            above_right = input[above_idx + 1];
        }
    }

    // Load words from the row below, if it exists.
    if (has_below) {
        std::size_t below_idx = global_idx + static_cast<std::size_t>(words_per_row);
        below = input[below_idx];
        if (has_left) {
            below_left = input[below_idx - 1];
        }
        if (has_right) {
            below_right = input[below_idx + 1];
        }
    }

    // Build neighbor direction masks for this word.
    // Each mask is a 64-bit word where bit i indicates if the corresponding
    // neighbor cell (in that direction) of cell i is alive.

    // Vertical neighbors (no intra-word shift is needed).
    std::uint64_t up_mask   = above;
    std::uint64_t down_mask = below;

    // Horizontal neighbors (current row).
    // left_mask: bit i = value of left neighbor of center bit i.
    //   - For i>0: center[i-1] -> achieved by (center << 1).
    //   - For i=0: left[63] -> injected from (left >> 63) if left exists.
    std::uint64_t left_mask = (center << 1);
    if (has_left) {
        left_mask |= (left >> 63);
    }

    // right_mask: bit i = value of right neighbor of center bit i.
    //   - For i<63: center[i+1] -> achieved by (center >> 1).
    //   - For i=63: right[0] -> injected from ((right & 1) << 63) if right exists.
    std::uint64_t right_mask = (center >> 1);
    if (has_right) {
        right_mask |= (right & 1ull) << 63;
    }

    // Diagonal neighbors: combine vertical words with shifts and cross-word bits.

    // up_left_mask: bit i is above-left neighbor of center bit i.
    //   - For i>0: above[i-1] -> (above << 1).
    //   - For i=0: above_left[63] -> injected from (above_left >> 63) if available.
    std::uint64_t up_left_mask = (above << 1);
    if (has_above && has_left) {
        up_left_mask |= (above_left >> 63);
    }

    // up_right_mask: bit i is above-right neighbor of center bit i.
    //   - For i<63: above[i+1] -> (above >> 1).
    //   - For i=63: above_right[0] -> injected from ((above_right & 1) << 63).
    std::uint64_t up_right_mask = (above >> 1);
    if (has_above && has_right) {
        up_right_mask |= (above_right & 1ull) << 63;
    }

    // down_left_mask: bit i is below-left neighbor of center bit i.
    //   - For i>0: below[i-1] -> (below << 1).
    //   - For i=0: below_left[63] -> injected from (below_left >> 63).
    std::uint64_t down_left_mask = (below << 1);
    if (has_below && has_left) {
        down_left_mask |= (below_left >> 63);
    }

    // down_right_mask: bit i is below-right neighbor of center bit i.
    //   - For i<63: below[i+1] -> (below >> 1).
    //   - For i=63: below_right[0] -> injected from ((below_right & 1) << 63).
    std::uint64_t down_right_mask = (below >> 1);
    if (has_below && has_right) {
        down_right_mask |= (below_right & 1ull) << 63;
    }

    // Compute next state for each of the 64 cells in this word.
    std::uint64_t next_word = 0;
    std::uint64_t bit_mask = 1ull;

    // Loop over all bits in the 64-bit word.
    // For each bit, collect the 8 neighbor bits into an 8-bit integer and use
    // __popc() to count set bits (live neighbors).
    for (int bit = 0; bit < 64; ++bit) {
        unsigned int neighbor_bits = 0u;

        // Each boolean expression ((mask & bit_mask) != 0) is converted to 0 or 1
        // and placed into a distinct bit position of neighbor_bits.
        neighbor_bits |= static_cast<unsigned int>((up_left_mask   & bit_mask) != 0ull) << 0;
        neighbor_bits |= static_cast<unsigned int>((up_mask        & bit_mask) != 0ull) << 1;
        neighbor_bits |= static_cast<unsigned int>((up_right_mask  & bit_mask) != 0ull) << 2;
        neighbor_bits |= static_cast<unsigned int>((left_mask      & bit_mask) != 0ull) << 3;
        neighbor_bits |= static_cast<unsigned int>((right_mask     & bit_mask) != 0ull) << 4;
        neighbor_bits |= static_cast<unsigned int>((down_left_mask & bit_mask) != 0ull) << 5;
        neighbor_bits |= static_cast<unsigned int>((down_mask      & bit_mask) != 0ull) << 6;
        neighbor_bits |= static_cast<unsigned int>((down_right_mask& bit_mask) != 0ull) << 7;

        // Count live neighbors using fast hardware popcount.
        unsigned int neighbors = __popc(neighbor_bits);

        // Current cell state (alive or dead).
        bool cell_alive = (center & bit_mask) != 0ull;

        // Apply Conway's Game of Life rules:
        // - Any live cell with 2 or 3 neighbors survives.
        // - Any dead cell with exactly 3 neighbors becomes alive.
        // - All other cells die or remain dead.
        bool next_alive = (!cell_alive && neighbors == 3) ||
                          (cell_alive && (neighbors == 2 || neighbors == 3));

        if (next_alive) {
            next_word |= bit_mask;
        }

        bit_mask <<= 1;
    }

    // Store the computed word to the output grid.
    output[global_idx] = next_word;
}

// Host function: launch one Game of Life step on the GPU.
//
// Parameters:
//   - input:  device pointer to the input grid (bit-packed, std::uint64_t).
//   - output: device pointer to the output grid (bit-packed, std::uint64_t).
//   - grid_dimensions: width/height of the square grid in cells (power of two,
//                      > 512). The number of 64-bit words per row is
//                      grid_dimensions / 64.
//
// Requirements:
//   - input and output must be allocated using cudaMalloc and large enough to
//     hold grid_dimensions * grid_dimensions bits.
//   - No device synchronization is performed here; the caller is responsible
//     for any required cudaDeviceSynchronize() or error checking.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row (guaranteed integer since grid_dimensions
    // is a power of two >= 512, hence divisible by 64).
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Total number of 64-bit words in the grid.
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) *
        static_cast<std::size_t>(words_per_row);

    // Choose a reasonable launch configuration for modern data center GPUs.
    // 256 threads per block is a good default.
    const int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    // Launch the kernel. No synchronization here; caller handles it.
    game_of_life_kernel<<<blocks, threads_per_block>>>(
        input, output, grid_dimensions, words_per_row, total_words);
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
