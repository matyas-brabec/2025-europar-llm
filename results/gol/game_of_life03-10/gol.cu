// This CUDA implementation of Conway’s Game of Life processes a bit‐packed grid.
// Each std::uint64_t in the input represents 64 consecutive cells (one bit per cell)
// of a row, where logical cell index 0 is the leftmost cell and index 63 is the rightmost.
// Each CUDA thread computes the next state of one such 64‐cell word using the standard
// Game of Life rules. For each cell the 8 neighboring cells (top, bottom, left, right, and diagonals)
// are examined. Cells outside the grid are assumed dead.
// Special handling is provided for the leftmost (logical index 0) and rightmost (logical index 63)
// cells, for which neighbors can come from adjacent words in the same row or in adjacent rows.
//
// The kernel launches a 2D grid over the logical grid:
//    - The y-dimension corresponds to grid rows (total grid_dimensions rows).
//    - The x-dimension corresponds to the word index in that row (total grid_dimensions/64 words).
//
// The run_game_of_life function computes one simulation step by invoking the kernel.
// All host/device memory allocation and synchronization are assumed to be managed externally.

#include <cstdint>
#include <cuda_runtime.h>

// Kernel function: each thread processes one uint64_t word (64 cells) from the grid.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row has grid_dimensions/64 words.
    const int grid_words = grid_dimensions >> 6;  // division by 64

    // Compute logical coordinates in the grid:
    // "row" is the cell row (0 <= row < grid_dimensions)
    // "col" is the word index in that row (0 <= col < grid_words).
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= grid_dimensions || col >= grid_words)
        return;

    // Compute the index within the bit-packed grid.
    const int index = row * grid_words + col;

    // Determine existence of neighboring rows/words.
    const bool has_top   = (row > 0);
    const bool has_bot   = (row < grid_dimensions - 1);
    const bool has_left  = (col > 0);
    const bool has_right = (col < grid_words - 1);

    // Load words for the current row.
    const std::uint64_t mid_mid = input[index];
    const std::uint64_t mid_left  = has_left  ? input[row * grid_words + col - 1] : 0;
    const std::uint64_t mid_right = has_right ? input[row * grid_words + col + 1] : 0;

    // Load words for the row above (if available).
    std::uint64_t top_mid = 0, top_left = 0, top_right = 0;
    if (has_top) {
        const int top_index = (row - 1) * grid_words + col;
        top_mid   = input[top_index];
        top_left  = has_left  ? input[(row - 1) * grid_words + col - 1] : 0;
        top_right = has_right ? input[(row - 1) * grid_words + col + 1] : 0;
    }
    
    // Load words for the row below (if available).
    std::uint64_t bot_mid = 0, bot_left = 0, bot_right = 0;
    if (has_bot) {
        const int bot_index = (row + 1) * grid_words + col;
        bot_mid   = input[bot_index];
        bot_left  = has_left  ? input[(row + 1) * grid_words + col - 1] : 0;
        bot_right = has_right ? input[(row + 1) * grid_words + col + 1] : 0;
    }

    // The new computed uint64_t word to be stored in output.
    std::uint64_t new_word = 0;

    // Helper lambda to extract the cell bit from a word.
    // The logical cell index "pos" is in [0,63] with 0 = leftmost.
    // In the stored 64-bit word, the bit corresponding to logical cell "pos"
    // is at bit-position (63 - pos).
    auto get_bit = [] (std::uint64_t word, int pos) -> int {
        return (word >> (63 - pos)) & 1ULL;
    };

    // Process the leftmost cell in this word (logical index 0).
    {
        const int j = 0;
        const int shift = 63 - j;  // position to set bit in new_word
        const int cell = get_bit(mid_mid, j);
        int sum = 0;
        // Top row neighbors.
        // For the leftmost cell, the top-left neighbor is in the adjacent left word.
        const int top_left_bit  = has_top ? get_bit(top_left, 63) : 0;
        const int top_bit       = has_top ? get_bit(top_mid, j) : 0;
        const int top_right_bit = has_top ? get_bit(top_mid, j + 1) : 0;
        sum += top_left_bit + top_bit + top_right_bit;
        // Same row neighbors (exclude center cell).
        const int left_bit  = has_left ? get_bit(mid_left, 63) : 0;
        const int right_bit = get_bit(mid_mid, j + 1);
        sum += left_bit + right_bit;
        // Bottom row neighbors.
        const int bot_left_bit  = has_bot ? get_bit(bot_left, 63) : 0;
        const int bot_bit       = has_bot ? get_bit(bot_mid, j) : 0;
        const int bot_right_bit = has_bot ? get_bit(bot_mid, j + 1) : 0;
        sum += bot_left_bit + bot_bit + bot_right_bit;

        // Game of Life rules:
        // - Live cell survives if sum==2 or sum==3.
        // - Dead cell becomes live if sum==3.
        const int new_cell = (cell && (sum == 2 || sum == 3)) || (!cell && sum == 3);
        new_word |= (static_cast<std::uint64_t>(new_cell) << shift);
    }

    // Process cells with logical indices 1 to 62.
    #pragma unroll
    for (int j = 1; j < 63; j++) {
        const int shift = 63 - j;
        const int cell = get_bit(mid_mid, j);
        int sum = 0;
        // Top row neighbors.
        const int top_left_bit  = has_top ? get_bit(top_mid, j - 1) : 0;
        const int top_bit       = has_top ? get_bit(top_mid, j)     : 0;
        const int top_right_bit = has_top ? get_bit(top_mid, j + 1) : 0;
        sum += top_left_bit + top_bit + top_right_bit;
        // Same row neighbors.
        const int left_bit  = get_bit(mid_mid, j - 1);
        const int right_bit = get_bit(mid_mid, j + 1);
        sum += left_bit + right_bit;
        // Bottom row neighbors.
        const int bot_left_bit  = has_bot ? get_bit(bot_mid, j - 1) : 0;
        const int bot_bit       = has_bot ? get_bit(bot_mid, j)     : 0;
        const int bot_right_bit = has_bot ? get_bit(bot_mid, j + 1) : 0;
        sum += bot_left_bit + bot_bit + bot_right_bit;

        const int new_cell = (cell && (sum == 2 || sum == 3)) || (!cell && sum == 3);
        new_word |= (static_cast<std::uint64_t>(new_cell) << shift);
    }

    // Process the rightmost cell in this word (logical index 63).
    {
        const int j = 63;
        const int shift = 63 - j;  // equals 0
        const int cell = get_bit(mid_mid, j);
        int sum = 0;
        // Top row neighbors.
        const int top_left_bit  = has_top ? get_bit(top_mid, j - 1) : 0;
        const int top_bit       = has_top ? get_bit(top_mid, j)     : 0;
        // For the rightmost cell, top-right neighbor is in the adjacent right word.
        const int top_right_bit = has_top ? get_bit(top_right, 0) : 0;
        sum += top_left_bit + top_bit + top_right_bit;
        // Same row neighbors.
        const int left_bit  = get_bit(mid_mid, j - 1);
        const int right_bit = has_right ? get_bit(mid_right, 0) : 0;
        sum += left_bit + right_bit;
        // Bottom row neighbors.
        const int bot_left_bit  = has_bot ? get_bit(bot_mid, j - 1) : 0;
        const int bot_bit       = has_bot ? get_bit(bot_mid, j)     : 0;
        const int bot_right_bit = has_bot ? get_bit(bot_right, 0)   : 0;
        sum += bot_left_bit + bot_bit + bot_right_bit;

        const int new_cell = (cell && (sum == 2 || sum == 3)) || (!cell && sum == 3);
        new_word |= (static_cast<std::uint64_t>(new_cell) << shift);
    }

    // Write the computed next-generation word.
    output[index] = new_word;
}

// Host function to run one Game of Life simulation step.
// Parameters:
//   input           - pointer to bit-packed input grid (allocated via cudaMalloc),
//                     each std::uint64_t encodes 64 cells from one row.
//   output          - pointer to bit-packed output grid (allocated via cudaMalloc).
//   grid_dimensions - width and height of the grid (in cells), a power of 2.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row has grid_dimensions/64 words.
    const int grid_words = grid_dimensions >> 6;
    // Configure a 2D grid of threads:
    //   x-dimension: word indices (0 .. grid_words-1)
    //   y-dimension: cell rows (0 .. grid_dimensions-1)
    const dim3 block(16, 16);
    const dim3 grid((grid_words + block.x - 1) / block.x, (grid_dimensions + block.y - 1) / block.y);
    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // Note: Any necessary host-device synchronization is handled by the caller.
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
