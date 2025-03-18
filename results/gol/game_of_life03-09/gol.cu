#include <cstdint>
#include <cuda_runtime.h>

// Kernel for one step of Conway's Game of Life on a bit‐packed grid.
// Each thread processes one std::uint64_t word (64 cells) from the grid.
// The grid is stored as an array of std::uint64_t words, where each word
// represents 64 contiguous cells (one bit per cell). The grid is organized
// as rows of words. The grid dimensions are given in cells (width and height)
// and are always a power of 2. There are (grid_dimensions/64) words per row.
// 
// For each cell, the eight neighbors are examined from the cells in the rows
// above, the same row, and below. Neighbor bits falling outside a word’s
// boundaries require special handling: for the 0th bit, the left neighbor comes
// from the previous word (if available), and for the 63rd bit, the right neighbor
// comes from the following word (if available).
//
// The rules of Conway's Game of Life are applied:
//   - A live cell with 2 or 3 neighbors stays alive.
//   - A dead cell with exactly 3 neighbors becomes alive.
//   - Otherwise, the cell becomes or remains dead.
//
// For each thread, the necessary neighboring words (from row above and below,
// and adjacent words in the same row) are loaded. Then, for each of the 64 bits,
// the neighbor count is computed by selecting the proper bit from the corresponding
// adjacent word using bit shifts and bit masks. To reduce overhead, the loop is split
// into three parts: one for bit 0 (left‐edge), one for bits 1..62 (middle bits), and one for bit 63 (right‐edge).
//
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions) {
    // Compute number of words per row (each word holds 64 cells).
    int words_per_row = grid_dimensions >> 6; // equivalent to grid_dimensions / 64

    // Compute thread coordinates: row index (cell row) and column index as word index.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure indices are within the grid bounds.
    if (row >= grid_dimensions || col >= words_per_row)
        return;

    // Compute the linear index for the current word.
    int index = row * words_per_row + col;

    // Load the 64-bit word representing the current 64 cells.
    std::uint64_t center   = input[index];

    // Load words of neighboring rows. If a neighbor row is out-of-bound, use 0.
    std::uint64_t top_center = (row > 0) ? input[(row - 1) * words_per_row + col] : 0ULL;
    std::uint64_t bot_center = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0ULL;

    // Load neighboring words from the same row (left and right).
    std::uint64_t left_word  = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t right_word = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;

    // Load the adjacent words for the top row.
    std::uint64_t top_left   = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t top_right  = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0ULL;

    // Load the adjacent words for the bottom row.
    std::uint64_t bot_left   = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t bot_right  = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0ULL;

    // Precompute flags for neighbor availability.
    bool has_top   = (row > 0);
    bool has_bot   = (row < grid_dimensions - 1);
    bool has_left  = (col > 0);
    bool has_right = (col < words_per_row - 1);

    // The result word for the next generation.
    std::uint64_t result = 0ULL;

    //------------------------------------------------------
    // Handle bit position 0 (left edge of the word)
    //------------------------------------------------------
    {
        int count = 0;
        // Top row neighbors for bit 0.
        if (has_top) {
            // Top-left neighbor: if available, comes from top-left word (bit 63); otherwise 0.
            count += (has_left ? (int)((top_left >> 63) & 1ULL) : 0);
            // Top neighbor: from the top_center word, bit 0.
            count += (int)((top_center >> 0) & 1ULL);
            // Top-right neighbor: for bit 0, from top_center word, bit 1.
            count += (int)((top_center >> 1) & 1ULL);
        }
        // Same row neighbors for bit 0.
        // Left neighbor: from left_word at bit 63.
        count += (has_left ? (int)((left_word >> 63) & 1ULL) : 0);
        // Right neighbor: from center word, bit 1.
        count += (int)((center >> 1) & 1ULL);

        // Bottom row neighbors for bit 0.
        if (has_bot) {
            // Bottom-left neighbor: from bot_left word at bit 63.
            count += (has_left ? (int)((bot_left >> 63) & 1ULL) : 0);
            // Bottom neighbor: from bot_center word, bit 0.
            count += (int)((bot_center >> 0) & 1ULL);
            // Bottom-right neighbor: from bot_center word, bit 1.
            count += (int)((bot_center >> 1) & 1ULL);
        }
        // Current cell value at bit 0.
        int cell = (int)((center >> 0) & 1ULL);
        int new_cell;
        // Apply the Game of Life rules.
        if (cell)
            new_cell = (count == 2 || count == 3) ? 1 : 0;
        else
            new_cell = (count == 3) ? 1 : 0;
        // Set the computed bit in the result.
        result |= ((std::uint64_t)new_cell << 0);
    }

    //------------------------------------------------------
    // Handle inner bits 1 to 62 (no cross–word boundary issues)
    //------------------------------------------------------
    for (int i = 1; i < 63; i++) {
        int count = 0;
        if (has_top) {
            // For inner bits, the top-left, top, and top-right neighbors all come from top_center.
            count += (int)((top_center >> (i - 1)) & 1ULL);
            count += (int)((top_center >> i) & 1ULL);
            count += (int)((top_center >> (i + 1)) & 1ULL);
        }
        // Same row: left and right neighbors from the current word.
        count += (int)((center >> (i - 1)) & 1ULL);
        count += (int)((center >> (i + 1)) & 1ULL);
        if (has_bot) {
            // Bottom row neighbors.
            count += (int)((bot_center >> (i - 1)) & 1ULL);
            count += (int)((bot_center >> i) & 1ULL);
            count += (int)((bot_center >> (i + 1)) & 1ULL);
        }
        int cell = (int)((center >> i) & 1ULL);
        int new_cell;
        if (cell)
            new_cell = (count == 2 || count == 3) ? 1 : 0;
        else
            new_cell = (count == 3) ? 1 : 0;
        result |= ((std::uint64_t)new_cell << i);
    }

    //------------------------------------------------------
    // Handle bit position 63 (right edge of the word)
    //------------------------------------------------------
    {
        int count = 0;
        if (has_top) {
            // Top row: left neighbor from top_center at bit 62 and top neighbor from top_center bit 63.
            count += (int)((top_center >> 62) & 1ULL);
            count += (int)((top_center >> 63) & 1ULL);
            // Top-right neighbor: not available in top_center, so use top_right word, bit 0.
            count += (has_right ? (int)((top_right >> 0) & 1ULL) : 0);
        }
        // Same row: left neighbor from center (bit 62) and right neighbor from right_word.
        count += (int)((center >> 62) & 1ULL);
        count += (has_right ? (int)((right_word >> 0) & 1ULL) : 0);
        if (has_bot) {
            // Bottom row: bottom-left from bot_center at bit 62, bottom from bot_center bit 63.
            count += (int)((bot_center >> 62) & 1ULL);
            count += (int)((bot_center >> 63) & 1ULL);
            // Bottom-right neighbor: from bot_right word, bit 0.
            count += (has_right ? (int)((bot_right >> 0) & 1ULL) : 0);
        }
        int cell63 = (int)((center >> 63) & 1ULL);
        int new_cell63;
        if (cell63)
            new_cell63 = (count == 2 || count == 3) ? 1 : 0;
        else
            new_cell63 = (count == 3) ? 1 : 0;
        result |= ((std::uint64_t)new_cell63 << 63);
    }

    // Write the computed next-generation word to the output grid.
    output[index] = result;
}

// Host function that launches one simulation step of Conway's Game of Life.
// The input and output grids are bit-packed arrays of std::uint64_t values.
// Each word represents 64 cells in a row. The grid is square with grid_dimensions
// cells per side, and grid_dimensions is a power of 2. This function launches the CUDA
// kernel with an appropriate grid configuration. All host-device synchronization is assumed
// to be handled by the caller.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Compute number of words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Set up a 2D grid of CUDA threads:
    // Each thread processes one 64-bit word. The overall grid size in threads is
    // grid_dimensions rows x words_per_row columns.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
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
