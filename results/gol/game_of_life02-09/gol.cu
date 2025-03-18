#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one generation step of Conway's Game of Life
// on a 2D bit‐packed grid. Each 64‐bit word encodes 64 consecutive cells
// (1 = alive, 0 = dead) in a row. The grid is of size grid_dimensions x grid_dimensions,
// and each row contains grid_dimensions/64 words.
// Each thread computes one 64‐bit output word based on the 8 neighbors for each cell.
// Neighbors outside the grid are assumed dead.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // Calculate number of 64‐bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Compute the global index for this thread (one output word per thread)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Determine the row index and the word index within the row.
    int row = idx / words_per_row;
    int word_col = idx % words_per_row;

    // Temporary variables to hold the 9 words (neighbors) required:
    // For each of the 3 rows (upper, current, lower),
    // we need the word for columns: left, center, and right.
    std::uint64_t upper_left = 0, upper_mid = 0, upper_right = 0;
    std::uint64_t current_left = 0, current_mid = 0, current_right = 0;
    std::uint64_t lower_left = 0, lower_mid = 0, lower_right = 0;
    int n = grid_dimensions; // total number of rows

    // Load words from the row above (if exists), else remain 0.
    if (row > 0)
    {
        int row_above = row - 1;
        int base = row_above * words_per_row;
        if (word_col > 0)
            upper_left = input[base + word_col - 1];
        upper_mid = input[base + word_col];
        if (word_col < words_per_row - 1)
            upper_right = input[base + word_col + 1];
    }

    // Load words from the current row.
    {
        int base = row * words_per_row;
        if (word_col > 0)
            current_left = input[base + word_col - 1];
        current_mid = input[base + word_col];
        if (word_col < words_per_row - 1)
            current_right = input[base + word_col + 1];
    }

    // Load words from the row below (if exists), else remain 0.
    if (row < n - 1)
    {
        int row_below = row + 1;
        int base = row_below * words_per_row;
        if (word_col > 0)
            lower_left = input[base + word_col - 1];
        lower_mid = input[base + word_col];
        if (word_col < words_per_row - 1)
            lower_right = input[base + word_col + 1];
    }

    // For each cell in the 64-bit word, count neighbors and update cell state.
    // The bits in a word correspond to 64 consecutive cells.
    // Note: At the bit boundaries (bit 0 and bit 63) we must consider
    // adjacent words (if available) for the neighbor that lies outside the word.
    std::uint64_t result = 0;

    // Process cell corresponding to bit position 0 (left boundary).
    {
        int i = 0;
        int count = 0;
        // Row above:
        // Left neighbor: if i==0, use the last bit of upper_left.
        count += (int)((upper_left >> 63) & 1ULL);
        // Center neighbor: bit 0 of upper_mid.
        count += (int)((upper_mid >> 0) & 1ULL);
        // Right neighbor: bit 1 of upper_mid.
        count += (int)((upper_mid >> 1) & 1ULL);
        // Current row (exclude cell itself):
        count += (int)((current_left >> 63) & 1ULL);
        count += (int)((current_mid >> 1) & 1ULL);
        // Row below:
        count += (int)((lower_left >> 63) & 1ULL);
        count += (int)((lower_mid >> 0) & 1ULL);
        count += (int)((lower_mid >> 1) & 1ULL);
        int cell = (int)((current_mid >> 0) & 1ULL);
        // Apply Game of Life rules:
        // Live cell survives with 2 or 3 neighbors; dead cell becomes live with exactly 3 neighbors.
        int new_cell = ((cell && (count == 2 || count == 3)) || (!cell && count == 3)) ? 1 : 0;
        result |= ((std::uint64_t)new_cell << 0);
    }

    // Process cells corresponding to bit positions 1 to 62.
    for (int i = 1; i < 63; i++)
    {
        int count = 0;
        // Row above: neighbors from current row's above row are always in upper_mid.
        count += (int)((upper_mid >> (i - 1)) & 1ULL);
        count += (int)((upper_mid >> i) & 1ULL);
        count += (int)((upper_mid >> (i + 1)) & 1ULL);
        // Same row: only left and right neighbors count.
        count += (int)((current_mid >> (i - 1)) & 1ULL);
        count += (int)((current_mid >> (i + 1)) & 1ULL);
        // Row below:
        count += (int)((lower_mid >> (i - 1)) & 1ULL);
        count += (int)((lower_mid >> i) & 1ULL);
        count += (int)((lower_mid >> (i + 1)) & 1ULL);
        int cell = (int)((current_mid >> i) & 1ULL);
        int new_cell = ((cell && (count == 2 || count == 3)) || (!cell && count == 3)) ? 1 : 0;
        result |= ((std::uint64_t)new_cell << i);
    }

    // Process cell corresponding to bit position 63 (right boundary).
    {
        int i = 63;
        int count = 0;
        // Row above: use upper_mid for neighbors, but for right neighbor use upper_right's bit 0.
        count += (int)((upper_mid >> 62) & 1ULL);
        count += (int)((upper_mid >> 63) & 1ULL);
        count += (int)((upper_right >> 0) & 1ULL);
        // Current row:
        count += (int)((current_mid >> 62) & 1ULL);
        count += (int)((current_right >> 0) & 1ULL);
        // Row below:
        count += (int)((lower_mid >> 62) & 1ULL);
        count += (int)((lower_mid >> 63) & 1ULL);
        count += (int)((lower_right >> 0) & 1ULL);
        int cell = (int)((current_mid >> 63) & 1ULL);
        int new_cell = ((cell && (count == 2 || count == 3)) || (!cell && count == 3)) ? 1 : 0;
        result |= ((std::uint64_t)new_cell << 63);
    }

    // Write the computed 64-bit output word back to global memory.
    output[idx] = result;
}

// Host function to execute one step of Conway's Game of Life.
// 'input' and 'output' point to GPU memory containing the bit-packed grids.
// 'grid_dimensions' is the width/height of the square grid (always a power of 2).
// This function launches the CUDA kernel to compute the next generation.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    // Total number of 64-bit words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Choose a reasonable block size (e.g., 256 threads per block).
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel. Any host-device synchronization is assumed to be handled by the caller.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
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
