#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel for one Game of Life iteration.
// Each thread processes one 64‐cell word (std::uint64_t) from the bit‐packed grid.
// The grid is arranged as rows of bit‐packed 64-bit words; each row has (grid_dimensions/64) words.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of 64–bit words per row.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Total number of words in the grid.
    const int total_words = grid_dimensions * words_per_row;

    // Global thread index.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Determine the row and column (word index) corresponding to this thread.
    int row = idx / words_per_row;
    int col = idx % words_per_row;

    // Determine neighboring row indices. If at the boundary, treat out‐of‐bound as -1.
    int top_row    = (row > 0) ? row - 1 : -1;
    int bottom_row = (row < grid_dimensions - 1) ? row + 1 : -1;

    // Determine neighboring word indices in the current row.
    int left_col  = (col > 0) ? col - 1 : -1;
    int right_col = (col < words_per_row - 1) ? col + 1 : -1;

    // Load the 9 neighboring words (from the three rows) if available; otherwise use 0.
    uint64_t top_left = 0, top_center = 0, top_right = 0;
    if (top_row != -1) {
        if (left_col != -1)
            top_left = input[top_row * words_per_row + left_col];
        top_center = input[top_row * words_per_row + col];
        if (right_col != -1)
            top_right = input[top_row * words_per_row + right_col];
    }

    uint64_t current_left = 0, current = 0, current_right = 0;
    if (left_col != -1)
        current_left = input[row * words_per_row + left_col];
    current = input[row * words_per_row + col];
    if (right_col != -1)
        current_right = input[row * words_per_row + right_col];

    uint64_t bottom_left = 0, bottom_center = 0, bottom_right = 0;
    if (bottom_row != -1) {
        if (left_col != -1)
            bottom_left = input[bottom_row * words_per_row + left_col];
        bottom_center = input[bottom_row * words_per_row + col];
        if (right_col != -1)
            bottom_right = input[bottom_row * words_per_row + right_col];
    }

    // For each direction we need to align the bits to our current word.
    // For diagonal/adjacent cells, bits may come from an adjacent word.
    // Shifting left by 1 moves bit i from position (i) to (i+1); however, we want the neighbor of cell i 
    // coming from a word to the left to appear in bit position 0. Thus, we combine shifts with extraction.
    //
    // For cells in the row above:
    //   - Diagonal left: for cell i, neighbor is at (r-1, (col*64 + i - 1)). For i>0, this is bit (i-1) of top_center;
    //     for i==0, it is bit 63 of top_left.
    uint64_t top_left_neighbors = (top_center << 1) | ((left_col != -1) ? (top_left >> 63) : 0);
    //   - Directly above: same column; simply top_center.
    uint64_t top_center_neighbors = top_center;
    //   - Diagonal right: for cell i, neighbor is at (r-1, (col*64 + i + 1)). For i<63, this is bit (i+1) of top_center;
    //     for i==63, it is bit 0 of top_right.
    uint64_t top_right_neighbors = (top_center >> 1) | ((right_col != -1) ? (top_right << 63) : 0);

    // For the current row (lateral neighbors only):
    //   - Left neighbor: for cell i, for i>0 use bit (i-1) from current; for i==0 use bit 63 from current_left.
    uint64_t current_left_neighbors = (current << 1) | ((left_col != -1) ? (current_left >> 63) : 0);
    //   - Right neighbor: for cell i, for i<63 use bit (i+1) from current; for i==63 use bit 0 from current_right.
    uint64_t current_right_neighbors = (current >> 1) | ((right_col != -1) ? (current_right << 63) : 0);

    // For the row below:
    uint64_t bottom_left_neighbors = (bottom_center << 1) | ((left_col != -1) ? (bottom_left >> 63) : 0);
    uint64_t bottom_center_neighbors = bottom_center;
    uint64_t bottom_right_neighbors = (bottom_center >> 1) | ((right_col != -1) ? (bottom_right << 63) : 0);

    // Compute the new state for the 64 cells of the current word.
    // Iterate over each bit position (cell) from 0 to 63 and count live neighbors.
    uint64_t new_word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        int count = 0;
        // Sum the contributions from each of the 8 neighbor directions.
        count += (int)((top_left_neighbors >> bit) & 1ULL);
        count += (int)((top_center_neighbors >> bit) & 1ULL);
        count += (int)((top_right_neighbors >> bit) & 1ULL);
        count += (int)((current_left_neighbors >> bit) & 1ULL);
        count += (int)((current_right_neighbors >> bit) & 1ULL);
        count += (int)((bottom_left_neighbors >> bit) & 1ULL);
        count += (int)((bottom_center_neighbors >> bit) & 1ULL);
        count += (int)((bottom_right_neighbors >> bit) & 1ULL);

        // Get the current cell state.
        int cell = (int)((current >> bit) & 1ULL);
        int new_cell = 0;
        // Apply Conway's Game of Life rules:
        // For a live cell: survives if it has 2 or 3 live neighbors.
        // For a dead cell: becomes live if it has exactly 3 live neighbors.
        if (cell)
            new_cell = ((count == 2) || (count == 3)) ? 1 : 0;
        else
            new_cell = (count == 3) ? 1 : 0;
        new_word |= ((uint64_t)new_cell << bit);
    }

    // Write the computed new word into the output grid.
    output[idx] = new_word;
}

// Host function to execute one iteration (step) of Conway's Game of Life on the GPU.
// The grid is a square of dimensions grid_dimensions x grid_dimensions,
// with each cell stored as one bit in a bit-packed std::uint64_t array.
// Each word encodes 64 consecutive cells in a row.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Choose an appropriate number of threads per block.
    int threadsPerBlock = 256;
    int blocks = (total_words + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to perform one simulation step.
    game_of_life_kernel<<<blocks, threadsPerBlock>>>(input, output, grid_dimensions);
    // Note: Host-device synchronization is handled by the caller.
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
