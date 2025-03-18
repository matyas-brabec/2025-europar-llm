#include <cuda_runtime.h>
#include <cstdint>

// This CUDA kernel implements one step of Conway’s Game of Life on a
// 2D grid stored in a bit‐packed format. Each word (std::uint64_t) holds 64 cells,
// where bit positions 0 and 63 are treated as the leftmost and rightmost cells respectively.
// The grid is stored row‐wise and each row consists of (grid_dimensions/64) words.
// Each thread is responsible for updating one 64‐cell block (a single word).
//
// For each cell in the 64-bit word, the 8 neighbor cells (from the row above,
// the same row – excluding the cell itself, and the row below) are examined.
// Special handling is needed for the first (bit index 0) and last (bit index 63) cells
// because their horizontal neighbors may reside in the adjacent word of the same row (or above/below).
//
// In this implementation we adopt the following convention:
//   • In each 64‐bit word, bit index 0 represents the leftmost cell and bit index 63 the rightmost.
//   • When a neighbor cell is out‐of‐bounds (either off the grid or off the current row’s word),
//     it is assumed dead (0).
//
// The next generation cell state is computed using the standard rules:
//   • An alive cell with 2 or 3 living neighbors survives; otherwise, it dies.
//   • A dead cell with exactly 3 living neighbors becomes alive.

__global__
void game_of_life_kernel(const std::uint64_t* input,
                           std::uint64_t* output,
                           int grid_dimensions,    // grid dimensions in cells (square grid)
                           int words_per_row)      // number of 64-bit words per row = grid_dimensions/64
{
    // Compute the 2D thread indices:
    // word_x: index of the 64-bit word in the row (0 <= word_x < words_per_row)
    // row: row index (0 <= row < grid_dimensions)
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= grid_dimensions || word_x >= words_per_row)
        return;
    
    // For a given cell block in row 'row' and word index 'word_x', we need to load
    // the corresponding input words for the row above, the current row, and the row below.
    // For each row, we also need the word immediately to the left and right
    // (if they exist) to correctly handle neighbor cells at the boundaries (bit index 0 and bit index 63).
    
    std::uint64_t A_L, A_C, A_R; // Words from the row ABOVE (if any)
    std::uint64_t M_L, M_C, M_R; // Words from the MIDDLE (current) row
    std::uint64_t B_L, B_C, B_R; // Words from the row BELOW (if any)
    
    // Load the "above" row data; if we are at the top of the grid, use 0.
    if (row > 0) {
        int row_above = row - 1;
        A_C = input[row_above * words_per_row + word_x];
        A_L = (word_x > 0) ? input[row_above * words_per_row + (word_x - 1)] : 0ULL;
        A_R = (word_x < words_per_row - 1) ? input[row_above * words_per_row + (word_x + 1)] : 0ULL;
    } else {
        A_C = 0; A_L = 0; A_R = 0;
    }
    
    // Load the current row data.
    M_C = input[row * words_per_row + word_x];
    M_L = (word_x > 0) ? input[row * words_per_row + (word_x - 1)] : 0ULL;
    M_R = (word_x < words_per_row - 1) ? input[row * words_per_row + (word_x + 1)] : 0ULL;
    
    // Load the "below" row data; if we are at the bottom of the grid, use 0.
    if (row < grid_dimensions - 1) {
        int row_below = row + 1;
        B_C = input[row_below * words_per_row + word_x];
        B_L = (word_x > 0) ? input[row_below * words_per_row + (word_x - 1)] : 0ULL;
        B_R = (word_x < words_per_row - 1) ? input[row_below * words_per_row + (word_x + 1)] : 0ULL;
    } else {
        B_C = 0; B_L = 0; B_R = 0;
    }
    
    // newWord will accumulate the updated state for the 64 cells.
    std::uint64_t newWord = 0;
    
    // For each bit in the word (cell index from 0 to 63), compute the number of live neighbors.
    // We unroll the boundary cases for bit index 0 and 63, and use a simple loop for the inner bits.
    
    // ---- Handle bit index 0 (leftmost cell in the block) ----
    {
        int i = 0;
        int count = 0;
        // Neighbors from the row ABOVE:
        //   left neighbor: comes from the left adjacent word's rightmost cell (bit 63) if it exists.
        count += (A_L >> 63) & 1;
        //   center neighbor: from the above row's current word, bit 0.
        count += (A_C >> 0) & 1;
        //   right neighbor: from the above row's current word, bit 1.
        count += (A_C >> 1) & 1;
        
        // Neighbors from the SAME row (excluding the cell itself):
        //   left neighbor: from the left word, bit 63.
        count += (M_L >> 63) & 1;
        //   right neighbor: from the current word, bit 1.
        count += (M_C >> 1) & 1;
        
        // Neighbors from the row BELOW:
        //   left neighbor: from the below row's left word, bit 63.
        count += (B_L >> 63) & 1;
        //   center neighbor: from the below row's current word, bit 0.
        count += (B_C >> 0) & 1;
        //   right neighbor: from the below row's current word, bit 1.
        count += (B_C >> 1) & 1;
        
        // Current cell state:
        int current = (M_C >> 0) & 1;
        // Conway’s Game of Life rule:
        // A cell is alive in the next generation if it has exactly 3 live neighbors,
        // or if it has 2 live neighbors and it is already alive.
        int alive = (count == 3 || (count == 2 && current)) ? 1 : 0;
        newWord |= ((std::uint64_t)alive << 0);
    }
    
    // ---- Handle inner bits (indices 1 to 62) ----
    for (int i = 1; i < 63; ++i) {
        int count = 0;
        // Row ABOVE:
        count += (A_C >> (i - 1)) & 1; // top-left
        count += (A_C >> i)       & 1; // top-center
        count += (A_C >> (i + 1)) & 1; // top-right
        
        // SAME row (only left and right neighbors):
        count += (M_C >> (i - 1)) & 1; // left
        count += (M_C >> (i + 1)) & 1; // right
        
        // Row BELOW:
        count += (B_C >> (i - 1)) & 1; // bottom-left
        count += (B_C >> i)       & 1; // bottom-center
        count += (B_C >> (i + 1)) & 1; // bottom-right
        
        int current = (M_C >> i) & 1;
        int alive = (count == 3 || (count == 2 && current)) ? 1 : 0;
        newWord |= ((std::uint64_t)alive << i);
    }
    
    // ---- Handle bit index 63 (rightmost cell in the block) ----
    {
        int i = 63;
        int count = 0;
        // Row ABOVE:
        count += (A_C >> 62) & 1;    // top-left (from current word, bit 62)
        count += (A_C >> 63) & 1;    // top-center (current word, bit 63)
        count += (A_R >> 0)  & 1;    // top-right (from right neighbor word, bit 0)
        
        // SAME row:
        count += (M_C >> 62) & 1;    // left neighbor (current word, bit 62)
        count += (M_R >> 0)  & 1;    // right neighbor (from right neighbor word, bit 0)
        
        // Row BELOW:
        count += (B_C >> 62) & 1;    // bottom-left (current word, bit 62)
        count += (B_C >> 63) & 1;    // bottom-center (current word, bit 63)
        count += (B_R >> 0)  & 1;    // bottom-right (from right neighbor word, bit 0)
        
        int current = (M_C >> 63) & 1;
        int alive = (count == 3 || (count == 2 && current)) ? 1 : 0;
        newWord |= ((std::uint64_t)alive << 63);
    }
    
    // Write the computed 64-cell block to the output grid.
    output[row * words_per_row + word_x] = newWord;
}

// Host function that launches the CUDA kernel to perform one generation update of Conway’s Game of Life.
// 'input' and 'output' point to device memory containing the bit-packed grids.
// grid_dimensions is the width and height (in cells) of the square grid.
// Note: Any required host-to-device data transfers or synchronizations (outside kernel launch)
// are assumed to be handled by the caller.
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions/64 words (each word represents 64 cells).
    int words_per_row = grid_dimensions / 64;
    
    // Configure a 2D grid of thread blocks.
    // We choose a 2D block (16x16 threads) for good occupancy; adjust if needed.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions, words_per_row);
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
