/*
Conway’s Game of Life CUDA Implementation
-------------------------------------------------
This implementation assumes a 2D grid with bit‐packed storage:
• Each std::uint64_t word encodes 64 cells in a row (bit 0 = left‐most cell,
  bit 63 = right‐most cell).
• The grid dimensions (width and height) are equal, a power‐of‐2 greater than 512.
• All cells outside the grid are dead.
• The kernel computes one generation (“step”) given the input grid, writing to output.

Design:
• Each CUDA thread processes one 64‐bit word (i.e. 64 consecutive cells in a given row).
• For each cell (bit) within the word, the eight neighboring cells are examined.
  The neighbors may be located in one of up to three contiguous 64‐bit words in the
  row above, the same row, or the row below.
• Special care is taken for cells on word boundaries (bit positions 0 and 63).
• All required neighbor words are loaded into registers.
• The inner loop over the 64 cells is unrolled for intermediate bits using the #pragma unroll directive.
• The next generation rule is: a cell becomes alive if it has exactly 3 live neighbors,
  or if it is already alive and has exactly 2 live neighbors.
• No shared/texture memory is used as it adds complexity without measurable benefit.

Assumptions:
• "input" and "output" are device pointers allocated via cudaMalloc.
• Host–device synchronization is handled externally.
*/

#include <cuda_runtime.h>
#include <cstdint>

// __global__ kernel: process one 64-bit word (i.e. 64 cells) of the grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    // Each row contains grid_dimensions/64 words.
    int words_per_row = grid_dimensions >> 6;  // division by 64

    // Compute global word index.
    int global_word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dimensions * words_per_row;
    if (global_word_idx >= total_words)
        return;

    // Compute row and column (word index) in the grid.
    int row = global_word_idx / words_per_row;
    int col = global_word_idx % words_per_row;

    // Load neighbor words from adjacent rows.
    // For cells outside the grid bounds (top, left, right, bottom) we use 0.
    //
    // For the row above (row-1):
    std::uint64_t top_mid   = (row > 0) ? input[(row - 1) * words_per_row + col] : 0ULL;
    std::uint64_t top_left  = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t top_right = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0ULL;

    // For the current row (row):
    std::uint64_t mid_current = input[row * words_per_row + col];
    std::uint64_t mid_left    = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t mid_right   = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;

    // For the row below (row+1):
    std::uint64_t bot_mid   = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0ULL;
    std::uint64_t bot_left  = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t bot_right = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0ULL;

    // The output word corresponding to the current 64 cells.
    std::uint64_t result = 0ULL;

    // Process each of the 64 cells (bits) in the current word.
    // For each cell at bit position b, we determine the 8 neighbors:
    // Neighbor cell coordinates (relative to current cell at (row, col*64 + b)):
    //   - Row above: columns (b-1, b, b+1)
    //   - Same row: columns (b-1, b+1)
    //   - Row below: columns (b-1, b, b+1)
    // When b-1 < 0 or b+1 >= 64, we fetch from the adjacent word (mid_left, mid_right, etc.)

    // Process cell at bit position 0.
    {
        int b = 0;
        int live_neighbors = 0;
        // Top row:
        //   Column: (b-1): if b==0, fetch from top_left, bit 63.
        live_neighbors += (row > 0 && col > 0) ? (int)((top_left >> 63) & 1ULL) : 0;
        //   Column: b (from top_mid, bit 0).
        live_neighbors += (row > 0) ? (int)((top_mid >> 0) & 1ULL) : 0;
        //   Column: (b+1): from top_mid, bit 1.
        live_neighbors += (row > 0) ? (int)((top_mid >> 1) & 1ULL) : 0;

        // Current row:
        //   Column: (b-1): for b==0, fetch from mid_left, bit 63.
        live_neighbors += (col > 0) ? (int)((mid_left >> 63) & 1ULL) : 0;
        //   Column: (b+1): from mid_current, bit 1.
        live_neighbors += (int)((mid_current >> 1) & 1ULL);

        // Bottom row:
        //   Column: (b-1): from bot_left, bit 63.
        live_neighbors += (row < grid_dimensions - 1 && col > 0) ? (int)((bot_left >> 63) & 1ULL) : 0;
        //   Column: b: from bot_mid, bit 0.
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> 0) & 1ULL) : 0;
        //   Column: (b+1): from bot_mid, bit 1.
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> 1) & 1ULL) : 0;

        int cell = (int)((mid_current >> 0) & 1ULL);
        // Apply Game of Life rules:
        //   A dead cell with exactly 3 live neighbors becomes alive.
        //   A live cell with 2 or 3 live neighbors survives.
        int next_cell = ((live_neighbors == 3) || (cell && live_neighbors == 2)) ? 1 : 0;
        result |= ((std::uint64_t)next_cell << 0);
    }

    // Process cells for bit positions 1 to 62 (the common case, no cross‐word fetch required).
    #pragma unroll
    for (int b = 1; b < 63; b++) {
        int live_neighbors = 0;
        // Top row (row-1): all neighbors come from top_mid.
        live_neighbors += (row > 0) ? (int)((top_mid >> (b - 1)) & 1ULL) : 0;  // top-left (b-1)
        live_neighbors += (row > 0) ? (int)((top_mid >> b) & 1ULL) : 0;          // top (b)
        live_neighbors += (row > 0) ? (int)((top_mid >> (b + 1)) & 1ULL) : 0;      // top-right (b+1)

        // Current row: neighbors from mid_current.
        live_neighbors += (int)((mid_current >> (b - 1)) & 1ULL);                // left neighbor (b-1)
        live_neighbors += (int)((mid_current >> (b + 1)) & 1ULL);                // right neighbor (b+1)

        // Bottom row (row+1): all neighbors come from bot_mid.
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> (b - 1)) & 1ULL) : 0; // bottom-left (b-1)
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> b) & 1ULL) : 0;         // bottom (b)
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> (b + 1)) & 1ULL) : 0;     // bottom-right (b+1)

        int cell = (int)((mid_current >> b) & 1ULL);
        int next_cell = ((live_neighbors == 3) || (cell && live_neighbors == 2)) ? 1 : 0;
        result |= ((std::uint64_t)next_cell << b);
    }

    // Process cell at bit position 63.
    {
        int b = 63;
        int live_neighbors = 0;
        // Top row: for b==63, neighbors: (b-1) from top_mid, (b) from top_mid, and (b+1) from top_right.
        live_neighbors += (row > 0) ? (int)((top_mid >> (b - 1)) & 1ULL) : 0;   // top-left (bit 62 from top_mid)
        live_neighbors += (row > 0) ? (int)((top_mid >> b) & 1ULL) : 0;           // top (bit 63 from top_mid)
        live_neighbors += (row > 0 && col < words_per_row - 1) ? (int)((top_right >> 0) & 1ULL) : 0; // top-right from top_right, bit 0

        // Current row: for b==63, left neighbor comes from mid_current and right neighbor from mid_right.
        live_neighbors += (int)((mid_current >> (b - 1)) & 1ULL);                // left neighbor (bit 62)
        live_neighbors += (col < words_per_row - 1) ? (int)((mid_right >> 0) & 1ULL) : 0; // right neighbor from mid_right, bit 0

        // Bottom row: for b==63, neighbors: (b-1) from bot_mid, (b) from bot_mid, and (b+1) from bot_right.
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> (b - 1)) & 1ULL) : 0; // bottom-left (bit 62)
        live_neighbors += (row < grid_dimensions - 1) ? (int)((bot_mid >> b) & 1ULL) : 0;         // bottom (bit 63)
        live_neighbors += (row < grid_dimensions - 1 && col < words_per_row - 1) ? (int)((bot_right >> 0) & 1ULL) : 0; // bottom-right

        int cell = (int)((mid_current >> b) & 1ULL);
        int next_cell = ((live_neighbors == 3) || (cell && live_neighbors == 2)) ? 1 : 0;
        result |= ((std::uint64_t)next_cell << b);
    }

    // Write the computed 64-bit result to the output grid.
    output[global_word_idx] = result;
}

// Host function that executes one step of Conway's Game of Life by launching the CUDA kernel.
// "input" and "output" are pointers to bit-packed grids residing in GPU memory.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    // Total number of 64-bit words in the entire grid.
    int total_words = grid_dimensions * words_per_row;

    // Set up CUDA launch configuration.
    const int blockSize = 256; // Threads per block.
    const int gridSize = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel. Host-device synchronization is assumed to be handled externally.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
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
