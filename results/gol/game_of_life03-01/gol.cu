#include <cstdint>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// This CUDA kernel computes one generation (one step) of Conway’s Game of Life.
// The input and output grids are bit‐packed: each std::uint64_t stores 64 cells,
// with each bit representing one cell (0 = dead, 1 = alive). In our layout the
// bits are numbered 0 to 63, where bit 0 corresponds to the leftmost cell and bit
// 63 to the rightmost cell in the word. For boundary cells within a word, adjacent
// words are consulted.
// Each thread processes one 64‐bit word (which represents 64 consecutive cells
// from one row).
//
// The eight neighbors of a cell in the current word come from three rows:
// the row above, the current row, and the row below. In each case the neighbor
// may be in an adjacent word if the cell is at the boundary (i.e. cell index 0 or 63).
// If a neighbor is outside the overall grid, it is considered dead (0).
//
// Conway’s rules are applied per cell: a cell is live in the next generation
// if (neighbors==3) or (neighbors==2 and the cell is already live).
//------------------------------------------------------------------------------
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row,
                         int total_words)
{
    // Compute the global word index (each word represents 64 cells).
    int word_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_index >= total_words) return;

    // Map the 1D word index to a 2D grid position.
    int row = word_index / words_per_row;
    int col = word_index % words_per_row;

    // For each cell we need the 3x3 neighborhood (excluding the center cell).
    // We load the three words (left, center, right) for the row above, the current row,
    // and the row below. If the neighbor row or word does not exist (grid boundary),
    // substitute 0.
    std::uint64_t a_left   = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t a_center = (row > 0)          ? input[(row - 1) * words_per_row + col]         : 0ULL;
    std::uint64_t a_right  = (row > 0 && col < words_per_row - 1)
                                ? input[(row - 1) * words_per_row + (col + 1)] : 0ULL;

    std::uint64_t c_left   = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t c_center = input[row * words_per_row + col];
    std::uint64_t c_right  = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;

    std::uint64_t b_left   = (row < grid_dim - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t b_center = (row < grid_dim - 1)          ? input[(row + 1) * words_per_row + col]         : 0ULL;
    std::uint64_t b_right  = (row < grid_dim - 1 && col < words_per_row - 1)
                                ? input[(row + 1) * words_per_row + (col + 1)] : 0ULL;

    // new_word will store the next state for the 64 cells in the current word.
    std::uint64_t new_word = 0ULL;

    // Process every cell (bit) in the current 64-bit word.
    // We assume that within a word the bit index i corresponds to the i-th cell,
    // with i==0 being the leftmost cell and i==63 being the rightmost.
    for (int i = 0; i < 64; i++) {
        int alive_neighbors = 0;

        // --- Row above ---
        // Top-left neighbor.
        if (i > 0)
            alive_neighbors += (int)((a_center >> (i - 1)) & 1ULL);
        else
            alive_neighbors += (int)((a_left >> 63) & 1ULL);
        // Top-center neighbor.
        alive_neighbors += (int)((a_center >> i) & 1ULL);
        // Top-right neighbor.
        if (i < 63)
            alive_neighbors += (int)((a_center >> (i + 1)) & 1ULL);
        else
            alive_neighbors += (int)((a_right >> 0) & 1ULL);

        // --- Current row (exclude center cell itself) ---
        // Left neighbor.
        if (i > 0)
            alive_neighbors += (int)((c_center >> (i - 1)) & 1ULL);
        else
            alive_neighbors += (int)((c_left >> 63) & 1ULL);
        // Right neighbor.
        if (i < 63)
            alive_neighbors += (int)((c_center >> (i + 1)) & 1ULL);
        else
            alive_neighbors += (int)((c_right >> 0) & 1ULL);

        // --- Row below ---
        // Bottom-left neighbor.
        if (i > 0)
            alive_neighbors += (int)((b_center >> (i - 1)) & 1ULL);
        else
            alive_neighbors += (int)((b_left >> 63) & 1ULL);
        // Bottom-center neighbor.
        alive_neighbors += (int)((b_center >> i) & 1ULL);
        // Bottom-right neighbor.
        if (i < 63)
            alive_neighbors += (int)((b_center >> (i + 1)) & 1ULL);
        else
            alive_neighbors += (int)((b_right >> 0) & 1ULL);

        // Read current cell state.
        bool current_alive = ((c_center >> i) & 1ULL) != 0;
        // Apply Conway's Game of Life rules:
        //  - A dead cell with exactly 3 live neighbors becomes live.
        //  - A live cell with 2 or 3 live neighbors stays live.
        bool next_alive = (alive_neighbors == 3) || (current_alive && alive_neighbors == 2);
        if (next_alive)
            new_word |= (1ULL << i); // Set bit i.
    }

    // Write computed word to the output grid.
    output[word_index] = new_word;
}

//------------------------------------------------------------------------------
// Host function that performs one simulation step of Conway’s Game of Life.
// The grid is square with grid_dimensions cells per side. The grid is stored
// in bit‐packed format: each std::uint64_t holds 64 consecutive cells along a row.
// The input and output pointers are assumed to be allocated on the device via cudaMalloc.
// This function sets up the CUDA kernel launch; any required host-device synchronization
// is handled by the caller.
//------------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row contains grid_dimensions cells, packed in words of 64 cells.
    // Therefore the number of 64-bit words per row is:
    int words_per_row = grid_dimensions / 64;
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Choose a reasonable block size.
    int blockSize = 256;
    int gridSize = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel. Note: We pass grid_dimensions, words_per_row, and total_words
    // as additional parameters to allow boundary checks.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions, words_per_row, total_words);
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
