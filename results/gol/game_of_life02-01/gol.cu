#include <cuda_runtime.h>
#include <cstdint>

// The CUDA kernel that computes one generation (step) of Conway's Game of Life.
// Since the grid is bit-packed, each thread processes one 64‐bit word (64 consecutive cells in a row).
// For each cell (bit) in the 64‐bit word, the kernel computes the number of alive neighbors
// by reading the corresponding bit from the three rows (above, current, and below)
// and from the left and right neighboring words when necessary. Boundary conditions (rows and words)
// that are outside the grid are considered dead (0).
// The rules of Game of Life are then applied:
//    - Any live cell with two or three live neighbors survives.
//    - Any dead cell with exactly three live neighbors becomes alive.
//    - Otherwise, the cell is dead in the next generation.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions,
                                    int num_words_per_row)
{
    // Determine the 2D position in the grid (row and word index).
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int wordIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread maps to a valid word in the grid.
    if (row >= grid_dimensions || wordIdx >= num_words_per_row)
        return;

    // Calculate the starting index for this row in the bit-packed grid.
    int row_start = row * num_words_per_row;

    // Load the current word for the row.
    std::uint64_t cur = input[row_start + wordIdx];

    // Load neighboring words from the same row.
    std::uint64_t cur_left  = (wordIdx > 0) ? input[row_start + wordIdx - 1] : 0ULL;
    std::uint64_t cur_right = (wordIdx < num_words_per_row - 1) ? input[row_start + wordIdx + 1] : 0ULL;

    // For the row above, if it exists, load the corresponding words.
    std::uint64_t above = 0ULL, above_left = 0ULL, above_right = 0ULL;
    if (row > 0)
    {
        int above_start = (row - 1) * num_words_per_row;
        above        = input[above_start + wordIdx];
        above_left   = (wordIdx > 0) ? input[above_start + wordIdx - 1] : 0ULL;
        above_right  = (wordIdx < num_words_per_row - 1) ? input[above_start + wordIdx + 1] : 0ULL;
    }

    // For the row below, if it exists, load the corresponding words.
    std::uint64_t below = 0ULL, below_left = 0ULL, below_right = 0ULL;
    if (row < grid_dimensions - 1)
    {
        int below_start = (row + 1) * num_words_per_row;
        below        = input[below_start + wordIdx];
        below_left   = (wordIdx > 0) ? input[below_start + wordIdx - 1] : 0ULL;
        below_right  = (wordIdx < num_words_per_row - 1) ? input[below_start + wordIdx + 1] : 0ULL;
    }

    // The output word that will hold the next state for this 64-bit block.
    std::uint64_t out_word = 0ULL;

    // Process each bit in the 64-bit word.
    // Each bit represents a cell state (alive=1, dead=0).
    // We iterate over all 64 cells in the word.
#pragma unroll
    for (int k = 0; k < 64; k++)
    {
        int nb = 0; // Number of alive neighbors.

        // --- Process the row above (if exists) ---
        if (row > 0)
        {
            // Above-left neighbor.
            if (k == 0)
                nb += (above_left >> 63) & 1ULL;
            else
                nb += (above >> (k - 1)) & 1ULL;
            // Directly above neighbor.
            nb += (above >> k) & 1ULL;
            // Above-right neighbor.
            if (k == 63)
                nb += (above_right >> 0) & 1ULL;
            else
                nb += (above >> (k + 1)) & 1ULL;
        }

        // --- Process the same row (neighbors left and right) ---
        // Left neighbor.
        if (k == 0)
            nb += (cur_left >> 63) & 1ULL;
        else
            nb += (cur >> (k - 1)) & 1ULL;
        // Right neighbor.
        if (k == 63)
            nb += (cur_right >> 0) & 1ULL;
        else
            nb += (cur >> (k + 1)) & 1ULL;

        // --- Process the row below (if exists) ---
        if (row < grid_dimensions - 1)
        {
            // Below-left neighbor.
            if (k == 0)
                nb += (below_left >> 63) & 1ULL;
            else
                nb += (below >> (k - 1)) & 1ULL;
            // Directly below neighbor.
            nb += (below >> k) & 1ULL;
            // Below-right neighbor.
            if (k == 63)
                nb += (below_right >> 0) & 1ULL;
            else
                nb += (below >> (k + 1)) & 1ULL;
        }

        // Extract current cell's state.
        int cell = (cur >> k) & 1ULL;

        // Apply Conway's Game of Life rules:
        // - A dead cell with exactly 3 live neighbors becomes live.
        // - A live cell with 2 or 3 live neighbors remains live.
        // - All other cases yield a dead cell.
        int new_state = (nb == 3) || (cell && (nb == 2));

        // Set the bit in the output word if the cell is alive.
        if (new_state)
            out_word |= (1ULL << k);
    }

    // Write the updated 64-bit word to the output grid.
    output[row_start + wordIdx] = out_word;
}

// Host function that launches one step of Conway's Game of Life.
// "input" and "output" point to device memory containing the bit-packed grids.
// "grid_dimensions" is the size (width and height) of the square grid.
// Each row contains grid_dimensions bits, stored in grid_dimensions/64 std::uint64_t words.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Calculate the number of 64-bit words per row.
    int num_words_per_row = grid_dimensions >> 6;  // equivalent to grid_dimensions / 64

    // Choose a suitable block size; here we use 32 threads in x (covering words in a row)
    // and 8 threads in y (covering rows). These values can be tuned for target hardware.
    dim3 block(32, 8);
    
    // Calculate grid dimensions so that each thread processes one 64-bit word.
    dim3 grid((num_words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the CUDA kernel. Note that host-device synchronization is assumed to be handled externally.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions, num_words_per_row);
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
