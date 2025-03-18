#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdint> /// @FIXED

// In this implementation, the grid is represented in a bitwise encoding where each 32-bit word
// packs 32 cells. The grid is square with dimensions 'grid_dimensions'.
// Each row consists of (grid_dimensions / 32) unsigned ints.
// Bit ordering: bit 0 corresponds to the leftmost cell in the 32-bit word,
// and bit 31 corresponds to the rightmost cell.

// Define block dimensions for kernel execution. Each thread processes one 32-bit word (32 cells).
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// CUDA kernel that computes one generation of Conway's Game of Life on a bit-packed grid.
// Each thread handles one 32-bit word (i.e. 32 cells) from a given row.
__global__ void game_of_life_bit_kernel(const unsigned int* input, unsigned int* output, int grid_dimensions) {
    // Calculate number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // equivalent to grid_dimensions / 32

    // Compute the row index and the word index within that row for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check that we are within grid bounds.
    if (row >= grid_dimensions || word_idx >= words_per_row)
        return;

    // Compute the index into the input/output arrays.
    int index = row * words_per_row + word_idx;

    // Load the 32-bit word for the current row.
    unsigned int cur = input[index];

    // Load adjacent words for the current row to handle horizontal neighbors.
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load words from the row above.
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Load words from the row below.
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dimensions - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Initialize the output word that will store the next state for 32 cells.
    unsigned int out_word = 0;

    // Process each bit in the 32-bit word.
    for (int bit = 0; bit < 32; bit++) {
        // Calculate the absolute column index for this bit.
        int col = (word_idx << 5) + bit;  // word_idx * 32 + bit
        // Ensure we do not process bits beyond grid_dimensions.
        if (col >= grid_dimensions)
            break;

        // Extract the current cell state (0 for dead, 1 for alive).
        int cell = (cur >> bit) & 1;
        int neighbors = 0;

        // Define a lambda to extract a bit from a word with automatic handling of word boundaries.
        // If the requested bit index is -1, it fetches the rightmost bit from the left adjacent word.
        // If the index is 32, it fetches the leftmost bit from the right adjacent word.
        auto getBit = [](unsigned int word, unsigned int left_word, unsigned int right_word, int b) -> int {
            if (b < 0) {
                // For b == -1, get bit 31 from the left word.
                return (left_word >> 31) & 1;
            } else if (b >= 32) {
                // For b == 32, get bit 0 from the right word.
                return (right_word >> 0) & 1;
            } else {
                return (word >> b) & 1;
            }
        };

        // Count neighbors from the top row (if it exists).
        if (row > 0) {
            neighbors += getBit(top, top_left, top_right, bit - 1);
            neighbors += getBit(top, top_left, top_right, bit);
            neighbors += getBit(top, top_left, top_right, bit + 1);
        }

        // Count horizontal neighbors from the current row (excluding the cell itself).
        neighbors += getBit(cur, cur_left, cur_right, bit - 1);
        neighbors += getBit(cur, cur_left, cur_right, bit + 1);

        // Count neighbors from the bottom row (if it exists).
        if (row < grid_dimensions - 1) {
            neighbors += getBit(bot, bot_left, bot_right, bit - 1);
            neighbors += getBit(bot, bot_left, bot_right, bit);
            neighbors += getBit(bot, bot_left, bot_right, bit + 1);
        }

        // Apply Conway's Game of Life rules:
        // - An alive cell survives if it has 2 or 3 alive neighbors.
        // - A dead cell becomes alive if it has exactly 3 alive neighbors.
        int next_cell = 0;
        if (cell) {
            next_cell = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            next_cell = (neighbors == 3) ? 1 : 0;
        }

        // Set the bit in the output word if the next state is alive.
        out_word |= (next_cell << bit);
    }

    // Write the computed output word back to global memory.
    output[index] = out_word;
}

// Host function that launches one simulation step of Conway's Game of Life on a bit-packed grid.
// The grid is represented as an array of unsigned ints (each storing 32 cells).
// 'grid_dimensions' is the width/height of the square grid.
/// @FIXED void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // grid_dimensions / 32

    // Set up the execution configuration.
    // The grid dimension in x corresponds to the number of words per row,
    // and in y corresponds to the number of rows.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Any required host-device synchronization is handled externally.
    /// @FIXED game_of_life_bit_kernel<<<grid, block>>>(input, output, grid_dimensions);
    game_of_life_bit_kernel<<<grid, block>>>((const unsigned int*)input, (unsigned int*)output, grid_dimensions);
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
