#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdint> /// @FIXED

// In this implementation the grid is stored in a bit-packed format,
// where each unsigned int packs 32 cells (each cell is 1 bit).
// The grid is square with dimensions 'grid_dimensions' (which is a power of 2).
// Each row consists of (grid_dimensions / 32) unsigned ints.
// For each cell, the eight neighbors are fetched from adjacent words as needed,
// and the __popc intrinsic is used to count the number of set bits (alive neighbors)
// in an 8-bit mask built for each cell.

// Define block dimensions; each thread processes one 32-bit word (32 cells)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// CUDA kernel: computes one generation of Conway's Game of Life on a bit-packed grid
// without using shared or texture memory. Each thread handles a 32-bit word.
__global__ void game_of_life_bit_kernel(const unsigned int* input, unsigned int* output, int grid_dim) {
    // Compute number of 32-bit words per row.
    int words_per_row = grid_dim >> 5;  // equivalent to grid_dim / 32

    // Determine the current row and word index.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check grid bounds.
    if (row >= grid_dim || word_idx >= words_per_row)
        return;

    // Calculate the linear index for the current word.
    int index = row * words_per_row + word_idx;

    // Load the current word and its horizontal neighbors.
    unsigned int cur = input[index];
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load the words from the row above (if available).
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Load the words from the row below (if available).
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Prepare the output word.
    unsigned int out_word = 0;

    // Process each bit in the current 32-bit word.
    // Each bit corresponds to one cell.
    for (int bit = 0; bit < 32; bit++) {
        // Compute the absolute column index for the current cell.
        int col = (word_idx << 5) + bit;  // word_idx * 32 + bit
        if (col >= grid_dim)
            break;

        // Build an 8-bit mask for the 8 neighbors of this cell.
        // Bit positions in the mask (0 to 7) are assigned as follows:
        // 0: top-left, 1: top, 2: top-right,
        // 3: left,             4: right,
        // 5: bottom-left, 6: bottom, 7: bottom-right.
        unsigned int neighbor_mask = 0;

        // Top row neighbors.
        if (row > 0) {
            int nb;
            // Top-left neighbor.
            if (bit == 0)
                nb = (top_left >> 31) & 1;
            else
                nb = (top >> (bit - 1)) & 1;
            neighbor_mask |= nb; // assign to bit 0

            // Top neighbor.
            nb = (top >> bit) & 1;
            neighbor_mask |= (nb << 1); // assign to bit 1

            // Top-right neighbor.
            if (bit == 31)
                nb = (top_right >> 0) & 1;
            else
                nb = (top >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 2); // assign to bit 2
        }

        // Current row neighbors.
        {
            int nb;
            // Left neighbor.
            if (bit == 0)
                nb = (cur_left >> 31) & 1;
            else
                nb = (cur >> (bit - 1)) & 1;
            neighbor_mask |= (nb << 3); // assign to bit 3

            // Right neighbor.
            if (bit == 31)
                nb = (cur_right >> 0) & 1;
            else
                nb = (cur >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 4); // assign to bit 4
        }

        // Bottom row neighbors.
        if (row < grid_dim - 1) {
            int nb;
            // Bottom-left neighbor.
            if (bit == 0)
                nb = (bot_left >> 31) & 1;
            else
                nb = (bot >> (bit - 1)) & 1;
            neighbor_mask |= (nb << 5); // assign to bit 5

            // Bottom neighbor.
            nb = (bot >> bit) & 1;
            neighbor_mask |= (nb << 6); // assign to bit 6

            // Bottom-right neighbor.
            if (bit == 31)
                nb = (bot_right >> 0) & 1;
            else
                nb = (bot >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 7); // assign to bit 7
        }

        // Use the __popc intrinsic to count the number of alive neighbors.
        int count = __popc(neighbor_mask);

        // Retrieve the current cell's state.
        int cell = (cur >> bit) & 1;
        int next_state = 0;

        // Apply Conway's Game of Life rules:
        // - An alive cell survives if it has 2 or 3 alive neighbors.
        // - A dead cell becomes alive if it has exactly 3 alive neighbors.
        if (cell)
            next_state = (count == 2 || count == 3) ? 1 : 0;
        else
            next_state = (count == 3) ? 1 : 0;

        // Set the corresponding bit in the output word.
        out_word |= (next_state << bit);
    }

    // Write the computed 32-bit word to the output grid.
    output[index] = out_word;
}

// Host function that launches one simulation step of Conway's Game of Life
// on a bit-packed grid. The grid is represented as an array of unsigned ints,
// where each bit represents a cell.
/// @FIXED void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // grid_dimensions / 32

    // Set up the execution configuration: grid dimensions in x correspond to the
    // number of words per row, and in y correspond to the number of rows.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled externally.
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
