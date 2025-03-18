#include <cuda_runtime.h>
#include <cstdint>

// The CUDA kernel that computes one generation of Conway's Game of Life.
// Each thread processes one 64‐bit word (i.e. 64 consecutive cells in a row).
// The grid is bit‐packed. Each word covers 64 cells in a row.
// The word at (row, word_idx) has neighboring cells coming from three rows:
// the row above (if any), the current row, and the row below (if any).
// In each row the left and right neighbor bits may come from an adjacent word if
// the cell being processed is at bit index 0 (left boundary) or 63 (right boundary).
//
// For each cell, we count the live neighbors from the 8 directions.
// Then we apply the Game of Life rules:
//   • A dead cell with exactly 3 live neighbors becomes live.
//   • A live cell with 2 or 3 live neighbors stays live. Otherwise it dies.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dim,       // grid dimensions in cells (square grid)
                                    int words_per_row)  // grid_dim / 64: number of 64-bit words per row
{
    // Compute the global word index this thread is responsible for.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dim * words_per_row;
    if (idx >= total_words)
        return;

    // Determine the row and column (word index within the row) for this thread.
    int row = idx / words_per_row;
    int col = idx % words_per_row;

    // Load the current word from the input grid.
    std::uint64_t curr_word = input[row * words_per_row + col];

    // For the current row, load the adjacent words to the left and right.
    // If at the boundary of the row, substitute 0 (all dead cells).
    std::uint64_t left_word  = (col > 0) ? input[row * words_per_row + col - 1] : 0;
    std::uint64_t right_word = (col < words_per_row - 1) ? input[row * words_per_row + col + 1] : 0;

    // For the row above: if we are on the top row, use 0; otherwise, load the three words.
    std::uint64_t above_center = 0, above_left = 0, above_right = 0;
    if (row > 0)
    {
        above_center = input[(row - 1) * words_per_row + col];
        above_left   = (col > 0) ? input[(row - 1) * words_per_row + col - 1] : 0;
        above_right  = (col < words_per_row - 1) ? input[(row - 1) * words_per_row + col + 1] : 0;
    }

    // For the row below: if we are on the bottom row, use 0; otherwise, load the three words.
    std::uint64_t below_center = 0, below_left = 0, below_right = 0;
    if (row < grid_dim - 1)
    {
        below_center = input[(row + 1) * words_per_row + col];
        below_left   = (col > 0) ? input[(row + 1) * words_per_row + col - 1] : 0;
        below_right  = (col < words_per_row - 1) ? input[(row + 1) * words_per_row + col + 1] : 0;
    }

    // The result word for this thread (each bit representing the next state of a cell).
    std::uint64_t result_word = 0;

    // Process each of the 64 bits (cells) in the current word.
    // We use loop unrolling to help the compiler optimize this inner loop.
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit)
    {
        // Extract the current cell state (0 for dead, 1 for live).
        int current = (curr_word >> bit) & 1ULL;

        // Count live neighbors from the 8 surrounding cells.
        int n = 0;

        // --- Top row neighbors ---
        // Top-left neighbor.
        if (row > 0)
        {
            if (bit == 0)
                n += (above_left >> 63) & 1ULL;
            else
                n += (above_center >> (bit - 1)) & 1ULL;
        }
        // Top center neighbor.
        if (row > 0)
        {
            n += (above_center >> bit) & 1ULL;
        }
        // Top-right neighbor.
        if (row > 0)
        {
            if (bit == 63)
                n += (above_right >> 0) & 1ULL;
            else
                n += (above_center >> (bit + 1)) & 1ULL;
        }

        // --- Same row neighbors (exclude the cell itself) ---
        // Left neighbor.
        if (bit == 0)
            n += (left_word >> 63) & 1ULL;
        else
            n += (curr_word >> (bit - 1)) & 1ULL;

        // Right neighbor.
        if (bit == 63)
            n += (right_word >> 0) & 1ULL;
        else
            n += (curr_word >> (bit + 1)) & 1ULL;

        // --- Bottom row neighbors ---
        // Bottom-left neighbor.
        if (row < grid_dim - 1)
        {
            if (bit == 0)
                n += (below_left >> 63) & 1ULL;
            else
                n += (below_center >> (bit - 1)) & 1ULL;
        }
        // Bottom center neighbor.
        if (row < grid_dim - 1)
        {
            n += (below_center >> bit) & 1ULL;
        }
        // Bottom-right neighbor.
        if (row < grid_dim - 1)
        {
            if (bit == 63)
                n += (below_right >> 0) & 1ULL;
            else
                n += (below_center >> (bit + 1)) & 1ULL;
        }

        // Apply the Game of Life rules:
        // - A cell becomes alive if it has exactly 3 live neighbors,
        //   or if it is already alive and has exactly 2 live neighbors.
        int new_state = (n == 3) || (current && (n == 2));

        // Set the corresponding bit in the result_word.
        result_word |= (static_cast<std::uint64_t>(new_state) << bit);
    }

    // Write the computed 64-bit word to the output grid.
    output[row * words_per_row + col] = result_word;
}

// Host function to run one step of Conway's Game of Life on the GPU.
// The grid is square with dimensions grid_dimensions x grid_dimensions (in cells).
// The grid is bit-packed: each std::uint64_t holds 64 consecutive cells in a row.
// The input and output pointers point to device memory (allocated with cudaMalloc).
//
// This function calculates the number of 64-bit words per row as grid_dimensions / 64,
// sets up the CUDA kernel launch parameters, and calls the kernel.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Choose a block size (number of threads per block); 256 is a common choice.
    int block_size = 256;
    int num_blocks = (total_words + block_size - 1) / block_size;

    // Launch the CUDA kernel.
    game_of_life_kernel<<<num_blocks, block_size>>>(input, output, grid_dimensions, words_per_row);
    
    // Note: Any host-device synchronization is handled by the caller.
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
