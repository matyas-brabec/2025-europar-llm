#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for one simulation step of Conway's Game of Life.
// Each thread processes one std::uint64_t word encoding 64 cells.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    // Each row has grid_dimensions cells. Each word packs 64 cells.
    // Compute number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Compute the global word index handled by this thread.
    int word_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dimensions * words_per_row;
    if (word_index >= total_words)
        return;

    // Determine the row and the column (word index within the row).
    int row = word_index / words_per_row;
    int col = word_index % words_per_row;

    // Load the current word (cell states in the current row).
    std::uint64_t curr = input[row * words_per_row + col];

    // Load neighbor words from the same row.
    std::uint64_t left = (col > 0) ? input[row * words_per_row + (col - 1)] : 0;
    std::uint64_t right = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0;

    // Load words from the row above.
    std::uint64_t up_curr = (row > 0) ? input[(row - 1) * words_per_row + col] : 0;
    std::uint64_t up_left = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0;
    std::uint64_t up_right = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0;

    // Load words from the row below.
    std::uint64_t down_curr = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0;
    std::uint64_t down_left = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0;
    std::uint64_t down_right = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0;

    // Compute next state for each of the 64 cells in the current word.
    std::uint64_t out = 0;
    // Loop over each bit (cell) in the 64-bit word.
    // For cell at bit position 'j', we compute the number of alive neighbor cells.
    for (int j = 0; j < 64; j++) {
        // Build an 8-bit value where each bit represents one neighbor's state.
        // The order of bits in the neighbor byte is irrelevant; we only need the popcount.
        unsigned int neighbor_byte = 0;

        // For the row above:
        // Northwest neighbor.
        if (j == 0)
            neighbor_byte |= (unsigned int)((up_left >> 63) & 1ULL);
        else
            neighbor_byte |= (unsigned int)((up_curr >> (j - 1)) & 1ULL);
        // North neighbor.
        neighbor_byte |= (unsigned int)(((up_curr >> j) & 1ULL) << 1);
        // Northeast neighbor.
        if (j == 63)
            neighbor_byte |= ((unsigned int)((up_right >> 0) & 1ULL) << 2);
        else
            neighbor_byte |= ((unsigned int)(((up_curr >> (j + 1)) & 1ULL) << 2));

        // For the same row (excluding the center cell itself):
        // West neighbor.
        if (j == 0)
            neighbor_byte |= (unsigned int)((left >> 63) & 1ULL) << 3;
        else
            neighbor_byte |= (unsigned int)(((curr >> (j - 1)) & 1ULL) << 3);
        // East neighbor.
        if (j == 63)
            neighbor_byte |= (unsigned int)((right >> 0) & 1ULL) << 4;
        else
            neighbor_byte |= (unsigned int)(((curr >> (j + 1)) & 1ULL) << 4);

        // For the row below:
        // Southwest neighbor.
        if (j == 0)
            neighbor_byte |= (unsigned int)((down_left >> 63) & 1ULL) << 5;
        else
            neighbor_byte |= (unsigned int)(((down_curr >> (j - 1)) & 1ULL) << 5);
        // South neighbor.
        neighbor_byte |= (unsigned int)(((down_curr >> j) & 1ULL) << 6);
        // Southeast neighbor.
        if (j == 63)
            neighbor_byte |= (unsigned int)((down_right >> 0) & 1ULL) << 7;
        else
            neighbor_byte |= (unsigned int)(((down_curr >> (j + 1)) & 1ULL) << 7);

        // Use the __popc intrinsic to count the number of bits set in neighbor_byte.
        // __popc operates on 32-bit integers.
        int count = __popc(neighbor_byte);

        // Extract the current cell's state.
        int cell = (int)((curr >> j) & 1ULL);

        // Apply Conway's Game of Life rules:
        //   - A live cell with 2 or 3 live neighbors survives.
        //   - A dead cell with exactly 3 live neighbors becomes alive.
        bool new_alive = (count == 3) || (cell && (count == 2));
        if (new_alive)
            out |= (1ULL << j);
    }

    // Store the computed next state word in the output grid.
    output[word_index] = out;
}

// Host function that launches one simulation step of Conway's Game of Life on the GPU.
// The grids (input and output) are bit-packed: each std::uint64_t contains 64 cells.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Choose a block size (number of threads per block).
    int block_size = 256;
    // Compute the number of blocks needed.
    int grid_size = (total_words + block_size - 1) / block_size;

    // Launch the kernel. Host-device synchronization and error checking are assumed to be done externally.
    game_of_life_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
