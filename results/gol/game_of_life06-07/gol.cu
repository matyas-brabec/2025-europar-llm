#include <cuda_runtime.h>
#include <cstdint>

//------------------------------------------------------------------------------
// CUDA kernel for one iteration of Conway's Game of Life over a bit‐packed grid.
// Each element of the grid is a 64‐bit word encoding 64 cells (1 = alive, 0 = dead).
// Each thread processes one such word. For each of the 64 bit–cells in the word,
// the kernel computes the number of live neighbors (from the 8 surrounding cells)
// and then applies the Game of Life rule:
//    new_state = (neighbors == 3) || (current && (neighbors == 2))
// Special handling is applied at the boundaries of each word (bit 0 and bit 63)
// by fetching bits from the neighboring word(s) when available.
// For grid rows/columns out of bounds the cells are assumed dead (0).
// Performance is achieved by doing the neighbor‐gathering with shifts and bit–ORs,
// so that each thread works independently without atomics or shared memory.
//------------------------------------------------------------------------------
__global__ 
void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions cells packed 64 per word.
    // Therefore, number of words per row:
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Determine the cell location (row, word index in row) for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || word_col >= words_per_row)
        return;
        
    // Compute flat index into the bit-packed grid.
    int index = row * words_per_row + word_col;
    
    // Load the current word (64 cells).
    std::uint64_t cur = input[index];

    // For neighbor contributions, we need to load words from adjacent rows.
    // For out-of-bound rows or columns, use zero.
    // For each neighbor "direction", we shift the loaded word so that its bits
    // line up with the cell positions in the current word.
    
    // Top row pointers.
    std::uint64_t top = (row > 0) ? input[(row - 1) * words_per_row + word_col] : 0ULL;
    std::uint64_t top_left = 0, top_right = 0;
    if (row > 0) {
        // For top-left, shift the top word left by 1 bit so that bit i gets the value
        // from column (i-1) within the same row; additionally, if there is a previous word, 
        // the bit for i==0 comes from the neighbor word's bit 63.
        std::uint64_t top_current = top;
        std::uint64_t top_prev = (word_col > 0) ? input[(row - 1) * words_per_row + word_col - 1] : 0ULL;
        top_left = (top_current << 1) | (top_prev >> 63);
        
        // For top-right, shift the top word right by 1 bit; for bit i==63, borrow bit 0
        // from the next word if available.
        std::uint64_t top_next = (word_col < words_per_row - 1) ? input[(row - 1) * words_per_row + word_col + 1] : 0ULL;
        top_right = (top_current >> 1) | (top_next << 63);
    }
    
    // Bottom row pointers.
    std::uint64_t bottom = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + word_col] : 0ULL;
    std::uint64_t bottom_left = 0, bottom_right = 0;
    if (row < grid_dimensions - 1) {
        std::uint64_t bottom_current = bottom;
        std::uint64_t bottom_prev = (word_col > 0) ? input[(row + 1) * words_per_row + word_col - 1] : 0ULL;
        bottom_left = (bottom_current << 1) | (bottom_prev >> 63);
        
        std::uint64_t bottom_next = (word_col < words_per_row - 1) ? input[(row + 1) * words_per_row + word_col + 1] : 0ULL;
        bottom_right = (bottom_current >> 1) | (bottom_next << 63);
    }
    
    // Left and right neighbors from the same row.
    std::uint64_t left = ( (cur << 1) | ((word_col > 0) ? (input[row * words_per_row + word_col - 1] >> 63) : 0ULL) );
    std::uint64_t right = ( (cur >> 1) | ((word_col < words_per_row - 1) ? (input[row * words_per_row + word_col + 1] << 63) : 0ULL) );
    
    // The 8 neighbor contributions for each cell (bit position) are:
    // top_left, top, top_right, left, right, bottom_left, bottom, bottom_right.
    // We now compute the next state for each of the 64 cells in the word.
    std::uint64_t new_word = 0ULL;
    
    // Loop over all bit positions in the 64-bit word.
    // Note: The compiler will likely unroll this small fixed loop.
    for (int bit = 0; bit < 64; ++bit) {
        // Extract neighbor bits for the cell corresponding to 'bit'
        int count = 0;
        count += int((top_left >> bit) & 1ULL);
        count += int((top >> bit) & 1ULL);
        count += int((top_right >> bit) & 1ULL);
        count += int((left >> bit) & 1ULL);
        count += int((right >> bit) & 1ULL);
        count += int((bottom_left >> bit) & 1ULL);
        count += int((bottom >> bit) & 1ULL);
        count += int((bottom_right >> bit) & 1ULL);
        
        // Current cell state (0 or 1)
        int cur_state = int((cur >> bit) & 1ULL);
        
        // Apply Game of Life rule:
        //   Alive in next generation if exactly 3 neighbors,
        //   or if cell was alive and has exactly 2 neighbors.
        int next_state = ((count == 3) || (cur_state && (count == 2))) ? 1 : 0;
        
        // Set the corresponding bit in the new word.
        new_word |= (std::uint64_t(next_state) << bit);
    }
    
    // Write the computed word to the output grid.
    output[index] = new_word;
}

//------------------------------------------------------------------------------
// Host function: run_game_of_life
// Executes one step (generation) of Conway's Game of Life on a bit–packed grid.
//   input: pointer to device memory containing the current grid state (bit-packed).
//   output: pointer to device memory for the resulting grid state (bit-packed).
//   grid_dimensions: the grid is square with grid_dimensions rows and columns.
//                  (grid_dimensions is a power-of-2, and >512).
//
// The grid is arranged such that each row consists of (grid_dimensions/64) 64–bit words.
// Host–device synchronization is assumed to be handled by the caller.
//------------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions/64 words.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Set up block and grid dimensions.
    // Using 16x16 threads per block.
    dim3 block(16, 16);
    // Grid dimensions: x dimension covers word columns, y dimension covers rows.
    dim3 grid((words_per_row + block.x - 1) / block.x, (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    
    // Note: Any error checking or synchronization is assumed to be handled externally.
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
