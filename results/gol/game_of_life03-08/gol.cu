#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel that computes one step of Conway's Game of Life on a bit‐packed grid.
// Each thread processes one 64‐bit word (64 cells) from the grid.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row consists of grid_dimensions bits stored in (grid_dimensions/64) 64-bit words.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    int total_words = grid_dimensions * words_per_row;
    
    // Compute the global thread index.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;
    
    // Determine the current row and column (in terms of 64-bit words).
    int row = tid / words_per_row;
    int col = tid % words_per_row;
    
    // Load neighbor words for the row above.
    std::uint64_t up_center, up_left, up_right;
    if (row > 0) {
        int up_row = row - 1;
        up_center = input[up_row * words_per_row + col];
        up_left   = (col > 0) ? input[up_row * words_per_row + (col - 1)] : 0ULL;
        up_right  = (col < words_per_row - 1) ? input[up_row * words_per_row + (col + 1)] : 0ULL;
    } else {
        up_center = 0ULL;
        up_left = 0ULL;
        up_right = 0ULL;
    }
    
    // Load neighbor words for the current row.
    std::uint64_t mid_center = input[row * words_per_row + col];
    std::uint64_t mid_left = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t mid_right = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;
    
    // Load neighbor words for the row below.
    std::uint64_t bot_center, bot_left, bot_right;
    if (row < grid_dimensions - 1) {
        int bot_row = row + 1;
        bot_center = input[bot_row * words_per_row + col];
        bot_left   = (col > 0) ? input[bot_row * words_per_row + (col - 1)] : 0ULL;
        bot_right  = (col < words_per_row - 1) ? input[bot_row * words_per_row + (col + 1)] : 0ULL;
    } else {
        bot_center = 0ULL;
        bot_left = 0ULL;
        bot_right = 0ULL;
    }
    
    // Compute the new state for each of the 64 cells in the current 64-bit word.
    std::uint64_t new_word = 0;
    
    // Unroll loop for performance.
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
        // Retrieve the state (0 or 1) of the current cell.
        int cell = (mid_center >> bit) & 1;
        
        // Compute the number of alive neighbors.
        int count = 0;
        
        // Row above.
        count += (up_center >> bit) & 1;  // Directly above.
        if (bit == 0) {
            count += (up_left >> 63) & 1; // Top-left from previous word.
        } else {
            count += (up_center >> (bit - 1)) & 1; // Top-left from same word.
        }
        if (bit == 63) {
            count += (up_right >> 0) & 1; // Top-right from next word.
        } else {
            count += (up_center >> (bit + 1)) & 1; // Top-right from same word.
        }
        
        // Same row.
        if (bit == 0) {
            count += (mid_left >> 63) & 1; // Left neighbor from previous word.
        } else {
            count += (mid_center >> (bit - 1)) & 1; // Left neighbor from same word.
        }
        if (bit == 63) {
            count += (mid_right >> 0) & 1; // Right neighbor from next word.
        } else {
            count += (mid_center >> (bit + 1)) & 1; // Right neighbor from same word.
        }
        
        // Row below.
        count += (bot_center >> bit) & 1;  // Directly below.
        if (bit == 0) {
            count += (bot_left >> 63) & 1;  // Bottom-left from previous word.
        } else {
            count += (bot_center >> (bit - 1)) & 1;  // Bottom-left from same word.
        }
        if (bit == 63) {
            count += (bot_right >> 0) & 1;  // Bottom-right from next word.
        } else {
            count += (bot_center >> (bit + 1)) & 1;  // Bottom-right from same word.
        }
        
        // Apply Conway's Game of Life rules.
        int newState = 0;
        if (cell == 1) {
            // Alive cell survives only if it has 2 or 3 neighbors.
            newState = (count == 2 || count == 3) ? 1 : 0;
        } else {
            // Dead cell becomes alive if it has exactly 3 neighbors.
            newState = (count == 3) ? 1 : 0;
        }
        
        // Set the new state bit in the result word.
        new_word |= (static_cast<std::uint64_t>(newState) << bit);
    }
    
    // Write the updated word to global memory.
    output[tid] = new_word;
}

// Host function that launches one iteration of Conway's Game of Life.
// The input and output pointers point to bit-packed device memory allocated via cudaMalloc.
// The grid is a square of grid_dimensions x grid_dimensions cells.
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;
    
    // Set up CUDA kernel launch parameters.
    int threads_per_block = 256;
    int blocks = (total_words + threads_per_block - 1) / threads_per_block;
    
    // Launch the kernel to compute one generation.
    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions);
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
