// Includes necessary CUDA and C++ standard headers.
#include <cuda_runtime.h>
#include <cstdint>

// The following CUDA kernel computes one generation (step) of Conway's Game of Life.
// The grid is a square of grid_dimensions x grid_dimensions cells. The cell states are
// stored in a bit‐packed format: each std::uint64_t value holds 64 consecutive cells
// in one row (with bit 0 representing the left‐most cell of that 64–cell group, and bit 63 the right–most).
// For a given cell, its eight neighbors are checked (cells outside the grid are considered dead).
// The kernel assigns one 64–cell word per thread.
__global__ void game_of_life_kernel(const std::uint64_t *input, std::uint64_t *output, int grid_dim) {
    // Each row has grid_dim cells and thus words_per_row = grid_dim/64 words.
    int words_per_row = grid_dim >> 6;  // division by 64

    // Compute the cell (word) coordinates in the grid.
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Row index (0 <= row < grid_dim)
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // Column index in bit-packed words (0 <= col < words_per_row)
    
    // Out-of-bound threads exit.
    if (row >= grid_dim || col >= words_per_row)
        return;

    // Compute linear index for the current 64-bit word.
    int idx = row * words_per_row + col;
    // Load the 64-bit word corresponding to the current row and block of 64 cells.
    std::uint64_t curr = input[idx];

    // For horizontal neighbors in the same row, load the previous and next words if present.
    std::uint64_t mid_prev = (col > 0) ? input[idx - 1] : 0ULL;
    std::uint64_t mid_next = (col < words_per_row - 1) ? input[idx + 1] : 0ULL;
    
    // Boolean flags for horizontal boundaries.
    bool has_left  = (col > 0);
    bool has_right = (col < words_per_row - 1);
    
    // For the vertical neighbors, determine if the top and bottom rows exist.
    bool has_top    = (row > 0);
    bool has_bottom = (row < grid_dim - 1);
    
    // Load the corresponding words for the row above (top) if available.
    std::uint64_t top_curr  = has_top ? input[(row - 1) * words_per_row + col] : 0ULL;
    std::uint64_t top_prev  = (has_top && has_left)  ? input[(row - 1) * words_per_row + col - 1] : 0ULL;
    std::uint64_t top_next  = (has_top && has_right) ? input[(row - 1) * words_per_row + col + 1] : 0ULL;
    
    // Load the corresponding words for the row below (bottom) if available.
    std::uint64_t bottom_curr = has_bottom ? input[(row + 1) * words_per_row + col] : 0ULL;
    std::uint64_t bottom_prev = (has_bottom && has_left)  ? input[(row + 1) * words_per_row + col - 1] : 0ULL;
    std::uint64_t bottom_next = (has_bottom && has_right) ? input[(row + 1) * words_per_row + col + 1] : 0ULL;
    
    // Compute the next state for each of the 64 cells in the current word.
    std::uint64_t result = 0;
    
    // Loop over each bit position (cell index within the 64-bit word).
    // The pragma unroll suggests the compiler unrolls this fixed-size loop.
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        // Extract the current cell state (0 for dead, 1 for alive).
        int cell = (curr >> bit) & 1ULL;
        int neighbor_count = 0;
        
        // Process top row neighbors (if the top row exists).
        if (has_top) {
            // Top-left neighbor:
            // If not at the left boundary of this word, use the same (top_curr) word.
            // Otherwise, if there is a left word available, fetch bit 63 from top_prev.
            if (bit > 0)
                neighbor_count += (int)((top_curr >> (bit - 1)) & 1ULL);
            else
                neighbor_count += (has_left ? (int)((top_prev >> 63) & 1ULL) : 0);
                
            // Top neighbor (center above).
            neighbor_count += (int)((top_curr >> bit) & 1ULL);
            
            // Top-right neighbor:
            // If not at the right boundary of this word, use top_curr.
            // Otherwise, if a right word exists, use bit 0 from top_next.
            if (bit < 63)
                neighbor_count += (int)((top_curr >> (bit + 1)) & 1ULL);
            else
                neighbor_count += (has_right ? (int)((top_next >> 0) & 1ULL) : 0);
        }
        
        // Process same row neighbors (left and right).
        // Left neighbor in the current row:
        if (bit > 0)
            neighbor_count += (int)((curr >> (bit - 1)) & 1ULL);
        else
            neighbor_count += (has_left ? (int)((mid_prev >> 63) & 1ULL) : 0);
        
        // Right neighbor in the current row:
        if (bit < 63)
            neighbor_count += (int)((curr >> (bit + 1)) & 1ULL);
        else
            neighbor_count += (has_right ? (int)((mid_next >> 0) & 1ULL) : 0);
        
        // Process bottom row neighbors (if the bottom row exists).
        if (has_bottom) {
            // Bottom-left neighbor:
            if (bit > 0)
                neighbor_count += (int)((bottom_curr >> (bit - 1)) & 1ULL);
            else
                neighbor_count += (has_left ? (int)((bottom_prev >> 63) & 1ULL) : 0);
            
            // Bottom neighbor (center below).
            neighbor_count += (int)((bottom_curr >> bit) & 1ULL);
            
            // Bottom-right neighbor:
            if (bit < 63)
                neighbor_count += (int)((bottom_curr >> (bit + 1)) & 1ULL);
            else
                neighbor_count += (has_right ? (int)((bottom_next >> 0) & 1ULL) : 0);
        }
        
        // Apply Conway's Game of Life rules:
        //  - Any live cell with 2 or 3 live neighbors survives.
        //  - Any dead cell with exactly 3 live neighbors becomes alive.
        int new_state = 0;
        if (cell)
            new_state = (neighbor_count == 2 || neighbor_count == 3);
        else
            new_state = (neighbor_count == 3);
        
        // Set the bit corresponding to the new state.
        result |= ((std::uint64_t)new_state << bit);
    }
    
    // Write the computed 64-bit word (representing 64 cells) to the output grid.
    output[idx] = result;
}

// The host function run_game_of_life executes one step of Conway's Game of Life on the GPU.
// This function is extern "C" to ensure C linkage.
/// @FIXED: extern "C" {

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Choose a 2D block configuration; 16x16 threads per block is a common optimal choice.
    dim3 blockDim(16, 16);
    // The grid dimensions in x-direction cover the word blocks per row,
    // and in y-direction cover the number of rows.
    dim3 gridDim( (words_per_row + blockDim.x - 1) / blockDim.x,
                  (grid_dimensions + blockDim.y - 1) / blockDim.y );
    
    // Launch the kernel.
    // The caller is responsible for any host-device synchronization.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
}

/// @FIXED: } // extern "C"} // } //


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
