#include <cstdint>
#include <cuda_runtime.h>

//--------------------------------------------------------------------------
// Kernel: game_of_life_kernel
//
// Each thread processes one 64‐bit word, which represents 64 consecutive cells
// in the grid row. Since the grid is bit‐packed (1 bit per cell) and each row
// contains grid_dimensions cells, each row is stored in (grid_dimensions/64)
// words.  The thread computes the next state for every cell (bit) in its word,
// by checking the 8-neighbor cells (from the row above, same row, and row below).
//
// Boundary conditions: When a neighbor is requested from outside the grid,
// the corresponding word is assumed to be 0 (all dead).
//
// For each cell bit index (0..63), the neighbor cell value is extracted from the
// proper word. For the left neighbor (i-1) and right neighbor (i+1), special care
// is taken if the bit is at the word boundary (i==0 or i==63) by loading bits from
// the adjacent left or right word respectively.
// 
// The Conway's Game of Life update is then applied as follows:
//   - A live cell (bit==1) survives if it has 2 or 3 live neighbors.
//   - A dead cell (bit==0) becomes live if it has exactly 3 live neighbors.
//  
// The __popc intrinsic is mentioned in the prompt to help with bit-counting.
// In this implementation we extract each bit and sum the 8 neighbor contributions.
// Although that introduces a loop over 64 iterations per thread, the kernel is
// launched with a huge number of threads and the loop is unrolled for performance.
//--------------------------------------------------------------------------

__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dim,         // grid dimensions in cells (rows == grid_dim)
                                    int words_per_row)    // number of 64-bit words per row = grid_dim / 64
{
    // Determine the 2D coordinates of the current thread in terms of words.
    int col = blockIdx.x * blockDim.x + threadIdx.x; // word index within the row
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
    // Check that the thread is within the valid grid bounds.
    if (row >= grid_dim || col >= words_per_row)
        return;
    
    // Compute the index for the current word in the bit-packed grid.
    int idx = row * words_per_row + col;

    //----------------------------------------------------------------------
    // Load neighbor words for the three rows: previous (top), current, next (bottom).
    // For each row, we need three words: left, center, and right.
    // If a neighbor row or word is outside the grid boundaries, we use 0.
    //----------------------------------------------------------------------
    
    // Current row words.
    std::uint64_t cur_center = input[row * words_per_row + col];
    std::uint64_t cur_left   = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t cur_right  = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;
    
    // Top row words.
    std::uint64_t top_center = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_row = row - 1;
        int base = top_row * words_per_row;
        top_center = input[base + col];
        top_left   = (col > 0) ? input[base + (col - 1)] : 0ULL;
        top_right  = (col < words_per_row - 1) ? input[base + (col + 1)] : 0ULL;
    }
    
    // Bottom row words.
    std::uint64_t bot_center = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_row = row + 1;
        int base = bot_row * words_per_row;
        bot_center = input[base + col];
        bot_left   = (col > 0) ? input[base + (col - 1)] : 0ULL;
        bot_right  = (col < words_per_row - 1) ? input[base + (col + 1)] : 0ULL;
    }
    
    //----------------------------------------------------------------------
    // For each cell (bit position 0 to 63) in the current word,
    // count the number of live neighbors and apply the Game of Life rules.
    //----------------------------------------------------------------------
    std::uint64_t result = 0;
    
    // The loop is unrolled for performance. Each iteration handles one cell.
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        int neighbors = 0;
        
        // --- Top row neighbors ---
        // For the top row the neighbor positions relative to the center cell
        // in the current word:
        //   top-left: bit (i-1) in top_center, except if i==0 take bit 63 from top_left.
        //   top-center: bit i in top_center.
        //   top-right: bit (i+1) in top_center, except if i==63 take bit 0 from top_right.
        int top_l = (i == 0) ? ((top_left >> 63) & 1ULL) : ((top_center >> (i - 1)) & 1ULL);
        int top_c = ((top_center >> i) & 1ULL);
        int top_r = (i == 63) ? ((top_right >> 0) & 1ULL) : ((top_center >> (i + 1)) & 1ULL);
        neighbors += top_l + top_c + top_r;
        
        // --- Current row neighbors ---
        // Only the left and right neighbors count:
        //   left: bit (i-1) in cur_center, except if i==0 then use bit 63 from cur_left.
        //   right: bit (i+1) in cur_center, except if i==63 then use bit 0 from cur_right.
        int cur_l = (i == 0) ? ((cur_left >> 63) & 1ULL) : ((cur_center >> (i - 1)) & 1ULL);
        int cur_r = (i == 63) ? ((cur_right >> 0) & 1ULL) : ((cur_center >> (i + 1)) & 1ULL);
        neighbors += cur_l + cur_r;
        
        // --- Bottom row neighbors ---
        // Similarly for the bottom row:
        int bot_l = (i == 0) ? ((bot_left >> 63) & 1ULL) : ((bot_center >> (i - 1)) & 1ULL);
        int bot_c = ((bot_center >> i) & 1ULL);
        int bot_r = (i == 63) ? ((bot_right >> 0) & 1ULL) : ((bot_center >> (i + 1)) & 1ULL);
        neighbors += bot_l + bot_c + bot_r;
        
        // Read the state of the center cell at bit i.
        int cell = ((cur_center >> i) & 1ULL);
        
        // Apply Conway's Game of Life rules:
        // - A cell becomes (or remains) alive if it has exactly 3 neighbors,
        //   or if it is already alive and has exactly 2 neighbors.
        int new_cell = (neighbors == 3 || (cell && neighbors == 2)) ? 1 : 0;
        
        // Set the new cell state in the result word.
        result |= (static_cast<std::uint64_t>(new_cell) << i);
    }
    
    // Store the computed word into the output grid.
    output[idx] = result;
}

//--------------------------------------------------------------------------
// Host function: run_game_of_life
//
// This function launches a CUDA kernel that computes one iteration (generation)
// of Conway's Game of Life. The grid is given in bit-packed format (std::uint64_t*
// for both input and output), where each 64-bit word encodes 64 cells of a row.
// The grid is square with 'grid_dimensions' cells per side. 
//
// Launch configuration: Each thread handles one 64-bit word. The grid dimensions
// in words are: (grid_dimensions/64) along the horizontal direction and
// grid_dimensions rows.
//
// Host-device synchronization is assumed to be handled by the caller.
//--------------------------------------------------------------------------

/// @FIXED: extern "C" 

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate the number of 64-bit words that represent one row.
    // Since each std::uint64_t holds 64 cells, words_per_row = grid_dimensions / 64.
    int words_per_row = grid_dimensions / 64;
    
    // Choose block dimensions. A 16x16 thread block is a practical choice.
    dim3 blockDim(16, 16);
    
    // Calculate grid dimensions in terms of thread blocks.
    // In the X-direction, we need enough blocks to cover all words per row.
    // In the Y-direction, we need grid_dimensions blocks (each block handles a part of the rows).
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel to compute one Game of Life iteration.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions, words_per_row);
    
    // Note: Any necessary host-device synchronization (such as cudaDeviceSynchronize)
    // is assumed to be handled by the caller.
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
