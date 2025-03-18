#include <cstdint>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// This CUDA kernel computes one Game‐of‐Life generation for a bit‐packed grid.
// Each 64‐bit unsigned integer encodes 64 consecutive cells (one bit per cell),
// where bit0 represents the leftmost cell and bit63 the rightmost cell in that word.
// Each CUDA thread processes one such word.
// 
// For each cell in the word (i.e. for each bit position from 0 to 63) the new state
// is computed by checking the eight neighbors (from the rows above, the same row,
// and the row below) according to Conway’s rules:
//   - A live cell (1) survives if it has 2 or 3 live neighbors.
//   - A dead cell (0) becomes live if it has exactly 3 live neighbors.
//   - Otherwise the cell is dead in the next generation.
//
// Because the grid is bit‐packed, a thread reading word “curr” (from row r, word-index c)
// must also load its horizontal neighbours “left” and “right” from the same row,
// and the three words each from the row above and below – taking great care on the
// boundary cells. In particular, for the leftmost cell in the current word (bit index 0)
// the left neighbour is taken from the rightmost bit (bit 63) of the adjacent word in the previous column,
// and similarly for the rightmost cell (bit index 63) using the adjacent word at the next column.
// (If a neighbour word is off‐grid, its value is taken as 0. This implements the “all outside cells are dead” rule.)
//
// To update a whole word simultaneously, we first compute for each of the eight directions a 64–bit
// "contribution" mask that is aligned with the current 64 cells. Then, for each bit position, we sum
// up the eight contributions (each one 0 or 1) to obtain the neighbour count.
// Finally the new state is given by:
//     new_state = ( (neighbour_count == 3)  ||  (current_state && neighbour_count == 2) )
// 
// Although there exist bit–parallel methods (using SIMD–like techniques) to compute a population
// count for multiple cells at once, here we use a loop over the 64 bits per word. This loop is unrolled
// (by pragma) so that each thread processes its 64 cells in registers. The extra cost of a 64–iteration loop
// is offset by the cost of global memory accesses.
//------------------------------------------------------------------------------
__global__ void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int grid_dimensions,  // number of cells per row (and number of rows)
    int words_per_row     // number of 64–bit words per row (i.e. grid_dimensions / 64)
)
{
    // Determine our word's (cell block's) coordinates:
    // x coordinate (word column) in the bit–packed representation.
    int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    // y coordinate (row index in cells)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process valid grid rows and word columns.
    if (row >= grid_dimensions || word_col >= words_per_row)
        return;
    
    // Compute the linear index into the input/output arrays.
    int idx = row * words_per_row + word_col;
    
    // Load the current word.
    std::uint64_t curr = input[idx];
    
    // Load horizontal neighbour words (for the same row).
    std::uint64_t left  = (word_col > 0) ? input[row * words_per_row + word_col - 1] : 0ULL;
    std::uint64_t right = (word_col < words_per_row - 1) ? input[row * words_per_row + word_col + 1] : 0ULL;
    
    // Load neighbour words from the row above (north) and below (south).
    std::uint64_t upper_center = (row > 0) ? input[(row - 1) * words_per_row + word_col] : 0ULL;
    std::uint64_t lower_center = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + word_col] : 0ULL;
    
    // For the row above, also load horizontal neighbours.
    std::uint64_t upper_left  = (row > 0 && word_col > 0) ? input[(row - 1) * words_per_row + word_col - 1] : 0ULL;
    std::uint64_t upper_right = (row > 0 && word_col < words_per_row - 1) ? input[(row - 1) * words_per_row + word_col + 1] : 0ULL;
    
    // For the row below, also load horizontal neighbours.
    std::uint64_t lower_left  = (row < grid_dimensions - 1 && word_col > 0) ? input[(row + 1) * words_per_row + word_col - 1] : 0ULL;
    std::uint64_t lower_right = (row < grid_dimensions - 1 && word_col < words_per_row - 1) ? input[(row + 1) * words_per_row + word_col + 1] : 0ULL;
    
    // Compute the neighbour contributions for each of the 8 directions.
    //
    // NOTE on shifting:
    // We use the convention that in a std::uint64_t, bit positions 0..63 represent cells 0..63 in a word.
    // A cell’s left neighbour is cell index (i-1) and its right neighbour is cell (i+1).
    // To align a neighbour from the same row:
    //   - For a left neighbour we shift the word left by 1 bit. (For bits 1..63, (curr << 1) delivers the appropriate neighbour.
    //     Bit 0 is then provided from the adjacent left word, by taking its bit 63.)
    //   - For a right neighbour we shift the word right by 1 bit, with bit 63 coming from the adjacent right word.
    //
    // Similar logic applies for diagonal neighbours.
    std::uint64_t nw = (upper_center << 1) | (((upper_left) >> 63) & 1ULL);
    std::uint64_t n  = upper_center; // directly above: same horizontal alignment.
    std::uint64_t ne = (upper_center >> 1) | (((upper_right) & 1ULL) << 63);
    
    std::uint64_t w  = (curr << 1) | (((left) >> 63) & 1ULL);
    std::uint64_t e  = (curr >> 1) | (((right) & 1ULL) << 63);
    
    std::uint64_t sw = (lower_center << 1) | (((lower_left) >> 63) & 1ULL);
    std::uint64_t s  = lower_center;
    std::uint64_t se = (lower_center >> 1) | (((lower_right) & 1ULL) << 63);
    
    // Compute next generation for the 64 cells in this word.
    // We process all 64 bit–lanes in a loop: for each bit position (cell) we count
    // the live neighbours and then apply the Game–of–Life rule:
    //   new_cell = 1 if (count == 3) or (current_cell == 1 and count == 2)
    std::uint64_t res = 0ULL;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++)
    {
        int count = 0;
        // Sum neighbor contributions from each of the 8 directions.
        count += (int)((nw >> bit) & 1ULL);
        count += (int)((n  >> bit) & 1ULL);
        count += (int)((ne >> bit) & 1ULL);
        count += (int)((w  >> bit) & 1ULL);
        count += (int)((e  >> bit) & 1ULL);
        count += (int)((sw >> bit) & 1ULL);
        count += (int)((s  >> bit) & 1ULL);
        count += (int)((se >> bit) & 1ULL);
        
        // Extract current cell state.
        int cell = (int)((curr >> bit) & 1ULL);
        
        // Apply Conway's rules.
        // A cell becomes (or remains) alive when it has exactly 3 live neighbours,
        // or it remains alive when it has exactly 2 live neighbours.
        int new_cell = (count == 3) || (cell && (count == 2));
        
        // Set the corresponding bit in the result word.
        res |= (std::uint64_t(new_cell) << bit);
    }
    
    // Write the computed word back to the output grid.
    output[idx] = res;
}

//------------------------------------------------------------------------------
// Host function that runs one step of Conway's Game of Life on a bit-packed grid.
// 
// Parameters:
//   input           - Pointer to device memory holding the current grid,
//                     where each std::uint64_t packs 64 cells.
//   output          - Pointer to device memory that will receive the next generation.
//   grid_dimensions - The width (and height) of the square grid (number of cells per side).
//                     (This value is always a power of 2.)
// 
// The function computes the number of 64–bit words per row as (grid_dimensions / 64)
// and then launches a 2D CUDA kernel such that each thread processes one word. Any
// host–device synchronization is assumed to be handled externally.
//------------------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                   std::uint64_t* output,
                                   int grid_dimensions)
{
    // Compute number of 64–cell words per row. Since grid_dimensions is a power of 2 and
    // is greater than 512, grid_dimensions is divisible by 64.
    int words_per_row = grid_dimensions / 64;
    
    // Configure CUDA grid and block dimensions.
    // We choose a 2D block of 16x16 threads.
    dim3 blockDim(16, 16);
    dim3 gridDim( (words_per_row + blockDim.x - 1) / blockDim.x,
                  (grid_dimensions + blockDim.y - 1) / blockDim.y );
    
    // Launch the Game of Life kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions, words_per_row);
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
