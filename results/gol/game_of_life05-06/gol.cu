// Conway’s Game of Life with bit‐packed grids in CUDA.
// Each thread processes one 64‐cell word (std::uint64_t) of the grid.
// The grid is square with “grid_dimensions” cells per side and each row is stored in
// grid_width_words = grid_dimensions/64 words. Cells are stored with the following convention:
// Bit 0 is the leftmost cell in the word and bit 63 is the rightmost.
// For a cell at bit index i (0 <= i < 64):
//   - Its west (left) neighbor is at index (i–1) if i>0; if i==0 it comes from the previous word’s bit 63.
//   - Its east (right) neighbor is at index (i+1) if i<63; if i==63 it comes from the next word’s bit 0.
// In the vertical direction, similar boundary handling applies for the rows above and below.
// 
// The Game of Life update rule is applied per cell: a cell is live
// in the next generation if and only if either (a) exactly 3 of its 8 neighbors are live,
// or (b) the cell is already live and exactly 2 of its neighbors are live.
//
// The kernel is written to maximize performance by having each thread load all the needed
// neighboring words from global memory once, and then compute the new state for all 64 packed cells
// in a loop that is unrolled by the compiler. Although one might attempt to “vectorize” the inner loop
// via bit‐parallel arithmetic, handling the per–bit carries correctly is too intricate given the
// 4–bit fields required per cell. In practice, a well–unrolled fixed 64–iteration loop works best.
//
// All pointers (input and output) are assumed allocated with cudaMalloc. Host-device sync is done
// by the caller.

#include <cstdint>
#include <cuda_runtime.h>

// __global__ kernel that computes one game-of-life step.
/// @FIXED: extern "C" __global__
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dimensions)
{
    // Each row of cells is bit-packed into words.
    // Number of words per row.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions/64

    // Determine the word coordinates (x: column index in word-space, y: row index in cell-space).
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    int word_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(word_y >= grid_dimensions || word_x >= words_per_row) return;

    // Compute linear index for current word.
    int index = word_y * words_per_row + word_x;

    // Load the current row's word (the word that holds 64 cells).
    std::uint64_t mid_center = input[index];

    // For the current row we also need left/right neighboring words.
    std::uint64_t mid_left  = (word_x > 0) ? input[word_y * words_per_row + word_x - 1] : 0ULL;
    std::uint64_t mid_right = (word_x < words_per_row - 1) ? input[word_y * words_per_row + word_x + 1] : 0ULL;

    // For the row above (top row), if available.
    std::uint64_t top_center = 0ULL, top_left = 0ULL, top_right = 0ULL;
    if(word_y > 0)
    {
        int top_index = (word_y - 1) * words_per_row + word_x;
        top_center = input[top_index];
        top_left = (word_x > 0) ? input[(word_y - 1) * words_per_row + word_x - 1] : 0ULL;
        top_right = (word_x < words_per_row - 1) ? input[(word_y - 1) * words_per_row + word_x + 1] : 0ULL;
    }

    // For the row below (bottom row), if available.
    std::uint64_t bot_center = 0ULL, bot_left = 0ULL, bot_right = 0ULL;
    if(word_y < grid_dimensions - 1)
    {
        int bot_index = (word_y + 1) * words_per_row + word_x;
        bot_center = input[bot_index];
        bot_left = (word_x > 0) ? input[(word_y + 1) * words_per_row + word_x - 1] : 0ULL;
        bot_right = (word_x < words_per_row - 1) ? input[(word_y + 1) * words_per_row + word_x + 1] : 0ULL;
    }

    // The new state for the current 64 cells (packed in one word).
    std::uint64_t result = 0ULL;

    // Process each cell (each bit 0..63) in this word.
    // Convention: bit 0 is the leftmost cell, bit 63 is the rightmost.
    // Thus, for a cell at bit index i:
    //   West neighbor: if (i > 0) -> mid_center bit (i-1); if (i==0) -> mid_left bit 63.
    //   East neighbor: if (i < 63) -> mid_center bit (i+1); if (i==63) -> mid_right bit 0.
    // Similarly for top and bottom rows.
    #pragma unroll
    for (int i = 0; i < 64; i++)
    {
        int ncount = 0;

        // Top row neighbors, if available.
        if(word_y > 0)
        {
            // NW (top-west)
            if(i > 0)
                ncount += int((top_center >> (i - 1)) & 1ULL);
            else
                ncount += int((top_left >> 63) & 1ULL);
            // N (top-center)
            ncount += int((top_center >> i) & 1ULL);
            // NE (top-east)
            if(i < 63)
                ncount += int((top_center >> (i + 1)) & 1ULL);
            else
                ncount += int((top_right >> 0) & 1ULL); // bit0 of top_right
        }

        // Current row neighbors (do not count the center cell itself).
        // West neighbor.
        if(i > 0)
            ncount += int((mid_center >> (i - 1)) & 1ULL);
        else
            ncount += int((mid_left >> 63) & 1ULL);
        // East neighbor.
        if(i < 63)
            ncount += int((mid_center >> (i + 1)) & 1ULL);
        else
            ncount += int((mid_right >> 0) & 1ULL);

        // Bottom row neighbors, if available.
        if(word_y < grid_dimensions - 1)
        {
            // SW (bottom-west)
            if(i > 0)
                ncount += int((bot_center >> (i - 1)) & 1ULL);
            else
                ncount += int((bot_left >> 63) & 1ULL);
            // S (bottom-center)
            ncount += int((bot_center >> i) & 1ULL);
            // SE (bottom-east)
            if(i < 63)
                ncount += int((bot_center >> (i + 1)) & 1ULL);
            else
                ncount += int((bot_right >> 0) & 1ULL);
        }

        // Current cell state.
        int cell = int((mid_center >> i) & 1ULL);

        // Game of Life rule:
        // If exactly 3 neighbors, cell becomes live.
        // If cell is live and exactly 2 neighbors, cell survives.
        int next_cell = (ncount == 3) || (cell && (ncount == 2));

        // Set the i-th bit in the result accordingly.
        result |= (std::uint64_t(next_cell) << i);
    }

    // Write the new 64-cell word to the output.
    output[index] = result;
}

// Host function to execute one step of Conway’s Game of Life.
// - input: pointer to the current generation (bit-packed grid).
// - output: pointer to the next generation (bit-packed grid).
// - grid_dimensions: both the width and height (number of cells per side).
//
// The input and output pointers are assumed to have been allocated on the GPU with cudaMalloc.
// The caller is responsible for synchronizing host/device.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Each row has grid_dimensions/64 words.
    int words_per_row = grid_dimensions >> 6; // divide by 64

    // Launch a 2D grid of CUDA threads: one thread per 64-bit word.
    // Choose block dimensions tuned for performance.
    dim3 block(16, 16);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel (using the current CUDA stream).
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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
