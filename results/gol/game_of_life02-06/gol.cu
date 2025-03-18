#include <cstdint>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Optimized CUDA kernel for a single step of Conway's Game of Life on a
// bit‐packed 2D grid. Each 64‐bit word encodes 64 consecutive cells in a row,
// with bit i representing the state of the cell at column (word_index*64 + i).
// 
// Each thread processes one 64‐bit word. To update the 64 cells in a word,
// the kernel loads not only the current word from the current row (center)
// but also adjacent words to the left and right, as well as the corresponding
// words from the row above and the row below. Boundary conditions (outside 
// the grid) are handled by substituting zero (dead cells).
//
// The update rule for each cell is as follows:
//   - A cell becomes alive in the next generation if:
//         (neighbor_count == 3) OR (neighbor_count == 2 and cell is alive)
//   - Otherwise the cell dies.
//
// For each of the 64 bits, the kernel computes the neighbor count by 
// conditionally reading from the proper 64-bit register using simple bit-shifts.
// Special handling is done for bits at positions 0 and 63, which rely on the 
// left/right adjacent words.
//------------------------------------------------------------------------------ 
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dim)
{
    // Each row has grid_dim cells, stored as (grid_dim / 64) 64-bit words.
    const int words_per_row = grid_dim / 64;

    // Determine the current thread's word index and row index.
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row      = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread is within grid bounds.
    if (row >= grid_dim || word_idx >= words_per_row)
        return;

    // Compute the linear offset for the current word.
    int offset = row * words_per_row + word_idx;

    // Load the center word for the current row.
    std::uint64_t current_center = input[offset];
    // Load the adjacent words in the current row if they exist; else use 0.
    std::uint64_t current_left  = (word_idx > 0) ? input[offset - 1] : 0ULL;
    std::uint64_t current_right = (word_idx < words_per_row - 1) ? input[offset + 1] : 0ULL;

    // Load the words from the row above.
    std::uint64_t upper_center = 0, upper_left = 0, upper_right = 0;
    if (row > 0) {
        int offset_up = (row - 1) * words_per_row + word_idx;
        upper_center = input[offset_up];
        upper_left   = (word_idx > 0) ? input[offset_up - 1] : 0ULL;
        upper_right  = (word_idx < words_per_row - 1) ? input[offset_up + 1] : 0ULL;
    }

    // Load the words from the row below.
    std::uint64_t lower_center = 0, lower_left = 0, lower_right = 0;
    if (row < grid_dim - 1) {
        int offset_down = (row + 1) * words_per_row + word_idx;
        lower_center = input[offset_down];
        lower_left   = (word_idx > 0) ? input[offset_down - 1] : 0ULL;
        lower_right  = (word_idx < words_per_row - 1) ? input[offset_down + 1] : 0ULL;
    }

    // Compute next generation for each of the 64 cells in the current word.
    std::uint64_t result = 0;

    // Unroll the loop to fully pipeline neighbor-count computation.
#pragma unroll
    for (int j = 0; j < 64; j++) {
        int count = 0;
        // Extract the current cell's state (true if alive).
        bool cell_alive = ((current_center >> j) & 1ULL) != 0;

        // For each cell, add contributions from its eight neighbors.
        // Handle boundary cells within the 64-bit word specially.
        if (j == 0) {
            // For the first cell in this 64-bit word, the left neighbor is in the adjacent word.
            // Neighbors from the row above:
            count += ((upper_left   >> 63) & 1ULL);   // upper left (from previous word)
            count += ((upper_center >> 0)  & 1ULL);      // upper center
            count += ((upper_center >> 1)  & 1ULL);      // upper right

            // Neighbors from the same row:
            count += ((current_left  >> 63) & 1ULL);     // left neighbor (from previous word)
            count += ((current_center >> 1)  & 1ULL);     // right neighbor

            // Neighbors from the row below:
            count += ((lower_left   >> 63) & 1ULL);      // lower left (from previous word)
            count += ((lower_center >> 0)  & 1ULL);       // lower center
            count += ((lower_center >> 1)  & 1ULL);       // lower right
        }
        else if (j == 63) {
            // For the last cell in this 64-bit word, the right neighbor is in the adjacent word.
            // Neighbors from the row above:
            count += ((upper_center >> 62) & 1ULL);       // upper left
            count += ((upper_center >> 63) & 1ULL);       // upper center
            count += ((upper_right  >> 0)  & 1ULL);        // upper right (from next word)

            // Neighbors from the same row:
            count += ((current_center >> 62) & 1ULL);     // left neighbor
            count += ((current_right  >> 0)  & 1ULL);      // right neighbor (from next word)

            // Neighbors from the row below:
            count += ((lower_center >> 62) & 1ULL);       // lower left
            count += ((lower_center >> 63) & 1ULL);       // lower center
            count += ((lower_right  >> 0)  & 1ULL);        // lower right (from next word)
        }
        else {
            // For cells not at the 64-bit boundary, all neighbors come from the center word.
            // Neighbors from the row above:
            count += ((upper_center >> (j - 1)) & 1ULL);  // upper left
            count += ((upper_center >> j)       & 1ULL);  // upper center
            count += ((upper_center >> (j + 1)) & 1ULL);  // upper right

            // Neighbors from the same row:
            count += ((current_center >> (j - 1)) & 1ULL); // left neighbor
            count += ((current_center >> (j + 1)) & 1ULL); // right neighbor

            // Neighbors from the row below:
            count += ((lower_center >> (j - 1)) & 1ULL);  // lower left
            count += ((lower_center >> j)       & 1ULL);  // lower center
            count += ((lower_center >> (j + 1)) & 1ULL);  // lower right
        }

        // Apply Conway's Game of Life rules:
        //   - A cell becomes alive if it has exactly 3 neighbors,
        //     or if it is currently alive and has exactly 2 neighbors.
        bool new_state = (count == 3) || (cell_alive && count == 2);

        // Set the computed bit in the result.
        result |= (static_cast<std::uint64_t>(new_state) << j);
    }

    // Write the result back to the output grid.
    output[offset] = result;
}

//------------------------------------------------------------------------------
// Host function that executes one step of Conway's Game of Life.
// The input and output grids are bit-packed and are assumed to have been allocated
// in GPU memory via cudaMalloc. The grid is square, with dimensions grid_dimensions,
// and each row consists of grid_dimensions/64 words, each word encoding 64 cells.
// 
// The performance of the kernel and simulation is the only concern; any data
// transformations needed (such as packing/unpacking) are not measured.
//------------------------------------------------------------------------------
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate the number of 64-bit words in each row.
    const int words_per_row = grid_dimensions / 64;

    // Choose a 2D block configuration.
    dim3 block(16, 16);

    // Calculate grid dimensions in blocks.
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel to perform one simulation step.
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
