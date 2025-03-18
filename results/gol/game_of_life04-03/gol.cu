/*
This CUDA implementation computes one generation of Conway’s Game of Life.
The grid is a square with “grid_dimensions” cells per side.
Cells are stored in bit‐packed format: each std::uint64_t contains 64 cells (one per bit)
with the 0th bit corresponding to one cell and the 63rd bit corresponding to the 64th cell.
Each CUDA thread processes one 64‐bit word (i.e. 64 consecutive cells in a row).

For each cell in the word the kernel computes the number of live (bit value 1) neighbors,
taking into account the three rows: (row above, same row, row below) and
three corresponding word “columns” (left, current, right).
Special handling is applied for the boundary bits (bit 0 and bit 63) where neighbor cells
may reside in an adjacent word. Cells outside the grid are assumed dead.

The Game of Life rules are then applied:
  – A live cell survives if it has 2 or 3 live neighbors.
  – A dead cell becomes live if it has exactly 3 live neighbors.
The result for each 64‐cell block is stored in the output grid in bit‐packed format.

All performance–critical operations (loads from global memory and bit–manipulations)
are done in registers. No shared or texture memory is used.
*/

#include <cstdint>
#include <cuda_runtime.h>

// The kernel that computes one step of Game of Life on a bit–packed grid.
// Each thread processes one std::uint64_t word (64 cells).
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64–bit words per row.
    // (grid_dimensions is a power of two and always >= 512)
    int words_per_row = grid_dimensions >> 6;  // equivalent to grid_dimensions / 64
    int total_words = grid_dimensions * words_per_row;

    // Each thread processes one std::uint64_t word.
    int word_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(word_index >= total_words)
        return;

    // Determine the row and column (word index within the row).
    int row = word_index / words_per_row;
    int col = word_index % words_per_row;

    // Compute the offset for the current row.
    int row_offset = row * words_per_row;

    // Load the center word (current row, current word).
    std::uint64_t center = input[row_offset + col];

    // Load left and right words (same row) if available; otherwise substitute with 0.
    std::uint64_t left  = (col > 0)                 ? input[row_offset + col - 1] : 0;
    std::uint64_t right = (col < words_per_row - 1)   ? input[row_offset + col + 1] : 0;

    // For the row above.
    std::uint64_t a_center = 0, a_left = 0, a_right = 0;
    if(row > 0)
    {
        int a_offset = (row - 1) * words_per_row;
        a_center = input[a_offset + col];
        a_left   = (col > 0)               ? input[a_offset + col - 1] : 0;
        a_right  = (col < words_per_row - 1) ? input[a_offset + col + 1] : 0;
    }

    // For the row below.
    std::uint64_t b_center = 0, b_left = 0, b_right = 0;
    if(row < grid_dimensions - 1)
    {
        int b_offset = (row + 1) * words_per_row;
        b_center = input[b_offset + col];
        b_left   = (col > 0)               ? input[b_offset + col - 1] : 0;
        b_right  = (col < words_per_row - 1) ? input[b_offset + col + 1] : 0;
    }

    // Process each of the 64 cells in this word.
    std::uint64_t result = 0;
    // Unroll the loop since the iteration count is fixed.
#pragma unroll
    for (int i = 0; i < 64; i++)
    {
        int live_neighbors = 0;

        //----------------------
        // Top row neighbors
        //----------------------
        if(row > 0)
        {
            // Top-left: if i==0, use the last bit of a_left; otherwise the bit (i-1) from a_center.
            int top_left = (i == 0) ? ((a_left >> 63) & 1ULL) : ((a_center >> (i - 1)) & 1ULL);
            // Top: from a_center at bit i.
            int top = (a_center >> i) & 1ULL;
            // Top-right: if i==63, use the first bit of a_right; otherwise the bit (i+1) from a_center.
            int top_right = (i == 63) ? ((a_right >> 0) & 1ULL) : ((a_center >> (i + 1)) & 1ULL);
            live_neighbors += top_left + top + top_right;
        }

        //----------------------
        // Same row neighbors (excluding self)
        //----------------------
        {
            // Left neighbor: if i==0, use the last bit of "left"; otherwise the bit (i-1) of center.
            int mid_left = (i == 0) ? ((left >> 63) & 1ULL) : ((center >> (i - 1)) & 1ULL);
            // Right neighbor: if i==63, use the first bit of "right"; otherwise the bit (i+1) of center.
            int mid_right = (i == 63) ? ((right >> 0) & 1ULL) : ((center >> (i + 1)) & 1ULL);
            live_neighbors += mid_left + mid_right;
        }

        //----------------------
        // Bottom row neighbors
        //----------------------
        if(row < grid_dimensions - 1)
        {
            // Bottom-left: if i==0, use the last bit of b_left; otherwise bit (i-1) from b_center.
            int bottom_left = (i == 0) ? ((b_left >> 63) & 1ULL) : ((b_center >> (i - 1)) & 1ULL);
            // Bottom: from b_center at bit i.
            int bottom = (b_center >> i) & 1ULL;
            // Bottom-right: if i==63, use the first bit of b_right; otherwise bit (i+1) from b_center.
            int bottom_right = (i == 63) ? ((b_right >> 0) & 1ULL) : ((b_center >> (i + 1)) & 1ULL);
            live_neighbors += bottom_left + bottom + bottom_right;
        }

        //----------------------
        // Compute next state for cell at bit position i.
        //----------------------
        int current = (center >> i) & 1ULL;
        // Apply the Game of Life rules:
        // - A dead cell becomes alive if it has exactly 3 live neighbors.
        // - A live cell survives if it has 2 or 3 live neighbors.
        int new_state = (live_neighbors == 3) || (current && (live_neighbors == 2));

        // Set the computed state at bit i in the result.
        result |= (static_cast<std::uint64_t>(new_state) << i);
    }

    // Write the computed 64–bit word to the output grid.
    output[word_index] = result;
}

// The host function that launches one generation update of Conway’s Game of Life.
// "input" and "output" are pointers to device memory containing bit–packed grids.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64–bit words per grid row.
    int words_per_row = grid_dimensions / 64;
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Configure the CUDA kernel launch parameters.
    // Using 256 threads per block (a multiple of the warp size) for good performance.
    int blockSize = 256;
    int gridSize = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel; note that host–device synchronization is expected to be performed by the caller.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
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
