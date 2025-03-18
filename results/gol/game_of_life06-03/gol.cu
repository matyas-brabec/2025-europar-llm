#include <cuda_runtime.h>
#include <cstdint>

//------------------------------------------------------------------------------
// This CUDA kernel implements one step of Conway’s Game of Life on a 2D grid.
// The grid is stored in bit‐packed form: each std::uint64_t holds 64 cells,
// with the following convention:
//   • Bit 0 represents the leftmost cell in the 64‐cell group,
//   • Bit 63 represents the rightmost cell.
// (This mapping was chosen so that simple bit shifts yield neighbor‐alignment:
//  shifting left (<< 1) “moves” a cell’s bit toward higher bit indices, which
//  corresponds to moving rightward in the grid; shifting right (>> 1) moves a cell’s
//  bit toward lower bit indices, i.e. leftward.)
//
// Each thread processes one 64‐cell word. The grid of bit-packed words has
// dimensions: grid_rows = grid_dimensions and grid_cols = grid_dimensions/64.
// Cells outside the grid are assumed dead (0). For boundary words, missing
// neighbors are replaced by 0. To correctly handle neighbors across word
// boundaries, we also use the adjacent word to the left (for the leftmost cell)
// and the adjacent word to the right (for the rightmost cell), applying an extra
// shift so that the correct neighbor bit is brought into position.
// 
// The eight neighbor contributions for each word are computed as follows:
//   • For the row above (if any): 
//       NW = (above-center << 1) OR (if available: above-left >> 63)
//       N  = above-center
//       NE = (above-center >> 1) OR (if available: above-right << 63)
//   • For the current row (neighbors only; do not include the cell itself):
//       W  = (curr << 1) OR (if available: left_curr >> 63)
//       E  = (curr >> 1) OR (if available: right_curr << 63)
//   • For the row below (if any):
//       SW = (below-center << 1) OR (if available: below-left >> 63)
//       S  = below-center
//       SE = (below-center >> 1) OR (if available: below-right << 63)
// 
// After obtaining these eight 64‐bit “neighbor” masks, each thread loops over
// the 64 bit positions (each representing a cell) and computes the neighbor count
// (an integer between 0 and 8). The Game of Life rule is then applied:
//   new cell = (neighbors == 3) OR (current cell AND (neighbors == 2))
// The resulting bit is stored into the correct bit position of the output word.
//------------------------------------------------------------------------------

// __global__ CUDA kernel. Each thread processes one 64-cell word.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // Number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;
    
    // Compute the global thread index.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Determine the row and column (word index) in the grid.
    int row = idx / words_per_row;
    int col = idx % words_per_row;

    // Load the current word from the input grid.
    std::uint64_t curr = input[idx];

    // For horizontal neighbors in the current row.
    std::uint64_t left_curr  = (col > 0) ? input[row * words_per_row + (col - 1)] : 0;
    std::uint64_t right_curr = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0;

    // For neighbors in the row above.
    std::uint64_t above_center = (row > 0) ? input[(row - 1) * words_per_row + col] : 0;
    std::uint64_t above_left   = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0;
    std::uint64_t above_right  = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0;

    // For neighbors in the row below.
    std::uint64_t below_center = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0;
    std::uint64_t below_left   = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0;
    std::uint64_t below_right  = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0;

    // Calculate neighbor contributions for the row above.
    // For NW: For each cell in the current word:
    //   If not at left edge (cell index > 0), NW comes from above_center at index (cell-1) – achieved by (above_center << 1).
    //   For the leftmost cell (cell index 0), use the rightmost bit of above_left, i.e. (above_left >> 63).
    std::uint64_t NW = (above_center << 1) | ((above_left)  ? (above_left >> 63) : 0);
    // North neighbor: no horizontal shift.
    std::uint64_t N  = above_center;
    // For NE: If not at right edge, NE comes from above_center at index (cell+1) – achieved by (above_center >> 1).
    //         For the rightmost cell, use the leftmost bit of above_right, i.e. (above_right << 63).
    std::uint64_t NE = (above_center >> 1) | ((above_right) ? (above_right << 63) : 0);

    // Calculate neighbor contributions for the current row (only horizontal neighbors).
    std::uint64_t W = (curr << 1) | ((left_curr)  ? (left_curr >> 63) : 0);
    std::uint64_t E = (curr >> 1) | ((right_curr) ? (right_curr << 63) : 0);

    // Calculate neighbor contributions for the row below.
    std::uint64_t SW = (below_center << 1) | ((below_left)  ? (below_left >> 63) : 0);
    std::uint64_t S  = below_center;
    std::uint64_t SE = (below_center >> 1) | ((below_right) ? (below_right << 63) : 0);

    // Now, for each cell (each bit position in the 64-bit word),
    // sum the contributions from the eight neighbors and apply the Game of Life rules.
    std::uint64_t new_word = 0;
    // We process each bit position (cell) independently.
    // Since each neighbor mask has only 0 or 1 in each cell's bit, summing them gives the neighbor count.
    for (int bit = 0; bit < 64; bit++) {
        int count = 0;
        count += (int)((NW >> bit) & 1ULL);
        count += (int)((N  >> bit) & 1ULL);
        count += (int)((NE >> bit) & 1ULL);
        count += (int)((W  >> bit) & 1ULL);
        count += (int)((E  >> bit) & 1ULL);
        count += (int)((SW >> bit) & 1ULL);
        count += (int)((S  >> bit) & 1ULL);
        count += (int)((SE >> bit) & 1ULL);

        // Extract the state of the current cell.
        int cell = (int)((curr >> bit) & 1ULL);
        // Apply Conway’s Game of Life rules:
        // A cell becomes live if it has exactly 3 live neighbors, or if it is already live and has exactly 2 neighbors.
        int new_cell = (count == 3) || (cell && (count == 2));
        if (new_cell)
            new_word |= (1ULL << bit);
    }

    output[idx] = new_word;
}

//------------------------------------------------------------------------------
// Host function that runs one simulation step of Conway’s Game of Life.
// - input: Pointer to the input grid (dynamically allocated on device via cudaMalloc).
//          Each std::uint64_t packs 64 cells, where a bit value of 1 represents a live cell.
// - output: Pointer to the output grid (also allocated on device).
// - grid_dimensions: The width (and height) of the square grid in cells (always a power of 2).
//
// This function prepares the kernel launch configuration and calls the kernel.
// Any necessary host-device synchronization is handled by the caller.
//------------------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Each row of cells is stored using grid_dimensions/64 64-bit words.
    int words_per_row = grid_dimensions / 64;
    int total_words = grid_dimensions * words_per_row;

    // Launch parameters; adjust block size for performance.
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
    // Note: Host-device synchronization (cudaDeviceSynchronize) is assumed to be performed by the caller.
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
