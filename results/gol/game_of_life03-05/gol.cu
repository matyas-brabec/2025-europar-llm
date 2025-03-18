#include <cuda_runtime.h>
#include <cstdint>

//-----------------------------------------------------------------------------
// This CUDA kernel performs one generation update for Conway’s Game of Life
// on a bit‐packed 2D grid. Each 64‐bit word encodes 64 cells in a row.
// Each thread handles one 64‐bit word. The grid’s width (in cells) is assumed
// to be a power of two, and the number of 64‐bit words per row is grid_dimensions/64.
// The rules are applied to each cell using bit‐masks and bit shifts.
// Boundary cells (cells outside the grid) are assumed dead (0).
//
// For each cell in a 64‐bit word the code loads the 3 words of the row above,
// the current row, and the row below. In each row, the “left” and “right” words
// are used to correctly supply neighbor bits for the left‐most (bit 0) and right‐most
// (bit 63) cells, respectively.
// For a given cell the eight neighbors are:
//   Row above: (col-1, col, col+1)
//   Same row: (col-1,         col+1)   (skipping the cell itself)
//   Row below: (col-1, col, col+1)
// For the cell at bit position 0, if a neighbor with offset -1 is needed, the bit
// is taken from the corresponding word on the left (bit index 63). Similarly, for
// the cell at bit position 63, a neighbor with offset +1 is taken from the word on
// the right (bit index 0).
//
// The next cell value is set to live (1) if either:
//   - exactly three live neighbors are present, or
//   - the cell is currently live and exactly two neighbors are live.
// 
// The kernel avoids extra overhead by unrolling the per-word loop into three regions:
//   - handling bit 0 (left boundary of the word)
//   - handling bits 1 to 62 (middle of the word)
//   - handling bit 63 (right boundary of the word)
//-----------------------------------------------------------------------------
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                           std::uint64_t* __restrict__ output,
                           int grid_dimensions)
{
    // Each thread processes one 64-bit word.
    // The number of words per row is grid_dimensions/64.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Calculate the current row and word index in that row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int wordIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || wordIdx >= words_per_row)
        return;

    // Compute base indices for accessing the input.
    // Boundary condition: if a neighbor row or word is out-of-bound, treat it as 0.
    // Row above:
    std::uint64_t up_left  = (row > 0 && wordIdx > 0) ? input[(row - 1) * words_per_row + (wordIdx - 1)] : 0ULL;
    std::uint64_t up_mid   = (row > 0)            ? input[(row - 1) * words_per_row + wordIdx]         : 0ULL;
    std::uint64_t up_right = (row > 0 && wordIdx < words_per_row - 1)
                             ? input[(row - 1) * words_per_row + (wordIdx + 1)] : 0ULL;

    // Current row:
    std::uint64_t mid_left  = (wordIdx > 0) ? input[row * words_per_row + (wordIdx - 1)] : 0ULL;
    std::uint64_t mid_mid   = input[row * words_per_row + wordIdx];
    std::uint64_t mid_right = (wordIdx < words_per_row - 1)
                              ? input[row * words_per_row + (wordIdx + 1)] : 0ULL;

    // Row below:
    std::uint64_t down_left  = (row < grid_dimensions - 1 && wordIdx > 0)
                               ? input[(row + 1) * words_per_row + (wordIdx - 1)] : 0ULL;
    std::uint64_t down_mid   = (row < grid_dimensions - 1)
                               ? input[(row + 1) * words_per_row + wordIdx] : 0ULL;
    std::uint64_t down_right = (row < grid_dimensions - 1 && wordIdx < words_per_row - 1)
                               ? input[(row + 1) * words_per_row + (wordIdx + 1)] : 0ULL;

    std::uint64_t result_word = 0ULL;

    //----------------------------------------------------------------------------
    // Handle the leftmost bit (bit index 0) separately.
    //----------------------------------------------------------------------------
    {
        int count = 0;
        // Row above:
        // up-left: For bit 0, use left word's rightmost bit.
        count += (up_left >> 63) & 1;
        // up-center: from up_mid at bit 0.
        count += (up_mid >> 0) & 1;
        // up-right: from up_mid at bit 1.
        count += (up_mid >> 1) & 1;

        // Current row:
        // left: from mid_left (bit 63).
        count += (mid_left >> 63) & 1;
        // right: from mid_mid (bit 1).
        count += (mid_mid >> 1) & 1;

        // Row below:
        // down-left: from down_left (bit 63).
        count += (down_left >> 63) & 1;
        // down-center: from down_mid (bit 0).
        count += (down_mid >> 0) & 1;
        // down-right: from down_mid (bit 1).
        count += (down_mid >> 1) & 1;

        int cell = (mid_mid >> 0) & 1;
        // Apply Game of Life rules:
        // Live cell survives with two or three neighbors.
        // Dead cell becomes live if exactly three neighbors.
        int next = ((count == 3) || (cell && (count == 2))) ? 1 : 0;
        result_word |= (static_cast<std::uint64_t>(next) << 0);
    }

    //----------------------------------------------------------------------------
    // Handle middle bits: indices 1 through 62.
    // For these bits all neighbor accesses are to the mid word of each row.
    //----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 1; i <= 62; ++i)
    {
        int count = 0;
        // Row above: use up_mid for all neighbor bits.
        count += (up_mid >> (i - 1)) & 1;
        count += (up_mid >> i)       & 1;
        count += (up_mid >> (i + 1))   & 1;
        // Current row: skip center; use mid_mid for left and right.
        count += (mid_mid >> (i - 1)) & 1;
        count += (mid_mid >> (i + 1)) & 1;
        // Row below: use down_mid.
        count += (down_mid >> (i - 1)) & 1;
        count += (down_mid >> i)       & 1;
        count += (down_mid >> (i + 1))   & 1;

        int cell = (mid_mid >> i) & 1;
        int next = ((count == 3) || (cell && (count == 2))) ? 1 : 0;
        result_word |= (static_cast<std::uint64_t>(next) << i);
    }

    //----------------------------------------------------------------------------
    // Handle the rightmost bit (bit index 63) separately.
    //----------------------------------------------------------------------------
    {
        int count = 0;
        // Row above:
        // left: from up_mid (bit 62).
        count += (up_mid >> 62) & 1;
        // center: from up_mid (bit 63).
        count += (up_mid >> 63) & 1;
        // right: for bit 63, use up_right from bit 0.
        count += (up_right >> 0) & 1;

        // Current row:
        // left: from mid_mid (bit 62).
        count += (mid_mid >> 62) & 1;
        // right: from mid_right (bit 0).
        count += (mid_right >> 0) & 1;

        // Row below:
        count += (down_mid >> 62) & 1;
        count += (down_mid >> 63) & 1;
        count += (down_right >> 0) & 1;

        int cell = (mid_mid >> 63) & 1;
        int next = ((count == 3) || (cell && (count == 2))) ? 1 : 0;
        result_word |= (static_cast<std::uint64_t>(next) << 63);
    }

    // Write the computed 64-bit word back to the output grid.
    output[row * words_per_row + wordIdx] = result_word;
}

//-----------------------------------------------------------------------------
// run_game_of_life
//
// This host function launches the CUDA kernel to perform one simulation
// step of Conway’s Game of Life. The grid is stored in bit-packed format:
// each std::uint64_t holds 64 cells (each bit represents one cell).
//
// Parameters:
//   input           - Pointer to the input grid in device memory.
//   output          - Pointer to the output grid in device memory.
//   grid_dimensions - The width and height (in cells) of the square grid.
//                     (Assumed to be a power of 2, and greater than 512.)
//
// The kernel launch configuration is chosen such that each CUDA thread
// processes one 64-bit word (64 consecutive cells) from the grid.
// Host-device synchronization is assumed to be handled by the caller.
//-----------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions/64

    // Define block and grid dimensions.
    // Each thread processes one 64-bit word.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
    
    // Note: any required host-device synchronization (e.g., cudaDeviceSynchronize)
    // should be done by the caller of run_game_of_life.
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
