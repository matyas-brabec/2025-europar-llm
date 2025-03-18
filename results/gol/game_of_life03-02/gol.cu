#include <cuda_runtime.h>
#include <cstdint>

//---------------------------------------------------------------------
// This __device__ inline helper function extracts a single bit from a
// bit-packed row given a horizontal offset. Each row is stored in a 64‐bit
// word ("current"). In case the requested column index (p + offset) falls
// outside [0,63], the function retrieves the corresponding bit from the
// neighbor word: if it’s less than 0, it uses the "left" word; if it’s 64
// or greater, it uses the "right" word. If the neighbor word does not
// exist (e.g. at the grid border), it has been pre-set to 0.
//---------------------------------------------------------------------
__device__ __forceinline__ int get_bit(std::uint64_t current, std::uint64_t left, std::uint64_t right, int p, int offset) {
    // Compute the effective column index.
    int col = p + offset;
    if (col < 0) {
        // For columns below 0, use bit 63 of the left neighbor word.
        return (int)((left >> 63) & 1ULL);
    } else if (col >= 64) {
        // For columns >= 64, use bit (col - 64) of the right neighbor word.
        return (int)((right >> (col - 64)) & 1ULL);
    } else {
        // Otherwise, return bit at index col from the current word.
        return (int)((current >> col) & 1ULL);
    }
}

//---------------------------------------------------------------------
// __global__ kernel to execute one iteration (generation) of Conway's
// Game of Life. Each CUDA thread handles one 64-bit word (64 consecutive
// cells in a row). The grid is bit-packed; each thread computes the 64
// cells in its word by examining the eight neighboring cells of each bit.
//---------------------------------------------------------------------
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    // Each row contains grid_dim cells, and since cells are bit-packed in 64-bit words,
    // the number of words (cells groups) per row is as follows.
    int words_per_row = grid_dim / 64;
    int total_words = grid_dim * words_per_row;

    // Compute the unique word index handled by this thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Determine the row and the word's column index in that row.
    int row = idx / words_per_row;
    int col_word = idx % words_per_row;

    // Compute the base index for the current row.
    int base = row * words_per_row;

    // Load the current 64-bit word.
    std::uint64_t cur = input[base + col_word];

    // Load the adjacent words in the same row.
    // For the left adjacent word (if col_word==0, it is outside the grid so we use 0).
    std::uint64_t cur_left  = (col_word > 0) ? input[base + col_word - 1] : 0;
    // For the right adjacent word.
    std::uint64_t cur_right = (col_word < words_per_row - 1) ? input[base + col_word + 1] : 0;

    // Load the words from the row above (if any). At grid boundaries, use 0.
    std::uint64_t up = 0, up_left = 0, up_right = 0;
    if (row > 0) {
        int base_up = (row - 1) * words_per_row;
        up       = input[base_up + col_word];
        up_left  = (col_word > 0) ? input[base_up + col_word - 1] : 0;
        up_right = (col_word < words_per_row - 1) ? input[base_up + col_word + 1] : 0;
    }

    // Load the words from the row below (if any). At grid boundaries, use 0.
    std::uint64_t down = 0, down_left = 0, down_right = 0;
    if (row < grid_dim - 1) {
        int base_down = (row + 1) * words_per_row;
        down       = input[base_down + col_word];
        down_left  = (col_word > 0) ? input[base_down + col_word - 1] : 0;
        down_right = (col_word < words_per_row - 1) ? input[base_down + col_word + 1] : 0;
    }

    // Variable to accumulate the updated state of the 64 cells.
    std::uint64_t result = 0;

    // Loop over all 64 bits (cells) in the current word.
    // Using #pragma unroll so that the loop is unrolled for performance.
#pragma unroll
    for (int p = 0; p < 64; ++p) {
        int neighborCount = 0;
        // Process the top row (neighbors above current cell).
        neighborCount += get_bit(up, up_left, up_right, p, -1);
        neighborCount += get_bit(up, up_left, up_right, p,  0);
        neighborCount += get_bit(up, up_left, up_right, p, +1);

        // Process the same row: only left and right neighbors.
        neighborCount += get_bit(cur, cur_left, cur_right, p, -1);
        neighborCount += get_bit(cur, cur_left, cur_right, p, +1);

        // Process the bottom row (neighbors below current cell).
        neighborCount += get_bit(down, down_left, down_right, p, -1);
        neighborCount += get_bit(down, down_left, down_right, p,  0);
        neighborCount += get_bit(down, down_left, down_right, p, +1);

        // Extract current cell state (0 = dead, 1 = alive).
        int cell = (int)((cur >> p) & 1ULL);
        int newState = 0;

        // Apply Conway's Game of Life rules:
        // - An alive cell survives if it has 2 or 3 live neighbors.
        // - A dead cell becomes alive only if it has exactly 3 live neighbors.
        if (cell) {
            newState = (neighborCount == 2 || neighborCount == 3) ? 1 : 0;
        } else {
            newState = (neighborCount == 3) ? 1 : 0;
        }

        // Set the computed state bit in the result word.
        result |= (std::uint64_t(newState) << p);
    }

    // Write the computed next generation word to the output grid.
    output[idx] = result;
}

//---------------------------------------------------------------------
// Host function that executes one time-step (generation) of Conway’s
// Game of Life using the CUDA kernel. The input and output grids are
// bit-packed (each 64-bit word encodes 64 cells) and are allocated on
// the device using cudaMalloc. Host-device synchronization is assumed
// to be handled by the caller.
//---------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions cells.
    // Since the cells are bit-packed into 64-bit words, the number of words per row is:
    int words_per_row = grid_dimensions / 64;
    // Total number of 64-bit words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Launch configuration: choose a block size (e.g., 256 threads per block)
    // and compute the grid size accordingly.
    int blockSize = 256;
    int gridSize = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel.
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
