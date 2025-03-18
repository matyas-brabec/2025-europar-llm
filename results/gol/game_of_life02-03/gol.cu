#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one iteration of Conway’s Game of Life over a grid whose cells are
// bit‐packed into std::uint64_t words (64 cells per word). Each thread processes one 64‐bit word,
// corresponding to 64 consecutive cells in a row. The grid is square with grid_dimensions cells per side.
// Each row is stored in (grid_dimensions/64) words. For a cell at global column = (word_index*64 + bit),
// the 8 neighbors come from the row above, the same row (excluding the cell itself), and the row below.
// For neighbor cells at the boundaries of a 64‐bit word, the corresponding adjacent word in the same row
// is used (if available), otherwise the neighbor is treated as dead (0).
//
// The kernel uses prefetching of the relevant neighbor words based on row and word boundaries to avoid
// costly conditionals within the inner loop that iterates over the 64 bits. Special handling is applied
// for bit positions 0 and 63; bits 1 through 62 are processed in a tight loop.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions,
                                    int words_per_row) {
    // Compute our position in terms of grid rows and word index within a row.
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int wordIx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || wordIx >= words_per_row)
        return;

    // Determine existence of neighboring rows and words.
    bool has_top    = (row > 0);
    bool has_bottom = (row < grid_dimensions - 1);
    bool has_left   = (wordIx > 0);
    bool has_right  = (wordIx < words_per_row - 1);

    // Prefetch neighbor words for the row above (if available).
    std::uint64_t top_left   = has_top && has_left ? input[(row - 1) * words_per_row + (wordIx - 1)] : 0ULL;
    std::uint64_t top_center = has_top ? input[(row - 1) * words_per_row + wordIx] : 0ULL;
    std::uint64_t top_right  = has_top && has_right ? input[(row - 1) * words_per_row + (wordIx + 1)] : 0ULL;

    // Prefetch neighbor words for the current row.
    std::uint64_t curr_left  = has_left ? input[row * words_per_row + (wordIx - 1)] : 0ULL;
    std::uint64_t curr       = input[row * words_per_row + wordIx];
    std::uint64_t curr_right = has_right ? input[row * words_per_row + (wordIx + 1)] : 0ULL;

    // Prefetch neighbor words for the row below (if available).
    std::uint64_t bottom_left   = has_bottom && has_left ? input[(row + 1) * words_per_row + (wordIx - 1)] : 0ULL;
    std::uint64_t bottom_center = has_bottom ? input[(row + 1) * words_per_row + wordIx] : 0ULL;
    std::uint64_t bottom_right  = has_bottom && has_right ? input[(row + 1) * words_per_row + (wordIx + 1)] : 0ULL;

    // Initialize the result for this 64-bit word.
    std::uint64_t result = 0ULL;
    int count, cell, new_state;

    // Handle bit index 0 separately (left boundary within the word).
    count = 0;
    if (has_top) {
        // For the top row, the left neighbor of bit 0 is from top_left (bit 63) if available,
        // otherwise the cell at top_center, bit 0 is the direct top neighbor and bit 1 is the top-right.
        count += has_left ? (int)((top_left >> 63) & 1ULL) : 0;
        count += (int)((top_center >> 0) & 1ULL);
        count += (int)((top_center >> 1) & 1ULL);
    }
    // Current row: left neighbor is in curr_left (bit 63) if available; right neighbor in curr (bit 1).
    count += has_left ? (int)((curr_left >> 63) & 1ULL) : 0;
    count += (int)((curr >> 1) & 1ULL);
    if (has_bottom) {
        count += has_left ? (int)((bottom_left >> 63) & 1ULL) : 0;
        count += (int)((bottom_center >> 0) & 1ULL);
        count += (int)((bottom_center >> 1) & 1ULL);
    }
    cell = (int)((curr >> 0) & 1ULL);
    new_state = (count == 3 || (cell && count == 2)) ? 1 : 0;
    result |= ((std::uint64_t)new_state << 0);

    // Process bits 1 to 62.
    for (int i = 1; i < 63; i++) {
        count = 0;
        if (has_top) {
            count += (int)((top_center >> (i - 1)) & 1ULL);
            count += (int)((top_center >> i) & 1ULL);
            count += (int)((top_center >> (i + 1)) & 1ULL);
        }
        // Current row: check left and right neighbors.
        count += (int)((curr >> (i - 1)) & 1ULL);
        count += (int)((curr >> (i + 1)) & 1ULL);
        if (has_bottom) {
            count += (int)((bottom_center >> (i - 1)) & 1ULL);
            count += (int)((bottom_center >> i) & 1ULL);
            count += (int)((bottom_center >> (i + 1)) & 1ULL);
        }
        cell = (int)((curr >> i) & 1ULL);
        new_state = (count == 3 || (cell && count == 2)) ? 1 : 0;
        result |= ((std::uint64_t)new_state << i);
    }

    // Handle bit index 63 separately (right boundary within the word).
    count = 0;
    if (has_top) {
        count += (int)((top_center >> 62) & 1ULL);
        count += (int)((top_center >> 63) & 1ULL);
        count += has_right ? (int)((top_right >> 0) & 1ULL) : 0;
    }
    count += (int)((curr >> 62) & 1ULL);
    count += has_right ? (int)((curr_right >> 0) & 1ULL) : 0;
    if (has_bottom) {
        count += (int)((bottom_center >> 62) & 1ULL);
        count += (int)((bottom_center >> 63) & 1ULL);
        count += has_right ? (int)((bottom_right >> 0) & 1ULL) : 0;
    }
    cell = (int)((curr >> 63) & 1ULL);
    new_state = (count == 3 || (cell && count == 2)) ? 1 : 0;
    result |= ((std::uint64_t)new_state << 63);

    // Write the computed 64-bit result into the output grid.
    output[row * words_per_row + wordIx] = result;
}

// Host function to execute one step of Conway’s Game of Life on the GPU.
// The grid is bit-packed so that each std::uint64_t holds 64 cells.
// grid_dimensions: the width (and height) of the square grid, which is a power of 2.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions) {
    // Calculate the number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;

    // Define CUDA block dimensions.
    // Each thread processes one word (64 cells) of a row.
    const dim3 blockSize(16, 16);
    const dim3 gridSize((words_per_row + blockSize.x - 1) / blockSize.x,
                        (grid_dimensions + blockSize.y - 1) / blockSize.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled by the caller.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions, words_per_row);
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
