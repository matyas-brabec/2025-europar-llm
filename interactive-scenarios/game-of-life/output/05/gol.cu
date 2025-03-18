#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdint> /// @FIXED

// In this optimized version, the grid is stored in a bit‐packed row‐based encoding,
// where each 32‐bit unsigned int packs 32 cells (one bit per cell).
// Each row consists of (grid_dimensions / 32) unsigned ints.
// To improve performance, each thread processes an entire 32‐bit word,
// but instead of iterating over each cell (32 iterations),
// the kernel processes 2 cells at a time (16 iterations per word).
// For each 2‐cell group, a 4‐bit “window” is extracted from each of the three rows
// (top, current, and bottom). This 4‐bit window covers the two cells plus one extra
// neighbor on each side. For each cell within the group, its 3×3 neighborhood
// is contained in 9 bits from the corresponding window (the two overlapping windows
// for the two cells are shifted by one bit). The __popc intrinsic is then used to
// count the alive neighbors in the 9‐bit mask.
// The new state for each cell is computed according to the standard rules:
//   - An alive cell survives if it has 2 or 3 alive neighbors.
//   - A dead cell becomes alive if it has exactly 3 alive neighbors.
// All cells outside the grid are considered dead.

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Inline helper to extract a 4-bit window from a row, given the current word and its horizontal neighbors.
// 'word'      : the 32-bit word for the row.
// 'left_word' : the word immediately to the left (or 0 if none).
// 'right_word': the word immediately to the right (or 0 if none).
// 'base'      : the starting bit position in 'word' for the 2-cell group (must be in [0, 31]).
// The desired window covers bits [base-1, base+2] (4 bits total), taking bits from adjacent words if needed.
__device__ inline unsigned int extract_window(unsigned int word, unsigned int left_word, unsigned int right_word, int base) {
    unsigned int window = 0;
    // Bit position -1 (left neighbor of first cell in the group)
    if (base == 0) {
        // Use the least-significant bit of the left word’s most-significant position.
        window |= ((left_word >> 31) & 1U) << 0;
    } else {
        window |= ((word >> (base - 1)) & 1U) << 0;
    }
    // Bits for positions corresponding to the two cells and the right neighbor of the group.
    // These come from the current word if they are in bounds.
    // Bit at position 0 in the group window corresponds to cell at index 'base'
    if (base < 32)
        window |= ((word >> base) & 1U) << 1;
    else
        window |= 0U << 1;
    // Next cell in the group.
    if (base + 1 < 32)
        window |= ((word >> (base + 1)) & 1U) << 2;
    else
        window |= 0U << 2;
    // Bit for the neighbor to the right of the group (cell at index base+2).
    if (base + 2 < 32)
        window |= ((word >> (base + 2)) & 1U) << 3;
    else {
        // If out-of-bound, take from right_word (bit 0)
        window |= ((right_word >> 0) & 1U) << 3;
    }
    return window; // window is in range [0,15]
}

// CUDA kernel that computes one generation of Conway's Game of Life on a bit‐packed grid.
// Each thread processes one 32‐bit word (32 cells) of the grid.
// Instead of processing one cell at a time, the kernel processes groups of 2 cells simultaneously
// (16 groups per word), using 9‐bit neighborhoods and __popc to count alive neighbors.
__global__ void game_of_life_bit_2cells_kernel(const unsigned int* input, unsigned int* output, int grid_dim) {
    // Number of 32-bit words per row.
    int words_per_row = grid_dim >> 5;  // grid_dim / 32

    // Compute the row index and the word index within that row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;

    // Calculate the index for the current 32-bit word.
    int index = row * words_per_row + word_idx;

    // Load the current word and its horizontal neighbors.
    unsigned int cur = input[index];
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load the corresponding words from the row above.
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    // Load the corresponding words from the row below.
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Prepare the output word.
    unsigned int out_word = 0;

    // Process the 32 cells in groups of 2 cells (16 groups per word).
    // 'base' is the starting bit index for the 2-cell group within the 32-bit word.
    for (int group = 0; group < 16; group++) {
        int base = group * 2;

        // Extract 4-bit windows from top, current, and bottom rows.
        unsigned int top_window   = extract_window(top, top_left, top_right, base);
        unsigned int cur_window   = extract_window(cur, cur_left, cur_right, base);
        unsigned int bot_window   = extract_window(bot, bot_left, bot_right, base);

        // For the two cells in the group, their 3x3 neighborhoods are taken from overlapping bits:
        // For cell 0 in the group, the neighborhood uses bits [0,1,2] from each window.
        // For cell 1 in the group, the neighborhood uses bits [1,2,3] from each window.
        // Build 9-bit masks for each cell.
        unsigned int mask0 = 0;
        unsigned int mask1 = 0;

        // Top row contributions.
        mask0 |= (top_window & 0x7) << 0;         // bits 0..2 for cell0
        mask1 |= ((top_window >> 1) & 0x7) << 0;    // bits 1..3 for cell1
        // Current row contributions (skip the center cell).
        // For cell0, take bit 0 and bit 2.
        mask0 |= ((cur_window >> 0) & 1) << 3;
        mask0 |= ((cur_window >> 2) & 1) << 4;
        // For cell1, take bit 1 and bit 3.
        mask1 |= ((cur_window >> 1) & 1) << 3;
        mask1 |= ((cur_window >> 3) & 1) << 4;
        // Bottom row contributions.
        mask0 |= (bot_window & 0x7) << 5;         // bits 0..2 for cell0
        mask1 |= ((bot_window >> 1) & 0x7) << 5;    // bits 1..3 for cell1

        // Count alive neighbors using __popc on the 9-bit masks.
        // __popc operates on 32-bit integers, so the 9-bit mask is in the lower bits.
        int count0 = __popc(mask0);
        int count1 = __popc(mask1);

        // The current state of the cells is in cur_window:
        // For cell0: bit 1, for cell1: bit 2.
        int cell0 = (cur_window >> 1) & 1;
        int cell1 = (cur_window >> 2) & 1;

        // Compute next state using Game of Life rules.
        // Alive cell survives if neighbor count is 2 or 3.
        // Dead cell becomes alive if neighbor count is exactly 3.
        int next0 = (cell0 && (count0 == 2 || count0 == 3)) || (!cell0 && (count0 == 3));
        int next1 = (cell1 && (count1 == 2 || count1 == 3)) || (!cell1 && (count1 == 3));

        // Set the corresponding bits in the output word.
        out_word |= (next0 << base) | (next1 << (base + 1));
    }

    // Write the computed 32-bit word to global memory.
    output[index] = out_word;
}

// Host function to execute one generation of Conway's Game of Life on a bit‐packed grid,
// using the optimized kernel that processes 2 cells simultaneously per thread.
// 'input' and 'output' are pointers to device memory (allocated via cudaMalloc),
// and 'grid_dimensions' is the width/height of the square grid.
/// @FIXED void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5; // grid_dimensions / 32

    // Define block dimensions.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    // Calculate grid dimensions (number of blocks) to cover the entire grid.
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled externally.
    /// @FIXED game_of_life_bit_2cells_kernel<<<grid, block>>>(input, output, grid_dimensions);
    game_of_life_bit_2cells_kernel<<<grid, block>>>((const unsigned int*)input, (unsigned int*)output, grid_dimensions);
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
