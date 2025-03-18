#include <cuda_runtime.h>
#include <cstdint>

// In this implementation we assume that each 64‐bit word encodes 64 consecutive cells,
// with bit index 0 representing the left‐edge cell of the word and bit index 63 representing the right‐edge cell.
// Thus, for each cell, the left neighbor is cell index (i-1) (and if i==0 then from the previous word)
// and the right neighbor is cell index (i+1) (and if i==63 then from the next word).
//
// Each CUDA thread processes one 64‐bit word (i.e. one word of 64 cells) from the grid.
// The grid is logically arranged in rows, each row containing (grid_dimensions/64) words.
// For a given word, we must consider the three rows: the one above, current, and below.
// For each row, we load three words: the word itself, and its left/right neighbors (if available).
// If a neighbor row or word does not exist (boundary), we treat it as all 0’s.
// Then, for each bit position in the 64‐bit word, we compute the number of live neighbors
// by reading the appropriate bit from the 9 surrounding words (3 from the row above, 2 from same row, 3 from row below).
// The live/dead update is then computed as per Conway’s Game of Life rules:
//    newState = (neighbors == 3) || (current && neighbors == 2) 
// We use the __popc intrinsic to count bits set in an 8–bit mask that gathers the eight neighbor bits,
// which can significantly improve performance relative to computing 8 separate additions.


// Device helper function to extract bit from a 64-bit word at a given position.
// Position is assumed to be in [0,63]. 0 corresponds to the leftmost cell (edge requiring special handling) 
// and 63 corresponds to the rightmost cell.
__device__ __forceinline__ int get_bit(std::uint64_t word, int pos) {
    return int((word >> pos) & 1ULL);
}

// CUDA kernel that computes one update step of Conway's Game of Life.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate number of 64-bit words per row.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions/64

    // Compute thread's position in terms of grid row and word column.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only proceed if within valid bounds.
    if (row >= grid_dimensions || word_idx >= words_per_row)
        return;

    // Compute base index for the current row.
    int row_base = row * words_per_row;

    // Load the three adjacent words (current row): left, current, and right.
    std::uint64_t mid = input[row_base + word_idx];
    std::uint64_t mid_left = (word_idx > 0) ? input[row_base + word_idx - 1] : 0ULL;
    std::uint64_t mid_right = (word_idx < words_per_row - 1) ? input[row_base + word_idx + 1] : 0ULL;

    // For the top row (row-1), if exists load adjacent words; else use 0.
    std::uint64_t top = 0ULL, top_left = 0ULL, top_right = 0ULL;
    if (row > 0) {
        int top_base = (row - 1) * words_per_row;
        top = input[top_base + word_idx];
        top_left = (word_idx > 0) ? input[top_base + word_idx - 1] : 0ULL;
        top_right = (word_idx < words_per_row - 1) ? input[top_base + word_idx + 1] : 0ULL;
    }

    // For the bottom row (row+1), if exists load adjacent words; else use 0.
    std::uint64_t bot = 0ULL, bot_left = 0ULL, bot_right = 0ULL;
    if (row < grid_dimensions - 1) {
        int bot_base = (row + 1) * words_per_row;
        bot = input[bot_base + word_idx];
        bot_left = (word_idx > 0) ? input[bot_base + word_idx - 1] : 0ULL;
        bot_right = (word_idx < words_per_row - 1) ? input[bot_base + word_idx + 1] : 0ULL;
    }

    // Result word for the next generation.
    std::uint64_t res = 0ULL;

    // Process each of the 64 cells in the current word.
    // We treat bits [0,63] directly. For each cell at bit position i:
    //   - For horizontal neighbors, if i == 0 the left neighbor comes from the left-adjacent word,
    //     and if i == 63 the right neighbor comes from the right-adjacent word.
    //   - Similarly for top and bottom rows.
    for (int i = 0; i < 64; i++) {
        // Build an 8-bit mask representing the eight neighbor cells.
        // Bit positions in the mask:
        //   bit0: top-left
        //   bit1: top
        //   bit2: top-right
        //   bit3: left   (same row)
        //   bit4: right  (same row)
        //   bit5: bottom-left
        //   bit6: bottom
        //   bit7: bottom-right

        int tl = (i == 0) ? get_bit(top_left, 63) : get_bit(top, i - 1);
        int t  = get_bit(top, i); // top cell
        int tr = (i == 63) ? get_bit(top_right, 0) : get_bit(top, i + 1);

        int l  = (i == 0) ? get_bit(mid_left, 63) : get_bit(mid, i - 1);
        int r  = (i == 63) ? get_bit(mid_right, 0) : get_bit(mid, i + 1);

        int bl = (i == 0) ? get_bit(bot_left, 63) : get_bit(bot, i - 1);
        int b  = get_bit(bot, i);
        int br = (i == 63) ? get_bit(bot_right, 0) : get_bit(bot, i + 1);

        // Pack all eight neighbor bits into a single 32-bit integer.
        int neighbor_bits = (tl << 0) | (t  << 1) | (tr << 2) |
                            (l  << 3) | (r  << 4) |
                            (bl << 5) | (b  << 6) | (br << 7);

        // Use __popc to count the number of live neighbors.
        int live_neighbors = __popc(neighbor_bits);

        // Get the current cell state from the middle word.
        int current = get_bit(mid, i);

        // Compute the next state according to Conway's Game of Life rules:
        //   - A dead cell with exactly 3 live neighbors becomes live.
        //   - A live cell with 2 or 3 live neighbors survives (otherwise dies).
        int next_state = (live_neighbors == 3) || (current && (live_neighbors == 2));

        // Set the corresponding bit in our result word if next_state is live.
        if (next_state)
            res |= (1ULL << i);
    }

    // Write the computed result for this word to the output grid.
    output[row_base + word_idx] = res;
}

// Host function that runs one simulation step of Conway's Game of Life on the GPU.
// 'input' and 'output' are pointers to device memory, where each std::uint64_t element is bit-packed
// representing 64 cells. 'grid_dimensions' is the width and height of the square grid.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Set a block size. We choose a 16x16 thread block.
    dim3 blockDim(16, 16);

    // The grid dimensions: x dimension over the words per row, y dimension over the grid rows.
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
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
