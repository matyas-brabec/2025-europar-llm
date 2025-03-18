#include <cstdint>
#include <cuda_runtime.h>

// In this implementation, each uint64_t element encodes 64 consecutive cells in a row.
// We adopt the following bit ordering convention for each 64‐bit word:
//   Bit 63 represents the leftmost cell and bit 0 represents the rightmost cell.
// The grid is a square of grid_dimensions x grid_dimensions cells.
// In the bit‐packed layout, each row contains (grid_dimensions/64) uint64_t words.
// Cells outside the grid are considered dead.

// CUDA kernel: each thread processes one 64‐bit word (i.e. 64 cells) of the grid.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dim) {
    // Compute the number of 64‐bit words per row.
    int words_per_row = grid_dim / 64;
    
    // Compute the row (r) and word (c) indices for the current thread.
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row index (in cells)
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Word index within the row
    
    // Process only threads corresponding to valid grid positions.
    if (r < grid_dim && c < words_per_row) {
        // Load the current word (middle row, center cell group).
        std::uint64_t mid_center = input[r * words_per_row + c];
        
        // Compute horizontal neighbors in the same row.
        // For our chosen bit ordering:
        // - The left neighbor for a cell is the cell immediately to its left,
        //   which, for cells within this word (except for the leftmost cell),
        //   can be obtained by shifting mid_center right by 1.
        // - For the leftmost cell (bit 63 of mid_center), if this word is not the first in the row (c > 0),
        //   its left neighbor comes from the rightmost bit (bit 0) of the previous word.
        std::uint64_t mid_left = mid_center >> 1;
        if (c > 0) {
            std::uint64_t left_word = input[r * words_per_row + c - 1];
            mid_left |= ((left_word & 1ULL) << 63);
        }
        
        // Similarly, the right neighbor for a cell is obtained by shifting mid_center left by 1.
        // For the rightmost cell (bit 0), if there is a next word (c < words_per_row-1),
        // its right neighbor comes from the leftmost bit (bit 63) of that word.
        std::uint64_t mid_right = mid_center << 1;
        if (c < words_per_row - 1) {
            std::uint64_t right_word = input[r * words_per_row + c + 1];
            mid_right |= ((right_word >> 63) & 1ULL);
        }
        
        // Compute neighbors from the row above.
        std::uint64_t top_center = 0, top_left = 0, top_right = 0;
        if (r > 0) {
            std::uint64_t tmp = input[(r - 1) * words_per_row + c];
            top_center = tmp;
            top_left = tmp >> 1;
            if (c > 0) {
                std::uint64_t left_word = input[(r - 1) * words_per_row + c - 1];
                top_left |= ((left_word & 1ULL) << 63);
            }
            top_right = tmp << 1;
            if (c < words_per_row - 1) {
                std::uint64_t right_word = input[(r - 1) * words_per_row + c + 1];
                top_right |= ((right_word >> 63) & 1ULL);
            }
        }
        
        // Compute neighbors from the row below.
        std::uint64_t bot_center = 0, bot_left = 0, bot_right = 0;
        if (r < grid_dim - 1) {
            std::uint64_t tmp = input[(r + 1) * words_per_row + c];
            bot_center = tmp;
            bot_left = tmp >> 1;
            if (c > 0) {
                std::uint64_t left_word = input[(r + 1) * words_per_row + c - 1];
                bot_left |= ((left_word & 1ULL) << 63);
            }
            bot_right = tmp << 1;
            if (c < words_per_row - 1) {
                std::uint64_t right_word = input[(r + 1) * words_per_row + c + 1];
                bot_right |= ((right_word >> 63) & 1ULL);
            }
        }
        
        // For each of the 64 cells in the word, compute the number of live neighbors
        // and apply the Game of Life rules:
        //   - A dead cell with exactly 3 live neighbors becomes live.
        //   - A live cell with 2 or 3 live neighbors survives.
        std::uint64_t res = 0;
        #pragma unroll
        for (int bit = 0; bit < 64; bit++) {
            int count = 0;
            count += (top_left   >> bit) & 1;
            count += (top_center >> bit) & 1;
            count += (top_right  >> bit) & 1;
            count += (mid_left  >> bit) & 1;
            count += (mid_right >> bit) & 1;
            count += (bot_left   >> bit) & 1;
            count += (bot_center >> bit) & 1;
            count += (bot_right  >> bit) & 1;
            
            int cell = (mid_center >> bit) & 1;
            int new_cell = (count == 3) || (cell && (count == 2));
            res |= (static_cast<std::uint64_t>(new_cell) << bit);
        }
        
        // Write the updated word to the output grid.
        output[r * words_per_row + c] = res;
    }
}

// Host function that launches one Game of Life iteration on the GPU.
// The input and output grids are assumed to be allocated using cudaMalloc.
// Any required host-device synchronization is handled externally.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    int words_per_row = grid_dimensions / 64;
    
    // Define block and grid dimensions.
    // Using 16x16 threads per block to achieve good occupancy.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel; caller is responsible for synchronization.
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
