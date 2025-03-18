#include <cuda_runtime.h>
#include <cstdint>

// __global__ kernel for one Game of Life step.
// Each thread processes one 64‐bit word (i.e. 64 contiguous cells) in the grid.
// The grid is organized as rows of bit‐packed cells. Each row has grid_dimensions/64 words.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    // Since grid_dimensions is a power of two, division is exact.
    int words_per_row = grid_dimensions / 64;

    // Compute the cell row and word-column index that this thread is responsible for.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // If out-of-bound then return.
    if (row >= grid_dimensions || word_col >= words_per_row)
        return;
    
    // Compute the 1D index for the word in the grid.
    int idx = row * words_per_row + word_col;
    
    // Load neighboring words for the 3 rows surrounding the current row.
    // For any neighbor outside the grid boundaries, assume a word of zero (i.e. all dead cells).
    //
    // For each of the three rows (top, current (mid), bottom) we need to load three words:
    // left neighbor word, current word, and right neighbor word.
    // These neighbor words are used to fetch the diagonal horizontal neighbors.
    //
    // Top row (row-1):
    std::uint64_t top_left   = 0, top_center = 0, top_right = 0;
    if (row > 0) {
        int top_row = row - 1;
        if (word_col > 0)
            top_left = input[top_row * words_per_row + (word_col - 1)];
        top_center = input[top_row * words_per_row + word_col];
        if (word_col < words_per_row - 1)
            top_right = input[top_row * words_per_row + (word_col + 1)];
    }
    
    // Current row (row):
    std::uint64_t mid_left  = 0, mid_center = 0, mid_right = 0;
    if (word_col > 0)
        mid_left = input[row * words_per_row + (word_col - 1)];
    mid_center = input[row * words_per_row + word_col];
    if (word_col < words_per_row - 1)
        mid_right = input[row * words_per_row + (word_col + 1)];
    
    // Bottom row (row+1):
    std::uint64_t bot_left   = 0, bot_center = 0, bot_right = 0;
    if (row < grid_dimensions - 1) {
        int bot_row = row + 1;
        if (word_col > 0)
            bot_left = input[bot_row * words_per_row + (word_col - 1)];
        bot_center = input[bot_row * words_per_row + word_col];
        if (word_col < words_per_row - 1)
            bot_right = input[bot_row * words_per_row + (word_col + 1)];
    }
    
    // Prepare the output word.
    std::uint64_t result = 0;
    
    // Loop over each bit in the 64-bit word.
    // The current thread's word (mid_center) represents 64 consecutive cells.
    // For each cell (bit position) we extract the eight neighbor bits and compute the next state.
    for (int bit = 0; bit < 64; bit++) {
        // To efficiently count the number of live neighbors (which are stored as bits),
        // we build an 8-bit integer where each bit corresponds to one neighbor:
        // Bit positions:
        //   0: top-left, 1: top-center, 2: top-right,
        //   3: middle-left, 4: middle-right,
        //   5: bottom-left, 6: bottom-center, 7: bottom-right.
        uint32_t neighbor_bits = 0;
        
        // --- Top row neighbors ---
        // For the top row the neighbor positions relative to the current cell are:
        // For top-left: if we are not at the left edge of the current word then use top_center shifted right by one;
        // otherwise use bit 63 from top_left.
        uint64_t tl = (bit > 0) ? (top_center >> (bit - 1)) : (top_left >> 63);
        neighbor_bits |= (uint32_t)(tl & 1ULL) << 0;
        
        // Top-center: directly from top_center, same bit position.
        neighbor_bits |= (uint32_t)((top_center >> bit) & 1ULL) << 1;
        
        // Top-right: if not at the right edge then use top_center at (bit+1); else use bit 0 from top_right.
        uint64_t tr = (bit < 63) ? (top_center >> (bit + 1)) : (top_right >> 0);
        neighbor_bits |= (uint32_t)(tr & 1ULL) << 2;
        
        // --- Current row neighbors (only horizontal: left and right) ---
        // Middle-left: if not at left edge then use mid_center at (bit-1); else use bit 63 from mid_left.
        uint64_t ml = (bit > 0) ? (mid_center >> (bit - 1)) : (mid_left >> 63);
        neighbor_bits |= (uint32_t)(ml & 1ULL) << 3;
        
        // Middle-right: if not at right edge then use mid_center at (bit+1); else use bit 0 from mid_right.
        uint64_t mr = (bit < 63) ? (mid_center >> (bit + 1)) : (mid_right >> 0);
        neighbor_bits |= (uint32_t)(mr & 1ULL) << 4;
        
        // --- Bottom row neighbors ---
        // Bottom-left: if not at left edge then use bot_center at (bit-1); else use bit 63 from bot_left.
        uint64_t bl = (bit > 0) ? (bot_center >> (bit - 1)) : (bot_left >> 63);
        neighbor_bits |= (uint32_t)(bl & 1ULL) << 5;
        
        // Bottom-center: directly from bot_center at same bit position.
        neighbor_bits |= (uint32_t)((bot_center >> bit) & 1ULL) << 6;
        
        // Bottom-right: if not at right edge then use bot_center at (bit+1); else use bit 0 from bot_right.
        uint64_t br = (bit < 63) ? (bot_center >> (bit + 1)) : (bot_right >> 0);
        neighbor_bits |= (uint32_t)(br & 1ULL) << 7;
        
        // Use the __popc intrinsic to count the number of set bits in the 8-bit neighbor mask.
        // This gives the total number of live neighbors.
        int neighbor_count = __popc(neighbor_bits);
        
        // Read the current cell's state (live=1, dead=0) from mid_center.
        int current_state = (int)((mid_center >> bit) & 1ULL);
        int new_state = 0;
        
        // Apply Conway's Game of Life rules:
        // For a live cell: survive if it has 2 or 3 live neighbors;
        // For a dead cell: become live if it has exactly 3 live neighbors.
        if (current_state)
            new_state = (neighbor_count == 2 || neighbor_count == 3) ? 1 : 0;
        else
            new_state = (neighbor_count == 3) ? 1 : 0;
        
        // Set the corresponding bit in the result.
        result |= ((std::uint64_t)new_state << bit);
    }
    
    // Store the computed next generation word.
    output[idx] = result;
}

// Host function to run one step of Conway's Game of Life using the CUDA kernel.
// 'input' and 'output' point to bit-packed grids allocated with cudaMalloc.
// 'grid_dimensions' specifies the width and height (in cells) of the square grid.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;
    
    // Set up a 2D grid of threads.
    // Each thread processes one 64-bit word.
    // Choose a block size that balances occupancy and ease-of-use.
    dim3 block(16, 16);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);
    
    // Launch the kernel.
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
