#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one generation of Conway's Game of Life.
// The grid is stored in a bit‐packed format: each uint64_t encodes 64 consecutive cells
// in a row (bit value 1 = alive, 0 = dead). Each thread processes one uint64_t word.
// For each cell (each bit within the word), the 8 neighbors are examined across three rows.
// Special care is applied for the 0th and 63rd bits to fetch neighbors that reside in
// adjacent words.
// The next state for each cell is determined by the rule:
//    next = (neighbors == 3) || (current && neighbors == 2)
// Boundary cells (outside the grid) are assumed to be dead.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    // Number of 64-bit words per row.
    int numWords = grid_dimensions >> 6;  // grid_dimensions / 64

    // Calculate thread coordinates:
    // Each thread corresponds to one word in a given row.
    int wordIndex = blockIdx.x * blockDim.x + threadIdx.x; // horizontal index (word index)
    int row = blockIdx.y * blockDim.y + threadIdx.y;         // row index (cell row)
    
    // If outside the grid, exit.
    if (row >= grid_dimensions || wordIndex >= numWords)
        return;
    
    // Compute the index into the input/output arrays.
    int index = row * numWords + wordIndex;
    
    // For neighboring rows, load the corresponding words if available; otherwise use 0.
    // Top row (row-1)
    std::uint64_t top_mid   = (row > 0) ? input[(row - 1) * numWords + wordIndex] : 0ULL;
    std::uint64_t top_left  = (row > 0 && wordIndex > 0) ? input[(row - 1) * numWords + (wordIndex - 1)] : 0ULL;
    std::uint64_t top_right = (row > 0 && wordIndex < numWords - 1) ? input[(row - 1) * numWords + (wordIndex + 1)] : 0ULL;
    
    // Current row (row)
    std::uint64_t mid       = input[index];
    std::uint64_t mid_left  = (wordIndex > 0) ? input[row * numWords + (wordIndex - 1)] : 0ULL;
    std::uint64_t mid_right = (wordIndex < numWords - 1) ? input[row * numWords + (wordIndex + 1)] : 0ULL;
    
    // Bottom row (row+1)
    std::uint64_t bot_mid   = (row < grid_dimensions - 1) ? input[(row + 1) * numWords + wordIndex] : 0ULL;
    std::uint64_t bot_left  = (row < grid_dimensions - 1 && wordIndex > 0) ? input[(row + 1) * numWords + (wordIndex - 1)] : 0ULL;
    std::uint64_t bot_right = (row < grid_dimensions - 1 && wordIndex < numWords - 1) ? input[(row + 1) * numWords + (wordIndex + 1)] : 0ULL;
    
    // The variable 'result' accumulates the updated 64 bits for this word.
    std::uint64_t result = 0ULL;
    
    // Process bit 0 separately (special handling for left-boundary in the word).
    {
        int n = 0;
        // Top row:
        n += (row > 0 && wordIndex > 0) ? (int)((top_left >> 63) & 1ULL) : 0;
        n += (row > 0) ? (int)((top_mid >> 0) & 1ULL) : 0;
        n += (row > 0) ? (int)((top_mid >> 1) & 1ULL) : 0;
        // Same row:
        n += (wordIndex > 0) ? (int)((mid_left >> 63) & 1ULL) : 0;
        n += (int)((mid >> 1) & 1ULL);
        // Bottom row:
        n += (row < grid_dimensions - 1 && wordIndex > 0) ? (int)((bot_left >> 63) & 1ULL) : 0;
        n += (row < grid_dimensions - 1) ? (int)((bot_mid >> 0) & 1ULL) : 0;
        n += (row < grid_dimensions - 1) ? (int)((bot_mid >> 1) & 1ULL) : 0;
        
        // Get current cell state (bit 0 of 'mid').
        int current = (int)((mid >> 0) & 1ULL);
        // Apply Game of Life rule: cell becomes live if exactly 3 neighbors,
        // or remains live if 2 neighbors and already live.
        int new_state = (n == 3) || (current && n == 2);
        result |= ((std::uint64_t)new_state << 0);
    }
    
    // Process bits 1 through 62 (the typical case with neighbors within the same word).
    #pragma unroll
    for (int bit = 1; bit < 63; bit++) {
        int n = 0;
        // Top row contributions (if available):
        if (row > 0) {
            n += (int)((top_mid >> (bit - 1)) & 1ULL);
            n += (int)((top_mid >> bit) & 1ULL);
            n += (int)((top_mid >> (bit + 1)) & 1ULL);
        }
        // Same row (neighbors are in the same word 'mid'):
        n += (int)((mid >> (bit - 1)) & 1ULL);
        n += (int)((mid >> (bit + 1)) & 1ULL);
        // Bottom row contributions (if available):
        if (row < grid_dimensions - 1) {
            n += (int)((bot_mid >> (bit - 1)) & 1ULL);
            n += (int)((bot_mid >> bit) & 1ULL);
            n += (int)((bot_mid >> (bit + 1)) & 1ULL);
        }
        int current = (int)((mid >> bit) & 1ULL);
        int new_state = (n == 3) || (current && n == 2);
        result |= ((std::uint64_t)new_state << bit);
    }
    
    // Process bit 63 separately (special handling for right-boundary in the word).
    {
        int n = 0;
        // Top row:
        if (row > 0) {
            n += (int)((top_mid >> 62) & 1ULL);  // top-left (from same word)
            n += (int)((top_mid >> 63) & 1ULL);  // top-center
            n += (wordIndex < numWords - 1) ? (int)((top_right >> 0) & 1ULL) : 0;  // top-right from adjacent word
        }
        // Same row:
        n += (int)((mid >> 62) & 1ULL); // mid-left
        n += (wordIndex < numWords - 1) ? (int)((mid_right >> 0) & 1ULL) : 0;    // mid-right
        // Bottom row:
        if (row < grid_dimensions - 1) {
            n += (int)((bot_mid >> 62) & 1ULL);
            n += (int)((bot_mid >> 63) & 1ULL);
            n += (wordIndex < numWords - 1) ? (int)((bot_right >> 0) & 1ULL) : 0;
        }
        int current = (int)((mid >> 63) & 1ULL);
        int new_state = (n == 3) || (current && n == 2);
        result |= ((std::uint64_t)new_state << 63);
    }
    
    // Write the computed result (next state for these 64 cells) to the output grid.
    output[index] = result;
}

// Host function that launches one simulation step of Conway's Game of Life.
// The grid is a square of grid_dimensions x grid_dimensions cells (grid_dimensions is a power of 2).
// The cells are bit-packed: each uint64_t word holds 64 cells in a row.
// This function sets up the CUDA kernel grid and block dimensions and launches the kernel.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int numWords = grid_dimensions >> 6;  // grid_dimensions / 64

    // Define a 2D block configuration. The values are chosen to offer a good occupancy on modern GPUs.
    dim3 block(16, 16);
    // Define a 2D grid configuration so that each thread covers one 64-bit word.
    dim3 gridDim((numWords + block.x - 1) / block.x, (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. All host-device synchronizations (if necessary) are assumed to be handled externally.
    game_of_life_kernel<<<gridDim, block>>>(input, output, grid_dimensions);
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
