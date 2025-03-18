#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel implements one step of Conway's Game of Life
// on a bit‐packed grid. Each uint64_t word represents 64 consecutive cells
// in a row. The least-significant bit corresponds to the leftmost cell in
// the 64‐cell group, and the most-significant bit corresponds to the rightmost cell.
// Each thread processes one 64‐cell word (block). It preloads the nine neighboring
// words (from the row above, current row, and row below, and from the left, center, and right blocks)
// and then iterates over the 64 bits in the current block to compute the new state
// for each cell based on its 8 neighbors. Out‐of‐bounds neighbors are treated as dead.
 
__global__ void gameOfLifeKernel(const std::uint64_t* __restrict__ input,
                                 std::uint64_t* __restrict__ output,
                                 int grid_dim)
{
    // Each row consists of (grid_dim / 64) words.
    const int words_per_row = grid_dim >> 6; // grid_dim/64

    // Compute global word (block) index.
    const int wordIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = grid_dim * words_per_row;
    if (wordIndex >= total_words)
        return;

    // Determine the current cell block's row (r) and block column (bx).
    const int r = wordIndex / words_per_row;
    const int bx = wordIndex % words_per_row;

    // Load the nine neighboring 64-bit words into a 3x3 array.
    // The indices in nbr[][] correspond to the following offsets:
    // nbr[0][0] -> (row-1, bx-1), nbr[0][1] -> (row-1, bx), nbr[0][2] -> (row-1, bx+1)
    // nbr[1][0] -> (row,   bx-1), nbr[1][1] -> (row,   bx), nbr[1][2] -> (row,   bx+1)
    // nbr[2][0] -> (row+1, bx-1), nbr[2][1] -> (row+1, bx), nbr[2][2] -> (row+1, bx+1)
    std::uint64_t nbr[3][3];
    for (int dr = -1; dr <= 1; dr++) {
        const int r_neighbor = r + dr;
        for (int dc = -1; dc <= 1; dc++) {
            const int b_neighbor = bx + dc;
            std::uint64_t val = 0;
            // Check boundaries: rows [0, grid_dim-1] and blocks [0, words_per_row-1]
            if (r_neighbor >= 0 && r_neighbor < grid_dim &&
                b_neighbor >= 0 && b_neighbor < words_per_row) {
                int idx = r_neighbor * words_per_row + b_neighbor;
                val = input[idx];
            }
            nbr[dr+1][dc+1] = val;
        }
    }

    // The center word for the current block.
    const std::uint64_t center = nbr[1][1];

    // Prepare result word by processing each of the 64 bits.
    // For each cell (bit), count the number of alive neighbors.
    // Note: For horizontal neighbors, we must properly handle the bit‐shift across the word boundary:
    // if bit==0 then the left neighbor comes from the adjacent block's bit63;
    // if bit==63 then the right neighbor comes from the adjacent block's bit0.
    std::uint64_t res = 0;
    for (int bit = 0; bit < 64; bit++) {
        int neighbors = 0;

        // Top row neighbors
        {
            // Top-left neighbor from nbr[0][0]
            const int tl_bit = (bit == 0) ? 63 : bit - 1;
            neighbors += int((nbr[0][0] >> tl_bit) & 1ULL);
            // Top neighbor from nbr[0][1]
            neighbors += int((nbr[0][1] >> bit) & 1ULL);
            // Top-right neighbor from nbr[0][2]
            const int tr_bit = (bit == 63) ? 0 : bit + 1;
            neighbors += int((nbr[0][2] >> tr_bit) & 1ULL);
        }
        
        // Middle row neighbors (excluding the center cell itself)
        {
            // Left neighbor from nbr[1][0]
            const int l_bit = (bit == 0) ? 63 : bit - 1;
            neighbors += int((nbr[1][0] >> l_bit) & 1ULL);
            // Right neighbor from nbr[1][2]
            const int r_bit = (bit == 63) ? 0 : bit + 1;
            neighbors += int((nbr[1][2] >> r_bit) & 1ULL);
        }
        
        // Bottom row neighbors
        {
            // Bottom-left neighbor from nbr[2][0]
            const int bl_bit = (bit == 0) ? 63 : bit - 1;
            neighbors += int((nbr[2][0] >> bl_bit) & 1ULL);
            // Bottom neighbor from nbr[2][1]
            neighbors += int((nbr[2][1] >> bit) & 1ULL);
            // Bottom-right neighbor from nbr[2][2]
            const int br_bit = (bit == 63) ? 0 : bit + 1;
            neighbors += int((nbr[2][2] >> br_bit) & 1ULL);
        }
        
        // Get current cell state from the center word.
        const int current = int((center >> bit) & 1ULL);
        // Apply Conway's rules:
        // A cell is alive if it has exactly 3 alive neighbors,
        // or if it is currently alive and has exactly 2 alive neighbors.
        const int alive = ((neighbors == 3) || ((neighbors == 2) && current));
        if (alive)
            res |= (1ULL << bit);
    }

    // Write the computed result to the output grid.
    output[wordIndex] = res;
}

// Host function that runs one step of Conway's Game of Life.
// It launches the CUDA kernel to compute the next generation.
// - input: pointer to device memory containing the bit-packed input grid.
// - output: pointer to device memory for the bit-packed output grid.
// - grid_dimensions: the width (and height) of the square grid in cells.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions/64 64-bit words.
    const int words_per_row = grid_dimensions >> 6; 
    const int total_words = grid_dimensions * words_per_row;

    // Choose a block size that is efficient for modern GPUs.
    const int blockSize = 256;
    const int numBlocks = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel. Any necessary host-device synchronization is handled by the caller.
    gameOfLifeKernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
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
