// This CUDA implementation performs one generation (step) of Conway’s Game of Life
// on a 2D grid of cells that is bit‐packed into 64‐bit words (each word holds 64 cells).
// We assume that cells are stored in each 64‐bit word with one bit per cell in the following order:
// the least significant bit (bit 0) corresponds to the leftmost cell in the 64‐cell block,
// and bit 63 corresponds to the rightmost cell.
// Special handling is needed for the leftmost (0th) and rightmost (63rd) cells in each word:
// for cell 0, the missing left neighbor is taken from the neighboring (left) word’s bit 63,
// and for cell 63, the missing right neighbor is taken from the neighboring (right) word’s bit 0.
// Similarly, vertical neighbors (from the rows above and below) are gathered from the
// corresponding words in neighboring rows, with analogous horizontal extractions.
// 
// To avoid examining one cell at a time, each thread works on a full 64‐cell word at once.
// The 8 neighbor contributions (from top‐left, top, top‐right, left, right, bottom‐left,
// bottom, and bottom‐right) are “added” in parallel using a bit‐sliced addition scheme.
// In this approach the per–cell neighbor count (ranging from 0 to 8) is stored in 4 bits
// for each cell across four 64–bit accumulators (s3, s2, s1, s0), where for each cell:
//   count = ( (s3_bit << 3) | (s2_bit << 2) | (s1_bit << 1) | (s0_bit) ) 
// Once the neighbor count is accumulated, the next state is computed according to the rule:
//   new cell = 1 if (neighbor count == 3) or (current cell is alive and neighbor count == 2)
// For testing equality we use simple bit–sliced comparisons: a cell’s count equals 2 if its 4–bit
// value equals binary 0010, and equals 3 if it equals 0011.
// 
// Each thread processes one 64–bit word. The grid is logically divided into rows of (grid_dimensions/64)
// words. Boundary conditions (outside the grid) are treated as dead (0).
//
// The following source code contains the CUDA kernel and the host function run_game_of_life,
// which you can compile with the latest CUDA toolkit and host compiler for target devices such as A100/H100.

#include <cuda_runtime.h>
#include <cstdint>

// __device__ function to add a binary digit (0 or 1 per cell)
// to a 4–bit per–cell, bit–sliced accumulator (s3,s2,s1,s0).
// Each of s0, s1, s2, and s3 is a 64–bit variable whose bit i holds the corresponding bit
// of the 4–bit sum for cell i.
__device__ __forceinline__ void add_bit(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3, uint64_t add)
{
    // Add the value in 'add' (which has 0/1 in each lane) to the accumulator.
    uint64_t t = s0 ^ add;      // sum in bit0 (without carry)
    uint64_t c = s0 & add;      // carry out from bit0
    s0 = t;
    
    t = s1 ^ c;
    c = s1 & c;
    s1 = t;
    
    t = s2 ^ c;
    c = s2 & c;
    s2 = t;
    
    t = s3 ^ c;  // maximum neighbor count is 8 so 4 bits are enough.
    s3 = t;
}

// CUDA kernel: each thread computes one 64–bit word of the output.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dim)
{
    // Compute number of 64–bit words per row.
    int words_per_row = grid_dim / 64;
    int total_words = grid_dim * words_per_row;  // grid_dim rows * words_per_row words per row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;
    
    // Map linear thread index to (row, col) in the grid of 64–bit words.
    int row = idx / words_per_row;
    int col = idx % words_per_row;
    
    // Linear index for the current word.
    int cur_index = row * words_per_row + col;
    std::uint64_t cur = input[cur_index];
    
    // ---------------------------------------------------------------------
    // Compute horizontal neighbors for the current row.
    // For each word, to get the left neighbor contribution we shift left by 1:
    //   (cur << 1) produces for cell i (i>=1) the value of cur[i-1],
    //   but cell 0 becomes 0; so OR in the missing bit from the left word, if it exists.
    // Similarly, for the right neighbor we use (cur >> 1)
    // and add the missing bit (from the right word) into bit 63.
    // Remember: In our chosen representation, cell i is stored in bit i,
    // with bit 0 = leftmost cell and bit 63 = rightmost cell.
    // ---------------------------------------------------------------------
    std::uint64_t cur_left = 0, cur_right = 0;
    if (col > 0)
    {
        int left_index = row * words_per_row + (col - 1);
        std::uint64_t left_word = input[left_index];
        // Extract the missing left neighbor from the rightmost cell of the left word:
        std::uint64_t extra_left = (left_word >> 63) & 1ULL;
        cur_left = (cur << 1) | extra_left;
    }
    else
    {
        cur_left = (cur << 1);
    }
    if (col < words_per_row - 1)
    {
        int right_index = row * words_per_row + (col + 1);
        std::uint64_t right_word = input[right_index];
        // Extract the missing right neighbor from the leftmost cell of the right word:
        std::uint64_t extra_right = right_word & 1ULL;
        cur_right = (cur >> 1) | (extra_right << 63);
    }
    else
    {
        cur_right = (cur >> 1);
    }
    
    // ---------------------------------------------------------------------
    // Compute horizontal neighbor contributions for the top row (row-1).
    // For the top row, if it exists, we gather three neighbors:
    //   top_nw (north-west): from top_center shifted left by 1, with extra bit from the left word.
    //   top_n  (north): direct copy of the top-center word.
    //   top_ne (north-east): from top_center shifted right by 1, with extra bit from the right word.
    // If the top row is off the grid (row==0) then all top contributions are 0.
    // ---------------------------------------------------------------------
    std::uint64_t top_nw = 0, top_n = 0, top_ne = 0;
    if (row > 0)
    {
        int top_index = (row - 1) * words_per_row + col;
        std::uint64_t top_center = input[top_index];
        std::uint64_t top_left_word = 0, top_right_word = 0;
        if (col > 0)
        {
            int top_left_index = (row - 1) * words_per_row + (col - 1);
            top_left_word = input[top_left_index];
        }
        if (col < words_per_row - 1)
        {
            int top_right_index = (row - 1) * words_per_row + (col + 1);
            top_right_word = input[top_right_index];
        }
        std::uint64_t extra_top_nw = (col > 0) ? ((top_left_word >> 63) & 1ULL) : 0;
        top_nw = (top_center << 1) | extra_top_nw;
        top_n = top_center;
        std::uint64_t extra_top_ne = (col < words_per_row - 1) ? (top_right_word & 1ULL) : 0;
        top_ne = (top_center >> 1) | (extra_top_ne << 63);
    }
    
    // ---------------------------------------------------------------------
    // Compute horizontal neighbor contributions for the bottom row (row+1).
    // For the bottom row, if it exists, we similarly compute:
    //   bottom_nw, bottom_n, bottom_ne.
    // ---------------------------------------------------------------------
    std::uint64_t bottom_nw = 0, bottom_n = 0, bottom_ne = 0;
    if (row < grid_dim - 1)
    {
        int bottom_index = (row + 1) * words_per_row + col;
        std::uint64_t bottom_center = input[bottom_index];
        std::uint64_t bottom_left_word = 0, bottom_right_word = 0;
        if (col > 0)
        {
            int bottom_left_index = (row + 1) * words_per_row + (col - 1);
            bottom_left_word = input[bottom_left_index];
        }
        if (col < words_per_row - 1)
        {
            int bottom_right_index = (row + 1) * words_per_row + (col + 1);
            bottom_right_word = input[bottom_right_index];
        }
        std::uint64_t extra_bottom_nw = (col > 0) ? ((bottom_left_word >> 63) & 1ULL) : 0;
        bottom_nw = (bottom_center << 1) | extra_bottom_nw;
        bottom_n = bottom_center;
        std::uint64_t extra_bottom_ne = (col < words_per_row - 1) ? (bottom_right_word & 1ULL) : 0;
        bottom_ne = (bottom_center >> 1) | (extra_bottom_ne << 63);
    }
    
    // ---------------------------------------------------------------------
    // Accumulate the neighbor counts in a bit-sliced manner.
    // We want to compute, for each of the 64 cells in the word, an integer count (0..8)
    // stored in 4 bits (in accumulators s3, s2, s1, s0).
    // We add the following eight neighbor contributions:
    //   Top row: top_nw, top_n, top_ne.
    //   Current row: cur_left, cur_right.
    //   Bottom row: bottom_nw, bottom_n, bottom_ne.
    // Each neighbor mask is a 64–bit value whose bit i is 1 if that neighbor cell is alive.
    // ---------------------------------------------------------------------
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    add_bit(s0, s1, s2, s3, top_nw);
    add_bit(s0, s1, s2, s3, top_n);
    add_bit(s0, s1, s2, s3, top_ne);
    add_bit(s0, s1, s2, s3, cur_left);
    add_bit(s0, s1, s2, s3, cur_right);
    add_bit(s0, s1, s2, s3, bottom_nw);
    add_bit(s0, s1, s2, s3, bottom_n);
    add_bit(s0, s1, s2, s3, bottom_ne);
    
    // ---------------------------------------------------------------------
    // Compute bit masks for “count==2” and “count==3” in bit–sliced form.
    // A cell's neighbor count is stored as a 4–bit number:
    //   count == 2  <=> binary 0010 (i.e., s3=0, s2=0, s1=1, s0=0)
    //   count == 3  <=> binary 0011 (i.e., s3=0, s2=0, s1=1, s0=1)
    // For each bit (i.e. for each cell), we produce a 1 if the condition is met.
    // ---------------------------------------------------------------------
    uint64_t eq2 = (~s3) & (~s2) & s1 & (~s0);
    uint64_t eq3 = (~s3) & (~s2) & s1 & s0;
    
    // Apply the Game of Life rule:
    // A cell becomes live if it has exactly 3 live neighbors, or if it is currently live and has exactly 2.
    std::uint64_t new_state = eq3 | (cur & eq2);
    
    // Write the computed 64 cells (packed as a 64–bit word) to the output grid.
    output[cur_index] = new_state;
}

// Host function to run one step of Conway’s Game of Life on the GPU.
// 'input' and 'output' are pointers to bit-packed grids allocated via cudaMalloc.
// 'grid_dimensions' is the width (and height) in cells (a power‐of‐2, >512).
//
// This function launches the kernel to compute one generation. (All necessary
// host-device data transformations have been considered non–critical for performance.)
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64–bit words per row.
    int words_per_row = grid_dimensions / 64;
    int total_words = grid_dimensions * words_per_row;
    
    // Set up a 1D grid of threads – one thread processes one 64–bit word.
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;
    
    // Launch the kernel.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
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
