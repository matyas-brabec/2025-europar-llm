#include <cstdint>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// This CUDA implementation performs one iteration (one “step”) of Conway’s Game of Life.
// The grid is bit‐packed: each std::uint64_t word holds 64 consecutive cells (1 bit per cell).
// The grid is square with dimensions grid_dimensions x grid_dimensions (grid_dimensions is a power‐of‐2).
// Each CUDA thread processes one 64‐bit word (i.e. 64 cells) in the grid.
// 
// The update rule is: a cell becomes live if it has exactly 3 live neighbors,
// or if it is currently live and has exactly 2 live neighbors.
// Neighbors are the eight adjacent cells (horizontal, vertical, diagonal).
// Cells outside the grid are considered dead.
// 
// To compute the per‐cell neighbor count (range 0–8) in parallel over 64 cells,
// we “bit‐slice” the addition. For each neighbor row we first compute the horizontal
// contributions from three adjacent cells (left, center, right) using full‐adder logic.
// For cells in rows above and below the current row, all three contributions are used,
// while for the current row only the immediate left and right neighbors count.
// Then we add the three 2‐bit numbers (from top, mid, bottom) using bit‐sliced adders,
// and finally decide the fate of each cell by checking if the total equals 3 or equals 2.
// Special care is taken for the boundary bits within each 64‐bit word:
//  • For the left neighbor of cell 0, we use the most–significant bit from the adjacent left word.
//  • For the right neighbor of cell 63, we use the least–significant bit from the adjacent right word.
// 
// The helper functions below operate on 64 “lanes” in parallel,
// where each lane corresponds to one cell within a 64–bit word.
// A 2–bit number is represented as two 64–bit masks: (lo, hi)
// so that for each bit position j the value is: ( (hi >> j) & 1 )*2 + ((lo >> j) & 1 ).
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Helper: For a given word (representing 64 cells), compute the left neighbor mask.
// The neighbor for cell j is the cell j-1. In the same word, shifting left by 1 shifts
// each bit into the position corresponding to the neighbor to its right.
// For cell 0, if a left-adjacent word exists, we take its bit63 (most-significant bit)
// and place it into bit0.
__device__ inline uint64_t get_left(uint64_t word, uint64_t left_word, bool has_left) {
    // Shift current word left: for j>=1, cell j gets bit (j-1); cell0 becomes 0.
    uint64_t res = word << 1;
    if (has_left) {
        // Extract the rightmost bit from the left-adjacent word.
        // Since we treat bit0 as the least-significant bit (cell0), the "rightmost" cell
        // in the left word is bit63.
        res |= ((left_word >> 63) & 1ULL);
    }
    return res;
}

// Helper: For a given word, compute the right neighbor mask.
// The neighbor for cell j is the cell j+1. In the same word, shifting right by 1 moves bit j+1 into j.
// For cell 63, if a right-adjacent word exists, take its bit0 and shift it into bit63.
__device__ inline uint64_t get_right(uint64_t word, uint64_t right_word, bool has_right) {
    uint64_t res = word >> 1;
    if (has_right) {
        res |= ((right_word & 1ULL) << 63);
    }
    return res;
}

// -----------------------------------------------------------------------------
// Horizontal addition for a row where three contributions (from left-diagonal,
// vertical, and right-diagonal) are summed up for each cell.
// The three inputs are assumed already shifted with appropriate boundary handling.
// That is, the caller computes:
//   comp_left = get_left(center, left_adj, has_left)
//   comp_center = center
//   comp_right = get_right(center, right_adj, has_right)
// Then the sum (a 2-bit number with value 0..3) per cell is computed using full adder logic.
__device__ inline void horz3(uint64_t comp_left, uint64_t comp_center, uint64_t comp_right,
                              uint64_t &out_lo, uint64_t &out_hi)
{
    // Full adder for three 1-bit numbers per cell.
    // Sum-bit = XOR of the three bits.
    // Carry-bit = majority function: (a&b) | (a&c) | (b&c).
    out_lo = comp_left ^ comp_center ^ comp_right;
    out_hi = (comp_left & comp_center) | (comp_left & comp_right) | (comp_center & comp_right);
}

// Horizontal addition for the current row where only two neighbors (left and right) are present.
__device__ inline void horz2(uint64_t comp_left, uint64_t comp_right,
                              uint64_t &out_lo, uint64_t &out_hi)
{
    // For two bits, the sum is:
    // Sum-bit = XOR, and carry (representing a value "2") = AND.
    out_lo = comp_left ^ comp_right;
    out_hi = comp_left & comp_right;
}

// -----------------------------------------------------------------------------
// Add two 2-bit numbers (each represented as (lo, hi)) and return a 3-bit result
// represented as (res_lo, res_hi, res_hi2), where the value per cell is:
//   result = (res_hi2 * 4) + (res_hi * 2) + (res_lo).
// Both inputs are in the range 0..3, so result is in 0..6.
__device__ inline void add2bit(uint64_t a_lo, uint64_t a_hi,
                               uint64_t b_lo, uint64_t b_hi,
                               uint64_t &res_lo, uint64_t &res_hi, uint64_t &res_hi2)
{
    // Add the least-significant bits.
    uint64_t r0 = a_lo ^ b_lo;
    uint64_t carry0 = a_lo & b_lo;
    // Add the next bits along with the carry from the LSB.
    uint64_t r1 = a_hi ^ b_hi ^ carry0;
    uint64_t carry1 = (a_hi & b_hi) | (a_hi & carry0) | (b_hi & carry0);
    uint64_t r2 = carry1;
    res_lo = r0;
    res_hi = r1;
    res_hi2 = r2;
}

// Add a 3-bit number (represented as (x_lo, x_hi, x_hi2)) and a 2-bit number (y_lo, y_hi)
// to produce a 4-bit result (res_lo, res_hi, res_hi2, res_hi3). The 3-bit number is in 0..7
// and the 2-bit number in 0..3; the result is in 0..10 (but for our Game of Life the sum never exceeds 8).
__device__ inline void add3bit(uint64_t x_lo, uint64_t x_hi, uint64_t x_hi2,
                               uint64_t y_lo, uint64_t y_hi,
                               uint64_t &res_lo, uint64_t &res_hi,
                               uint64_t &res_hi2, uint64_t &res_hi3)
{
    // First add the LSBs.
    uint64_t s0 = x_lo ^ y_lo;
    uint64_t c0 = x_lo & y_lo;
    // Add next bits: x_hi, y_hi, and the carry from the LSB.
    uint64_t s1 = x_hi ^ y_hi ^ c0;
    uint64_t c1 = (x_hi & y_hi) | (x_hi & c0) | (y_hi & c0);
    // Add the third bit from x and the carry from previous stage.
    uint64_t s2 = x_hi2 ^ c1;
    uint64_t c2 = x_hi2 & c1;
    // The fourth bit is just the carry.
    uint64_t s3 = c2;
    res_lo = s0;
    res_hi = s1;
    res_hi2 = s2;
    res_hi3 = s3;
}

// -----------------------------------------------------------------------------
// The kernel: each thread processes one 64-bit word (64 cells).
// It reads the required neighbors from the input grid (bit-packed).
// The grid is logically arranged into rows, each of which consists of (grid_dimensions/64) words.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    int total_words = grid_dimensions * words_per_row;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;
    
    // Determine the row and column (word index) in the grid.
    int row = idx / words_per_row;
    int col = idx % words_per_row;
    
    // Convenience: compute base indices.
    int cur_index = row * words_per_row + col;
    
    // Read the current cell word.
    std::uint64_t cur = input[cur_index];
    
    // -------------------------------------------------------------------------
    // For neighbors, we define indices for adjacent rows and words.
    // If a neighbor is out-of-bound, we treat it as 0.
    
    // Row above (top)
    std::uint64_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_base = (row - 1) * words_per_row;
        top = input[top_base + col];
        if (col > 0)
            top_left = input[top_base + col - 1];
        if (col < words_per_row - 1)
            top_right = input[top_base + col + 1];
    }
    
    // Row below (bottom)
    std::uint64_t bottom = 0, bottom_left = 0, bottom_right = 0;
    if (row < grid_dimensions - 1) {
        int bottom_base = (row + 1) * words_per_row;
        bottom = input[bottom_base + col];
        if (col > 0)
            bottom_left = input[bottom_base + col - 1];
        if (col < words_per_row - 1)
            bottom_right = input[bottom_base + col + 1];
    }
    
    // Same row neighbors will be used for mid contribution.
    std::uint64_t mid_left = 0, mid_right = 0;
    {
        int cur_row_base = row * words_per_row;
        if (col > 0)
            mid_left = input[cur_row_base + col - 1];
        if (col < words_per_row - 1)
            mid_right = input[cur_row_base + col + 1];
    }
    
    // -------------------------------------------------------------------------
    // Compute horizontal neighbor sums for each of the three rows:
    // For the top and bottom rows, three neighbors contribute: left-diagonal, vertical, and right-diagonal.
    // For the current row, only left and right contribute.
    
    // --- Top row contribution (2-bit number per cell: value 0..3) ---
    std::uint64_t top_sum_lo = 0, top_sum_hi = 0;
    if (row > 0) {
        // For top row, prepare the three components from the top-center word.
        uint64_t comp_left_top = get_left(top, top_left, (col > 0));
        uint64_t comp_center_top = top;  // no shift for the vertical neighbor
        uint64_t comp_right_top = get_right(top, top_right, (col < words_per_row - 1));
        horz3(comp_left_top, comp_center_top, comp_right_top, top_sum_lo, top_sum_hi);
    }
    else {
        // No row above: contribution is 0.
        top_sum_lo = 0; top_sum_hi = 0;
    }
    
    // --- Bottom row contribution (2-bit number per cell: value 0..3) ---
    std::uint64_t bottom_sum_lo = 0, bottom_sum_hi = 0;
    if (row < grid_dimensions - 1) {
        uint64_t comp_left_bot = get_left(bottom, bottom_left, (col > 0));
        uint64_t comp_center_bot = bottom;
        uint64_t comp_right_bot = get_right(bottom, bottom_right, (col < words_per_row - 1));
        horz3(comp_left_bot, comp_center_bot, comp_right_bot, bottom_sum_lo, bottom_sum_hi);
    }
    else {
        bottom_sum_lo = 0; bottom_sum_hi = 0;
    }
    
    // --- Mid (current row) contribution (2-bit number per cell: value 0..2) ---
    std::uint64_t mid_sum_lo = 0, mid_sum_hi = 0;
    {
        uint64_t comp_left_mid = get_left(cur, mid_left, (col > 0));
        uint64_t comp_right_mid = get_right(cur, mid_right, (col < words_per_row - 1));
        // For the current row, the two neighbors are simply added.
        horz2(comp_left_mid, comp_right_mid, mid_sum_lo, mid_sum_hi);
    }
    
    // -------------------------------------------------------------------------
    // Now add the three contributions: top (2-bit), mid (2-bit), and bottom (2-bit).
    // First, add top and mid. The result is a 3-bit number per cell (range 0..5).
    std::uint64_t interm_lo, interm_hi, interm_hi2;
    add2bit(top_sum_lo, top_sum_hi, mid_sum_lo, mid_sum_hi, interm_lo, interm_hi, interm_hi2);
    
    // Then add the bottom contribution (2-bit) to the intermediate 3-bit sum,
    // to get a 4-bit total (range 0..8).
    std::uint64_t total_lo, total_hi, total_hi2, total_hi3;
    add3bit(interm_lo, interm_hi, interm_hi2, bottom_sum_lo, bottom_sum_hi,
            total_lo, total_hi, total_hi2, total_hi3);
    
    // -------------------------------------------------------------------------
    // Determine the new state for each cell.
    // According to the Game of Life rules, a cell becomes alive if it has exactly 3 live neighbors,
    // or if it is already alive and has exactly 2 live neighbors.
    // In our 4-bit total, for each cell:
    //   total == 3: bits (total_hi3,total_hi2,total_hi,total_lo) == 0,0,1,1.
    //   total == 2: bits == 0,0,1,0.
    // We compute masks for (total==3) and (total==2) for all 64 bits concurrently.
    //
    // Note: total_hi3 is the MSB (bit3), total_hi2 is bit2, total_hi is bit1, and total_lo is bit0.
    std::uint64_t eq3 = (~total_hi3) & (~total_hi2) & total_hi & total_lo;  // equals 3 if 0,0,1,1.
    std::uint64_t eq2 = (~total_hi3) & (~total_hi2) & total_hi & (~total_lo); // equals 2 if 0,0,1,0.
    
    // A cell becomes live if eq3 is true OR (if currently live and eq2 is true).
    std::uint64_t new_state = eq3 | (cur & eq2);
    
    // Write the result.
    output[cur_index] = new_state;
}

// -----------------------------------------------------------------------------
// Host function that launches the kernel to run one step of Conway's Game of Life.
// The input and output grids are bit-packed arrays of std::uint64_t, allocated on the GPU.
// The grid is square with dimensions grid_dimensions x grid_dimensions.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Determine the number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    int total_words = grid_dimensions * words_per_row;
    
    // Choose block and grid sizes. Using 256 threads per block.
    int blockSize = 256;
    int gridSize = (total_words + blockSize - 1) / blockSize;
    
    // Launch the kernel.
    game_of_life_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
    
    // Note: Synchronization (if needed) is handled by the caller.
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
