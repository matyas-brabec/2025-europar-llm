#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>

//---------------------------------------------------------------------------
// This implementation computes one Game‐of‐Life step using bit‐packed grids.
// Each 64‐bit word encodes 64 cells (LSB = cell0, MSB = cell63). Each CUDA thread
// processes one word (64 cells) in a grid row. To update all 64 cells concurrently,
// we process them in 8 “blocks” of 8 cells each using a precomputed lookup table.
// 
// For each cell the 3x3 neighbourhood is needed (neighbors above, same row, below),
// but the center cell is not counted towards the neighbor sum. Its next state is:
//    live if (popcount(neighbors)==3) or (cell is live && popcount(neighbors)==2)
// Boundary words (0th and last in each row) are handled specially by “importing” one bit
// from the adjacent word (from the left for bit0 and from the right for bit63).
//
// In our approach each thread loads 9 words for the block:
//    For the current row: { left, center, right }
//    For the row above:  { top_left, top_center, top_right }  (if not at top boundary, else 0)
//    For the row below:  { bottom_left, bottom_center, bottom_right }  (if not at bottom boundary, else 0)
// Then the 64 cells in the center word are split into 8 groups of 8 contiguous cells.
// For each 8-cell block we extract (for each row) a 10-bit window that spans the 8 cells
// plus one extra bit on each side. In that 10‐bit window the cell of interest is always
// at positions 1..8; then for each cell (index j in [0,7]) the three bits from that row
// are extracted from positions j, j+1, j+2. Gathering the 3‐bit slices from the three rows
// (top, current, bottom) produces a 9–bit index. We use a lookup table (of 512 entries, one
// per possible 3x3 block) that returns the new state (0 or 1) for the cell.
// 
// The lookup table is built on the host and copied to device constant memory before
// launching the kernel.
//---------------------------------------------------------------------------

// Device constant lookup table: 512 entries for each possible 3x3 block
// The 9 bits are arranged as follows (bit positions in the index):
//    bit0: top NW, bit1: top N, bit2: top NE,
//    bit3: middle W, bit4: center cell, bit5: middle E,
//    bit6: bottom NW, bit7: bottom N, bit8: bottom NE.
// The neighbor sum is computed from all bits except the center (bit4).
// Next state rule: new = 1 if (neighbor count == 3) or (center==1 && neighbor count == 2), else 0.
__constant__ unsigned char d_lut[512];

//---------------------------------------------------------------------------
// __device__ inline function to extract a 10–bit window from a row.
// For a given row we need to extract a contiguous block of 8 cells together
// with one extra cell on the left and one extra on the right.
// The 10-bit window is arranged as follows:
//    bit0: extra left neighbor, bits1–8: the 8 cells, bit9: extra right neighbor.
// "center" is the 64–bit word containing the block.
// "left" and "right" are the neighboring words, used only when the block touches
// the interior boundary of the word.
// "offset" is the starting bit index in the center word (must be 0,8,16,...,56).
// For offset==0, the left extra is taken from the adjacent left word (if available),
// and for offset+8==64 the right extra is taken from the adjacent right word.
__device__ inline uint32_t extract_window(uint64_t left, uint64_t center, uint64_t right, int offset)
{
    uint32_t window = 0;
    uint32_t left_extra;
    uint32_t mid;
    uint32_t right_extra;
    
    if (offset == 0) {
        // For block starting at bit 0, left extra comes from the adjacent left word.
        // If left word is 0 (i.e. outside grid), we treat as dead.
        left_extra = left ? (uint32_t)((left >> 63) & 1ULL) : 0;
        mid = (uint32_t)((center >> 0) & 0xFFULL);
        // Right extra comes from bit8 of the center word.
        right_extra = (uint32_t)((center >> 8) & 1ULL);
    }
    else if (offset + 8 < 64) {
        // Block is entirely within the center word.
        left_extra = (uint32_t)((center >> (offset - 1)) & 1ULL);
        mid = (uint32_t)((center >> offset) & 0xFFULL);
        right_extra = (uint32_t)((center >> (offset + 8)) & 1ULL);
    }
    else { // offset + 8 == 64, which happens when offset == 56.
        left_extra = (uint32_t)((center >> (offset - 1)) & 1ULL); // (center >> 55) & 1
        mid = (uint32_t)((center >> offset) & 0xFFULL); // bits 56..63
        // Right extra comes from the adjacent right word.
        right_extra = right ? (uint32_t)(right & 1ULL) : 0;
    }
    
    // Pack the three parts into a 10–bit number:
    // bits: [9] = right extra, [8:1] = mid 8 bits, [0] = left extra.
    window = left_extra | (mid << 1) | (right_extra << 9);
    return window;
}

//---------------------------------------------------------------------------
// The CUDA kernel that computes one step of Conway's Game of Life.
// Each thread processes one 64-bit word (i.e. 64 cells) from the grid.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The grid is square with grid_dimensions cells per side.
    // Each row has (grid_dimensions / 64) 64-bit words.
    int words_per_row = grid_dimensions / 64;
    
    // Determine which cell (word) this thread is responsible for.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || col >= words_per_row)
        return;
    
    // Compute indices for the current word and its neighbors.
    int idx = row * words_per_row + col;
    
    // Load current row's words.
    std::uint64_t center   = input[idx];
    std::uint64_t curr_left  = (col > 0) ? input[row * words_per_row + (col - 1)] : 0;
    std::uint64_t curr_right = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0;
    
    // Load top row's words; if at the top boundary, use 0.
    std::uint64_t top_center = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_idx = (row - 1) * words_per_row + col;
        top_center = input[top_idx];
        top_left   = (col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0;
        top_right  = (col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0;
    }
    
    // Load bottom row's words; if at the bottom boundary, use 0.
    std::uint64_t bottom_center = 0, bottom_left = 0, bottom_right = 0;
    if (row < grid_dimensions - 1) {
        int bot_idx = (row + 1) * words_per_row + col;
        bottom_center = input[bot_idx];
        bottom_left   = (col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0;
        bottom_right  = (col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0;
    }
    
    // The result word to be computed.
    std::uint64_t result = 0;
    
    // Process the 64 cells in the word in 8 groups of 8.
    for (int b = 0; b < 8; b++) {
        int offset = b * 8;
        // For each row, compute the 10-bit window for the current 8-cell block.
        // For top row: if no top row data available, window is 0.
        uint32_t top_window = (row > 0) ? extract_window(top_left, top_center, top_right, offset) : 0;
        // For current row, always available.
        uint32_t mid_window = extract_window(curr_left, center, curr_right, offset);
        // For bottom row.
        uint32_t bot_window = (row < grid_dimensions - 1) ? extract_window(bottom_left, bottom_center, bottom_right, offset) : 0;
        
        // Process 8 cells in this 8-bit block.
        for (int j = 0; j < 8; j++) {
            int bit_pos = offset + j;
            // Build the 9-bit index by extracting 3 consecutive bits from each window.
            // For a given cell in the block, the 3x3 block is:
            //   top: bits at positions [j], [j+1], [j+2] from top_window
            //   middle: bits at positions [j] (W), [j+1] (center), [j+2] (E) from mid_window
            //   bottom: bits at positions [j], [j+1], [j+2] from bot_window
            uint32_t idx3x3 =  (((top_window    >> j)   & 1)      ) |
                               (((top_window    >> (j+1)) & 1) << 1 ) |
                               (((top_window    >> (j+2)) & 1) << 2 ) |
                               (((mid_window    >> j)   & 1) << 3 ) |
                               (((mid_window    >> (j+1)) & 1) << 4 ) |
                               (((mid_window    >> (j+2)) & 1) << 5 ) |
                               (((bot_window    >> j)   & 1) << 6 ) |
                               (((bot_window    >> (j+1)) & 1) << 7 ) |
                               (((bot_window    >> (j+2)) & 1) << 8 );
            // Lookup the next state for this cell (0 or 1) from constant memory.
            std::uint64_t new_state = d_lut[idx3x3] & 1;
            result |= (new_state << bit_pos);
        }
    }
    
    // Write the computed word to the output grid.
    output[idx] = result;
}

//---------------------------------------------------------------------------
// Host function to run one step of Conway's Game of Life on the GPU.
// "input" and "output" point to bit-packed grid data in device memory.
// "grid_dimensions" is the width/height of the square grid (a power of 2, >512).
//---------------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // !!! THE SECTION BELOW **WAS** GENERATED BY GPT !!!
    //     to measure performance properly we initialized the lookup 
    //     table only once in `initialize_internal_data_structures`

    // // Precompute the 512-entry lookup table in host memory.
    // // For each possible 3x3 block (9 bits), compute the new state for the center cell.
    // unsigned char h_lut[512];
    // for (int i = 0; i < 512; i++) {
    //     // Extract bits from the 9-bit index.
    //     int top_nw = (i >> 0) & 1;
    //     int top_n  = (i >> 1) & 1;
    //     int top_ne = (i >> 2) & 1;
    //     int mid_w  = (i >> 3) & 1;
    //     int center = (i >> 4) & 1; // center cell (its own state)
    //     int mid_e  = (i >> 5) & 1;
    //     int bot_nw = (i >> 6) & 1;
    //     int bot_n  = (i >> 7) & 1;
    //     int bot_ne = (i >> 8) & 1;
    //     // Sum the eight neighbor bits (all except the center, bit4).
    //     int sum = top_nw + top_n + top_ne + mid_w + mid_e + bot_nw + bot_n + bot_ne;
    //     // The Game of Life rule:
    //     //   A cell becomes (or remains) alive if it has exactly 3 neighbors,
    //     //   or if it is currently alive and has exactly 2 neighbors.
    //     h_lut[i] = (sum == 3 || (center && sum == 2)) ? 1 : 0;
    // }
    // // Copy the lookup table to device constant memory.
    // cudaMemcpyToSymbol(d_lut, h_lut, sizeof(h_lut));
    
    // Compute grid layout for kernel launch.
    int words_per_row = grid_dimensions / 64;
    // We'll launch one thread per 64-bit word.
    // Choose a 2D block size (adjustable for your hardware).
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
    
    // Note: Host-device synchronization is assumed to be handled by the caller.
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;

    // Precompute the 512-entry lookup table in host memory.
    // For each possible 3x3 block (9 bits), compute the new state for the center cell.
    unsigned char h_lut[512];
    for (int i = 0; i < 512; i++) {
        // Extract bits from the 9-bit index.
        int top_nw = (i >> 0) & 1;
        int top_n  = (i >> 1) & 1;
        int top_ne = (i >> 2) & 1;
        int mid_w  = (i >> 3) & 1;
        int center = (i >> 4) & 1; // center cell (its own state)
        int mid_e  = (i >> 5) & 1;
        int bot_nw = (i >> 6) & 1;
        int bot_n  = (i >> 7) & 1;
        int bot_ne = (i >> 8) & 1;
        // Sum the eight neighbor bits (all except the center, bit4).
        int sum = top_nw + top_n + top_ne + mid_w + mid_e + bot_nw + bot_n + bot_ne;
        // The Game of Life rule:
        //   A cell becomes (or remains) alive if it has exactly 3 neighbors,
        //   or if it is currently alive and has exactly 2 neighbors.
        h_lut[i] = (sum == 3 || (center && sum == 2)) ? 1 : 0;
    }
    // Copy the lookup table to device constant memory.
    cudaMemcpyToSymbol(d_lut, h_lut, sizeof(h_lut));
}

// This label is used by the framework to identify the memory layout of the input and output arrays.
// MEMORY_LAYOUT: ROWS
