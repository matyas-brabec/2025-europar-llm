#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdint> /// @FIXED

// This implementation uses a row‐based bitwise encoding with 16‐bit words,
// where each word packs 16 cells (one bit per cell). To compute the neighbor
// count for each cell, we apply full adder logic in parallel on the bit
// columns. Each neighbor (from one of the 8 directions) is first converted into
// a “digit” whose value is either 0 or 1. These 1‐bit digits are then added
// using bit‐sliced full adder logic so that each cell’s neighbor sum (ranging from 0 to 8)
// is represented by a 4‐bit number (stored in 4 separate bitmasks, one per bit).
//
// The grid accesses for neighbor cells account for horizontal boundaries by
// fetching bits from adjacent words when needed. Vertical boundaries are handled
// by checking the row index (cells outside the grid are considered dead).
//
// After computing the neighbor count for each cell, the Game of Life rules are applied:
//   - For an alive cell (bit = 1): it survives if the neighbor count is 2 or 3.
//   - For a dead cell (bit = 0): it becomes alive if the neighbor count is exactly 3.
//
// The final result is repacked into a 16‐bit word for each block of 16 cells.


// Structure to hold a “digit” in 4 bits per cell.
// For each cell (each bit position in a 16‐bit word), the neighbor count is represented as:
//   value = d0 + 2*d1 + 4*d2 + 8*d3, where each dX is a bitmask (16 bits) holding the Xth bit
// for each cell independently.
struct Digit {
    uint16_t r0;
    uint16_t r1;
    uint16_t r2;
    uint16_t r3;
};

// Convert a 1‐bit value (0 or 1 for each cell) into a 4‐bit digit representation.
__device__ inline Digit make_digit(uint16_t bit) {
    Digit d;
    d.r0 = bit;  // the value is in the least‐significant bit position
    d.r1 = 0;
    d.r2 = 0;
    d.r3 = 0;
    return d;
}

// Add two digits using full adder logic per cell (i.e. per bit column).
// Each addition is performed independently without propagating carry across cell boundaries.
__device__ inline Digit add_digit(const Digit &A, const Digit &B) {
    Digit R;
    // Add the LSBs.
    uint16_t s0 = A.r0 ^ B.r0;
    uint16_t c0 = A.r0 & B.r0;
    // Add the next bits along with the carry from the previous column.
    uint16_t s1 = A.r1 ^ B.r1 ^ c0;
    uint16_t c1 = (A.r1 & B.r1) | (A.r1 & c0) | (B.r1 & c0);
    // Continue for the third bit.
    uint16_t s2 = A.r2 ^ B.r2 ^ c1;
    uint16_t c2 = (A.r2 & B.r2) | (A.r2 & c1) | (B.r2 & c1);
    // And the fourth bit.
    uint16_t s3 = A.r3 ^ B.r3 ^ c2;
    // The final carry is ignored (maximum sum is 8, fitting in 4 bits).
    R.r0 = s0;
    R.r1 = s1;
    R.r2 = s2;
    R.r3 = s3;
    return R;
}

// Add eight digits using pairwise full-adder additions.
__device__ inline Digit add_multiple(Digit a, Digit b, Digit c, Digit d,
                                      Digit e, Digit f, Digit g, Digit h) {
    Digit t0 = add_digit(a, b);
    Digit t1 = add_digit(c, d);
    Digit t2 = add_digit(e, f);
    Digit t3 = add_digit(g, h);
    Digit u0 = add_digit(t0, t1);
    Digit u1 = add_digit(t2, t3);
    Digit sum = add_digit(u0, u1);
    return sum;
}

// Extract the 4-bit neighbor count for a given cell (bit position) from the digit.
__device__ inline int extract_digit(const Digit &d, int bit_index) {
    int b0 = (d.r0 >> bit_index) & 1;
    int b1 = (d.r1 >> bit_index) & 1;
    int b2 = (d.r2 >> bit_index) & 1;
    int b3 = (d.r3 >> bit_index) & 1;
    return b0 + (b1 << 1) + (b2 << 2) + (b3 << 3);
}

// Block dimensions for kernel execution.
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

// CUDA kernel that computes one generation of Conway's Game of Life using full adder logic
// to sum neighbor bits concurrently. The grid is stored in 16-bit words (16 cells per word).
__global__ void game_of_life_bit_adder_kernel(const uint16_t* input, uint16_t* output, int grid_dim) {
    // Compute the number of 16-bit words per row.
    int words_per_row = grid_dim / 16;
    
    // Determine the row and word index for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;
    
    // Linear index into the grid.
    int index = row * words_per_row + word_idx;
    
    // Load the current word (16 cells) for this row.
    uint16_t cur = input[index];
    
    // Fetch horizontal neighbors from the same row.
    uint16_t cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    uint16_t cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;
    
    // Fetch words from the row above.
    uint16_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Fetch words from the row below.
    uint16_t bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Compute neighbor bit vectors for each of the 8 directions.
    // For horizontal shifts, a left shift (<< 1) gives the left neighbor bits,
    // and a right shift (>> 1) gives the right neighbor bits.
    // Boundary bits are fixed by fetching the appropriate bit from adjacent words.
    uint16_t n_top_left  = (word_idx > 0) ? ((top_left << 1) | ((top_left >> 15) & 1)) : (top << 1);
    uint16_t n_top_center = top;
    uint16_t n_top_right = (word_idx < words_per_row - 1) ? ((top >> 1) | ((top_right & 1) << 15)) : (top >> 1);
    
    uint16_t n_left  = (word_idx > 0) ? ((cur_left << 1) | ((cur_left >> 15) & 1)) : (cur << 1);
    uint16_t n_right = (word_idx < words_per_row - 1) ? ((cur >> 1) | ((cur_right & 1) << 15)) : (cur >> 1);
    
    uint16_t n_bot_left  = (word_idx > 0) ? ((bot_left << 1) | ((bot_left >> 15) & 1)) : (bot << 1);
    uint16_t n_bot_center = bot;
    uint16_t n_bot_right = (word_idx < words_per_row - 1) ? ((bot >> 1) | ((bot_right & 1) << 15)) : (bot >> 1);
    
    // Convert each neighbor contribution (16-bit, one bit per cell) into a 4-bit digit.
    Digit d_top_left    = make_digit(n_top_left);
    Digit d_top_center  = make_digit(n_top_center);
    Digit d_top_right   = make_digit(n_top_right);
    Digit d_left        = make_digit(n_left);
    Digit d_right       = make_digit(n_right);
    Digit d_bot_left    = make_digit(n_bot_left);
    Digit d_bot_center  = make_digit(n_bot_center);
    Digit d_bot_right   = make_digit(n_bot_right);
    
    // Sum all 8 neighbor contributions using full adder logic.
    Digit neighbor_sum = add_multiple(d_top_left, d_top_center, d_top_right,
                                      d_left, d_right, d_bot_left, d_bot_center, d_bot_right);
    
    // Determine the new state for each cell according to the Game of Life rules.
    // For each cell (each bit position in the 16-bit word), extract the neighbor count
    // (a 4-bit number) and apply:
    //   - If the cell is alive: survive if count == 2 or count == 3.
    //   - If the cell is dead: become alive if count == 3.
    uint16_t new_word = 0;
    for (int i = 0; i < 16; i++) {
        int count = extract_digit(neighbor_sum, i);
        int cell = (cur >> i) & 1;
        int new_state = 0;
        if (cell)
            new_state = (count == 2 || count == 3) ? 1 : 0;
        else
            new_state = (count == 3) ? 1 : 0;
        new_word |= (new_state << i);
    }
    
    // Write the computed new state for the 16 cells back to global memory.
    output[index] = new_word;
}

// Host function to execute one simulation step of Conway's Game of Life using the
// full adder (bitwise) approach on a bit-packed grid represented with 16-bit words.
// 'grid_dimensions' is the number of cells per row (must be divisible by 16).
/// @FIXED void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate the number of 16-bit words per row.
    int words_per_row = grid_dimensions / 16;
    
    // Define block dimensions.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);
    
    // Launch the kernel. Any necessary host-device synchronization is assumed to be handled externally.
    /// @FIXED game_of_life_bit_adder_kernel<<<grid, block>>>(input, output, grid_dimensions);
    game_of_life_bit_adder_kernel<<<grid, block>>>((uint16_t*)input, (uint16_t*)output, grid_dimensions);
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
