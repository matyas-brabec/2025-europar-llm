#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdint> /// @FIXED

// In this implementation the grid is stored in a 16‐bit word per block of 16 cells.
// Bit ordering: bit 0 is the leftmost cell, bit 15 is the rightmost cell.
// For each cell, its 8 neighbors (top‐left, top, top‐right, left, right, bottom‐left, bottom, bottom‐right)
// are extracted from the corresponding neighbor words with proper shifts (using 0 for out‐of‐bounds).
// To compute the neighbor count per cell in parallel, we represent each one‐bit neighbor
// as a “digit” (in a bit‐sliced form) and sum eight such digits using full‐adder logic.
// The final 4‐bit result per cell (stored across 4 bitmasks) gives the neighbor count (0–8).
// Game of Life rules are then applied to produce the next state.

struct Digit {
    uint16_t r0;
    uint16_t r1;
    uint16_t r2;
    uint16_t r3;
};

// Convert a 16-bit mask (each bit 0 or 1) into a digit (bit-sliced 4-bit number per cell).
__device__ inline Digit make_digit(uint16_t bit) {
    Digit d;
    d.r0 = bit;  // the least-significant slice holds the value (0 or 1)
    d.r1 = 0;
    d.r2 = 0;
    d.r3 = 0;
    return d;
}

// Add two digits using bitwise full-adder logic per cell (each cell is one bit in the 16-bit mask).
// Since each digit is only 0 or 1 initially (and sums up to 8), no cross-cell carry occurs.
__device__ inline Digit add_digit(const Digit &A, const Digit &B) {
    Digit R;
    uint16_t s0 = A.r0 ^ B.r0;          // sum of LSBs
    uint16_t c0 = A.r0 & B.r0;          // carry from LSBs
    uint16_t s1 = A.r1 ^ B.r1 ^ c0;
    uint16_t c1 = (A.r1 & B.r1) | (A.r1 & c0) | (B.r1 & c0);
    uint16_t s2 = A.r2 ^ B.r2 ^ c1;
    uint16_t c2 = (A.r2 & B.r2) | (A.r2 & c1) | (B.r2 & c1);
    uint16_t s3 = A.r3 ^ B.r3 ^ c2;
    // Final carry is ignored (maximum sum is 8, fitting in 4 bits).
    R.r0 = s0;
    R.r1 = s1;
    R.r2 = s2;
    R.r3 = s3;
    return R;
}

// Sum eight digits by pairwise additions.
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

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

// CUDA kernel implementing one Game of Life generation using full-adder logic on a 16-bit packed grid.
// Each thread processes one 16-bit word representing 16 cells.
__global__ void game_of_life_bit_adder_kernel(const uint16_t* input, uint16_t* output, int grid_dim) {
    // Compute the number of 16-bit words per row.
    int words_per_row = grid_dim / 16;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;
    
    int index = row * words_per_row + word_idx;
    uint16_t cur = input[index];
    
    // Fetch horizontal neighbors from the current row.
    uint16_t cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    uint16_t cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;
    
    // Fetch vertical neighbor rows.
    uint16_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    uint16_t bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Compute neighbor masks with corrected shifts.
    // For left neighbors: each cell's left neighbor is in the same row, one position to the left.
    // Compute by shifting the current word left by 1; for cell 0, use previous word's rightmost bit if available.
    uint16_t n_left = cur << 1;
    if (word_idx > 0) {
        uint16_t extra = (cur_left >> 15) & 1;
        n_left = (n_left & 0xFFFE) | extra;  // clear bit 0 and set it to extra
    }
    // For right neighbors: shift the current word right by 1; for cell 15, use next word's leftmost bit if available.
    uint16_t n_right = cur >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = cur_right & 1;
        n_right = (n_right & 0x7FFF) | (extra << 15);  // clear bit 15 and set it to extra
    }
    
    // Top neighbors (no horizontal shift).
    uint16_t n_top_center = top;
    // Top-left: shift top row left by 1; for cell 0, use top_left's rightmost bit.
    uint16_t n_top_left = top << 1;
    if (word_idx > 0) {
        uint16_t extra = (top_left >> 15) & 1;
        n_top_left = (n_top_left & 0xFFFE) | extra;
    }
    // Top-right: shift top row right by 1; for cell 15, use top_right's leftmost bit.
    uint16_t n_top_right = top >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = top_right & 1;
        n_top_right = (n_top_right & 0x7FFF) | (extra << 15);
    }
    
    // Bottom neighbors.
    uint16_t n_bot_center = bot;
    uint16_t n_bot_left = bot << 1;
    if (word_idx > 0) {
        uint16_t extra = (bot_left >> 15) & 1;
        n_bot_left = (n_bot_left & 0xFFFE) | extra;
    }
    uint16_t n_bot_right = bot >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = bot_right & 1;
        n_bot_right = (n_bot_right & 0x7FFF) | (extra << 15);
    }
    
    // Convert each neighbor mask into a digit.
    Digit d_top_left    = make_digit(n_top_left);
    Digit d_top_center  = make_digit(n_top_center);
    Digit d_top_right   = make_digit(n_top_right);
    Digit d_left        = make_digit(n_left);
    Digit d_right       = make_digit(n_right);
    Digit d_bot_left    = make_digit(n_bot_left);
    Digit d_bot_center  = make_digit(n_bot_center);
    Digit d_bot_right   = make_digit(n_bot_right);
    
    // Sum all 8 neighbor contributions using full-adder logic.
    Digit neighbor_sum = add_multiple(d_top_left, d_top_center, d_top_right,
                                      d_left, d_right, d_bot_left, d_bot_center, d_bot_right);
    
    // Compute next state for each cell (each bit in the 16-bit word) based on Game of Life rules.
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
    
    output[index] = new_word;
}

// Host function to launch one generation of Conway's Game of Life using the full-adder method.
// 'input' and 'output' are pointers to device memory (allocated via cudaMalloc).
// 'grid_dimensions' is the number of cells per row (must be divisible by 16).
/// @FIXED void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    int words_per_row = grid_dimensions / 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);
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
