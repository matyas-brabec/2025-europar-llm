#include <cuda_runtime.h>
#include <cstdint>

//---------------------------------------------------------------------
// In this implementation each thread processes one 64‐cell word 
// (std::uint64_t) in a bit‐packed grid. The grid is square with
// grid_dimensions cells per row and each row consists of (grid_dimensions/64)
// 64‐bit words. We assume that the bit packing is such that bit 0 is the left‐most
// cell and bit 63 is the right‐most cell in that word.
// 
// For each cell, its 8 neighbors come from the 3 rows (upper, same, lower)
// and 3 horizontal positions (left, center, right). For the same row, we only 
// take the left and right words (shifting the current word appropriately) 
// because the cell itself (center) is not a neighbor.
// 
// To obtain the horizontal neighbor bits “aligned” with the current word, we 
// use two helper functions that shift the word while “injecting” the bit from 
// the neighboring word as needed at the boundary:
//
//   shift_left(center, left):
//     For each bit position i in [0,63] in the current word,
//       if (i==0) then result[i] = (left word’s bit 63) else result[i] = center[i-1].
//     This is implemented as: (center << 1) | (left >> 63)
//     
//   shift_right(center, right):
//     For each bit position i in [0,63],
//       if (i==63) then result[i] = (right word’s bit 0) else result[i] = center[i+1].
//     This is implemented as: (center >> 1) | ((right & 1ULL) << 63)
//
// We then compute eight neighbor masks for the top, same, and bottom rows:
//   Top row:    top-left, top-center, top-right
//   Same row:   left,       (center is not used), right
//   Bottom row: bottom-left, bottom-center, bottom-right
//
// Each of these is a 64‐bit word where every bit is either 0 or 1 (a neighbor’s “alive” flag).
//
// To compute the neighbor count for each cell (each bit lane) we add the 8 bit masks
// using bit‐parallel “full‐adder” logic. Because each thread “lives” on a 64‐bit word,
// we are performing 64 independent 4–bit additions (each cell gets a count between 0 and 8).
// Instead of storing the 4–bit sum in one integer (which would require 256 bits),
// we store the four bits (bit0, bit1, bit2, bit3) in four separate 64–bit variables.
// 
// To add a single “bit–value” (mask m) to the current 4–bit count (stored in s0,s1,s2,s3)
// we use a bit–slice ripple–carry adder that works lane–wise:
//
//   For each lane, let the current count be represented as (s3,s2,s1,s0) (s0 is LSB).
//   Then we add m (which is 0 or 1) to each lane. Since m is 64–bit with each lane
//   holding the bit to add, we do:
//       carry = m;
//       s0 = s0 XOR carry;       carry = s0_old AND carry;
//       s1 = s1 XOR carry;       carry = s1_old AND carry;
//       s2 = s2 XOR carry;       carry = s2_old AND carry;
//       s3 = s3 XOR carry;       // carry beyond s3 is discarded (max count is 8).
//
// After accumulating the eight neighbor bits, we obtain a 4–bit count for each cell.
// 
// Finally, the Game of Life rule is applied per cell:
//   A cell will be alive in the next generation if either:
//     (a) it has exactly 3 neighbors, OR
//     (b) it is currently alive and has exactly 2 neighbors.
// 
// To test equality of the 4–bit count to a constant (e.g. 2 or 3) we reconstruct
// each cell’s 4–bit value from the four bit–planes. Since each thread’s 4 separate
// 64–bit integers hold one bit per cell, we loop over the 64 lanes, gather the four bits,
// and compare to the constant. (This loop is unrolled by the compiler for performance.)
//---------------------------------------------------------------------

// Inline device function: Shift the current word left to align left neighbors.
// 'center' is the word from the current row (or neighbor row) and 
// 'left' is the adjacent word to its left (or 0 if none).
// The result: for each bit i, if i==0, then result[i] = left's bit63; else result[i] = center[i-1].
__device__ __forceinline__ uint64_t shift_left(uint64_t center, uint64_t left) {
    return (center << 1) | (left >> 63);
}

// Inline device function: Shift the current word right to align right neighbors.
// 'center' is the word from the current row (or neighbor row) and
// 'right' is the adjacent word to its right (or 0 if none).
// The result: for each bit i, if i==63, then result[i] = right's bit0; else result[i] = center[i+1].
__device__ __forceinline__ uint64_t shift_right(uint64_t center, uint64_t right) {
    return (center >> 1) | ((right & 1ULL) << 63);
}

// Device function: add a single bit (mask m) to the bit-sliced 4-bit counter (s0,s1,s2,s3)
// where each of s0, s1, s2, s3 is a 64-bit integer holding one bit from each cell's counter.
__device__ __forceinline__ void add_bit(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3, uint64_t m) {
    uint64_t carry = m;
    uint64_t tmp;
    // Add to bit 0 (LSB)
    tmp = s0;
    s0 = tmp ^ carry;
    carry = tmp & carry;
    // Add to bit 1
    tmp = s1;
    s1 = tmp ^ carry;
    carry = tmp & carry;
    // Add to bit 2
    tmp = s2;
    s2 = tmp ^ carry;
    carry = tmp & carry;
    // Add to bit 3 (highest needed, since maximum count is 8)
    tmp = s3;
    s3 = tmp ^ carry;
    // We discard any carry beyond bit 3.
}

// Device function: Given the four bit-planes s0,s1,s2,s3 (each 64-bit) representing
// the 4-bit counts for 64 cells (one per bit/lane), return a 64-bit mask in which bit i is 1
// if and only if the 4-bit number for lane i equals the constant 'val' (which is in the range [0,8]).
// This implementation loops over the 64 lanes and assembles the 4-bit number for each lane.
__device__ __forceinline__ uint64_t equals4bit(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, int val) {
    uint64_t eq_mask = 0;
    // For each lane (bit position i), extract its 4-bit count.
    // Note: Since each sX has either 0 or 1 in each lane (but located at bit position i),
    // we extract that bit and combine into a 4-bit number.
#pragma unroll
    for (int i = 0; i < 64; i++) {
        // Extract bit i from each plane.
        unsigned int bit0 = (s0 >> i) & 1U;
        unsigned int bit1 = (s1 >> i) & 1U;
        unsigned int bit2 = (s2 >> i) & 1U;
        unsigned int bit3 = (s3 >> i) & 1U;
        unsigned int count = (bit3 << 3) | (bit2 << 2) | (bit1 << 1) | bit0;
        // If count equals the desired value, set the corresponding bit in eq_mask.
        if (count == static_cast<unsigned int>(val))
            eq_mask |= (1ULL << i);
    }
    return eq_mask;
}

//---------------------------------------------------------------------
// CUDA kernel for one Game of Life iteration. Each thread processes one
// std::uint64_t representing 64 cells (a word) in the grid.
//---------------------------------------------------------------------
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Calculate indices:
    // Let each thread handle one 64-bit word.
    // The grid is organized such that there are (grid_dimensions) rows,
    // and each row has (grid_dimensions/64) words.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    
    // Compute row index (cell row) and column index (word index in the row)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds: row must be in [0, grid_dimensions) and col in [0, words_per_row)
    if (row >= grid_dimensions || col >= words_per_row)
        return;
    
    // Compute the base index into the input/output arrays.
    int base_idx = row * words_per_row + col;
    
    // Load the "center" word (current row, this word).
    std::uint64_t center = input[base_idx];
    
    // For neighbor computation we will need words from adjacent columns.
    // For left and right words in the same row:
    std::uint64_t left_word   = (col > 0) ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t right_word  = (col < words_per_row - 1) ? input[row * words_per_row + (col + 1)] : 0ULL;
    
    // For the row above:
    std::uint64_t top_center = (row > 0) ? input[(row - 1) * words_per_row + col] : 0ULL;
    std::uint64_t top_left_word  = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t top_right_word = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + (col + 1)] : 0ULL;
    
    // For the row below:
    std::uint64_t bot_center = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0ULL;
    std::uint64_t bot_left_word  = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t bot_right_word = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + (col + 1)] : 0ULL;
    
    // Compute neighbor masks with proper horizontal alignment.
    // Top row neighbors:
    std::uint64_t top_left   = shift_left(top_center, top_left_word);
    std::uint64_t top_center_aligned = top_center;  // no horizontal shift.
    std::uint64_t top_right  = shift_right(top_center, top_right_word);
    
    // Same row neighbors (only left and right, not center):
    std::uint64_t same_left  = shift_left(center, left_word);
    std::uint64_t same_right = shift_right(center, right_word);
    
    // Bottom row neighbors:
    std::uint64_t bot_left   = shift_left(bot_center, bot_left_word);
    std::uint64_t bot_center_aligned = bot_center;  // no shift.
    std::uint64_t bot_right  = shift_right(bot_center, bot_right_word);
    
    // We have eight neighbor contributions:
    // Top row: top_left, top_center_aligned, top_right
    // Same row: same_left, same_right
    // Bottom row: bot_left, bot_center_aligned, bot_right
    
    // Use bit-sliced addition to sum neighbor bits (per cell/lane).
    // We use four 64-bit accumulators to hold the 4-bit count for 64 cells.
    std::uint64_t count0 = 0, count1 = 0, count2 = 0, count3 = 0;
    
    // Add each neighbor's contribution (each is a bit mask with 0 or 1 per lane).
    add_bit(count0, count1, count2, count3, top_left);
    add_bit(count0, count1, count2, count3, top_center_aligned);
    add_bit(count0, count1, count2, count3, top_right);
    add_bit(count0, count1, count2, count3, same_left);
    add_bit(count0, count1, count2, count3, same_right);
    add_bit(count0, count1, count2, count3, bot_left);
    add_bit(count0, count1, count2, count3, bot_center_aligned);
    add_bit(count0, count1, count2, count3, bot_right);
    
    // According to the Game of Life rules, a cell is live next generation if:
    //   (neighbors == 3) OR (cell is live AND neighbors == 2)
    // Compute masks for (neighbor count == 2) and (neighbor count == 3).
    std::uint64_t mask_eq2 = equals4bit(count0, count1, count2, count3, 2);
    std::uint64_t mask_eq3 = equals4bit(count0, count1, count2, count3, 3);
    
    // Next state: cells that have exactly 3 neighbors, or live cells with exactly 2 neighbors.
    std::uint64_t next = mask_eq3 | (mask_eq2 & center);
    
    // Write the result back to the output grid.
    output[base_idx] = next;
}

//---------------------------------------------------------------------
// Host function to perform one step of Conway's Game of Life on the GPU.
// 'input'  - Device pointer to the bit-packed input grid (std::uint64_t*).
// 'output' - Device pointer to the bit-packed output grid (std::uint64_t*).
// 'grid_dimensions' - Number of cells per row (and per column); always a power of 2.
//---------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row has grid_dimensions/64 words.
    int words_per_row = grid_dimensions >> 6;
    
    // Define CUDA block and grid dimensions.
    // We'll use a 2D block; choose 16x16 threads per block.
    dim3 blockDim(16, 16);
    // Grid dimensions: total threads in x = words_per_row, in y = grid_dimensions.
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
}

// This label is used by the framework to identify the memory layout of the input and output arrays.
// MEMORY_LAYOUT: ROWS
