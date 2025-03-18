// Conway’s Game of Life in CUDA.
// This implementation processes the grid in its bit‐packed form: each
// std::uint64_t holds 64 cells (one bit per cell). Each thread processes one
// such word (i.e. 64 consecutive cells in a row). To update a cell one must sum
// the 8 neighbor bits (from the row above, same row [neighbors only], and row below).
// In order to process multiple cells in parallel without a 64‐iteration inner loop,
// we “accumulate” the neighbor counts in a bit‐sliced fashion. In this scheme each
// cell (bit lane) gets its 4–bit count stored across four 64–bit registers (cnt0, cnt1, cnt2, cnt3),
// where for each cell the least–significant bit is in cnt0, then cnt1, cnt2, cnt3.
// (For example, for a given bit position j, the neighbor count is:
//    count = ( (cnt3 >> j) & 1 ) * 8 + ((cnt2 >> j)&1)*4 + ((cnt1 >> j)&1)*2 + ((cnt0 >> j)&1)
//)
// Then the Game–of–Life rule (a live cell survives if its neighbor count is 2 or 3,
// but a dead cell becomes live only if its neighbor count is exactly 3) is applied
// using bit–logical operations.
// 
// Special handling is required at word boundaries. For each word, its 8 neighbors
// come from the three words above and the three words below, plus left and right words
// on the same row; for the leftmost and rightmost cells within a word, the adjacent word’s
// bits are used (if available) to supply the missing neighbor bits; if the neighbor row/word
// is outside the grid, those bits are taken as 0 (dead).
//
// This implementation avoids shared or texture memory and focuses on performance.
// Each thread loads its “current” word and the 8 neighboring words (when available)
// and then computes the 8 shifted “masks” corresponding to the 8 neighbor directions.
//
// The neighbor contribution for a given direction is computed as follows:
//   • For a vertical neighbor (e.g. top‐center) the word is used as is.
//   • For a diagonal neighbor (e.g. top–left) we shift the “top center” word left by 1,
//     and for bit0 we supplement from the “top left” word (if available).
//   • For the same row, left‐neighbor is obtained by (current << 1) plus the carry from
//     the left word’s bit63 (if available), and right–neighbor similarly.
// These 8 masks are then “added” into bit–sliced 4–bit counters using a custom inline
// full–adder that works lane–wise (each bit of the 64–bit word represents one cell).
//
// Finally, the new state for each cell is computed by comparing the neighbor count:
// live if (count==3) OR (current cell live AND count==2).
//
// The run_game_of_life() function sets up and launches the kernel.
// All memory (input and output) is assumed allocated using cudaMalloc.
// Host/device synchronization is assumed to be handled externally.

#include <cstdint>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Inline device function that adds 1 (for lanes indicated by mask m)
// to a bit-sliced nibble stored in (a, b, c, d).
// Here each of a,b,c,d is a 64-bit integer whose bit j stores one bit
// (of the 4-bit count for cell j). We add (m?1:0) lane–wise.
// The full adder is computed independently for each lane.
inline __device__ void add_one(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t m) {
    // For each cell (bit position j), let a = bit0, b = bit1, c = bit2, d = bit3.
    // We add: nibble = nibble + ( (m>>j)&1 ).
    uint64_t carry = a & m;
    uint64_t new_a = a ^ m;
    uint64_t old_b = b;
    b = b ^ carry;
    carry = old_b & carry;
    uint64_t old_c = c;
    c = c ^ carry;
    carry = old_c & carry;
    d = d ^ carry;
    a = new_a;
}

//---------------------------------------------------------------------
// The CUDA kernel that processes one word (64 cells) per thread.
__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dimensions) {
    // Each row of the grid is grid_dimensions bits.
    // Number of 64-bit words per row:
    int words_per_row = grid_dimensions >> 6; // equivalent to grid_dimensions/64
    int total_words = grid_dimensions * words_per_row;
    int word_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_index >= total_words) return;

    // Compute row and column (in word units)
    int row = word_index / words_per_row;
    int col = word_index % words_per_row;

    // Index for current word:
    int curr_index = row * words_per_row + col;
    uint64_t current = input[curr_index];

    // Load neighboring words from adjacent rows and columns.
    // For rows above/below.
    uint64_t topC = (row > 0) ? input[(row - 1) * words_per_row + col] : 0ULL;
    uint64_t bottomC = (row < grid_dimensions - 1) ? input[(row + 1) * words_per_row + col] : 0ULL;
    // Left and right words in the same row.
    uint64_t left_current = (col > 0) ? input[row * words_per_row + col - 1] : 0ULL;
    uint64_t right_current = (col < words_per_row - 1) ? input[row * words_per_row + col + 1] : 0ULL;
    // Diagonal neighbors.
    uint64_t top_left = (row > 0 && col > 0) ? input[(row - 1) * words_per_row + col - 1] : 0ULL;
    uint64_t top_right = (row > 0 && col < words_per_row - 1) ? input[(row - 1) * words_per_row + col + 1] : 0ULL;
    uint64_t bottom_left = (row < grid_dimensions - 1 && col > 0) ? input[(row + 1) * words_per_row + col - 1] : 0ULL;
    uint64_t bottom_right = (row < grid_dimensions - 1 && col < words_per_row - 1) ? input[(row + 1) * words_per_row + col + 1] : 0ULL;

    // Compute shifted neighbor masks.
    // For top-left: we want for each cell in current word,
    // the top-left neighbor comes from topC shifted left by 1 (so that for bit i, we get topC bit i-1)
    // and for cell 0 (i==0) use the top_left word's bit63.
    uint64_t top_left_mask = 0;
    if (row > 0) {
        uint64_t shifted = topC << 1;
        uint64_t extra = (col > 0) ? ((top_left >> 63) & 1ULL) : 0ULL;
        top_left_mask = shifted | extra;
    }
    // Top-center: just the top row (if exists)
    uint64_t top_center_mask = (row > 0) ? topC : 0ULL;
    // Top-right: topC shifted right by 1; for cell 63, supplement from top_right's bit0.
    uint64_t top_right_mask = 0;
    if (row > 0) {
        uint64_t shifted = topC >> 1;
        uint64_t extra = (col < words_per_row - 1) ? ((top_right & 1ULL) << 63) : 0ULL;
        top_right_mask = shifted | extra;
    }
    // Left neighbor in same row: current shifted left by 1; for cell 0, use left_current's bit63.
    uint64_t left_mask = (col > 0) ? ((current << 1) | ((left_current >> 63) & 1ULL))
                                  : (current << 1);
    // Right neighbor in same row: current shifted right by 1; for cell 63, use right_current's bit0.
    uint64_t right_mask = (col < words_per_row - 1) ? ((current >> 1) | ((right_current & 1ULL) << 63))
                                                    : (current >> 1);
    // Bottom row:
    uint64_t bottom_left_mask = 0;
    if (row < grid_dimensions - 1) {
        uint64_t shifted = bottomC << 1;
        uint64_t extra = (col > 0) ? ((bottom_left >> 63) & 1ULL) : 0ULL;
        bottom_left_mask = shifted | extra;
    }
    uint64_t bottom_center_mask = (row < grid_dimensions - 1) ? bottomC : 0ULL;
    uint64_t bottom_right_mask = 0;
    if (row < grid_dimensions - 1) {
        uint64_t shifted = bottomC >> 1;
        uint64_t extra = (col < words_per_row - 1) ? ((bottom_right & 1ULL) << 63) : 0ULL;
        bottom_right_mask = shifted | extra;
    }

    // Now, the 8 neighbor masks (each bit in a mask represents a contribution of 1 for that cell).
    // They are:
    //    top-left_mask, top_center_mask, top_right_mask,
    //    left_mask, right_mask,
    //    bottom_left_mask, bottom_center_mask, bottom_right_mask

    // Accumulate neighbor counts in bit-sliced 4-bit registers.
    // For each cell (bit position) the 4–bit nibble (cnt3 cnt2 cnt1 cnt0) will hold the neighbor count.
    uint64_t cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
    add_one(cnt0, cnt1, cnt2, cnt3, top_left_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, top_center_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, top_right_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, left_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, right_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, bottom_left_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, bottom_center_mask);
    add_one(cnt0, cnt1, cnt2, cnt3, bottom_right_mask);

    // Now for each cell, the neighbor count is in the nibble (cnt3 cnt2 cnt1 cnt0).
    // According to Game-of-Life:
    //   - A dead cell becomes alive if it has exactly 3 live neighbors.
    //   - A live cell remains alive if it has 2 or 3 live neighbors.
    // Since our counts do NOT include the current cell, the rule becomes:
    //   next = (neighbor_count == 3) OR (current_cell AND neighbor_count == 2)
    //
    // For our 4–bit representation, we can test equality with 2 and 3.
    // Note: 2 == 0010 (bits: cnt3=0, cnt2=0, cnt1=1, cnt0=0)
    //       3 == 0011 (bits: cnt3=0, cnt2=0, cnt1=1, cnt0=1)
    uint64_t eq2 = ((~cnt3) & (~cnt2) & cnt1 & (~cnt0));
    uint64_t eq3 = ((~cnt3) & (~cnt2) & cnt1 & cnt0);
    uint64_t next = eq3 | (current & eq2);

    // Write the updated 64 cells (in bit-packed form) to the output grid.
    output[word_index] = next;
}

//---------------------------------------------------------------------
// The run_game_of_life function launches one step of the simulation.
// It assumes that 'input' and 'output' point to device memory (allocated via cudaMalloc)
// and that the grid is square with side length grid_dimensions in cells.
// The grid is stored in bit-packed form (64 cells per std::uint64_t).
/// @FIXED: extern "C" 

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row has grid_dimensions bits; therefore, number of 64-bit words per row:
    int words_per_row = grid_dimensions >> 6; // grid_dimensions/64
    int total_words = grid_dimensions * words_per_row;
    // Launch configuration: use 256 threads per block.
    int block_size = 256;
    int num_blocks = (total_words + block_size - 1) / block_size;
    game_of_life_kernel<<<num_blocks, block_size>>>(input, output, grid_dimensions);
    // Note: Host/device synchronization is assumed to be handled externally.
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
