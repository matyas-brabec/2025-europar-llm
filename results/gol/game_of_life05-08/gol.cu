#include <cstdint>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// This helper function expands a 16‐bit value (representing 16 cells, one bit per cell)
// into a 64‐bit integer with 16 independent 4‐bit “lanes.” In lane i (i from 0 to 15)
// the lower 4 bits will hold a value of either 0 or 1. With 4 bits per lane, we
// can add several such “lane‐values” without any carry affecting adjacent lanes
// (since maximum sum is 8, well below 16).
// We use a simple loop (unrolled to help optimization) to do the expansion.
__device__ __forceinline__ uint64_t expand16(uint16_t bits) {
    uint64_t result = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // If bit i is set, place a 1 in the 4-bit field at nibble position i.
        if (bits & (1u << i))
            result |= (uint64_t)1 << (i * 4);
    }
    return result;
}

//---------------------------------------------------------------------
// The CUDA kernel that computes one generation (step) of Conway's Game of Life.
// The grid is bit‐packed: each 64‐bit word holds 64 cells (1 bit per cell).
// Each thread is responsible for one 64‐bit word (i.e. a block of 64 cells).
//
// Because the update rule for a cell depends on its 8 neighbors (in rows above,
// same, and below), each thread reads 3 rows × 3 words (left, center, right) -- with
// proper boundary checks. For cells in the middle of the word (bits 1..62) the neighbor
// contributions come from the same word, but special handling is needed for bit0 and bit63
// to pull in bits from the "left" or "right" neighboring 64‐bit words.
//
// To combine contributions from the 8 neighbor “sources” without processing each cell
// individually, we use “lane‐wise” arithmetic. We split the 64 cells of a word into 4 groups
// of 16. For each neighbor mask (which is a 64‐bit word with 0/1 per cell), we extract its
// four 16‐bit chunks and use the helper function expand16() to “widen” each 1-bit value into a
// 4‐bit lane. Then we add contributions from all 8 neighbor masks into 4 accumulators (one per group).
// Finally, for each of the 16 lanes in each accumulator, we check whether the neighbor count (0..8)
// satisfies the rule: a cell comes alive if its neighbor count is 3, or if it is already live and the count is 2.
// We then combine the 4 groups (each yielding 16 result bits) into a 64‐bit answer.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions,
                                    int words_per_row)
{
    // Compute global thread index (each thread processes one word).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dimensions * words_per_row;
    if (idx >= total_words) return;
    
    // Determine the row and column (word index) for this thread.
    int row = idx / words_per_row;
    int col = idx % words_per_row;
    
    // Compute indices of adjacent rows.
    int row_above = row - 1;
    int row_below = row + 1;
    
    // For boundary handling, cells outside the grid are considered dead (0).
    // Load words for the row above (if available); otherwise 0.
    std::uint64_t above = 0, aboveLeft = 0, aboveRight = 0;
    if (row_above >= 0) {
        int baseIndex = row_above * words_per_row;
        above     = input[baseIndex + col];
        aboveLeft = (col > 0) ? input[baseIndex + col - 1] : 0ULL;
        aboveRight= (col < words_per_row - 1) ? input[baseIndex + col + 1] : 0ULL;
    }
    
    // Load words for the current row.
    int curIndex = row * words_per_row;
    std::uint64_t curWord = input[curIndex + col];
    std::uint64_t curLeft = (col > 0) ? input[curIndex + col - 1] : 0ULL;
    std::uint64_t curRight= (col < words_per_row - 1) ? input[curIndex + col + 1] : 0ULL;
    
    // Load words for the row below (if available); otherwise 0.
    std::uint64_t below = 0, belowLeft = 0, belowRight = 0;
    if (row_below < grid_dimensions) {
        int baseIndex = row_below * words_per_row;
        below      = input[baseIndex + col];
        belowLeft  = (col > 0) ? input[baseIndex + col - 1] : 0ULL;
        belowRight = (col < words_per_row - 1) ? input[baseIndex + col + 1] : 0ULL;
    }
    
    // For each of the three rows (above, current, below) we consider appropriate horizontal neighbors:
    // For a given row, the neighbor for a cell at bit position i is:
    //   - For left neighbor: if i > 0, use bit (i-1) from the center word; if i == 0, use bit 63 from the left word.
    //   - For right neighbor: if i < 63, use bit (i+1) from the center word; if i == 63, use bit 0 from the right word.
    // For the above and below rows, we also include the directly aligned cell (no horizontal shift) as a neighbor.
    //
    // Compute neighbor masks for the above row.
    std::uint64_t above_left_mask  = (above << 1)  | ((col > 0) ? ((aboveLeft >> 63) & 1ULL) : 0ULL);
    std::uint64_t above_center_mask= above;
    std::uint64_t above_right_mask = (above >> 1)  | ((col < words_per_row - 1) ? ((aboveRight & 1ULL) << 63) : 0ULL);
    
    // For the current row, we exclude the center cell itself.
    std::uint64_t current_left_mask = (curWord << 1) | ((col > 0) ? ((curLeft >> 63) & 1ULL) : 0ULL);
    std::uint64_t current_right_mask= (curWord >> 1) | ((col < words_per_row - 1) ? ((curRight & 1ULL) << 63) : 0ULL);
    
    // For the below row.
    std::uint64_t below_left_mask  = (below << 1)  | ((col > 0) ? ((belowLeft >> 63) & 1ULL) : 0ULL);
    std::uint64_t below_center_mask= below;
    std::uint64_t below_right_mask = (below >> 1)  | ((col < words_per_row - 1) ? ((belowRight & 1ULL) << 63) : 0ULL);
    
    // The eight neighbor contributions for each cell (excluding the center cell)
    // are coming from: above-left, above-center, above-right,
    //                       current-left,         current-right,
    //                      below-left, below-center, below-right.
    const int NUM_MASKS = 8;
    std::uint64_t masks[NUM_MASKS] = { above_left_mask, above_center_mask, above_right_mask,
                                       current_left_mask, current_right_mask,
                                       below_left_mask, below_center_mask, below_right_mask };
    
    // We split the 64 bits (cells) into 4 groups of 16 cells.
    // For each neighbor mask, we extract its 16‐bit chunks and “expand” them into lane‐wise 4‐bit numbers.
    uint64_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    #pragma unroll
    for (int m = 0; m < NUM_MASKS; m++) {
        uint64_t mask = masks[m];
        uint16_t group0 = (uint16_t)(mask & 0xFFFFULL);
        uint16_t group1 = (uint16_t)((mask >> 16) & 0xFFFFULL);
        uint16_t group2 = (uint16_t)((mask >> 32) & 0xFFFFULL);
        uint16_t group3 = (uint16_t)((mask >> 48) & 0xFFFFULL);
        sum0 += expand16(group0);
        sum1 += expand16(group1);
        sum2 += expand16(group2);
        sum3 += expand16(group3);
    }
    
    // Now, for each of the 64 cells (organized as 4 groups of 16),
    // the neighbor count for each cell is stored in a 4‐bit lane within sum0, sum1, sum2, sum3.
    // We now apply the Game of Life rules:
    //   A cell becomes alive if:
    //      (neighbor count == 3) OR ((neighbor count == 2) and (cell is already alive)).
    //
    // For each 16‐cell group, we extract the corresponding 16 bits of the current cell state,
    // loop over each cell in the group and compute its new state.
    uint16_t res0 = 0, res1 = 0, res2 = 0, res3 = 0;
    
    // Group 0: cells 0..15
    uint16_t curGroup0 = (uint16_t)(curWord & 0xFFFFULL);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Extract neighbor count from the i-th nibble.
        uint8_t count = (uint8_t)((sum0 >> (i * 4)) & 0xF);
        // Extract current cell state (0 or 1).
        uint8_t cell = (curGroup0 >> i) & 1;
        uint8_t newState = (count == 3) || ((count == 2) && (cell != 0));
        res0 |= (newState << i);
    }
    
    // Group 1: cells 16..31
    uint16_t curGroup1 = (uint16_t)((curWord >> 16) & 0xFFFFULL);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t count = (uint8_t)((sum1 >> (i * 4)) & 0xF);
        uint8_t cell = (curGroup1 >> i) & 1;
        uint8_t newState = (count == 3) || ((count == 2) && (cell != 0));
        res1 |= (newState << i);
    }
    
    // Group 2: cells 32..47
    uint16_t curGroup2 = (uint16_t)((curWord >> 32) & 0xFFFFULL);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t count = (uint8_t)((sum2 >> (i * 4)) & 0xF);
        uint8_t cell = (curGroup2 >> i) & 1;
        uint8_t newState = (count == 3) || ((count == 2) && (cell != 0));
        res2 |= (newState << i);
    }
    
    // Group 3: cells 48..63
    uint16_t curGroup3 = (uint16_t)((curWord >> 48) & 0xFFFFULL);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t count = (uint8_t)((sum3 >> (i * 4)) & 0xF);
        uint8_t cell = (curGroup3 >> i) & 1;
        uint8_t newState = (count == 3) || ((count == 2) && (cell != 0));
        res3 |= (newState << i);
    }
    
    // Reassemble the four 16-bit groups into one 64-bit word and store the result.
    uint64_t result_word = ((uint64_t)res3 << 48) | ((uint64_t)res2 << 32) |
                           ((uint64_t)res1 << 16) | ((uint64_t)res0);
    output[row * words_per_row + col] = result_word;
}

//---------------------------------------------------------------------
// Host function to run one step of Conway's Game of Life on the GPU.
// 'input' and 'output' are pointers to bit‐packed grids allocated with cudaMalloc.
// Each std::uint64_t packs 64 consecutive cells (1 bit per cell).
// 'grid_dimensions' is the width/height of the square grid.
// All necessary host–device synchronization is assumed to be handled by the caller.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Compute the number of 64‐bit words per row.
    int words_per_row = grid_dimensions / 64;
    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;
    
    // Choose a block size (e.g., 256 threads per block) and compute grid dimensions.
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;
    
    // Launch the kernel.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions, words_per_row);
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
