#include <cuda_runtime.h>
#include <cstdint>

//------------------------------------------------------------------------------
// This CUDA implementation of Conway's Game of Life uses a bit‐packed
// representation for the grid: each std::uint64_t word encodes 64 consecutive
// cells in a row (bit 0 = dead, 1 = alive). Each CUDA thread computes one such
// word (i.e. 64 cells) in the new generation.  The rules are:
//
//   - Any live cell with fewer than 2 live neighbours dies.
//   - Any live cell with 2 or 3 live neighbours lives on.
//   - Any live cell with more than 3 live neighbours dies.
//   - Any dead cell with exactly 3 live neighbours becomes a live cell.
//
// The kernel computes the neighbor count for each cell “in parallel” by
// collecting 8 neighbor contributions (from the three rows: above, same, below)
// – taking into account that a neighbor may reside in an adjacent word if the
// cell is at bit index 0 or 63.  To avoid costly per‐bit loops we “vectorize” the
// per‐cell addition using bit‐sliced arithmetic: a 4‐bit number for each cell is
// stored in four 64–bit variables (each bit position in these variables holds
// one bit of the 4–bit count for one cell).  After adding eight 1–bit contributions
// (each corresponding to one neighbor, already aligned into the correct bit‐lane)
// we check: a cell becomes alive if its neighbor count is exactly 3, or if
// (neighbor count is exactly 2 and the cell itself was alive).
//
// The function add_bit() below adds a 1–bit value (per cell, stored in a 64–bit word)
// to a lane–wise 4–bit accumulator represented by (sum0, sum1, sum2, sum3).
//
// Boundary cells are handled by treating cells outside the grid as dead (0).
// This kernel is tuned for modern NVIDIA GPUs (A100/H100) and uses no shared or
// texture memory.
//------------------------------------------------------------------------------

__device__ inline void add_bit(uint64_t &sum0, uint64_t &sum1, uint64_t &sum2, uint64_t &sum3, uint64_t d)
{
    // The accumulator (sum3, sum2, sum1, sum0) holds a 4–bit number per cell;
    // d contains a 1–bit operand per cell. We perform a ripple–carry style addition,
    // but note that each bit (i.e. each cell) is processed independently because the bit
    // for cell j in each variable is isolated.
    uint64_t carry;
    // Add to LSB (bit0)
    carry = sum0 & d;         // carry bits where both are 1
    sum0 ^= d;                // new LSB
    d = carry;                // propagate carry to next bit

    // Add to bit1 field
    carry = sum1 & d;
    sum1 ^= d;
    d = carry;

    // Add to bit2 field
    carry = sum2 & d;
    sum2 ^= d;
    d = carry;

    // Add to bit3 field
    carry = sum3 & d;
    sum3 ^= d;
    // Any further carry is discarded (max sum is 8, so four bits suffice)
}

__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dim, int nwords)
{
    // Compute grid coordinates:
    // Each thread computes one uint64_t word corresponding to 64 cells.
    int word_x = blockIdx.x * blockDim.x + threadIdx.x; // column index in word units
    int row = blockIdx.y * blockDim.y + threadIdx.y;      // row index (cell row)

    if (row >= grid_dim || word_x >= nwords)
        return;

    // Compute index into bit-packed grid array
    int idx = row * nwords + word_x;

    // Define lambda to safely load a word from (r, w),
    // returning 0 if out-of-bound.
/// @FIXED: auto load_word = [=, input] (int r, int w) -> uint64_t {
    auto load_word = [=] (int r, int w) -> uint64_t {
        if(r < 0 || r >= grid_dim || w < 0 || w >= nwords)
            return 0ULL;
        return input[r * nwords + w];
    };

    // Load necessary words for this cell.
    // For rows: r-1 (top), r (mid), r+1 (bottom).
    uint64_t top_center   = load_word(row - 1, word_x);
    uint64_t mid_center   = load_word(row, word_x);  // current cell state (center of mid row)
    uint64_t bot_center   = load_word(row + 1, word_x);

    // For horizontal neighbors in the same row, use adjacent words.
    uint64_t mid_left     = load_word(row, word_x - 1);
    uint64_t mid_right    = load_word(row, word_x + 1);

    // For top row horizontal neighbors.
    uint64_t top_left     = load_word(row - 1, word_x - 1);
    uint64_t top_right    = load_word(row - 1, word_x + 1);

    // For bottom row horizontal neighbors.
    uint64_t bot_left     = load_word(row + 1, word_x - 1);
    uint64_t bot_right    = load_word(row + 1, word_x + 1);

    //-----------------------------------------------------------------------------
    // For each neighboring cell, we now align its bit into the proper "lane" for the
    // central word.  For a given cell in the output word at bit position i (0 <= i < 64),
    // its neighbors come from various sources:
    //
    // Top row neighbors (row-1):
    //   - Top-left neighbor: if i > 0 then from top_center word bit (i-1),
    //       otherwise (if i==0) use the rightmost bit (bit 63) of top_left.
    //       => top_nw = (top_center << 1) OR (top_left >> 63)
    //   - Top-center neighbor: from top_center word bit i.
    //       => top_n = top_center.
    //   - Top-right neighbor: if i < 63 then from top_center word bit (i+1),
    //       otherwise (if i==63) use the leftmost bit (bit 0) of top_right.
    //       => top_ne = (top_center >> 1) OR (top_right << 63)
    //
    // Middle row neighbors (row):
    //   (Exclude the center cell.)
    //   - West neighbor:   mid_w = (mid_center << 1) OR (mid_left >> 63)
    //   - East neighbor:   mid_e = (mid_center >> 1) OR (mid_right << 63)
    //
    // Bottom row neighbors (row+1):
    //   - Bottom-left neighbor: bot_nw = (bot_center << 1) OR (bot_left >> 63)
    //   - Bottom-center neighbor: bot_n = bot_center.
    //   - Bottom-right neighbor: bot_ne = (bot_center >> 1) OR (bot_right << 63)
    //-----------------------------------------------------------------------------
    uint64_t top_nw = (top_center << 1) | ( (word_x > 0) ? (top_left >> 63) : 0ULL );
    uint64_t top_n  = top_center;
    uint64_t top_ne = (top_center >> 1) | ( (word_x < nwords - 1) ? (top_right << 63) : 0ULL );

    uint64_t mid_w  = (mid_center << 1) | ( (word_x > 0) ? (mid_left >> 63) : 0ULL );
    uint64_t mid_e  = (mid_center >> 1) | ( (word_x < nwords - 1) ? (mid_right << 63) : 0ULL );

    uint64_t bot_nw = (bot_center << 1) | ( (word_x > 0) ? (bot_left >> 63) : 0ULL );
    uint64_t bot_n  = bot_center;
    uint64_t bot_ne = (bot_center >> 1) | ( (word_x < nwords - 1) ? (bot_right << 63) : 0ULL );

    // Our 8 operands (each a 64-bit value whose j-th bit is 0 or 1)
    // corresponding to the 8 neighbors of each cell.
    uint64_t n0 = top_nw;
    uint64_t n1 = top_n;
    uint64_t n2 = top_ne;
    uint64_t n3 = mid_w;
    uint64_t n4 = mid_e;
    uint64_t n5 = bot_nw;
    uint64_t n6 = bot_n;
    uint64_t n7 = bot_ne;

    //------------------------------------------------------------------------------
    // Now we add (lane–wise) the eight neighbor contributions.
    // For each of the 64 cell lanes, we want to compute an integer in [0,8]
    // equal to the sum n0+n1+...+n7 (each n_i is 0 or 1).
    // We represent the 4–bit result for each cell in four 64–bit words:
    //    count = (count3, count2, count1, count0)  where count0 is the LSB.
    // We initialize the 4–bit accumulator to zero.
    // Then add each neighbor's bit (using the function add_bit() defined above).
    //------------------------------------------------------------------------------
    uint64_t count0 = 0, count1 = 0, count2 = 0, count3 = 0;

    add_bit(count0, count1, count2, count3, n0);
    add_bit(count0, count1, count2, count3, n1);
    add_bit(count0, count1, count2, count3, n2);
    add_bit(count0, count1, count2, count3, n3);
    add_bit(count0, count1, count2, count3, n4);
    add_bit(count0, count1, count2, count3, n5);
    add_bit(count0, count1, count2, count3, n6);
    add_bit(count0, count1, count2, count3, n7);

    //------------------------------------------------------------------------------
    // The output state for each cell is determined by the classic Game of Life rules:
    // A cell becomes/ stays alive if:
    //    (exactly 3 live neighbours) OR (cell was alive and exactly 2 live neighbours)
    // For each lane (cell) we check if the 4–bit neighbor count equals 2 or 3.
    //   2 in binary is 0010; 3 is 0011.
    // Because count3 and count2 are the two most–significant bits, we can compute:
    //    eq2 = (not count3) AND (not count2) AND count1 AND (not count0)
    //    eq3 = (not count3) AND (not count2) AND count1 AND count0
    //------------------------------------------------------------------------------
    uint64_t eq2 = (~count3) & (~count2) & count1 & (~count0);
    uint64_t eq3 = (~count3) & (~count2) & count1 & count0;

    // Next-state: cells with exactly 3 neighbours, or those that are currently alive with exactly 2.
    uint64_t next_state = eq3 | ( mid_center & eq2 );

    // Write the computed 64 new cells (packed in one word) to the output grid.
    output[idx] = next_state;
}

//------------------------------------------------------------------------------
// Host function: run_game_of_life
//
// This function launches one step of Conway's Game of Life on the GPU.
//   - input: pointer to device memory holding the current generation (bit-packed)
//   - output: pointer to device memory for the next generation (bit-packed)
//   - grid_dimensions: width (and height) of the square grid. Guaranteed to be a
//     power of two, and > 512.
//------------------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of 64-bit words per row.
    int nwords = grid_dimensions >> 6; // equivalent to grid_dimensions / 64

    // Choose a 2D block and grid configuration.
    // Each thread computes one 64–cell word.
    dim3 block(16, 16);
    dim3 grid((nwords + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions, nwords);
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
