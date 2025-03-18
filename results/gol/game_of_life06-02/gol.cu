#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one step of Conway’s Game of Life on a bit‐packed grid.
// The grid is square with grid_dimensions rows and grid_dimensions columns.
// Each row is stored as grid_dimensions/64 64‐bit unsigned integers.
// In each 64‐bit word, bit0 represents the leftmost cell and bit63 the rightmost cell.
// Each thread processes one 64‐bit word (i.e. one contiguous block of 64 cells in a row).
//
// For every cell, the eight neighbors (upper-left, up, upper‐right, left, right, lower‐left, down, lower‐right)
// are summed using “bit‐sliced” full adder logic. Because the grid is bit–packed, we must properly align
// contributions coming from adjacent words (handling the 0th and 63rd bit boundaries).
//
// The next generation is computed using the standard rules:
//  • A cell is born if it has exactly 3 live neighbors.
//  • A live cell survives if it has exactly 2 or 3 live neighbors; otherwise it dies.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // Each row has a number of 64-bit words equal to grid_dimensions / 64.
    int words_per_row = grid_dimensions / 64;
    
    // Compute the row index (cell row) and column index (word index within the row).
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || col >= words_per_row)
        return;
    
    // Compute the linear index for the current word.
    int idx = row * words_per_row + col;
    
    // Load the current word from the input grid.
    std::uint64_t cur = input[idx];
    
    // For same-row neighbor contributions, load the adjacent words.
    std::uint64_t word_left  = (col > 0) ? input[row * words_per_row + col - 1] : 0;
    std::uint64_t word_right = (col < words_per_row - 1) ? input[row * words_per_row + col + 1] : 0;
    
    // For the row above.
    std::uint64_t up = 0, up_left = 0, up_right = 0;
    if (row > 0) {
        int row_above = row - 1;
        up       = input[row_above * words_per_row + col];
        up_left  = (col > 0) ? input[row_above * words_per_row + col - 1] : 0;
        up_right = (col < words_per_row - 1) ? input[row_above * words_per_row + col + 1] : 0;
    }
    
    // For the row below.
    std::uint64_t down = 0, down_left = 0, down_right = 0;
    if (row < grid_dimensions - 1) {
        int row_below = row + 1;
        down       = input[row_below * words_per_row + col];
        down_left  = (col > 0) ? input[row_below * words_per_row + col - 1] : 0;
        down_right = (col < words_per_row - 1) ? input[row_below * words_per_row + col + 1] : 0;
    }
    
    //-------------------------------------------------------------------------
    // Compute aligned neighbor contributions for the 8 directions.
    // We must align the bits from adjacent words to the current word’s cell positions.
    // The convention: In a 64-bit word, bit0 is the leftmost cell and bit63 is the rightmost cell.
    // Shifting right (>>) moves bits toward lower-index positions (i.e. from index i to i-1),
    // and shifting left (<<) moves bits toward higher-index positions.
    //
    // For a neighbor that is diagonally offset horizontally, we combine a shifted version
    // of the neighbor word from the same row and, for the edge cell of the word, a bit from the adjacent word.
    //-------------------------------------------------------------------------
    
    // 1. Neighbors from the row above.
    // Directly above: no horizontal shift.
    std::uint64_t n_up = up;
    // Upper-left: For cells with index >0, use (up >> 1); for the leftmost cell (bit0), use bit63 of up_left.
    //   - (up >> 1) shifts bits so that for cell i (i>=1) the bit comes from column i-1.
    //   - We clear bit0 from (up >> 1) and then set it from up_left if available.
    std::uint64_t up_left_bulk = (up >> 1) & ~1ULL; 
    std::uint64_t up_left_edge = (up_left & (1ULL << 63)) ? 1ULL : 0ULL;
    std::uint64_t n_up_left = up_left_bulk | up_left_edge;
    
    // Upper-right: For cells with index <63, use (up << 1); for the rightmost cell (bit63), use bit0 of up_right.
    std::uint64_t up_right_bulk = (up << 1) & 0x7FFFFFFFFFFFFFFFULL; // mask to clear bit63
    std::uint64_t up_right_edge = (up_right & 1ULL) ? (1ULL << 63) : 0ULL;
    std::uint64_t n_up_right = up_right_bulk | up_right_edge;
    
    // 2. Neighbors from the same row.
    // Left: For cells with index >0, use (cur >> 1); for bit0, use bit63 from word_left.
    std::uint64_t left_bulk = (cur >> 1) & ~1ULL;
    std::uint64_t left_edge = (word_left & (1ULL << 63)) ? 1ULL : 0ULL;
    std::uint64_t n_left = left_bulk | left_edge;
    
    // Right: For cells with index <63, use (cur << 1); for bit63, use bit0 from word_right.
    std::uint64_t right_bulk = (cur << 1) & 0x7FFFFFFFFFFFFFFFULL;
    std::uint64_t right_edge = (word_right & 1ULL) ? (1ULL << 63) : 0ULL;
    std::uint64_t n_right = right_bulk | right_edge;
    
    // 3. Neighbors from the row below.
    // Directly below: no horizontal shift.
    std::uint64_t n_down = down;
    // Down-left: For cells with index >0, use (down >> 1); for bit0, use bit63 from down_left.
    std::uint64_t down_left_bulk = (down >> 1) & ~1ULL;
    std::uint64_t down_left_edge = (down_left & (1ULL << 63)) ? 1ULL : 0ULL;
    std::uint64_t n_down_left = down_left_bulk | down_left_edge;
    
    // Down-right: For cells with index <63, use (down << 1); for bit63, use bit0 from down_right.
    std::uint64_t down_right_bulk = (down << 1) & 0x7FFFFFFFFFFFFFFFULL;
    std::uint64_t down_right_edge = (down_right & 1ULL) ? (1ULL << 63) : 0ULL;
    std::uint64_t n_down_right = down_right_bulk | down_right_edge;
    
    //--------------------------------------------------------------------------
    // We now have 8 bit masks, each representing the contribution (0 or 1) to the neighbor
    // count for each cell in the current word from one of the 8 directions:
    //    n_up_left, n_up, n_up_right, n_left, n_right, n_down_left, n_down, n_down_right.
    //
    // To compute the per-cell neighbor count (a number between 0 and 8) in a bit–parallel fashion,
    // we use "bit-sliced" addition.  For each of the 64 cell-positions (lanes) within the word,
    // we add 8 one-bit values.  We represent each lane’s 4–bit counter using 4 bit–planes
    // (sum0, sum1, sum2, sum3) stored in separate 64–bit variables.
    //--------------------------------------------------------------------------
    
    // Initialize bit-sliced counters to 0.
    std::uint64_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    
    // Helper macro to add a one-bit mask (X) to the bit-sliced counter.
    // This implements a full-adder that adds the single-bit contribution X to each lane.
    #define ADD_BIT_MASK(X)  {                 \
        std::uint64_t carry = (X);              \
        std::uint64_t s = sum0 ^ carry;         \
        std::uint64_t c = sum0 & carry;         \
        sum0 = s;                             \
        carry = c;                            \
        s = sum1 ^ carry;                     \
        c = sum1 & carry;                     \
        sum1 = s;                             \
        carry = c;                            \
        s = sum2 ^ carry;                     \
        c = sum2 & carry;                     \
        sum2 = s;                             \
        carry = c;                            \
        s = sum3 ^ carry;                     \
        c = sum3 & carry;                     \
        sum3 = s;                             \
        /* No propagation beyond 4 bits (max count=8) */ \
    }
    
    // Accumulate the 8 neighbor contributions.
    ADD_BIT_MASK(n_up_left);
    ADD_BIT_MASK(n_up);
    ADD_BIT_MASK(n_up_right);
    ADD_BIT_MASK(n_left);
    ADD_BIT_MASK(n_right);
    ADD_BIT_MASK(n_down_left);
    ADD_BIT_MASK(n_down);
    ADD_BIT_MASK(n_down_right);
    
    #undef ADD_BIT_MASK
    
    //--------------------------------------------------------------------------
    // Now for each cell (each bit position in the 64-bit word), the neighbor count
    // is encoded in 4-bit binary with sum3 as the most-significant and sum0 as the least-significant bit.
    //
    // Conway's Game of Life rules require that:
    //   - A cell becomes alive if it has exactly 3 live neighbors.
    //   - A live cell survives if it has exactly 2 live neighbors (in addition to the 3-neighbor reproduction).
    //
    // We form bit masks for positions where the neighbor count equals 3 (eq3) and equals 2 (eq2).
    // The binary encoding for 3 is 0011 and for 2 is 0010.
    //--------------------------------------------------------------------------
    std::uint64_t eq3 = ((~sum3) & (~sum2) & sum1 & sum0);   // equals 3: 0,0,1,1
    std::uint64_t eq2 = ((~sum3) & (~sum2) & sum1 & (~sum0));  // equals 2: 0,0,1,0
    
    // Apply the rules:
    //   - A cell will be alive in the next generation if it has exactly 3 live neighbors,
    //     or if it is currently alive and has exactly 2 live neighbors.
    std::uint64_t next = eq3 | (cur & eq2);
    
    // Write the updated word into the output grid.
    output[idx] = next;
}

// Host-side function that launches one Game of Life step on the GPU.
// The 'input' and 'output' pointers refer to device (GPU) memory holding bit-packed grid data.
// The grid is square with grid_dimensions rows and columns; each row is encoded as grid_dimensions/64 64–bit words.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;
    
    // Choose a thread block size. The values below are chosen to provide good occupancy on modern GPUs.
    dim3 blockDim(16, 16);
    // Grid dimensions: one thread per 64-bit word.
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
