#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// This CUDA implementation of Conway's Game of Life works on a 
// bit‐packed grid where each std::uint64_t holds 64 consecutive cells.
// Each cell is represented by one bit – 1 for live, 0 for dead.
// 
// For proper neighbor addressing in a bit‐packed grid, we align the 8
// neighbor contributions via bit‐shifts. For example, the left neighbor of 
// a cell at bit i (0 ≤ i ≤ 63) is in the same word at bit i–1 when i>0, and if 
// i==0 it comes from bit 63 of the left adjacent word. Similar logic is applied
// for the right neighbor and for diagonal neighbors in adjacent rows.
// 
// To compute the neighbor count for 64 cells simultaneously without loop‐ing 
// over each bit, we use a 3‐bit binary counter per cell (bits: cnt2 cnt1 cnt0) 
// that we update using bitwise ripple–carry addition. That is, each neighbor 
// contribution is a 64-bit word where each bit is 0 or 1; we “add” these words 
// into our per–cell 3–bit counter.  Note that the maximum neighbor count is 8. 
// Our counter works correctly for sums 0–7; for 8 the counter wraps to 0. This 
// is acceptable because a cell with 8 neighbors dies (does not match any rule).
//
// Finally, the Game of Life rule is applied:
//   • Live cell with exactly 2 live neighbors stays alive (2 -> 010).
//   • Live cell with 3 live neighbors stays alive; dead cell becomes live if exactly 3 neighbors (3 -> 011).
//   • Otherwise the cell becomes/stays dead.
// Thus, next state = (neighbor_count == 3) OR (current_state AND neighbor_count == 2).
//
// Each CUDA thread handles one 64-bit word (which encodes 64 cells). The grid 
// of threads is arranged in 2D such that there are (grid_dimensions) rows and 
// (grid_dimensions/64) columns of words.
// ---------------------------------------------------------------------------


// __device__ inline function to add a 1–bit neighbor word (X) into a 3–bit per–cell counter.
// The counter is stored in three 64–bit variables (cnt2, cnt1, cnt0) where each bit position 
// (lane) holds the 3–bit value: value = cnt0 + 2*cnt1 + 4*cnt2. This addition is done with 
// ripple–carry, and any overflow (beyond 7) is discarded – acceptable for our Game of Life.
// Each bit in X is either 0 or 1.
__device__ inline void add_bit(uint64_t &cnt0, uint64_t &cnt1, uint64_t &cnt2, uint64_t X) {
    uint64_t new0   = cnt0 ^ X;            // add bit X to the 0th bit (LSB)
    uint64_t carry0 = cnt0 & X;            // compute carry from bit0 addition
    uint64_t new1   = cnt1 ^ carry0;       // add carry0 to bit1
    uint64_t carry1 = cnt1 & carry0;        // compute carry from bit1 addition
    uint64_t new2   = cnt2 ^ carry1;       // add carry1 to bit2
    // (Any carry from bit2 producing a 4th bit is ignored.)
    cnt0 = new0;
    cnt1 = new1;
    cnt2 = new2;
}

// ---------------------------------------------------------------------------
// The kernel processes one generation of the Game of Life. Each thread handles one
// word (64 cells) identified by row index r and word column index c.
// 
// To compute a cell's 8–neighbor sum, we fetch the corresponding neighbor words from 
// the row above (r-1), the current row (r), and the row below (r+1). For each row, we 
// need three contributions: left, center, and right. However, because the grid is stored 
// in 64–bit words, special care is taken at the 0th and 63rd bit positions to pick the 
// correct neighbor from an adjacent word.
// 
// We construct each neighbor's contribution as follows (for current thread at (r,c)):
//   For the current row:
//     left = (current word << 1) OR (if c > 0, the MSB of word (r, c-1))
//     right = (current word >> 1) OR (if c < last, the LSB of word (r, c+1))
//   For row above and below, similar shifts are applied.
// All shifted words become aligned with the bits of the current word, so that for each bit 
// (cell) in the current word, its neighbor bits are in the corresponding bit positions.
// ---------------------------------------------------------------------------
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dim, int words_per_row) {
    // Compute row index and word (column) index for this thread.
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= grid_dim || c >= words_per_row) return;
    
    // Helper lambda to get a word from input if in bounds; otherwise return 0.
    auto get_word = [=] __device__ (int row, int col) -> uint64_t {
        if (row < 0 || row >= grid_dim || col < 0 || col >= words_per_row) {
            return 0ULL;
        }
        return input[row * words_per_row + col];
    };
    
    // Load current cell word.
    uint64_t curr = get_word(r, c);
    
    // Fetch neighbor words from the same row.
    uint64_t left_curr  = get_word(r, c - 1);
    uint64_t right_curr = get_word(r, c + 1);
    
    // For the row above.
    uint64_t center_up  = get_word(r - 1, c);
    uint64_t left_up    = get_word(r - 1, c - 1);
    uint64_t right_up   = get_word(r - 1, c + 1);
    
    // For the row below.
    uint64_t center_down = get_word(r + 1, c);
    uint64_t left_down   = get_word(r + 1, c - 1);
    uint64_t right_down  = get_word(r + 1, c + 1);
    
    // Construct shifted neighbor contributions, aligning them with the current word's bits.
    // For current row:
    //   The left neighbor for a cell is:
    //      if bit index > 0: (curr << 1) provides cell from i-1.
    //      if bit index = 0: use MSB of left_curr (if available).
    uint64_t left = (curr << 1) | ((c > 0) ? (left_curr >> 63) : 0ULL);
    // Right neighbor for current row:
    //      if bit index < 63: (curr >> 1) provides cell from i+1.
    //      if bit index = 63: use LSB of right_curr.
    uint64_t right = (curr >> 1) | ((c < words_per_row - 1) ? ((right_curr & 1ULL) << 63) : 0ULL);
    
    // For row above:
    uint64_t up = center_up;
    uint64_t up_left = (center_up << 1) | ((c > 0) ? (left_up >> 63) : 0ULL);
    uint64_t up_right = (center_up >> 1) | ((c < words_per_row - 1) ? ((right_up & 1ULL) << 63) : 0ULL);
    
    // For row below:
    uint64_t down = center_down;
    uint64_t down_left = (center_down << 1) | ((c > 0) ? (left_down >> 63) : 0ULL);
    uint64_t down_right = (center_down >> 1) | ((c < words_per_row - 1) ? ((right_down & 1ULL) << 63) : 0ULL);
    
    // Accumulate the 8 neighbor contributions using our 3-bit per–cell counter.
    // Counters: cnt0 (LSB), cnt1 (2's place), cnt2 (4's place).
    uint64_t cnt0 = 0ULL, cnt1 = 0ULL, cnt2 = 0ULL;
    
    add_bit(cnt0, cnt1, cnt2, up);
    add_bit(cnt0, cnt1, cnt2, up_left);
    add_bit(cnt0, cnt1, cnt2, up_right);
    add_bit(cnt0, cnt1, cnt2, left);
    add_bit(cnt0, cnt1, cnt2, right);
    add_bit(cnt0, cnt1, cnt2, down);
    add_bit(cnt0, cnt1, cnt2, down_left);
    add_bit(cnt0, cnt1, cnt2, down_right);
    
    // At this point, for each bit position (cell), the neighbor count is encoded in binary as:
    //   value = cnt0 + 2*cnt1 + 4*cnt2   (ranging correctly 0..7; note: 8 wraps to 0)
    //
    // The Game of Life rule requires:
    //   - A cell becomes (or remains) live if it has exactly 3 neighbors, or if it is currently live and has exactly 2 neighbors.
    // In binary, 2 = 010 and 3 = 011.
    //
    // Compute masks for positions where neighbor count equals 2 and equals 3.
    uint64_t eq2_mask = ((~cnt2) & cnt1 & (~cnt0));  // bits equal to 010
    uint64_t eq3_mask = ((~cnt2) & cnt1 & cnt0);       // bits equal to 011
    
    // The next state for each cell is:
    //   next = eq3_mask  OR  (curr & eq2_mask)
    uint64_t next = eq3_mask | (curr & eq2_mask);
    
    // Write the next state word to the output grid.
    output[r * words_per_row + c] = next;
}

// ---------------------------------------------------------------------------
// Host function: run_game_of_life
//
// Executes one step (one generation) of Conway's Game of Life on a bit‐packed grid.
//   • input: pointer to device memory input grid (std::uint64_t*, bit-packed)
//   • output: pointer to device memory for the output grid (same format)
//   • grid_dimensions: width and height (in cells) of the grid (a power of 2, >512)
// 
// The grid is stored as rows of bit-packed cells. Each row contains (grid_dimensions/64) 
// std::uint64_t words. Any necessary data transformations are done herein. Host-device 
// synchronizations are assumed to be performed by the caller.
// ---------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Compute the number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;
    
    // Define 2D block and grid dimensions.
    // We choose a block size of 16x16 threads.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions, words_per_row);
    // Note: Any host-device synchronization is assumed to be handled by the caller.
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
