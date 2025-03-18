#include <cstdint>
#include <cuda_runtime.h>

// This device‐inline function performs a “ripple‐carry” addition in a bit‐sliced manner.
// Each of the four 64‐bit variables (s0, s1, s2, s3) represent one “digit” of a 4‐bit number
// for 64 independent lanes (one lane per bit position). When adding a neighbor mask (which is a 
// binary number, 0 or 1 for each cell) the addition is performed independently on each lane.
__device__ inline void add_bit(uint64_t m, uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3) {
    // Add m (which has a 1 where the neighbor is alive) to the existing 4‐bit value stored bitwise.
    uint64_t t = s0 ^ m;       // Sum without carry (low “digit”)
    uint64_t c = s0 & m;       // Carry: where both had a 1.
    s0 = t;
    t = s1 ^ c;
    c = s1 & c;
    s1 = t;
    t = s2 ^ c;
    c = s2 & c;
    s2 = t;
    s3 ^= c;  // Maximum neighbor count is 8, so we only need 4 bits; any final carry is absorbed here.
}

// Kernel for one generation update of Conway's Game of Life.
// The grid is stored as a bit‐packed array of std::uint64_t, where each 64‐bit word encodes 64 cells.
// Bits are arranged such that bit 0 (the least‐significant bit) represents the leftmost cell in the word,
// and bit 63 represents the rightmost cell. Consequently, the 0th bit requires special handling (neighbors
// from the word to the left) and the 63rd bit requires special handling (neighbors from the right word).
// Each thread computes the next state for one 64‐cell word.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dim) {
    // grid_dim is the number of cells per row (and the number of rows, since the grid is square).
    // Each row is stored as grid_dim/64 words.
    int words_per_row = grid_dim / 64;

    // Determine the current word's coordinates.
    // word_col indexes the bit‐packed word within the row.
    int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= grid_dim || word_col >= words_per_row)
        return;
    
    // Compute the index into the 1D bit‐packed grid.
    int index = row * words_per_row + word_col;
    
    // Load the “center” word (current row, current word).
    uint64_t center = input[index];
    
    // Load adjacent words from the current row as needed.
    uint64_t left   = (word_col > 0) ? input[row * words_per_row + (word_col - 1)] : 0ULL;
    uint64_t right  = (word_col < words_per_row - 1) ? input[row * words_per_row + (word_col + 1)] : 0ULL;
    
    // For the row above (north) and below (south), load the center and adjacent words if available.
    uint64_t n_center = 0, n_left = 0, n_right = 0;
    uint64_t s_center = 0, s_left = 0, s_right = 0;
    if (row > 0) {
        int n_index = (row - 1) * words_per_row + word_col;
        n_center = input[n_index];
        n_left   = (word_col > 0) ? input[(row - 1) * words_per_row + (word_col - 1)] : 0ULL;
        n_right  = (word_col < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_col + 1)] : 0ULL;
    }
    if (row < grid_dim - 1) {
        int s_index = (row + 1) * words_per_row + word_col;
        s_center = input[s_index];
        s_left   = (word_col > 0) ? input[(row + 1) * words_per_row + (word_col - 1)] : 0ULL;
        s_right  = (word_col < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_col + 1)] : 0ULL;
    }
    
    // With our chosen representation:
    // - bit 0 (LSB) is the leftmost cell in the word; bit 63 is the rightmost.
    // Therefore, to compute horizontal neighbor contributions:
    // West (left) neighbor for a cell at bit position i comes from bit (i-1) if i > 0.
    //   For intra-word neighbors, shifting left by 1 aligns bit (i-1) into bit i.
    //   The leftmost cell (bit 0) must get its west neighbor from the previous word’s rightmost cell (bit 63).
    // East (right) neighbor for a cell at bit position i comes from bit (i+1) if i < 63.
    //   For intra-word neighbors, shifting right by 1 aligns bit (i+1) into bit i.
    //   The rightmost cell (bit 63) gets its east neighbor from the next word’s leftmost cell (bit 0).
    
    // West (W) mask.
    uint64_t W_intra = center << 1;  // For bits 1..63, this yields the cell to the left.
    uint64_t W_extra = (word_col > 0) ? ((left >> 63) & 1ULL) : 0ULL; // Left word's rightmost cell becomes the west neighbor for bit 0.
    uint64_t W = W_intra | W_extra;
    
    // East (E) mask.
    uint64_t E_intra = center >> 1;  // For bits 0..62.
    uint64_t E_extra = (word_col < words_per_row - 1) ? ((right & 1ULL) << 63) : 0ULL;  // Right word's leftmost cell becomes the east neighbor for bit 63.
    uint64_t E = E_intra | E_extra;
    
    // North (N) mask: from the row above, same word (no horizontal shift).
    uint64_t N = n_center;
    
    // South (S) mask: from the row below.
    uint64_t S = s_center;
    
    // North-West (NW): from the row above, similar to W.
    uint64_t NW_intra = n_center << 1;
    uint64_t NW_extra = (word_col > 0) ? ((n_left >> 63) & 1ULL) : 0ULL;
    uint64_t NW = NW_intra | NW_extra;
    
    // North-East (NE): from the row above, similar to E.
    uint64_t NE_intra = n_center >> 1;
    uint64_t NE_extra = (word_col < words_per_row - 1) ? ((n_right & 1ULL) << 63) : 0ULL;
    uint64_t NE = NE_intra | NE_extra;
    
    // South-West (SW): from the row below.
    uint64_t SW_intra = s_center << 1;
    uint64_t SW_extra = (word_col > 0) ? ((s_left >> 63) & 1ULL) : 0ULL;
    uint64_t SW = SW_intra | SW_extra;
    
    // South-East (SE): from the row below.
    uint64_t SE_intra = s_center >> 1;
    uint64_t SE_extra = (word_col < words_per_row - 1) ? ((s_right & 1ULL) << 63) : 0ULL;
    uint64_t SE = SE_intra | SE_extra;
    
    // Sum up the contributions of all eight neighbors using bit-sliced addition.
    // Each of the 64 lanes (cells) will have its count (ranging 0..8) stored as a 4-bit number,
    // with the bits distributed across sum3 (MSB), sum2, sum1, and sum0 (LSB).
    uint64_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    add_bit(NW, sum0, sum1, sum2, sum3);
    add_bit(N,  sum0, sum1, sum2, sum3);
    add_bit(NE, sum0, sum1, sum2, sum3);
    add_bit(W,  sum0, sum1, sum2, sum3);
    add_bit(E,  sum0, sum1, sum2, sum3);
    add_bit(SW, sum0, sum1, sum2, sum3);
    add_bit(S,  sum0, sum1, sum2, sum3);
    add_bit(SE, sum0, sum1, sum2, sum3);
    
    // According to the Game of Life rules:
    //   - A cell becomes (or remains) alive if it has exactly 3 live neighbors.
    //   - A cell remains alive if it has exactly 2 live neighbors.
    // Since we do not include the central cell in the neighbor sum,
    // the rule to be applied is: new_state = (neighbors == 3) OR (current_cell AND (neighbors == 2))
    //
    // The bit-sliced neighbor count for a given cell is stored in the 4-bit number:
    //     count = (sum3, sum2, sum1, sum0)
    // We can check for equality with 3 (binary 0011) and 2 (binary 0010) by testing:
    //     eq_3: (sum3==0) & (sum2==0) & (sum1==1) & (sum0==1)
    //     eq_2: (sum3==0) & (sum2==0) & (sum1==1) & (sum0==0)
    uint64_t eq3 = (~sum3) & (~sum2) & sum1 & sum0;
    uint64_t eq2 = (~sum3) & (~sum2) & sum1 & (~sum0);
    
    // Compute the next state mask for the 64 cells within this word.
    uint64_t next_state = eq3 | (center & eq2);
    
    // Store the resulting 64-bit word into the output grid.
    output[index] = next_state;
}

// run_game_of_life executes one simulation step of Conway's Game of Life by launching the CUDA kernel.
// The grid is a square of 'grid_dimensions' cells per side, stored in bit‐packed format:
// each std::uint64_t holds 64 consecutive cells (1 bit per cell). The 0th bit (least‐significant bit)
// corresponds to the leftmost cell of the word and the 63rd bit to its rightmost cell (thus, special handling
// is required at the word boundaries).
//
// The function assumes that both input and output pointers refer to memory allocated with cudaMalloc,
// and that any necessary host-device synchronization will be performed by the caller.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Compute how many 64‐bit words comprise a row.
    int words_per_row = grid_dimensions / 64;
    int total_rows = grid_dimensions;
    
    // Select a 2D thread block configuration. A 16×16 block is chosen for good occupancy.
    dim3 blockDim(16, 16);
    // The grid dimensions are computed in terms of the word grid (columns = words_per_row, rows = total_rows).
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (total_rows + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel.  Note that any required synchronization (such as cudaDeviceSynchronize)
    // is assumed to be handled by the caller.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
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
