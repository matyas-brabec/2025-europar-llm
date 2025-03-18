#include <cstdint>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// In this implementation, the 2D grid is bit‐packed: each 64‐bit word
// holds 64 cells (one cell per bit). The grid is square with
// grid_dimensions cells per side. Each row contains (grid_dimensions/64)
// words. Each CUDA thread processes one 64‐bit word (i.e. 64 cells).
//
// The Game of Life rule is:
//   next_state = (neighbors == 3) OR (current_cell AND (neighbors == 2))
// where neighbors is computed from the eight neighboring cells.
// 
// Bit ordering convention (per 64‐bit word):
//   • Bit 0 (LSB) corresponds to the left‐most cell in that 64‐cell segment.
//   • Bit 63 corresponds to the right‐most cell.
// 
// Neighbor positions for a cell in the current word require careful handling.
// For horizontal neighbors in the current row:
//   – For a cell with bit index j (0 ≤ j < 64), its left neighbor is at index j–1,
//     except for j==0 where the left neighbor is taken from the adjacent left word’s bit 63.
//   – Similarly, its right neighbor is at index j+1 unless j==63, in which case it comes
//     from the adjacent right word’s bit 0.
// For rows above and below, the contributions are computed analogously using three words:
// left, center, and right. For a given adjacent row, the neighbors for a cell in the
// current word are:
//   • Diagonal left: use the center word shifted (or, for cell 0, take the left adjacent word’s bit 63).
//   • Vertical: the same bit position from the center word of that row.
//   • Diagonal right: use the center word shifted the opposite way (or, for cell 63, take the
//     right adjacent word’s bit 0).
//
// To perform the neighbor count we add eight binary values (each either 0 or 1) per cell,
// but the addition must be performed “in parallel” (i.e. per cell) without inter‐cell carries.
// We represent the per–cell count (which can be 0–8) in bit–sliced form with 4 bit–planes:
// s0 (LSB), s1, s2, s3. Each of these is a 64–bit word where bit i holds the corresponding bit
// of the count for cell i in the current word.
// Bit–wise addition is implemented by adding one 1–bit (in parallel over 64 cells) at a time,
// using a ripple–carry approach that works independently on each cell.
//---------------------------------------------------------------------

// Device helper: add a 1-bit value (mask "bit") to the bit–sliced accumulator (s0,s1,s2,s3)
// using ripple–carry addition (each bit position corresponds to one cell; no inter–cell carry).
__device__ inline void add_bit(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3, uint64_t bit)
{
    uint64_t carry = bit;
    uint64_t temp;

    // Add to the 0th bit-plane.
    temp   = s0 ^ carry;
    carry  = s0 & carry;
    s0     = temp;

    // Propagate carry to the 1st bit-plane.
    temp   = s1 ^ carry;
    carry  = s1 & carry;
    s1     = temp;

    // Propagate carry to the 2nd bit-plane.
    temp   = s2 ^ carry;
    carry  = s2 & carry;
    s2     = temp;

    // Propagate carry to the 3rd bit-plane.
    temp   = s3 ^ carry;
    carry  = s3 & carry;
    s3     = temp;
    // Maximum count is 8 so 4 bits suffice.
}

// CUDA kernel: processes one 64-bit word (64 cells)
__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    // Each row contains (grid_dimensions / 64) words.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    int total_words   = grid_dimensions * words_per_row;  // total words in grid

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    // Determine 2D location: row and column (word index in row)
    int row = idx / words_per_row;
    int col = idx % words_per_row;

    // Load the "center" word (current row, current word).
    uint64_t center = input[row * words_per_row + col];

    // Load current row's adjacent words (left and right) if available; otherwise use 0.
    uint64_t left_word  = (col > 0)                     ? input[row * words_per_row + col - 1] : 0ULL;
    uint64_t right_word = (col < words_per_row - 1)       ? input[row * words_per_row + col + 1] : 0ULL;

    // Load the words from the row above ("top") if available.
    uint64_t top_center      = 0ULL, top_left_word = 0ULL, top_right_word = 0ULL;
    if (row > 0) {
        top_center      = input[(row - 1) * words_per_row + col];
        top_left_word   = (col > 0)               ? input[(row - 1) * words_per_row + col - 1] : 0ULL;
        top_right_word  = (col < words_per_row - 1) ? input[(row - 1) * words_per_row + col + 1] : 0ULL;
    }

    // Load the words from the row below ("bottom") if available.
    uint64_t bottom_center      = 0ULL, bottom_left_word = 0ULL, bottom_right_word = 0ULL;
    if (row < grid_dimensions - 1) {
        bottom_center     = input[(row + 1) * words_per_row + col];
        bottom_left_word  = (col > 0)               ? input[(row + 1) * words_per_row + col - 1] : 0ULL;
        bottom_right_word = (col < words_per_row - 1) ? input[(row + 1) * words_per_row + col + 1] : 0ULL;
    }

    //-----------------------------------------------------------------
    // For each neighbor, we need to extract a 1-bit mask for each cell in the
    // current word. The extraction uses bit shifts with "wrap" from adjacent words.
    //
    // Bit–ordering convention:
    //   • Bit 0 (LSB) is leftmost cell.
    //   • Bit 63 is rightmost cell.
    //
    // Horizontal neighbor extraction:
    //   – Left neighbor: For cells with index > 0, use (word << 1) so that bit j
    //     becomes the cell from index j-1. For cell 0, use the adjacent left word’s
    //     rightmost cell: (left_word >> 63).
    //   – Right neighbor: For cells with index < 63, use (word >> 1). For cell 63,
    //     use the adjacent right word’s leftmost cell: ((right_word & 1ULL) << 63).
    //-----------------------------------------------------------------

    // Current row neighbors (do not include the center cell).
    uint64_t current_left  = (center << 1) | ((col > 0) ? (left_word >> 63) : 0ULL);
    uint64_t current_right = (center >> 1) | ((col < words_per_row - 1) ? ((right_word & 1ULL) << 63) : 0ULL);

    // Top row neighbor contributions.
    // If no top row, these default to 0.
    uint64_t top_left    = (top_center << 1) | ((row > 0 && col > 0) ? (top_left_word >> 63) : 0ULL);
    uint64_t top_center_contrib = top_center;  // vertical neighbor from top row.
    uint64_t top_right   = (top_center >> 1) | ((row > 0 && col < words_per_row - 1) ? ((top_right_word & 1ULL) << 63) : 0ULL);

    // Bottom row neighbor contributions.
    // If no bottom row, these default to 0.
    uint64_t bottom_left  = (bottom_center << 1) | ((row < grid_dimensions - 1 && col > 0) ? (bottom_left_word >> 63) : 0ULL);
    uint64_t bottom_center_contrib = bottom_center;  // vertical neighbor from bottom row.
    uint64_t bottom_right = (bottom_center >> 1) | ((row < grid_dimensions - 1 && col < words_per_row - 1) ? ((bottom_right_word & 1ULL) << 63) : 0ULL);

    //-----------------------------------------------------------------
    // Sum the eight neighbor bits (from top, current, and bottom rows) for
    // each cell in this 64–cell word. The eight neighbor bits are added in a
    // bit–sliced manner so that each cell’s sum (0..8) is represented with 4 bits.
    //
    // We use four 64-bit accumulators: s0 (LSB), s1, s2, s3.
    // Each call to add_bit(...) adds one 1-bit value (mask) per cell.
    //-----------------------------------------------------------------
    uint64_t s0 = 0ULL, s1 = 0ULL, s2 = 0ULL, s3 = 0ULL;
    add_bit(s0, s1, s2, s3, top_left);
    add_bit(s0, s1, s2, s3, top_center_contrib);
    add_bit(s0, s1, s2, s3, top_right);
    add_bit(s0, s1, s2, s3, current_left);
    add_bit(s0, s1, s2, s3, current_right);
    add_bit(s0, s1, s2, s3, bottom_left);
    add_bit(s0, s1, s2, s3, bottom_center_contrib);
    add_bit(s0, s1, s2, s3, bottom_right);

    //-----------------------------------------------------------------
    // At this point, for each cell (each bit in the word), the neighbor count
    // is encoded in four bits:
    //   count = s0 + 2*s1 + 4*s2 + 8*s3.
    //
    // We now compute masks for the conditions:
    //   (neighbors == 3) and (neighbors == 2).
    // For count 3, the binary form is 0011 (s3=0, s2=0, s1=1, s0=1).
    // For count 2, it is 0010 (s3=0, s2=0, s1=1, s0=0).
    //-----------------------------------------------------------------
    uint64_t eq3 = s0 & s1 & ~(s2) & ~(s3);         // equals 3
    uint64_t eq2 = (~s0) & s1 & ~(s2) & ~(s3);        // equals 2

    //-----------------------------------------------------------------
    // Apply the Game of Life rules:
    //   A cell becomes live if it has exactly 3 live neighbors or if it is
    //   currently live and has exactly 2 live neighbors.
    //-----------------------------------------------------------------
    uint64_t next = eq3 | (center & eq2);

    // Write the computed next state word to output.
    output[row * words_per_row + col] = next;
}

//---------------------------------------------------------------------
// Host function: run_game_of_life
// Executes one step of Conway’s Game of Life using the CUDA kernel.
//   • input: pointer to input grid (bit–packed), allocated on device.
//   • output: pointer to output grid (bit–packed), allocated on device.
//   • grid_dimensions: width and height (in cells) of the square grid.
//---------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input,
void run_game_of_life(const std::uint64_t* input,
                                 std::uint64_t* output,
                                 int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    // Total words in the grid.
    int total_words = grid_dimensions * words_per_row;

    // Set CUDA launch configuration.
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
    // Device synchronization (if needed) is assumed to be handled by the caller.
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
