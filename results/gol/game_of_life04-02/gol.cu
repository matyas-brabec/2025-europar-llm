// This file implements one iteration of Conway's Game of Life in CUDA.
// The grid is 2D, bit‐packed: each std::uint64_t represents 64 consecutive cells in a row,
// where bit index 0 is defined to be the leftmost cell and bit index 63 the rightmost cell.
// Neighbors outside the grid are considered dead.
// Each CUDA thread processes one 64‐cell word (one std::uint64_t) from the grid.
// For each cell in a word, we compute the number of live neighbors from three rows:
// the row above, the current row, and the row below.  For cells at bit index 0 or 63
// we need to access bits from the adjacent words in the same row, as well as for the rows
// above and below.
// We use a lookup table (LUT) to decide the next state for a cell given its current state
// (0 or 1) and its live‐neighbor count (0..8).  The rules are:
//   • A dead cell becomes alive if it has exactly 3 live neighbors.
//   • A live cell survives if it has 2 or 3 live neighbors, and dies otherwise.
// The LUT is indexed as follows:
//    lut[ (cell << 3) + count ]
// where cell is 0 or 1 and count is in the range 0..8.
// (For cell==0: only index 3 yields 1; for cell==1: indices 2 and 3 yield 1.)
//
// Note: We assume that no shared or texture memory is needed and all grids (input and output)
// are allocated with cudaMalloc.  Thread synchronization is assumed to be handled externally.
//
// Each thread loops over the 64 bit positions in its word. For each bit position,
// we use an inline helper to extract the bit value (0 or 1) from a word,
// taking care to interpret bit position 0 as the left‐most cell (i.e. using shift by (63-pos)).

#include <cstdint>
#include <cuda_runtime.h>

#define UNROLL_LOOP _Pragma("unroll")

// Device inline function to extract a bit from a 64‐bit word.
// The bit positions are defined so that pos==0 corresponds to the leftmost bit
// and pos==63 the rightmost bit.
__device__ __forceinline__
int get_bit(std::uint64_t word, int pos) {
    return int((word >> (63 - pos)) & 1ULL);
}

// The device kernel that computes one generation of Conway's Game of Life.
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                           std::uint64_t* __restrict__ output,
                           int grid_dimensions)
{
    // Compute the number of 64-cell words per row.
    const int words_per_row = grid_dimensions / 64;

    // Compute the row (in cell-space) and the column (word index) in the grid.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // col is the index of the 64-cell word

    // Boundary check.
    if (row >= grid_dimensions || col >= words_per_row)
        return;

    // Compute base indices.
    int index = row * words_per_row + col;

    // Load the current row's words.
    std::uint64_t b1 = input[index];
    std::uint64_t b0 = (col > 0) ? input[row * words_per_row + col - 1] : 0ULL;
    std::uint64_t b2 = (col < words_per_row - 1) ? input[row * words_per_row + col + 1] : 0ULL;

    // Load the row above (if any).
    std::uint64_t a0, a1, a2;
    if (row > 0) {
        int rowAbove = row - 1;
        a1 = input[rowAbove * words_per_row + col];
        a0 = (col > 0) ? input[rowAbove * words_per_row + col - 1] : 0ULL;
        a2 = (col < words_per_row - 1) ? input[rowAbove * words_per_row + col + 1] : 0ULL;
    } else {
        a0 = a1 = a2 = 0ULL;
    }

    // Load the row below (if any).
    std::uint64_t c0, c1, c2;
    if (row < grid_dimensions - 1) {
        int rowBelow = row + 1;
        c1 = input[rowBelow * words_per_row + col];
        c0 = (col > 0) ? input[rowBelow * words_per_row + col - 1] : 0ULL;
        c2 = (col < words_per_row - 1) ? input[rowBelow * words_per_row + col + 1] : 0ULL;
    } else {
        c0 = c1 = c2 = 0ULL;
    }

    // Lookup table for next cell state.
    // For a dead cell (cell=0), only count==3 leads to alive.
    // For a live cell (cell=1), count==2 or count==3 lead to survival.
    // The LUT is indexed by: (cell * 9 + count)
    // LUT layout:
    //    For cell==0: indices [0..8]: {0,0,0,1,0,0,0,0,0}
    //    For cell==1: indices [9..17]: {0,0,1,1,0,0,0,0,0}
    const int lut[18] = {0,0,0,1,0,0,0,0,0,
                           0,0,1,1,0,0,0,0,0};

    // The word that will hold the next state for the 64 cells in this block.
    std::uint64_t result = 0ULL;

    // Process each cell (bit) in the 64-cell word.
    UNROLL_LOOP
    for (int pos = 0; pos < 64; pos++) {
        int count = 0;

        // For row above:
        // NW: if pos > 0, take from a1 at pos-1; else (for leftmost cell) take from a0 at rightmost bit (pos 63)
        count += (pos > 0) ? get_bit(a1, pos - 1) : get_bit(a0, 63);
        // N: take from a1 at same pos.
        count += get_bit(a1, pos);
        // NE: if pos < 63, take from a1 at pos+1; else take from a2 at leftmost bit (pos 0)
        count += (pos < 63) ? get_bit(a1, pos + 1) : get_bit(a2, 0);

        // For current row (neighbors from same row, excluding center cell):
        // W: if pos > 0, take from b1 at pos-1; else from b0 at pos 63.
        count += (pos > 0) ? get_bit(b1, pos - 1) : get_bit(b0, 63);
        // E: if pos < 63, take from b1 at pos+1; else from b2 at pos 0.
        count += (pos < 63) ? get_bit(b1, pos + 1) : get_bit(b2, 0);

        // For row below:
        // SW:
        count += (pos > 0) ? get_bit(c1, pos - 1) : get_bit(c0, 63);
        // S:
        count += get_bit(c1, pos);
        // SE:
        count += (pos < 63) ? get_bit(c1, pos + 1) : get_bit(c2, 0);

        // Get the current cell state from b1.
        int cell = get_bit(b1, pos);
        // Use LUT to determine new state.
        int new_state = lut[(cell << 3) + count];

        // Set the new state into the result word.
        // Since bit position 0 represents the leftmost cell, we store the new cell in bit (63-pos).
        if (new_state)
            result |= (1ULL << (63 - pos));
    }

    // Write the computed next-generation word back to the output grid.
    output[index] = result;
}

// Host function that runs one iteration of the Game of Life.
// The grid is a square of dimension grid_dimensions x grid_dimensions,
// where grid_dimensions is a power of 2 and each row is stored in bit‐packed
// format (64 cells per std::uint64_t). The "input" and "output" pointers refer
// to device memory allocated via cudaMalloc.
/// @FIXED: extern "C"

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate the number of 64-cell words per row.
    int words_per_row = grid_dimensions / 64;

    // Configure 2D grid: each thread processes one word.
    // We choose a 2D block size; here 16x16 threads per block.
    dim3 blockDim(16, 16);
    // Grid dimensions: x dimension covers words_per_row, y dimension covers grid_dimensions (rows).
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
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
