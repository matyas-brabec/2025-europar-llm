#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Overview:
// This CUDA implementation performs one iteration (step) of Conway’s Game of Life
// on a 2D grid that is stored in bit‐packed form. Each std::uint64_t stores 64
// consecutive cells (one bit per cell). The grid is square with “grid_dim” cells
// per side (a power‐of‐2, e.g. 1024 or 2048). Thus each row contains (grid_dim/64)
// words. We assume that the bits are interpreted in “logical order” such that
// cell column 0 is the leftmost and cell column grid_dim-1 is the rightmost.
// To “vectorize” the computation, each CUDA thread is responsible for updating one
// 64–cell block (one word). Using “bit–parallel” operations we precompute the
// eight neighbor masks of the entire word. However, the two extreme cells in a word
// (the leftmost and rightmost, respectively) need “special handling” because their
// required neighbors come partly from an adjacent word.
// 
// To resolve ambiguity between machine–bit order and logical cell ordering, we adopt
// the following convention: We store each 64–cell block in a std::uint64_t so that
// its bits are arranged in “machine order” (bit 63 is the most–significant and bit 0 is
// the least–significant). We interpret the block so that the leftmost (logical) cell is
// in bit 63 and the rightmost cell is in bit 0. Then (with standard C/C++ shifts):
//    • The west (left) neighbor of a cell is obtained by shifting LEFT (<< 1).
//    • The east (right) neighbor is obtained by shifting RIGHT (>> 1).
// In a given word, for any cell not on a horizontal boundary the shift produces the 
// correct neighbor. But for the leftmost cell (logical col 0, machine bit 63) its west
// neighbor comes from the adjacent word to the left (if any) and similarly the east
// neighbor of the rightmost cell (logical col grid/64-1, machine bit 0) comes from the
// word immediately to the right.
// 
// For each of the three rows (row above, current row, and row below) we compute the 
// contributions (neighbors) in three “columns”: the block in question, its left neighbor 
// block and its right neighbor block. (That is why for the extreme cells we “must also 
// consider the three words” to the left or to the right.) Finally, the Game of Life
// transition is computed cell–wise by the standard rule:
//      new = (neighbors==3) || (alive && neighbors==2)
// 
// For performance reasons the kernel computes the eight neighbor masks in parallel
// (using whole–word bit–operations) and then loops over the 64 bit–positions.
// (Note: Although we loop over 64 positions per thread, there is massive thread–level
// parallelism. This “per–word” processing is much better than processing one cell at a
// time across threads.)
// -----------------------------------------------------------------------------


// CUDA kernel: each thread updates one 64–cell (64–bit) word.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Compute word coordinates.
    // wx: index of the 64–bit word in this row.
    // wy: row index (in cell units, one entire row holds grid_dim cells).
    int wx = blockIdx.x * blockDim.x + threadIdx.x;
    int wy = blockIdx.y * blockDim.y + threadIdx.y;
    if (wx >= words_per_row || wy >= grid_dim)
        return;

    // Compute index for current word.
    int idx = wy * words_per_row + wx;

    //------------------------------------------------------------------------
    // Load the needed words for the three rows:
    //   • Current row: load the current word plus its immediate left/right words.
    //   • Row above (if any): similarly.
    //   • Row below (if any): similarly.
    // Out‐of–boundary accesses yield 0 (dead cells).
    //------------------------------------------------------------------------
    // Current row.
    std::uint64_t cur       = input[idx];
    std::uint64_t left_cur  = (wx > 0)              ? input[wy * words_per_row + (wx - 1)] : 0ULL;
    std::uint64_t right_cur = (wx < words_per_row-1) ? input[wy * words_per_row + (wx + 1)] : 0ULL;

    // Row above.
    std::uint64_t up = 0, left_up = 0, right_up = 0;
    if (wy > 0) {
        int up_base = (wy - 1) * words_per_row;
        up       = input[up_base + wx];
        left_up  = (wx > 0)              ? input[up_base + (wx - 1)] : 0ULL;
        right_up = (wx < words_per_row-1) ? input[up_base + (wx + 1)] : 0ULL;
    }

    // Row below.
    std::uint64_t down = 0, left_down = 0, right_down = 0;
    if (wy < grid_dim - 1) {
        int down_base = (wy + 1) * words_per_row;
        down       = input[down_base + wx];
        left_down  = (wx > 0)              ? input[down_base + (wx - 1)] : 0ULL;
        right_down = (wx < words_per_row-1) ? input[down_base + (wx + 1)] : 0ULL;
    }

    //------------------------------------------------------------------------
    // In our convention, each 64–bit word holds 64 cells with:
    //    Logical column 0 (leftmost)  -> bit 63 (MSB)
    //    Logical column 63 (rightmost) -> bit 0 (LSB)
    // This allows us to use the standard shift operators as follows:
    //    • To get the west (left) neighbor of a cell, use (word << 1).
    //    • To get the east (right) neighbor, use (word >> 1).
    //
    // For cells on horizontal boundaries in the current word, we must fetch the missing
    // bit from the adjacent word. For example:
    //    • For the current row, the left neighbor mask is:
    //         same_west = (cur << 1) OR (if (wx > 0) then the rightmost cell from left_cur)
    //      and the right neighbor mask is:
    //         same_east = (cur >> 1) OR (if (wx < words_per_row-1) then the leftmost cell
    //                    from right_cur, which is in bit 63 of that word).
    // The same is done for the row above and below.
    //------------------------------------------------------------------------
    // Current row horizontal neighbors.
    std::uint64_t same_west = (cur << 1)
                            | ((wx > 0) ? ((left_cur >> 0) & 1ULL) : 0ULL);
    std::uint64_t same_east = (cur >> 1)
                            | ((wx < words_per_row-1) ? (((right_cur >> 63) & 1ULL) << 63) : 0ULL);

    // Row above horizontal neighbors.
    std::uint64_t up_west = (up << 1)
                          | ((wx > 0) ? ((left_up >> 0) & 1ULL) : 0ULL);
    std::uint64_t up_east = (up >> 1)
                          | ((wx < words_per_row-1) ? (((right_up >> 63) & 1ULL) << 63) : 0ULL);

    // Row below horizontal neighbors.
    std::uint64_t down_west = (down << 1)
                            | ((wx > 0) ? ((left_down >> 0) & 1ULL) : 0ULL);
    std::uint64_t down_east = (down >> 1)
                            | ((wx < words_per_row-1) ? (((right_down >> 63) & 1ULL) << 63) : 0ULL);

    //------------------------------------------------------------------------
    // Each cell’s eight neighbors come from:
    //    • Row above: up_west, up, up_east.
    //    • Same row: same_west, same_east.
    //    • Row below: down_west, down, down_east.
    //
    // To update the 64 cells in the current word in parallel (i.e. in one thread),
    // we would ideally “sum” the corresponding bits (each 0 or 1) in eight separate 64–bit
    // masks without inter–bit carries. Because each cell’s neighbor count fits in 4 bits,
    // an ideal “nibble–parallel” addition would pack 64 independent 4–bit counters into
    // a 256–bit word. In our implementation we choose a simpler route: although we first
    // compute full–word masks using bit–operations, we then loop over the 64 bit–positions.
    // The loop is expected to be unrolled and the massive thread parallelism amortizes the cost.
    //------------------------------------------------------------------------
    std::uint64_t new_word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        // Determine the machine bit position corresponding to the logical cell.
        // Logical column 0 is stored in bit position 63, and logical column 63 in bit 0.
        int pos = 63 - bit;
        // Sum the eight contributions from neighbor masks.
        int count = 0;
        count += (int)((up_west   >> pos) & 1ULL);
        count += (int)((up          >> pos) & 1ULL);
        count += (int)((up_east   >> pos) & 1ULL);
        count += (int)((same_west >> pos) & 1ULL);
        count += (int)((same_east >> pos) & 1ULL);
        count += (int)((down_west >> pos) & 1ULL);
        count += (int)((down      >> pos) & 1ULL);
        count += (int)((down_east >> pos) & 1ULL);
        // Get the current cell state.
        int cell = (int)((cur >> pos) & 1ULL);
        // Compute the new cell state using the Game of Life rules:
        //   - A live cell survives if it has 2 or 3 live neighbors.
        //   - A dead cell becomes live if it has exactly 3 live neighbors.
        int new_cell = 0;
        if (cell)
            new_cell = (count == 2 || count == 3);
        else
            new_cell = (count == 3);
        // Write the new cell state into the proper bit position.
        new_word |= ((std::uint64_t)new_cell << pos);
    }

    // Write the updated 64–cell word to the output grid.
    output[idx] = new_word;
}

// -----------------------------------------------------------------------------
// Host function: run_game_of_life
//
// Executes one simulation step. The input and output grids are allocated on the device
// using cudaMalloc and are stored in bit–packed format (std::uint64_t per 64 cells).
//
// grid_dim: the width (and height) of the grid in cells.
// It is assumed that grid_dim is a power of 2 (e.g., 1024, 2048) and at least 512.
// -----------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dim)
{
    // Compute how many 64–cell words are in one row.
    int words_per_row = grid_dim / 64;

    // Choose a 2D block and grid configuration.
    // With modern GPUs (e.g. A100 or H100), a block of 16x16 threads (256 threads)
    // is a good starting point.
    dim3 blockDim(16, 16);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dim + blockDim.y - 1) / blockDim.y);

    // Launch the kernel. (Note: host-device synchronization is handled by the caller.)
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dim, words_per_row);
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
