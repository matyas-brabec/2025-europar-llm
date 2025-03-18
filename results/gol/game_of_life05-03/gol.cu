#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// This CUDA implementation of Conway's Game of Life operates on a square grid
// where each cell is represented by one bit. The grid is stored in bit‐packed
// format, with each std::uint64_t holding 64 consecutive cells in a row.
// 
// Each thread is assigned one 64‐bit word (i.e. 64 cells) from the grid, and
// computes the evolution for all 64 cells concurrently. For each cell the eight
// neighbors are examined (upper-left, up, upper-right, left, right, lower-left,
// down, lower-right) following the rules:
//
//   • Any live cell with 2 or 3 live neighbors survives.
//   • Any dead cell with exactly 3 live neighbors becomes live.
//   • Otherwise the cell dies or remains dead.
//
// For boundary conditions, cells beyond the grid are treated as dead. In order
// to correctly compute neighbor counts for cells on word boundaries (bit index
// 0 and 63 of a word), the kernel loads not only the current word but also its
// left and right neighbors from the same row, and similarly for the rows above
// and below.
// 
// In the inner loop the 64-bit word is processed 8 bits at a time (8 cells per
// inner iteration) to improve performance relative to testing each cell in a
// separate loop iteration. The inner loops are unrolled so that the compiler
// can optimize the bit arithmetic.
// ----------------------------------------------------------------------------- 

// Device kernel: Each thread processes one std::uint64_t word (64 cells).
// The grid is arranged in rows; each row is composed of (grid_dimensions/64)
// words. For a given thread we compute the row index and column (word index).
// For proper neighbor access the kernel loads the 3x3 neighborhood of words (if
// within bounds; otherwise missing words are replaced by 0). Then for every cell
// (bit) in the current word we compute the neighbor sum and apply the rules.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of 64-bit words per row.
    int word_cols = grid_dimensions >> 6; // grid_dimensions / 64
    // Global thread index – each thread processes one 64-bit word.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dimensions * word_cols;  // total words in the grid

    if(index >= total_words)
        return;

    // Determine the row and column (word index within the row) this thread is
    // responsible for.
    int row = index / word_cols;
    int col = index % word_cols;

    // -------------------------------------------------------------------------
    // Load the neighboring words needed for computing the eight neighbors.
    // For a cell in the current word (mid_center), the neighbors lie in the row
    // above (top_), in the current row (mid_) and in the row below (bot_).
    // For each row, we load three words:
    //   - Left: word at column index (col-1) (if available, else 0)
    //   - Center: word at column index col
    //   - Right: word at column index (col+1) (if available, else 0)
    // Boundary conditions (row==0 or row==grid_dimensions-1) and (col==0 or
    // col==word_cols-1) are handled by substituting 0.
    // -------------------------------------------------------------------------
    std::uint64_t top_left = 0, top_center = 0, top_right = 0;
    std::uint64_t mid_left = 0, mid_center, mid_right = 0;
    std::uint64_t bot_left = 0, bot_center = 0, bot_right = 0;

    // Row above.
    if(row > 0) {
        int top_row_index = (row - 1) * word_cols;
        if(col > 0)
            top_left = input[top_row_index + col - 1];
        top_center = input[top_row_index + col];
        if(col < word_cols - 1)
            top_right = input[top_row_index + col + 1];
    }
    // Current row.
    int mid_row_index = row * word_cols;
    if(col > 0)
        mid_left = input[mid_row_index + col - 1];
    mid_center = input[mid_row_index + col];
    if(col < word_cols - 1)
        mid_right = input[mid_row_index + col + 1];
    // Row below.
    if(row < grid_dimensions - 1) {
        int bot_row_index = (row + 1) * word_cols;
        if(col > 0)
            bot_left = input[bot_row_index + col - 1];
        bot_center = input[bot_row_index + col];
        if(col < word_cols - 1)
            bot_right = input[bot_row_index + col + 1];
    }

    // -------------------------------------------------------------------------
    // Process the 64 cells (bits) in the current word.
    // To increase performance we process 8 cells at a time (an 8-bit chunk) in the
    // outer loop; inside an inner loop (unrolled) we process each bit in the chunk.
    // The neighbor for a cell at bit position 'pos' (0 <= pos < 64) is obtained by:
    //    • Top row: positions (pos-1, pos, pos+1) from top_center, except that when
    //      pos==0, the left neighbor comes from top_left (bit 63), and when pos==63,
    //      the right neighbor comes from top_right (bit 0).
    //    • Same row: positions (pos-1) from mid_center for left and (pos+1) for
    //      right, with the same adjustments from mid_left and mid_right at the word
    //      boundaries.
    //    • Bottom row: similar to the top row, but using bot_center, bot_left,
    //      and bot_right.
    // The current cell’s state is taken from mid_center (bit at position pos).
    // The Game of Life rule is then applied: a cell becomes live if exactly 3
    // neighbors are live or if it is live and has exactly 2 live neighbors.
    // -------------------------------------------------------------------------
    std::uint64_t new_word = 0;

    #pragma unroll
    for (int chunk = 0; chunk < 8; chunk++) {
        // Process an 8-bit chunk; result for these 8 cells will be stored in res_chunk.
        unsigned char res_chunk = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            int pos = (chunk << 3) + bit;  // pos = chunk*8 + bit, 0 <= pos < 64

            // Extract the current cell's state (0 or 1) from mid_center.
            int current = (mid_center >> pos) & 1;

            int count = 0; // count of live neighbors

            // Top-left neighbor.
            if (row > 0) {
                if (pos == 0)
                    count += (col > 0 ? (top_left >> 63) & 1 : 0);
                else
                    count += (top_center >> (pos - 1)) & 1;
            }

            // Top neighbor.
            if (row > 0) {
                count += (top_center >> pos) & 1;
            }

            // Top-right neighbor.
            if (row > 0) {
                if (pos == 63)
                    count += (col < word_cols - 1 ? (top_right >> 0) & 1 : 0);
                else
                    count += (top_center >> (pos + 1)) & 1;
            }

            // Left neighbor.
            if (pos == 0)
                count += (col > 0 ? (mid_left >> 63) & 1 : 0);
            else
                count += (mid_center >> (pos - 1)) & 1;

            // Right neighbor.
            if (pos == 63)
                count += (col < word_cols - 1 ? (mid_right >> 0) & 1 : 0);
            else
                count += (mid_center >> (pos + 1)) & 1;

            // Bottom-left neighbor.
            if (row < grid_dimensions - 1) {
                if (pos == 0)
                    count += (col > 0 ? (bot_left >> 63) & 1 : 0);
                else
                    count += (bot_center >> (pos - 1)) & 1;
            }

            // Bottom neighbor.
            if (row < grid_dimensions - 1) {
                count += (bot_center >> pos) & 1;
            }

            // Bottom-right neighbor.
            if (row < grid_dimensions - 1) {
                if (pos == 63)
                    count += (col < word_cols - 1 ? (bot_right >> 0) & 1 : 0);
                else
                    count += (bot_center >> (pos + 1)) & 1;
            }

            // Apply Conway's Game of Life rules:
            // A cell is live in the next generation if either:
            //    (a) It has exactly 3 live neighbors, or
            //    (b) It is currently live and has exactly 2 live neighbors.
            int next_state = (count == 3) || (current && (count == 2));

            // Set the computed next state bit in the 8-bit result (res_chunk) at 'bit' pos.
            res_chunk |= static_cast<unsigned char>(next_state << bit);
        }
        // Combine the 8-bit chunk into new_word at the correct position.
        new_word |= (static_cast<std::uint64_t>(res_chunk) << (chunk << 3));
    }

    // Write the updated 64-bit word to the output grid.
    output[index] = new_word;
}

// -----------------------------------------------------------------------------
// run_game_of_life:
// This host function launches one simulation step of Conway's Game of Life.
// The grids are stored in bit-packed format: each std::uint64_t holds 64 cells,
// arranged consecutively within a row. The grid is square with dimensions
// grid_dimensions x grid_dimensions, where grid_dimensions is a power-of-2.
// 
// The input and output pointers refer to memory allocated by cudaMalloc.
// Any host-device synchronization is handled externally.
// -----------------------------------------------------------------------------
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int word_cols = grid_dimensions >> 6; // grid_dimensions / 64
    // Total 64-bit words in the grid.
    int total_words = grid_dimensions * word_cols;

    // Choose a block size that is a multiple of the warp size.
    int blockSize = 256;
    int numBlocks = (total_words + blockSize - 1) / blockSize;

    // Launch the kernel: each thread processes one 64-bit word.
    game_of_life_kernel<<<numBlocks, blockSize>>>(input, output, grid_dimensions);
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
