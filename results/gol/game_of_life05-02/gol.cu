#include <cstdint>
#include <cuda_runtime.h>

// This kernel implements one iteration (one generation) of Conway’s Game of Life
// on a 2D grid whose cells are stored in a bit‐packed format. Each uint64_t holds
// 64 consecutive cells (one bit per cell) in a row. The grid is square with
// grid_dimensions cells per side. Because grid dimensions are a power of 2 and
// each row is divided into 64–bit words, the number of words per row is (grid_dimensions / 64).
//
// Each CUDA thread processes one 64–bit word (i.e. 64 cells) at once. In order to
// decide the fate of each cell in the word, the thread gathers the “neighbor” bits
// from eight directions.
// 
// When processing a word, most cells’ neighbors come from the current neighbor words
// (from the same word, its left/right neighbor in the same row, and the corresponding
// word(s) from the row above and the row below). However, the 0th and 63rd bits of a word
// are “boundary” cells: for the 0th bit, the left–shifted neighbor comes from the previous
// word (if it exists), and for the 63rd bit the right–shifted neighbor comes from the next word.
// Thus, for each neighbor direction we combine a horizontally shifted version of the
// central neighbor word with an extra bit from the adjacent word when needed.
//
// The “rules” are applied per cell (bit):
//   • A live cell (bit==1) survives if it has 2 or 3 live neighbors.
//   • A dead cell (bit==0) becomes live if it has exactly 3 live neighbors.
// Otherwise, the cell is dead in the next generation.
//
// In order to avoid atomic operations, each thread computes 64 cells at once and writes
// the result into the output word.
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Compute the current thread’s location in the grid of words
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;  // word index within the row
    int row     = blockIdx.y * blockDim.y + threadIdx.y;  // row index (each row has grid_dim cells)
    if (row >= grid_dim || word_x >= words_per_row)
        return;

    // Compute the flat index for the current word
    int idx = row * words_per_row + word_x;

    // Load the current word from global memory.
    std::uint64_t cur = input[idx];

    // Load horizontal neighbors from the same row.
    // For cells at the left and right boundaries of the word, we must fetch bits
    // from the adjacent word if available, or use 0 if at the grid border.
    std::uint64_t left_word  = (word_x > 0) ? input[row * words_per_row + word_x - 1] : 0;
    std::uint64_t right_word = (word_x < words_per_row - 1) ? input[row * words_per_row + word_x + 1] : 0;

    // Load the neighbor words for the row above, if it exists.
    std::uint64_t top      = (row > 0) ? input[(row - 1) * words_per_row + word_x] : 0;
    std::uint64_t top_left = (row > 0 && word_x > 0) ? input[(row - 1) * words_per_row + word_x - 1] : 0;
    std::uint64_t top_right = (row > 0 && word_x < words_per_row - 1) ?
                              input[(row - 1) * words_per_row + word_x + 1] : 0;

    // Load the neighbor words for the row below, if it exists.
    std::uint64_t bottom      = (row < grid_dim - 1) ? input[(row + 1) * words_per_row + word_x] : 0;
    std::uint64_t bottom_left = (row < grid_dim - 1 && word_x > 0) ?
                                input[(row + 1) * words_per_row + word_x - 1] : 0;
    std::uint64_t bottom_right = (row < grid_dim - 1 && word_x < words_per_row - 1) ?
                                 input[(row + 1) * words_per_row + word_x + 1] : 0;

    // For each neighbor “direction” we now produce a 64–bit mask that holds the neighbor value
    // for each cell in the current word (cells 0..63). Because cells at bit boundaries in a word
    // have neighbors in a different (adjacent) word, we combine a shifted version of the “central”
    // word with one bit from the adjacent word.
    //
    // SAME ROW:
    // Left neighbors: for cell j, if j > 0 then from current word bit (j-1),
    // and for j==0 from left_word (bit 63).
    std::uint64_t left_neighbors = (cur >> 1) | (((left_word >> 63) & 1ULL) << 0);

    // Right neighbors: for cell j, if j < 63 then from current word bit (j+1),
    // and for j==63 from right_word (bit 0).
    std::uint64_t right_neighbors = ((cur << 1) & ~(1ULL << 63)) | (((right_word & 1ULL)) << 63);

    // ROW ABOVE:
    // Center neighbors: directly aligned.
    std::uint64_t top_center = top;
    // Top-left: for cell j, if j > 0 then top >> 1, and for j==0 use top_left (bit 63).
    std::uint64_t top_left_neighbors = (top >> 1) | (((top_left >> 63) & 1ULL) << 0);
    // Top-right: for cell j, if j < 63 then top << 1 (with bit 63 cleared), and for j==63 use top_right (bit 0).
    std::uint64_t top_right_neighbors = ((top << 1) & ~(1ULL << 63)) | (((top_right & 1ULL)) << 63);

    // ROW BELOW:
    std::uint64_t bottom_center = bottom;
    std::uint64_t bottom_left_neighbors = (bottom >> 1) | (((bottom_left >> 63) & 1ULL) << 0);
    std::uint64_t bottom_right_neighbors = ((bottom << 1) & ~(1ULL << 63)) | (((bottom_right & 1ULL)) << 63);

    // For each cell (each bit in the word) we now compute the sum of live neighbors.
    // We iterate over each bit position from 0 to 63. The sum is obtained by extracting the
    // corresponding bit from each of the eight neighbor masks and adding them.
    // Then, using the Game of Life rules, we determine the cell’s state in the next generation:
    //   - A dead cell with exactly 3 live neighbors becomes live.
    //   - A live cell with 2 or 3 live neighbors survives.
    // (All cells outside the grid are dead.)
    std::uint64_t next_word = 0;
#pragma unroll
    for (int bit = 0; bit < 64; ++bit)
    {
        int count = 0;
        count += (int)((top_left_neighbors >> bit) & 1ULL);
        count += (int)((top_center       >> bit) & 1ULL);
        count += (int)((top_right_neighbors>> bit) & 1ULL);
        count += (int)((left_neighbors     >> bit) & 1ULL);
        count += (int)((right_neighbors    >> bit) & 1ULL);
        count += (int)((bottom_left_neighbors >> bit) & 1ULL);
        count += (int)((bottom_center      >> bit) & 1ULL);
        count += (int)((bottom_right_neighbors>> bit) & 1ULL);

        int cell = (int)((cur >> bit) & 1ULL);
        // A cell becomes live if it has exactly 3 neighbors,
        // or if it is already live and has exactly 2 neighbors.
        int new_cell = (count == 3) || (cell && (count == 2));
        next_word |= ((std::uint64_t)new_cell << bit);
    }
    output[idx] = next_word;
}

// Host function that runs one simulation step of Conway’s Game of Life on a CUDA device.
// The grid is bit‐packed so that each std::uint64_t holds 64 cells. The grid is assumed
// to be square (grid_dimensions x grid_dimensions), and grid_dimensions is a power of 2.
// The input and output pointers refer to GPU memory allocated via cudaMalloc.
// Any host–device synchronization is assumed to be handled by the caller.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64–bit words per row.
    int words_per_row = grid_dimensions / 64;

    // Choose a 2D block configuration. A 16x16 block is a reasonable choice for modern GPUs.
    dim3 blockDim(16, 16);
    // The grid dimensions are computed from the number of words in a row and the number of rows.
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel. The simulation step is run on the GPU.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions, words_per_row);
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
