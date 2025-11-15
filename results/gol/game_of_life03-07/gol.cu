#include <cstdint>
#include <cuda_runtime.h>

using std::uint64_t;

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed
 * square grid.
 *
 * Memory layout:
 *   - The logical grid is N x N cells (N == grid_dimensions, power of 2).
 *   - Each row is stored as N / 64 contiguous 64-bit words.
 *   - Within a word, bit i (0 <= i < 64) represents the cell at column
 *     (word_index * 64 + i) in that row.
 *   - A bit value of 1 means alive, 0 means dead.
 *
 * Kernel mapping:
 *   - Each CUDA thread is responsible for computing the next state of a single
 *     64-bit word (i.e., 64 cells) in the grid.
 *   - Threads are organized in a 2D grid:
 *       x-dimension: word index within a row.
 *       y-dimension: row index.
 *
 * Neighborhood handling:
 *   - For a given word at (row, word_col), we load up to 9 words:
 *       current row:     left, center, right
 *       row above:       left, center, right
 *       row below:       left, center, right
 *     Missing words (outside the grid boundaries) are treated as all zeros
 *     to model dead cells outside the grid.
 *
 *   - From these 9 words we construct 8 "neighbor bitboards" for the current
 *     64 cells: N, S, E, W, NE, NW, SE, SW. Each bitboard is a 64-bit value
 *     where bit i corresponds to one of the 8 neighbors of the cell at bit i
 *     in the center word.
 *
 *   - Horizontal neighbors with word-boundary handling:
 *       - East neighbors:
 *           E = (center >> 1) | (right << 63)
 *         For bits 0..62, E.bit[i] = center.bit[i+1].
 *         For bit 63, E.bit[63] = right.bit[0] (or 0 if no right word).
 *
 *       - West neighbors:
 *           W = (center << 1) | (left >> 63)
 *         For bits 1..63, W.bit[i] = center.bit[i-1].
 *         For bit 0,  W.bit[0]  = left.bit[63] (or 0 if no left word).
 *
 *   - Diagonal neighbors use the same pattern applied to the north/south rows:
 *       NW = (N << 1) | (north_left  >> 63)
 *       NE = (N >> 1) | (north_right << 63)
 *       SW = (S << 1) | (south_left  >> 63)
 *       SE = (S >> 1) | (south_right << 63)
 *
 * Neighbor counting and rule application:
 *   - For each of the 64 bits in the center word, we need the sum of the 8
 *     neighbors (0..8). Instead of extracting each neighbor bit independently
 *     with shifts-by-variable amounts, we process all words bit-by-bit in a
 *     loop, shifting the bitboards right by 1 each iteration:
 *
 *       - On iteration b (0 <= b < 64), the least significant bit (LSB) of each
 *         neighbor bitboard corresponds to the neighbor of cell b in the center
 *         word. After processing, we shift all bitboards right by 1, so the
 *         next LSBs correspond to the next cell.
 *
 *   - For each bit:
 *       center = center_mask & 1
 *       neighbors = sum of LSBs of eight neighbor bitboards
 *
 *       Life rule:
 *         - A live cell survives if neighbors == 2 or 3.
 *         - A dead cell becomes live if neighbors == 3.
 *
 *       Implemented branchlessly as:
 *         alive_next = (neighbors == 3) | (center & (neighbors == 2))
 *
 * Performance considerations:
 *   - Each thread performs only 9 global loads (9 x 8 bytes).
 *   - All arithmetic is on 64-bit integers and small 32-bit counters.
 *   - No atomics, no shared or texture memory; all neighbor handling uses
 *     simple bit operations and local registers.
 */

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    // Number of 64-bit words per row: grid_dimensions is guaranteed
    // to be a power of two and thus divisible by 64.
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // 2D index: word column within the row, and row index.
    const int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row      = blockIdx.y * blockDim.y + threadIdx.y;

    // Out-of-bounds threads do nothing.
    if (row >= grid_dimensions || word_col >= words_per_row) {
        return;
    }

    const int row_offset = row * words_per_row;
    const std::uint64_t* row_ptr = input + row_offset;

    // Load center word and horizontal neighbors (same row).
    const std::uint64_t center = row_ptr[word_col];

    const std::uint64_t left   = (word_col > 0)
                                 ? row_ptr[word_col - 1]
                                 : 0ull;
    const std::uint64_t right  = (word_col + 1 < words_per_row)
                                 ? row_ptr[word_col + 1]
                                 : 0ull;

    // Load words from the row above (north) and below (south), with boundary checks.
    std::uint64_t north_center = 0ull;
    std::uint64_t north_left   = 0ull;
    std::uint64_t north_right  = 0ull;

    if (row > 0) {
        const std::uint64_t* north_row = input + (row - 1) * words_per_row;
        north_center = north_row[word_col];
        north_left   = (word_col > 0) ? north_row[word_col - 1] : 0ull;
        north_right  = (word_col + 1 < words_per_row) ? north_row[word_col + 1] : 0ull;
    }

    std::uint64_t south_center = 0ull;
    std::uint64_t south_left   = 0ull;
    std::uint64_t south_right  = 0ull;

    if (row + 1 < grid_dimensions) {
        const std::uint64_t* south_row = input + (row + 1) * words_per_row;
        south_center = south_row[word_col];
        south_left   = (word_col > 0) ? south_row[word_col - 1] : 0ull;
        south_right  = (word_col + 1 < words_per_row) ? south_row[word_col + 1] : 0ull;
    }

    // Construct neighbor bitboards for the 64 cells in the center word.
    //
    // Vertical neighbors are directly aligned.
    const std::uint64_t N = north_center;
    const std::uint64_t S = south_center;

    // Horizontal neighbors in the same row, with cross-word handling.
    const std::uint64_t E = (center >> 1) | (right << 63);
    const std::uint64_t W = (center << 1) | (left  >> 63);

    // Diagonal neighbors.
    const std::uint64_t NE = (N >> 1) | (north_right << 63);
    const std::uint64_t NW = (N << 1) | (north_left  >> 63);
    const std::uint64_t SE = (S >> 1) | (south_right << 63);
    const std::uint64_t SW = (S << 1) | (south_left  >> 63);

    // Prepare masks for bit-by-bit processing. On iteration b, the LSB of each
    // mask corresponds to the neighbor of cell bit b in the center word.
    std::uint64_t center_mask = center;
    std::uint64_t n_mask      = N;
    std::uint64_t s_mask      = S;
    std::uint64_t e_mask      = E;
    std::uint64_t w_mask      = W;
    std::uint64_t ne_mask     = NE;
    std::uint64_t nw_mask     = NW;
    std::uint64_t se_mask     = SE;
    std::uint64_t sw_mask     = SW;

    std::uint64_t next_word = 0ull;

    // Process all 64 bits in the word. Unroll for better ILP and performance.
#pragma unroll 64
    for (int bit = 0; bit < 64; ++bit) {
        // Current cell state: 0 or 1.
        const unsigned center_bit = static_cast<unsigned>(center_mask & 1ull);

        // Sum of the 8 neighbor bits for this cell.
        unsigned neighbors =
            static_cast<unsigned>(n_mask  & 1ull) +
            static_cast<unsigned>(s_mask  & 1ull) +
            static_cast<unsigned>(e_mask  & 1ull) +
            static_cast<unsigned>(w_mask  & 1ull) +
            static_cast<unsigned>(ne_mask & 1ull) +
            static_cast<unsigned>(nw_mask & 1ull) +
            static_cast<unsigned>(se_mask & 1ull) +
            static_cast<unsigned>(sw_mask & 1ull);

        // Apply Conway's Game of Life rules in a branchless manner:
        // - Birth: neighbors == 3
        // - Survival: center_bit == 1 and neighbors == 2
        const unsigned is3        = (neighbors == 3u);
        const unsigned is2        = (neighbors == 2u);
        const unsigned alive_next = is3 | (is2 & center_bit);

        // Set bit 'bit' in next_word if the cell is alive in the next generation.
        next_word |= (static_cast<std::uint64_t>(alive_next) << bit);

        // Shift masks so that the next LSB corresponds to the next cell.
        center_mask >>= 1;
        n_mask      >>= 1;
        s_mask      >>= 1;
        e_mask      >>= 1;
        w_mask      >>= 1;
        ne_mask     >>= 1;
        nw_mask     >>= 1;
        se_mask     >>= 1;
        sw_mask     >>= 1;
    }

    // Write the result for this 64-cell block.
    output[row_offset + word_col] = next_word;
}

/*
 * Host API: execute one Game of Life step on a square grid.
 *
 * Parameters:
 *   - input:  device pointer to the input grid (bit-packed, read-only).
 *   - output: device pointer to the output grid (bit-packed).
 *   - grid_dimensions: width and height of the square grid in cells.
 *
 * Requirements:
 *   - grid_dimensions is a power of 2 greater than 512.
 *   - grid_dimensions is divisible by 64 (guaranteed by the above).
 *   - input and output are allocated in device memory via cudaMalloc, with
 *     at least (grid_dimensions * grid_dimensions / 64) uint64_t elements.
 *   - No synchronization is performed here; the caller is responsible for
 *     cudaDeviceSynchronize() if needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int cells_per_side = grid_dimensions;
    const int words_per_row  = cells_per_side >> 6; // divide by 64

    // Use a 2D launch configuration. Each thread handles one 64-bit word.
    // 256 threads per block is a good default for modern NVIDIA GPUs.
    const dim3 block_dim(32, 8);
    const dim3 grid_dim(
        (words_per_row  + block_dim.x - 1) / block_dim.x,
        (cells_per_side + block_dim.y - 1) / block_dim.y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, grid_dimensions);
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
