#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA kernel implementing one step of Conway's Game of Life on a bit-packed grid.
 *
 * Grid representation:
 * - The grid is square with dimension grid_dim x grid_dim.
 * - Each row is stored as (grid_dim / 64) 64-bit words.
 * - In each std::uint64_t word, bit 0 represents the left-most cell of the 64-cell segment,
 *   bit 63 the right-most.
 *
 * Thread mapping:
 * - Each thread processes one 64-bit word (64 cells) for a single row.
 * - Threads are assigned linearly over all words: tid in [0, totalWords).
 *
 * Neighbor computation:
 * - For a given word in row r and word index w, the thread loads up to 9 words:
 *     prev_left,   prev_center,   prev_right
 *     cur_left,    cur_center,    cur_right
 *     next_left,   next_center,   next_right
 *   where "prev" is row r-1, "cur" is row r, "next" is row r+1 and "left/center/right"
 *   are column-wise neighbors (word indices w-1, w, w+1).
 * - Words outside the grid boundaries are treated as zero (dead cells).
 *
 * Bitfield construction:
 * - From these 9 words we derive 8 64-bit bitfields, each indicating the presence of a specific
 *   neighbor direction for each of the 64 cells in cur_center:
 *
 *     north_west, north, north_east,
 *     west,                 east,
 *     south_west, south, south_east
 *
 * - For example, "west" is computed as:
 *     west = (cur_center << 1) | (cur_left >> 63);
 *
 *   Here, for each bit position j (0..63):
 *     - The bit j of (cur_center << 1) corresponds to the cell at column (base + j - 1)
 *       within the same word, i.e. the west neighbor, except for j=0 where it would underflow.
 *     - The bit 63 of cur_left (shifted down to bit 0) fills in the missing west neighbor
 *       for the first cell in this word when a left word exists, otherwise it is 0.
 *
 *   Similar logic is applied for east and diagonals using shifts and carry-in bits from
 *   neighbor words.
 *
 * Per-cell update:
 * - For each of the 64 bits (cells) in cur_center, we:
 *     1. Count the number of live neighbors by checking the corresponding bit in each of the
 *        8 neighbor bitfields.
 *     2. Read the current cell's state from cur_center.
 *     3. Apply Conway's rules:
 *           - If count == 3, the cell becomes alive.
 *           - Else if count == 2, the cell retains its current state.
 *           - Else the cell becomes dead.
 *     4. Set/clear the corresponding bit in the output word accordingly.
 *
 * This approach keeps memory access minimal and highly coalesced, and all per-cell logic
 * is performed in registers via simple bit operations and small integer arithmetic.
 */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    const int words_per_row = grid_dim >> 6;  // grid_dim / 64, grid_dim is power-of-two >= 1024

    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * grid_dim;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= total_words) {
        return;
    }

    // Map linear word index to (row, word_idx)
    const int row = static_cast<int>(tid / words_per_row);
    const int word_idx = static_cast<int>(tid % words_per_row);

    const bool first_row = (row == 0);
    const bool last_row  = (row == grid_dim - 1);
    const bool first_word = (word_idx == 0);
    const int last_word_idx = words_per_row - 1;
    const bool last_word  = (word_idx == last_word_idx);

    const int base_idx = row * words_per_row + word_idx;

    // Load center word for this thread's 64 cells
    const std::uint64_t cur_center = input[base_idx];

    // Load neighbor words in the current row
    const std::uint64_t cur_left  = first_word ? 0ULL : input[base_idx - 1];
    const std::uint64_t cur_right = last_word  ? 0ULL : input[base_idx + 1];

    // Load neighbor words in the previous row
    const std::uint64_t prev_center = first_row ? 0ULL : input[(row - 1) * words_per_row + word_idx];
    const std::uint64_t prev_left   = (first_row || first_word) ? 0ULL
                                   : input[(row - 1) * words_per_row + (word_idx - 1)];
    const std::uint64_t prev_right  = (first_row || last_word) ? 0ULL
                                   : input[(row - 1) * words_per_row + (word_idx + 1)];

    // Load neighbor words in the next row
    const std::uint64_t next_center = last_row ? 0ULL : input[(row + 1) * words_per_row + word_idx];
    const std::uint64_t next_left   = (last_row || first_word) ? 0ULL
                                   : input[(row + 1) * words_per_row + (word_idx - 1)];
    const std::uint64_t next_right  = (last_row || last_word) ? 0ULL
                                   : input[(row + 1) * words_per_row + (word_idx + 1)];

    // Construct neighbor bitfields for each of the 8 directions.
    // These bitfields are aligned with cur_center: bit j corresponds to the neighbor
    // (if any) of the cell at bit j of cur_center in the given direction.

    const std::uint64_t north      = prev_center;
    const std::uint64_t south      = next_center;

    const std::uint64_t north_west = (prev_center << 1) | (prev_left >> 63);
    const std::uint64_t north_east = (prev_center >> 1) | ((prev_right & 1ULL) << 63);

    const std::uint64_t south_west = (next_center << 1) | (next_left >> 63);
    const std::uint64_t south_east = (next_center >> 1) | ((next_right & 1ULL) << 63);

    const std::uint64_t west       = (cur_center << 1) | (cur_left >> 63);
    const std::uint64_t east       = (cur_center >> 1) | ((cur_right & 1ULL) << 63);

    // Iterate over each bit (cell) in this word, computing neighbor counts and
    // applying the Game of Life rules.
    std::uint64_t result = 0ULL;
    std::uint64_t mask = 1ULL;

#pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        int count = 0;

        // Count alive neighbors from the 8 directions
        count += (north_west & mask) ? 1 : 0;
        count += (north      & mask) ? 1 : 0;
        count += (north_east & mask) ? 1 : 0;
        count += (west       & mask) ? 1 : 0;
        count += (east       & mask) ? 1 : 0;
        count += (south_west & mask) ? 1 : 0;
        count += (south      & mask) ? 1 : 0;
        count += (south_east & mask) ? 1 : 0;

        const bool alive = (cur_center & mask) != 0;

        // Conway's Game of Life rules:
        // - A live cell with 2 or 3 neighbors survives.
        // - A dead cell with exactly 3 neighbors becomes alive.
        // - All other cells die or remain dead.
        const bool survives = (count == 2);
        const bool born     = (count == 3);
        const bool new_alive = (alive && survives) || born;

        if (new_alive) {
            result |= mask;
        }

        mask <<= 1;
    }

    output[base_idx] = result;
}

/*
 * Host function to execute one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 * - input:  Device pointer to the input grid, bit-packed as std::uint64_t words.
 * - output: Device pointer to the output grid, same layout as input.
 * - grid_dimensions: Side length (width and height) of the square grid. It is guaranteed
 *                    to be a power of 2, greater than 512, and such that both input and
 *                    output grids fit into GPU memory.
 *
 * Assumptions:
 * - 'input' and 'output' have been allocated with cudaMalloc by the caller.
 * - Any required synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
 * - No additional host-device transfers are performed here for maximum performance.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Choose a reasonable block size for modern GPUs.
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((total_words + block_size - 1) / block_size);

    game_of_life_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
