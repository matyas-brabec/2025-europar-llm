#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel computes one Game of Life step on a bit‐packed grid.
// Each thread processes one 64‐bit word (64 cells) in the grid. The grid is stored
// in row-major order, with every uint64_t representing 64 consecutive cells in a row.
// Boundary cells (neighbors outside the grid) are considered dead.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim, int words_per_row)
{
    // Compute the word index in the current row.
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the row index.
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds: each thread corresponds to one 64-bit word.
    if (y >= grid_dim || word_x >= words_per_row) {
        return;
    }

    // Compute index in the bit-packed grid.
    int idx = y * words_per_row + word_x;

    // Precompute boundary availability for current block.
    bool has_left  = (word_x > 0);
    bool has_right = (word_x < words_per_row - 1);
    bool has_prev  = (y > 0);
    bool has_next  = (y < grid_dim - 1);

    // Load current row's words.
    std::uint64_t curr       = input[idx];
    std::uint64_t curr_left  = has_left  ? input[y * words_per_row + (word_x - 1)] : 0;
    std::uint64_t curr_right = has_right ? input[y * words_per_row + (word_x + 1)] : 0;

    // Load previous row's words (if available).
    std::uint64_t prev       = has_prev ? input[(y - 1) * words_per_row + word_x] : 0;
    std::uint64_t prev_left  = (has_prev && has_left)  ? input[(y - 1) * words_per_row + (word_x - 1)] : 0;
    std::uint64_t prev_right = (has_prev && has_right) ? input[(y - 1) * words_per_row + (word_x + 1)] : 0;

    // Load next row's words (if available).
    std::uint64_t next       = has_next ? input[(y + 1) * words_per_row + word_x] : 0;
    std::uint64_t next_left  = (has_next && has_left)  ? input[(y + 1) * words_per_row + (word_x - 1)] : 0;
    std::uint64_t next_right = (has_next && has_right) ? input[(y + 1) * words_per_row + (word_x + 1)] : 0;

    std::uint64_t new_word = 0;

    // Loop over each bit (cell) in the 64-bit word.
    // The loop is unrolled to help the compiler optimize the constant 64 iterations.
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        int count = 0;

        // Accumulate live neighbor counts from row above.
        if (has_prev) {
            // Left neighbor in the row above.
            if (i == 0)
                count += (has_left) ? (int)((prev_left >> 63) & 1ULL) : 0;
            else
                count += (int)((prev >> (i - 1)) & 1ULL);
            // Center neighbor in the row above.
            count += (int)((prev >> i) & 1ULL);
            // Right neighbor in the row above.
            if (i == 63)
                count += (has_right) ? (int)(prev_right & 1ULL) : 0;
            else
                count += (int)((prev >> (i + 1)) & 1ULL);
        }

        // Accumulate live neighbor counts from the current row (skip the center cell).
        if (i == 0)
            count += (has_left) ? (int)((curr_left >> 63) & 1ULL) : 0;
        else
            count += (int)((curr >> (i - 1)) & 1ULL);
        if (i == 63)
            count += (has_right) ? (int)(curr_right & 1ULL) : 0;
        else
            count += (int)((curr >> (i + 1)) & 1ULL);

        // Accumulate live neighbor counts from the row below.
        if (has_next) {
            if (i == 0)
                count += (has_left) ? (int)((next_left >> 63) & 1ULL) : 0;
            else
                count += (int)((next >> (i - 1)) & 1ULL);
            count += (int)((next >> i) & 1ULL);
            if (i == 63)
                count += (has_right) ? (int)(next_right & 1ULL) : 0;
            else
                count += (int)((next >> (i + 1)) & 1ULL);
        }

        // Determine the state of the current cell.
        int cell = (int)((curr >> i) & 1ULL);
        int new_cell = 0;
        // Apply Conway's Game of Life rules:
        // - A live cell survives if it has 2 or 3 live neighbors.
        // - A dead cell becomes live if it has exactly 3 live neighbors.
        if (cell) {
            new_cell = ((count == 2) || (count == 3)) ? 1 : 0;
        } else {
            new_cell = (count == 3) ? 1 : 0;
        }
        // Set the corresponding bit in the output word.
        new_word |= (static_cast<std::uint64_t>(new_cell) << i);
    }

    // Write the computed word to the output grid.
    output[idx] = new_word;
}

// This host function launches one step of Conway's Game of Life by calling the CUDA kernel.
// The grid is bit-packed; each uint64_t represents 64 consecutive cells in a row.
// grid_dimensions is the width (and height) of the square grid.
// It is assumed that grid_dimensions is a power of 2 and at least 512.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each row contains grid_dimensions cells, packed into 64-bit words.
    int words_per_row = grid_dimensions / 64;

    // Choose a block size that yields good occupancy on modern GPUs.
    // Here we choose a 32x8 block, which can be tuned per GPU.
    dim3 blockDim(32, 8);
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel. All host-device synchronizations are handled by the caller.
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
