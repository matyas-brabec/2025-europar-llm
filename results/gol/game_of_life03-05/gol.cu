#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
 * High-performance CUDA implementation of one step of Conway's Game of Life
 * on a bit-packed square grid.
 *
 * Each std::uint64_t encodes 64 horizontally adjacent cells in a row.
 * Bit 0 (LSB) is the leftmost cell in the word; bit 63 is the rightmost.
 *
 * - The grid is grid_dim x grid_dim cells (grid_dim is a power of 2, > 512).
 * - words_per_row = grid_dim / 64.
 * - The total number of 64-bit words is grid_dim * words_per_row.
 *
 * Each CUDA thread processes exactly one 64-bit word (64 cells) and writes
 * one 64-bit word of output. This avoids the need for atomics.
 *
 * Boundary handling:
 * - Cells outside the grid are treated as dead (0).
 * - For bit 0 of a word, neighbors to the left are in the three words to the
 *   immediate left (previous word in the current row and the two adjacent rows).
 * - For bit 63 of a word, neighbors to the right are in the three words to the
 *   immediate right (next word in the current row and the two adjacent rows).
 *
 * We avoid shared and texture memory as requested. All neighbor accesses are
 * done directly from global memory, which is coalesced given the data layout.
 */

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = grid_dim * words_per_row;
    if (idx >= total_words) {
        return;
    }

    const int row = idx / words_per_row;
    const int col = idx - row * words_per_row;

    const std::uint64_t* current_row =
        input + static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row);

    // Central word for this thread
    const std::uint64_t mid = current_row[col];

    // Neighbor words; initialize to 0 to naturally handle boundaries.
    std::uint64_t left          = 0;
    std::uint64_t right         = 0;
    std::uint64_t top           = 0;
    std::uint64_t top_left      = 0;
    std::uint64_t top_right     = 0;
    std::uint64_t bottom        = 0;
    std::uint64_t bottom_left   = 0;
    std::uint64_t bottom_right  = 0;

    // Horizontal neighbors in the same row
    if (col > 0) {
        left = current_row[col - 1];
    }
    if (col + 1 < words_per_row) {
        right = current_row[col + 1];
    }

    // Row above (if any)
    if (row > 0) {
        const std::uint64_t* above_row =
            input + static_cast<std::size_t>(row - 1) * static_cast<std::size_t>(words_per_row);
        top = above_row[col];
        if (col > 0) {
            top_left = above_row[col - 1];
        }
        if (col + 1 < words_per_row) {
            top_right = above_row[col + 1];
        }
    }

    // Row below (if any)
    if (row + 1 < grid_dim) {
        const std::uint64_t* below_row =
            input + static_cast<std::size_t>(row + 1) * static_cast<std::size_t>(words_per_row);
        bottom = below_row[col];
        if (col > 0) {
            bottom_left = below_row[col - 1];
        }
        if (col + 1 < words_per_row) {
            bottom_right = below_row[col + 1];
        }
    }

    // Precompute horizontally shifted neighbor words for each of the three rows.
    // For a given bit position i in the result word:
    //
    // - top_center[i]      : above neighbor (row-1, same column)
    // - top_left_word[i]   : above-left neighbor (row-1, column-1)
    // - top_right_word[i]  : above-right neighbor (row-1, column+1)
    // - mid_left[i]        : left neighbor (row, column-1)
    // - mid_right[i]       : right neighbor (row, column+1)
    // - bottom_center[i]   : below neighbor (row+1, same column)
    // - bottom_left_word[i]: below-left neighbor (row+1, column-1)
    // - bottom_right_word[i]: below-right neighbor (row+1, column+1)
    //
    // The shifts propagate neighbor bits across 64-bit word boundaries by
    // mixing in bits from the left/right words. Missing neighbors (outside
    // the grid) are already 0 due to initialization above.
    constexpr int WORD_BITS = 64;

    const std::uint64_t mid_center     = mid;
    const std::uint64_t top_center     = top;
    const std::uint64_t bottom_center  = bottom;

    const std::uint64_t mid_left_word  = (mid_center << 1) | (left >> (WORD_BITS - 1));
    const std::uint64_t mid_right_word = (mid_center >> 1) | (right << (WORD_BITS - 1));

    const std::uint64_t top_left_word  = (top_center << 1) | (top_left >> (WORD_BITS - 1));
    const std::uint64_t top_right_word = (top_center >> 1) | (top_right << (WORD_BITS - 1));

    const std::uint64_t bottom_left_word  = (bottom_center << 1) | (bottom_left >> (WORD_BITS - 1));
    const std::uint64_t bottom_right_word = (bottom_center >> 1) | (bottom_right << (WORD_BITS - 1));

    // Compute next generation bits for this 64-cell segment.
    std::uint64_t result = 0;
    std::uint64_t mask = 1ull;

    // The loop is explicitly unrolled to let the compiler generate straight-line
    // code for all 64 bits, minimizing loop overhead and enabling aggressive
    // instruction scheduling.
#pragma unroll 64
    for (int bit = 0; bit < WORD_BITS; ++bit) {
        // Count live neighbors at this bit position by sampling the eight
        // corresponding bits from the neighbor words.
        unsigned int neighbors = 0;
        neighbors += (top_left_word      & mask) ? 1u : 0u;
        neighbors += (top_center         & mask) ? 1u : 0u;
        neighbors += (top_right_word     & mask) ? 1u : 0u;
        neighbors += (mid_left_word      & mask) ? 1u : 0u;
        neighbors += (mid_right_word     & mask) ? 1u : 0u;
        neighbors += (bottom_left_word   & mask) ? 1u : 0u;
        neighbors += (bottom_center      & mask) ? 1u : 0u;
        neighbors += (bottom_right_word  & mask) ? 1u : 0u;

        // Current cell state at this bit
        const std::uint64_t cell_alive = (mid_center & mask) ? 1ull : 0ull;

        // Apply Conway's Game of Life rules in a branchless manner:
        // - A live cell survives if it has exactly 2 live neighbors.
        // - A cell (dead or alive) becomes alive if it has exactly 3 neighbors.
        const std::uint64_t stay_alive    = cell_alive & (neighbors == 2u);
        const std::uint64_t become_alive  = (neighbors == 3u) ? 1ull : 0ull;

        const std::uint64_t new_cell_bit = (stay_alive | become_alive) << bit;

        result |= new_cell_bit;
        mask <<= 1;
    }

    output[idx] = result;
}

/*
 * Host-side wrapper that launches one simulation step.
 *
 * Parameters:
 *   input           - device pointer to input grid (bit-packed).
 *   output          - device pointer to output grid (bit-packed).
 *   grid_dimensions - width/height of the square grid in cells.
 *
 * Requirements:
 *   - grid_dimensions is a power of 2, > 512.
 *   - grid_dimensions is divisible by 64 (since we pack 64 cells per word).
 *   - input and output point to device memory allocated with cudaMalloc and
 *     have at least (grid_dimensions * grid_dimensions / 64) elements.
 *   - No synchronization is performed here; the caller is responsible for
 *     any required cudaDeviceSynchronize() or stream synchronization.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words   = grid_dimensions * words_per_row;

    // Reasonable default launch configuration for modern data-center GPUs
    // such as A100/H100; tuned for high occupancy and throughput.
    constexpr int threads_per_block = 256;
    const int blocks =
        (total_words + threads_per_block - 1) / threads_per_block;

    game_of_life_kernel<<<blocks, threads_per_block>>>(
        input, output, grid_dimensions, words_per_row);
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
