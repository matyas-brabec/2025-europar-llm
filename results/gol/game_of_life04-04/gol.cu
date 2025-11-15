// CUDA implementation of a single step of Conway's Game of Life on a bit-packed grid.
// Each std::uint64_t word encodes 64 cells in a row (bit 0 = least-significant).
//
// Design notes:
// - Each CUDA thread processes exactly one 64-bit word (64 cells).
// - The grid is square (grid_dimensions x grid_dimensions), power of two, and >= 512.
// - Each row has grid_dimensions / 64 words.
// - Neighbor cells outside the grid are treated as dead (0).
// - We avoid shared/texture memory as requested; rely on global memory and L1/L2 caching.
// - For each word, we load its 3x3 neighborhood in terms of words (up to 9 words).
// - Using those 9 words, we construct 8 "neighbor bitboards", each 64 bits wide,
//   that, for every bit position, contain one of the 8 neighbor directions.
// - We then iterate over the 64 bits of the word, shifting all neighbor bitboards
//   right by 1 each iteration. The least-significant bit (LSB) of each bitboard at
//   iteration `bit` corresponds to the `bit`-th cell's neighbor in that direction.
// - For each bit, we pack the 8 neighbor bits into an 8-bit integer and use __popc()
//   to count neighbors quickly.
// - Game-of-Life rules are then applied per bit, and the result is written back
//   into a 64-bit output word.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

/**
 * CUDA kernel: compute one Game of Life step on a bit-packed grid.
 *
 * @param input           Pointer to input grid (device memory), bit-packed (uint64_t).
 * @param output          Pointer to output grid (device memory), same layout as input.
 * @param grid_dim        Side length of the square grid, in cells (power of two).
 * @param words_per_row   Number of 64-bit words in each row (grid_dim / 64).
 */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;  // word index within the row
    int row    = blockIdx.y * blockDim.y + threadIdx.y;  // row index (in cells/words)

    if (row >= grid_dim || word_x >= words_per_row)
        return;

    const std::uint64_t* in  = input;
    std::uint64_t*       out = output;

    const int row_offset   = row * words_per_row;
    const int center_index = row_offset + word_x;

    // Load the current word (center word).
    std::uint64_t c_mid = in[center_index];

    // Load words from the same row: left and right neighbors (if they exist).
    std::uint64_t c_left  = 0;
    std::uint64_t c_right = 0;
    if (word_x > 0) {
        c_left = in[center_index - 1];
    }
    if (word_x + 1 < words_per_row) {
        c_right = in[center_index + 1];
    }

    // Load the three words from the row above (a_left, a_mid, a_right).
    std::uint64_t a_left = 0, a_mid = 0, a_right = 0;
    if (row > 0) {
        const int above_offset = row_offset - words_per_row;
        if (word_x > 0) {
            a_left = in[above_offset + word_x - 1];
        }
        a_mid = in[above_offset + word_x];
        if (word_x + 1 < words_per_row) {
            a_right = in[above_offset + word_x + 1];
        }
    }

    // Load the three words from the row below (b_left, b_mid, b_right).
    std::uint64_t b_left = 0, b_mid = 0, b_right = 0;
    if (row + 1 < grid_dim) {
        const int below_offset = row_offset + words_per_row;
        if (word_x > 0) {
            b_left = in[below_offset + word_x - 1];
        }
        b_mid = in[below_offset + word_x];
        if (word_x + 1 < words_per_row) {
            b_right = in[below_offset + word_x + 1];
        }
    }

    // Construct neighbor bitboards by horizontally shifting each row.
    //
    // For a given row R (a_mid, c_mid, b_mid) and its left/right neighbor words R_left, R_right:
    // - left_neighbors  = (R << 1) | (R_left >> 63)
    //   -> for each bit position i, left_neighbors[i] = cell at column (col-1).
    // - right_neighbors = (R >> 1) | (R_right << 63)
    //   -> for each bit position i, right_neighbors[i] = cell at column (col+1).
    //
    // When there is no neighbor word to the left/right (grid boundary), we treat R_left/R_right as 0,
    // which correctly yields dead neighbors outside the grid.
    const std::uint64_t above_left_neighbors  = (a_mid << 1) | (a_left  >> 63);
    const std::uint64_t above_right_neighbors = (a_mid >> 1) | (a_right << 63);

    const std::uint64_t same_left_neighbors   = (c_mid << 1) | (c_left  >> 63);
    const std::uint64_t same_right_neighbors  = (c_mid >> 1) | (c_right << 63);

    const std::uint64_t below_left_neighbors  = (b_mid << 1) | (b_left  >> 63);
    const std::uint64_t below_right_neighbors = (b_mid >> 1) | (b_right << 63);

    // Neighbor bitboards for all 8 neighbor directions, aligned so that bit `i` corresponds to the
    // neighbor of the cell represented by bit `i` in the center word.
    //
    // Direction mapping:
    // - n0: above-left
    // - n1: directly above
    // - n2: above-right
    // - n3: left
    // - n4: right
    // - n5: below-left
    // - n6: directly below
    // - n7: below-right
    std::uint64_t n0 = above_left_neighbors;
    std::uint64_t n1 = a_mid;
    std::uint64_t n2 = above_right_neighbors;
    std::uint64_t n3 = same_left_neighbors;
    std::uint64_t n4 = same_right_neighbors;
    std::uint64_t n5 = below_left_neighbors;
    std::uint64_t n6 = b_mid;
    std::uint64_t n7 = below_right_neighbors;

    // Copy of the center word for extracting the current cell bit.
    std::uint64_t cur = c_mid;

    // Result word for this thread.
    std::uint64_t out_word = 0;

    // Iterate over all 64 bits/cells in this word.
    // At iteration `bit`, the LSB of each n0..n7 holds the corresponding neighbor bit for that cell.
    // After processing a bit, shift all bitboards right by 1 to align the next cell's neighbors.
#pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        // Pack the 8 neighbor bits (one from each direction) into an 8-bit integer.
        // Each (nX & 1ull) yields either 0 or 1.
        unsigned int neighbor_bits =
            (unsigned int)(n0 & 1ull) |
            ((unsigned int)(n1 & 1ull) << 1) |
            ((unsigned int)(n2 & 1ull) << 2) |
            ((unsigned int)(n3 & 1ull) << 3) |
            ((unsigned int)(n4 & 1ull) << 4) |
            ((unsigned int)(n5 & 1ull) << 5) |
            ((unsigned int)(n6 & 1ull) << 6) |
            ((unsigned int)(n7 & 1ull) << 7);

        // Count neighbor bits using the fast hardware popcount.
        unsigned int neighbor_count = (unsigned int)__popc(neighbor_bits);

        // Current cell: 0 (dead) or 1 (alive).
        unsigned int center_alive = (unsigned int)(cur & 1ull);

        // Apply Conway's Game of Life rules:
        // - Any live cell with fewer than 2 live neighbors dies (underpopulation).
        // - Any live cell with 2 or 3 live neighbors survives.
        // - Any live cell with more than 3 live neighbors dies (overpopulation).
        // - Any dead cell with exactly 3 live neighbors becomes a live cell (reproduction).
        //
        // Implemented branchlessly with boolean arithmetic:
        // next_alive = (neighbor_count == 3) OR (center_alive AND neighbor_count == 2)
        unsigned int count_is_3  = (neighbor_count == 3u);
        unsigned int count_is_2  = (neighbor_count == 2u);
        unsigned int next_alive  = count_is_3 | (center_alive & count_is_2);

        // Set the corresponding bit in the output word.
        out_word |= (static_cast<std::uint64_t>(next_alive) << bit);

        // Shift all neighbor bitboards and the center word right by 1 bit to prepare
        // for the next cell (next bit position).
        n0 >>= 1;
        n1 >>= 1;
        n2 >>= 1;
        n3 >>= 1;
        n4 >>= 1;
        n5 >>= 1;
        n6 >>= 1;
        n7 >>= 1;
        cur >>= 1;
    }

    // Write the computed word to the output grid.
    out[center_index] = out_word;
}

} // unnamed namespace

/**
 * Host function: execute one step of Conway's Game of Life on the GPU.
 *
 * @param input            Device pointer to the input grid (std::uint64_t bit-packed).
 * @param output           Device pointer to the output grid (std::uint64_t bit-packed).
 * @param grid_dimensions  Width and height of the square grid in cells (power of two).
 *
 * The input and output pointers must point to device memory allocated with cudaMalloc.
 * This function launches the CUDA kernel but does not synchronize the device; any
 * required synchronization should be performed by the caller.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each 64-bit word encodes 64 cells.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Choose a 2D block configuration to ensure good occupancy and coalesced accesses.
    // - blockDim.x: horizontal dimension in words
    // - blockDim.y: vertical dimension in rows
    //
    // Here we use 32x8 = 256 threads per block, which is usually a good balance on
    // modern NVIDIA data-center GPUs (A100/H100).
    dim3 block_dim(32, 8);
    dim3 grid_dim(
        (words_per_row   + block_dim.x - 1) / block_dim.x,
        (grid_dimensions + block_dim.y - 1) / block_dim.y);

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, grid_dimensions, words_per_row);
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
