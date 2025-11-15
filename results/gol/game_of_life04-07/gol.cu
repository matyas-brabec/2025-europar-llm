// Conway's Game of Life - CUDA implementation
//
// This implementation operates on a bit-packed representation of the grid,
// where each std::uint64_t encodes 64 consecutive cells in a row (LSB is
// the leftmost cell of the word). Each CUDA thread processes exactly one
// 64-bit word, computing the next generation for its 64 cells.
//
// Strategy:
//   - No shared or texture memory is used; all accesses are global loads/stores.
//   - For each word, we load up to 9 neighboring words (3x3 block) from the
//     rows above, current, and below.
//   - For bits 1..62 in the word: all neighbor cells reside within the three
//     words at the same column index (above, center, below). Their neighborhoods
//     are computed using bit-shifts and masked extraction of 3-bit segments.
//   - Bits 0 and 63 of each word require special handling because some of their
//     neighbors may reside in words to the left or right (aboveLeft, left,
//     belowLeft, aboveRight, right, belowRight).
//   - Neighbor counts are computed using the __popc intrinsic on an 8-bit mask
//     collecting the 8 neighbors of the current cell.
//   - Rule application (survival/birth/death) is implemented branchlessly using
//     boolean arithmetic to minimize branch divergence.
//
// Assumptions:
//   - grid_dimensions is the width and height of the square grid in cells.
//   - grid_dimensions is a power of two, >= 512.
//   - grid_dimensions is divisible by 64 (implied by power-of-two >= 512).
//   - input and output are device pointers allocated with cudaMalloc.
//   - Caller handles synchronization (cudaDeviceSynchronize) if needed.

#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel: compute one Game of Life step for a bit-packed grid.
// Each thread processes one 64-bit word (64 cells) in the grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,           // grid dimension in cells (width == height)
                                    int words_per_row)      // grid_dim / 64
{
    int global_word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = words_per_row * grid_dim;

    if (global_word_idx >= total_words)
        return;

    // Compute 2D coordinates of this word in the grid of words
    int row = global_word_idx / words_per_row;       // 0 .. grid_dim-1
    int col = global_word_idx - row * words_per_row; // 0 .. words_per_row-1

    // Load center word and its neighbors. Outside the grid is treated as 0.
    const std::uint64_t center =
        input[global_word_idx];

    const bool has_row_above = (row > 0);
    const bool has_row_below = (row + 1 < grid_dim);
    const bool has_col_left  = (col > 0);
    const bool has_col_right = (col + 1 < words_per_row);

    const std::uint64_t above =
        has_row_above ? input[(row - 1) * words_per_row + col] : 0ull;
    const std::uint64_t below =
        has_row_below ? input[(row + 1) * words_per_row + col] : 0ull;

    const std::uint64_t left =
        has_col_left ? input[row * words_per_row + (col - 1)] : 0ull;
    const std::uint64_t right =
        has_col_right ? input[row * words_per_row + (col + 1)] : 0ull;

    const std::uint64_t above_left =
        (has_row_above && has_col_left) ? input[(row - 1) * words_per_row + (col - 1)] : 0ull;
    const std::uint64_t above_right =
        (has_row_above && has_col_right) ? input[(row - 1) * words_per_row + (col + 1)] : 0ull;
    const std::uint64_t below_left =
        (has_row_below && has_col_left) ? input[(row + 1) * words_per_row + (col - 1)] : 0ull;
    const std::uint64_t below_right =
        (has_row_below && has_col_right) ? input[(row + 1) * words_per_row + (col + 1)] : 0ull;

    std::uint64_t new_word = 0ull;

    // Helper lambda to apply Game of Life rule in a branchless way.
    auto apply_rule = [](unsigned int alive_neighbors, unsigned int self_alive) -> unsigned int {
        // A cell is alive in the next generation if:
        //   - exactly 3 neighbors are alive, or
        //   - it is currently alive and exactly 2 neighbors are alive.
        //
        // This is encoded as:
        //   will_live = (alive_neighbors == 3) | (self_alive & (alive_neighbors == 2));
        unsigned int is3 = (alive_neighbors == 3);
        unsigned int is2 = (alive_neighbors == 2);
        unsigned int will_live = is3 | (self_alive & is2);
        return will_live;
    };

    // Handle bit 0 (LSB) - requires neighbors from "left" words for west/NW/SW.
    {
        unsigned int neighbors_mask = 0u;

        // Assign each of the 8 neighbors to a unique bit in neighbors_mask.
        // Bit positions (for clarity):
        //   0: NW, 1: N, 2: NE, 3: W, 4: E, 5: SW, 6: S, 7: SE
        neighbors_mask |= static_cast<unsigned int>((above_left >> 63) & 1ull) << 0; // NW
        neighbors_mask |= static_cast<unsigned int>((above       >>  0) & 1ull) << 1; // N
        neighbors_mask |= static_cast<unsigned int>((above       >>  1) & 1ull) << 2; // NE

        neighbors_mask |= static_cast<unsigned int>((left        >> 63) & 1ull) << 3; // W
        neighbors_mask |= static_cast<unsigned int>((center      >>  1) & 1ull) << 4; // E

        neighbors_mask |= static_cast<unsigned int>((below_left  >> 63) & 1ull) << 5; // SW
        neighbors_mask |= static_cast<unsigned int>((below       >>  0) & 1ull) << 6; // S
        neighbors_mask |= static_cast<unsigned int>((below       >>  1) & 1ull) << 7; // SE

        unsigned int alive_neighbors = static_cast<unsigned int>(__popc(neighbors_mask));
        unsigned int self_alive = static_cast<unsigned int>(center & 1ull);

        unsigned int will_live = apply_rule(alive_neighbors, self_alive);
        new_word |= (static_cast<std::uint64_t>(will_live) << 0);
    }

    // Handle bits 1..62 - all neighbors are within the three words:
    //   above, center, below (same column index). This avoids dependency
    //   on left/right words and simplifies address calculations markedly.
#pragma unroll
    for (int b = 1; b <= 62; ++b)
    {
        // neighbors_mask layout:
        //   bit 0: NW  (above bit b-1)
        //   bit 1: N   (above bit b)
        //   bit 2: NE  (above bit b+1)
        //   bit 3: W   (center bit b-1)
        //   bit 4: E   (center bit b+1)
        //   bit 5: SW  (below bit b-1)
        //   bit 6: S   (below bit b)
        //   bit 7: SE  (below bit b+1)
        unsigned int neighbors_mask = 0u;

        // Extract 3-bit segments from above and below at [b-1, b, b+1].
        std::uint64_t above_seg = (above >> (b - 1)) & 0x7ull;
        std::uint64_t below_seg = (below >> (b - 1)) & 0x7ull;

        neighbors_mask |= static_cast<unsigned int>(above_seg) << 0;                // NW,N,NE -> bits 0..2
        neighbors_mask |= static_cast<unsigned int>((center >> (b - 1)) & 1ull) << 3; // W       -> bit 3
        neighbors_mask |= static_cast<unsigned int>((center >> (b + 1)) & 1ull) << 4; // E       -> bit 4
        neighbors_mask |= static_cast<unsigned int>(below_seg) << 5;                // SW,S,SE -> bits 5..7

        unsigned int alive_neighbors = static_cast<unsigned int>(__popc(neighbors_mask));
        unsigned int self_alive = static_cast<unsigned int>((center >> b) & 1ull);

        unsigned int will_live = apply_rule(alive_neighbors, self_alive);
        new_word |= (static_cast<std::uint64_t>(will_live) << b);
    }

    // Handle bit 63 (MSB) - requires neighbors from "right" words for east/NE/SE.
    {
        unsigned int neighbors_mask = 0u;

        // Bit positions match the mapping used above.
        neighbors_mask |= static_cast<unsigned int>((above       >> 62) & 1ull) << 0; // NW
        neighbors_mask |= static_cast<unsigned int>((above       >> 63) & 1ull) << 1; // N
        neighbors_mask |= static_cast<unsigned int>((above_right >>  0) & 1ull) << 2; // NE

        neighbors_mask |= static_cast<unsigned int>((center      >> 62) & 1ull) << 3; // W
        neighbors_mask |= static_cast<unsigned int>((right       >>  0) & 1ull) << 4; // E

        neighbors_mask |= static_cast<unsigned int>((below       >> 62) & 1ull) << 5; // SW
        neighbors_mask |= static_cast<unsigned int>((below       >> 63) & 1ull) << 6; // S
        neighbors_mask |= static_cast<unsigned int>((below_right >>  0) & 1ull) << 7; // SE

        unsigned int alive_neighbors = static_cast<unsigned int>(__popc(neighbors_mask));
        unsigned int self_alive = static_cast<unsigned int>((center >> 63) & 1ull);

        unsigned int will_live = apply_rule(alive_neighbors, self_alive);
        new_word |= (static_cast<std::uint64_t>(will_live) << 63);
    }

    // Store the computed word to the output grid.
    output[global_word_idx] = new_word;
}

// Host function: run one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   - input:  device pointer to the current grid (bit-packed, std::uint64_t).
//   - output: device pointer where the next grid state will be stored.
//   - grid_dimensions: number of cells in each dimension (square grid).
//
// The caller is responsible for:
//   - Ensuring input and output are allocated with cudaMalloc.
//   - Ensuring grid_dimensions is a power of two, >= 512, and divisible by 64.
//   - Performing any required synchronization (e.g., cudaDeviceSynchronize).
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;    // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    // Use a 1D launch configuration with enough threads to cover all words.
    // Block size 256 is a good default for modern NVIDIA GPUs (A100, H100).
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (total_words + BLOCK_SIZE - 1) / BLOCK_SIZE;

    game_of_life_kernel<<<grid_size, BLOCK_SIZE>>>(input, output,
                                                   grid_dimensions,
                                                   words_per_row);
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
