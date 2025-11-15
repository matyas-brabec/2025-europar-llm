#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Grid representation:
 *   - The grid is square: grid_dimensions x grid_dimensions cells.
 *   - Each cell is a single bit: 1 = alive, 0 = dead.
 *   - The grid is stored row-major, bit-packed into 64-bit words.
 *     Thus, each row contains (grid_dimensions / 64) 64-bit words.
 *
 * Thread mapping:
 *   - Each CUDA thread processes one 64-bit word (i.e., 64 cells in the same row).
 *   - For each word, a thread reads up to 9 neighboring words:
 *       [north-west] [north] [north-east]
 *       [west]       [self ] [east     ]
 *       [south-west] [south] [south-east]
 *     These words cover all possible neighbors for any bit in the central word.
 *   - Boundary conditions:
 *       * Cells outside the grid are considered dead.
 *       * Therefore, neighbors that lie outside the grid are treated as 0.
 *
 * Per-bit neighbor computation:
 *   - For each bit position b in [0, 63] of the central word:
 *       self = (center >> b) & 1
 *       neighbors = sum of the 8 neighboring bits:
 *         N, S (vertical neighbors)
 *         W, E (horizontal neighbors)
 *         NW, NE, SW, SE (diagonals)
 *
 *   - For interior bits (1 ≤ b ≤ 62), horizontal and diagonal neighbors reside
 *     in the same words (center, north, south) with bit indices (b-1) or (b+1).
 *   - For bit 0 and bit 63, neighbors cross 64-bit word boundaries:
 *       * bit 0:
 *           W, NW, SW reside in west, north-west, south-west words at bit 63.
 *           E, NE, SE reside in center, north, south words at bit 1.
 *       * bit 63:
 *           W, NW, SW reside in center, north, south words at bit 62.
 *           E, NE, SE reside in east, north-east, south-east words at bit 0.
 *
 * Game of Life rules:
 *   - Let neighbors = number of alive neighbors (0..8)
 *   - Let self = current cell state (0 or 1)
 *   - Next state:
 *       alive_next = (neighbors == 3) || (self == 1 && neighbors == 2)
 *
 * Implementation details for performance:
 *   - All neighbor checks are done via shifts and bitwise AND (& 1ULL),
 *     avoiding branches per bit.
 *   - The rule is applied in a branchless form:
 *       next = (neighbors == 3) | (self & (neighbors == 2));
 *   - The per-word bit loop is fully unrolled (#pragma unroll 64) so that
 *     the compiler can optimize conditions for bit 0 and 63 at compile time.
 *   - No shared or texture memory is used; all reads are from global memory.
 */

namespace {
    // Number of bits in a 64-bit word (for clarity and self-documentation).
    constexpr int BITS_PER_WORD = 64;
}

/**
 * CUDA kernel implementing one step of Conway's Game of Life for a bit-packed grid.
 *
 * @param input           Device pointer to input grid (bit-packed, read-only).
 * @param output          Device pointer to output grid (bit-packed, write-only).
 * @param grid_dim_cells  Width/height of the square grid in cells (power of 2).
 * @param words_per_row   Number of 64-bit words per row: grid_dim_cells / 64.
 * @param total_words     Total number of 64-bit words in the grid: grid_dim_cells * words_per_row.
 */
__global__ void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int grid_dim_cells,
    int words_per_row,
    int total_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) {
        return;
    }

    // Determine row and column (in word units) for this thread.
    int row = idx / words_per_row;          // row index in [0, grid_dim_cells-1]
    int col = idx - row * words_per_row;    // word column index in [0, words_per_row-1]

    // Load the central word.
    std::uint64_t center = input[idx];

    // Pre-load neighboring words, using 0 for out-of-bound neighbors.
    // These words cover all potential neighbors for all 64 bits in "center".
    std::uint64_t north     = (row > 0)                          ? input[idx - words_per_row]         : 0ull;
    std::uint64_t south     = (row + 1 < grid_dim_cells)         ? input[idx + words_per_row]         : 0ull;
    std::uint64_t west      = (col > 0)                          ? input[idx - 1]                     : 0ull;
    std::uint64_t east      = (col + 1 < words_per_row)          ? input[idx + 1]                     : 0ull;
    std::uint64_t northWest = (row > 0 && col > 0)               ? input[idx - words_per_row - 1]     : 0ull;
    std::uint64_t northEast = (row > 0 && col + 1 < words_per_row)
                                                                ? input[idx - words_per_row + 1]     : 0ull;
    std::uint64_t southWest = (row + 1 < grid_dim_cells && col > 0)
                                                                ? input[idx + words_per_row - 1]     : 0ull;
    std::uint64_t southEast = (row + 1 < grid_dim_cells && col + 1 < words_per_row)
                                                                ? input[idx + words_per_row + 1]     : 0ull;

    std::uint64_t result = 0ull;

    // Process each bit (cell) in this word.
    // The loop is unrolled to let the compiler optimize bit-specific branches.
    #pragma unroll 64
    for (int bit = 0; bit < BITS_PER_WORD; ++bit) {
        // Count alive neighbors.
        unsigned int neighbors = 0u;

        // Vertical neighbors: North and South.
        neighbors += static_cast<unsigned int>((north >> bit) & 1ull);
        neighbors += static_cast<unsigned int>((south >> bit) & 1ull);

        // West, North-West, South-West neighbors.
        if (bit > 0) {
            unsigned int shift1 = static_cast<unsigned int>(bit - 1);
            neighbors += static_cast<unsigned int>((center >> shift1) & 1ull); // W
            neighbors += static_cast<unsigned int>((north  >> shift1) & 1ull); // NW
            neighbors += static_cast<unsigned int>((south  >> shift1) & 1ull); // SW
        } else {
            // bit == 0: neighbors to the left come from the "west" column of words.
            neighbors += static_cast<unsigned int>((west      >> 63) & 1ull); // W
            neighbors += static_cast<unsigned int>((northWest >> 63) & 1ull); // NW
            neighbors += static_cast<unsigned int>((southWest >> 63) & 1ull); // SW
        }

        // East, North-East, South-East neighbors.
        if (bit < 63) {
            unsigned int shift1 = static_cast<unsigned int>(bit + 1);
            neighbors += static_cast<unsigned int>((center >> shift1) & 1ull); // E
            neighbors += static_cast<unsigned int>((north  >> shift1) & 1ull); // NE
            neighbors += static_cast<unsigned int>((south  >> shift1) & 1ull); // SE
        } else {
            // bit == 63: neighbors to the right come from the "east" column of words.
            neighbors += static_cast<unsigned int>( east      & 1ull); // E (bit 0)
            neighbors += static_cast<unsigned int>( northEast & 1ull); // NE (bit 0)
            neighbors += static_cast<unsigned int>( southEast & 1ull); // SE (bit 0)
        }

        // Current cell state.
        unsigned int self = static_cast<unsigned int>((center >> bit) & 1ull);

        // Apply Game of Life rules in a branchless way:
        // next = (neighbors == 3) || (self == 1 && neighbors == 2)
        unsigned int next =
            (neighbors == 3u) |
            (self & (neighbors == 2u));

        // Set the corresponding bit in the result word.
        result |= (static_cast<std::uint64_t>(next) << bit);
    }

    // Write the computed next-generation word to output.
    output[idx] = result;
}

/**
 * Host function that executes one Game of Life step on the GPU.
 *
 * @param input            Device pointer to the current grid state (bit-packed).
 * @param output           Device pointer to the next grid state (bit-packed).
 * @param grid_dimensions  Width/height of the square grid in cells.
 *
 * Preconditions:
 *   - grid_dimensions is a power of 2, >= 512.
 *   - grid_dimensions is a multiple of 64 (true for 2^n with n >= 6).
 *   - input and output point to device memory allocated via cudaMalloc.
 *   - The caller handles any host-device synchronization.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row.
    const int words_per_row = grid_dimensions / BITS_PER_WORD;
    // Total number of 64-bit words in the grid.
    const int total_words   = words_per_row * grid_dimensions;

    // Choose a reasonable block size for H100/A100.
    const int block_size = 256;
    const int grid_size  = (total_words + block_size - 1) / block_size;

    // Launch kernel: one thread per 64-bit word.
    game_of_life_kernel<<<grid_size, block_size>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
        total_words
    );

    // No cudaDeviceSynchronize() here; the caller is responsible for synchronization.
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
