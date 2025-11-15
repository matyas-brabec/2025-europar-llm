// Conway's Game of Life on a bit-packed grid using CUDA.
// Each std::uint64_t word stores 64 consecutive cells in a row (one bit per cell).
// Each CUDA thread processes one 64-bit word and computes the next-generation
// state of its 64 cells.
//
// Key implementation details:
// - The grid is square with dimension grid_dimensions x grid_dimensions.
//   grid_dimensions is a power of 2 and >= 512, and divisible by 64.
// - words_per_row = grid_dimensions / 64
// - Total words = words_per_row * grid_dimensions
// - The cell at (row, col_bit) lives in word index:
//     word_index = row * words_per_row + (col_bit / 64)
//   and bit index within that word = col_bit % 64.
//
// Neighbor counting strategy:
// - For bits 1..62 within a word, all 8 neighbors live in the same word in
//   the same column block, or in the words above/below in the same column.
//   They never cross word boundaries horizontally. Therefore, for bits 1..62,
//   we only need three 64-bit words:
//     n (row-1), c (row), s (row+1) for the same column.
//   For each bit "b" (1 <= b <= 62), we extract three 3-bit fields:
//     from n: bits [b-1, b, b+1] => NW, N, NE
//     from c: bits [b-1, b, b+1] => W,  C, E
//     from s: bits [b-1, b, b+1] => SW, S, SE
//   These 9 bits form a 3x3 neighborhood around the cell; a __popc on this
//   9-bit mask minus the center bit gives the neighbor count.
// - Bits 0 and 63 of each word may have neighbors crossing to the left/right
//   word, so they are handled explicitly using the 8 surrounding words:
//     nw, n, ne
//     w,  c, e
//     sw, s, se
//   All boundary conditions (outside the grid) are handled by treating
//   missing neighbor words as 0 (dead cells).
//
// Performance notes:
// - Only global memory is used; no shared or texture memory.
// - The inner loop over bits 1..62 uses simple bit-field extracts plus
//   __popc, which is efficient on modern NVIDIA GPUs.
// - Branches inside the bit loop are avoided; neighbor sums are computed
//   in a branchless manner.

#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel: process one 64-bit word (64 cells) per thread.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    const long long global_word_index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    const long long total_words =
        static_cast<long long>(words_per_row) * grid_dimensions;

    if (global_word_index >= total_words) {
        return;
    }

    const int row = static_cast<int>(global_word_index / words_per_row);
    const int col = static_cast<int>(global_word_index - row * words_per_row);
    const int last_row = grid_dimensions - 1;

    // Load the 9 relevant words, treating out-of-bounds as 0 (dead).
    const std::uint64_t c  = input[global_word_index];                                      // center
    const std::uint64_t w  = (col > 0)                         ? input[global_word_index - 1]           : 0ULL;
    const std::uint64_t e  = (col + 1 < words_per_row)         ? input[global_word_index + 1]           : 0ULL;
    const std::uint64_t n  = (row > 0)                         ? input[global_word_index - words_per_row] : 0ULL;
    const std::uint64_t s  = (row < last_row)                  ? input[global_word_index + words_per_row] : 0ULL;
    const std::uint64_t nw = (row > 0 && col > 0)
                             ? input[global_word_index - words_per_row - 1] : 0ULL;
    const std::uint64_t ne = (row > 0 && col + 1 < words_per_row)
                             ? input[global_word_index - words_per_row + 1] : 0ULL;
    const std::uint64_t sw = (row < last_row && col > 0)
                             ? input[global_word_index + words_per_row - 1] : 0ULL;
    const std::uint64_t se = (row < last_row && col + 1 < words_per_row)
                             ? input[global_word_index + words_per_row + 1] : 0ULL;

    std::uint64_t result = 0ULL;

    // Helper lambda to apply Conway's rules, returning 0 or 1.
    auto apply_rules = [] __device__ (int alive, int neighbors) -> std::uint64_t {
        // Alive in next generation if:
        // - exactly 3 neighbors, or
        // - currently alive and exactly 2 neighbors.
        const int survive = (neighbors == 3) | (alive & (neighbors == 2));
        return static_cast<std::uint64_t>(survive);
    };

    // ----- Bit 0 (may depend on left neighbor words) -----
    {
        // Build a 9-bit mask (NW..SE) in bits [0..8].
        //   bits 0..2: NW, N, NE
        //   bits 3..5: W,  C, E
        //   bits 6..8: SW, S, SE
        std::uint32_t mask = 0;

        mask |= static_cast<std::uint32_t>((nw >> 63) & 1ULL) << 0; // NW
        mask |= static_cast<std::uint32_t>((n  >> 0)  & 1ULL) << 1; // N
        mask |= static_cast<std::uint32_t>((n  >> 1)  & 1ULL) << 2; // NE

        mask |= static_cast<std::uint32_t>((w  >> 63) & 1ULL) << 3; // W
        mask |= static_cast<std::uint32_t>((c  >> 0)  & 1ULL) << 4; // C
        mask |= static_cast<std::uint32_t>((c  >> 1)  & 1ULL) << 5; // E

        mask |= static_cast<std::uint32_t>((sw >> 63) & 1ULL) << 6; // SW
        mask |= static_cast<std::uint32_t>((s  >> 0)  & 1ULL) << 7; // S
        mask |= static_cast<std::uint32_t>((s  >> 1)  & 1ULL) << 8; // SE

        const int center_alive = (mask >> 4) & 1;
        const int neighbors    = __popc(mask) - center_alive;

        const std::uint64_t next_bit = apply_rules(center_alive, neighbors);
        result |= next_bit << 0;
    }

    // ----- Bits 1..62 (no horizontal word-crossing) -----
    // For these bits, all neighbors are in the same column block
    // (words: n, c, s) and don't require left/right words.
    #pragma unroll
    for (int bit = 1; bit <= 62; ++bit) {
        // Extract 3 consecutive bits (bit-1, bit, bit+1) from each row.
        // These correspond to:
        //   from n: NW, N, NE
        //   from c: W,  C, E
        //   from s: SW, S, SE
        const std::uint32_t top3    = static_cast<std::uint32_t>((n >> (bit - 1)) & 0x7ULL);
        const std::uint32_t middle3 = static_cast<std::uint32_t>((c >> (bit - 1)) & 0x7ULL);
        const std::uint32_t bottom3 = static_cast<std::uint32_t>((s >> (bit - 1)) & 0x7ULL);

        std::uint32_t mask = 0;
        mask |= top3;              // bits 0..2: NW, N, NE
        mask |= middle3 << 3;      // bits 3..5: W,  C, E
        mask |= bottom3 << 6;      // bits 6..8: SW, S, SE

        const int center_alive = (middle3 >> 1) & 1; // middle bit of middle3
        const int neighbors    = __popc(mask) - center_alive;

        const std::uint64_t next_bit = apply_rules(center_alive, neighbors);
        result |= next_bit << bit;
    }

    // ----- Bit 63 (may depend on right neighbor words) -----
    {
        std::uint32_t mask = 0;

        mask |= static_cast<std::uint32_t>((n  >> 62) & 1ULL) << 0; // NW
        mask |= static_cast<std::uint32_t>((n  >> 63) & 1ULL) << 1; // N
        mask |= static_cast<std::uint32_t>((ne >> 0)  & 1ULL) << 2; // NE

        mask |= static_cast<std::uint32_t>((c  >> 62) & 1ULL) << 3; // W
        mask |= static_cast<std::uint32_t>((c  >> 63) & 1ULL) << 4; // C
        mask |= static_cast<std::uint32_t>((e  >> 0)  & 1ULL) << 5; // E

        mask |= static_cast<std::uint32_t>((s  >> 62) & 1ULL) << 6; // SW
        mask |= static_cast<std::uint32_t>((s  >> 63) & 1ULL) << 7; // S
        mask |= static_cast<std::uint32_t>((se >> 0)  & 1ULL) << 8; // SE

        const int center_alive = (mask >> 4) & 1;
        const int neighbors    = __popc(mask) - center_alive;

        const std::uint64_t next_bit = apply_rules(center_alive, neighbors);
        result |= next_bit << 63;
    }

    // Store the 64 updated cells for this word.
    output[global_word_index] = result;
}

// Host function: run a single Game of Life step on the GPU.
//
// Parameters:
//   input  - device pointer to the current grid (bit-packed std::uint64_t[*])
//   output - device pointer to the next grid (bit-packed std::uint64_t[*])
//   grid_dimensions - width/height of the square grid; power of 2, >= 512
//
// Notes:
//   - input and output must be allocated with cudaMalloc and large enough
//     to hold grid_dimensions * grid_dimensions bits each.
//   - No synchronization is performed here; the caller is responsible for
//     any required cudaDeviceSynchronize or CUDA event handling.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const long long total_words =
        static_cast<long long>(words_per_row) * grid_dimensions;

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
