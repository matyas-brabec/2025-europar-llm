#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a square grid.
 *
 * Representation:
 * - The grid is grid_dimensions x grid_dimensions cells.
 * - Each cell is 0 (dead) or 1 (alive).
 * - The grid is stored in row-major order, bit-packed:
 *     * Each std::uint64_t word encodes 64 consecutive cells in a row.
 *     * Bit 0 of a word is the left-most cell of that 64-cell segment.
 *     * Bit 63 is the right-most cell of that segment.
 * - Outside-grid cells are treated as dead (0).
 *
 * Parallelization strategy:
 * - Each CUDA thread processes exactly one 64-bit word (64 cells in one row).
 * - This eliminates the need for atomic operations.
 * - Threads are arranged so that x-dimension spans words within a row,
 *   and y-dimension spans rows. This yields coalesced memory accesses.
 *
 * Neighbor handling:
 * - For a word at (row, word_index), we may need up to 8 neighboring words:
 *     UL, U, UR   (row-1)
 *     L,  C, R    (row)
 *     DL, D, DR   (row+1)
 *   where C is the current word.
 * - For interior bits (1..62), all neighbors lie within the three vertically
 *   aligned words: U, C, D (no need to touch left/right words).
 * - For bit 0, we additionally use UL, L, DL.
 * - For bit 63, we additionally use UR, R, DR.
 * - Any neighbor that would fall outside the grid is treated as 0 by
 *   substituting the corresponding word with 0.
 *
 * Performance notes:
 * - For bits 1..62 in each word, we use a sliding-window scheme over U, C, D
 *   that shifts these three 64-bit registers by one bit per iteration. This
 *   avoids recomputing large shifts for each bit.
 * - For each cell, we pack the 8 neighbors into the low 8 bits of a 32-bit
 *   integer and use __popc() to count neighbors efficiently.
 * - The next state for each bit is computed branchlessly and written into
 *   a result 64-bit word, which is finally stored to global memory.
 */

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Determine this thread's word coordinates.
    const int word_x = blockIdx.x * blockDim.x + threadIdx.x; // word index within row
    const int row    = blockIdx.y;                            // row index

    if (row >= grid_dim || word_x >= words_per_row)
        return;

    // Compute base indices for this row and neighboring rows.
    const std::size_t row_offset      = static_cast<std::size_t>(row) * words_per_row;
    const std::size_t row_above_off   = (row > 0) ? row_offset - words_per_row : 0;
    const std::size_t row_below_off   = (row + 1 < grid_dim) ? row_offset + words_per_row : 0;
    const bool has_row_above          = (row > 0);
    const bool has_row_below          = (row + 1 < grid_dim);
    const bool has_word_left          = (word_x > 0);
    const bool has_word_right         = (word_x + 1 < words_per_row);

    // Load center word.
    const std::uint64_t C = input[row_offset + word_x];

    // Load neighbor words, using 0 where neighbors fall outside the grid.
    std::uint64_t U  = 0;
    std::uint64_t UL = 0;
    std::uint64_t UR = 0;
    std::uint64_t D  = 0;
    std::uint64_t DL = 0;
    std::uint64_t DR = 0;
    std::uint64_t L  = 0;
    std::uint64_t R  = 0;

    if (has_row_above) {
        const std::size_t base = row_above_off;
        U  = input[base + word_x];
        if (has_word_left)  UL = input[base + word_x - 1];
        if (has_word_right) UR = input[base + word_x + 1];
    }

    if (has_row_below) {
        const std::size_t base = row_below_off;
        D  = input[base + word_x];
        if (has_word_left)  DL = input[base + word_x - 1];
        if (has_word_right) DR = input[base + word_x + 1];
    }

    if (has_word_left)  L = input[row_offset + word_x - 1];
    if (has_word_right) R = input[row_offset + word_x + 1];

    std::uint64_t result_word = 0;

    // --- Handle bit 0 (left-most bit) with special neighbor handling. ---
    {
        std::uint32_t neighbors = 0;

        // Top row neighbors: NW, N, NE
        neighbors |= (static_cast<std::uint32_t>((UL >> 63) & 1ull) << 0); // NW
        neighbors |= (static_cast<std::uint32_t>((U  >>  0) & 1ull) << 1); // N
        neighbors |= (static_cast<std::uint32_t>((U  >>  1) & 1ull) << 2); // NE

        // Middle row neighbors: W, E
        neighbors |= (static_cast<std::uint32_t>((L  >> 63) & 1ull) << 3); // W
        neighbors |= (static_cast<std::uint32_t>((C  >>  1) & 1ull) << 4); // E

        // Bottom row neighbors: SW, S, SE
        neighbors |= (static_cast<std::uint32_t>((DL >> 63) & 1ull) << 5); // SW
        neighbors |= (static_cast<std::uint32_t>((D  >>  0) & 1ull) << 6); // S
        neighbors |= (static_cast<std::uint32_t>((D  >>  1) & 1ull) << 7); // SE

        const std::uint32_t neighbor_count = static_cast<std::uint32_t>(__popc(neighbors));
        const std::uint32_t center_bit     = static_cast<std::uint32_t>((C >> 0) & 1ull);

        // Game of Life rule:
        // next_alive = (neighbor_count == 3) || (center_bit && neighbor_count == 2)
        const std::uint32_t next_alive =
            (neighbor_count == 3u) |
            (center_bit & (neighbor_count == 2u));

        result_word |= (static_cast<std::uint64_t>(next_alive) << 0);
    }

    // --- Handle bits 1..62 using a sliding-window over U, C, D. ---
    {
        // up_shift, mid_shift, down_shift are aligned so that:
        //  - For current bit index "bit", up_shift = U >> (bit - 1)
        //  - mid_shift = C >> (bit - 1)
        //  - down_shift = D >> (bit - 1)
        //
        // This alignment lets us extract NW, N, NE, W, center, E, SW, S, SE
        // with just simple masks and small constant shifts.
        std::uint64_t up_shift  = U;
        std::uint64_t mid_shift = C;
        std::uint64_t down_shift = D;

        // Process bits 1 through 62.
        #pragma unroll
        for (int bit = 1; bit <= 62; ++bit) {
            std::uint32_t neighbors = 0;

            const std::uint64_t u_s = up_shift;   // U >> (bit - 1)
            const std::uint64_t c_s = mid_shift;  // C >> (bit - 1)
            const std::uint64_t d_s = down_shift; // D >> (bit - 1)

            // Top row: NW, N, NE in bits 0..2 of neighbors.
            neighbors |= static_cast<std::uint32_t>(u_s & 0x7ull);

            // Middle row: W (bit 3), E (bit 4) from c_s.
            neighbors |= static_cast<std::uint32_t>((c_s & 0x1ull) << 3);         // W
            neighbors |= static_cast<std::uint32_t>(((c_s >> 2) & 0x1ull) << 4);  // E

            // Bottom row: SW, S, SE in bits 5..7 of neighbors.
            neighbors |= static_cast<std::uint32_t>((d_s & 0x7ull) << 5);

            const std::uint32_t neighbor_count = static_cast<std::uint32_t>(__popc(neighbors));

            // Center bit for this position is at bit 1 of c_s.
            const std::uint32_t center_bit = static_cast<std::uint32_t>((c_s >> 1) & 0x1ull);

            const std::uint32_t next_alive =
                (neighbor_count == 3u) |
                (center_bit & (neighbor_count == 2u));

            result_word |= (static_cast<std::uint64_t>(next_alive) << bit);

            // Advance sliding window for next bit.
            up_shift   >>= 1;
            mid_shift  >>= 1;
            down_shift >>= 1;
        }
    }

    // --- Handle bit 63 (right-most bit) with special neighbor handling. ---
    {
        std::uint32_t neighbors = 0;

        // Top row neighbors: NW, N, NE
        neighbors |= (static_cast<std::uint32_t>((U  >> 62) & 1ull) << 0); // NW
        neighbors |= (static_cast<std::uint32_t>((U  >> 63) & 1ull) << 1); // N
        neighbors |= (static_cast<std::uint32_t>((UR >>  0) & 1ull) << 2); // NE

        // Middle row neighbors: W, E
        neighbors |= (static_cast<std::uint32_t>((C  >> 62) & 1ull) << 3); // W
        neighbors |= (static_cast<std::uint32_t>((R  >>  0) & 1ull) << 4); // E

        // Bottom row neighbors: SW, S, SE
        neighbors |= (static_cast<std::uint32_t>((D  >> 62) & 1ull) << 5); // SW
        neighbors |= (static_cast<std::uint32_t>((D  >> 63) & 1ull) << 6); // S
        neighbors |= (static_cast<std::uint32_t>((DR >>  0) & 1ull) << 7); // SE

        const std::uint32_t neighbor_count = static_cast<std::uint32_t>(__popc(neighbors));
        const std::uint32_t center_bit     = static_cast<std::uint32_t>((C >> 63) & 1ull);

        const std::uint32_t next_alive =
            (neighbor_count == 3u) |
            (center_bit & (neighbor_count == 2u));

        result_word |= (static_cast<std::uint64_t>(next_alive) << 63);
    }

    // Store the resulting 64-bit word (next generation) to global memory.
    output[row_offset + word_x] = result_word;
}

/*
 * Host function: run one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 * - input: pointer to device memory holding the current grid, bit-packed
 *          as described above (grid_dimensions x grid_dimensions cells).
 * - output: pointer to device memory where the next grid state will be stored
 *           (same layout as input).
 * - grid_dimensions: width and height of the square grid (power of 2 > 512).
 *
 * Assumptions:
 * - input and output are allocated with cudaMalloc and have enough space
 *   for grid_dimensions * grid_dimensions bits.
 * - All necessary synchronization (e.g., cudaDeviceSynchronize) is handled
 *   by the caller; this function only launches the kernel.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each row contains grid_dimensions cells; at 64 cells per word:
    const int words_per_row = grid_dimensions / 64;

    // Configure a 2D grid:
    // - x-dimension: words within a row
    // - y-dimension: rows
    //
    // Choose a moderate block size for good occupancy. 256 threads per block
    // in x-dimension is usually a good starting point on modern GPUs.
    const int threads_per_block_x = 256;
    const int blocks_x = (words_per_row + threads_per_block_x - 1) / threads_per_block_x;

    dim3 block_dim(threads_per_block_x, 1, 1);
    dim3 grid_dim(blocks_x, grid_dimensions, 1);

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
