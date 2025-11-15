#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Representation:
 * - The grid is N x N cells, where N is a power of 2 and N >= 512.
 * - Each row is stored as N/64 64-bit words.
 * - Bit i (0 = least significant) of a word represents one cell in that row.
 * - Outside-grid cells are treated as dead (0).
 *
 * Kernel strategy:
 * - Each thread processes one 64-bit word (64 cells) at a time, with a grid-stride loop.
 * - For the word at (row, col_word), the thread loads up to 9 words:
 *   - From the current row: left, center, right (c_w, c_c, c_e)
 *   - From the row above:   left, center, right (n_w, n_c, n_e)
 *   - From the row below:   left, center, right (s_w, s_c, s_e)
 *   Missing neighbors outside the grid are treated as 0.
 *
 * Neighbor computation:
 * - For each of the three rows (above, same, below), we compute bitboards:
 *   - west   = row << 1 | (left_word >> 63)   (horizontal west neighbors)
 *   - east   = row >> 1 | (right_word << 63)  (horizontal east neighbors)
 *   - center = row                             (vertical neighbors for above/below rows)
 *
 * - This gives 8 bitboards, one per neighbor direction:
 *   - NW, N, NE, W, E, SW, S, SE
 *
 * Neighbor counting (bit-parallel, modulo 8):
 * - Each bit position corresponds to one cell. Its eight neighbors are represented by
 *   the 8 bitboards above.
 * - We compute the neighbor count per cell modulo 8 using a 3-bit accumulator per bit:
 *   - nb0: least significant bit of count
 *   - nb1: second bit
 *   - nb2: third bit
 *
 * - We add each neighbor bitboard into (nb0, nb1, nb2) using a bitwise 3-bit ripple adder:
 *
 *   function add_bitboard(x):
 *     carry0 = nb0 & x; nb0 ^= x
 *     carry1 = nb1 & carry0; nb1 ^= carry0
 *     carry2 = nb2 & carry1; nb2 ^= carry1
 *     // carry2 is discarded (modulo 8 arithmetic)
 *
 * - Since each neighbor bit is 0 or 1, and we only have 8 neighbors, counts ∈ [0, 8].
 *   We only care whether the count is 2 or 3. Representing the count modulo 8 is sufficient:
 *   - 2 ≡ 2 (010b)
 *   - 3 ≡ 3 (011b)
 *   - 8 ≡ 0 (000b), which is not 2 or 3, so classification remains correct.
 *
 * Applying Game of Life rules:
 * - Let c_c be the current word for the row (current cell states).
 * - After accumulating neighbors:
 *   - count == 3: (~nb2) & nb1 &  nb0
 *   - count == 2: (~nb2) & nb1 & ~nb0
 *
 * - New state bitboard:
 *   new = (count == 3) | ((count == 2) & c_c)
 *
 * This kernel avoids per-cell loops and uses only bitwise operations and shifts,
 * giving good performance on modern GPUs without needing shared memory.
 */

namespace {

// Add one neighbor bitboard `x` into the 3-bit per-cell accumulator (nb0, nb1, nb2),
// performing addition modulo 8 independently for each bit position.
__device__ __forceinline__
void add_bitboard(std::uint64_t x,
                  std::uint64_t &nb0,
                  std::uint64_t &nb1,
                  std::uint64_t &nb2)
{
    // First bit (LSB) addition: nb0 = nb0 + x
    std::uint64_t carry0 = nb0 & x;
    nb0 ^= x;

    // Second bit addition: nb1 = nb1 + carry0
    std::uint64_t carry1 = nb1 & carry0;
    nb1 ^= carry0;

    // Third bit addition: nb2 = nb2 + carry1
    std::uint64_t carry2 = nb2 & carry1;
    nb2 ^= carry1;

    // carry2 would be the 4th bit (8's place) and is discarded (modulo 8).
    (void)carry2;
}

// Kernel that computes one Game of Life step on a bit-packed N x N grid.
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row,
                         int log2_words_per_row)
{
    const std::uint64_t total_words =
        static_cast<std::uint64_t>(grid_dim) *
        static_cast<std::uint64_t>(words_per_row);

    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t idx =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += stride)
    {
        // Map 1D word index to (row, col_word) using power-of-two decomposition:
        const int row = static_cast<int>(idx >> log2_words_per_row);
        const int col = static_cast<int>(idx &
                           (static_cast<std::uint64_t>(words_per_row) - 1ULL));

        const std::uint64_t zero = 0ULL;

        const std::uint64_t row_base =
            static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(words_per_row);

        // Current row words: left, center, right
        std::uint64_t c_w = zero;
        std::uint64_t c_c = input[row_base + col];
        std::uint64_t c_e = zero;

        if (col > 0) {
            c_w = input[row_base + (col - 1)];
        }
        if (col + 1 < words_per_row) {
            c_e = input[row_base + (col + 1)];
        }

        // North row words: left, center, right (if row > 0)
        std::uint64_t n_w = zero, n_c = zero, n_e = zero;
        if (row > 0) {
            const std::uint64_t n_base = row_base - static_cast<std::uint64_t>(words_per_row);
            n_c = input[n_base + col];
            if (col > 0) {
                n_w = input[n_base + (col - 1)];
            }
            if (col + 1 < words_per_row) {
                n_e = input[n_base + (col + 1)];
            }
        }

        // South row words: left, center, right (if row + 1 < grid_dim)
        std::uint64_t s_w = zero, s_c = zero, s_e = zero;
        if (row + 1 < grid_dim) {
            const std::uint64_t s_base = row_base + static_cast<std::uint64_t>(words_per_row);
            s_c = input[s_base + col];
            if (col > 0) {
                s_w = input[s_base + (col - 1)];
            }
            if (col + 1 < words_per_row) {
                s_e = input[s_base + (col + 1)];
            }
        }

        // Neighbor bitboards from the north row
        const std::uint64_t n_center = n_c;                        // N
        const std::uint64_t n_west   = (n_c << 1) | (n_w >> 63);   // NW
        const std::uint64_t n_east   = (n_c >> 1) | (n_e << 63);   // NE

        // Neighbor bitboards from the current row (W, E)
        const std::uint64_t c_west   = (c_c << 1) | (c_w >> 63);   // W
        const std::uint64_t c_east   = (c_c >> 1) | (c_e << 63);   // E

        // Neighbor bitboards from the south row
        const std::uint64_t s_center = s_c;                        // S
        const std::uint64_t s_west   = (s_c << 1) | (s_w >> 63);   // SW
        const std::uint64_t s_east   = (s_c >> 1) | (s_e << 63);   // SE

        // Accumulate neighbor counts modulo 8 for each bit position
        std::uint64_t nb0 = 0ULL; // LSB of neighbor count
        std::uint64_t nb1 = 0ULL; // second bit
        std::uint64_t nb2 = 0ULL; // third bit

        add_bitboard(n_west,   nb0, nb1, nb2);
        add_bitboard(n_center, nb0, nb1, nb2);
        add_bitboard(n_east,   nb0, nb1, nb2);
        add_bitboard(c_west,   nb0, nb1, nb2);
        add_bitboard(c_east,   nb0, nb1, nb2);
        add_bitboard(s_west,   nb0, nb1, nb2);
        add_bitboard(s_center, nb0, nb1, nb2);
        add_bitboard(s_east,   nb0, nb1, nb2);

        // Determine where neighbor count is exactly 2 or 3:
        // count == 3 -> 0b011: nb2=0, nb1=1, nb0=1
        // count == 2 -> 0b010: nb2=0, nb1=1, nb0=0
        const std::uint64_t not_nb2   = ~nb2;
        const std::uint64_t count_is3 = not_nb2 & nb1 &  nb0;
        const std::uint64_t count_is2 = not_nb2 & nb1 & ~nb0;

        // Apply Game of Life rules:
        // New cell is alive if:
        // - it has exactly 3 neighbors (birth), or
        // - it is currently alive and has exactly 2 neighbors (survival).
        const std::uint64_t result = count_is3 | (count_is2 & c_c);

        output[idx] = result;
    }
}

} // anonymous namespace

// Host function that launches one Game of Life step on the GPU.
// `input` and `output` are device pointers (cudaMalloc'ed) to bit-packed grids.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int grid_dim = grid_dimensions;

    // Each row has grid_dim / 64 words; grid_dim is guaranteed to be a power of 2,
    // so this division by 64 is exact.
    const int words_per_row = grid_dim >> 6;

    // Compute log2(words_per_row) on the host (words_per_row is a power of 2).
    int log2_words_per_row = 0;
    int tmp = words_per_row;
    while (tmp > 1) {
        tmp >>= 1;
        ++log2_words_per_row;
    }

    const std::uint64_t total_words =
        static_cast<std::uint64_t>(grid_dim) *
        static_cast<std::uint64_t>(words_per_row);

    // Launch configuration: 256 threads per block, enough blocks to cover the grid.
    const int block_size = 256;
    int grid_size = static_cast<int>((total_words + block_size - 1ULL) / block_size);

    // Launch the kernel. Host-device synchronization, if needed, is handled by the caller.
    game_of_life_kernel<<<grid_size, block_size>>>(
        input,
        output,
        grid_dim,
        words_per_row,
        log2_words_per_row);
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
