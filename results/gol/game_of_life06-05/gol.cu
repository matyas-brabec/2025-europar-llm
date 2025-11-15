#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a square grid.
 *
 * - The grid is grid_dimensions x grid_dimensions cells, where each cell is
 *   encoded as a single bit inside a std::uint64_t word.
 * - Each word encodes 64 consecutive cells in a row (bit 0 = leftmost cell
 *   in that word, bit 63 = rightmost).
 * - input/output point to device memory allocated via cudaMalloc.
 * - Each CUDA thread processes exactly one 64-bit word (64 cells).
 *
 * Neighbor counting is done in bit-parallel fashion:
 *   - For each word, we construct 64-bit masks corresponding to the 8
 *     neighbor directions (N, S, E, W, NE, NW, SE, SW).
 *   - We then accumulate these eight bitplanes into three bitplanes
 *     (ones, twos, fours) that represent a 3-bit counter (0..7) per cell.
 *   - This uses bit-sliced full-adder logic, implemented with simple
 *     bitwise operations (XOR and AND) and no cross-bit carries.
 *   - Finally, we derive masks for neighbor counts of exactly 2 or 3 and
 *     apply the Game of Life rules in parallel for all 64 cells.
 */

/**
 * @brief Add a 1-bit-per-cell bitplane into a 3-bit-per-cell counter.
 *
 * For each bit position i, (ones[i], twos[i], fours[i]) form a 3-bit integer
 * in binary: count = ones + 2*twos + 4*fours (mod 8).
 * This function performs:
 *
 *     count_i = count_i + bit_i   (for all i in parallel)
 *
 * using standard ripple-carry adder logic, but applied independently to each
 * bit position via bitwise operations:
 *   - bit 0 (ones) is updated by XOR with the new bit.
 *   - The carry from bit 0 propagates into bit 1 (twos).
 *   - The carry from bit 1 propagates into bit 2 (fours).
 *
 * Carry out from bit 2 (i.e., transitioning from 7 to 8) is discarded, which
 * is equivalent to addition modulo 8. Since we only need to distinguish
 * neighbor counts 0..7 and 8 is irrelevant to "==2" or "==3", this is safe.
 */
__device__ __forceinline__
void add_neighbor_bitplane(std::uint64_t bits,
                           std::uint64_t &ones,
                           std::uint64_t &twos,
                           std::uint64_t &fours)
{
    // Full-adder bit 0: (ones, bits) -> new ones and carry into twos.
    std::uint64_t carry01 = ones & bits;   // majority(ones, bits, 0)
    ones ^= bits;                          // sum bit (ones XOR bits)

    // Full-adder bit 1: (twos, carry01) -> new twos and carry into fours.
    std::uint64_t carry12 = twos & carry01;
    twos ^= carry01;

    // Full-adder bit 2: (fours, carry12) -> new fours (carry to bit 3 ignored).
    fours ^= carry12;
}

/**
 * @brief CUDA kernel: compute one Game of Life step for a bit-packed grid.
 *
 * Each thread processes one 64-bit word corresponding to 64 cells of a row.
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row,
                         int words_per_row_log2,
                         int total_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) {
        return;
    }

    const int max_row = grid_dim - 1;
    const int max_col = words_per_row - 1;

    // Decode (row, colWord) from linear index.
    // words_per_row is a power of two, so we can replace division/modulo
    // by a right shift and a bitmask.
    const int row = idx >> words_per_row_log2;
    const int col = idx & (words_per_row - 1);

    const std::uint64_t center = input[idx];

    // Load neighbor words where they exist; cells outside the grid are dead.
    std::uint64_t north        = 0;
    std::uint64_t south        = 0;
    std::uint64_t west_word    = 0;
    std::uint64_t east_word    = 0;
    std::uint64_t north_west_w = 0;
    std::uint64_t north_east_w = 0;
    std::uint64_t south_west_w = 0;
    std::uint64_t south_east_w = 0;

    if (row > 0) {
        int idx_north = idx - words_per_row;
        north = input[idx_north];
        if (col > 0) {
            north_west_w = input[idx_north - 1];
        }
        if (col < max_col) {
            north_east_w = input[idx_north + 1];
        }
    }

    if (row < max_row) {
        int idx_south = idx + words_per_row;
        south = input[idx_south];
        if (col > 0) {
            south_west_w = input[idx_south - 1];
        }
        if (col < max_col) {
            south_east_w = input[idx_south + 1];
        }
    }

    if (col > 0) {
        west_word = input[idx - 1];
    }
    if (col < max_col) {
        east_word = input[idx + 1];
    }

    // Build the eight neighbor direction masks (bitplanes).
    // For any given bit position b in "center":
    //   - N/S neighbors are at bit b in the north/south words.
    //   - W neighbor is at bit (b-1) or bit 63 of the word to the left
    //     when b == 0.
    //   - E neighbor is at bit (b+1) or bit 0 of the word to the right
    //     when b == 63.
    // Similar logic applies for diagonal neighbors using the appropriate
    // neighbor words above/below and left/right of the current word.
    const std::uint64_t n  = north;
    const std::uint64_t s  = south;

    const std::uint64_t w  = (center >> 1) | (west_word << 63);
    const std::uint64_t e  = (center << 1) | (east_word >> 63);

    const std::uint64_t nw = (north >> 1) | (north_west_w << 63);
    const std::uint64_t ne = (north << 1) | (north_east_w >> 63);
    const std::uint64_t sw = (south >> 1) | (south_west_w << 63);
    const std::uint64_t se = (south << 1) | (south_east_w >> 63);

    // Accumulate neighbor counts with bit-sliced 3-bit counters.
    std::uint64_t ones  = 0;
    std::uint64_t twos  = 0;
    std::uint64_t fours = 0;

    add_neighbor_bitplane(n,  ones, twos, fours);
    add_neighbor_bitplane(s,  ones, twos, fours);
    add_neighbor_bitplane(e,  ones, twos, fours);
    add_neighbor_bitplane(w,  ones, twos, fours);
    add_neighbor_bitplane(ne, ones, twos, fours);
    add_neighbor_bitplane(nw, ones, twos, fours);
    add_neighbor_bitplane(se, ones, twos, fours);
    add_neighbor_bitplane(sw, ones, twos, fours);

    // For each bit position:
    //   neighbor_count = ones + 2*twos + 4*fours  (mod 8)
    //
    // We derive masks for counts equal to 2 or 3. Using binary:
    //   count == 3  => (fours,twos,ones) == 0b011
    //   count == 2  => (fours,twos,ones) == 0b010
    const std::uint64_t not_fours   = ~fours;
    const std::uint64_t count_eq_3  = not_fours & twos &  ones;     // 0b011
    const std::uint64_t count_eq_2  = not_fours & twos & (~ones);   // 0b010

    // Game of Life update rule (per bit):
    //   new_state = (neighbor_count == 3) OR (alive & neighbor_count == 2)
    const std::uint64_t next = count_eq_3 | (center & count_eq_2);

    output[idx] = next;
}

/**
 * @brief Host wrapper to run one Game of Life step on the GPU.
 *
 * @param input           Device pointer to input grid (bit-packed).
 * @param output          Device pointer to output grid (bit-packed).
 * @param grid_dimensions Width and height of the square grid (power of 2).
 *
 * Both input and output must point to arrays of std::uint64_t large enough
 * to store grid_dimensions * grid_dimensions bits, i.e.:
 *
 *   total_words = (grid_dimensions * grid_dimensions) / 64
 *
 * No device synchronization is performed here; the caller is responsible
 * for any required cudaDeviceSynchronize or stream management.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each row contains grid_dimensions cells, i.e., grid_dimensions / 64 words.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    // words_per_row is a power of two since grid_dimensions is a power of two.
    int words_per_row_log2 = 0;
    int tmp = words_per_row;
    while (tmp > 1) {
        tmp >>= 1;
        ++words_per_row_log2;
    }

    // One thread per 64-cell word.
    const int block_size = 256;
    const int grid_size  = (total_words + block_size - 1) / block_size;

    game_of_life_kernel<<<grid_size, block_size>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
        words_per_row_log2,
        total_words
    );
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
