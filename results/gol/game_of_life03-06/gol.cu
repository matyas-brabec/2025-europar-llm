#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single step of Conway's Game of Life on a bit-packed
 * 2D grid.
 *
 * - The grid is square, of size grid_dimensions x grid_dimensions cells.
 * - Each cell is a single bit: 1 = alive, 0 = dead.
 * - The grid is stored in row-major order, bit-packed into std::uint64_t words:
 *     * Each word encodes 64 consecutive cells within the same row.
 *     * Bit 0 is the least significant bit, bit 63 is the most significant bit.
 * - Cells outside the grid are considered dead (no wrap-around).
 * - Each CUDA thread processes exactly one 64-bit word (64 cells).
 *
 * Neighbour handling
 * ------------------
 * For a given word C_C (current row, current word), its neighboring words are:
 *
 *   Row above: A_L, A_C, A_R  (left, center, right)
 *   Same row:  C_L,  C_C, C_R
 *   Row below: B_L, B_C, B_R
 *
 * To construct bit masks for the 8 neighbor directions, we need to account for
 * intra-word neighbors (left/right within the same 64-bit word) and inter-word
 * neighbors (bit 0 / bit 63 that spill into words to the left/right):
 *
 *   - For a word pair (left, center):
 *       shift_left_combine(center, left)  =
 *            (center << 1) | (left >> 63)
 *
 *     This constructs a mask where bit j corresponds to the "left neighbor" of
 *     the original bit j in the center word. For bit 0, the left neighbor is
 *     bit 63 of the left word.
 *
 *   - For a word pair (center, right):
 *       shift_right_combine(center, right) =
 *            (center >> 1) | (right << 63)
 *
 *     This constructs a mask where bit j corresponds to the "right neighbor" of
 *     the original bit j in the center word. For bit 63, the right neighbor is
 *     bit 0 of the right word.
 *
 * Using these helpers, the eight neighbor masks for C_C are:
 *
 *   north      = A_C
 *   south      = B_C
 *   west       = shift_left_combine(C_C, C_L)
 *   east       = shift_right_combine(C_C, C_R)
 *   north_west = shift_left_combine(A_C, A_L)
 *   north_east = shift_right_combine(A_C, A_R)
 *   south_west = shift_left_combine(B_C, B_L)
 *   south_east = shift_right_combine(B_C, B_R)
 *
 * Bit-sliced neighbor counting
 * ----------------------------
 * We need, for each bit position, the number of alive neighbors (0..8).
 * Instead of counting per bit, we use bit-sliced addition:
 *
 *   - Maintain three 64-bit planes (s0, s1, s2) that represent the neighbor
 *     count modulo 8:
 *       count = s0 + 2*s1 + 4*s2  (per bit position)
 *
 *   - For each neighbor mask 'bits' (one of the eight masks), we perform
 *     a per-bit full-adder into (s0, s1, s2) without cross-bit carries:
 *
 *       carry0 = s0 & bits;
 *       s0    ^= bits;
 *       carry1 = s1 & carry0;
 *       s1    ^= carry0;
 *       s2    ^= carry1;
 *
 *     Each lane (bit position) behaves like a 3-bit counter modulo 8, and
 *     there is no carry between different bit positions because all operations
 *     are bitwise.
 *
 *   - The maximum neighbor count is 8, so representing counts modulo 8 is
 *     sufficient to distinguish counts 0..7 and 8:
 *       * 8 (1000b) becomes 0 (000b) in modulo-8 representation, which does
 *         not interfere with equality tests for 2 or 3 (since 8 != 2,3).
 *
 * After accumulating all 8 neighbor masks:
 *
 *   - neighbor_count == 2  <=>  (s2==0, s1==1, s0==0)
 *   - neighbor_count == 3  <=>  (s2==0, s1==1, s0==1)
 *
 * Which translates to bit masks:
 *
 *   eq2 = (~s2) & s1 & (~s0);
 *   eq3 = (~s2) & s1 &  s0;
 *
 * Game of Life rule per bit (cell):
 *
 *   alive' = (neighbor_count == 3) ||
 *             (alive && neighbor_count == 2)
 *
 * In bit form:
 *
 *   next = eq3 | (current & eq2)
 *
 * Kernel launch configuration
 * ---------------------------
 * We map the 2D grid of words to a 2D CUDA grid:
 *
 *   - X-dimension: words within a row (columns of 64-bit words)
 *   - Y-dimension: rows
 *
 *   words_per_row = grid_dimensions / 64
 *
 *   dim3 block(THREADS_PER_BLOCK, 1, 1);
 *   dim3 grid( (words_per_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
 *              grid_dimensions,
 *              1 );
 *
 * Each thread handles one word:
 *
 *   row      = blockIdx.y
 *   col_word = blockIdx.x * blockDim.x + threadIdx.x
 *
 * If col_word >= words_per_row, the thread exits early. Otherwise, it processes
 * word index:
 *
 *   idx = row * words_per_row + col_word
 *
 * The caller is responsible for synchronization if needed; run_game_of_life
 * only enqueues the kernel.
 */

///////////////////////////////////////////////////////////////////////////////
// Device helpers
///////////////////////////////////////////////////////////////////////////////

/**
 * Shift a 64-bit word left by 1 bit and insert bit 63 from 'left' into bit 0.
 *
 * This is used to construct "west" or "north-west"/"south-west" neighbor masks
 * so that the neighbor of bit 0 comes from the previous word.
 */
__device__ __forceinline__
std::uint64_t shift_left_combine(std::uint64_t center, std::uint64_t left)
{
    // (center << 1) shifts intra-word neighbors.
    // (left >> 63) brings bit 63 of the left word into bit 0.
    return (center << 1) | (left >> 63);
}

/**
 * Shift a 64-bit word right by 1 bit and insert bit 0 from 'right' into bit 63.
 *
 * This is used to construct "east" or "north-east"/"south-east" neighbor masks
 * so that the neighbor of bit 63 comes from the next word.
 */
__device__ __forceinline__
std::uint64_t shift_right_combine(std::uint64_t center, std::uint64_t right)
{
    // (center >> 1) shifts intra-word neighbors.
    // (right << 63) brings bit 0 of the right word into bit 63.
    return (center >> 1) | (right << 63);
}

/**
 * Add a single-bit mask 'bits' to the per-bit 3-bit counter (s2 s1 s0)
 * representing neighbor counts modulo 8.
 *
 * For each bit position i:
 *   s0[i], s1[i], s2[i] represent the current count in binary:
 *     count_i = s0[i] + 2*s1[i] + 4*s2[i]  (mod 8)
 *
 * Adding 'bits[i]' is done via a ripple of 1-bit full adders, but because
 * all operations are bitwise, no carries propagate between different bit
 * positions.
 *
 * We ignore overflow into the 4th bit, effectively working modulo 8.
 */
__device__ __forceinline__
void add_bit_plane(std::uint64_t bits,
                   std::uint64_t &s0,
                   std::uint64_t &s1,
                   std::uint64_t &s2)
{
    // First bit (LSB) addition: s0 += bits
    std::uint64_t carry0 = s0 & bits;
    s0 ^= bits;

    // Second bit: s1 += carry0
    std::uint64_t carry1 = s1 & carry0;
    s1 ^= carry0;

    // Third bit: s2 += carry1  (mod 8, ignore any further carry)
    s2 ^= carry1;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel
///////////////////////////////////////////////////////////////////////////////

/**
 * Kernel that computes one step of Conway's Game of Life on a bit-packed grid.
 *
 * Parameters:
 *   input         - device pointer to input grid (bit-packed, row-major)
 *   output        - device pointer to output grid (next generation)
 *   grid_dim      - number of cells per dimension (grid is grid_dim x grid_dim)
 *   words_per_row - number of 64-bit words per row (grid_dim / 64)
 *
 * The kernel uses a 2D grid:
 *   - gridDim.y = number of rows (grid_dim)
 *   - gridDim.x = ceil(words_per_row / blockDim.x)
 *   - blockDim.x threads cover words within a row.
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row)
{
    const int row      = blockIdx.y;
    const int col_word = blockIdx.x * blockDim.x + threadIdx.x;

    // Rows are mapped one-to-one with gridDim.y, so row is always valid.
    // Columns (words) may exceed words_per_row because of block rounding.
    if (col_word >= words_per_row)
        return;

    // Compute linear index for this word.
    const std::size_t row_offset = static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(words_per_row);
    const std::size_t idx = row_offset + static_cast<std::size_t>(col_word);

    // Current row: center and its left/right neighbors.
    const std::uint64_t C_C = input[idx];
    const std::uint64_t C_L = (col_word > 0)
                              ? input[idx - 1]
                              : std::uint64_t(0);
    const std::uint64_t C_R = (col_word + 1 < words_per_row)
                              ? input[idx + 1]
                              : std::uint64_t(0);

    // Row above: A_L, A_C, A_R (zero if row == 0).
    std::uint64_t A_L, A_C, A_R;
    if (row > 0)
    {
        const std::size_t above_offset = row_offset -
                                         static_cast<std::size_t>(words_per_row);
        const std::size_t idxA = above_offset + static_cast<std::size_t>(col_word);

        A_C = input[idxA];
        A_L = (col_word > 0)
              ? input[idxA - 1]
              : std::uint64_t(0);
        A_R = (col_word + 1 < words_per_row)
              ? input[idxA + 1]
              : std::uint64_t(0);
    }
    else
    {
        A_L = A_C = A_R = std::uint64_t(0);
    }

    // Row below: B_L, B_C, B_R (zero if row == grid_dim - 1).
    std::uint64_t B_L, B_C, B_R;
    if (row + 1 < grid_dim)
    {
        const std::size_t below_offset = row_offset +
                                         static_cast<std::size_t>(words_per_row);
        const std::size_t idxB = below_offset + static_cast<std::size_t>(col_word);

        B_C = input[idxB];
        B_L = (col_word > 0)
              ? input[idxB - 1]
              : std::uint64_t(0);
        B_R = (col_word + 1 < words_per_row)
              ? input[idxB + 1]
              : std::uint64_t(0);
    }
    else
    {
        B_L = B_C = B_R = std::uint64_t(0);
    }

    // Construct the eight neighbor masks via bit shifts and cross-word carries.
    const std::uint64_t north      = A_C;
    const std::uint64_t south      = B_C;
    const std::uint64_t west       = shift_left_combine(C_C, C_L);
    const std::uint64_t east       = shift_right_combine(C_C, C_R);
    const std::uint64_t north_west = shift_left_combine(A_C, A_L);
    const std::uint64_t north_east = shift_right_combine(A_C, A_R);
    const std::uint64_t south_west = shift_left_combine(B_C, B_L);
    const std::uint64_t south_east = shift_right_combine(B_C, B_R);

    // Bit-sliced neighbor count: s0 (LSB), s1, s2 (MSB) represent count mod 8.
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;

    // Accumulate the eight neighbor directions.
    add_bit_plane(north_west, s0, s1, s2);
    add_bit_plane(north,      s0, s1, s2);
    add_bit_plane(north_east, s0, s1, s2);
    add_bit_plane(west,       s0, s1, s2);
    add_bit_plane(east,       s0, s1, s2);
    add_bit_plane(south_west, s0, s1, s2);
    add_bit_plane(south,      s0, s1, s2);
    add_bit_plane(south_east, s0, s1, s2);

    // Compute masks for neighbor_count == 2 and neighbor_count == 3.
    const std::uint64_t not_s0 = ~s0;
    const std::uint64_t not_s2 = ~s2;

    const std::uint64_t eq2 = not_s2 & s1 & not_s0;  // 010b
    const std::uint64_t eq3 = not_s2 & s1 & s0;      // 011b

    // Current live cells.
    const std::uint64_t alive = C_C;

    // Apply Game of Life rules:
    // - Survive if alive and neighbor_count == 2.
    // - Born if neighbor_count == 3.
    const std::uint64_t survive = alive & eq2;
    const std::uint64_t born    = eq3;

    const std::uint64_t next = survive | born;

    // Store result.
    output[idx] = next;
}

///////////////////////////////////////////////////////////////////////////////
// Host entry point
///////////////////////////////////////////////////////////////////////////////

/**
 * Host function to run one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *   input            - device pointer to input grid (bit-packed, row-major)
 *   output           - device pointer to output grid (bit-packed, row-major)
 *   grid_dimensions  - width/height of the square grid (power of 2, > 512)
 *
 * Both input and output must be allocated with cudaMalloc and large enough
 * to hold grid_dimensions * grid_dimensions cells, packed into 64-bit words:
 *
 *   total_words = (grid_dimensions * grid_dimensions) / 64
 *
 * No synchronization is performed; the caller is responsible for calling
 * cudaDeviceSynchronize() if needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 64-bit words per row.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Configure a 2D grid:
    //   - X dimension: words within a row.
    //   - Y dimension: rows.
    //
    // Each block covers THREADS_PER_BLOCK words in X; any extra threads in the
    // last block of a row simply exit early.
    constexpr int THREADS_PER_BLOCK = 256;

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((words_per_row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                 static_cast<unsigned int>(grid_dimensions),
                 1);

    game_of_life_kernel<<<gridDim, blockDim>>>(input, output,
                                               grid_dimensions, words_per_row);
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
