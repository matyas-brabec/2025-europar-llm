#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single step of Conway's Game of Life
 * on a bit-packed grid.
 *
 * Representation:
 *   - The grid is square: grid_dimensions x grid_dimensions.
 *   - Each row is stored as grid_dimensions / 64 std::uint64_t words.
 *   - Bit i (0 = least significant) of word w in row y represents the cell
 *     at column (w * 64 + i) in that row.
 *   - Cells outside the grid are implicitly dead.
 *
 * Parallelization:
 *   - Each CUDA thread processes one 64-bit word (i.e., 64 cells).
 *   - The kernel is launched with a 2D grid:
 *       gridDim.y = number of rows = grid_dimensions
 *       gridDim.x = ceil(words_per_row / blockDim.x)
 *
 * Neighborhood handling:
 *   For the word at (row = y, word_x = wx):
 *     - Load up to 9 words:
 *         above_left, above_center, above_right
 *         center_left, center,       center_right
 *         below_left, below_center,  below_right
 *       Words outside the grid are treated as 0.
 *
 *   - From these words, build 8 "neighbor bitboards" (each 64-bit):
 *       northwest, north, northeast,
 *       west,               east,
 *       southwest, south, southeast
 *
 *     For a word segment representing columns k..k+63:
 *       Let row bits R[c] with c = column index.
 *       Mapping (bit 0 is column k, bit 63 is column k+63):
 *         - West neighbors:  West[i] = R[k + i - 1]
 *              West = (center << 1) | (left >> 63)
 *         - East neighbors:  East[i] = R[k + i + 1]
 *              East = (center >> 1) | (right << 63)
 *
 *     For rows above/below we use the same west/east construction to obtain
 *     the diagonal neighbors NW, NE, SW, SE.
 *
 * Neighbor counting:
 *   - We must count, for each bit, how many of the 8 neighbor bitboards
 *     have a 1 at that position.
 *   - We do this using a 4-bit parallel counter per cell:
 *       s0: bit 0 of neighbor count (LSB)
 *       s1: bit 1
 *       s2: bit 2
 *       s3: bit 3 (only needed for count == 8)
 *
 *   - Start with s0 = s1 = s2 = s3 = 0.
 *   - For each neighbor bitboard nb, we increment the 4-bit counter at every
 *     bit position using ripple-carry addition:
 *
 *       carry0 = s0 & nb;
 *       s0    ^= nb;
 *       carry1 = s1 & carry0;
 *       s1    ^= carry0;
 *       carry2 = s2 & carry1;
 *       s2    ^= carry1;
 *       s3    ^= carry2;   // no overflow beyond s3 since max count is 8
 *
 *   - After processing all 8 neighbors, the per-bit neighbor count is
 *     encoded as (s3 s2 s1 s0).
 *
 * Applying Game of Life rules (B3/S23):
 *   - A cell is alive in the next generation if:
 *       - It has exactly 3 live neighbors, or
 *       - It is currently alive and has exactly 2 live neighbors.
 *
 *   - To compute masks for neighbor count == 2 or 3:
 *       Binary patterns:
 *         2 -> 0b0010: s3=0, s2=0, s1=1, s0=0
 *         3 -> 0b0011: s3=0, s2=0, s1=1, s0=1
 *
 *       Let no_high = ~(s2 | s3);   // ensures s2 == 0 and s3 == 0
 *       eq2 = no_high & s1 & ~s0;
 *       eq3 = no_high & s1 &  s0;
 *
 *   - Let 'alive' be the current word (center).
 *       next = eq3 | (alive & eq2);
 *
 *     Explanation:
 *       - eq3: cells with exactly 3 neighbors become alive (birth or survival).
 *       - alive & eq2: cells that are currently alive and have 2 neighbors.
 */

__device__ __forceinline__
void add_neighbor_bitboard(std::uint64_t nb,
                           std::uint64_t& s0,
                           std::uint64_t& s1,
                           std::uint64_t& s2,
                           std::uint64_t& s3)
{
    // Ripple-carry increment of a 4-bit counter per bit position.
    std::uint64_t carry0 = s0 & nb;
    s0 ^= nb;

    std::uint64_t carry1 = s1 & carry0;
    s1 ^= carry0;

    std::uint64_t carry2 = s2 & carry1;
    s2 ^= carry1;

    s3 ^= carry2;
}

__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int words_per_row,
                         int height)
{
    // word_x: index of 64-bit word within the row
    const int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y      = blockIdx.y;  // row index

    if (word_x >= words_per_row || y >= height)
        return;

    const int idx_center = y * words_per_row + word_x;

    // Determine boundary conditions.
    const bool has_left  = (word_x > 0);
    const bool has_right = (word_x + 1 < words_per_row);
    const bool has_above = (y > 0);
    const bool has_below = (y + 1 < height);

    // Load center row words.
    const std::uint64_t centerC = input[idx_center];
    const std::uint64_t centerL = has_left  ? input[idx_center - 1] : 0ull;
    const std::uint64_t centerR = has_right ? input[idx_center + 1] : 0ull;

    // Load above row words, if present.
    std::uint64_t aboveL = 0ull, aboveC = 0ull, aboveR = 0ull;
    if (has_above) {
        const int idx_above_center = idx_center - words_per_row;
        aboveC = input[idx_above_center];
        aboveL = has_left  ? input[idx_above_center - 1] : 0ull;
        aboveR = has_right ? input[idx_above_center + 1] : 0ull;
    }

    // Load below row words, if present.
    std::uint64_t belowL = 0ull, belowC = 0ull, belowR = 0ull;
    if (has_below) {
        const int idx_below_center = idx_center + words_per_row;
        belowC = input[idx_below_center];
        belowL = has_left  ? input[idx_below_center - 1] : 0ull;
        belowR = has_right ? input[idx_below_center + 1] : 0ull;
    }

    // Build neighbor bitboards.
    // Horizontal neighbors in the same row.
    const std::uint64_t west = (centerC << 1) | (centerL >> 63);
    const std::uint64_t east = (centerC >> 1) | (centerR << 63);

    // Neighbors from the row above.
    const std::uint64_t north     = aboveC;
    const std::uint64_t northwest = (aboveC << 1) | (aboveL >> 63);
    const std::uint64_t northeast = (aboveC >> 1) | (aboveR << 63);

    // Neighbors from the row below.
    const std::uint64_t south     = belowC;
    const std::uint64_t southwest = (belowC << 1) | (belowL >> 63);
    const std::uint64_t southeast = (belowC >> 1) | (belowR << 63);

    // 4-bit neighbor count per bit position: s3 s2 s1 s0.
    std::uint64_t s0 = 0ull;
    std::uint64_t s1 = 0ull;
    std::uint64_t s2 = 0ull;
    std::uint64_t s3 = 0ull;

    // Accumulate counts from the 8 neighbor bitboards.
    add_neighbor_bitboard(northwest, s0, s1, s2, s3);
    add_neighbor_bitboard(north,     s0, s1, s2, s3);
    add_neighbor_bitboard(northeast, s0, s1, s2, s3);
    add_neighbor_bitboard(west,      s0, s1, s2, s3);
    add_neighbor_bitboard(east,      s0, s1, s2, s3);
    add_neighbor_bitboard(southwest, s0, s1, s2, s3);
    add_neighbor_bitboard(south,     s0, s1, s2, s3);
    add_neighbor_bitboard(southeast, s0, s1, s2, s3);

    // Compute masks for neighbor count == 2 and == 3.
    const std::uint64_t no_high = ~(s2 | s3);      // ensure s2 == 0 and s3 == 0
    const std::uint64_t eq2     = no_high & s1 & ~s0;
    const std::uint64_t eq3     = no_high & s1 &  s0;

    // Apply Game of Life rules (B3/S23).
    const std::uint64_t alive = centerC;
    const std::uint64_t next  = eq3 | (alive & eq2);

    output[idx_center] = next;
}

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is the number of cells per row/column.
    // Each word encodes 64 cells.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int height        = grid_dimensions;

    // Choose a 1D block size along X; Y is implicit via gridDim.y = height.
    const int block_size_x = 256;
    const int grid_size_x  = (words_per_row + block_size_x - 1) / block_size_x;

    dim3 block(block_size_x, 1, 1);
    dim3 grid(grid_size_x, height, 1);

    game_of_life_kernel<<<grid, block>>>(input, output, words_per_row, height);
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
