#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * Data layout:
 * - The grid is N x N cells, with N a power of 2 >= 512.
 * - Each std::uint64_t word stores 64 consecutive cells from the same row.
 *   Bit 0 is the least significant bit, bit 63 the most significant.
 * - Words are stored in row-major order: row 0 words, then row 1, etc.
 *
 * Parallelization:
 * - Each CUDA thread is responsible for one 64-bit word (i.e., 64 cells).
 * - Threads are organized in a 2D grid: (word_x, row_y).
 *
 * Neighborhood handling:
 * - For a given word (x,y), we load up to 9 words:
 *     (y-1,x-1) (y-1,x) (y-1,x+1)
 *     (y  ,x-1) (y  ,x) (y  ,x+1)
 *     (y+1,x-1) (y+1,x) (y+1,x+1)
 *   Words outside grid bounds are treated as zero (all dead).
 *
 * Bit-parallel neighbor counting:
 * - For the central word C, and its neighbors we build eight 64-bit masks:
 *     north, south, west, east, northwest, northeast, southwest, southeast.
 *   Each mask has a '1' bit at positions where that direction's neighbor is alive.
 *
 * - We then count neighbors per cell using a bit-sliced ripple-carry scheme:
 *     We maintain four 64-bit planes c0, c1, c2, c3 representing a 4-bit
 *     neighbor count for each cell (0..8), where:
 *       N = c0*1 + c1*2 + c2*4 + c3*8  (for each bit position independently).
 *
 *   For each neighbor mask M, we "add 1" to the count wherever M has a 1 bit:
 *     carry0 = c0 & M;  c0 ^= M;
 *     carry1 = c1 & carry0; c1 ^= carry0;
 *     carry2 = c2 & carry1; c2 ^= carry1;
 *     c3 ^= carry2;
 *
 *   Repeating this for all 8 neighbor directions yields exact counts 0..8
 *   for every cell in the word.
 *
 * Game of Life rule application:
 * - A cell lives in the next generation if:
 *     neighbors == 3  OR  (cell_is_alive AND neighbors == 2)
 *
 * - Using the count bit-planes (c0..c3), we compute:
 *     neighbors == 2  <=>  (~c3 & ~c2 &  c1 & ~c0)
 *     neighbors == 3  <=>  (~c3 & ~c2 &  c1 &  c0)
 *
 * - Let 'alive' be the current word; then:
 *     next = eq3 | (eq2 & alive)
 */

namespace {

/**
 * Device helper to increment per-cell neighbor count by a mask of neighbors.
 *
 * The count is represented in 4 bitplanes (c0..c3) per cell, giving a 4-bit
 * counter 0..15, but our actual range is 0..8 (8 neighbors).
 *
 * For each bit position i:
 *   if mask_i == 1, then (c3,c2,c1,c0) at lane i is incremented by 1
 *   using a ripple-carry binary adder.
 */
__device__ __forceinline__
void add_neighbor_mask(std::uint64_t mask,
                       std::uint64_t &c0,
                       std::uint64_t &c1,
                       std::uint64_t &c2,
                       std::uint64_t &c3)
{
    // Bit 0 addition: c0 += mask (1-bit add)
    std::uint64_t carry0 = c0 & mask;
    c0 ^= mask;

    // Bit 1 addition: c1 += carry0
    std::uint64_t carry1 = c1 & carry0;
    c1 ^= carry0;

    // Bit 2 addition: c2 += carry1
    std::uint64_t carry2 = c2 & carry1;
    c2 ^= carry1;

    // Bit 3 addition: c3 += carry2 (we ignore overflow beyond this)
    c3 ^= carry2;
}

/**
 * CUDA kernel: compute one Game of Life step on a bit-packed square grid.
 *
 * Parameters:
 *   input          - pointer to device memory with current grid state
 *   output         - pointer to device memory for next grid state
 *   grid_dim       - grid width/height (N)
 *   words_per_row  - number of 64-bit words per row (N / 64)
 */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Compute the word coordinates this thread is responsible for
    int word_x = blockIdx.x * blockDim.x + threadIdx.x; // index of 64-bit word in row
    int word_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (word_x >= words_per_row || word_y >= grid_dim)
        return;

    int idx = word_y * words_per_row + word_x;

    // Flags to check if neighbor words exist (grid boundaries)
    bool has_left   = (word_x > 0);
    bool has_right  = (word_x + 1 < words_per_row);
    bool has_top    = (word_y > 0);
    bool has_bottom = (word_y + 1 < grid_dim);

    // Load central word
    std::uint64_t c = input[idx];

    // Load horizontal neighbors in same row
    std::uint64_t l = has_left  ? input[idx - 1] : 0ull;
    std::uint64_t r = has_right ? input[idx + 1] : 0ull;

    // Load words from row above (north)
    std::uint64_t n  = has_top ? input[idx - words_per_row] : 0ull;
    std::uint64_t nl = (has_top && has_left)  ? input[idx - words_per_row - 1] : 0ull;
    std::uint64_t nr = (has_top && has_right) ? input[idx - words_per_row + 1] : 0ull;

    // Load words from row below (south)
    std::uint64_t s  = has_bottom ? input[idx + words_per_row] : 0ull;
    std::uint64_t sl = (has_bottom && has_left)  ? input[idx + words_per_row - 1] : 0ull;
    std::uint64_t sr = (has_bottom && has_right) ? input[idx + words_per_row + 1] : 0ull;

    // Build 8 neighbor bitboards for this word:
    //
    // Vertical:
    std::uint64_t north = n;
    std::uint64_t south = s;

    // Horizontal (within current row, including cross-word neighbors)
    std::uint64_t west  = (c << 1) | (l >> 63);
    std::uint64_t east  = (c >> 1) | (r << 63);

    // Diagonals (northwest, northeast, southwest, southeast)
    std::uint64_t northwest = (n << 1) | (nl >> 63);
    std::uint64_t northeast = (n >> 1) | (nr << 63);
    std::uint64_t southwest = (s << 1) | (sl >> 63);
    std::uint64_t southeast = (s >> 1) | (sr << 63);

    // Initialize neighbor count bit-planes to zero
    std::uint64_t c0 = 0ull; // bit 0 of neighbor count (LSB)
    std::uint64_t c1 = 0ull; // bit 1
    std::uint64_t c2 = 0ull; // bit 2
    std::uint64_t c3 = 0ull; // bit 3 (enough for max 8 neighbors)

    // Accumulate contributions from 8 neighbor directions
    add_neighbor_mask(north,     c0, c1, c2, c3);
    add_neighbor_mask(south,     c0, c1, c2, c3);
    add_neighbor_mask(west,      c0, c1, c2, c3);
    add_neighbor_mask(east,      c0, c1, c2, c3);
    add_neighbor_mask(northwest, c0, c1, c2, c3);
    add_neighbor_mask(northeast, c0, c1, c2, c3);
    add_neighbor_mask(southwest, c0, c1, c2, c3);
    add_neighbor_mask(southeast, c0, c1, c2, c3);

    // Compute masks where neighbor count == 2 or 3.
    // For count N, we have N = c0*1 + c1*2 + c2*4 + c3*8.
    // Thus:
    //   N == 2  <=>  (c3,c2,c1,c0) == 0010b
    //   N == 3  <=>  (c3,c2,c1,c0) == 0011b
    std::uint64_t not_c0 = ~c0;
    std::uint64_t not_c1 = ~c1;
    std::uint64_t not_c2 = ~c2;
    std::uint64_t not_c3 = ~c3;

    std::uint64_t eq2 = not_c3 & not_c2 &  c1 & not_c0;
    std::uint64_t eq3 = not_c3 & not_c2 &  c1 &  c0;

    // Current alive cells
    std::uint64_t alive = c;

    // Apply Game of Life rules:
    // next = (neighbors == 3) OR (alive AND neighbors == 2)
    std::uint64_t next = eq3 | (eq2 & alive);

    // Store result
    output[idx] = next;
}

} // anonymous namespace


/**
 * Host function to run one Game of Life step on the GPU.
 *
 * Parameters:
 *   input           - device pointer to current grid (bit-packed)
 *   output          - device pointer to next grid (bit-packed)
 *   grid_dimensions - grid width/height (N; must be divisible by 64)
 *
 * Notes:
 *   - input and output must be allocated by cudaMalloc.
 *   - No synchronization or error checking is performed here; the caller is
 *     responsible for any needed cudaDeviceSynchronize or error handling.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int grid_dim = grid_dimensions;
    const int words_per_row = grid_dim / 64;

    // Choose a 2D block configuration that promotes good occupancy and coalescing.
    // 32 threads in x-direction (one warp) operate on consecutive words in a row,
    // 8 threads in y-direction operate on adjacent rows.
    dim3 block_dim(32, 8, 1);

    dim3 grid_dim_2d(
        (words_per_row + block_dim.x - 1) / block_dim.x,
        (grid_dim      + block_dim.y - 1) / block_dim.y,
        1
    );

    game_of_life_kernel<<<grid_dim_2d, block_dim>>>(input, output, grid_dim, words_per_row);
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
