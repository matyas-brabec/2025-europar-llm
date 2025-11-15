#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single step of Conway's Game of Life on a
 * bit-packed grid:
 *
 * - Each std::uint64_t word encodes 64 consecutive cells of one row.
 *   Bit 0 is the leftmost cell in the word, bit 63 the rightmost.
 * - The grid is square with side length grid_dim, a power of two > 512.
 * - Cells outside the grid are treated as dead.
 *
 * Parallelization strategy:
 * - Each CUDA thread processes one 64-bit word (64 cells) independently.
 * - Threads are organized in a 2D grid: X over words-in-row, Y over rows.
 * - We only use global memory (no shared/texture memory) as requested.
 *
 * Neighbor counting:
 * - For each word we load up to 9 words: the word itself and its
 *   horizontal/vertical neighbors (left/right, above/below).
 * - From these words we construct eight 64-bit bitmasks:
 *     NW, N, NE, W, E, SW, S, SE
 *   via shifts and cross-word bit carry between adjacent words.
 * - Each bit of these masks represents a neighbor cell for the corresponding
 *   cell in the center word.
 *
 * - We then compute the per-cell neighbor count using a bit-sliced ripple
 *   adder:
 *   * Maintain four 64-bit planes (c0, c1, c2, c3) representing the
 *     4-bit neighbor count (0..8) for all 64 cells in the word in parallel:
 *       count = c0 + 2*c1 + 4*c2 + 8*c3
 *   * For each of the 8 neighbor masks we "add 1" to this 4-bit counter using
 *     simple bitwise logic equivalent to cascaded full adders.
 *
 * Life rules:
 * - From the final count planes, we derive masks for:
 *   - "neighbors in {2,3}"  -> used for survival
 *   - "neighbors == 3"      -> used for birth
 * - New state:
 *     new = (alive & (neighbors == 2 or 3))  |  (~alive & (neighbors == 3))
 *
 * All operations are performed with bitwise ops on 64-bit words, exploiting
 * full-word parallelism.
 */

//////////////////////////////////////////////////////////////////
// Device helper: add one 1-bit neighbor mask to a 4-bit counter //
//////////////////////////////////////////////////////////////////

/*
 * add_neighbor_mask:
 *   Adds a 1-bit-per-cell neighbor mask to the 4-bit per-cell count
 *   held in (c0, c1, c2, c3). Each parameter is a 64-bit word where
 *   each bit position corresponds to one cell.
 *
 *   For each bit position i (0..63):
 *     input neighbor bit n = neighbor[i] ∈ {0,1}
 *     current count is a 4-bit integer:
 *         count = c0[i] + 2*c1[i] + 4*c2[i] + 8*c3[i]
 *     We compute:
 *         count' = count + n
 *
 *   The logic is implemented as a ripple of half-adders, but at the
 *   word level. This is equivalent to chaining full adders per bit,
 *   using:
 *       sum   = a ^ b ^ c
 *       carry = (a & b) | (b & c) | (a & c)
 *   but specialized to the case of adding a 1-bit operand.
 */
__device__ __forceinline__
void add_neighbor_mask(std::uint64_t neighbor,
                       std::uint64_t &c0,
                       std::uint64_t &c1,
                       std::uint64_t &c2,
                       std::uint64_t &c3)
{
    // Add neighbor to least significant bit plane c0
    std::uint64_t sum  = c0 ^ neighbor;
    std::uint64_t carry = c0 & neighbor;
    c0 = sum;

    // Propagate carry into next bit plane c1
    sum   = c1 ^ carry;
    std::uint64_t carry2 = c1 & carry;
    c1 = sum;

    // Propagate into c2
    sum   = c2 ^ carry2;
    std::uint64_t carry3 = c2 & carry2;
    c2 = sum;

    // Propagate final carry into most significant bit plane c3
    c3 ^= carry3;
}

//////////////////////////////////////////////////////////////
// Kernel: one Game of Life step on a bit-packed 2D grid    //
//////////////////////////////////////////////////////////////

__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim)
{
    // Number of 64-bit words per row (grid_dim is a power of two)
    const int words_per_row = grid_dim >> 6; // grid_dim / 64

    // 2D index: word_x within row, row index
    const int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= grid_dim || word_x >= words_per_row)
        return;

    const int idx = row * words_per_row + word_x;

    // Load center word and its horizontal neighbors (same row)
    const std::uint64_t m  = input[idx];
    const std::uint64_t mL = (word_x > 0) ? input[idx - 1] : 0ull;
    const std::uint64_t mR = (word_x + 1 < words_per_row) ? input[idx + 1] : 0ull;

    // Load words from the row above (if it exists)
    std::uint64_t u  = 0ull;
    std::uint64_t uL = 0ull;
    std::uint64_t uR = 0ull;
    if (row > 0) {
        const int idx_up = idx - words_per_row;
        u  = input[idx_up];
        uL = (word_x > 0) ? input[idx_up - 1] : 0ull;
        uR = (word_x + 1 < words_per_row) ? input[idx_up + 1] : 0ull;
    }

    // Load words from the row below (if it exists)
    std::uint64_t d  = 0ull;
    std::uint64_t dL = 0ull;
    std::uint64_t dR = 0ull;
    if (row + 1 < grid_dim) {
        const int idx_down = idx + words_per_row;
        d  = input[idx_down];
        dL = (word_x > 0) ? input[idx_down - 1] : 0ull;
        dR = (word_x + 1 < words_per_row) ? input[idx_down + 1] : 0ull;
    }

    // Construct neighbor bitmasks for this word.
    //
    // Bit indexing convention:
    //   - Bit 0 is the leftmost cell in the word
    //   - Bit 63 is the rightmost
    //
    // Horizontal neighbors:
    //   Left neighbor W:
    //     - For bit i > 0: comes from m bit (i-1)  -> (m << 1)
    //     - For bit i = 0: comes from mL bit 63    -> (mL >> 63)
    //   Right neighbor E:
    //     - For bit i < 63: comes from m bit (i+1) -> (m >> 1)
    //     - For bit i = 63: comes from mR bit 0    -> (mR << 63)
    //
    // Vertical + diagonal neighbors use the same pattern with u/uL/uR and d/dL/dR.
    const std::uint64_t nw = (u << 1) | (uL >> 63); // above-left
    const std::uint64_t n  = u;                     // above
    const std::uint64_t ne = (u >> 1) | (uR << 63); // above-right

    const std::uint64_t w  = (m << 1) | (mL >> 63); // left
    const std::uint64_t e  = (m >> 1) | (mR << 63); // right

    const std::uint64_t sw = (d << 1) | (dL >> 63); // below-left
    const std::uint64_t s  = d;                     // below
    const std::uint64_t se = (d >> 1) | (dR << 63); // below-right

    // Accumulate neighbor counts using 4 bitplanes c0..c3.
    // Each add_neighbor_mask call processes all 64 cells in this word.
    std::uint64_t c0 = 0ull; // bit 0 of neighbor count
    std::uint64_t c1 = 0ull; // bit 1
    std::uint64_t c2 = 0ull; // bit 2
    std::uint64_t c3 = 0ull; // bit 3 (only needed for count == 8)

    add_neighbor_mask(nw, c0, c1, c2, c3);
    add_neighbor_mask(n,  c0, c1, c2, c3);
    add_neighbor_mask(ne, c0, c1, c2, c3);
    add_neighbor_mask(w,  c0, c1, c2, c3);
    add_neighbor_mask(e,  c0, c1, c2, c3);
    add_neighbor_mask(sw, c0, c1, c2, c3);
    add_neighbor_mask(s,  c0, c1, c2, c3);
    add_neighbor_mask(se, c0, c1, c2, c3);

    // At this point, for each bit position i:
    //   neighbor_count[i] = c0[i] + 2*c1[i] + 4*c2[i] + 8*c3[i]
    //
    // We need:
    //   - neighbors in {2,3} for survival
    //   - neighbors == 3     for birth
    //
    // Binary encoding:
    //   count == 2 : 0010 -> c0=0, c1=1, c2=0, c3=0
    //   count == 3 : 0011 -> c0=1, c1=1, c2=0, c3=0
    //
    // So:
    //   base = cells with neighbors in {2,3} = (c1 & ~c2 & ~c3)
    //   eq3  = cells with neighbors == 3     = (base & c0)
    const std::uint64_t not_c2 = ~c2;
    const std::uint64_t not_c3 = ~c3;
    const std::uint64_t base   = c1 & not_c2 & not_c3; // neighbors == 2 or 3
    const std::uint64_t eq3    = base & c0;            // neighbors == 3

    const std::uint64_t alive = m;

    // Survive: alive cells with 2 or 3 neighbors
    const std::uint64_t survive = alive & base;

    // Birth: dead cells with exactly 3 neighbors
    const std::uint64_t birth = (~alive) & eq3;

    // Next generation for this 64-cell block
    output[idx] = survive | birth;
}

//////////////////////////////////////////////////////////////
// Host function: run one Game of Life step on the GPU      //
//////////////////////////////////////////////////////////////

/*
 * run_game_of_life:
 *   Executes a single generation update of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *   - input           : device pointer to bit-packed input grid
 *   - output          : device pointer to bit-packed output grid
 *   - grid_dimensions : width and height of the square grid (power of two)
 *
 * Memory layout:
 *   - The grid is stored row-major.
 *   - Each row consists of (grid_dimensions / 64) 64-bit words.
 *   - Both input and output pointers are assumed to be cudaMalloc'ed.
 *
 * Synchronization:
 *   - This function does not synchronize; the caller is responsible for any
 *     necessary cudaDeviceSynchronize/cudaStreamSynchronize.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Choose a 2D block configuration that aligns warps along the X dimension
    // for good memory coalescing when accessing words within a row.
    const dim3 block(32, 8); // 256 threads per block

    const dim3 grid(
        (words_per_row   + block.x - 1) / block.x,
        (grid_dimensions + block.y - 1) / block.y
    );

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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
