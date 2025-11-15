#include <cstdint>
#include <cuda_runtime.h>

// Simple alias for convenience in both host and device code.
using uint64_t = std::uint64_t;

/*
 * Bitwise carry-save adder (CSA) for three 64-bit bitboards.
 *
 * For each bit position i, this computes:
 *   sum_i   = a_i ^ b_i ^ c_i
 *   carry_i = majority(a_i, b_i, c_i)   (i.e., at least two of them are 1)
 *
 * Interpreting each bit position as a separate 1-bit integer, we have:
 *   a + b + c = sum + 2*carry
 *
 * Crucially, this does NOT propagate carries between bit positions, so it
 * correctly performs 64 independent 1-bit additions in parallel.
 */
__device__ __forceinline__
void bitwise_csa(uint64_t a, uint64_t b, uint64_t c,
                 uint64_t &sum, uint64_t &carry)
{
    uint64_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (u & c);
}

/*
 * Given eight neighbor direction bitboards (one per neighbor direction),
 * compute the neighbor count for each cell as four bitplanes:
 *
 *   b1: least significant bit  (1)
 *   b2: second bit             (2)
 *   b4: third bit              (4)
 *   b8: fourth bit             (8)
 *
 * For each bit position i, the number of live neighbors is:
 *   neighbors_i = b1_i + 2*b2_i + 4*b4_i + 8*b8_i   (0..8)
 *
 * This uses a tree of carry-save adders to sum eight 1-bit values without
 * cross-bit carries. The structure is:
 *
 *   CSA1: (n0, n1, n2) -> s1 (1), c1 (2)
 *   CSA2: (n3, n4, n5) -> s2 (1), c2 (2)
 *   CSA3: (n6, n7, 0)  -> s3 (1), c3 (2)
 *   CSA4: (s1, s2, s3) -> s4 (1), c4 (2)
 *   CSA5: (c1, c2, c3) -> s5 (2), c5 (4)
 *   CSA6: (s5, c4, 0)  -> s6 (2), c6 (4)
 *   CSA7: (c5, c6, 0)  -> s7 (4), c7 (8)
 *
 * Final bitplanes:
 *   b1 = s4
 *   b2 = s6
 *   b4 = s7
 *   b8 = c7
 */
__device__ __forceinline__
void count_neighbors_bitplanes(uint64_t n0, uint64_t n1,
                               uint64_t n2, uint64_t n3,
                               uint64_t n4, uint64_t n5,
                               uint64_t n6, uint64_t n7,
                               uint64_t &b1, uint64_t &b2,
                               uint64_t &b4, uint64_t &b8)
{
    uint64_t s1, c1;
    uint64_t s2, c2;
    uint64_t s3, c3;
    uint64_t s4, c4;
    uint64_t s5, c5;
    uint64_t s6, c6;
    uint64_t s7, c7;

    bitwise_csa(n0, n1, n2, s1, c1);
    bitwise_csa(n3, n4, n5, s2, c2);
    bitwise_csa(n6, n7, 0ull, s3, c3);

    bitwise_csa(s1, s2, s3, s4, c4);   // weight 1, carries weight 2
    bitwise_csa(c1, c2, c3, s5, c5);   // s5 weight 2, c5 weight 4

    bitwise_csa(s5, c4, 0ull, s6, c6); // s6 weight 2, c6 weight 4
    bitwise_csa(c5, c6, 0ull, s7, c7); // s7 weight 4, c7 weight 8

    b1 = s4;
    b2 = s6;
    b4 = s7;
    b8 = c7;
}

/*
 * Kernel configuration constant: number of 64-bit words handled by each thread
 * along the X (horizontal) direction.
 *
 * Using >1 words per thread improves memory efficiency by reusing row data
 * loaded from global memory across multiple output words, reducing the number
 * of loads compared to one-word-per-thread while keeping register usage
 * moderate.
 */
constexpr int WORDS_PER_THREAD = 2;

/*
 * CUDA kernel: compute one step of Conway's Game of Life on a bit-packed grid.
 *
 * Each 64-bit word encodes 64 horizontally adjacent cells in a row.
 * The grid is square with side length grid_dim, and each row therefore has
 * words_per_row = grid_dim / 64 words.
 *
 * Threads are organized such that each thread processes up to WORDS_PER_THREAD
 * contiguous words within a single row, to improve memory reuse.
 *
 * Boundary conditions:
 *   - Cells outside the grid are considered dead.
 *   - This is implemented by substituting zero-valued words when neighbors
 *     would fall outside the valid row/column range.
 */
template <int WPT>
__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Group index along X: each group covers WPT consecutive words.
    int group_x = blockIdx.x * blockDim.x + threadIdx.x;
    int row     = blockIdx.y * blockDim.y + threadIdx.y;

    // Starting word index in this row for this thread.
    int word_base = group_x * WPT;

    if (row >= grid_dim || word_base >= words_per_row)
        return;

    // Number of words actually processed by this thread (handles row-end).
    int words_this_thread = words_per_row - word_base;
    if (words_this_thread > WPT)
        words_this_thread = WPT;

    // Base linear index in the input/output arrays.
    int idx_base = row * words_per_row + word_base;

    // We load three rows of data: row-1, row, row+1.
    // For each row we load WPT+2 words: one word left of our first word,
    // WPT center words, and one right of our last word.
    //
    // row_*[0]         : left neighbor of first word (or 0 at left boundary)
    // row_*[1..WPT]    : center words to be updated
    // row_*[WPT+1]     : right neighbor of last word (or 0 at right boundary)
    uint64_t row_up[WPT + 2];
    uint64_t row_mid[WPT + 2];
    uint64_t row_dn[WPT + 2];

    // Load row above (row-1) or fill with zeros at top boundary.
    if (row > 0) {
        int idx_up_base = idx_base - words_per_row;

        // Left neighbor word for the group.
        if (word_base > 0) {
            row_up[0] = input[idx_up_base - 1];
        } else {
            row_up[0] = 0ull;
        }

        // Center words.
        #pragma unroll
        for (int i = 0; i < words_this_thread; ++i) {
            row_up[1 + i] = input[idx_up_base + i];
        }

        // Right neighbor word for the group.
        int right_word_index = word_base + words_this_thread;
        if (right_word_index < words_per_row) {
            row_up[1 + words_this_thread] = input[idx_up_base + words_this_thread];
        } else {
            row_up[1 + words_this_thread] = 0ull;
        }
    } else {
        // No row above: all neighbors from above are dead.
        #pragma unroll
        for (int i = 0; i < WPT + 2; ++i) {
            row_up[i] = 0ull;
        }
    }

    // Load current row.
    {
        // Left neighbor word for the group.
        if (word_base > 0) {
            row_mid[0] = input[idx_base - 1];
        } else {
            row_mid[0] = 0ull;
        }

        // Center words.
        #pragma unroll
        for (int i = 0; i < words_this_thread; ++i) {
            row_mid[1 + i] = input[idx_base + i];
        }

        // Right neighbor word for the group.
        int right_word_index = word_base + words_this_thread;
        if (right_word_index < words_per_row) {
            row_mid[1 + words_this_thread] = input[idx_base + words_this_thread];
        } else {
            row_mid[1 + words_this_thread] = 0ull;
        }
    }

    // Load row below (row+1) or fill with zeros at bottom boundary.
    if (row + 1 < grid_dim) {
        int idx_dn_base = idx_base + words_per_row;

        // Left neighbor word for the group.
        if (word_base > 0) {
            row_dn[0] = input[idx_dn_base - 1];
        } else {
            row_dn[0] = 0ull;
        }

        // Center words.
        #pragma unroll
        for (int i = 0; i < words_this_thread; ++i) {
            row_dn[1 + i] = input[idx_dn_base + i];
        }

        // Right neighbor word for the group.
        int right_word_index = word_base + words_this_thread;
        if (right_word_index < words_per_row) {
            row_dn[1 + words_this_thread] = input[idx_dn_base + words_this_thread];
        } else {
            row_dn[1 + words_this_thread] = 0ull;
        }
    } else {
        // No row below: all neighbors from below are dead.
        #pragma unroll
        for (int i = 0; i < WPT + 2; ++i) {
            row_dn[i] = 0ull;
        }
    }

    // Process each word handled by this thread.
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        if (i >= words_this_thread)
            break;

        // For word i, take a 3x3 neighborhood of words:
        // [row_up[i],   row_up[i+1],   row_up[i+2]]
        // [row_mid[i],  row_mid[i+1],  row_mid[i+2]]
        // [row_dn[i],   row_dn[i+1],   row_dn[i+2]]
        uint64_t L_up   = row_up[i];
        uint64_t C_up   = row_up[i + 1];
        uint64_t R_up   = row_up[i + 2];

        uint64_t L_mid  = row_mid[i];
        uint64_t C_mid  = row_mid[i + 1]; // current word (cells to update)
        uint64_t R_mid  = row_mid[i + 2];

        uint64_t L_dn   = row_dn[i];
        uint64_t C_dn   = row_dn[i + 1];
        uint64_t R_dn   = row_dn[i + 2];

        // Construct bitboards for the eight neighbor directions for each cell
        // in C_mid, using bit shifts and adjacent words to handle boundaries
        // between 64-bit words.
        //
        // Vertical neighbors:
        uint64_t nN  = C_up;  // one row above, same column
        uint64_t nS  = C_dn;  // one row below, same column

        // Horizontal neighbors on the same row:
        // For each bit position j in C_mid:
        //   left neighbor  is bit j-1 in (L_mid, C_mid)
        //   right neighbor is bit j+1 in (C_mid, R_mid)
        uint64_t left_mid  = (C_mid << 1) | (L_mid >> 63);
        uint64_t right_mid = (C_mid >> 1) | (R_mid << 63);
        uint64_t nW  = left_mid;
        uint64_t nE  = right_mid;

        // Diagonal neighbors from row above:
        uint64_t left_up   = (C_up << 1) | (L_up >> 63);
        uint64_t right_up  = (C_up >> 1) | (R_up << 63);
        uint64_t nNW = left_up;
        uint64_t nNE = right_up;

        // Diagonal neighbors from row below:
        uint64_t left_dn   = (C_dn << 1) | (L_dn >> 63);
        uint64_t right_dn  = (C_dn >> 1) | (R_dn << 63);
        uint64_t nSW = left_dn;
        uint64_t nSE = right_dn;

        // Count neighbors per bit into four bitplanes (b1, b2, b4, b8).
        uint64_t b1, b2, b4, b8;
        count_neighbors_bitplanes(nN, nS, nE, nW, nNE, nNW, nSE, nSW,
                                  b1, b2, b4, b8);

        // Next state rules:
        //   neighbors == 3 -> cell becomes alive
        //   neighbors == 2 -> cell stays as-is (alive if currently alive)
        //
        // neighbors in binary: n = b1 + 2*b2 + 4*b4 + 8*b8
        //   n == 2 -> (b8,b4,b2,b1) = (0,0,1,0)
        //   n == 3 -> (b8,b4,b2,b1) = (0,0,1,1)
        //
        // eq3 = !b8 & !b4 & b2 & b1
        // eq2 = !b8 & !b4 & b2 & !b1
        uint64_t not_b4_or_b8 = ~(b4 | b8);
        uint64_t eq3 = not_b4_or_b8 & b2 &  b1;
        uint64_t eq2 = not_b4_or_b8 & b2 & ~b1;

        // Apply Conway's rules.
        // next = (neighbors == 3) | (alive & neighbors == 2)
        uint64_t next = eq3 | (C_mid & eq2);

        output[idx_base + i] = next;
    }
}

/*
 * Host function to run one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *   input           - device pointer to input grid (bit-packed, uint64_t per 64 cells)
 *   output          - device pointer to output grid (same layout as input)
 *   grid_dimensions - width and height of the square grid (power of two, > 512)
 *
 * Notes:
 *   - The grid is assumed to be square: grid_dimensions x grid_dimensions.
 *   - Each row has grid_dimensions / 64  64-bit words.
 *   - Input and output arrays are assumed to have been allocated with cudaMalloc.
 *   - This function launches the CUDA kernel but does not synchronize; any
 *     necessary synchronization is handled by the caller.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Reasonable block configuration for modern data-center GPUs (A100/H100).
    // 256 threads per block: 32 (X) * 8 (Y).
    dim3 block(32, 8);

    // Each thread processes WORDS_PER_THREAD words horizontally.
    int groups_per_row = (words_per_row + WORDS_PER_THREAD * block.x - 1)
                         / (WORDS_PER_THREAD * block.x);

    dim3 grid(groups_per_row,
              (grid_dimensions + block.y - 1) / block.y);

    game_of_life_kernel<WORDS_PER_THREAD>
        <<<grid, block>>>(reinterpret_cast<const uint64_t*>(input),
                          reinterpret_cast<uint64_t*>(output),
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
