#include <cuda_runtime.h>
#include <cstdint>

/*
 * CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
 *
 * - The grid is square with size grid_dimensions x grid_dimensions.
 * - Each row is stored as a sequence of std::uint64_t words.
 *   Each word encodes 64 consecutive cells in the row, bit 0 = leftmost cell.
 * - The input and output arrays are laid out row-major:
 *     word_index = row * row_words + word_in_row
 *   where row_words = ceil(grid_dimensions / 64.0).
 *
 * - Each CUDA thread processes one 64-bit word (i.e., 64 cells in a row).
 * - All cells outside the grid are considered dead.
 *
 * Neighbor computation:
 *
 * For a given word 'c' at (row, col_word), we load up to 8 neighboring words:
 *   Row above: n_w, n, n_e
 *   Same row:  c_w,    c, c_e
 *   Row below: s_w, s, s_e
 *
 * where *_w is the word to the left, *_e to the right.
 *
 * Horizontal shifts across word boundaries are handled with helper functions:
 *   shift_left(left, center)  -> one-cell shift towards higher column index
 *   shift_right(center, right)-> one-cell shift towards lower column index
 *
 * Example: same-row left neighbor mask:
 *   same_left = shift_left(c_w, c);
 * For each bit position j (0..63), same_left[j] is 1 iff the cell at column j-1
 * in the same row is alive. For j=0, it uses bit 63 of c_w.
 *
 * We build eight 64-bit masks for the neighbors of each bit in the central word:
 *   above_left, above, above_right,
 *   same_left,             same_right,
 *   below_left, below, below_right
 *
 * We then compute the per-cell neighbor count in a bit-sliced way: each bit
 * position is a separate 4-bit counter stored as four 64-bit masks (s0..s3):
 *   neighbor_count = s0*1 + s1*2 + s2*4 + s3*8
 * using a small "add 1-bit mask into 4-bit counter" routine implemented with
 * bitwise operations (no cross-bit carries).
 *
 * Finally, we apply the Game of Life rules:
 *   - A live cell survives if it has 2 or 3 live neighbors.
 *   - A dead cell becomes alive if it has exactly 3 live neighbors.
 *
 * Using the bit-sliced count, for every cell:
 *   neighbors == 2 or 3  <=>  s3==0, s2==0, s1==1          (s0 don't care)
 *   neighbors == 3       <=>  s3==0, s2==0, s1==1, s0==1
 *
 * Thus:
 *   eq23 = ~s3 & ~s2 & s1;
 *   eq3  = eq23 & s0;
 *   next = (current & eq23) | (~current & eq3);
 *
 * For the last word in each row when the grid width is not a multiple of 64,
 * we use a mask last_word_mask such that bits >= grid_width are forced dead
 * in both input and output, and are ignored as neighbors.
 */


/* Bitwise one-cell shift to the "left" (towards higher bit index / column).
 * 'left'   : word immediately to the left of 'center' in the row
 * 'center' : current word in the row
 *
 * Returns a word where bit j corresponds to the cell at column j-1 in the row:
 *  - For j>0: comes from center bit j-1
 *  - For j==0: comes from left bit 63
 */
static __device__ __forceinline__ std::uint64_t
shift_left(std::uint64_t left, std::uint64_t center)
{
    return (center << 1) | (left >> 63);
}

/* Bitwise one-cell shift to the "right" (towards lower bit index / column).
 * 'right'  : word immediately to the right of 'center' in the row
 * 'center' : current word in the row
 *
 * Returns a word where bit j corresponds to the cell at column j+1 in the row:
 *  - For j<63: comes from center bit j+1
 *  - For j==63: comes from right bit 0
 */
static __device__ __forceinline__ std::uint64_t
shift_right(std::uint64_t center, std::uint64_t right)
{
    return (center >> 1) | (right << 63);
}

/* Add a 1-bit mask 'x' into a 4-bit per-cell counter (s0,s1,s2,s3).
 *
 * Each of s0,s1,s2,s3 is a 64-bit word; bit i of these four words together
 * store a 4-bit integer count (0..8) for cell i:
 *   count_i = s0_i*1 + s1_i*2 + s2_i*4 + s3_i*8
 *
 * This routine increments the count by 'x' (0 or 1) in each bit position,
 * implemented as a ripple-carry adder using bitwise operations. Carries do
 * not propagate across bit positions, only vertically within the 4-bit slice.
 */
static __device__ __forceinline__ void
add_bit_to_counter(std::uint64_t x,
                   std::uint64_t &s0,
                   std::uint64_t &s1,
                   std::uint64_t &s2,
                   std::uint64_t &s3)
{
    // Add to bit 0
    std::uint64_t c0 = s0 & x;
    s0 ^= x;

    // Propagate carry to bit 1
    std::uint64_t c1 = s1 & c0;
    s1 ^= c0;

    // Propagate carry to bit 2
    std::uint64_t c2 = s2 & c1;
    s2 ^= c1;

    // Propagate carry to bit 3 (no overflow beyond bit 3 for max count 8)
    s3 ^= c2;
}

/* CUDA kernel: compute one Game of Life step on a bit-packed grid.
 *
 * Parameters:
 *   input          - pointer to device memory, bit-packed input grid
 *   output         - pointer to device memory, bit-packed output grid
 *   grid_dim       - grid width and height (square grid)
 *   row_words      - number of 64-bit words per row
 *   last_word_mask - bitmask for the last word in each row.
 *                    If grid_dim % 64 == 0, last_word_mask == ~0ULL.
 *                    Otherwise, only the lowest (grid_dim % 64) bits are 1.
 */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int row_words,
                                    std::uint64_t last_word_mask)
{
    int word_x = blockIdx.x * blockDim.x + threadIdx.x; // word index within row
    int row    = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (row >= grid_dim || word_x >= row_words)
        return;

    const int last_word_idx = row_words - 1;

    const bool is_top    = (row == 0);
    const bool is_bottom = (row == grid_dim - 1);
    const bool is_left   = (word_x == 0);
    const bool is_right  = (word_x == last_word_idx);

    const int row_offset      = row * row_words;
    const int row_above_off   = row_offset - (is_top    ? 0 : row_words);
    const int row_below_off   = row_offset + (is_bottom ? 0 : row_words);

    // Load central word and same-row neighbors
    std::uint64_t c   = input[row_offset + word_x];
    std::uint64_t c_w = is_left  ? 0 : input[row_offset + word_x - 1];
    std::uint64_t c_e = is_right ? 0 : input[row_offset + word_x + 1];

    // Load words from the row above (if it exists)
    std::uint64_t n   = 0;
    std::uint64_t n_w = 0;
    std::uint64_t n_e = 0;
    if (!is_top) {
        n   = input[row_above_off + word_x];
        n_w = is_left  ? 0 : input[row_above_off + word_x - 1];
        n_e = is_right ? 0 : input[row_above_off + word_x + 1];
    }

    // Load words from the row below (if it exists)
    std::uint64_t s   = 0;
    std::uint64_t s_w = 0;
    std::uint64_t s_e = 0;
    if (!is_bottom) {
        s   = input[row_below_off + word_x];
        s_w = is_left  ? 0 : input[row_below_off + word_x - 1];
        s_e = is_right ? 0 : input[row_below_off + word_x + 1];
    }

    // If this is the last word in the row and the row is not a multiple of 64,
    // mask off bits that are outside the grid. This ensures that they are
    // treated as dead and do not influence neighbor counts.
    if (is_right && last_word_mask != ~std::uint64_t(0)) {
        c &= last_word_mask;
        if (!is_top)    n &= last_word_mask;
        if (!is_bottom) s &= last_word_mask;
    }

    // Build neighbor masks (8 directions), aligned to the central word bits.
    // Top row neighbors
    std::uint64_t above_left  = 0;
    std::uint64_t above       = 0;
    std::uint64_t above_right = 0;
    if (!is_top) {
        above       = n;
        above_left  = shift_left(n_w, n);
        above_right = shift_right(n, n_e);
    }

    // Same row neighbors (left and right)
    std::uint64_t same_left  = shift_left(c_w, c);
    std::uint64_t same_right = shift_right(c, c_e);

    // Bottom row neighbors
    std::uint64_t below_left  = 0;
    std::uint64_t below       = 0;
    std::uint64_t below_right = 0;
    if (!is_bottom) {
        below       = s;
        below_left  = shift_left(s_w, s);
        below_right = shift_right(s, s_e);
    }

    // Bit-sliced neighbor count for each bit in the central word.
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;

    // Accumulate the 8 neighbor masks into the 4-bit counters
    add_bit_to_counter(above_left,  s0, s1, s2, s3);
    add_bit_to_counter(above,       s0, s1, s2, s3);
    add_bit_to_counter(above_right, s0, s1, s2, s3);
    add_bit_to_counter(same_left,   s0, s1, s2, s3);
    add_bit_to_counter(same_right,  s0, s1, s2, s3);
    add_bit_to_counter(below_left,  s0, s1, s2, s3);
    add_bit_to_counter(below,       s0, s1, s2, s3);
    add_bit_to_counter(below_right, s0, s1, s2, s3);

    // Compute masks for "neighbors == 2 or 3" and "neighbors == 3".
    // neighbors == 2 or 3  <=> s3==0, s2==0, s1==1
    // neighbors == 3       <=> s3==0, s2==0, s1==1, s0==1
    std::uint64_t not_s3 = ~s3;
    std::uint64_t not_s2 = ~s2;

    std::uint64_t eq23 = not_s3 & not_s2 & s1;
    std::uint64_t eq3  = eq23 & s0;

    // Apply Game of Life rules:
    //   next = (alive & (neighbors==2 or 3)) | (dead & neighbors==3)
    std::uint64_t survive = c & eq23;
    std::uint64_t born    = (~c) & eq3;
    std::uint64_t next    = survive | born;

    // Mask off invalid bits for the last word in each row
    if (is_right && last_word_mask != ~std::uint64_t(0)) {
        next &= last_word_mask;
    }

    output[row_offset + word_x] = next;
}

/*
 * Host function: run one step of Conway's Game of Life on the GPU.
 *
 * Parameters:
 *   input           - device pointer to bit-packed input grid
 *   output          - device pointer to bit-packed output grid
 *   grid_dimensions - width and height of the square grid (power of 2)
 *
 * Both input and output must have been allocated with cudaMalloc.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int grid_dim  = grid_dimensions;
    const int row_words = (grid_dim + 63) / 64;

    // Compute mask for the last word in each row.
    // If grid_dim is a multiple of 64, all bits are valid.
    std::uint64_t last_word_mask;
    int rem = grid_dim & 63;
    if (rem == 0) {
        last_word_mask = ~std::uint64_t(0);
    } else {
        last_word_mask = (rem == 64) ? ~std::uint64_t(0)
                                     : ((std::uint64_t(1) << rem) - 1);
    }

    // Launch configuration: 2D grid with one thread per 64-bit word.
    // Use a 1D block in x for coalesced accesses along rows.
    dim3 block(256, 1, 1);
    dim3 grid((row_words + block.x - 1) / block.x,
              (grid_dim   + block.y - 1) / block.y,
              1);

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dim, row_words, last_word_mask);
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
