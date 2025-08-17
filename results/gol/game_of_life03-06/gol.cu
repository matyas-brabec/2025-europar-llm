#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
  CUDA implementation of a single Conway's Game of Life step on a bit-packed grid.

  Data layout:
  - The grid is square with dimension N = grid_dimensions (power of 2, > 512).
  - Each row has W = N / 64 64-bit words.
  - Each std::uint64_t packs 64 consecutive cells in a row.
    Bit 0 is the leftmost cell within the word; bit 63 is the rightmost.

  Thread mapping:
  - One CUDA thread processes one 64-bit word (i.e., 64 cells).
  - The kernel uses a 1D grid-stride loop to process all words.

  Neighborhood handling:
  - Cells outside the grid are considered dead (zero).
  - To compute neighbors across word boundaries, we inject boundary bits from
    the left/right neighbor words when shifting, but clamp at grid edges.

  Algorithm:
  - For each of the three relevant rows (above, current, below), compute the three
    horizontally shifted bitboards: left-shifted, unshifted (center), and right-shifted.
    For the current row, we exclude the center (self) because the cell itself is not a neighbor.
    For above and below rows we include center (above and below neighbors).
  - Use bit-sliced addition to sum the three horizontal contributions per row. This yields
    two bitboards per row: lo (LSB) and hi (MSB), representing counts in {0..3}.
    This addition uses boolean logic without carry propagation across bit positions.
  - Sum the three rows' 2-bit counts (above, current, below) using bit-sliced addition
    to produce a per-bit neighbor count representation across four weight bitboards:
    w1 (1's place), w2 (2's), w4 (4's), w8 (8's). The 3x3 neighborhood count is in 0..8.
  - Apply Game of Life rules:
      next = (count == 3) | (alive & (count == 2))
    Using the bit-sliced representation:
      count == 2  -> w2=1, w1=0, w4=0, w8=0
      count == 3  -> w2=1, w1=1, w4=0, w8=0

  Notes:
  - No shared or texture memory is used; all operations are on registers and global memory.
  - Loads are read-only; on modern GPUs, the compiler will use the appropriate cache path.
*/

#if __CUDA_ARCH__ >= 350
  #define LDG64(ptr) __ldg(ptr)
#else
  #define LDG64(ptr) (*(ptr))
#endif

// Bit-sliced 3-input adder for 64-wide bit-vectors.
// For each bit position k:
//   sum[k]   = a[k] XOR b[k] XOR c[k]            (LSB of count)
//   carry[k] = majority(a[k], b[k], c[k])        (MSB of count, value 1 means +2)
static __device__ __forceinline__
void add3_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
              std::uint64_t& sum, std::uint64_t& carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// The main kernel: computes one generation of Game of Life.
// in/out: bit-packed grid as described above.
// words_per_row: number of 64-bit words per row (grid_dimensions / 64).
// rows: number of rows (grid_dimensions).
// row_shift: log2(words_per_row), since words_per_row is a power of two.
// total_words: total number of 64-bit words in the grid.
static __global__
void life_kernel(const std::uint64_t* __restrict__ in,
                 std::uint64_t* __restrict__ out,
                 int words_per_row, int rows, int row_shift, std::size_t total_words)
{
    // Grid-stride loop over all words
    for (std::size_t idx = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
         idx < total_words;
         idx += (std::size_t)blockDim.x * gridDim.x)
    {
        // Compute 2D word coordinates (row, col) using power-of-two arithmetic.
        const std::size_t row = idx >> row_shift;
        const std::size_t col = idx & (std::size_t)(words_per_row - 1);

        // Boundary flags
        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < (std::size_t)words_per_row);
        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < (std::size_t)rows);

        // Precompute neighbor word indices for vertical moves
        const std::size_t idx_up   = idx - (std::size_t)words_per_row; // only valid if has_up
        const std::size_t idx_down = idx + (std::size_t)words_per_row; // only valid if has_down

        // Load current row words (current, left, right)
        const std::uint64_t c  = LDG64(in + idx);
        const std::uint64_t cL = has_left  ? LDG64(in + (idx - 1)) : 0ULL;
        const std::uint64_t cR = has_right ? LDG64(in + (idx + 1)) : 0ULL;

        // Load above row words (above, above-left, above-right)
        const std::uint64_t a  = has_up ? LDG64(in + idx_up) : 0ULL;
        const std::uint64_t aL = (has_up && has_left)  ? LDG64(in + (idx_up - 1)) : 0ULL;
        const std::uint64_t aR = (has_up && has_right) ? LDG64(in + (idx_up + 1)) : 0ULL;

        // Load below row words (below, below-left, below-right)
        const std::uint64_t b  = has_down ? LDG64(in + idx_down) : 0ULL;
        const std::uint64_t bL = (has_down && has_left)  ? LDG64(in + (idx_down - 1)) : 0ULL;
        const std::uint64_t bR = (has_down && has_right) ? LDG64(in + (idx_down + 1)) : 0ULL;

        // Horizontal shifts with cross-word bit injection.
        // For each row R in {a, c, b}:
        //   R_left  = (R << 1) with bit0 injected from MSB of neighbor-left word.
        //   R_right = (R >> 1) with bit63 injected from LSB of neighbor-right word.
        const std::uint64_t c_left  = (c << 1) | (cL >> 63);
        const std::uint64_t c_right = (c >> 1) | ((cR & 1ULL) << 63);

        const std::uint64_t a_left  = (a << 1) | (aL >> 63);
        const std::uint64_t a_right = (a >> 1) | ((aR & 1ULL) << 63);

        const std::uint64_t b_left  = (b << 1) | (bL >> 63);
        const std::uint64_t b_right = (b >> 1) | ((bR & 1ULL) << 63);

        // For the current row, we must EXCLUDE the center (self).
        // So the horizontal triplet for current row is {left, 0, right}.
        // For above and below rows, include center.
        std::uint64_t rowA_lo, rowA_hi;
        add3_u64(a_left, a, a_right, rowA_lo, rowA_hi);             // Above row: contributions from UL, U, UR

        std::uint64_t rowC_lo, rowC_hi;
        add3_u64(c_left, 0ULL, c_right, rowC_lo, rowC_hi);          // Current row: contributions from L, R (no center)

        std::uint64_t rowB_lo, rowB_hi;
        add3_u64(b_left, b, b_right, rowB_lo, rowB_hi);             // Below row: contributions from DL, D, DR

        // Sum the three rows' 2-bit counts:
        // Sum of LSBs (weight 1 across rows)
        std::uint64_t loS0, loS1;
        add3_u64(rowA_lo, rowC_lo, rowB_lo, loS0, loS1);            // loS0: weight1, loS1: carry to weight2

        // Sum of MSBs (each is weight 2 per row, but we first sum as counts and then scale)
        std::uint64_t hiS0, hiS1;
        add3_u64(rowA_hi, rowC_hi, rowB_hi, hiS0, hiS1);            // hiS0: count in {0,1} ; hiS1: carry (>=2)
                                                                    // After scaling by 2: hiS0 contributes to weight2, hiS1 to weight4.

        // Combine weights:
        // w1 = loS0
        // w2 = loS1 + hiS0      (both at weight2; addition is bitwise with carry to weight4)
        // w4 = hiS1 + carry(w2)
        // w8 = carry from (w4 accumulation), although the range here ensures at most one carry source.
        const std::uint64_t w1 = loS0;

        const std::uint64_t w2   = loS1 ^ hiS0;
        const std::uint64_t c_w4 = loS1 & hiS0;

        const std::uint64_t w4   = hiS1 ^ c_w4;
        const std::uint64_t w8   = hiS1 & c_w4;

        // Apply Game of Life rules:
        // next = (count == 3) | (alive & (count == 2))
        const std::uint64_t not_w4 = ~w4;
        const std::uint64_t not_w8 = ~w8;

        const std::uint64_t eq2 = w2 & (~w1) & not_w4 & not_w8;
        const std::uint64_t eq3 = w2 & ( w1 ) & not_w4 & not_w8;

        const std::uint64_t next = eq3 | (c & eq2);

        out[idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute grid geometry for bit-packed layout.
    // words_per_row is a power of two since grid_dimensions is a power of two and we divide by 64.
    const int words_per_row = grid_dimensions >> 6;
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Compute row_shift = log2(words_per_row) using simple loop (words_per_row is power of two).
    int row_shift = 0;
    unsigned int wpr = static_cast<unsigned int>(words_per_row);
    while ((1u << row_shift) < wpr) { ++row_shift; }

    // Launch parameters: a good default for modern GPUs; grid-stride loop handles any size.
    const int threads = 256;
    int blocks = static_cast<int>((total_words + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    life_kernel<<<blocks, threads>>>(input, output, words_per_row, grid_dimensions, row_shift, total_words);
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
