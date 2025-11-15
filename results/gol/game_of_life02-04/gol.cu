// Conway's Game of Life - CUDA implementation optimized for bit-packed grids.
//
// Each cell is stored as a single bit in a 64-bit word (std::uint64_t).
// - Bit value 1: cell is alive.
// - Bit value 0: cell is dead.
// 64 consecutive cells in a row are packed into one 64-bit word.
// The grid is square (grid_dimensions x grid_dimensions), with grid_dimensions
// being a power-of-two greater than 512. Thus:
//   words_per_row = grid_dimensions / 64
//
// This implementation performs one Game of Life step entirely on the GPU,
// operating directly on the bit-packed representation without using shared
// or texture memory.
//
// Mapping of work:
//   - One thread processes one 64-bit word (i.e., 64 cells in a row).
//   - Threads are launched in a 1D grid over all words in the grid.
//   - For each word, the thread loads up to 9 words from the input:
//       the word itself, its left and right neighbors in the same row,
//       and the three corresponding words in the row above and below.
//   - From these 9 words, the thread computes 8 neighbor bitboards
//       (N, NE, E, SE, S, SW, W, NW), each indicating which neighbor
//       is alive for each of the 64 cells.
//   - It then accumulates the 8 neighbor bitboards into a 4-bit integer
//       per cell (stored as 4 separate 64-bit bitplanes).
//   - Using these per-cell neighbor counts, it applies the Game of Life
//       rules in bit-parallel fashion to produce the next generation word.
//
// Coordinate system within a word:
//   - For a given 64-bit word, bit index i (0 <= i <= 63) corresponds to
//     column (word_column * 64 + i) in the grid row.
//   - Bit index 0 is the least significant bit (LSB).
//
// Neighbor mapping for horizontal shifts:
//   - West neighbor (column-1):
//       For bit i > 0: comes from bit (i-1) in the same word.
//       For bit i == 0: comes from bit 63 in the left neighbor word.
//     west = (center << 1) | (left >> 63)
//
//   - East neighbor (column+1):
//       For bit i < 63: comes from bit (i+1) in the same word.
//       For bit i == 63: comes from bit 0 in the right neighbor word.
//     east = (center >> 1) | (right << 63)
//
// Vertical neighbors:
//   - North (row-1): same bit index in the word directly above.
//   - South (row+1): same bit index in the word directly below.
//   - NE, NW, SE, SW are derived by combining above/below with horizontal
//     shifts as above.
//
// Boundary conditions:
//   - All cells outside the grid are treated as dead.
//   - For words at the left/right edges, missing neighbors are 0.
//   - For words in the top/bottom rows, missing neighbors are 0.
//
// Neighbor count accumulation:
//   - We compute eight 64-bit bitboards for neighbor directions:
//       n, ne, e, se, s, sw, w, nw
//   - For each cell (bit position), these eight bits are the neighbor states.
//   - We use a bit-parallel binary adder network to accumulate the eight bits
//     into a 4-bit integer per cell, stored across four bitplanes:
//       c0 = bit 0 (LSB), c1 = bit 1, c2 = bit 2, c3 = bit 3.
//     The maximum possible count is 8 (1000b), so 4 bits are sufficient.
//   - The accumulator is updated by repeatedly adding a 1-bit value (bitboard)
//     using a ripple-carry adder implemented with bitwise operations.
//
// Game of Life rules (per cell):
//   Let alive be 1 if the cell is alive in the current generation.
//   Let neighbors be the number of alive neighbors (0..8).
//   - If alive:
//       survives if neighbors == 2 or neighbors == 3
//   - If dead:
//       becomes alive if neighbors == 3
//
// In bit-parallel form:
//   - eq2: bitboard where neighbor count == 2
//   - eq3: bitboard where neighbor count == 3
//   - next = eq3 | (alive & eq2)
//     (cells with exactly 3 neighbors become alive;
//      cells that are alive with 2 neighbors survive)
//
// Performance considerations:
//   - Global memory access is coalesced because threads process consecutive
//     64-bit words.
//   - No shared or texture memory is used (as requested).
//   - The kernel uses only simple bitwise operations and a small number of
//     integer operations per thread, which is efficient on modern GPUs.
//
// Host function:
//   void run_game_of_life(const std::uint64_t* input,
//                         std::uint64_t*       output,
//                         int                  grid_dimensions);
//   - input, output are device pointers allocated with cudaMalloc.
//   - Executes a single Game of Life step.
//   - Does not perform any host-device synchronization; the caller is
//     responsible for synchronization if needed.

#include <cstdint>
#include <cuda_runtime.h>

using u64 = std::uint64_t;

//------------------------------------------------------------------------------
// Device helper: add a 1-bit bitboard into a 4-bit per-cell counter.
//
// The counter is stored across four 64-bit bitplanes (c0..c3), representing
// the bits of an integer count per cell:
//   count = c0*1 + c1*2 + c2*4 + c3*8
//
// 'b' is a bitboard where each bit is 0 or 1 and is added as +1 in that lane.
//
// This is a bit-parallel ripple-carry adder for all 64 lanes at once.
// The maximum count is 8, so 4 bits suffice and no overflow beyond bit 3 occurs.
//------------------------------------------------------------------------------
__device__ __forceinline__
void add_bitboard(u64 b, u64 &c0, u64 &c1, u64 &c2, u64 &c3)
{
    // Add to bit 0 and get carry into bit 1
    u64 carry0 = c0 & b;
    c0 ^= b;

    // Propagate into bit 1
    u64 carry1 = c1 & carry0;
    c1 ^= carry0;

    // Propagate into bit 2
    u64 carry2 = c2 & carry1;
    c2 ^= carry1;

    // Propagate into bit 3 (no need to propagate further; max count is 8)
    c3 ^= carry2;
}

//------------------------------------------------------------------------------
// CUDA kernel: compute one Game of Life step on a bit-packed grid.
//
// Parameters:
//   input            - pointer to device input grid (bit-packed)
//   output           - pointer to device output grid (bit-packed)
//   grid_dim         - width/height of the square grid (power of 2)
//   words_per_row    - number of 64-bit words per row = grid_dim / 64
//   log2_words_per_row - log2(words_per_row); used to decode (row, col)
//                        from a flat word index without division.
//------------------------------------------------------------------------------
__global__
void game_of_life_kernel(const u64* __restrict__ input,
                         u64*       __restrict__ output,
                         int        grid_dim,
                         int        words_per_row,
                         int        log2_words_per_row)
{
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dim) *
        static_cast<std::size_t>(words_per_row);

    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        static_cast<std::size_t>(threadIdx.x);

    if (idx >= total_words)
        return;

    // Decode row and column index for this word.
    // Since words_per_row is a power of two:
    //   row = idx / words_per_row = idx >> log2_words_per_row
    //   col = idx % words_per_row = idx & (words_per_row - 1)
    const std::size_t words_per_row_sz = static_cast<std::size_t>(words_per_row);
    const std::size_t row = idx >> log2_words_per_row;
    const std::size_t col = idx & (words_per_row_sz - 1);

    const int row_int = static_cast<int>(row);
    const int col_int = static_cast<int>(col);

    // Base indices for this row and its neighbors
    const std::size_t row_base   = row * words_per_row_sz;
    const bool has_row_above     = (row_int > 0);
    const bool has_row_below     = (row_int + 1 < grid_dim);
    const bool has_col_left      = (col_int > 0);
    const bool has_col_right     = (col_int + 1 < words_per_row);

    // Pointers to the three relevant rows
    const u64* row_ptr        = input + row_base;
    const u64* row_above_ptr  = has_row_above ? (row_ptr - words_per_row_sz) : nullptr;
    const u64* row_below_ptr  = has_row_below ? (row_ptr + words_per_row_sz) : nullptr;

    // Load center word (current row, this column)
    const u64 center = row_ptr[col_int];

    // Load left and right neighbors in the same row (0 if out of bounds)
    const u64 left_center  = has_col_left  ? row_ptr[col_int - 1] : 0ull;
    const u64 right_center = has_col_right ? row_ptr[col_int + 1] : 0ull;

    // Load words from the row above (or 0 if out of bounds)
    const u64 above_center =
        has_row_above ? row_above_ptr[col_int] : 0ull;
    const u64 above_left =
        (has_row_above && has_col_left)  ? row_above_ptr[col_int - 1] : 0ull;
    const u64 above_right =
        (has_row_above && has_col_right) ? row_above_ptr[col_int + 1] : 0ull;

    // Load words from the row below (or 0 if out of bounds)
    const u64 below_center =
        has_row_below ? row_below_ptr[col_int] : 0ull;
    const u64 below_left =
        (has_row_below && has_col_left)  ? row_below_ptr[col_int - 1] : 0ull;
    const u64 below_right =
        (has_row_below && has_col_right) ? row_below_ptr[col_int + 1] : 0ull;

    // Compute the 8 neighbor direction bitboards.

    // North, South (same column bits from neighboring rows)
    const u64 north = above_center;
    const u64 south = below_center;

    // East and West (horizontal neighbors within same row, with cross-word carry)
    const u64 east  = (center >> 1) | (right_center << 63);
    const u64 west  = (center << 1) | (left_center  >> 63);

    // Diagonals: NE, NW, SE, SW
    const u64 north_east = (above_center >> 1) | (above_right << 63);
    const u64 north_west = (above_center << 1) | (above_left  >> 63);
    const u64 south_east = (below_center >> 1) | (below_right << 63);
    const u64 south_west = (below_center << 1) | (below_left  >> 63);

    // Accumulate neighbor counts using a 4-bit per-cell counter.
    u64 c0 = 0ull;  // bit 0 of neighbor count
    u64 c1 = 0ull;  // bit 1
    u64 c2 = 0ull;  // bit 2
    u64 c3 = 0ull;  // bit 3

    add_bitboard(north,      c0, c1, c2, c3);
    add_bitboard(south,      c0, c1, c2, c3);
    add_bitboard(east,       c0, c1, c2, c3);
    add_bitboard(west,       c0, c1, c2, c3);
    add_bitboard(north_east, c0, c1, c2, c3);
    add_bitboard(north_west, c0, c1, c2, c3);
    add_bitboard(south_east, c0, c1, c2, c3);
    add_bitboard(south_west, c0, c1, c2, c3);

    // Compute masks for neighbor count == 2 and neighbor count == 3.
    //
    // Binary representations:
    //   2 = 0b0010 -> (c3=0, c2=0, c1=1, c0=0)
    //   3 = 0b0011 -> (c3=0, c2=0, c1=1, c0=1)
    //
    // Therefore:
    //   eq2 = (~c3) & (~c2) &  c1  & (~c0)
    //   eq3 = (~c3) & (~c2) &  c1  &  c0
    const u64 not_c3 = ~c3;
    const u64 not_c2 = ~c2;

    const u64 eq2 = not_c3 & not_c2 &  c1  & (~c0);
    const u64 eq3 = not_c3 & not_c2 &  c1  &  c0;

    // Alive cells in the current generation.
    const u64 alive = center;

    // Game of Life rule:
    //   next = eq3 | (alive & eq2)
    // (cells with exactly 3 neighbors become alive;
    //  cells that are alive with exactly 2 neighbors survive)
    const u64 next = eq3 | (alive & eq2);

    // Store the result.
    output[idx] = next;
}

//------------------------------------------------------------------------------
// Host function: run one Game of Life step on the GPU.
//
// Parameters:
//   input            - device pointer to input grid (bit-packed), allocated by cudaMalloc
//   output           - device pointer to output grid (bit-packed), allocated by cudaMalloc
//   grid_dimensions  - width/height of the square grid (power-of-two > 512)
//
// Notes:
//   - This function launches a CUDA kernel but does not perform cudaDeviceSynchronize.
//     The caller is responsible for synchronization if needed.
//   - The function assumes that the total number of cells is small enough that both
//     input and output arrays fit in GPU memory.
//------------------------------------------------------------------------------
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t*       output,
                      int                  grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    // grid_dimensions is a power of two, so grid_dimensions / 64 is also a power of two.
    const int words_per_row = grid_dimensions >> 6;

    // Compute log2(words_per_row) on the host. This is used in the kernel to decode
    // row and column indices from a flat word index without division.
    int log2_words_per_row = 0;
    {
        int x = words_per_row;
        while (x > 1) {
            x >>= 1;
            ++log2_words_per_row;
        }
    }

    const std::size_t total_words =
        static_cast<std::size_t>(grid_dimensions) *
        static_cast<std::size_t>(words_per_row);

    // Configure CUDA launch parameters.
    // Use a 1D grid over all words. A block size of 256 threads is a good
    // general choice for modern data center GPUs (A100/H100).
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_words + block_size - 1) / block_size
    );

    // Launch the kernel. Synchronization (if desired) is handled by the caller.
    game_of_life_kernel<<<grid_size, block_size>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
        log2_words_per_row
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
