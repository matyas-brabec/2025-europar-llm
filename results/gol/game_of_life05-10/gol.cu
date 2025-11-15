// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// 
// Representation:
//   - The grid is square: grid_dimensions x grid_dimensions.
//   - Each row is packed into 64-bit words: one std::uint64_t encodes 64 cells.
//   - Bit value 1 = alive, 0 = dead.
//   - Words are laid out in row-major order.
//   - The number of columns is always a power of two > 512, hence a multiple of 64,
//     so the number of words per row is grid_dimensions / 64.
//   - Cells outside the grid are treated as dead (zero).
//
// Parallelization strategy:
//   - Each CUDA thread processes exactly one 64-bit word (i.e., 64 cells) of the grid.
//   - No atomics are required, since each word is uniquely owned by a thread.
//
// Neighborhood computation:
//   - For a given word W at (row, col_word), we need cells from:
//       rows: row-1, row, row+1
//       word columns: col_word-1, col_word, col_word+1
//   - We load up to 9 words:
//       Nl, Nc, Nr  (row-1)
//       Cl, Cc, Cr  (row)
//       Sl, Sc, Sr  (row+1)
//   - Out-of-bounds rows/columns are treated as zero.
//
//   - From these words we derive the 8 neighbor masks for each of the 64 cells in Cc:
//       west       : (Cc >> 1) | (Cl << 63)
//       east       : (Cc << 1) | (Cr >> 63)
//       north      : Nc
//       south      : Sc
//       north_west : (Nc >> 1) | (Nl << 63)
//       north_east : (Nc << 1) | (Nr >> 63)
//       south_west : (Sc >> 1) | (Sl << 63)
//       south_east : (Sc << 1) | (Sr >> 63)
//
// Bit-parallel neighbor counting:
//   - For each of the 8 neighbor direction masks (64-bit each), we conceptually add
//     one to a per-bit neighbor counter where the neighbor has a 1.
//   - We maintain three bit-planes (ones, twos, fours) that encode the neighbor
//     count in 3-bit binary *modulo 8* at each bit position:
//         count = ones * 1 + twos * 2 + fours * 4   (mod 8)
//   - Starting from zero, we add each neighbor mask x with a 3-bit ripple-carry
//     adder, done in parallel for all 64 bits:
//
//       carry0 = ones & x;
//       ones   = ones ^ x;
//
//       carry1 = twos & carry0;
//       twos   = twos ^ carry0;
//
//       fours  = fours ^ carry1;
//       // Carry out of 'fours' would indicate count >= 8 (i.e., exactly 8 in
//       // this context), but we don't need it because Game of Life rules only
//       // care about counts 2 and 3. For counts of 8 neighbors, the modulo-8
//       // representation is 0, which is not mistaken as 2 or 3.
//
//   - After accumulating all 8 neighbors, each bit position's count c in 0..8
//     has the following encoding in (ones, twos, fours) (mod 8):
//         c = 0 -> 000, 1 -> 001, 2 -> 010, 3 -> 011,
//             4 -> 100, 5 -> 101, 6 -> 110, 7 -> 111, 8 -> 000
//
//   - We only need masks for counts == 2 and counts == 3:
//       eq2_mask = (~ones) &  twos  & (~fours)   // 010
//       eq3_mask =   ones  &  twos  & (~fours)   // 011
//
// Game of Life rule in bit form:
//   - Let 'alive' be the current state bits (Cc).
//   - Next state bit is 1 if:
//       - neighbor_count == 3 (eq3_mask), OR
//       - alive && neighbor_count == 2 (alive & eq2_mask).
//
//   -> next = eq3_mask | (alive & eq2_mask)
//
// Performance notes:
//   - Each thread linearly accesses neighboring words, resulting in coalesced
//     global memory loads when launched with a simple 1D grid over words.
//   - All neighbor counting is fully bit-parallel via integer logic; no per-cell
//     loops are used.
//   - Shared and texture memory are intentionally avoided per the problem
//     statement; global memory plus on-the-fly computation is sufficient.
//
// The run_game_of_life() function launches the kernel for one simulation step.
// Synchronization (e.g., cudaDeviceSynchronize) is intentionally not performed here
// and must be handled by the caller if needed.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

using std::uint64_t;

// Device helper: accumulate a neighbor mask into (ones, twos, fours) bit-planes.
//
// Each bit set in 'x' corresponds to a +1 increment of the neighbor count
// at that bit position. 'ones', 'twos', and 'fours' store the count in
// binary modulo 8 as described above.
//
// This is essentially a 3-bit ripple-carry addition of x to the current count.
__device__ __forceinline__
void accumulate_neighbor(uint64_t x, uint64_t &ones, uint64_t &twos, uint64_t &fours)
{
    // Add x to the ones bit-plane, record carry into 'twos'.
    uint64_t carry0 = ones & x;
    ones ^= x;

    // Add carry0 to the twos bit-plane, record carry into 'fours'.
    uint64_t carry1 = twos & carry0;
    twos ^= carry0;

    // Add carry1 to the fours bit-plane; ignore overflow beyond 4 (count==8),
    // because we never need to distinguish 8 from 0 for the Life rules.
    fours ^= carry1;
}

// Kernel: one thread per 64-bit word (64 cells).
__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dimensions,
                                    int words_per_row)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    if (tid >= total_words)
        return;

    // Compute 2D coordinates: row index and word-column index.
    const int row = static_cast<int>(tid / words_per_row);
    const int col = static_cast<int>(tid - static_cast<std::size_t>(row) * words_per_row);

    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < grid_dimensions);
    const bool has_left   = (col > 0);
    const bool has_right  = (col + 1 < words_per_row);

    // Load center row words.
    const uint64_t Cc = input[tid];
    const uint64_t Cl = has_left  ? input[tid - 1] : 0ull;
    const uint64_t Cr = has_right ? input[tid + 1] : 0ull;

    // Load north row words (row - 1).
    const std::size_t north_base = tid - static_cast<std::size_t>(words_per_row);
    const uint64_t Nc = has_top ? input[north_base] : 0ull;
    const uint64_t Nl = (has_top && has_left)  ? input[north_base - 1] : 0ull;
    const uint64_t Nr = (has_top && has_right) ? input[north_base + 1] : 0ull;

    // Load south row words (row + 1).
    const std::size_t south_base = tid + static_cast<std::size_t>(words_per_row);
    const uint64_t Sc = has_bottom ? input[south_base] : 0ull;
    const uint64_t Sl = (has_bottom && has_left)  ? input[south_base - 1] : 0ull;
    const uint64_t Sr = (has_bottom && has_right) ? input[south_base + 1] : 0ull;

    // Construct neighbor masks by shifting and combining adjacent words.
    // Central row horizontal neighbors.
    const uint64_t west = (Cc >> 1) | (Cl << 63);  // W neighbors
    const uint64_t east = (Cc << 1) | (Cr >> 63);  // E neighbors

    // North row: vertical and diagonal neighbors.
    const uint64_t north      = Nc;                               // N
    const uint64_t north_west = (Nc >> 1) | (Nl << 63);           // NW
    const uint64_t north_east = (Nc << 1) | (Nr >> 63);           // NE

    // South row: vertical and diagonal neighbors.
    const uint64_t south      = Sc;                               // S
    const uint64_t south_west = (Sc >> 1) | (Sl << 63);           // SW
    const uint64_t south_east = (Sc << 1) | (Sr >> 63);           // SE

    // Bit-parallel neighbor counting.
    uint64_t ones  = 0ull;
    uint64_t twos  = 0ull;
    uint64_t fours = 0ull;

    // Accumulate all 8 neighbor directions.
    accumulate_neighbor(west,       ones, twos, fours);
    accumulate_neighbor(east,       ones, twos, fours);
    accumulate_neighbor(north,      ones, twos, fours);
    accumulate_neighbor(south,      ones, twos, fours);
    accumulate_neighbor(north_west, ones, twos, fours);
    accumulate_neighbor(north_east, ones, twos, fours);
    accumulate_neighbor(south_west, ones, twos, fours);
    accumulate_neighbor(south_east, ones, twos, fours);

    // Decode counts == 2 and counts == 3 from (ones, twos, fours).
    const uint64_t not_ones  = ~ones;
    const uint64_t not_fours = ~fours;

    const uint64_t eq2_mask = (not_ones & twos & not_fours);  // 010
    const uint64_t eq3_mask = (ones     & twos & not_fours);  // 011

    const uint64_t alive = Cc;

    // Apply Game of Life rules:
    //   - A cell is alive in the next generation if:
    //       neighbors == 3, OR
    //       (currently alive AND neighbors == 2).
    const uint64_t next = eq3_mask | (alive & eq2_mask);

    output[tid] = next;
}

// Host function: runs one Game of Life step on the given bit-packed grid.
//
// Parameters:
//   input           - device pointer to input grid (bit-packed).
//   output          - device pointer to output grid (bit-packed).
//   grid_dimensions - width/height of the square grid (power of 2, > 512).
//
// Both input and output must be allocated with cudaMalloc and large enough to
// hold grid_dimensions * grid_dimensions bits (i.e., grid_dimensions *
// grid_dimensions / 64 uint64_t elements).
//
// This function does not perform any explicit device synchronization; the caller
// is responsible for synchronizing if needed.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64, safe since power of 2
    const std::size_t total_words =
        static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Choose a reasonable block size; 256 threads is a good default on modern GPUs.
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    game_of_life_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output),
        grid_dimensions,
        words_per_row
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
