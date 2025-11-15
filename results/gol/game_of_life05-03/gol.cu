// Conway's Game of Life using CUDA with 64-bit bit-packed rows.
// Each CUDA thread processes exactly one 64-bit word (64 cells).
//
// The grid is square with dimensions `grid_dimensions x grid_dimensions`,
// where `grid_dimensions` is a power of 2 > 512 and divisible by 64.
// Each row therefore contains `grid_dimensions / 64` 64-bit words.
//
// Encoding:
//   - Each std::uint64_t encodes 64 consecutive cells in a row.
//   - Bit value 1 -> cell is alive, 0 -> cell is dead.
//   - Bit index 0 is the least significant bit (LSB), representing the
//     leftmost cell of the 64-cell block; bit index 63 is the rightmost.
//
// Neighbor handling:
//   For every word at row r and word index c, we load the nine words
//   (if they exist) around it: left, right, up, down, and the 4 diagonals.
//   Then we build eight 64-bit masks corresponding to the eight neighbor
//   directions (N, S, E, W, NE, NW, SE, SW) for all 64 cells in parallel.
//
//   Horizontal neighbors and diagonals are formed by bitwise shifts across
//   64-bit word boundaries (using the adjacent word for carry-in), which
//   correctly handles the 0th and 63rd bits of each word.
//
// Neighbor counting:
//   For each of the 64 cell positions in the word we need the number of
//   alive neighbors (0..8). We compute this for all 64 bits in parallel:
//     - Represent the per-cell neighbor count modulo 8 as three 64-bit
//       bitmaps: ones, twos, fours (LSB..MSB).
//     - Increment these counters by each of the eight neighbor bitboards
//       using a ripple-carry scheme implemented with AND/XOR operations.
//     - This yields counts mod 8; note that actual counts are in 0..8,
//       and 8 ≡ 0 mod 8. In Conway's Game of Life, both 0 and 8 neighbors
//       cause the cell to be dead, so modulo-8 arithmetic is sufficient.
//
// Applying the Game of Life rules:
//   Let count be the per-cell neighbor count (mod 8):
//       - A dead cell becomes alive iff count == 3.
//       - A live cell survives iff count == 2 or count == 3.
//   Using bitboards:
//       - count == 2 -> ones=0, twos=1, fours=0
//       - count == 3 -> ones=1, twos=1, fours=0
//   So,
//       mask2 = twos & ~ones & ~fours  (exactly 2 neighbors)
//       mask3 = twos &  ones & ~fours  (exactly 3 neighbors)
//   Then,
//       next = (mask3) | (mask2 & current)
//
// Boundary conditions:
//   All cells outside the grid are treated as dead. We implement this by
//   setting "missing" neighbor words (beyond edges) to zero before
//   computing the shifts.
//
// Performance notes:
//   - One thread per 64-bit word (64 cells).
//   - All neighbor operations are done with word-wise bit operations.
//   - No shared or texture memory is used; accesses are coalesced and
//     cached via hardware. This is sufficient on modern GPUs (A100/H100).

#include <cstdint>
#include <cuda_runtime.h>

// Convenience alias for brevity.
using u64 = std::uint64_t;

// Bitwise shift helpers for horizontal neighbors within a row.
// These operate on a "segment" of the row (one 64-bit word) but
// include interaction with the neighboring word to correctly
// handle the 0th and 63rd bits.
//
// With bit 0 as LSB and bit indices increasing to the right:
//
//   - shift_east(row)[x] = row[x + 1] (cell's east neighbor)
//   - shift_west(row)[x] = row[x - 1] (cell's west neighbor)
//
// Implemented per 64-bit word with carry-in from right/left neighbor.
__device__ __forceinline__
u64 shift_east(u64 center, u64 right)
{
    // Equivalent to (center | (right << 64)) >> 1 on a 128-bit value.
    return (center >> 1) | (right << 63);
}

__device__ __forceinline__
u64 shift_west(u64 center, u64 left)
{
    // Equivalent to ((left << 64) | center) << 1 on a 128-bit value.
    return (center << 1) | (left >> 63);
}

// Add a 1-bit-per-cell neighbor bitboard `b` into the per-cell
// modulo-8 neighbor count represented by (ones, twos, fours).
//
// For each bit position i (0..63), (ones[i], twos[i], fours[i]) is the
// 3-bit binary representation of the current neighbor count at that
// position, modulo 8. Adding a single new neighbor bit b[i] is
// equivalent to incrementing a 3-bit counter:
//
//   ones, twos, fours := (ones, twos, fours) + b
//
// Done via ripple-carry:
//   - carry into `twos` is where both ones and b are 1.
//   - carry into `fours` is where both twos and that carry are 1.
__device__ __forceinline__
void add_neighbor(u64 b, u64 &ones, u64 &twos, u64 &fours)
{
    u64 carry = ones & b;
    ones ^= b;

    u64 carry2 = twos & carry;
    twos ^= carry;

    fours ^= carry2;
}

// CUDA kernel: compute one Game of Life step on a bit-packed grid.
// Each thread processes one 64-bit word (64 cells).
__global__
void game_of_life_kernel(const u64* __restrict__ in,
                         u64* __restrict__ out,
                         int wordsPerRow,
                         int numRows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWords = wordsPerRow * numRows;
    if (tid >= totalWords) return;

    // Compute 2D coordinates of this word within the grid.
    int row = tid / wordsPerRow;      // y-coordinate
    int col = tid - row * wordsPerRow; // x-coordinate in 64-cell words

    const u64 center = in[tid];

    // Determine existence of neighboring words in this row/column.
    const bool hasLeft  = (col > 0);
    const bool hasRight = (col + 1 < wordsPerRow);
    const bool hasUp    = (row > 0);
    const bool hasDown  = (row + 1 < numRows);

    // Load neighboring words, defaulting to 0 for out-of-bounds
    // to implement the "outside cells are dead" rule.
    u64 left      = 0;
    u64 right     = 0;
    u64 up        = 0;
    u64 up_left   = 0;
    u64 up_right  = 0;
    u64 down      = 0;
    u64 down_left = 0;
    u64 down_right= 0;

    if (hasLeft) {
        left = in[tid - 1];
    }
    if (hasRight) {
        right = in[tid + 1];
    }

    if (hasUp) {
        int upBase = (row - 1) * wordsPerRow;
        up = in[upBase + col];
        if (hasLeft) {
            up_left = in[upBase + col - 1];
        }
        if (hasRight) {
            up_right = in[upBase + col + 1];
        }
    }

    if (hasDown) {
        int downBase = (row + 1) * wordsPerRow;
        down = in[downBase + col];
        if (hasLeft) {
            down_left = in[downBase + col - 1];
        }
        if (hasRight) {
            down_right = in[downBase + col + 1];
        }
    }

    // Build neighbor bitboards for all 64 cells simultaneously.

    // Vertical neighbors.
    u64 north = up;
    u64 south = down;

    // Horizontal neighbors (same row), using cross-word shifts.
    u64 east = shift_east(center, right);
    u64 west = shift_west(center, left);

    // Diagonal neighbors: shift the rows above/below.
    u64 ne = shift_east(up,   up_right);
    u64 nw = shift_west(up,   up_left);
    u64 se = shift_east(down, down_right);
    u64 sw = shift_west(down, down_left);

    // Accumulate neighbor counts (modulo 8) into ones, twos, fours.
    u64 ones  = 0;
    u64 twos  = 0;
    u64 fours = 0;

    add_neighbor(north, ones, twos, fours);
    add_neighbor(south, ones, twos, fours);
    add_neighbor(east,  ones, twos, fours);
    add_neighbor(west,  ones, twos, fours);
    add_neighbor(ne,    ones, twos, fours);
    add_neighbor(nw,    ones, twos, fours);
    add_neighbor(se,    ones, twos, fours);
    add_neighbor(sw,    ones, twos, fours);

    // Decode neighbor counts:
    //   count == 2  => ones=0, twos=1, fours=0
    //   count == 3  => ones=1, twos=1, fours=0
    u64 notOnes  = ~ones;
    u64 notFours = ~fours;

    u64 count_eq_2 = twos & notOnes & notFours;
    u64 count_eq_3 = twos &  ones   & notFours;

    // Apply Game of Life rules:
    //   - Alive in next step if:
    //       * currently alive and exactly 2 neighbors, or
    //       * exactly 3 neighbors (birth or survival).
    u64 stayAlive = count_eq_2 & center;
    u64 born      = count_eq_3;

    u64 next = stayAlive | born;

    out[tid] = next;
}

// Host function: run a single Game of Life step on the GPU.
// `input` and `output` are device pointers allocated with cudaMalloc.
// `grid_dimensions` is the width/height of the square grid and is
// a power of two greater than 512. Each row has grid_dimensions/64
// 64-bit words. This function launches the CUDA kernel but does not
// perform any host-device synchronization; the caller is responsible
// for synchronization and error checking if desired.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int wordsPerRow = grid_dimensions >> 6;  // grid_dimensions / 64
    const int numRows     = grid_dimensions;
    const int totalWords  = wordsPerRow * numRows;

    // Configure a 1D grid of threads.
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = (totalWords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    game_of_life_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const u64*>(input),
        reinterpret_cast<u64*>(output),
        wordsPerRow,
        numRows
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
