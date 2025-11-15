// Conway's Game of Life - Bit-packed CUDA implementation
//
// Each cell is represented as a single bit. The grid is square with
// dimension `grid_dimensions`, always a power of two and >= 512.
//
// Layout:
//   - The grid is stored row-major.
//   - Each row is composed of `words_per_row = grid_dimensions / 64`
//     64-bit words.
//   - Within a word, bit 0 is the least significant bit (LSB) and
//     bit 63 is the most significant bit (MSB).
//
// A CUDA thread processes exactly one 64-bit word (64 cells). Instead of
// processing each cell individually, we process all 64 cells in parallel
// using bitwise operations (SWAR / bit-slicing). This avoids atomics and
// drastically reduces per-cell computational overhead.
//
// Neighbor counting strategy:
//
// For a given word W at position (row, col):
//   - Let U, D be the words directly above and below W.
//   - Let L, R be the words directly left and right.
//   - Let UL, UR, DL, DR be the diagonal neighbors.
//
// For each of the three rows (U, W, D), we construct three shifted versions:
//   - left shift (neighbors coming from left in that row)
//   - center (neighbors directly above/below for U/D, not used for W)
//   - right shift (neighbors coming from right in that row)
//
// Example for row W (current row):
//   - C_left  = W shifted so that for each bit k, C_left[k] = W[k-1],
//              including cross-word contributions from L for bit 0.
//   - C_right = W shifted so that for each bit k, C_right[k] = W[k+1],
//              including cross-word contributions from R for bit 63.
//   - The cell itself (W) is not included as a neighbor.
//
// For rows U and D we additionally include the center word itself
// (neighbors directly above and below).
//
// Total of 8 neighbor bitmasks per word:
//   N0 = U_left,   N1 = U_center,  N2 = U_right,
//   N3 = C_left,                  N4 = C_right,
//   N5 = D_left,   N6 = D_center,  N7 = D_right.
//
// We then compute, for all 64 cells in the word simultaneously, the
// neighbor counts modulo 8 using a bit-sliced incrementer for each
// neighbor mask. We keep three 64-bit planes:
//
//   count1: bit 0 of the count (1's place)
//   count2: bit 1 of the count (2's place)
//   count4: bit 2 of the count (4's place)
//
// For each neighbor mask `x` (N0..N7), we "add 1" to the per-bit
// counters wherever `x` has a 1. This is a full binary increment
// done in parallel for all 64 columns:
//
//   carry1 = count1 & x;
//   count1 ^= x;
//   carry2 = count2 & carry1;
//   count2 ^= carry1;
//   count4 ^= carry2;
//
// This addition is modulo 8, but the real neighbor count is at most 8.
// The only case where modulo 8 differs from the true count in [0,8]
// is when the count is exactly 8 (wraps to 0). Since we only care about
// counts equal to 2 or 3, and 8 is "more than 3" (cell dies), this is
// safe: counts of 8 will behave exactly like 0 for our equality tests,
// and neither 0 nor 8 should cause a birth or survival based on neighbor
// count.
//
// After accumulating all neighbors, for each bit we have count in
// binary: count = count1 + 2*count2 + 4*count4 (mod 8).
//
// We derive two masks:
//   eq2: bits where count == 2 (010b)
//   eq3: bits where count == 3 (011b)
//
//   eq2 = ~count4 &  count2 & ~count1
//   eq3 = ~count4 &  count2 &  count1
//
// The Game of Life rule becomes:
//   next_bit = (eq3) | (eq2 & current_bit)
//
// Boundary handling:
//
// - Cells outside the grid are considered dead. For words on the edges,
//   missing neighbor words are replaced with 0.
// - Bit 0 and bit 63 of each word are handled by using the left/right
//   neighbor words when constructing the shifted neighbor masks.
//
// This implementation does not use shared or texture memory; all neighbor
// data is loaded directly from global memory. The memory layout ensures
// that these accesses are coalesced when possible.

#include <cstdint>
#include <cuda_runtime.h>

// Convenience alias
using u64 = std::uint64_t;

// Device helper: add a neighbor bitmask to the bit-sliced counters.
//
// For each bit position i (0..63):
//   If x[i] == 1, then we increment the 3-bit counter
//   (count4[i], count2[i], count1[i]) by 1 modulo 8.
static __device__ __forceinline__
void add_neighbor(u64 x, u64 &count1, u64 &count2, u64 &count4)
{
    // Increment least significant bit (count1) where x has ones
    u64 carry1 = count1 & x;
    count1 ^= x;

    // Propagate carry into count2
    u64 carry2 = count2 & carry1;
    count2 ^= carry1;

    // Propagate carry into count4 (no need to keep beyond bit 2)
    count4 ^= carry2;
}

// CUDA kernel: perform one step of Conway's Game of Life on a bit-packed grid.
//
// Each thread processes one 64-bit word (64 cells).
__global__
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              int grid_dimensions,
                              int words_per_row)
{
    const int global_word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = grid_dimensions * words_per_row;

    if (global_word_idx >= total_words)
        return;

    // Compute (row, col_word) from linear word index.
    const int row       = global_word_idx / words_per_row;
    const int col_word  = global_word_idx - row * words_per_row;

    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dimensions);
    const bool has_left  = (col_word > 0);
    const bool has_right = (col_word + 1 < words_per_row);

    // Index helpers
    const int row_offset      = row * words_per_row;
    const int up_row_offset   = (row - 1) * words_per_row;
    const int down_row_offset = (row + 1) * words_per_row;

    // Load center word
    const u64 self = input[global_word_idx];

    // Load neighbor words where they exist; otherwise 0.
    const u64 left       = has_left  ? input[global_word_idx - 1]                : 0ull;
    const u64 right      = has_right ? input[global_word_idx + 1]                : 0ull;
    const u64 up         = has_up    ? input[up_row_offset   + col_word]         : 0ull;
    const u64 down       = has_down  ? input[down_row_offset + col_word]         : 0ull;
    const u64 up_left    = (has_up   && has_left)  ? input[up_row_offset   + col_word - 1] : 0ull;
    const u64 up_right   = (has_up   && has_right) ? input[up_row_offset   + col_word + 1] : 0ull;
    const u64 down_left  = (has_down && has_left)  ? input[down_row_offset + col_word - 1] : 0ull;
    const u64 down_right = (has_down && has_right) ? input[down_row_offset + col_word + 1] : 0ull;

    // Construct neighbor masks for the three rows:
    // Above row (U): left, center, right neighbors
    u64 U_center = up;
    u64 U_left   = up << 1;
    if (has_left) {
        // For bit 0, left neighbor comes from bit 63 of up_left
        U_left |= (up_left >> 63);
    }
    u64 U_right  = up >> 1;
    if (has_right) {
        // For bit 63, right neighbor comes from bit 0 of up_right
        U_right |= (up_right << 63);
    }

    // Current row (self): left and right neighbors only (self is not counted)
    u64 C_left   = self << 1;
    if (has_left) {
        C_left |= (left >> 63);
    }
    u64 C_right  = self >> 1;
    if (has_right) {
        C_right |= (right << 63);
    }

    // Below row (D): left, center, right neighbors
    u64 D_center = down;
    u64 D_left   = down << 1;
    if (has_left) {
        D_left |= (down_left >> 63);
    }
    u64 D_right  = down >> 1;
    if (has_right) {
        D_right |= (down_right << 63);
    }

    // Accumulate neighbor counts modulo 8 in bit-sliced form.
    u64 count1 = 0ull; // 1's bit
    u64 count2 = 0ull; // 2's bit
    u64 count4 = 0ull; // 4's bit

    // Eight neighbor direction masks:
    //  U_left, U_center, U_right,
    //  C_left,           C_right,
    //  D_left, D_center, D_right
    add_neighbor(U_left,   count1, count2, count4);
    add_neighbor(U_center, count1, count2, count4);
    add_neighbor(U_right,  count1, count2, count4);
    add_neighbor(C_left,   count1, count2, count4);
    add_neighbor(C_right,  count1, count2, count4);
    add_neighbor(D_left,   count1, count2, count4);
    add_neighbor(D_center, count1, count2, count4);
    add_neighbor(D_right,  count1, count2, count4);

    // Determine where neighbor count == 2 or 3.
    //
    // count = (count4 << 2) | (count2 << 1) | count1  (mod 8)
    // 2 -> 010b, 3 -> 011b.
    const u64 not_count4 = ~count4;
    const u64 eq2 = not_count4 &  count2 & ~count1; // count == 2
    const u64 eq3 = not_count4 &  count2 &  count1; // count == 3

    // Apply Game of Life rules:
    //   - Any live cell with 2 or 3 neighbors survives.
    //   - Any dead cell with exactly 3 neighbors becomes alive.
    //   - Otherwise the cell is dead.
    const u64 next = (eq3) | (eq2 & self);

    output[global_word_idx] = next;
}

// Host function: run one simulation step.
//
// `input`  - device pointer to current grid (bit-packed).
// `output` - device pointer to next grid (bit-packed).
// `grid_dimensions` - width and height of the square grid (power of 2).
//
// Both `input` and `output` must point to device memory of size:
//   (grid_dimensions * grid_dimensions / 64) * sizeof(std::uint64_t)
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    // Launch with a 1D grid of threads, one thread per 64-bit word.
    constexpr int THREADS_PER_BLOCK = 256;
    const int blocks = (total_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    game_of_life_step_kernel<<<blocks, THREADS_PER_BLOCK>>>(input,
                                                            output,
                                                            grid_dimensions,
                                                            words_per_row);
    // Synchronization (if any) is handled by the caller.
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
