#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  High-performance CUDA implementation of a single Conway's Game of Life step on a bit-packed grid.

  - The grid is square with side length grid_dimensions (power of 2, >512).
  - Each uint64_t encodes 64 consecutive horizontal cells in a row (bit 0 is the leftmost cell in the word).
  - Each CUDA thread processes exactly one 64-bit word (64 cells).
  - Outside-grid cells are considered dead (0), so boundary words handle missing neighbors as zeros.
  - No shared or texture memory; global loads are coalesced due to row-major packing and per-word threading.
  - Neighbor counting is done with bit-sliced counters (ones, twos, fours) via ripple-carry add of the 8 neighbor bitboards.
    This avoids cross-bit carries and computes eq2 and eq3 directly without scalar loops or atomics.

  The evolution rule:
    next = (neighbors == 3) | (current & (neighbors == 2))
*/

static __device__ __forceinline__ std::uint64_t shift_left_with_carry(std::uint64_t center, std::uint64_t left_word) {
    // For each bit i in 'center', its left neighbor is bit (i-1) in the same row.
    // Bit 0's left neighbor comes from bit 63 of the word to the left (left_word).
    return (center << 1) | (left_word >> 63);
}

static __device__ __forceinline__ std::uint64_t shift_right_with_carry(std::uint64_t center, std::uint64_t right_word) {
    // For each bit i in 'center', its right neighbor is bit (i+1) in the same row.
    // Bit 63's right neighbor comes from bit 0 of the word to the right (right_word).
    return (center >> 1) | (right_word << 63);
}

static __device__ __forceinline__ void add_bitboard_to_counters(std::uint64_t b,
                                                                std::uint64_t& ones,
                                                                std::uint64_t& twos,
                                                                std::uint64_t& fours) {
    // Ripple-carry addition of a 1-bit-per-lane bitboard 'b' into bit-sliced counters ones/twos/fours.
    // After processing all 8 neighbor bitboards:
    //   ones  holds the 1's bit of the per-bit neighbor count,
    //   twos  holds the 2's bit,
    //   fours holds the 4's bit.
    // This is sufficient to test for counts == 2 (010) and == 3 (011) without computing an 8's bit plane.
    std::uint64_t t = ones & b;    // carry from ones
    ones ^= b;                      // ones = ones + b (mod 2)
    std::uint64_t u = twos & t;     // carry from twos
    twos ^= t;                      // twos = twos + carry (mod 2)
    fours ^= u;                     // fours = fours + carry (mod 2)
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int grid_dim,
                                    int words_per_row) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_words = static_cast<size_t>(grid_dim) * static_cast<size_t>(words_per_row);
    if (tid >= total_words) return;

    // Map thread to (row, col_word)
    const int row = static_cast<int>(tid / words_per_row);
    const int col = static_cast<int>(tid - static_cast<size_t>(row) * static_cast<size_t>(words_per_row));

    // Boundary flags
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim);
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    const size_t base      = static_cast<size_t>(row) * static_cast<size_t>(words_per_row);
    const size_t base_up   = base - static_cast<size_t>(words_per_row);
    const size_t base_down = base + static_cast<size_t>(words_per_row);

    // Load the 9 relevant words (3x3 neighborhood in word-space) with zero-padding at boundaries.
    // Current row
    const std::uint64_t cL = has_left  ? in[base + static_cast<size_t>(col - 1)] : 0ull;
    const std::uint64_t cC =             in[base + static_cast<size_t>(col)];
    const std::uint64_t cR = has_right ? in[base + static_cast<size_t>(col + 1)] : 0ull;

    // Up row
    const std::uint64_t uL = (has_up && has_left)  ? in[base_up + static_cast<size_t>(col - 1)] : 0ull;
    const std::uint64_t uC =  has_up               ? in[base_up + static_cast<size_t>(col)]     : 0ull;
    const std::uint64_t uR = (has_up && has_right) ? in[base_up + static_cast<size_t>(col + 1)] : 0ull;

    // Down row
    const std::uint64_t dL = (has_down && has_left)  ? in[base_down + static_cast<size_t>(col - 1)] : 0ull;
    const std::uint64_t dC =  has_down               ? in[base_down + static_cast<size_t>(col)]     : 0ull;
    const std::uint64_t dR = (has_down && has_right) ? in[base_down + static_cast<size_t>(col + 1)] : 0ull;

    // Compute the 8 neighbor bitboards for this word using cross-word shifts where needed.
    const std::uint64_t UL = shift_left_with_carry(uC, uL);
    const std::uint64_t U  = uC;
    const std::uint64_t UR = shift_right_with_carry(uC, uR);

    const std::uint64_t L  = shift_left_with_carry(cC, cL);
    const std::uint64_t R  = shift_right_with_carry(cC, cR);

    const std::uint64_t DL = shift_left_with_carry(dC, dL);
    const std::uint64_t D  = dC;
    const std::uint64_t DR = shift_right_with_carry(dC, dR);

    // Bit-sliced neighbor counters via ripple-carry addition of the 8 neighbor bitboards.
    std::uint64_t ones = 0ull, twos = 0ull, fours = 0ull;
    add_bitboard_to_counters(UL, ones, twos, fours);
    add_bitboard_to_counters(U,  ones, twos, fours);
    add_bitboard_to_counters(UR, ones, twos, fours);
    add_bitboard_to_counters(L,  ones, twos, fours);
    add_bitboard_to_counters(R,  ones, twos, fours);
    add_bitboard_to_counters(DL, ones, twos, fours);
    add_bitboard_to_counters(D,  ones, twos, fours);
    add_bitboard_to_counters(DR, ones, twos, fours);

    // Determine cells with exactly 2 or exactly 3 neighbors:
    // Count == 2 -> binary 010 -> ~ones &  twos & ~fours
    // Count == 3 -> binary 011 ->  ones &  twos & ~fours
    const std::uint64_t not_ones  = ~ones;
    const std::uint64_t not_fours = ~fours;
    const std::uint64_t eq2 = (not_ones & twos) & not_fours;
    const std::uint64_t eq3 = (ones & twos) & not_fours;

    // Apply Game of Life rules:
    // - Alive survives with 2 or 3 neighbors: (current & eq2) | (current & eq3 == included in eq3)
    // - Dead becomes alive with exactly 3 neighbors: eq3
    // So next = eq3 | (current & eq2)
    const std::uint64_t next = eq3 | (cC & eq2);

    out[base + static_cast<size_t>(col)] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row holds words_per_row 64-bit words.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const size_t total_words = static_cast<size_t>(grid_dimensions) * static_cast<size_t>(words_per_row);

    // Launch configuration: one thread per word.
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    game_of_life_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions, words_per_row);
    // No sync here; the caller handles synchronization as specified.
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
