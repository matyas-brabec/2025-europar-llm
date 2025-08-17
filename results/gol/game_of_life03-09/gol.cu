#include <cstdint>
#include <cuda_runtime.h>

// This implementation simulates one step of Conway's Game of Life on a bit-packed grid.
// Each uint64_t encodes a run of 64 horizontally consecutive cells in a row (LSB is the leftmost bit).
// Each thread processes one 64-bit word (exactly 64 cells).
//
// Core idea:
// - For each word, load the necessary 9 words (left/center/right across the upper/middle/lower rows).
// - Build three horizontally shifted masks per row (left/center/right) using funnel-like shifts that
//   pull in the neighbor bit from adjacent words, taking care of boundaries (neighbor word is 0 outside the grid).
// - Sum the three masks per row using carry-less (per-bit) addition (add3) to get a 2-bit result per bit (0..3).
// - Sum the three row results (upper/middle/lower), again using carry-less addition, to get a 3-bit neighbor count (0..8),
//   excluding the center cell of the current row by omitting it from the middle row sum.
// - Apply the Life rules: next = (neighbors == 3) | (alive & (neighbors == 2)).
//
// We use bit-sliced addition for performance; no shared memory or textures are used as requested.

static_assert(sizeof(std::uint64_t) == 8, "This code assumes 64-bit unsigned integers");

// Carry-less addition of three one-bit-per-lane bitmaps.
// For each bit position i: lo[i] = (a[i] + b[i] + c[i]) & 1; hi[i] = ((a[i] + b[i] + c[i]) >> 1) & 1
// Implemented via two half-adders: classic bitwise logic without inter-bit carries.
__device__ __forceinline__ void add3_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                         std::uint64_t& lo, std::uint64_t& hi)
{
    // First half-adder for a + b
    std::uint64_t s = a ^ b;     // sum without carry
    std::uint64_t carry1 = a & b;// carry bits from a + b

    // Second half-adder: (a + b) + c
    lo = s ^ c;                  // final sum bit (mod 2)
    std::uint64_t carry2 = s & c;

    // The carry (value 2) is set if either:
    // - both a and b are 1 (carry1), or
    // - one of a/b is 1 and c is 1 (carry2)
    hi = carry1 | carry2;
}

// Shift-left by 1 with incoming bit from the MSB of 'left' word.
// For bit i, the source is bit (i-1). For i=0, we use left's bit63, which is zero at left boundary.
__device__ __forceinline__ std::uint64_t shl1_with_left(std::uint64_t center, std::uint64_t left)
{
    return (center << 1) | (left >> 63);
}

// Shift-right by 1 with incoming bit from the LSB of 'right' word.
// For bit i, the source is bit (i+1). For i=63, we use right's bit0, which is zero at right boundary.
__device__ __forceinline__ std::uint64_t shr1_with_right(std::uint64_t center, std::uint64_t right)
{
    return (center >> 1) | (right << 63);
}

// CUDA kernel: compute one evolution step of Conway's Game of Life.
// - in/out: bit-packed grids
// - n: grid dimension (number of cells per side, power of two, > 512)
// - words_per_row: n / 64
__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ in,
                                         std::uint64_t* __restrict__ out,
                                         int n, int words_per_row)
{
    const std::size_t total_words = static_cast<std::size_t>(n) * static_cast<std::size_t>(words_per_row);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_words; idx += stride)
    {
        // Compute row and column (word column within the row)
        const int row = static_cast<int>(idx / words_per_row);
        const int col = static_cast<int>(idx - static_cast<std::size_t>(row) * words_per_row);

        // Flags to avoid branches where possible
        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < n);
        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < words_per_row);

        // Compute base indices for neighbor rows
        const std::size_t idx_up   = idx - static_cast<std::size_t>(words_per_row);
        const std::size_t idx_down = idx + static_cast<std::size_t>(words_per_row);

        // Load center-row words
        const std::uint64_t midC = in[idx];
        const std::uint64_t midL = has_left  ? in[idx - 1] : 0ull;
        const std::uint64_t midR = has_right ? in[idx + 1] : 0ull;

        // Load upper-row words (or zeros at top boundary)
        const std::uint64_t upC = has_up ? in[idx_up] : 0ull;
        const std::uint64_t upL = (has_up && has_left)  ? in[idx_up - 1] : 0ull;
        const std::uint64_t upR = (has_up && has_right) ? in[idx_up + 1] : 0ull;

        // Load lower-row words (or zeros at bottom boundary)
        const std::uint64_t dnC = has_down ? in[idx_down] : 0ull;
        const std::uint64_t dnL = (has_down && has_left)  ? in[idx_down - 1] : 0ull;
        const std::uint64_t dnR = (has_down && has_right) ? in[idx_down + 1] : 0ull;

        // Build horizontally shifted masks for each row:
        // For neighbors above and below, include left/center/right; for the middle row exclude the center (self).
        // The shl/shr helpers automatically pull in boundary bits from neighbor words.
        const std::uint64_t lu = shl1_with_left(upC, upL);
        const std::uint64_t cu = upC;
        const std::uint64_t ru = shr1_with_right(upC, upR);

        const std::uint64_t lm = shl1_with_left(midC, midL);
        const std::uint64_t rm = shr1_with_right(midC, midR);
        // Note: the middle "center" (self) is intentionally excluded from neighbor accumulation.

        const std::uint64_t ld = shl1_with_left(dnC, dnL);
        const std::uint64_t cd = dnC;
        const std::uint64_t rd = shr1_with_right(dnC, dnR);

        // Row-wise sums (per-bit 0..3) using carry-less add3.
        std::uint64_t u_lo, u_hi;
        add3_u64(lu, cu, ru, u_lo, u_hi);          // upper row sum: lo + 2*hi

        std::uint64_t m_lo, m_hi;
        // middle row sum is lm + rm (no center), so add3 with zero is fine
        add3_u64(lm, rm, 0ull, m_lo, m_hi);        // middle row sum: lo + 2*hi

        std::uint64_t d_lo, d_hi;
        add3_u64(ld, cd, rd, d_lo, d_hi);          // lower row sum: lo + 2*hi

        // Sum the three row sums:
        // total_neighbors = (u_lo + m_lo + d_lo) + 2*(u_hi + m_hi + d_hi)
        // Compute per-bit sums of lo parts and hi parts separately using add3.
        std::uint64_t lo_s, lo_c; // lo_s is bit0 of total (LSB), lo_c is carry from lo sum (weight 2)
        add3_u64(u_lo, m_lo, d_lo, lo_s, lo_c);

        std::uint64_t hi_s, hi_c; // hi_s is sum mod2 of hi parts (weight 2), hi_c is carry (weight 4)
        add3_u64(u_hi, m_hi, d_hi, hi_s, hi_c);

        // Reconstruct neighbor count bits (bit-sliced, per lane):
        // Let:
        // - n0 = bit0 (LSB) of neighbor count
        // - n1 = bit1 of neighbor count
        // - n2 = bit2 of neighbor count
        //
        // From total_neighbors = lo_s + 2*(lo_c + hi_s) + 4*hi_c:
        // - n0 = lo_s
        // - n1 = (lo_c ^ hi_s)         (parity of the 2's place contributions)
        // - n2 = hi_c ^ (lo_c & hi_s)  (carry from two 2's plus the 4's place contribution)
        const std::uint64_t n0 = lo_s;
        const std::uint64_t n1 = lo_c ^ hi_s;
        const std::uint64_t n2 = hi_c ^ (lo_c & hi_s);

        // Apply Conway's rules:
        // - A live cell survives if neighbor count == 2 or 3.
        // - A dead cell becomes live if neighbor count == 3.
        //
        // We compute masks for (neighbors == 2) and (neighbors == 3) using the three bit-planes (n2 n1 n0):
        // eq2: 010 => (~n2) &  n1  & (~n0)
        // eq3: 011 => (~n2) &  n1  &  n0
        const std::uint64_t eq2 = (~n2) & n1 & (~n0);
        const std::uint64_t eq3 = (~n2) & n1 & n0;

        const std::uint64_t cur_alive = midC;
        const std::uint64_t next = eq3 | (cur_alive & eq2);

        out[idx] = next;
    }
}

// Public API: run one step of Conway's Game of Life.
// - input:  pointer to device memory holding the current state (bit-packed, row-major).
// - output: pointer to device memory where the next state will be written (same layout).
// - grid_dimensions: number of cells per side (power of two; > 512).
//
// Notes:
// - Both input and output must be allocated with cudaMalloc by the caller.
// - This function launches the CUDA kernel but does not synchronize; the caller handles synchronization if needed.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    const int n = grid_dimensions;
    const int words_per_row = n >> 6; // n / 64

    const std::size_t total_words = static_cast<std::size_t>(n) * static_cast<std::size_t>(words_per_row);

    // Choose a moderate block size for good occupancy on modern data center GPUs.
    // Each thread handles one 64-bit word. Use a grid-stride loop inside the kernel.
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    game_of_life_step_kernel<<<blocks, threads_per_block>>>(input, output, n, words_per_row);
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
