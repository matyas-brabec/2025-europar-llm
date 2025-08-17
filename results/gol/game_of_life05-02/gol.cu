#include <cuda_runtime.h>
#include <cstdint>

// Each thread processes one 64-bit word (64 cells). The algorithm uses broadword (bit-sliced)
// arithmetic to compute the 8-neighbor counts for all 64 cells in parallel.
//
// Overview of the method:
// - For the three relevant rows (above, current, below) we compute bitmasks representing the
//   left, center, and right neighbor contributions for the current word, with cross-word carry:
//     L = (center << 1) | (left_word >> 63)
//     C = center
//     R = (center >> 1) | (right_word << 63)
//   This handles the 0th and 63rd bits by importing edge bits from the neighboring words.
//   Outside-grid words are treated as zero.
// - For the top and bottom rows, we sum the three contributions (L + C + R) into two bit-planes:
//     ones = L ^ C ^ R
//     twos = (L & C) | (L & R) | (C & R)
//   representing a per-bit value: ones + 2*twos (in 0..3).
// - For the middle row (current row), neighbors exclude the center cell, so we sum only L + R:
//     ones = L ^ R
//     twos = L & R
//   giving values in 0..2.
// - We then add the three per-bit numbers (top, mid, bottom) using bit-sliced addition that
//   produces planes for 1s, 2s, 4s, and 8s (n0, n1, n2, n3).
// - Finally, we compute next-state bits with Game of Life rules using the bit-sliced count:
//     births     (count == 3) => n0=1, n1=1, n2=0, n3=0
//     survivals  (count == 2) AND (current alive) => n0=0, n1=1, n2=0, n3=0 AND current bit = 1
//   next = births | (survivals & current)
//
// Notes:
// - No shared or texture memory is used; global loads are coalesced when threads map to consecutive words.
// - Boundary handling: missing neighbors (outside the grid) are treated as zeros.
// - Each thread reads up to 9 words: (left, center, right) for above/current/below rows; boundary
//   threads reduce this. This is fast and simple on modern GPUs (A100/H100).

// Add two bit-sliced numbers (a and b) with up to three bit-planes each.
// a = a0 + 2*a1 + 4*a2
// b = b0 + 2*b1 + 4*b2
// Result r = a + b = r0 + 2*r1 + 4*r2 + 8*r3
static __device__ __forceinline__
void add_planes3(uint64_t a0, uint64_t a1, uint64_t a2,
                 uint64_t b0, uint64_t b1, uint64_t b2,
                 uint64_t &r0, uint64_t &r1, uint64_t &r2, uint64_t &r3)
{
    // Plane 0 (ones)
    uint64_t s0 = a0 ^ b0;
    uint64_t c0 = a0 & b0; // carry into plane1

    // Plane 1 (twos)
    uint64_t t1 = a1 ^ b1;
    uint64_t s1 = t1 ^ c0;
    uint64_t c1 = (a1 & b1) | (c0 & t1); // carry into plane2

    // Plane 2 (fours)
    uint64_t t2 = a2 ^ b2;
    uint64_t s2 = t2 ^ c1;
    uint64_t c2 = (a2 & b2) | (c1 & t2); // carry into plane3 (eights)

    r0 = s0;
    r1 = s1;
    r2 = s2;
    r3 = c2;
}

static __device__ __forceinline__
uint64_t shift_left_with_carry(uint64_t center, uint64_t left_word)
{
    // For bit 0, bring in bit 63 from the left neighbor word.
    return (center << 1) | (left_word >> 63);
}

static __device__ __forceinline__
uint64_t shift_right_with_carry(uint64_t center, uint64_t right_word)
{
    // For bit 63, bring in bit 0 from the right neighbor word.
    return (center >> 1) | (right_word << 63);
}

__global__ void life_kernel(const std::uint64_t* __restrict__ in,
                            std::uint64_t* __restrict__ out,
                            int grid_dim)
{
    const int words_per_row = grid_dim >> 6; // grid_dim / 64
    const std::size_t total_words = static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    const std::size_t gid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_words) return;

    // Map gid -> (row, col)
    const int row = static_cast<int>(gid / words_per_row);
    const int col = static_cast<int>(gid % words_per_row);

    // Determine boundary existence
    const bool has_left   = (col > 0);
    const bool has_right  = (col + 1 < words_per_row);
    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < grid_dim);

    // Index helpers
    const std::size_t row_base      = static_cast<std::size_t>(row) * words_per_row;
    const std::size_t above_base    = has_top    ? (row_base - words_per_row) : 0;
    const std::size_t below_base    = has_bottom ? (row_base + words_per_row) : 0;

    // Load current row words
    const std::uint64_t curC = in[row_base + col];
    const std::uint64_t curL = has_left  ? in[row_base + col - 1] : 0ull;
    const std::uint64_t curR = has_right ? in[row_base + col + 1] : 0ull;

    // Load above row words (if any)
    const std::uint64_t topC = has_top ? in[above_base + col] : 0ull;
    const std::uint64_t topL = (has_top && has_left)  ? in[above_base + col - 1] : 0ull;
    const std::uint64_t topR = (has_top && has_right) ? in[above_base + col + 1] : 0ull;

    // Load below row words (if any)
    const std::uint64_t botC = has_bottom ? in[below_base + col] : 0ull;
    const std::uint64_t botL = (has_bottom && has_left)  ? in[below_base + col - 1] : 0ull;
    const std::uint64_t botR = (has_bottom && has_right) ? in[below_base + col + 1] : 0ull;

    // Horizontal neighbor masks for each of the three rows
    // Above row: three contributions (left/center/right)
    const std::uint64_t aL = shift_left_with_carry(topC, topL);
    const std::uint64_t aC = topC;
    const std::uint64_t aR = shift_right_with_carry(topC, topR);

    // Middle row: only left and right (exclude center cell from neighbor count)
    const std::uint64_t mL = shift_left_with_carry(curC, curL);
    const std::uint64_t mR = shift_right_with_carry(curC, curR);

    // Below row: three contributions (left/center/right)
    const std::uint64_t bL = shift_left_with_carry(botC, botL);
    const std::uint64_t bC = botC;
    const std::uint64_t bR = shift_right_with_carry(botC, botR);

    // Horizontal sums within each row
    // Top row (3 inputs): ones_t + 2*twos_t
    const std::uint64_t ones_t = aL ^ aC ^ aR;
    const std::uint64_t twos_t = (aL & aC) | (aL & aR) | (aC & aR);

    // Middle row (2 inputs): ones_m + 2*twos_m
    const std::uint64_t ones_m = mL ^ mR;
    const std::uint64_t twos_m = mL & mR;

    // Bottom row (3 inputs): ones_b + 2*twos_b
    const std::uint64_t ones_b = bL ^ bC ^ bR;
    const std::uint64_t twos_b = (bL & bC) | (bL & bR) | (bC & bR);

    // Sum the three per-row horizontal counts using bit-sliced addition:
    // First add top + bottom -> s0 + 2*s1 + 4*s2 (+8*s3, expected zero here), then add middle.
    uint64_t s0, s1, s2, s3;
    add_planes3(ones_t, twos_t, 0ull, ones_b, twos_b, 0ull, s0, s1, s2, s3);

    uint64_t n0, n1, n2, n3;
    add_planes3(s0, s1, s2, ones_m, twos_m, 0ull, n0, n1, n2, n3);

    // n0,n1,n2,n3 are the bit-planes for the per-bit neighbor counts:
    // count = n0 + 2*n1 + 4*n2 + 8*n3 (n3 can be 1 only when count==8)

    // Game of Life rules:
    // births: count == 3 -> n0=1, n1=1, n2=0, n3=0
    // survivals: count == 2 -> n0=0, n1=1, n2=0, n3=0, AND current cell alive (curC bit)
    const uint64_t not_n2 = ~n2;
    const uint64_t not_n3 = ~n3;

    const uint64_t eq3 = n0 & n1 & not_n2 & not_n3;
    const uint64_t eq2 = (~n0) & n1 & not_n2 & not_n3;

    const uint64_t next = eq3 | (eq2 & curC);

    out[gid] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Launch configuration: 1D grid where each thread processes one 64-bit word.
    // Choose a block size that maps well to the hardware; 256 is a good default on A100/H100.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64, guaranteed integer
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    life_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
}