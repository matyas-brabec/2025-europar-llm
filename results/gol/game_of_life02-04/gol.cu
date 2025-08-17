#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// High-level approach:
// - The grid is bit-packed: each uint64_t word encodes 64 cells in a row.
// - Each thread updates exactly one 64-bit word (i.e., 64 cells) for one row.
// - For a cell at position x in row y, we sum eight neighbors:
//     rows y-1, y, y+1 and columns x-1, x, x+1 (excluding the center cell).
// - We build eight 64-bit masks for these neighbors aligned to the current word,
//   then compute the per-bit neighbor count using bit-parallel binary counters.
// - The new state per bit is: (count == 3) | (alive & (count == 2)).
// - Outside-the-grid neighbors are treated as dead (0).
// - No shared or texture memory; memory accesses rely on L1/L2 caches.
// - Grid dimensions are power-of-two > 512, so width is always a multiple of 64.
//
// Performance notes:
// - 64-bit bitwise operations are very fast on A100/H100.
// - Memory accesses are coalesced: threads in a warp read neighboring words.
// - Avoids divergent branches except at borders.
// - Uses ripple-carry bitplane counters to sum the 8 neighbor masks without
//   cross-bit carries, preserving per-bit independence.

namespace {

// Adders for bitplane counters:
// We maintain a 4-plane counter (b0,b1,b2,b3) per bit position,
// representing the binary neighbor count (weights 1,2,4,8).
struct Counter4 {
    std::uint64_t b0;  // weight 1
    std::uint64_t b1;  // weight 2
    std::uint64_t b2;  // weight 4
    std::uint64_t b3;  // weight 8
};

// Add a weight-1 bitmask m into the counter.
__device__ __forceinline__ void add1(Counter4& c, std::uint64_t m) {
    std::uint64_t carry = c.b0 & m;
    c.b0 ^= m;
    m = carry;
    carry = c.b1 & m;
    c.b1 ^= m;
    m = carry;
    carry = c.b2 & m;
    c.b2 ^= m;
    m = carry;
    c.b3 ^= m;
}

// Add a weight-2 bitmask m into the counter (equivalent to adding m to plane b1).
__device__ __forceinline__ void add2(Counter4& c, std::uint64_t m) {
    std::uint64_t carry = c.b1 & m;
    c.b1 ^= m;
    m = carry;
    carry = c.b2 & m;
    c.b2 ^= m;
    m = carry;
    c.b3 ^= m;
}

// Shift helpers with cross-word carry from adjacent 64-bit words within the same row.
// These align neighbor cells into the current word's bit positions.
// - shift_left_with_carry: aligns west neighbors (x-1) to x; inject MSB of left word into bit 0.
// - shift_right_with_carry: aligns east neighbors (x+1) to x; inject LSB of right word into bit 63.
__device__ __forceinline__ std::uint64_t shift_left_with_carry(std::uint64_t x, std::uint64_t left_word) {
    return (x << 1) | (left_word >> 63);
}
__device__ __forceinline__ std::uint64_t shift_right_with_carry(std::uint64_t x, std::uint64_t right_word) {
    return (x >> 1) | ((right_word & 1ull) << 63);
}

// Compute the ones and twos planes (bitwise) for the sum of three 1-bit masks.
// Given a,b,c in {0,1} per bit position, returns:
//   ones = (a + b + c) mod 2
//   twos = ((a + b + c) >= 2) as a mask (i.e., carries of weight 2)
__device__ __forceinline__ void sum3(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                     std::uint64_t& ones, std::uint64_t& twos) {
    ones = a ^ b ^ c;
    twos = (a & b) | (a & c) | (b & c);
}

} // namespace

// Kernel: compute one Game-of-Life step on a square bit-packed grid.
__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int grid_dim) {
    // words_per_row is grid_dim / 64; grid_dim is a power of two > 512, so divisible by 64.
    const int words_per_row = grid_dim >> 6;

    const int xw = blockIdx.x * blockDim.x + threadIdx.x; // word index along the row
    const int y  = blockIdx.y;                            // row index

    if (xw >= words_per_row) return;

    const int row_stride = words_per_row;

    // Helper lambdas to safely fetch words with boundary checks.
    auto load_word = [&](int ry, int rx) -> std::uint64_t {
        if (ry < 0 || ry >= grid_dim || rx < 0 || rx >= words_per_row) return 0ull;
        return in[ry * row_stride + rx];
    };

    // Load center words for the three rows (top, mid, bottom).
    const std::uint64_t top    = load_word(y - 1, xw);
    const std::uint64_t middle = load_word(y,     xw);
    const std::uint64_t bottom = load_word(y + 1, xw);

    // Load neighbor words for cross-word shifts (left and right) for each row.
    const std::uint64_t topL    = load_word(y - 1, xw - 1);
    const std::uint64_t topR    = load_word(y - 1, xw + 1);
    const std::uint64_t midL    = load_word(y,     xw - 1);
    const std::uint64_t midR    = load_word(y,     xw + 1);
    const std::uint64_t bottomL = load_word(y + 1, xw - 1);
    const std::uint64_t bottomR = load_word(y + 1, xw + 1);

    // Build horizontally shifted masks for each row to align neighbor cells to the current positions.
    // For the middle row, exclude the center cell by not including 'middle' itself.
    const std::uint64_t t_west  = shift_left_with_carry(top,    topL);
    const std::uint64_t t_mid   = top; // center column from the top row
    const std::uint64_t t_east  = shift_right_with_carry(top,   topR);

    const std::uint64_t m_west  = shift_left_with_carry(middle, midL);
    const std::uint64_t m_east  = shift_right_with_carry(middle,midR);

    const std::uint64_t b_west  = shift_left_with_carry(bottom, bottomL);
    const std::uint64_t b_mid   = bottom; // center column from the bottom row
    const std::uint64_t b_east  = shift_right_with_carry(bottom,bottomR);

    // Sum the three columns per row into ones/twos bitplanes.
    std::uint64_t top_ones, top_twos;
    std::uint64_t bot_ones, bot_twos;
    sum3(t_west, t_mid, t_east, top_ones, top_twos);
    sum3(b_west, b_mid, b_east, bot_ones, bot_twos);

    // Middle row has only two contributors (left and right), no center.
    const std::uint64_t mid_ones = m_west ^ m_east;
    const std::uint64_t mid_twos = m_west & m_east;

    // Accumulate into a 4-plane counter.
    Counter4 c{0ull, 0ull, 0ull, 0ull};
    add1(c, top_ones);
    add2(c, top_twos);
    add1(c, mid_ones);
    add2(c, mid_twos);
    add1(c, bot_ones);
    add2(c, bot_twos);

    // Compute masks for counts == 2 and counts == 3.
    // count == 2: b3=0, b2=0, b1=1, b0=0
    // count == 3: b3=0, b2=0, b1=1, b0=1
    const std::uint64_t not_b2 = ~c.b2;
    const std::uint64_t not_b3 = ~c.b3;
    const std::uint64_t low_ok = not_b2 & not_b3;
    const std::uint64_t is_two   = low_ok & c.b1 & (~c.b0);
    const std::uint64_t is_three = low_ok & c.b1 & c.b0;

    const std::uint64_t alive = middle; // current state

    const std::uint64_t next = is_three | (is_two & alive);

    out[y * row_stride + xw] = next;
}

// Host function to launch one step of Conway’s Game of Life.
// - input/output are device pointers allocated with cudaMalloc.
// - grid_dimensions is the width/height of the square grid (power of 2).
// - The function launches a kernel that updates the entire grid once.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    // Choose a block size that balances occupancy and avoids too many idle threads.
    // Limit to 256 threads per block along X, but not exceeding words_per_row.
    const int block_x = std::min(256, words_per_row);
    dim3 block(block_x, 1, 1);
    dim3 grid((words_per_row + block.x - 1) / block.x, grid_dimensions, 1);

    gol_step_kernel<<<grid, block>>>(input, output, grid_dimensions);
}