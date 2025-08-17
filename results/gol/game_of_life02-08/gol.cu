#include <cuda_runtime.h>
#include <cstdint>

// This implementation computes one generation update of Conway's Game of Life
// on a square, bit-packed grid where each 64-bit word encodes 64 cells in a row.
// The algorithm uses bit-sliced arithmetic to sum the 8 neighbors for all 64 cells
// in a word in parallel, then computes the next state using logical operations.
// It handles boundaries by treating all cells outside the grid as dead (zeros).
//
// Key points:
// - Each thread processes one 64-bit word (64 cells) at coordinates (row, word_index).
// - Neighbor contributions are computed from three rows (north, current, south)
//   and three horizontal positions (west, center, east) with cross-word carry for shifts.
// - The 8 neighbor masks (NW,N,NE,W,E,SW,S,SE) are summed using bit-sliced adders,
//   yielding a 4-bit count per bit-position (0..8).
// - Next state per bit-position is: (neighbors == 3) OR (self AND neighbors == 2).
// - No shared or texture memory is used; global loads are coalesced by mapping thread.x
//   across words within a row.

static __device__ __forceinline__ std::uint64_t shl_with_carry(std::uint64_t center, std::uint64_t left, bool has_left) {
    // Shift left within a row (towards MSB). For interior words, inject the MSB of the left word into bit 0.
    // At the leftmost grid boundary (has_left == false), inject 0 as the outside world is dead.
    return (center << 1) | (has_left ? (left >> 63) : 0ULL);
}

static __device__ __forceinline__ std::uint64_t shr_with_carry(std::uint64_t center, std::uint64_t right, bool has_right) {
    // Shift right within a row (towards LSB). For interior words, inject the LSB of the right word into bit 63.
    // At the rightmost grid boundary (has_right == false), inject 0 as the outside world is dead.
    return (center >> 1) | (has_right ? ((right & 1ULL) << 63) : 0ULL);
}

static __device__ __forceinline__ void add3_bits(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                 std::uint64_t &low, std::uint64_t &high) {
    // Bit-sliced addition of three 1-bit numbers per bit position (a + b + c).
    // Returns:
    //   low  = sum bit (LSB) of the 2-bit result per position
    //   high = carry bit (MSB) per position
    // Truth: low = a ^ b ^ c
    //        high = (a&b) | (b&c) | (a&c) which can be computed as (a&b) | ((a^b)&c)
    std::uint64_t t = a ^ b;
    low  = t ^ c;
    high = (a & b) | (t & c);
}

static __device__ __forceinline__ void add2_bits(std::uint64_t a, std::uint64_t b,
                                                 std::uint64_t &low, std::uint64_t &high) {
    // Bit-sliced addition of two 1-bit numbers per position.
    // Returns low (sum bit) and high (carry bit).
    low  = a ^ b;
    high = a & b;
}

__global__ void life_step_kernel(const std::uint64_t* __restrict__ in,
                                 std::uint64_t* __restrict__ out,
                                 int grid_dim, int words_per_row)
{
    // Map each thread to a (row, word_index) position
    int xw = blockIdx.x * blockDim.x + threadIdx.x; // word index in row
    int y  = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (y >= grid_dim || xw >= words_per_row) return;

    const int W = words_per_row;
    const int idx = y * W + xw;

    // Boundary flags
    const bool has_left  = (xw > 0);
    const bool has_right = (xw + 1 < W);
    const bool has_north = (y > 0);
    const bool has_south = (y + 1 < grid_dim);

    // Load current row words (left, center, right)
    const std::uint64_t cC = in[idx];
    const std::uint64_t cL = has_left  ? in[idx - 1] : 0ULL;
    const std::uint64_t cR = has_right ? in[idx + 1] : 0ULL;

    // Load north row words (left, center, right), or 0 if outside
    std::uint64_t nL = 0ULL, nC = 0ULL, nR = 0ULL;
    if (has_north) {
        const int nIdx = idx - W;
        nC = in[nIdx];
        nL = has_left  ? in[nIdx - 1] : 0ULL;
        nR = has_right ? in[nIdx + 1] : 0ULL;
    }

    // Load south row words (left, center, right), or 0 if outside
    std::uint64_t sL = 0ULL, sC = 0ULL, sR = 0ULL;
    if (has_south) {
        const int sIdx = idx + W;
        sC = in[sIdx];
        sL = has_left  ? in[sIdx - 1] : 0ULL;
        sR = has_right ? in[sIdx + 1] : 0ULL;
    }

    // Compute the 8 neighbor bitmasks using shifts with cross-word carries.
    // Current row (no center contribution):
    const std::uint64_t Wm = shl_with_carry(cC, cL, has_left);   // West neighbor (cell at col-1)
    const std::uint64_t Em = shr_with_carry(cC, cR, has_right);  // East neighbor (cell at col+1)

    // North row (three contributions): NW, N, NE
    const std::uint64_t Nm  = nC;
    const std::uint64_t NWm = shl_with_carry(nC, nL, has_left);
    const std::uint64_t NEm = shr_with_carry(nC, nR, has_right);

    // South row (three contributions): SW, S, SE
    const std::uint64_t Sm  = sC;
    const std::uint64_t SWm = shl_with_carry(sC, sL, has_left);
    const std::uint64_t SEm = shr_with_carry(sC, sR, has_right);

    // Sum the 8 neighbor masks using bit-sliced arithmetic.
    // First, sum triples in north and south rows, and the pair in current row.
    std::uint64_t n_low, n_high;
    add3_bits(NWm, Nm, NEm, n_low, n_high); // north triple -> value = n_low + 2*n_high

    std::uint64_t s_low, s_high;
    add3_bits(SWm, Sm, SEm, s_low, s_high); // south triple -> value = s_low + 2*s_high

    std::uint64_t m_low, m_high;
    add2_bits(Wm, Em, m_low, m_high);       // current row pair -> value = m_low + 2*m_high

    // Add north and south partial sums: (n_low + 2*n_high) + (s_low + 2*s_high) = A0 + 2*A1 + 4*A2
    const std::uint64_t A0 = n_low ^ s_low;            // units place
    const std::uint64_t C0 = n_low & s_low;            // carry to 2's place

    const std::uint64_t t1 = n_high ^ s_high;          // for 2's place full-adder
    const std::uint64_t A1 = t1 ^ C0;                  // 2's place
    const std::uint64_t C1 = (n_high & s_high) | (t1 & C0); // carry to 4's place

    const std::uint64_t A2 = C1;                       // 4's place

    // Now add current row pair (m_low + 2*m_high): Total = (A0 + 2*A1 + 4*A2) + (m_low + 2*m_high)
    const std::uint64_t S0 = A0 ^ m_low;               // units place
    const std::uint64_t C2 = A0 & m_low;               // carry to 2's place

    const std::uint64_t t2 = A1 ^ m_high;
    const std::uint64_t S1 = t2 ^ C2;                  // 2's place
    const std::uint64_t C3 = (A1 & m_high) | (t2 & C2);// carry to 4's place

    const std::uint64_t S2 = A2 ^ C3;                  // 4's place
    const std::uint64_t S3 = A2 & C3;                  // 8's place (only set when neighbors==8)

    // Compute next state:
    // next = (neighbors == 3) | (self & (neighbors == 2))
    // neighbors == 3 => S3=0, S2=0, S1=1, S0=1
    // neighbors == 2 => S3=0, S2=0, S1=1, S0=0
    const std::uint64_t not_S3 = ~S3;
    const std::uint64_t not_S2 = ~S2;

    const std::uint64_t eq3 = not_S3 & not_S2 & S1 & S0;
    const std::uint64_t eq2 = not_S3 & not_S2 & S1 & (~S0);

    const std::uint64_t next_word = eq3 | (cC & eq2);

    out[idx] = next_word;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions: width and height in cells (square grid). Assumed power of two and >= 512.
    // Each 64 cells are packed into one 64-bit word.
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // Configure launch parameters:
    // Use 2D blocks to keep loads coalesced across words in a row and to cover multiple rows per block,
    // reducing gridDim.y requirements and improving occupancy. 128x8 = 1024 threads/block (max).
    dim3 block(128, 8, 1);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y,
              1);

    life_step_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
