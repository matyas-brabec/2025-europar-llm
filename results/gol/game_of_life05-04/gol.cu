#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life
  on a bit-packed grid where each 64-bit word encodes 64 consecutive cells
  in a row (LSB -> lowest column index within the word).
  
  Key ideas for performance:
  - Each thread processes one 64-bit word (64 cells) to avoid atomics.
  - Use bit-sliced arithmetic (logical half/full adders) to count neighbors
    for all 64 cells in parallel without cross-bit carries.
  - Handle word boundaries by injecting carry-in bits from neighbor words
    when shifting horizontally.
  - Handle grid boundaries by treating out-of-bounds neighbor words as zeros.
  
  Notation:
  - For a given row bitboard W at word position (y, x):
      left-shift-with-carry(W, W_L):   inject bit63 from left neighbor word into bit0
      right-shift-with-carry(W, W_R):  inject bit0 from right neighbor word into bit63
  - Horizontal sums:
      For upper and lower rows: sum of {left, center, right} -> 2-bit result (0..3)
      For current row: sum of {left, right} -> 2-bit result (0..2), excludes center cell
  - Vertical combination:
      Add the three 2-bit horizontal sums to produce a 4-bit count (0..8) per bit position.
      Then apply Life rule: next = (count == 3) | (alive & (count == 2))
*/

// Safe, branchless horizontal shifts with boundary injection from neighbor words
static __device__ __forceinline__
std::uint64_t shl1_in(std::uint64_t w, std::uint64_t w_left) {
    // Bring MSB of left word into bit0
    return (w << 1) | (w_left >> 63);
}

static __device__ __forceinline__
std::uint64_t shr1_in(std::uint64_t w, std::uint64_t w_right) {
    // Bring LSB of right word into bit63
    return (w >> 1) | (w_right << 63);
}

// Half-adder for per-bit logic: adds two 1-bit bitboards (no cross-bit carry propagation)
static __device__ __forceinline__
void half_add_u1(std::uint64_t a, std::uint64_t b, std::uint64_t& sum, std::uint64_t& carry) {
    sum   = a ^ b;   // bitwise sum (LSB)
    carry = a & b;   // bitwise carry (MSB)
}

// Full-adder for per-bit logic: adds three 1-bit bitboards (no cross-bit carry propagation)
static __device__ __forceinline__
void full_add_u1(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                 std::uint64_t& sum, std::uint64_t& carry) {
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// Add three 1-bit bitboards -> 2-bit result (lo, hi) in bit-sliced form
static __device__ __forceinline__
void add3_u1_to_u2(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                   std::uint64_t& lo, std::uint64_t& hi) {
    full_add_u1(a, b, c, lo, hi);
}

// Add two 1-bit bitboards -> 2-bit result (lo, hi) in bit-sliced form
static __device__ __forceinline__
void add2_u1_to_u2(std::uint64_t a, std::uint64_t b,
                   std::uint64_t& lo, std::uint64_t& hi) {
    half_add_u1(a, b, lo, hi);
}

// Add two 2-bit numbers (a1:a0) + (b1:b0) -> 3-bit result (s2:s1:s0)
static __device__ __forceinline__
void add_u2_u2_to_u3(std::uint64_t a0, std::uint64_t a1,
                     std::uint64_t b0, std::uint64_t b1,
                     std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2) {
    // Low bit addition
    std::uint64_t c0;
    half_add_u1(a0, b0, s0, c0); // s0 = a0 ^ b0, c0 = a0 & b0

    // Middle bit addition with carry-in c0
    std::uint64_t s1_tmp, c1_ab;
    half_add_u1(a1, b1, s1_tmp, c1_ab); // s1_tmp = a1 ^ b1, c1_ab = a1 & b1

    // Full add for s1: s1_tmp + c0
    s1 = s1_tmp ^ c0;
    std::uint64_t c1 = c1_ab | (s1_tmp & c0); // carry out from s1

    // High bit is the remaining carry
    s2 = c1;
}

// Add a 3-bit number (a2:a1:a0) and a 2-bit number (b1:b0) -> 4-bit result (s3:s2:s1:s0)
static __device__ __forceinline__
void add_u3_u2_to_u4(std::uint64_t a0, std::uint64_t a1, std::uint64_t a2,
                     std::uint64_t b0, std::uint64_t b1,
                     std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) {
    // Bit 0
    std::uint64_t c0;
    half_add_u1(a0, b0, s0, c0);

    // Bit 1 with carry-in c0
    std::uint64_t s1_tmp, c1_ab;
    half_add_u1(a1, b1, s1_tmp, c1_ab);
    s1 = s1_tmp ^ c0;
    std::uint64_t c1 = c1_ab | (s1_tmp & c0);

    // Bit 2 with carry-in c1
    s2 = a2 ^ c1;
    std::uint64_t c2 = a2 & c1;

    // Bit 3 is final carry
    s3 = c2;
}

// CUDA kernel: processes one 64-bit word (64 cells) per thread
static __global__
void life_step_kernel(const std::uint64_t* __restrict__ input,
                      std::uint64_t* __restrict__ output,
                      int grid_dim, int words_per_row)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_words = static_cast<unsigned int>(grid_dim) * static_cast<unsigned int>(words_per_row);
    if (idx >= total_words) return;

    // Compute 2D word coordinates (y, x) from linear index
    const unsigned int y = idx / static_cast<unsigned int>(words_per_row);
    const unsigned int x = idx - y * static_cast<unsigned int>(words_per_row);

    // Boundary flags
    const bool hasL = (x > 0);
    const bool hasR = (x + 1u < static_cast<unsigned int>(words_per_row));
    const bool hasU = (y > 0);
    const bool hasD = (y + 1u < static_cast<unsigned int>(grid_dim));

    // Helper to safely load neighbor words (coalesced reads for common case)
    auto ldg = [&](unsigned int i) -> std::uint64_t {
#if __CUDA_ARCH__ >= 350
        return __ldg(&input[i]);
#else
        return input[i];
#endif
    };

    // Indices for this word and neighbors
    const unsigned int row_stride = static_cast<unsigned int>(words_per_row);
    const unsigned int idxC = idx;
    const unsigned int idxL = idxC - 1u;
    const unsigned int idxR = idxC + 1u;
    const unsigned int idxU = idxC - row_stride;
    const unsigned int idxD = idxC + row_stride;
    const unsigned int idxUL = idxU - 1u;
    const unsigned int idxUR = idxU + 1u;
    const unsigned int idxDL = idxD - 1u;
    const unsigned int idxDR = idxD + 1u;

    // Load current row words
    const std::uint64_t C  = ldg(idxC);
    const std::uint64_t CL = hasL ? ldg(idxL) : 0ull;
    const std::uint64_t CR = hasR ? ldg(idxR) : 0ull;

    // Load upper row words (or zeros at boundary)
    const std::uint64_t U  = hasU ? ldg(idxU)           : 0ull;
    const std::uint64_t UL = (hasU && hasL) ? ldg(idxUL) : 0ull;
    const std::uint64_t UR = (hasU && hasR) ? ldg(idxUR) : 0ull;

    // Load lower row words (or zeros at boundary)
    const std::uint64_t D  = hasD ? ldg(idxD)           : 0ull;
    const std::uint64_t DL = (hasD && hasL) ? ldg(idxDL) : 0ull;
    const std::uint64_t DR = (hasD && hasR) ? ldg(idxDR) : 0ull;

    // Horizontal contributions for upper row (includes center)
    const std::uint64_t U_left   = shl1_in(U, UL);
    const std::uint64_t U_center = U;
    const std::uint64_t U_right  = shr1_in(U, UR);
    std::uint64_t HUp0, HUp1; // 2-bit horizontal sum for upper row
    add3_u1_to_u2(U_left, U_center, U_right, HUp0, HUp1);

    // Horizontal contributions for current row (exclude center cell)
    const std::uint64_t C_left  = shl1_in(C, CL);
    const std::uint64_t C_right = shr1_in(C, CR);
    std::uint64_t HCur0, HCur1; // 2-bit horizontal sum for current row (0..2)
    add2_u1_to_u2(C_left, C_right, HCur0, HCur1);

    // Horizontal contributions for lower row (includes center)
    const std::uint64_t D_left   = shl1_in(D, DL);
    const std::uint64_t D_center = D;
    const std::uint64_t D_right  = shr1_in(D, DR);
    std::uint64_t HDn0, HDn1; // 2-bit horizontal sum for lower row
    add3_u1_to_u2(D_left, D_center, D_right, HDn0, HDn1);

    // Vertical accumulation: (Upper + Lower) -> 3-bit
    std::uint64_t S0, S1, S2; // partial sum 3-bit
    add_u2_u2_to_u3(HUp0, HUp1, HDn0, HDn1, S0, S1, S2);

    // Add current row horizontal sum to get full 4-bit neighbor count (0..8)
    std::uint64_t N0, N1, N2, N3;
    add_u3_u2_to_u4(S0, S1, S2, HCur0, HCur1, N0, N1, N2, N3);

    // Apply Game of Life rules:
    // - alive next if (count == 3) OR (alive AND count == 2)
    // count == 2 -> 0010: ~N3 & ~N2 &  N1 & ~N0
    // count == 3 -> 0011: ~N3 & ~N2 &  N1 &  N0
    const std::uint64_t notN3 = ~N3;
    const std::uint64_t notN2 = ~N2;
    const std::uint64_t eq2   = notN3 & notN2 &  N1 & ~N0;
    const std::uint64_t eq3   = notN3 & notN2 &  N1 &  N0;

    const std::uint64_t next = eq3 | (C & eq2);

    output[idxC] = next;
}

// Host function: launches one kernel step of Game of Life
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each row has grid_dimensions / 64 words (grid_dimensions is power-of-two >= 512, hence divisible by 64).
    const int words_per_row = grid_dimensions >> 6;
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Choose a block size that offers good occupancy on A100/H100
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row);
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
