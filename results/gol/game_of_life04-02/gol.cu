#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel implementing one step of Conway's Game of Life using bit-parallel logic.
// Each thread processes one 64-bit word (64 cells in a row). The grid is square with
// side length 'grid_dim' cells, and words_per_row = grid_dim / 64.
// The input and output arrays are word-packed row-major buffers of size words_per_row * grid_dim.
//
// Algorithm overview (all per-thread, per-word):
// - Load up to 9 neighboring 64-bit words: the current word (C) and its 8 neighbors at the word level:
//     [AL | A | AR]
//     [ L | C |  R]
//     [BL | B | BR]
//   where A is the word above C, B is the word below C, etc. Out-of-bound words are treated as zero.
// - Build 8 bitboards of neighbors by shifting with injected carry bits from neighbor words
//   so that the 0th and 63rd bit neighborhoods are correctly handled:
//     UL = (A << 1) | (AL >> 63)
//     U  = A
//     UR = (A >> 1) | (AR << 63)
//     L  = (C << 1) | (L  >> 63)
//     R  = (C >> 1) | (R  << 63)
//     DL = (B << 1) | (BL >> 63)
//     D  = B
//     DR = (B >> 1) | (BR << 63)
//   Note: If the corresponding neighbor word does not exist (grid boundary), its contribution is zero.
// - Compute the per-bit neighbor counts (0..8) across these 8 bitboards using a carry-save adder (CSA) network,
//   producing 4 bitplanes p0..p3 that represent the binary count for all 64 lanes simultaneously.
//   This avoids cross-lane carry propagation and is highly efficient on modern GPUs.
// - Apply the Life rules using these bitplanes:
//     next = (count == 3) | (C & (count == 2))
//   where equality tests are done bitwise on the bitplanes:
//     count == 3 -> ~p3 & ~p2 &  p1 &  p0
//     count == 2 -> ~p3 & ~p2 &  p1 & ~p0
//
// Notes:
// - This approach naturally includes the 0th and 63rd bit edge handling via neighbor word injections.
// - Using shared or texture memory is unnecessary; global loads are coalesced for most accesses.
// - We avoid per-cell __popc calls by computing 64 cells in parallel using bit-sliced addition.
//   While __popc/__popcll can be effective in other designs, the CSA bit-slicing generally outperforms
//   per-bit popcount for this packed representation on data-center GPUs.
// - Division/modulo to map linear index -> (row,col) is avoided since words_per_row is a power of two.
//   We pass row_shift = log2(words_per_row) and compute row = idx >> row_shift, col = idx & (words_per_row - 1).

// Carry-Save Adder: computes sum and carry (next bitplane) for a + b + c per bit-lane.
// sum = a ^ b ^ c
// carry (to next plane) = (a & b) | ((a ^ b) & c)
// All operations are 64-bit bitwise and do not propagate carries across bit positions (lanes).
static __device__ __forceinline__
void csa_u64(uint64_t a, uint64_t b, uint64_t c, uint64_t &sum, uint64_t &carry) {
    uint64_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (u & c);
}

__global__ void life_kernel_bitcs(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    int grid_dim,                  // number of cells per row/column (power of two)
    int words_per_row,             // grid_dim / 64
    int row_shift                  // log2(words_per_row)
) {
    // 1D launch: one thread per 64-bit word
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_words = static_cast<unsigned int>(grid_dim) * static_cast<unsigned int>(words_per_row);
    if (idx >= total_words) return;

    // Compute (row, col_in_words) without division using the fact that words_per_row is a power of two
    const unsigned int row = idx >> row_shift;
    const unsigned int col = idx & (static_cast<unsigned int>(words_per_row) - 1);

    // Boundary checks for neighboring words
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1u < static_cast<unsigned int>(grid_dim));
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1u < static_cast<unsigned int>(words_per_row));

    // Compute base indices for the three rows in word space
    const unsigned int row_base     = row * static_cast<unsigned int>(words_per_row);
    const unsigned int row_above    = has_up   ? (row_base - static_cast<unsigned int>(words_per_row)) : 0u;
    const unsigned int row_below    = has_down ? (row_base + static_cast<unsigned int>(words_per_row)) : 0u;

    // Load the 9 neighboring words (out-of-bounds -> zero)
    // Using __ldg to leverage read-only caching on supported architectures; safe fallback otherwise.
    const uint64_t C  = __ldg(input + (row_base + col));
    const uint64_t Lw = has_left  ? __ldg(input + (row_base + col - 1u)) : 0ull;
    const uint64_t Rw = has_right ? __ldg(input + (row_base + col + 1u)) : 0ull;

    const uint64_t A  = has_up    ? __ldg(input + (row_above + col)) : 0ull;
    const uint64_t AL = (has_up && has_left)  ? __ldg(input + (row_above + col - 1u)) : 0ull;
    const uint64_t AR = (has_up && has_right) ? __ldg(input + (row_above + col + 1u)) : 0ull;

    const uint64_t B  = has_down  ? __ldg(input + (row_below + col)) : 0ull;
    const uint64_t BL = (has_down && has_left)  ? __ldg(input + (row_below + col - 1u)) : 0ull;
    const uint64_t BR = (has_down && has_right) ? __ldg(input + (row_below + col + 1u)) : 0ull;

    // Build 8 neighbor bitboards with proper cross-word bit injection for bit 0 and bit 63.
    // Up-Left
    const uint64_t UL = (A << 1) | (has_left  ? (AL >> 63) : 0ull);
    // Up
    const uint64_t U  = A;
    // Up-Right
    const uint64_t UR = (A >> 1) | (has_right ? (AR << 63) : 0ull);
    // Left
    const uint64_t Lh = (C << 1) | (has_left  ? (Lw >> 63) : 0ull);
    // Right
    const uint64_t Rh = (C >> 1) | (has_right ? (Rw << 63) : 0ull);
    // Down-Left
    const uint64_t DL = (B << 1) | (has_left  ? (BL >> 63) : 0ull);
    // Down
    const uint64_t D  = B;
    // Down-Right
    const uint64_t DR = (B >> 1) | (has_right ? (BR << 63) : 0ull);

    // Carry-save adder network to sum eight 1-bit operands per lane into 4 bitplanes p0..p3.
    uint64_t s1, c1;
    csa_u64(UL, U,  UR, s1, c1);

    uint64_t s2, c2;
    csa_u64(Lh, Rh, DL, s2, c2);

    uint64_t s3, c3;
    csa_u64(D,  DR, 0ull, s3, c3);

    // Plane 0 (LSB of neighbor count) and carry into plane 1
    uint64_t p0, c4;
    csa_u64(s1, s2, s3, p0, c4);

    // Accumulate plane-1 carries from c1, c2, c3 and c4
    uint64_t u1, c5;
    csa_u64(c1, c2, c3, u1, c5);

    uint64_t p1, c6;
    csa_u64(u1, c4, 0ull, p1, c6);

    // Accumulate plane-2 carries
    uint64_t p2, c7;
    csa_u64(c5, c6, 0ull, p2, c7);

    // Plane 3 (MSB of neighbor count)
    const uint64_t p3 = c7;

    // Apply Game of Life rules using bitplanes:
    // eq3: neighbor count == 3
    // eq2: neighbor count == 2
    const uint64_t not_p3 = ~p3;
    const uint64_t not_p2 = ~p2;
    const uint64_t eq3 = not_p3 & not_p2 & p1 & p0;
    const uint64_t eq2 = not_p3 & not_p2 & p1 & (~p0);

    const uint64_t next = eq3 | (C & eq2);

    // Write out the next generation for this 64-bit word
    output[idx] = next;
}

// Host utility: compute integer log2 for a power-of-two 32-bit unsigned value.
// Used to avoid division when mapping linear index to (row,col) in the kernel.
static inline int ilog2_pow2_u32(unsigned int x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, x);
    return static_cast<int>(idx);
#else
    // Portable fallback (x is power of two): count trailing zeros
    int r = 0;
    while ((x >> r) != 1u) ++r;
    return r;
#endif
}

// Public API: Executes one step of Conway’s Game of Life on a bit-packed square grid.
// - input:  device pointer to the input bit grid (one uint64_t per 64 horizontal cells).
// - output: device pointer to store the next generation using the same packing.
// - grid_dimensions: width and height in cells (power of two, > 512).
//
// Assumptions:
// - input and output are valid device allocations (cudaMalloc).
// - The caller handles any necessary synchronization around this call.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions / 64; // guaranteed to be an integer and a power of two
    const int row_shift = ilog2_pow2_u32(static_cast<unsigned int>(words_per_row));
    const unsigned int total_words = static_cast<unsigned int>(grid_dimensions) * static_cast<unsigned int>(words_per_row);

    // Launch configuration: tuned for high occupancy and memory throughput
    constexpr int THREADS = 256;
    const unsigned int blocks = (total_words + THREADS - 1u) / THREADS;

    life_kernel_bitcs<<<blocks, THREADS>>>(input, output, grid_dimensions, words_per_row, row_shift);
}