#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// - Each 64-bit word encodes 64 horizontally consecutive cells within a row.
// - Each CUDA thread processes exactly one 64-bit word.
// - The algorithm uses bit-sliced addition to compute per-bit neighbor counts (0..8)
//   without per-bit loops or atomics, handling cross-word (bit 0 and bit 63) neighbors
//   by incorporating carry bits from adjacent words.
// - All out-of-bounds cells are treated as dead (zero padding for boundary handling).
// - Shared/texture memory is intentionally not used; global memory is sufficient.
// - The grid dimension (width == height) is always a power of two; thus, the number
//   of 64-bit words per row is also a power of two. We exploit that to replace
//   division/mod by bitwise shifts & masks for indexing.

// Convenience alias
using u64 = std::uint64_t;

// Load from global memory with read-only cache hint (when available).
// On modern GPUs, the compiler generally optimizes const loads appropriately;
// __ldg can still be beneficial. Fallback is a plain load if not supported.
static __device__ __forceinline__ u64 ldg64(const u64* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Sum of three 1-bit bitboards per bit position producing a 2-bit result (0..3).
// Inputs a, b, c are 64-bit masks where each bit represents a 0/1 value to sum.
// Output: lo = bit0 of the sum, hi = bit1 of the sum.
static __device__ __forceinline__ void sum3_1bit(u64 a, u64 b, u64 c, u64& lo, u64& hi) {
    u64 ab_x = a ^ b;                      // partial sum of a+b, bitwise
    lo = ab_x ^ c;                         // low bit of a+b+c
    // hi bit is set if at least two of (a,b,c) are 1:
    // hi = (a&b) | (a&c) | (b&c) == (a&b) | (c&(a^b))
    hi = (a & b) | (c & ab_x);
}

// Sum of two 1-bit bitboards per bit position producing a 2-bit result (0..2).
// Inputs a, b are 64-bit masks. Outputs lo = bit0, hi = bit1.
static __device__ __forceinline__ void sum2_1bit(u64 a, u64 b, u64& lo, u64& hi) {
    lo = a ^ b;       // low bit
    hi = a & b;       // carry (bit1)
}

// Add two 2-bit numbers per bit position: (a1 a0) + (b1 b0) -> (s2 s1 s0), range 0..6.
// Each ai/bi/si is a 64-bit mask holding that bit across 64 lanes.
// s2 is the carry out of bit1 (the 3rd bit of the sum).
static __device__ __forceinline__ void add2bit_2bit(u64 a0, u64 a1, u64 b0, u64 b1,
                                                    u64& s0, u64& s1, u64& s2) {
    // Add bit0
    u64 t0 = a0 ^ b0;
    u64 c0 = a0 & b0;

    // Add bit1 with carry c0
    u64 t1 = a1 ^ b1;
    u64 c1 = a1 & b1;

    s0 = t0;
    s1 = t1 ^ c0;
    s2 = c1 | (t1 & c0);
}

// Add a 2-bit number (m1 m0) to a 3-bit number (s2 s1 s0) -> 4-bit (r3 r2 r1 r0), range 0..8.
static __device__ __forceinline__ void add2bit_3bit(u64 s0, u64 s1, u64 s2, u64 m0, u64 m1,
                                                    u64& r0, u64& r1, u64& r2, u64& r3) {
    // Add bit0
    r0 = s0 ^ m0;
    u64 c0 = s0 & m0;

    // Add bit1 with carry c0
    u64 t1 = s1 ^ m1;
    u64 c1 = s1 & m1;
    r1 = t1 ^ c0;
    u64 c01 = c1 | (t1 & c0);

    // Add bit2 with carry c01
    r2 = s2 ^ c01;
    r3 = s2 & c01; // final carry is bit3
}

// Compute (lo,hi) for the three horizontally adjacent bits from a row word C,
// using neighboring words L and R for cross-word carries (bit 0 and bit 63).
// This computes, for each bit position, the count of 1s among {left, center, right} in that row.
static __device__ __forceinline__ void horizontal_sum3(const u64 L, const u64 C, const u64 R,
                                                       u64& lo, u64& hi) {
    // Left neighbor aligned to center: shift left with carry from L's bit63 -> bit0
    u64 left = (C << 1) | (L >> 63);
    // Right neighbor aligned to center: shift right with carry from R's bit0 -> bit63
    u64 right = (C >> 1) | (R << 63);
    sum3_1bit(left, C, right, lo, hi);
}

// Compute (lo,hi) for the two horizontal neighbors from the current row word C,
// using neighboring words L and R for cross-word carries (bit 0 and bit 63).
// This omits the center cell itself (since a cell is not its own neighbor).
static __device__ __forceinline__ void horizontal_sum2_neighbors_only(const u64 L, const u64 C, const u64 R,
                                                                      u64& lo, u64& hi) {
    u64 left  = (C << 1) | (L >> 63);
    u64 right = (C >> 1) | (R << 63);
    sum2_1bit(left, right, lo, hi);
}

// Compute one step of Game of Life for a bit-packed square grid.
// Each thread processes one 64-bit word (64 adjacent horizontal cells).
// words_per_row is power of two; log2_words_per_row = log2(words_per_row) for fast index math.
__global__ void gol_step_kernel(const u64* __restrict__ in,
                                u64* __restrict__ out,
                                int grid_dim,                 // number of rows == number of columns
                                int words_per_row,            // grid_dim / 64, power-of-two
                                int log2_words_per_row) {     // log2(words_per_row)
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_words = static_cast<unsigned int>(grid_dim) * static_cast<unsigned int>(words_per_row);
    if (tid >= total_words) return;

    // Map linear word index -> (row y, word index x) using power-of-two properties
    const int x = static_cast<int>(tid & (words_per_row - 1));
    const int y = static_cast<int>(tid >> log2_words_per_row);

    const bool has_left  = (x > 0);
    const bool has_right = (x + 1 < words_per_row);
    const bool has_up    = (y > 0);
    const bool has_down  = (y + 1 < grid_dim);

    // Row base pointers (conditionally valid)
    const u64* row_cur   = in + static_cast<size_t>(y) * words_per_row;
    const u64* row_up    = has_up   ? (row_cur - words_per_row) : nullptr;
    const u64* row_down  = has_down ? (row_cur + words_per_row) : nullptr;

    // Load 9 relevant words: (Left, Center, Right) for Up, Cur, Down rows.
    // Out-of-bounds neighbors are treated as 0.
    u64 UL = (has_up   && has_left ) ? ldg64(row_up  + (x - 1)) : 0ull;
    u64 UC = (has_up               ) ? ldg64(row_up  +  x      ) : 0ull;
    u64 UR = (has_up   && has_right) ? ldg64(row_up  + (x + 1)) : 0ull;

    u64 CL = (has_left            ) ? ldg64(row_cur + (x - 1)) : 0ull;
    u64 CC =                        ldg64(row_cur +  x      );
    u64 CR = (has_right           ) ? ldg64(row_cur + (x + 1)) : 0ull;

    u64 DL = (has_down && has_left ) ? ldg64(row_down + (x - 1)) : 0ull;
    u64 DC = (has_down            ) ? ldg64(row_down +  x      ) : 0ull;
    u64 DR = (has_down && has_right) ? ldg64(row_down + (x + 1)) : 0ull;

    // Horizontal sums per row:
    // Up and Down rows: sum of three neighbors (left, center, right)
    u64 up_lo, up_hi;
    horizontal_sum3(UL, UC, UR, up_lo, up_hi);

    u64 dn_lo, dn_hi;
    horizontal_sum3(DL, DC, DR, dn_lo, dn_hi);

    // Current row: sum of two neighbors (left, right), excluding center cell
    u64 mid_lo, mid_hi;
    horizontal_sum2_neighbors_only(CL, CC, CR, mid_lo, mid_hi);

    // Sum up (Up + Down) -> 3-bit number per bit (range 0..6)
    u64 s0, s1, s2;
    add2bit_2bit(up_lo, up_hi, dn_lo, dn_hi, s0, s1, s2);

    // Add Mid -> 4-bit number per bit (range 0..8)
    u64 n0, n1, n2, n3;
    add2bit_3bit(s0, s1, s2, mid_lo, mid_hi, n0, n1, n2, n3);

    // Compute masks for neighbor count == 2 and == 3:
    // exactly2: n3=0, n2=0, n1=1, n0=0
    // exactly3: n3=0, n2=0, n1=1, n0=1
    u64 high_zero = ~(n3 | n2);
    u64 exactly3 = high_zero & n1 &  n0;
    u64 exactly2 = high_zero & n1 & ~n0;

    // Apply Game of Life rules:
    // Alive next if (neighbors == 3) OR (alive and neighbors == 2)
    u64 next = exactly3 | (exactly2 & CC);

    out[tid] = next;
}

// Host function to launch one Game of Life step.
// - input:  device pointer to bit-packed input grid (grid_dim x grid_dim cells)
// - output: device pointer to bit-packed output grid (same layout)
// - grid_dimensions: width == height of the grid, power of two (> 512)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Number of 64-bit words per row (grid_dim / 64). grid_dimensions is a power of two,
    // hence words_per_row is also a power of two.
    const int words_per_row = grid_dimensions >> 6;

    // Compute log2(words_per_row) to replace division/modulo with shifts/masks in the kernel.
    // Since words_per_row is power-of-two, count trailing zeros equals the log2.
    int log2_words_per_row = 0;
#if defined(__GNUG__) || defined(__clang__)
    log2_words_per_row = __builtin_ctz(words_per_row);
#else
    {
        int w = words_per_row;
        while ((w >> log2_words_per_row) > 1) { ++log2_words_per_row; }
    }
#endif

    const size_t total_words = static_cast<size_t>(grid_dimensions) * static_cast<size_t>(words_per_row);

    // Launch configuration: one thread per 64-bit word.
    // Choose a performant block size; 256 is a good default for A100/H100.
    const int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    gol_step_kernel<<<grid_size, block_size>>>(input, output,
                                               grid_dimensions,
                                               words_per_row,
                                               log2_words_per_row);
    // Synchronization is intentionally omitted as per problem statement; the caller handles it.
}