#include <cuda_runtime.h>
#include <cstdint>

/*
Optimized CUDA implementation of one step of Conway’s Game of Life using bit-packed rows.

Key properties exploited:
- Each thread processes one 64-bit word (64 cells) of the grid, ensuring no atomics are needed.
- Grid width and height are powers of two; thus words_per_row is also a power of two. We use this to avoid divisions via bit-shifts and masks.
- Bit-parallel neighbor counting:
  - We compute horizontal neighbor contributions per row by shifting and combining adjacent 64-bit words to handle bit 0 and bit 63 boundaries using cross-word carry (from left/right neighbor words).
  - For each of the three relevant rows (above, current, below), we sum the three horizontal positions (west, center, east) using a 3:2 carry-save adder (CSA) to produce two bitmasks per row: s (LSB) and c (twos place).
  - We exclude the center cell from the middle row sum to match the 8-neighbor definition.
  - We then vertically add the three row results using bit-sliced arithmetic to produce the 4-bit per-cell neighbor count (bits b3 b2 b1 b0).
  - Rules application:
      next = (count == 3) | (alive & (count == 2))
    where count==2 or count==3 are derived from bit-sliced count bits.
- Memory accesses:
  - Each thread reads at most 9 words: its own word and its 8 adjacent words (left/right and three rows).
  - Boundary conditions are handled by treating out-of-bounds reads as zeros (outside-grid cells are dead).
- No shared or texture memory is used, as requested.

This kernel is optimized for modern data center GPUs (e.g., A100/H100) and uses a 1D grid-stride loop to handle arbitrarily large grids while avoiding division/modulus in the hot path.
*/

using u64 = std::uint64_t;

// 3-input bitwise adder: computes per-bit sum of a+b+c with no inter-bit carries.
// Returns: s = (a + b + c) & 1 (LSB), c2 = ((a + b + c) >> 1) & 1 (twos place)
// That is, numeric sum equals s + 2*c2 per bit position.
static __device__ __forceinline__
void add3(u64 a, u64 b, u64 c, u64 &s, u64 &c2)
{
    u64 t = a ^ b;
    s = t ^ c;
    c2 = (a & b) | (t & c);
}

// Shift right by 1 with cross-word carry from left neighbor (for "west" neighbors)
// Equivalent to: west = (x >> 1) | (left << 63)
static __device__ __forceinline__
u64 shift_west(u64 x, u64 left)
{
    return (x >> 1) | (left << 63);
}

// Shift left by 1 with cross-word carry from right neighbor (for "east" neighbors)
// Equivalent to: east = (x << 1) | (right >> 63)
static __device__ __forceinline__
u64 shift_east(u64 x, u64 right)
{
    return (x << 1) | (right >> 63);
}

__global__
void gol_step_kernel(const u64* __restrict__ in,
                     u64* __restrict__ out,
                     int words_per_row,
                     int rows,
                     int wshift) // wshift = log2(words_per_row), since words_per_row is power of two
{
    const int mask = words_per_row - 1; // for fast modulus (power of two)
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(rows);

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        // Compute (y, x) from idx without division/modulo using power-of-two property
        int y = static_cast<int>(idx >> wshift);
        int x = static_cast<int>(idx) & mask;

        // Determine boundary availability
        const bool hasU = (y > 0);
        const bool hasD = (y + 1 < rows);
        const bool hasL = (x > 0);
        const bool hasR = (x + 1 < words_per_row);

        // Load center row words
        const u64 C  = in[idx];
        const u64 L  = hasL ? in[idx - 1] : 0ull;
        const u64 R  = hasR ? in[idx + 1] : 0ull;

        // Load upper row words
        u64 UC = 0ull, UL = 0ull, UR = 0ull;
        if (hasU) {
            const std::size_t up_idx = idx - static_cast<std::size_t>(words_per_row);
            UC = in[up_idx];
            UL = hasL ? in[up_idx - 1] : 0ull;
            UR = hasR ? in[up_idx + 1] : 0ull;
        }

        // Load lower row words
        u64 DC = 0ull, DL = 0ull, DR = 0ull;
        if (hasD) {
            const std::size_t dn_idx = idx + static_cast<std::size_t>(words_per_row);
            DC = in[dn_idx];
            DL = hasL ? in[dn_idx - 1] : 0ull;
            DR = hasR ? in[dn_idx + 1] : 0ull;
        }

        // Construct neighbor bitmasks via shifts with cross-word carry.
        // Upper row contributions: NW, N, NE
        const u64 up_w = shift_west(UC, UL);
        const u64 up_c = UC;
        const u64 up_e = shift_east(UC, UR);

        // Middle row contributions: W, E (exclude center itself)
        const u64 mid_w = shift_west(C, L);
        const u64 mid_e = shift_east(C, R);

        // Lower row contributions: SW, S, SE
        const u64 dn_w = shift_west(DC, DL);
        const u64 dn_c = DC;
        const u64 dn_e = shift_east(DC, DR);

        // Sum horizontally per row using 3:2 compressors (carry-save adders).
        // For upper row: add up_w + up_c + up_e -> u1 (LSB) and u2 (twos place)
        u64 u1, u2;
        add3(up_w, up_c, up_e, u1, u2);

        // For middle row: add mid_w + mid_e (exclude center), which is a 2-input sum
        // Equivalent to s = mid_w ^ mid_e; c = mid_w & mid_e;
        const u64 m1 = (mid_w ^ mid_e);
        const u64 m2 = (mid_w & mid_e);

        // For lower row: add dn_w + dn_c + dn_e -> d1 (LSB) and d2 (twos place)
        u64 d1, d2;
        add3(dn_w, dn_c, dn_e, d1, d2);

        // Vertical accumulation:
        // Total neighbor count per bit = (u1 + m1 + d1) + 2*(u2 + m2 + d2)
        // Compute bit-sliced count bits b0 (1's), b1 (2's), b2 (4's), b3 (8's).

        // Sum of ones (u1, m1, d1) -> b0 and carry k2 to twos place.
        const u64 b0 = u1 ^ m1 ^ d1;
        const u64 k2 = (u1 & m1) | (u1 & d1) | (m1 & d1); // number of pairs among ones (0 or 1)

        // Sum of "twos units": k2 + u2 + m2 + d2.
        // Compute its 2-3 bit result (T = 0..4) using pairwise CSA-inspired logic.
        // Pairwise combine (k2,u2) and (m2,d2)
        const u64 s_ab = k2 ^ u2;
        const u64 c_ab = k2 & u2;

        const u64 s_cd = m2 ^ d2;
        const u64 c_cd = m2 & d2;

        // LSB of T (bit1 of final count)
        const u64 b1 = s_ab ^ s_cd;

        // Carry from adding s_ab + s_cd (contributes one more "two" unit)
        const u64 carry_l = s_ab & s_cd;

        // Now U2 = c_ab + c_cd + carry_l (0..2)
        // Its LSB is b2, and a carry when U2==2 produces b3.
        const u64 b2 = c_ab ^ c_cd ^ carry_l;
        const u64 b3 = (c_ab & c_cd) | (carry_l & (c_ab ^ c_cd)); // set only if U2 == 2

        // Apply Game of Life rules:
        // next = (count == 3) | (alive & (count == 2))
        // count==2: b3==0, b2==0, b1==1, b0==0
        // count==3: b3==0, b2==0, b1==1, b0==1
        const u64 not_b3 = ~b3;
        const u64 not_b2 = ~b2;
        const u64 eq_b1 = b1;       // b1 must be 1
        const u64 eq_b0 = b0;       // for ==3 requires b0=1; for ==2 requires b0=0

        const u64 common_mask = not_b3 & not_b2 & eq_b1;

        const u64 eq2 = common_mask & (~eq_b0);
        const u64 eq3 = common_mask & eq_b0;

        const u64 next = eq3 | (eq2 & C);

        out[idx] = next;
    }
}

static inline int ilog2_pow2_host(unsigned int x)
{
    // x is guaranteed to be a power of two in this context.
    int s = 0;
    while ((1u << s) != x) ++s;
    return s;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is the width and height (square), power of two, >= 512.
    // Each row has grid_dimensions / 64 words.
    const int words_per_row = grid_dimensions >> 6;
    const std::size_t total_words = static_cast<std::size_t>(grid_dimensions) * static_cast<std::size_t>(words_per_row);

    // Compute shift for fast div/mod by words_per_row
    const int wshift = ilog2_pow2_host(static_cast<unsigned int>(words_per_row));

    // Launch configuration: 1D grid-stride kernel
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const int threads_per_block = 256;

    // Choose a reasonable number of blocks to saturate the GPU while using a grid-stride loop.
    // Using a multiple of SM count helps with load balancing; adjust factor based on memory-bound nature.
    const int sm_count = prop.multiProcessorCount;
    const int target_blocks = sm_count * 20; // heuristic multiplier; 20 often good for memory-bound kernels

    // Ensure at least enough blocks to cover all work, but don't exceed very large grids unnecessarily.
    std::size_t needed_blocks = (total_words + threads_per_block - 1) / threads_per_block;
    int blocks = static_cast<int>(needed_blocks < static_cast<std::size_t>(target_blocks) ? needed_blocks : static_cast<std::size_t>(target_blocks));
    if (blocks < 1) blocks = 1;

    gol_step_kernel<<<blocks, threads_per_block>>>(input, output, words_per_row, grid_dimensions, wshift);
}