#include <cuda_runtime.h>
#include <cstdint>

/*
High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

Key details:
- The grid is square with dimensions N x N (N is a power of 2 > 512). Each row is bit-packed into 64-bit words.
- Each 64-bit word encodes 64 consecutive cells in the same row; bit 0 is the least significant bit.
- Each CUDA thread processes exactly one 64-bit word (64 cells), avoiding atomics.
- Neighbor counting is fully bit-parallel: 64 cells are processed simultaneously using simple bitwise operations.
- The eight neighbor masks (N, S, E, W, NE, NW, SE, SW) are built with proper cross-word injection for bit 0 and bit 63.
- Outside-the-grid cells are zero (dead). Boundary handling is done via guarded loads and zero injection.
- Neighbor counts (0..8) are accumulated using a small carry-save adder (CSA) tree built from 3:2 "full-adder" logic.
  This produces bit-planes for the count (ones, twos, fours, eights). The Game of Life rule only requires checking
  counts equal to 2 or 3, which depend on ones/twos/fours only; eights can be computed but is not needed for the rule.
- Next state: next = (neighbors == 3) | (alive & (neighbors == 2)).

No shared or texture memory is used because the bit-packed representation and coalesced loads suffice on modern GPUs.

Thread mapping:
- words_per_row = N / 64, height = N
- thread (x, y) handles word index x of row y
*/

static __device__ __forceinline__ void add3_u64(const std::uint64_t a,
                                                const std::uint64_t b,
                                                const std::uint64_t c,
                                                std::uint64_t& sum,
                                                std::uint64_t& carry)
{
    // 3:2 carry-save adder across 64 independent bit positions:
    // sum   = a ^ b ^ c
    // carry = majority(a, b, c) = (a & b) | (a & c) | (b & c)
    // Use a slightly optimized form for carry to save an AND:
    const std::uint64_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

static __global__ void life_step_kernel(const std::uint64_t* __restrict__ in,
                                        std::uint64_t* __restrict__ out,
                                        int words_per_row,
                                        int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // word index within row
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x >= words_per_row || y >= height) return;

    const int idx = y * words_per_row + x;

    // Flags for boundary handling
    const bool has_left  = (x > 0);
    const bool has_right = (x + 1 < words_per_row);
    const bool has_up    = (y > 0);
    const bool has_down  = (y + 1 < height);

    // Load current row words
    const std::uint64_t sc = in[idx]; // same-row center
    const std::uint64_t sl = has_left  ? in[idx - 1] : 0; // same-row left neighbor word
    const std::uint64_t sr = has_right ? in[idx + 1] : 0; // same-row right neighbor word

    // Load above row words (or 0 if outside grid)
    const int idx_up = idx - words_per_row;
    const std::uint64_t ac = has_up    ? in[idx_up]                  : 0; // above center
    const std::uint64_t al = (has_up && has_left)  ? in[idx_up - 1]  : 0; // above left
    const std::uint64_t ar = (has_up && has_right) ? in[idx_up + 1]  : 0; // above right

    // Load below row words (or 0 if outside grid)
    const int idx_dn = idx + words_per_row;
    const std::uint64_t bc = has_down  ? in[idx_dn]                  : 0; // below center
    const std::uint64_t bl = (has_down && has_left)  ? in[idx_dn - 1]: 0; // below left
    const std::uint64_t br = (has_down && has_right) ? in[idx_dn + 1]: 0; // below right

    // Build 8 neighbor masks aligned to this word's 64 cells.
    // For shifts, inject boundary bits from adjacent words so the 0th and 63rd bits get correct neighbors.
    // - For "west" (left neighbor), shift left (i-1) and inject the MSB from the left word into bit 0.
    // - For "east" (right neighbor), shift right (i+1) and inject the LSB from the right word into bit 63.
    const std::uint64_t west = (sc << 1) | (sl >> 63);
    const std::uint64_t east = (sc >> 1) | (sr << 63);

    // Above row: include N, NW, NE
    const std::uint64_t north = ac;
    const std::uint64_t nw    = (ac << 1) | (al >> 63);
    const std::uint64_t ne    = (ac >> 1) | (ar << 63);

    // Below row: include S, SW, SE
    const std::uint64_t south = bc;
    const std::uint64_t sw    = (bc << 1) | (bl >> 63);
    const std::uint64_t se    = (bc >> 1) | (br << 63);

    // Sum the 8 neighbor masks with a small CSA tree.
    // Stage 1: three groups of up to three inputs -> (sum, carry) pairs
    std::uint64_t s0, c0; add3_u64(nw, north, ne, s0, c0);      // weight-1 sum s0, weight-2 carry c0
    std::uint64_t s1, c1; add3_u64(west, east, south, s1, c1);  // weight-1 sum s1, weight-2 carry c1
    std::uint64_t s2, c2; add3_u64(sw, se, 0ull, s2, c2);       // weight-1 sum s2, weight-2 carry c2

    // Stage 2: combine weight-1 sums
    std::uint64_t s3, c3; add3_u64(s0, s1, s2, s3, c3);         // s3: weight-1 (ones), c3: weight-2

    // Stage 3: combine weight-2 carries
    std::uint64_t t0, d0; add3_u64(c0, c1, c2, t0, d0);         // t0: weight-2, d0: weight-4
    std::uint64_t t1, d1; add3_u64(t0, c3, 0ull, t1, d1);       // t1: weight-2 (twos), d1: weight-4

    // Final planes: ones, twos, fours, eights
    const std::uint64_t ones   = s3;                 // weight 1
    const std::uint64_t twos   = t1;                 // weight 2
    const std::uint64_t fours  = d0 ^ d1;            // add two weight-4 planes
    // const std::uint64_t eights = d0 & d1;         // weight 8 (not needed for rule checks)

    // Game of Life rule:
    // next = (neighbors == 3) | (alive & (neighbors == 2))
    // neighbors == 3 <=> ones=1, twos=1, fours=0
    // neighbors == 2 <=> ones=0, twos=1, fours=0
    const std::uint64_t eq3 =  ones &  twos & ~fours;
    const std::uint64_t eq2 = ~ones &  twos & ~fours;
    const std::uint64_t next = eq3 | (sc & eq2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is N (width == height), and guaranteed to be a power of 2 and > 512.
    const int height = grid_dimensions;
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // Launch configuration:
    // - Use a 2D grid: X covers words in a row, Y covers rows.
    // - Favor blockDim.x for coalesced access along rows; choose 256 threads per block (tuned for A100/H100).
    const dim3 block(256, 1, 1);
    const dim3 grid((words_per_row + block.x - 1) / block.x,
                    (height       + block.y - 1) / block.y,
                    1);

    life_step_kernel<<<grid, block>>>(input, output, words_per_row, height);
}