#include <cuda_runtime.h>
#include <cstdint>

/*
High-performance CUDA implementation of one step of Conway’s Game of Life
on a bit-packed grid. Each 64-bit word encodes 64 horizontally adjacent
cells (bit 0 = leftmost cell of the word, bit 63 = rightmost cell).

Key points:
- Every CUDA thread processes exactly one 64-bit word.
- Neighbor counts are computed with bit-sliced carry-save adders (CSAs),
  avoiding cross-bit carries and enabling parallel per-bit arithmetic.
- Boundary handling: all cells outside the grid are dead. Threads at the
  boundaries explicitly zero out missing neighbors. Interior threads use
  a fast path with no boundary checks.
- No shared or texture memory is used; global memory accesses are coalesced.
*/

// Type alias for readability
using u64 = std::uint64_t;

// Carry-Save Adder (CSA) for three 1-bit-per-lane inputs.
// Produces two 1-bit-per-lane outputs: sum (ones place) and carry (twos place).
// For each bit position i:
//   sum[i]   = a[i] XOR b[i] XOR c[i]  (bitwise sum modulo 2)
//   carry[i] = majority(a[i], b[i], c[i]) of pairs -> (a&b) | (b&c) | (a&c)
// Both outputs are independent per bit and do not carry across bit positions.
static __device__ __forceinline__ void csa3(u64 a, u64 b, u64 c, u64 &sum, u64 &carry) {
    sum   = a ^ b ^ c;
    carry = (a & b) | (b & c) | (a & c);
}

// Compute west (left) neighbor mask given current word and its left-adjacent word.
// For each cell bit position b in the current word, this returns whether its west neighbor
// is alive. It aligns the neighbor to the current cell's bit position.
// Handles cross-word bit transfer at bit 0 via (left >> 63).
static __device__ __forceinline__ u64 west_mask(u64 cur, u64 left) {
    return (cur << 1) | (left >> 63);
}

// Compute east (right) neighbor mask given current word and its right-adjacent word.
// For each cell bit position b in the current word, this returns whether its east neighbor
// is alive. It aligns the neighbor to the current cell's bit position.
// Handles cross-word bit transfer at bit 63 via (right << 63).
static __device__ __forceinline__ u64 east_mask(u64 cur, u64 right) {
    return (cur >> 1) | (right << 63);
}

static __global__ void game_of_life_kernel(const u64* __restrict__ input,
                                           u64* __restrict__ output,
                                           int words_per_row,
                                           int rows) {
    // 2D grid: y indexes row, x indexes word within the row
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= words_per_row) return;

    const size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(words_per_row);
    const size_t idx      = row_base + static_cast<size_t>(col);

    // Fast path for interior words (no boundary checks required).
    // This handles the vast majority of threads when grid_dimensions >> 1.
    if (row > 0 && row + 1 < rows && col > 0 && col + 1 < words_per_row) {
        const size_t up_base   = row_base - static_cast<size_t>(words_per_row);
        const size_t down_base = row_base + static_cast<size_t>(words_per_row);

        const u64 L   = input[idx - 1];
        const u64 C   = input[idx];
        const u64 R   = input[idx + 1];

        const u64 UpL = input[up_base + col - 1];
        const u64 Up  = input[up_base + col];
        const u64 UpR = input[up_base + col + 1];

        const u64 DnL = input[down_base + col - 1];
        const u64 Dn  = input[down_base + col];
        const u64 DnR = input[down_base + col + 1];

        // Build the eight neighbor masks aligned to current word bit positions.
        const u64 NW = west_mask(Up, UpL);
        const u64 N  = Up;
        const u64 NE = east_mask(Up, UpR);

        const u64 W  = west_mask(C, L);
        const u64 E  = east_mask(C, R);

        const u64 SW = west_mask(Dn, DnL);
        const u64 S  = Dn;
        const u64 SE = east_mask(Dn, DnR);

        // Carry-save adder tree to sum eight 1-bit masks per bit position:
        // First level: group into three CSAs (3:2 reduction).
        u64 sA, cA; csa3(NW, N,  NE, sA, cA);
        u64 sB, cB; csa3(W,  E,  SW, sB, cB);
        u64 sC, cC; csa3(S,  SE, 0ULL, sC, cC);

        // Second level: combine the ones (weight-1) from first level.
        u64 ones, carry2_from_ones; csa3(sA, sB, sC, ones, carry2_from_ones); // carry2_from_ones has weight 2

        // Combine all weight-2 terms: cA, cB, cC, and carry2_from_ones.
        u64 twos_partial, fours_from_three; csa3(cA, cB, cC, twos_partial, fours_from_three); // twos_partial weight 2, fours_from_three weight 4
        const u64 twos = twos_partial ^ carry2_from_ones;   // add the last weight-2 input
        const u64 extra_fours = twos_partial & carry2_from_ones; // carry to weight 4 from the last addition

        // Combine weight-4 terms; may produce weight-8 carries.
        const u64 fours  = fours_from_three ^ extra_fours;
        const u64 eights = fours_from_three & extra_fours;

        // Apply Game of Life rules:
        // next = (neighbors == 3) | (current & (neighbors == 2))
        const u64 eq3 = (ones & twos) & ~fours & ~eights;    // 3 = 1+2
        const u64 eq2 = (~ones & twos) & ~fours & ~eights;   // 2 = 2 only

        output[idx] = eq3 | (C & eq2);
        return;
    }

    // Boundary path (handles any missing neighbors by zeroing them).
    const bool hasUp    = (row > 0);
    const bool hasDown  = (row + 1 < rows);
    const bool hasLeft  = (col > 0);
    const bool hasRight = (col + 1 < words_per_row);

    const size_t up_base   = hasUp   ? (row_base - static_cast<size_t>(words_per_row)) : 0;
    const size_t down_base = hasDown ? (row_base + static_cast<size_t>(words_per_row)) : 0;

    const u64 L   = hasLeft  ? input[idx - 1] : 0ULL;
    const u64 C   = input[idx];
    const u64 R   = hasRight ? input[idx + 1] : 0ULL;

    const u64 UpL = (hasUp && hasLeft)   ? input[up_base + col - 1] : 0ULL;
    const u64 Up  = hasUp                ? input[up_base + col]     : 0ULL;
    const u64 UpR = (hasUp && hasRight)  ? input[up_base + col + 1] : 0ULL;

    const u64 DnL = (hasDown && hasLeft)  ? input[down_base + col - 1] : 0ULL;
    const u64 Dn  = hasDown               ? input[down_base + col]     : 0ULL;
    const u64 DnR = (hasDown && hasRight) ? input[down_base + col + 1] : 0ULL;

    const u64 NW = west_mask(Up, UpL);
    const u64 N  = Up;
    const u64 NE = east_mask(Up, UpR);

    const u64 W  = west_mask(C, L);
    const u64 E  = east_mask(C, R);

    const u64 SW = west_mask(Dn, DnL);
    const u64 S  = Dn;
    const u64 SE = east_mask(Dn, DnR);

    u64 sA, cA; csa3(NW, N,  NE, sA, cA);
    u64 sB, cB; csa3(W,  E,  SW, sB, cB);
    u64 sC, cC; csa3(S,  SE, 0ULL, sC, cC);

    u64 ones, carry2_from_ones; csa3(sA, sB, sC, ones, carry2_from_ones);

    u64 twos_partial, fours_from_three; csa3(cA, cB, cC, twos_partial, fours_from_three);
    const u64 twos = twos_partial ^ carry2_from_ones;
    const u64 extra_fours = twos_partial & carry2_from_ones;

    const u64 fours  = fours_from_three ^ extra_fours;
    const u64 eights = fours_from_three & extra_fours;

    const u64 eq3 = (ones & twos) & ~fours & ~eights;
    const u64 eq2 = (~ones & twos) & ~fours & ~eights;

    output[idx] = eq3 | (C & eq2);
}

// Launch function to compute one step of Conway’s Game of Life.
// - input:  bit-packed input grid (device pointer), one 64-bit word per 64 consecutive cells horizontally.
// - output: bit-packed output grid (device pointer), same layout as input.
// - grid_dimensions: width and height of the square grid; guaranteed power of two, >512, and fits in device memory.
// Assumptions:
// - input/output are allocated by cudaMalloc with sufficient size: (grid_dimensions/64) * grid_dimensions 64-bit words.
// - The caller performs any needed synchronization after this function returns.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int rows = grid_dimensions;
    const int words_per_row = grid_dimensions >> 6; // divide by 64

    // Choose a block size that balances occupancy and instruction throughput.
    // 256 threads per block is a good default for A100/H100 on this workload.
    const dim3 block(256, 1, 1);
    const dim3 grid((words_per_row + block.x - 1) / block.x, rows, 1);

    game_of_life_kernel<<<grid, block>>>(input, output, words_per_row, rows);
}