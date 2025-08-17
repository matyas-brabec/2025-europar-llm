#include <cuda_runtime.h>
#include <cstdint>

/*
    CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

    - Grid is square: width = height = grid_dimensions (power of 2, >512).
    - Each 64-bit word encodes 64 consecutive cells in a row; one bit per cell (LSB = leftmost cell in the word).
    - One CUDA thread processes one 64-bit word (eliminates atomics).
    - Outside-the-grid cells are treated as dead (0).
    - The kernel uses only global memory and bitwise arithmetic; no shared/texture memory.

    Bit mapping and neighbor masks:
    - For a row word wC and its immediate left/right words wL, wR:
        East (x+1):   E  = (wC >> 1) | (wR << 63)  // bring next word's bit0 into bit63
        West (x-1):   W  = (wC << 1) | (wL >> 63)  // bring prev word's bit63 into bit0
    - For row above and below (u*, d*):
        North (y-1):  N  = uC
        South (y+1):  S  = dC
        North-East:   NE = (uC >> 1) | (uR << 63)
        North-West:   NW = (uC << 1) | (uL >> 63)
        South-East:   SE = (dC >> 1) | (dR << 63)
        South-West:   SW = (dC << 1) | (dL >> 63)

    Neighbor count accumulation:
    - We build the neighbor count per bit using a bit-sliced adder tree:
        - Pairwise add 8 one-bit masks -> 4 two-bit numbers (low/2's planes).
        - Add those -> 2 numbers with planes for 1,2,4.
        - Add final pair -> planes for 1,2,4,8.
    - No carries propagate across bit positions; all operations are bitwise and lane-wise.

    Rule application:
    - next = (count == 3) | (current & (count == 2))
    - With bit planes (b1,b2,b4,b8) representing the neighbor count in binary (1,2,4,8 positions):
        eq3 =  b1 &  b2 & ~b4 & ~b8
        eq2 = ~b1 &  b2 & ~b4 & ~b8
        next = eq3 | (current & eq2)
*/

static __device__ __forceinline__ void sum8_bitmasks(
    std::uint64_t m0, std::uint64_t m1, std::uint64_t m2, std::uint64_t m3,
    std::uint64_t m4, std::uint64_t m5, std::uint64_t m6, std::uint64_t m7,
    std::uint64_t &b1, std::uint64_t &b2, std::uint64_t &b4, std::uint64_t &b8)
{
    // Pairwise add (two 1-bit numbers -> 2-bit result: low bit (1's) + carry (2's))
    std::uint64_t l01 = m0 ^ m1;
    std::uint64_t h01 = m0 & m1;

    std::uint64_t l23 = m2 ^ m3;
    std::uint64_t h23 = m2 & m3;

    std::uint64_t l45 = m4 ^ m5;
    std::uint64_t h45 = m4 & m5;

    std::uint64_t l67 = m6 ^ m7;
    std::uint64_t h67 = m6 & m7;

    // Add pairs to get sums over 4 inputs
    // Sum of low bits (1's place)
    std::uint64_t t0   = l01 & l23;           // carry into 2's place from 1's addition
    std::uint64_t l0123 = l01 ^ l23;

    // 2's place: sum of three 1-bit numbers: h01 + h23 + t0
    std::uint64_t x0   = h01 ^ h23;
    std::uint64_t m0123 = x0 ^ t0;            // 2's bit (parity of three inputs)
    std::uint64_t h0123 = (h01 & h23) | (t0 & x0); // carry to 4's place from 2's addition

    std::uint64_t t1   = l45 & l67;
    std::uint64_t l4567 = l45 ^ l67;

    std::uint64_t x1   = h45 ^ h67;
    std::uint64_t m4567 = x1 ^ t1;
    std::uint64_t h4567 = (h45 & h67) | (t1 & x1);

    // Add the two 4-wide sums to get 8-wide sum
    // 1's place
    std::uint64_t c1   = l0123 & l4567;
    b1 = l0123 ^ l4567;

    // 2's place: sum m0123 + m4567 + c1
    std::uint64_t x2   = m0123 ^ m4567;
    b2 = x2 ^ c1;
    std::uint64_t c2   = (m0123 & m4567) | (c1 & x2);

    // 4's place: sum h0123 + h4567 + c2
    std::uint64_t x3   = h0123 ^ h4567;
    b4 = x3 ^ c2;
    b8 = (h0123 & h4567) | (c2 & x3);
}

static __device__ __forceinline__ std::uint64_t compute_next_word(
    std::uint64_t wL, std::uint64_t wC, std::uint64_t wR,
    std::uint64_t uL, std::uint64_t uC, std::uint64_t uR,
    std::uint64_t dL, std::uint64_t dC, std::uint64_t dR)
{
    // Build neighbor masks for the 8 adjacent directions.
    // Horizontal neighbors within the same row:
    const std::uint64_t W  = (wC << 1) | (wL >> 63);
    const std::uint64_t E  = (wC >> 1) | (wR << 63);

    // Vertical and diagonal neighbors:
    const std::uint64_t N  = uC;
    const std::uint64_t S  = dC;
    const std::uint64_t NW = (uC << 1) | (uL >> 63);
    const std::uint64_t NE = (uC >> 1) | (uR << 63);
    const std::uint64_t SW = (dC << 1) | (dL >> 63);
    const std::uint64_t SE = (dC >> 1) | (dR << 63);

    // Sum the eight neighbor masks using bit-sliced arithmetic.
    std::uint64_t b1, b2, b4, b8;
    sum8_bitmasks(NW, N, NE, W, E, SW, S, SE, b1, b2, b4, b8);

    // Apply Game of Life rules:
    // - survive if alive and count == 2
    // - born if count == 3
    const std::uint64_t not_b4 = ~b4;
    const std::uint64_t not_b8 = ~b8;

    const std::uint64_t eq3 = (b1 & b2) & not_b4 & not_b8;          // count == 3
    const std::uint64_t eq2 = (~b1 & b2) & not_b4 & not_b8;         // count == 2

    const std::uint64_t next = eq3 | (wC & eq2);
    return next;
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int grid_dim_cells)
{
    // Compute indices
    const int words_per_row = grid_dim_cells >> 6;     // grid_dim_cells / 64
    const int height_rows   = grid_dim_cells;          // number of rows
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(height_rows);

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    const int row = static_cast<int>(idx / words_per_row);
    const int col = static_cast<int>(idx - static_cast<std::size_t>(row) * words_per_row);

    // Load center word
    const std::uint64_t wC = input[idx];

    // Neighbor word indices conditions (treat out-of-bounds as 0)
    const bool hasLeft  = (col > 0);
    const bool hasRight = (col + 1 < words_per_row);
    const bool hasUp    = (row > 0);
    const bool hasDown  = (row + 1 < height_rows);

    // Same row neighbors
    const std::uint64_t wL = hasLeft  ? input[idx - 1] : 0ull;
    const std::uint64_t wR = hasRight ? input[idx + 1] : 0ull;

    // Above row neighbors
    const std::size_t up_idx = hasUp ? (idx - static_cast<std::size_t>(words_per_row)) : idx; // safe base
    const std::uint64_t uC = hasUp ? input[up_idx] : 0ull;
    const std::uint64_t uL = (hasUp && hasLeft)  ? input[up_idx - 1] : 0ull;
    const std::uint64_t uR = (hasUp && hasRight) ? input[up_idx + 1] : 0ull;

    // Below row neighbors
    const std::size_t dn_idx = hasDown ? (idx + static_cast<std::size_t>(words_per_row)) : idx; // safe base
    const std::uint64_t dC = hasDown ? input[dn_idx] : 0ull;
    const std::uint64_t dL = (hasDown && hasLeft)  ? input[dn_idx - 1] : 0ull;
    const std::uint64_t dR = (hasDown && hasRight) ? input[dn_idx + 1] : 0ull;

    // Compute and write next state for this 64-cell block
    output[idx] = compute_next_word(wL, wC, wR, uL, uC, uR, dL, dC, dR);
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Launch one thread per 64-bit word in the grid.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Tuned block size for modern NVIDIA data center GPUs (A100/H100).
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    gol_step_kernel<<<blocks, threads_per_block>>>(input, output, grid_dimensions);

    // Synchronization is handled by the caller as per problem statement.
}