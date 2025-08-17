#include <cuda_runtime.h>
#include <cstdint>

/*
High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

Key design decisions:
- Bit-packed layout: each 64-bit word encodes a run of 64 horizontally adjacent cells in a row (LSB = leftmost bit).
- One CUDA thread processes exactly one 64-bit word, avoiding atomics and enabling fully coalesced memory access.
- No shared or texture memory: modern H100/A100 GPUs are bandwidth rich; coalesced global loads are efficient here.
- Avoid per-cell loops: use a bit-sliced adder network (carry-save adders) to count neighbors for all 64 cells in parallel.
  This computes 3 bitplanes (1,2,4) of the 0..8 neighbor count per bit without cross-bit carries.
- Boundary handling: outside-the-grid cells are treated as dead. For horizontal neighbors, handle cross-word bit carries
  explicitly by injecting the MSB/LSB from left/right words; for top/bottom rows, use zero.
- Rule application: next = (neighbors == 3) | (alive & neighbors == 2)

Notes on the adder network (per bit position, across 64 bits in parallel):
- From 8 one-bit masks (N, S, E, W, NE, NW, SE, SW), we compute three binary digit planes of the neighbor count:
  n1 (1's bit), n2 (2's bit), n4 (4's bit). The 8's bit is not needed for rules (==2 or ==3).
- Using carry-save adders (CSA):
    Let CSA(s, c, a, b, d) produce:
      s = a ^ b ^ d
      c = (a & b) | (a & d) | (b & d)   // per-bit "majority" (carry)
    Level 1:
      CSA(s10,c10, N, S, E)
      CSA(s11,c11, W, NE, NW)
      CSA(s12,c12, SE, SW, 0)
    Level 2:
      CSA(s20,c20, s10, s11, s12)   // sums of 1's
      CSA(s21,c21, c10, c11, c12)   // sums of 2's
    Final digit planes:
      n1 = s20
      n2 = s21 ^ c20
      n4 = c21 ^ (s21 & c20)
- Then:
    mask2 = (n2 & ~n1 & ~n4)   // exactly 2 neighbors
    mask3 = (n2 &  n1 & ~n4)   // exactly 3 neighbors
    next  = mask3 | (alive & mask2)

Performance considerations:
- Each thread performs 9 coalesced 64-bit loads (current, left/right; up row left/center/right; down row left/center/right),
  ~8 shifts, ~8 ors, and a fixed sequence of bitwise ops for the CSA tree. This is compute-bound and scales well.
- The use of __popc/__popcll for per-cell counting is intentionally avoided; the CSA-based bit-sliced reduction is faster
  and minimizes branch divergence and per-bit work.

API:
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions);

Assumptions:
- grid_dimensions is a power of two, >= 512.
- grid is square of size (grid_dimensions x grid_dimensions).
- Each row contains (grid_dimensions / 64) 64-bit words.
- Input and output device buffers are allocated with cudaMalloc.
- Synchronization is performed by the caller.
*/

static __device__ __forceinline__ void csa(uint64_t& s, uint64_t& c,
                                           const uint64_t a,
                                           const uint64_t b,
                                           const uint64_t d) {
    // Carry-Save Adder: per-bit full adder across 64 lanes in parallel
    const uint64_t ab = a ^ b;
    s = ab ^ d;
    c = (a & b) | (d & ab);
}

static __device__ __forceinline__ uint64_t shift_east(const uint64_t center, const uint64_t right) {
    // For each target bit i, set bit i if the cell at (i+1) (east neighbor) is alive.
    // Achieved by shifting the source row right by 1 and injecting the LSB of 'right' into MSB.
    return (center >> 1) | (right << 63); // right<<63 moves bit0 of 'right' into bit63.
}

static __device__ __forceinline__ uint64_t shift_west(const uint64_t center, const uint64_t left) {
    // For each target bit i, set bit i if the cell at (i-1) (west neighbor) is alive.
    // Achieved by shifting the source row left by 1 and injecting the MSB of 'left' into LSB.
    return (center << 1) | (left >> 63); // left>>63 moves bit63 of 'left' into bit0.
}

__global__ void life_kernel_bitpacked(const std::uint64_t* __restrict__ in,
                                      std::uint64_t* __restrict__ out,
                                      int grid_dim) {
    const int words_per_row = grid_dim >> 6; // grid_dim / 64; grid_dim is a power of two
    const int total_words = words_per_row * grid_dim;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;

    // Compute 2D position in word coordinates
    const int y = idx / words_per_row;
    const int x = idx - y * words_per_row;

    // Row base pointers for current, above, and below rows
    const std::uint64_t* rowC = in + static_cast<std::size_t>(y) * words_per_row;
    const std::uint64_t* rowU = (y > 0) ? (in + static_cast<std::size_t>(y - 1) * words_per_row) : nullptr;
    const std::uint64_t* rowD = (y + 1 < grid_dim) ? (in + static_cast<std::size_t>(y + 1) * words_per_row) : nullptr;

    // Load current row words (center, left, right)
    const bool hasL = (x > 0);
    const bool hasR = (x + 1 < words_per_row);

    const uint64_t mC = rowC[x];
    const uint64_t mL = hasL ? rowC[x - 1] : 0ull;
    const uint64_t mR = hasR ? rowC[x + 1] : 0ull;

    // Load above row words (center, left, right), or zero at grid boundary
    const uint64_t uC = rowU ? rowU[x] : 0ull;
    const uint64_t uL = (rowU && hasL) ? rowU[x - 1] : 0ull;
    const uint64_t uR = (rowU && hasR) ? rowU[x + 1] : 0ull;

    // Load below row words (center, left, right), or zero at grid boundary
    const uint64_t dC = rowD ? rowD[x] : 0ull;
    const uint64_t dL = (rowD && hasL) ? rowD[x - 1] : 0ull;
    const uint64_t dR = (rowD && hasR) ? rowD[x + 1] : 0ull;

    // Build neighbor masks for the 8 directions (per-bit neighbor presence)
    const uint64_t N  = uC;
    const uint64_t S  = dC;
    const uint64_t E  = shift_east(mC, mR);
    const uint64_t W  = shift_west(mC, mL);
    const uint64_t NE = shift_east(uC, uR);
    const uint64_t NW = shift_west(uC, uL);
    const uint64_t SE = shift_east(dC, dR);
    const uint64_t SW = shift_west(dC, dL);

    // Carry-save adder tree to compute digit planes (n1,n2,n4) of the 0..8 neighbor count per bit
    uint64_t s10, c10;
    uint64_t s11, c11;
    uint64_t s12, c12;

    csa(s10, c10, N,  S,  E);
    csa(s11, c11, W,  NE, NW);
    csa(s12, c12, SE, SW, 0ull);

    uint64_t s20, c20;
    uint64_t s21, c21;

    csa(s20, c20, s10, s11, s12);   // sums of 1's (LSB plane and carry->2's)
    csa(s21, c21, c10, c11, c12);   // sums of 2's (LSB plane and carry->4's)

    // Neighbor count digit planes:
    // n1 = s20
    // n2 = s21 ^ c20
    // n4 = c21 ^ (s21 & c20)
    const uint64_t n1 = s20;
    const uint64_t n2 = s21 ^ c20;
    const uint64_t n4 = c21 ^ (s21 & c20);

    // Masks for exactly 2 and exactly 3 neighbors (no need to handle 8 specifically; n2=0 there)
    const uint64_t mask2 = (n2 & ~n1 & ~n4); // 010
    const uint64_t mask3 = (n2 &  n1 & ~n4); // 011

    // Current cell state
    const uint64_t alive = mC;

    // Next state: birth on 3; survive on 2
    const uint64_t next = mask3 | (alive & mask2);

    out[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    /*
    Executes one generation step of Conway's Game of Life on a square, bit-packed grid.

    Parameters:
    - input:  device pointer to the current grid (bit-packed: 1 bit per cell, 64 cells per uint64_t)
    - output: device pointer to store the next generation (same packing)
    - grid_dimensions: width and height of the square grid (power of two, >= 512)

    Notes:
    - Each CUDA thread computes the next state for exactly one 64-bit word (64 adjacent horizontal cells).
    - No synchronization is performed here; the caller must handle it if needed.
    */
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words = words_per_row * grid_dimensions;

    // Choose a launch configuration that balances occupancy and register usage.
    // 256 threads per block works well on modern data center GPUs.
    constexpr int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

    life_kernel_bitpacked<<<blocks, threads_per_block>>>(input, output, grid_dimensions);
}