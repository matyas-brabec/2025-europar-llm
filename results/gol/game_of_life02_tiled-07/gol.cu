#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA-optimized single-step Conway's Game of Life for a bit-packed grid.

  Encoding and layout:
  - The grid is square of size grid_dimensions x grid_dimensions cells.
  - Cells are bit-packed into 8x8 tiles; each std::uint64_t encodes one 8x8 tile.
  - Tiles are laid out in row-major order: tile_index = tile_y * tiles_per_row + tile_x,
    where tiles_per_row = grid_dimensions / 8.
  - Within each 64-bit word (tile), bits are arranged as 8 bytes, each byte is a tile row.
    - Byte 0 (least-significant byte) is the top row of the tile (row 0).
    - Within each byte, bit 0 (LSB) is column 0 (leftmost), bit 7 (MSB) is column 7 (rightmost).
    - Therefore, the bit index for tile-local coordinates (r, c) is (r * 8 + c).
  - All cells outside the grid are considered dead (0). Boundary tiles use zero-filled neighbors.

  Algorithm:
  - Each CUDA thread processes one 8x8 tile.
  - For the target tile C and its 8 neighbors (W, E, N, S, NW, NE, SW, SE), we compute
    neighbor counts using bit-parallel operations:
      * Horizontal neighbor fields within each row (west/east), with cross-tile carry handled.
      * Vertical alignment across tiles by shifting in whole bytes (8-bit rows).
      * Use bitwise half-/full-adders to sum 1-bit fields into multi-bit per-cell counts
        without cross-cell carry propagation.
  - The rules applied per cell:
      * A live cell with 2 or 3 live neighbors survives.
      * A dead cell with exactly 3 live neighbors becomes alive.
    Implemented via masks for (neighbors == 2) and (neighbors == 3).

  Notes:
  - Shared/texture memory is not used by design; all work is done with registers and global loads.
  - The kernel expects grid_dimensions to be a power of two and >= 512, but only requires
    that it is divisible by 8 (so tiles_per_row is integer).
*/

static __device__ __forceinline__ std::uint64_t vshift_north(std::uint64_t mid, std::uint64_t north) {
    // Align "row above" each cell into the same row position:
    // y0 = north.b7, y1..y7 = mid.b0..b6
    return (mid << 8) | (north >> 56);
}

static __device__ __forceinline__ std::uint64_t vshift_south(std::uint64_t mid, std::uint64_t south) {
    // Align "row below" each cell into the same row position:
    // y0..y6 = mid.b1..b7, y7 = south.b0
    return (mid >> 8) | (south << 56);
}

static __device__ __forceinline__ std::uint64_t west_field(std::uint64_t mid, std::uint64_t left) {
    // For each cell, produce the bit of the west neighbor in the same row.
    // - Shift right by 1 within each byte, zeroing MSB (bit 7) per byte to prevent cross-row leakage.
    // - For column 0 in each row, bring in the MSB (bit 7) of the left tile's corresponding row.
    const std::uint64_t BYTE_MSB_MASK = 0x8080808080808080ull; // MSB of each byte
    const std::uint64_t BYTE_KEEP_MASK_R = 0x7f7f7f7f7f7f7f7full; // per byte: 01111111
    std::uint64_t intra = (mid >> 1) & BYTE_KEEP_MASK_R;
    std::uint64_t carry_in = (left & BYTE_MSB_MASK) >> 7;       // map bit7->bit0 per byte
    return intra | carry_in;
}

static __device__ __forceinline__ std::uint64_t east_field(std::uint64_t mid, std::uint64_t right) {
    // For each cell, produce the bit of the east neighbor in the same row.
    // - Shift left by 1 within each byte, zeroing LSB (bit 0) per byte to prevent cross-row leakage.
    // - For column 7 in each row, bring in the LSB (bit 0) of the right tile's corresponding row.
    const std::uint64_t BYTE_LSB_MASK = 0x0101010101010101ull; // LSB of each byte
    const std::uint64_t BYTE_KEEP_MASK_L = 0xfefefefefefefefeull; // per byte: 11111110
    std::uint64_t intra = (mid << 1) & BYTE_KEEP_MASK_L;
    std::uint64_t carry_in = (right & BYTE_LSB_MASK) << 7;      // map bit0->bit7 per byte
    return intra | carry_in;
}

// Sum of three 1-bit fields per cell -> two bitplanes (lo, hi) representing values 0..3.
static __device__ __forceinline__ void add3_bits(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                 std::uint64_t &lo, std::uint64_t &hi) {
    // lo = a ^ b ^ c
    // hi = (a & b) | (b & c) | (a & c)
    std::uint64_t ab_x = a ^ b;
    std::uint64_t ab_c = a & b;
    lo = ab_x ^ c;
    hi = ab_c | (ab_x & c);
}

// Sum of two 1-bit fields per cell -> two bitplanes (lo, hi) representing values 0..2.
static __device__ __forceinline__ void add2_bits(std::uint64_t a, std::uint64_t b,
                                                 std::uint64_t &lo, std::uint64_t &hi) {
    // lo = a ^ b
    // hi = a & b
    lo = a ^ b;
    hi = a & b;
}

// Add two 2-bit numbers per cell: (a0 + 2*a1) + (b0 + 2*b1) -> s0 + 2*s1 + 4*s2.
static __device__ __forceinline__ void add2bit_2bit(std::uint64_t a0, std::uint64_t a1,
                                                    std::uint64_t b0, std::uint64_t b1,
                                                    std::uint64_t &s0, std::uint64_t &s1, std::uint64_t &s2) {
    // Bit 0:
    s0 = a0 ^ b0;
    std::uint64_t c0 = a0 & b0; // carry into bit 1

    // Bit 1:
    std::uint64_t t1 = a1 ^ b1;
    std::uint64_t c1 = a1 & b1; // carry into bit 2 from MSB add
    s1 = t1 ^ c0;
    std::uint64_t c01 = t1 & c0;

    // Bit 2:
    s2 = c1 | c01;
}

// Add a 3-bit number per cell (s0 + 2*s1 + 4*s2) with a 2-bit number (t0 + 2*t1)
// -> r0 + 2*r1 + 4*r2 + 8*r3
static __device__ __forceinline__ void add3bit_2bit(std::uint64_t s0, std::uint64_t s1, std::uint64_t s2,
                                                    std::uint64_t t0, std::uint64_t t1,
                                                    std::uint64_t &r0, std::uint64_t &r1,
                                                    std::uint64_t &r2, std::uint64_t &r3) {
    // Bit 0:
    r0 = s0 ^ t0;
    std::uint64_t c0 = s0 & t0;

    // Bit 1:
    std::uint64_t t1sum = s1 ^ t1;
    std::uint64_t c1a = s1 & t1;
    r1 = t1sum ^ c0;
    std::uint64_t c1b = t1sum & c0;
    std::uint64_t c1 = c1a | c1b; // carry into bit 2

    // Bit 2:
    r2 = s2 ^ c1;
    r3 = s2 & c1; // carry into bit 3
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int tiles_per_dim)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // tile x
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // tile y
    if (tx >= tiles_per_dim || ty >= tiles_per_dim) return;

    int idx = ty * tiles_per_dim + tx;

    // Load center tile and its eight neighbors (zeros beyond boundary).
    // Using simple boundary checks; divergence occurs only at borders (small fraction).
    std::uint64_t C = input[idx];

    std::uint64_t W = (tx > 0) ? input[idx - 1] : 0ull;
    std::uint64_t E = (tx + 1 < tiles_per_dim) ? input[idx + 1] : 0ull;

    std::uint64_t N = (ty > 0) ? input[idx - tiles_per_dim] : 0ull;
    std::uint64_t S = (ty + 1 < tiles_per_dim) ? input[idx + tiles_per_dim] : 0ull;

    std::uint64_t NW = (tx > 0 && ty > 0) ? input[idx - tiles_per_dim - 1] : 0ull;
    std::uint64_t NE = (tx + 1 < tiles_per_dim && ty > 0) ? input[idx - tiles_per_dim + 1] : 0ull;
    std::uint64_t SW = (tx > 0 && ty + 1 < tiles_per_dim) ? input[idx + tiles_per_dim - 1] : 0ull;
    std::uint64_t SE = (tx + 1 < tiles_per_dim && ty + 1 < tiles_per_dim) ? input[idx + tiles_per_dim + 1] : 0ull;

    // Vertical alignment for north and south bands.
    std::uint64_t nL = vshift_north(W, NW);
    std::uint64_t nM = vshift_north(C, N);
    std::uint64_t nR = vshift_north(E, NE);

    std::uint64_t sL = vshift_south(W, SW);
    std::uint64_t sM = vshift_south(C, S);
    std::uint64_t sR = vshift_south(E, SE);

    // Horizontal neighbor fields for each band and their per-cell sums.

    // North band: west + center + east
    std::uint64_t nW = west_field(nM, nL);
    std::uint64_t nE = east_field(nM, nR);
    std::uint64_t n_lo, n_hi;
    add3_bits(nW, nM, nE, n_lo, n_hi); // values 0..3

    // Same row band: only west + east (exclude center to avoid counting the cell itself)
    std::uint64_t mW = west_field(C, W);
    std::uint64_t mE = east_field(C, E);
    std::uint64_t m_lo, m_hi;
    add2_bits(mW, mE, m_lo, m_hi); // values 0..2

    // South band: west + center + east
    std::uint64_t sW = west_field(sM, sL);
    std::uint64_t sE = east_field(sM, sR);
    std::uint64_t s_lo, s_hi;
    add3_bits(sW, sM, sE, s_lo, s_hi); // values 0..3

    // Sum bands: (north 2-bit) + (middle 2-bit) -> 3-bit
    std::uint64_t nm0, nm1, nm2;
    add2bit_2bit(n_lo, n_hi, m_lo, m_hi, nm0, nm1, nm2);

    // Then add (south 2-bit) -> 4-bit total neighbor count per cell (0..8)
    std::uint64_t f0, f1, f2, f3;
    add3bit_2bit(nm0, nm1, nm2, s_lo, s_hi, f0, f1, f2, f3);

    // Compute masks for neighbors == 3 and neighbors == 2
    std::uint64_t not_f3 = ~f3;
    std::uint64_t not_f2 = ~f2;

    // neighbors == 3 -> (~f3) & (~f2) & f1 & f0
    std::uint64_t eq3 = not_f3 & not_f2 & f1 & f0;

    // neighbors == 2 -> (~f3) & (~f2) & f1 & (~f0)
    std::uint64_t eq2 = not_f3 & not_f2 & f1 & (~f0);

    // Next state: alive if (neighbors == 3) OR (alive AND neighbors == 2)
    std::uint64_t next = eq3 | (C & eq2);

    output[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is divisible by 8 and sufficiently large; tiles_per_dim fits into int.
    int tiles_per_dim = grid_dimensions >> 3;

    // Choose a 2D block size that maps well to warp scheduling and favors coalesced accesses.
    // 32x8 = 256 threads per block; threads advance in x for coalesced 64-bit loads.
    dim3 block(32, 8, 1);
    dim3 grid((tiles_per_dim + block.x - 1) / block.x,
              (tiles_per_dim + block.y - 1) / block.y,
              1);

    gol_step_kernel<<<grid, block>>>(input, output, tiles_per_dim);
}