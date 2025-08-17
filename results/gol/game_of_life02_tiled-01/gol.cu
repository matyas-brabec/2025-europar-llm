#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Encoding:
  - The global grid is square (grid_dimensions x grid_dimensions) and bit-packed into 8x8 tiles.
  - Each std::uint64_t word encodes a single 8x8 tile in row-major order within the tile:
      bit index = (row * 8 + col), where row, col in [0,7].
    - bit 0   corresponds to (row=0, col=0) of the tile
    - bit 7   corresponds to (row=0, col=7)
    - bit 8   corresponds to (row=1, col=0)
    - ...
    - bit 63  corresponds to (row=7, col=7)
  - The tiles themselves are laid out in row-major order in memory:
      tile_index = tile_y * tiles_per_side + tile_x, where tiles_per_side = grid_dimensions / 8.

  Rules:
  - Classic Conway's Game of Life with 8-neighborhood and dead boundary outside the grid.

  Implementation notes:
  - One thread processes one 8x8 tile (one 64-bit word).
  - No shared or texture memory is used; only global loads and bitwise arithmetic.
  - We load the 3x3 neighborhood of tiles around the current tile (clamping to zero out-of-bounds).
  - We build eight 64-bit bitboards for the eight neighbor directions (N, S, E, W, NE, NW, SE, SW)
    such that each bit in each direction bitboard contributes to the appropriate destination cell bit in the current tile.
    This involves intra-tile shifts and importing edge columns/rows from neighboring tiles.
  - We then compute the per-cell neighbor count using a carry-save adder (CSA) tree to generate bit-sliced sums:
      ones (bit0), twos (bit1), fours (bit2), eights (bit3).
    From these, we derive masks for exactly-2 and exactly-3 neighbors, and apply the Life rule:
      next = (neighbors == 3) | (alive & (neighbors == 2)).
  - The kernel writes the resulting 64-bit word to the output buffer at the same tile index.
*/

static __device__ __forceinline__ void csa(uint64_t a, uint64_t b, uint64_t c, uint64_t &sum, uint64_t &carry) {
    // Carry-save adder for bitboards:
    // sum   = a ^ b ^ c               (bitwise sum modulo 2)
    // carry = (a&b) | (a&c) | (b&c)   (carry bits to next binary digit)
    uint64_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

static __global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                           std::uint64_t* __restrict__ output,
                                           int grid_dimensions)
{
    // Compute tile coordinates for this thread
    const int tiles_per_side = grid_dimensions >> 3; // grid_dimensions / 8
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tiles_per_side || ty >= tiles_per_side) return;

    const int idx = ty * tiles_per_side + tx;

    // Boundary checks for neighbor tiles
    const bool has_w = (tx > 0);
    const bool has_e = (tx + 1 < tiles_per_side);
    const bool has_n = (ty > 0);
    const bool has_s = (ty + 1 < tiles_per_side);

    // Load center and neighbor tiles. Outside-grid tiles are treated as zero.
    const std::uint64_t c  = __ldg(&input[idx]);
    const std::uint64_t w  = has_w ? __ldg(&input[idx - 1]) : 0ull;
    const std::uint64_t e  = has_e ? __ldg(&input[idx + 1]) : 0ull;
    const std::uint64_t n  = has_n ? __ldg(&input[idx - tiles_per_side]) : 0ull;
    const std::uint64_t s  = has_s ? __ldg(&input[idx + tiles_per_side]) : 0ull;
    const std::uint64_t nw = (has_n && has_w) ? __ldg(&input[idx - tiles_per_side - 1]) : 0ull;
    const std::uint64_t ne = (has_n && has_e) ? __ldg(&input[idx - tiles_per_side + 1]) : 0ull;
    const std::uint64_t sw = (has_s && has_w) ? __ldg(&input[idx + tiles_per_side - 1]) : 0ull;
    const std::uint64_t se = (has_s && has_e) ? __ldg(&input[idx + tiles_per_side + 1]) : 0ull;

    // Column and row masks for 8x8 tile layout (row-major, 8-bit rows)
    constexpr std::uint64_t COL_0     = 0x0101010101010101ULL; // bit 0 of each byte
    constexpr std::uint64_t COL_7     = 0x8080808080808080ULL; // bit 7 of each byte
    constexpr std::uint64_t ROW_0     = 0x00000000000000FFULL; // lowest 8 bits
    constexpr std::uint64_t ROW_7     = 0xFF00000000000000ULL; // highest 8 bits
    constexpr std::uint64_t NCOL_0    = ~COL_0;
    constexpr std::uint64_t NCOL_7    = ~COL_7;
    constexpr std::uint64_t NROW_0    = ~ROW_0;
    constexpr std::uint64_t NROW_7    = ~ROW_7;

    // Build neighbor contribution bitboards for the current tile.
    // Each bitboard has bits set at positions of cells that receive a neighbor from that direction.
    // Intra-tile shifts use masks to avoid wrap across row/col boundaries; cross-tile edges import the needed row/col.
    const std::uint64_t N  = ((c  & NROW_7) << 8)  | ((n  & ROW_7) >> 56);
    const std::uint64_t S  = ((c  & NROW_0) >> 8)  | ((s  & ROW_0) << 56);
    const std::uint64_t W  = ((c  & NCOL_7) << 1)  | ((w  & COL_7) >> 7);
    const std::uint64_t E  = ((c  & NCOL_0) >> 1)  | ((e  & COL_0) << 7);

    const std::uint64_t NW = ((c  & NROW_7 & NCOL_7) << 9)
                           | ((n  & ROW_7 & NCOL_7) >> 55)
                           | ((w  & NROW_7 & COL_7)  << 1)
                           | ((nw & ROW_7 & COL_7)   >> 63);

    const std::uint64_t NE = ((c  & NROW_7 & NCOL_0) << 7)
                           | ((n  & ROW_7 & NCOL_0) >> 57)
                           | ((e  & NROW_7 & COL_0) << 15)
                           | ((ne & ROW_7 & COL_0)  >> 49);

    const std::uint64_t SW = ((c  & NROW_0 & NCOL_7) >> 7)
                           | ((s  & ROW_0 & NCOL_7) << 57)
                           | ((w  & NROW_0 & COL_7) >> 15)
                           | ((sw & ROW_0 & COL_7)  << 49);

    const std::uint64_t SE = ((c  & NROW_0 & NCOL_0) >> 9)
                           | ((s  & ROW_0 & NCOL_0) << 55)
                           | ((e  & NROW_0 & COL_0) >> 1)
                           | ((se & ROW_0 & COL_0)  << 63);

    // Sum the eight neighbor bitboards using a CSA tree to obtain bit-sliced counts:
    // ones (bit 0), twos (bit 1), fours (bit 2), eights (bit 3) for each bit position independently.
    uint64_t s1, c1, s2, c2, s3, c3, s4, c4, s5, c5;

    // Level 1: three-way additions
    csa(N,  S,  E,  s1, c1);
    csa(W,  NE, NW, s2, c2);
    csa(SE, SW, 0ull, s3, c3);

    // Level 2: combine the sums (ones layer)
    csa(s1, s2, s3, s4, c4);
    const uint64_t ones = s4;

    // Combine carries from Level 1 and Level 2 to form twos/fours/eights bit-planes.
    // First, sum c1, c2, c3 (these are carries to the twos plane).
    csa(c1, c2, c3, s5, c5); // s5: partial twos, c5: partial fours

    // Add c4 (carry from combining s1,s2,s3) into the twos plane.
    const uint64_t twos_partial = s5 ^ c4;
    const uint64_t carry_twos   = s5 & c4; // carry from twos to fours

    // Fours plane is the XOR of carries from previous step
    const uint64_t fours_partial = c5 ^ carry_twos;
    const uint64_t eights        = c5 & carry_twos; // eights plane (max 1 bit per position)

    const uint64_t twos  = twos_partial;
    const uint64_t fours = fours_partial;

    // Masks for exact neighbor counts: 2 (0010) and 3 (0011)
    const uint64_t eq3 = (ones & twos) & ~fours & ~eights;
    const uint64_t eq2 = (~ones & twos) & ~fours & ~eights;

    // Apply Life rule: survive with 2 or 3; birth with exactly 3
    const std::uint64_t next = eq3 | (c & eq2);

    // Store result
    output[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Each tile is 8x8 cells → tiles_per_side = grid_dimensions / 8
    const int tiles_per_side = grid_dimensions >> 3; // grid_dimensions / 8
    // Launch one thread per tile. Use a 2D launch for good coalescing and occupancy.
    dim3 block(32, 8, 1); // 256 threads per block: good balance for modern GPUs
    dim3 grid((tiles_per_side + block.x - 1) / block.x,
              (tiles_per_side + block.y - 1) / block.y,
              1);

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // Host-device synchronization (if desired) is handled by the caller.
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
// MEMORY_LAYOUT: TILES
