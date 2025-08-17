#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Encoding and layout:
  - The grid is square with side length grid_dimensions (power of 2 > 512).
  - The grid is stored in tiles; each std::uint64_t encodes an 8x8 tile of cells.
  - Within each 64-bit tile:
      * Bit index = (row * 8 + col), where row and col are in [0, 7].
      * Bit 0 corresponds to the top-left cell of the tile (row = 0, col = 0).
      * Bit 7 is top-right (row = 0, col = 7).
      * Bit 56 is bottom-left (row = 7, col = 0).
      * Bit 63 is bottom-right (row = 7, col = 7).
  - Tiles are arranged in row-major order: tile_index = tile_y * tiles_per_dim + tile_x.

  Boundary conditions:
  - All cells outside the grid are dead. Therefore, any access to a non-existent neighbor tile is treated as 0.

  Kernel strategy:
  - Each CUDA thread processes one 8x8 tile (one 64-bit word).
  - For a tile, all eight directional neighbor bitboards (N, NE, E, SE, S, SW, W, NW) of the current tile are constructed
    by combining the current tile and up to eight neighboring tiles with carefully masked bit shifts that avoid wrap
    across 8-bit row lanes.
  - The eight neighbor bitboards are then accumulated using a bit-sliced (per-bit) ripple-carry population counter to
    produce four bitplanes (ones, twos, fours, eights) representing the neighbor count (0..8) per cell.
  - The next state is computed via:
      next = (neighbors == 3) | (alive & (neighbors == 2))
    evaluated using bitplanes without per-bit loops or scalar branching.

  Notes:
  - Shared or texture memory is not used; the global-memory access pattern is coalesced for the main tile array, and
    L1/L2 caches will capture spatial locality for neighbor tiles.
  - The approach avoids any data-dependent divergence except at grid borders (where only a small fraction of threads are affected).
*/

static __device__ __forceinline__ void add_to_counter(uint64_t bits,
                                                      uint64_t &ones,
                                                      uint64_t &twos,
                                                      uint64_t &fours,
                                                      uint64_t &eights)
{
    // Ripple-carry bit-sliced counter:
    // Incorporate 'bits' into the current per-bit count represented by planes (ones, twos, fours, eights).
    uint64_t c0 = ones & bits;
    ones ^= bits;

    uint64_t c1 = twos & c0;
    twos ^= c0;

    uint64_t c2 = fours & c1;
    fours ^= c1;

    eights ^= c2;
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int tiles_per_dim)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tiles = tiles_per_dim * tiles_per_dim;
    if (tile_idx >= total_tiles) return;

    // Tile coordinates
    int ty = tile_idx / tiles_per_dim;
    int tx = tile_idx % tiles_per_dim;

    // Load center tile and all 8 neighbor tiles (0 if out-of-bounds)
    uint64_t T = input[tile_idx];

    uint64_t N  = (ty > 0) ? input[tile_idx - tiles_per_dim] : 0ULL;
    uint64_t S  = (ty + 1 < tiles_per_dim) ? input[tile_idx + tiles_per_dim] : 0ULL;
    uint64_t W  = (tx > 0) ? input[tile_idx - 1] : 0ULL;
    uint64_t E  = (tx + 1 < tiles_per_dim) ? input[tile_idx + 1] : 0ULL;

    uint64_t NWt = (ty > 0 && tx > 0) ? input[tile_idx - tiles_per_dim - 1] : 0ULL;
    uint64_t NEt = (ty > 0 && tx + 1 < tiles_per_dim) ? input[tile_idx - tiles_per_dim + 1] : 0ULL;
    uint64_t SWt = (ty + 1 < tiles_per_dim && tx > 0) ? input[tile_idx + tiles_per_dim - 1] : 0ULL;
    uint64_t SEt = (ty + 1 < tiles_per_dim && tx + 1 < tiles_per_dim) ? input[tile_idx + tiles_per_dim + 1] : 0ULL;

    // Lane masks
    constexpr uint64_t COL0_MASK = 0x0101010101010101ULL; // bit 0 in each byte (column 0)
    constexpr uint64_t COL7_MASK = 0x8080808080808080ULL; // bit 7 in each byte (column 7)
    constexpr uint64_t NOT_COL0  = 0xFEFEFEFEFEFEFEFEULL; // ~COL0_MASK
    constexpr uint64_t NOT_COL7  = 0x7F7F7F7F7F7F7F7FULL; // ~COL7_MASK

    constexpr uint64_t ROW0_MASK = 0x00000000000000FFULL; // top row bits
    constexpr uint64_t ROW7_MASK = 0xFF00000000000000ULL; // bottom row bits
    constexpr uint64_t NOT_ROW0  = 0xFFFFFFFFFFFFFF00ULL; // ~ROW0_MASK
    constexpr uint64_t NOT_ROW7  = 0x00FFFFFFFFFFFFFFULL; // ~ROW7_MASK

    // Construct directional neighbor bitboards for the center tile.
    // Cardinal directions:
    uint64_t nbN = (T << 8)  | (N >> 56);
    uint64_t nbS = (T >> 8)  | (S << 56);
    uint64_t nbW = ((T & NOT_COL0) >> 1) | ((W & COL7_MASK) >> 7);
    uint64_t nbE = ((T & NOT_COL7) << 1) | ((E & COL0_MASK) << 7);

    // Diagonals:
    // NW: (x-1, y-1)
    uint64_t nbNW = ((T & NOT_COL0 & NOT_ROW0) << 9)
                  | (((N >> 56) & 0x7FULL) << 1)
                  | (((W & COL7_MASK & NOT_ROW7) << 1))
                  | (NWt >> 63);

    // NE: (x+1, y-1)
    uint64_t nbNE = ((T & NOT_COL7 & NOT_ROW0) << 7)
                  | ((N >> 56) >> 1)
                  | ((E & COL0_MASK & NOT_ROW7) << 15)
                  | (((NEt >> 56) & 0x1ULL) << 7);

    // SW: (x-1, y+1)
    uint64_t nbSW = ((T & NOT_COL0 & NOT_ROW7) >> 7)
                  | (((S & 0x7FULL) << 1) << 56)
                  | ((W & COL7_MASK & NOT_ROW0) >> 15)
                  | ((SWt & 0x0000000000000080ULL) << 49);

    // SE: (x+1, y+1)
    uint64_t nbSE = ((T & NOT_COL7 & NOT_ROW7) >> 9)
                  | (((S & 0xFFULL) >> 1) << 56)
                  | ((E & COL0_MASK & NOT_ROW0) >> 1)
                  | ((SEt & 0x1ULL) << 63);

    // Accumulate neighbor counts (0..8) per bit using bit-sliced counters.
    uint64_t ones = 0, twos = 0, fours = 0, eights = 0;
    add_to_counter(nbN,  ones, twos, fours, eights);
    add_to_counter(nbNE, ones, twos, fours, eights);
    add_to_counter(nbE,  ones, twos, fours, eights);
    add_to_counter(nbSE, ones, twos, fours, eights);
    add_to_counter(nbS,  ones, twos, fours, eights);
    add_to_counter(nbSW, ones, twos, fours, eights);
    add_to_counter(nbW,  ones, twos, fours, eights);
    add_to_counter(nbNW, ones, twos, fours, eights);

    // Compute masks for "neighbors == 2" and "neighbors == 3"
    uint64_t zero_fours = ~fours;
    uint64_t zero_eights = ~eights;

    uint64_t eq3 = (ones & twos) & zero_fours & zero_eights;          // 0011
    uint64_t eq2 = ((~ones) & twos) & zero_fours & zero_eights;       // 0010

    // Apply Game of Life rules: survive with 2, born with 3
    uint64_t alive = T;
    uint64_t next  = eq3 | (alive & eq2);

    output[tile_idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is in cells; tiles are 8x8, so tiles_per_dim is grid_dimensions / 8.
    const int tiles_per_dim = grid_dimensions >> 3;
    const int total_tiles = tiles_per_dim * tiles_per_dim;

    // Launch with a 1D grid: one thread per 8x8 tile.
    constexpr int TPB = 256; // Threads per block; good occupancy on modern GPUs.
    int blocks = (total_tiles + TPB - 1) / TPB;

    gol_step_kernel<<<blocks, TPB>>>(input, output, tiles_per_dim);
}