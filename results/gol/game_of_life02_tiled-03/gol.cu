#include <cuda_runtime.h>
#include <cstdint>

/*
High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

Encoding and layout:
- The full grid is square with side length grid_dimensions (power of 2, >512).
- The grid is partitioned into 8x8 tiles. Each std::uint64_t encodes one tile:
  bit index b = row*8 + col, where row, col are in [0..7], row-major within the 64-bit word.
  - Bit 0 is row 0, col 0 (least significant bit).
  - Bit 7 is row 0, col 7.
  - Bit 8 is row 1, col 0.
  - ...
  - Bit 63 is row 7, col 7 (most significant bit).
- The array of tiles is laid out in row-major order across the 2D tile grid.
  tiles_per_dim = grid_dimensions / 8
  tile_index(tx, ty) = ty * tiles_per_dim + tx

Rules and implementation:
- For each 8x8 tile, we load its 3x3 neighborhood of tiles (9 values: NW, N, NE, W, C, E, SW, S, SE).
- We construct 8 "neighbor-direction" bitboards (N, S, E, W, NE, NW, SE, SW) that indicate,
  for each destination cell in the center tile, whether it has a live neighbor in that direction.
  These use bit shifts and masks to handle intra-tile shifts and cross-tile boundary bits.
- We then compute the per-cell neighbor count (0..8) across the eight direction bitboards using
  carry-save adders (bitwise parallel 3:2 compressors), producing three bitboards representing the
  binary digits of the count (bit1, bit2, bit4).
- The next-state bit is 1 if (count == 3) OR (alive AND count == 2).
  Implemented in bitwise form:
    eq2 = (~bit1) & bit2 & (~bit4)
    eq3 = ( bit1) & bit2 & (~bit4)
    next = eq3 | (alive & eq2)
- Boundary handling: Tiles outside the grid are treated as all-dead (0), as specified.

Performance notes:
- No shared or texture memory is used (as requested).
- Each thread processes one tile. Threads are organized in a 2D grid (x,y) over the tile grid to
  avoid divisions/modulos in index calculations and to promote coalesced global memory access.
- The kernel uses __forceinline__ and constexpr masks to keep critical operations in registers.
*/

namespace {

// Column and row masks for 8x8 tiles in the 64-bit word.
__device__ __constant__ std::uint64_t COL0_MASK = 0x0101010101010101ULL; // bits at col 0 of each row
__device__ __constant__ std::uint64_t COL7_MASK = 0x8080808080808080ULL; // bits at col 7 of each row
__device__ __constant__ std::uint64_t ROW0_MASK = 0x00000000000000FFULL; // bits of row 0 (bits 0..7)
__device__ __constant__ std::uint64_t ROW7_MASK = 0xFF00000000000000ULL; // bits of row 7 (bits 56..63)

// These combined masks select single corner bits when needed.
__device__ __constant__ std::uint64_t ROW7_COL0_MASK = 0x0100000000000000ULL; // row7 col0
__device__ __constant__ std::uint64_t ROW0_COL7_MASK = 0x0000000000000080ULL; // row0 col7

// Full adder for bitboards: sums 3 one-bit-per-lane inputs into sum (weight 1) and carry (weight 2).
__device__ __forceinline__
void full_adder(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                std::uint64_t &sum, std::uint64_t &carry)
{
    sum   = a ^ b ^ c;
    // Majority function for the carry: true if at least two inputs are 1
    carry = (a & b) | (a & c) | (b & c);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ input,
                                std::uint64_t* __restrict__ output,
                                int tiles_per_dim)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tiles_per_dim || ty >= tiles_per_dim) return;

    const int stride = tiles_per_dim;
    const int idxC   = ty * stride + tx;

    // In-bounds flags for neighbors
    const bool has_w = (tx > 0);
    const bool has_e = (tx + 1 < tiles_per_dim);
    const bool has_n = (ty > 0);
    const bool has_s = (ty + 1 < tiles_per_dim);

    // Load 3x3 neighbor tiles; tiles outside the grid are treated as 0 (dead).
    const std::uint64_t C  = input[idxC];
    const std::uint64_t W  = has_w ? input[idxC - 1]        : 0ULL;
    const std::uint64_t E  = has_e ? input[idxC + 1]        : 0ULL;
    const std::uint64_t N  = has_n ? input[idxC - stride]   : 0ULL;
    const std::uint64_t S  = has_s ? input[idxC + stride]   : 0ULL;
    const std::uint64_t NW = (has_n && has_w) ? input[idxC - stride - 1] : 0ULL;
    const std::uint64_t NE = (has_n && has_e) ? input[idxC - stride + 1] : 0ULL;
    const std::uint64_t SW = (has_s && has_w) ? input[idxC + stride - 1] : 0ULL;
    const std::uint64_t SE = (has_s && has_e) ? input[idxC + stride + 1] : 0ULL;

    // Directional neighbor bitboards.
    // For each destination bit (cell) in the center tile, these bitboards indicate whether an
    // alive neighbor exists in that direction. They account for cross-tile boundaries explicitly.

    // Horizontal neighbors within same row: West (left) and East (right)
    const std::uint64_t west  = ((C & ~COL7_MASK) << 1) | ((W & COL7_MASK) >> 7);
    const std::uint64_t east  = ((C & ~COL0_MASK) >> 1) | ((E & COL0_MASK) << 7);

    // Vertical neighbors: North (up) and South (down)
    const std::uint64_t north = (C << 8) | (N >> 56);
    const std::uint64_t south = (C >> 8) | (S << 56);

    // Diagonal neighbors:
    // NE: combines contributions from C (internal), N (top row), E (right col), NE (top-right corner).
    const std::uint64_t ne =
        ((C & ~COL0_MASK) << 7) |
        (N >> 57) |
        ((E & COL0_MASK & ~ROW7_MASK) << 15) |
        ((NE & ROW7_COL0_MASK) >> 49);

    // NW: combines contributions from C (internal), N (top row), W (left col), NW (top-left corner).
    const std::uint64_t nw =
        ((C & ~COL7_MASK) << 9) |
        ((N & ~COL7_MASK) >> 55) |
        ((W & COL7_MASK & ~ROW7_MASK) << 1) |
        (NW >> 63); // only the single bit (row7,col7) can contribute after >>63

    // SE: combines contributions from C (internal), S (bottom row), E (right col), SE (bottom-right corner).
    const std::uint64_t se =
        ((C & ~COL0_MASK) >> 9) |
        ((S & ~COL0_MASK) << 55) |
        ((E & COL0_MASK & ~ROW0_MASK) >> 1) |
        (SE << 63); // only the single bit (row0,col0) can contribute after <<63

    // SW: combines contributions from C (internal), S (bottom row), W (left col), SW (bottom-left corner).
    const std::uint64_t sw =
        ((C & ~COL7_MASK) >> 7) |
        ((S & ~COL7_MASK) << 57) |
        ((W & COL7_MASK & ~ROW0_MASK) >> 15) |
        ((SW & ROW0_COL7_MASK) << 49);

    // Sum the eight direction bitboards using carry-save adders to produce 3-bit counts per cell.
    // Stage 1: three 3-input adders to reduce 8 inputs to 3 sums and 3 carries (all 64-bit bitboards).
    std::uint64_t sA, cA, sB, cB, sC, cC;
    full_adder(north, south, east, sA, cA);
    full_adder(west,  ne,    nw,   sB, cB);
    full_adder(se,    sw,    0ULL, sC, cC);

    // Stage 2: combine the sums and carries separately.
    std::uint64_t sum1, carry1; // weight-1 sum and its carry (weight-2)
    std::uint64_t sum2, carry2; // weight-2 sum and its carry (weight-4)
    full_adder(sA, sB, sC, sum1, carry1);
    full_adder(cA, cB, cC, sum2, carry2);

    // Final digits of the neighbor count:
    const std::uint64_t bit1 = sum1;                 // 1's place
    const std::uint64_t bit2 = sum2 ^ carry1;        // 2's place
    const std::uint64_t bit4 = carry2 | (sum2 & carry1); // 4's place

    // Game of Life rules: next = (count == 3) | (alive & (count == 2))
    const std::uint64_t alive = C;
    const std::uint64_t eq3   = (bit1 & bit2) & ~bit4;      // count == 3 (011)
    const std::uint64_t eq2   = (~bit1 & bit2) & ~bit4;     // count == 2 (010)
    const std::uint64_t next  = eq3 | (alive & eq2);

    output[idxC] = next;
}

} // anonymous namespace

// Host function that launches one step of the Game of Life on the GPU.
// - input:  pointer to bit-packed 8x8 tiles (cudaMalloc'd device memory).
// - output: pointer to bit-packed 8x8 tiles (cudaMalloc'd device memory).
// - grid_dimensions: side length of the full grid in cells (power of 2, >512).
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of tiles along one dimension.
    const int tiles_per_dim = grid_dimensions >> 3; // divide by 8

    // Configure a 2D launch grid over the tile grid. 32x8 threads per block gives 256 threads per block.
    dim3 block(32, 8, 1);
    dim3 grid((tiles_per_dim + block.x - 1) / block.x,
              (tiles_per_dim + block.y - 1) / block.y,
              1);

    gol_step_kernel<<<grid, block>>>(input, output, tiles_per_dim);

    // No synchronization here; caller handles synchronization if needed.
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
