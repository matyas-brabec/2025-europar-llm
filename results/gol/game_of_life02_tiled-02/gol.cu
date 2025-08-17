#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life.

  Important details:
  - The input/output formats are bit-packed with each std::uint64_t representing an 8x8 tile of cells.
  - We internally transform to a row-major bitset (1 bit per cell, contiguous along rows) to exploit fast
    wide bitwise operations for neighbor counting. After computing the next generation, we convert back.
  - Transform kernels (tile <-> row-major) are designed to be simple and free of write hazards.
  - The Game of Life step itself uses carry-save addition (CSA) to count the 8 neighbors per bit in parallel,
    producing only the masks for "exactly 2 neighbors" and "exactly 3 neighbors" that are needed by the rules.
  - No shared or texture memory is used as requested.

  Bit layout conventions (self-consistent for IO):
  - Inside each 8x8 tile uint64_t:
      bit index = (row_in_tile * 8 + col_in_tile), with LSB = bit 0 = (row=0, col=0).
  - Inside the row-major representation:
      Each row is an array of 64-bit words; bit k of word w represents column (w*64 + k).
      LSB (bit 0) of a row word is the leftmost column within that 64-wide block.

  Boundary conditions:
  - All cells outside the grid are considered dead. This is enforced by zero-padding logically at the edges
    when shifting across row word boundaries and when accessing the first/last rows.
*/

using u64 = std::uint64_t;

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Device-side helper: shift left by 1 across 64-bit word boundary, bringing in prev_word's MSB.
__device__ __forceinline__ u64 shl1_carry(u64 x, u64 prev_word) {
    return (x << 1) | (prev_word >> 63);
}

// Device-side helper: shift right by 1 across 64-bit word boundary, bringing in next_word's LSB.
__device__ __forceinline__ u64 shr1_carry(u64 x, u64 next_word) {
    return (x >> 1) | (next_word << 63);
}

// Carry-Save Adder for three 1-bit-per-lane bitboards.
// Given a, b, c (bitboards), produces:
//   l = a ^ b ^ c  (sum bits, weight 1)
//   h = (a & b) | (a & c) | (b & c)  (carry bits, to be added at next weight)
__device__ __forceinline__ void csa(u64& h, u64& l, u64 a, u64 b, u64 c) {
    u64 u = a ^ b;
    l = u ^ c;
    h = (a & b) | (u & c);
}

/*
  Kernel: Convert from tile-packed (8x8 per u64) to row-major (1 bit per cell laid out along rows).
  Each thread outputs one 64-bit row word, assembling it from 8 tiles (8 bytes each) without write conflicts.

  rows_out[y * words_per_row + w] aggregates columns [w*64 .. w*64+63] of row y.
*/
__global__ void kernel_tiles_to_rows(const u64* __restrict__ tiles_in,
                                     u64* __restrict__ rows_out,
                                     int grid_dim) {
    const int words_per_row = grid_dim / 64;
    int w = blockIdx.x * blockDim.x + threadIdx.x; // 64-bit word index within the row
    int y = blockIdx.y;                             // row index
    if (y >= grid_dim || w >= words_per_row) return;

    const int tiles_per_row = grid_dim / 8;
    const int ty = y >> 3;      // tile row index
    const int r_in_tile = y & 7;
    const int tx0 = w << 3;     // starting tile column for this 64-bit row word (8 tiles per word)

    u64 acc = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int tx = tx0 + k;
        const size_t tile_idx = static_cast<size_t>(ty) * tiles_per_row + tx;
        u64 t = tiles_in[tile_idx];
        u64 row8 = (t >> (r_in_tile * 8)) & 0xFFu; // 8 bits for this row inside the tile
        acc |= (row8 << (k * 8));
    }
    rows_out[static_cast<size_t>(y) * words_per_row + w] = acc;
}

/*
  Kernel: One Game of Life step on the row-major bitset.
  Each thread processes one 64-bit word (columns [i*64 .. i*64+63]) for a single row y.
  Neighbor counts are computed using carry-save addition of the 8 neighbor bitboards.

  Eight neighbor contributions per word:
    - From the row above (if any): NW, N, NE  -> (shifted left, same, shifted right)
    - From the same row: W, E                -> (shifted left, shifted right)
    - From the row below (if any): SW, S, SE -> (shifted left, same, shifted right)
  The central cell itself is not added (rules exclude the center when counting neighbors).
*/
__global__ void kernel_gol_step_rows(const u64* __restrict__ rows_in,
                                     u64* __restrict__ rows_out,
                                     int height,
                                     int words_per_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // word index within row
    int y = blockIdx.y;                             // row index
    if (y >= height || i >= words_per_row) return;

    const u64* rowU = (y > 0) ? (rows_in + static_cast<size_t>(y - 1) * words_per_row) : nullptr;
    const u64* rowC = rows_in + static_cast<size_t>(y) * words_per_row;
    const u64* rowD = (y + 1 < height) ? (rows_in + static_cast<size_t>(y + 1) * words_per_row) : nullptr;

    // Load current row and neighbors (with zero-padding at edges)
    const u64 cC = rowC[i];
    const u64 cL = (i > 0) ? rowC[i - 1] : 0;
    const u64 cR = (i + 1 < words_per_row) ? rowC[i + 1] : 0;

    const u64 uC = rowU ? rowU[i] : 0;
    const u64 uL = (rowU && i > 0) ? rowU[i - 1] : 0;
    const u64 uR = (rowU && i + 1 < words_per_row) ? rowU[i + 1] : 0;

    const u64 dC = rowD ? rowD[i] : 0;
    const u64 dL = (rowD && i > 0) ? rowD[i - 1] : 0;
    const u64 dR = (rowD && i + 1 < words_per_row) ? rowD[i + 1] : 0;

    // Eight neighbor bitboards aligned to center positions
    const u64 t0 = shl1_carry(uC, uL);          // NW
    const u64 t1 = uC;                          // N
    const u64 t2 = shr1_carry(uC, uR);          // NE
    const u64 t3 = shl1_carry(cC, cL);          // W
    const u64 t4 = shr1_carry(cC, cR);          // E
    const u64 t5 = shl1_carry(dC, dL);          // SW
    const u64 t6 = dC;                          // S
    const u64 t7 = shr1_carry(dC, dR);          // SE

    // Carry-save addition tree to compute neighbor count bit-planes (b0,b1,b2,b3)
    u64 h1, l1; csa(h1, l1, t0, t1, t2);       // sums of top band
    u64 h2, l2; csa(h2, l2, t3, t4, t5);       // sums of middle-left/right + SW
    u64 h3, l3; csa(h3, l3, t6, t7, 0ull);     // sums of bottom band

    u64 h4, l4; csa(h4, l4, l1, l2, l3);       // combine (LSB plane)
    u64 h5, l5; csa(h5, l5, h1, h2, h3);       // combine carries (these are weight 2)

    u64 h6, l6; csa(h6, l6, h4, l5, 0ull);     // merge 2's carries with 2's sums

    const u64 b0 = l4;              // 1's bit-plane
    const u64 b1 = l6;              // 2's bit-plane
    const u64 b2 = h5 ^ h6;         // 4's bit-plane
    const u64 b3 = h5 & h6;         // 8's bit-plane

    const u64 eq2 = b1 & ~b0 & ~b2 & ~b3;   // exactly 2 neighbors
    const u64 eq3 = b1 &  b0 & ~b2 & ~b3;   // exactly 3 neighbors

    const u64 next = eq3 | (cC & eq2);      // births or survival

    rows_out[static_cast<size_t>(y) * words_per_row + i] = next;
}

/*
  Kernel: Convert from row-major bitset back to tile-packed (8x8 per u64).
  Each thread builds one tile word by gathering 8 bytes from 8 rows.
*/
__global__ void kernel_rows_to_tiles(const u64* __restrict__ rows_in,
                                     u64* __restrict__ tiles_out,
                                     int grid_dim) {
    const int tiles_per_row = grid_dim / 8;
    const int words_per_row = grid_dim / 64;

    int tx = blockIdx.x * blockDim.x + threadIdx.x; // tile column
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // tile row
    if (tx >= tiles_per_row || ty >= tiles_per_row) return;

    const int word_idx = tx >> 3;        // which 64-bit row word contains this tile's 8 columns
    const int bit_off = (tx & 7) * 8;    // bit offset within that 64-bit row word

    u64 tile = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int y = ty * 8 + r;
        const size_t row_word_idx = static_cast<size_t>(y) * words_per_row + word_idx;
        u64 row_word = rows_in[row_word_idx];
        u64 row8 = (row_word >> bit_off) & 0xFFu;
        tile |= (row8 << (r * 8));
    }

    tiles_out[static_cast<size_t>(ty) * tiles_per_row + tx] = tile;
}

/*
  Public API: run one step of Game of Life.

  input  - device pointer to tile-packed grid (each u64 encodes an 8x8 tile).
  output - device pointer to tile-packed grid for the next generation.
  grid_dimensions - width/height (square), power of two > 512. We assume it is divisible by 64.
*/
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int N = grid_dimensions;
    const int tiles_per_row = N / 8;
    const int words_per_row = N / 64;
    const size_t row_word_count = static_cast<size_t>(N) * words_per_row;

    // Allocate temporary row-major buffers (device)
    u64* d_rows_in = nullptr;
    u64* d_rows_out = nullptr;
    cudaMalloc(&d_rows_in, row_word_count * sizeof(u64));
    cudaMalloc(&d_rows_out, row_word_count * sizeof(u64));

    // Launch: tiles -> rows
    {
        dim3 block(256, 1, 1);
        dim3 grid(ceil_div_int(words_per_row, block.x), N, 1);
        kernel_tiles_to_rows<<<grid, block>>>(reinterpret_cast<const u64*>(input), d_rows_in, N);
    }

    // Launch: one Game of Life step on row-major bitset
    {
        dim3 block(256, 1, 1);
        dim3 grid(ceil_div_int(words_per_row, block.x), N, 1);
        kernel_gol_step_rows<<<grid, block>>>(d_rows_in, d_rows_out, N, words_per_row);
    }

    // Launch: rows -> tiles
    {
        dim3 block(16, 16, 1);
        dim3 grid(ceil_div_int(tiles_per_row, block.x), ceil_div_int(tiles_per_row, block.y), 1);
        kernel_rows_to_tiles<<<grid, block>>>(d_rows_out, reinterpret_cast<u64*>(output), N);
    }

    // Free temporaries (caller handles any synchronization)
    cudaFree(d_rows_in);
    cudaFree(d_rows_out);
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
