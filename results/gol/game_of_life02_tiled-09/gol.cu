#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Bit packing and layout:
  - The grid is square with side length grid_dimensions (power of 2, > 512).
  - Each 64-bit word encodes an 8x8 tile of cells in row-major order:
      bit index b = (row * 8 + col), where row, col in [0..7].
      - bit 0 (LSB) corresponds to the cell at (row=0, col=0) within the tile.
      - bits 0..7 are row 0, bits 8..15 are row 1, ..., bits 56..63 are row 7.
  - The tiles themselves are laid out row-major in memory:
      tile index = tile_y * (grid_dimensions/8) + tile_x.
  - All cells outside the grid are treated as dead (0).
  - The input and output arrays are allocated with cudaMalloc.

  Kernel strategy:
  - One thread processes one 8x8 tile (one 64-bit word).
  - It loads its 8 neighboring tiles (W, E, N, S, NW, NE, SW, SE) plus itself (C).
  - It constructs three vertical "layers" (north, same row, south) for W/C/E tiles:
      up_X   = one-row-up of tile X, with vertical injection from its north neighbor
      mid_X  = tile X itself
      down_X = one-row-down of tile X, with vertical injection from its south neighbor
  - For each layer, it computes horizontal neighbor vectors (left-of-cell, same-column, right-of-cell)
    with proper horizontal injection from W/E tiles of the same layer.
  - It then adds the three per-layer 3-neighbour horizontal sums via bit-parallel binary adders
    to form bit-planes for the total 8-neighbor count:
      n1: 1's bit, n2: 2's bit, n4: 4's bit (we ignore the 8's bit; it's not needed for Life rules).
  - The Life rule is applied via the boolean formula:
      next = (~n4) & n2 & (n1 | C)
    which is equivalent to:
      (neighbors == 3) | (C & (neighbors == 2))
*/

static __device__ __forceinline__ std::uint64_t shift_left_neighbors(std::uint64_t x, std::uint64_t left_layer) {
    // For each cell position p, produce the value of the left neighbor (west) at p.
    // Implementation: shift x left by 1 within each 8-bit row and inject MSB-of-each-byte from the left-layer.
    const std::uint64_t MASK_FE = 0xfefefefefefefefeULL;  // clear bit0 in each byte after <<1
    const std::uint64_t FILE_H  = 0x8080808080808080ULL;  // MSB of each byte
    return ((x << 1) & MASK_FE) | ((left_layer & FILE_H) >> 7);
}

static __device__ __forceinline__ std::uint64_t shift_right_neighbors(std::uint64_t x, std::uint64_t right_layer) {
    // For each cell position p, produce the value of the right neighbor (east) at p.
    // Implementation: shift x right by 1 within each 8-bit row and inject LSB-of-each-byte from the right-layer.
    const std::uint64_t MASK_7F = 0x7f7f7f7f7f7f7f7fULL;  // clear bit7 in each byte after >>1
    const std::uint64_t FILE_A  = 0x0101010101010101ULL;  // LSB of each byte
    return ((x >> 1) & MASK_7F) | ((right_layer & FILE_A) << 7);
}

static __device__ __forceinline__ void add3_bitwise(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                    std::uint64_t& s1, std::uint64_t& s2) {
    // Bitwise addition of three 1-bit values per position: a + b + c -> two bit-planes s1 (1's), s2 (2's)
    // s1 = a ^ b ^ c
    // s2 = (a & b) | (a & c) | (b & c)
    s1 = a ^ b ^ c;
    s2 = (a & b) | (a & c) | (b & c);
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ in,
                                         std::uint64_t* __restrict__ out,
                                         int tiles_per_row) {
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tiles_total = tiles_per_row * tiles_per_row;
    if (tile_idx >= tiles_total) return;

    // Compute 2D tile coordinates
    int ty = tile_idx / tiles_per_row;
    int tx = tile_idx - ty * tiles_per_row;

    // Neighbor presence masks (edges are zero-filled)
    bool has_left  = (tx > 0);
    bool has_right = (tx + 1 < tiles_per_row);
    bool has_up    = (ty > 0);
    bool has_down  = (ty + 1 < tiles_per_row);

    int idxC = tile_idx;

    // Load 3x3 neighborhood tiles (9 words), zero for out-of-bounds.
    std::uint64_t C  = in[idxC];
    std::uint64_t W  = has_left  ? in[idxC - 1]               : 0ULL;
    std::uint64_t E  = has_right ? in[idxC + 1]               : 0ULL;
    std::uint64_t N  = has_up    ? in[idxC - tiles_per_row]   : 0ULL;
    std::uint64_t S  = has_down  ? in[idxC + tiles_per_row]   : 0ULL;
    std::uint64_t NW = (has_up && has_left)   ? in[idxC - tiles_per_row - 1] : 0ULL;
    std::uint64_t NE = (has_up && has_right)  ? in[idxC - tiles_per_row + 1] : 0ULL;
    std::uint64_t SW = (has_down && has_left) ? in[idxC + tiles_per_row - 1] : 0ULL;
    std::uint64_t SE = (has_down && has_right)? in[idxC + tiles_per_row + 1] : 0ULL;

    // Build vertical layers (up, mid, down) for W, C, E with vertical injection at tile borders.
    // up_X: for each cell position p, holds the cell directly above (north) of p.
    // down_X: holds the cell directly below (south) of p.
    std::uint64_t up_C   = (C >> 8) | (N >> 56);                   // N row7 -> row0
    std::uint64_t mid_C  = C;
    std::uint64_t down_C = (C << 8) | ((S & 0xffULL) << 56);       // S row0 -> row7

    std::uint64_t up_W   = (W >> 8) | (NW >> 56);
    std::uint64_t mid_W  = W;
    std::uint64_t down_W = (W << 8) | ((SW & 0xffULL) << 56);

    std::uint64_t up_E   = (E >> 8) | (NE >> 56);
    std::uint64_t mid_E  = E;
    std::uint64_t down_E = (E << 8) | ((SE & 0xffULL) << 56);

    // Horizontal neighbor vectors for the three vertical layers:
    // For each layer Lx:
    //   left-of-cell (west)  = shift_left_neighbors(Lx_C, Lx_W)
    //   same-column          = Lx_C   (except for mid layer where it's excluded, i.e., 0)
    //   right-of-cell (east) = shift_right_neighbors(Lx_C, Lx_E)
    // Top (north) layer
    std::uint64_t L_up = shift_left_neighbors(up_C, up_W);
    std::uint64_t C_up = up_C;
    std::uint64_t R_up = shift_right_neighbors(up_C, up_E);

    // Middle (same row) layer - exclude the cell itself
    std::uint64_t L_mid = shift_left_neighbors(mid_C, mid_W);
    std::uint64_t C_mid = 0ULL;
    std::uint64_t R_mid = shift_right_neighbors(mid_C, mid_E);

    // Bottom (south) layer
    std::uint64_t L_down = shift_left_neighbors(down_C, down_W);
    std::uint64_t C_down = down_C;
    std::uint64_t R_down = shift_right_neighbors(down_C, down_E);

    // Per-layer horizontal 3-neighbor sums -> two bit-planes per layer: s1 (1's), s2 (2's)
    std::uint64_t s1_up,   s2_up;
    std::uint64_t s1_mid,  s2_mid;
    std::uint64_t s1_down, s2_down;

    add3_bitwise(L_up,  C_up,  R_up,  s1_up,  s2_up);
    add3_bitwise(L_mid, C_mid, R_mid, s1_mid, s2_mid); // same as add2(L_mid, R_mid)
    add3_bitwise(L_down,C_down,R_down,s1_down,s2_down);

    // Accumulate the three layer sums using bit-parallel binary addition,
    // tracking 1's (n1), 2's (n2), and 4's (n4) bit-planes. 8's plane is ignored.
    // First, add top + mid
    std::uint64_t carry2_a = s1_up & s1_mid;
    std::uint64_t t1 = s1_up ^ s1_mid;
    std::uint64_t t2 = s2_up ^ s2_mid ^ carry2_a;
    std::uint64_t t4 = (s2_up & s2_mid) | (s2_up & carry2_a) | (s2_mid & carry2_a);

    // Then add bottom
    std::uint64_t carry2_b = t1 & s1_down;
    std::uint64_t n1 = t1 ^ s1_down;
    std::uint64_t n2 = t2 ^ s2_down ^ carry2_b;
    std::uint64_t carry4_b = (t2 & s2_down) | (t2 & carry2_b) | (s2_down & carry2_b);
    std::uint64_t n4 = t4 ^ carry4_b; // ignoring any carry into the 8's plane (not needed for Life rules)

    // Apply Game of Life rule:
    // next = (~n4) & n2 & (n1 | mid_C)
    std::uint64_t next = (~n4) & n2 & (n1 | mid_C);

    out[idxC] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the number of cells per side (power of 2, > 512, divisible by 8).
    const int tiles_per_row = grid_dimensions >> 3; // grid_dimensions / 8
    const int tiles_total = tiles_per_row * tiles_per_row;

    // Choose a launch configuration that balances occupancy and simplicity.
    // One thread per 8x8 tile.
    const int threads_per_block = 256;
    const int blocks = (tiles_total + threads_per_block - 1) / threads_per_block;

    game_of_life_step_kernel<<<blocks, threads_per_block>>>(input, output, tiles_per_row);
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
