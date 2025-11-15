// Conway's Game of Life CUDA implementation using bit-packed 8×8 tiles.
//
// Representation:
// - The grid is square with size grid_dimensions × grid_dimensions cells.
// - grid_dimensions is a power of 2, >= 512, and divisible by 8.
// - Each std::uint64_t in the input/output encodes an 8×8 tile of cells.
// - Tiles are laid out in row-major order:
//     tile_index = tile_y * tiles_per_dim + tile_x,
//     where tiles_per_dim = grid_dimensions / 8.
// - Within each tile, bits are laid out row-major, with bit 0 representing
//   the cell at local coordinates (row=0, col=0), bit 1 => (0,1), ..., bit 7 => (0,7),
//   bit 8 => (1,0), etc.
//   In general: bit_index = row * 8 + col.
//
// Boundary handling:
// - Cells outside the global grid are considered dead (0).
// - This is implemented by treating neighbor tiles outside the grid as zero.
//
// Kernel strategy:
// - One CUDA thread processes one 8×8 tile (one uint64_t).
// - For each tile, the thread loads the 3×3 neighborhood of tiles (up to 9 × uint64_t).
// - For each of the 8 rows inside the central tile, it builds 8 bitmasks corresponding
//   to the 8 neighbor directions (N, NE, E, SE, S, SW, W, NW), each mask holding
//   neighbor bits for all 8 cells in that row at once.
// - It then performs a bit-sliced accumulation to get the neighbor count per cell
//   (0–8) in 4 bit-planes (c0..c3).
// - Using these bit-planes, it computes masks for "neighbor count == 2" and
//   "neighbor count == 3", and applies the Game of Life rules in parallel for
//   all 8 cells in the row, producing the next row as another 8-bit mask.
// - The 8 new rows are packed back into a uint64_t and written to the output.
//
// Performance notes:
// - No shared or texture memory is used; all reads/writes are from/to global memory.
// - Arithmetic is done with bitwise operations on 8-bit masks, minimizing per-cell
//   branching and memory traffic.
// - Grid/block configuration is chosen so that threads in a warp access consecutive
//   tiles, yielding coalesced 64-bit memory accesses.

#include <cstdint>
#include <cuda_runtime.h>

// Increment bit-sliced neighbor counters by a mask of "1"s (one per live neighbor).
// Each bit position across the bytes (c0, c1, c2, c3) forms a 4-bit counter for the
// corresponding cell in the row. The mask "bits" contains 1s at positions where we
// need to add 1 to the counter (i.e., the neighbor is alive).
//
// The counters c0..c3 represent:
//   count = c0 * 2^0 + c1 * 2^1 + c2 * 2^2 + c3 * 2^3
//
// This function adds 'bits' (0 or 1) to the count at each bit position in parallel.
__device__ __forceinline__
void add1ToCounters(std::uint8_t bits,
                    std::uint8_t &c0,
                    std::uint8_t &c1,
                    std::uint8_t &c2,
                    std::uint8_t &c3)
{
    std::uint8_t carry = bits;
    std::uint8_t carry_next;

    // Add carry to c0
    carry_next = carry & c0;
    c0 ^= carry;
    carry = carry_next;

    // Add carry to c1
    carry_next = carry & c1;
    c1 ^= carry;
    carry = carry_next;

    // Add carry to c2
    carry_next = carry & c2;
    c2 ^= carry;
    carry = carry_next;

    // Add carry to c3 (final bit-plane)
    c3 ^= carry;
}

// Kernel that processes one generation of Conway's Game of Life on a bit-packed grid.
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int tiles_per_dim)
{
    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tile_x >= tiles_per_dim || tile_y >= tiles_per_dim)
        return;

    const int tile_index = tile_y * tiles_per_dim + tile_x;

    const int stride = tiles_per_dim;

    // Load center tile and neighbor tiles.
    const bool has_n = tile_y > 0;
    const bool has_s = tile_y + 1 < tiles_per_dim;
    const bool has_w = tile_x > 0;
    const bool has_e = tile_x + 1 < tiles_per_dim;

    const std::uint64_t center = input[tile_index];

    std::uint64_t north  = 0;
    std::uint64_t south  = 0;
    std::uint64_t west   = 0;
    std::uint64_t east   = 0;
    std::uint64_t nw     = 0;
    std::uint64_t ne     = 0;
    std::uint64_t sw     = 0;
    std::uint64_t se     = 0;

    if (has_n) north = input[tile_index - stride];
    if (has_s) south = input[tile_index + stride];
    if (has_w) west  = input[tile_index - 1];
    if (has_e) east  = input[tile_index + 1];
    if (has_n && has_w) nw = input[tile_index - stride - 1];
    if (has_n && has_e) ne = input[tile_index - stride + 1];
    if (has_s && has_w) sw = input[tile_index + stride - 1];
    if (has_s && has_e) se = input[tile_index + stride + 1];

    std::uint64_t out_tile = 0;

    // Process each of the 8 rows within this 8×8 tile.
    #pragma unroll
    for (int r = 0; r < 8; ++r)
    {
        // Extract the current row and same-row neighbors from tiles.
        const std::uint8_t row_center = static_cast<std::uint8_t>((center >> (r * 8)) & 0xFFu);
        const std::uint8_t row_west   = static_cast<std::uint8_t>((west   >> (r * 8)) & 0xFFu);
        const std::uint8_t row_east   = static_cast<std::uint8_t>((east   >> (r * 8)) & 0xFFu);

        // Rows directly above and below the current row, plus their W/E neighbors.
        std::uint8_t row_above, row_above_w, row_above_e;
        std::uint8_t row_below, row_below_w, row_below_e;

        if (r > 0)
        {
            // Above is within the same tile.
            row_above   = static_cast<std::uint8_t>((center >> ((r - 1) * 8)) & 0xFFu);
            row_above_w = static_cast<std::uint8_t>((west   >> ((r - 1) * 8)) & 0xFFu);
            row_above_e = static_cast<std::uint8_t>((east   >> ((r - 1) * 8)) & 0xFFu);
        }
        else
        {
            // Above row comes from the tile to the north (or zero if out-of-bounds).
            row_above   = static_cast<std::uint8_t>((north >> (7 * 8)) & 0xFFu);
            row_above_w = static_cast<std::uint8_t>((nw    >> (7 * 8)) & 0xFFu);
            row_above_e = static_cast<std::uint8_t>((ne    >> (7 * 8)) & 0xFFu);
        }

        if (r < 7)
        {
            // Below is within the same tile.
            row_below   = static_cast<std::uint8_t>((center >> ((r + 1) * 8)) & 0xFFu);
            row_below_w = static_cast<std::uint8_t>((west   >> ((r + 1) * 8)) & 0xFFu);
            row_below_e = static_cast<std::uint8_t>((east   >> ((r + 1) * 8)) & 0xFFu);
        }
        else
        {
            // Below row comes from the tile to the south (or zero if out-of-bounds).
            row_below   = static_cast<std::uint8_t>( south        & 0xFFu);
            row_below_w = static_cast<std::uint8_t>( sw           & 0xFFu);
            row_below_e = static_cast<std::uint8_t>( se           & 0xFFu);
        }

        // Build neighbor direction masks (each is an 8-bit mask for this row).
        const std::uint8_t n_mask  = row_above;
        const std::uint8_t s_mask  = row_below;
        const std::uint8_t w_mask  = static_cast<std::uint8_t>((row_center << 1) | (row_west >> 7));
        const std::uint8_t e_mask  = static_cast<std::uint8_t>((row_center >> 1) | ((row_east & 0x01u) << 7));
        const std::uint8_t nw_mask = static_cast<std::uint8_t>((row_above  << 1) | (row_above_w >> 7));
        const std::uint8_t ne_mask = static_cast<std::uint8_t>((row_above  >> 1) | ((row_above_e & 0x01u) << 7));
        const std::uint8_t sw_mask = static_cast<std::uint8_t>((row_below  << 1) | (row_below_w >> 7));
        const std::uint8_t se_mask = static_cast<std::uint8_t>((row_below  >> 1) | ((row_below_e & 0x01u) << 7));

        // Bit-sliced neighbor counters for this row (4-bit counters per cell).
        std::uint8_t c0 = 0;
        std::uint8_t c1 = 0;
        std::uint8_t c2 = 0;
        std::uint8_t c3 = 0;

        // Accumulate contributions from all 8 neighbor directions.
        add1ToCounters(nw_mask, c0, c1, c2, c3);
        add1ToCounters(n_mask,  c0, c1, c2, c3);
        add1ToCounters(ne_mask, c0, c1, c2, c3);
        add1ToCounters(w_mask,  c0, c1, c2, c3);
        add1ToCounters(e_mask,  c0, c1, c2, c3);
        add1ToCounters(sw_mask, c0, c1, c2, c3);
        add1ToCounters(s_mask,  c0, c1, c2, c3);
        add1ToCounters(se_mask, c0, c1, c2, c3);

        // Compute masks where neighbor count == 2 or == 3.
        const std::uint8_t not_c0 = static_cast<std::uint8_t>(~c0);
        const std::uint8_t not_c1 = static_cast<std::uint8_t>(~c1);
        const std::uint8_t not_c2 = static_cast<std::uint8_t>(~c2);
        const std::uint8_t not_c3 = static_cast<std::uint8_t>(~c3);

        const std::uint8_t eq2 = static_cast<std::uint8_t>(not_c3 & not_c2 &  c1    & not_c0); // 0010
        const std::uint8_t eq3 = static_cast<std::uint8_t>(not_c3 & not_c2 &  c1    &  c0);    // 0011

        // Apply Game of Life rules:
        // - Any live cell with 2 or 3 neighbors survives.
        // - Any dead cell with exactly 3 neighbors becomes alive.
        const std::uint8_t survive_or_birth = static_cast<std::uint8_t>(eq3 | (row_center & eq2));

        // Insert this row into the output tile.
        out_tile |= (static_cast<std::uint64_t>(survive_or_birth) << (r * 8));
    }

    output[tile_index] = out_tile;
}

// Host function to launch one step of Conway's Game of Life on the GPU.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Number of 8×8 tiles along one side of the grid.
    const int tiles_per_dim = grid_dimensions >> 3; // grid_dimensions / 8

    // Choose a block size that yields good occupancy and coalesced accesses.
    const dim3 block_dim(32, 8); // 256 threads per block
    const dim3 grid_dim((tiles_per_dim + block_dim.x - 1) / block_dim.x,
                        (tiles_per_dim + block_dim.y - 1) / block_dim.y);

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, tiles_per_dim);
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
