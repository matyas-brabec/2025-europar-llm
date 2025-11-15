#include <cstdint>
#include <cuda_runtime.h>

// Bring std::uint64_t into the global namespace for convenience on device.
using std::uint64_t;

/*
 * Bitwise helpers
 *
 * Each CUDA thread processes one 64‑bit word, which represents 64 horizontally
 * adjacent cells in one row. Bit 0 is the leftmost cell, bit 63 the rightmost.
 *
 * To compute neighbors across 64‑bit word boundaries, we need to construct
 * masks where each bit k corresponds to one of the 8 neighboring cells of
 * the cell at position k in the current word.
 *
 * For horizontal neighbors we use the current word and its immediate
 * left/right neighbor words and shift+OR them to propagate neighbor bits
 * across 64‑bit boundaries.
 */

/**
 * Compute the mask of west (left) neighbors for all 64 cells in a row segment.
 *
 * For each bit position k in the returned word:
 *   result_bit[k] = cell_state[x-1] (west of x), where x is the cell at bit k.
 *
 * 'cur'  contains cells [x, x+63]
 * 'left' contains cells [x-64, x-1] (0 if there is no left word).
 *
 * Within 'cur', cur << 1 shifts cell at bit (k-1) into position k.
 * For k = 0, the west neighbor is bit 63 of 'left', which is added via OR.
 */
__device__ __forceinline__
uint64_t shift_west(uint64_t cur, uint64_t left)
{
    return (cur << 1) | (left >> 63);
}

/**
 * Compute the mask of east (right) neighbors for all 64 cells in a row segment.
 *
 * For each bit position k in the returned word:
 *   result_bit[k] = cell_state[x+1] (east of x), where x is the cell at bit k.
 *
 * 'cur'   contains cells [x, x+63]
 * 'right' contains cells [x+64, x+127] (0 if there is no right word).
 *
 * Within 'cur', cur >> 1 shifts cell at bit (k+1) into position k.
 * For k = 63, the east neighbor is bit 0 of 'right', moved into bit 63.
 */
__device__ __forceinline__
uint64_t shift_east(uint64_t cur, uint64_t right)
{
    return (cur >> 1) | ((right & 1uLL) << 63);
}

/**
 * Add a 1‑bit neighbor mask to a 4‑bit per‑cell accumulator, in parallel for
 * all 64 cells.
 *
 * The neighbor count for each cell is represented in bit‑sliced form:
 *   count = c0 + 2*c1 + 4*c2 + 8*c3
 *
 * Each ci is a 64‑bit word holding that bitplane for all 64 cells.
 *
 * 'bits' is a 64‑bit mask of a single neighbor direction (e.g. "north").
 *
 * We perform a ripple‑carry addition of a 1‑bit value ('bits') into the
 * 4‑bit accumulator using only bitwise XOR/AND operations. This is SWAR-style
 * (SIMD Within A Register) arithmetic: every bit position is updated in
 * parallel.
 *
 * Maximum number of neighbors is 8, so 4 bits are sufficient (0..15).
 */
__device__ __forceinline__
void add_neighbor_bits(uint64_t bits,
                       uint64_t &c0, uint64_t &c1,
                       uint64_t &c2, uint64_t &c3)
{
    uint64_t carry = bits;

    // Add to bit 0
    uint64_t sum0 = c0 ^ carry;
    carry         = c0 & carry;

    // Add carry to bit 1
    uint64_t sum1 = c1 ^ carry;
    carry         = c1 & carry;

    // Add carry to bit 2
    uint64_t sum2 = c2 ^ carry;
    carry         = c2 & carry;

    // Add carry to bit 3
    uint64_t sum3 = c3 ^ carry;
    carry         = c3 & carry; // carry beyond 4 bits can't occur (max sum 8).

    c0 = sum0;
    c1 = sum1;
    c2 = sum2;
    c3 = sum3;
}

/**
 * CUDA kernel: one Game of Life step on a bit‑packed square grid.
 *
 * Layout:
 *   - The grid is grid_dim x grid_dim cells.
 *   - Each row is packed into 'words_per_row' 64‑bit words.
 *   - Thread i processes the i‑th 64‑bit word in row‑major order.
 *
 * For each 64‑bit word, we:
 *   1. Load up to 9 neighboring words (3 rows x 3 columns) from global memory.
 *   2. Construct 8 neighbor-direction masks (N, S, E, W, NE, NW, SE, SW)
 *      using shift_west/shift_east to cross word boundaries.
 *   3. Accumulate neighbor counts using bit‑sliced addition over 8 masks.
 *   4. Compute masks for "neighbors == 2" and "neighbors == 3".
 *   5. Apply Life rules in parallel to all 64 cells:
 *        - birth where neighbors == 3
 *        - survival where cell is alive and neighbors == 2
 */
__global__
void game_of_life_kernel(const uint64_t* __restrict__ input,
                         uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row,
                         int words_log2,
                         int words_mask,
                         int total_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words)
        return;

    // Map linear word index -> (row, col_word) using power-of-two arithmetic.
    int row = idx >> words_log2;      // row = idx / words_per_row
    int col = idx &  words_mask;      // col = idx % words_per_row

    const uint64_t* center_row = input + row * words_per_row;

    uint64_t rowN   = 0, rowN_L = 0, rowN_R = 0;
    uint64_t rowC   = 0, rowC_L = 0, rowC_R = 0;
    uint64_t rowS   = 0, rowS_L = 0, rowS_R = 0;

    // Fast path for interior region: no boundary checks needed.
    if (row > 0 && row < grid_dim - 1 && col > 0 && col < words_per_row - 1)
    {
        const uint64_t* north_row = center_row - words_per_row;
        const uint64_t* south_row = center_row + words_per_row;

        rowN_L = north_row[col - 1];
        rowN   = north_row[col];
        rowN_R = north_row[col + 1];

        rowC_L = center_row[col - 1];
        rowC   = center_row[col];
        rowC_R = center_row[col + 1];

        rowS_L = south_row[col - 1];
        rowS   = south_row[col];
        rowS_R = south_row[col + 1];
    }
    else
    {
        // Boundary handling: treat cells outside the grid as dead (0).
        // We only load neighbors that are within bounds, otherwise we leave 0.
        // Center row always exists.
        rowC   = center_row[col];
        rowC_L = (col > 0) ? center_row[col - 1] : 0;
        rowC_R = (col + 1 < words_per_row) ? center_row[col + 1] : 0;

        if (row > 0)
        {
            const uint64_t* north_row = center_row - words_per_row;
            rowN   = north_row[col];
            rowN_L = (col > 0) ? north_row[col - 1] : 0;
            rowN_R = (col + 1 < words_per_row) ? north_row[col + 1] : 0;
        }
        // else rowN, rowN_L, rowN_R remain 0

        if (row + 1 < grid_dim)
        {
            const uint64_t* south_row = center_row + words_per_row;
            rowS   = south_row[col];
            rowS_L = (col > 0) ? south_row[col - 1] : 0;
            rowS_R = (col + 1 < words_per_row) ? south_row[col + 1] : 0;
        }
        // else rowS, rowS_L, rowS_R remain 0
    }

    // Construct 8 neighbor direction masks for this 64‑bit segment.
    uint64_t north_west = shift_west(rowN, rowN_L);
    uint64_t north      = rowN;
    uint64_t north_east = shift_east(rowN, rowN_R);

    uint64_t west       = shift_west(rowC, rowC_L);
    uint64_t east       = shift_east(rowC, rowC_R);

    uint64_t south_west = shift_west(rowS, rowS_L);
    uint64_t south      = rowS;
    uint64_t south_east = shift_east(rowS, rowS_R);

    // Bit-sliced neighbor count: accumulate 8 one-bit masks into 4 bitplanes.
    uint64_t c0 = 0; // least significant bit of neighbor count
    uint64_t c1 = 0;
    uint64_t c2 = 0;
    uint64_t c3 = 0; // most significant bit (up to 8 neighbors)

    add_neighbor_bits(north_west, c0, c1, c2, c3);
    add_neighbor_bits(north,      c0, c1, c2, c3);
    add_neighbor_bits(north_east, c0, c1, c2, c3);
    add_neighbor_bits(west,       c0, c1, c2, c3);
    add_neighbor_bits(east,       c0, c1, c2, c3);
    add_neighbor_bits(south_west, c0, c1, c2, c3);
    add_neighbor_bits(south,      c0, c1, c2, c3);
    add_neighbor_bits(south_east, c0, c1, c2, c3);

    // Masks for neighbor_count == 2 and neighbor_count == 3.
    //
    // count == 2  -> binary 0010 -> ~c3 & ~c2 &  c1 & ~c0
    // count == 3  -> binary 0011 -> ~c3 & ~c2 &  c1 &  c0
    uint64_t not_c0 = ~c0;
    uint64_t not_c1 = ~c1;
    uint64_t not_c2 = ~c2;
    uint64_t not_c3 = ~c3;

    (void)not_c1; // not_c1 is not needed for eq2/eq3 but kept for clarity.

    uint64_t eq2 = not_c3 & not_c2 &  c1 & not_c0;
    uint64_t eq3 = not_c3 & not_c2 &  c1 &  c0;

    // Apply Game of Life rules:
    //   next = (neighbors == 3) OR (alive & neighbors == 2)
    uint64_t alive = rowC;
    uint64_t next  = eq3 | (alive & eq2);

    output[idx] = next;
}

/**
 * Host function: perform one step of Conway's Game of Life on a bit‑packed grid.
 *
 * Parameters:
 *   - input  : device pointer to current grid state (bit‑packed).
 *   - output : device pointer to next grid state (bit‑packed).
 *   - grid_dimensions : number of cells per dimension (square grid).
 *
 * Layout:
 *   - Grid is grid_dimensions x grid_dimensions cells.
 *   - Each row is stored as grid_dimensions / 64 consecutive uint64_t words.
 *
 * Notes:
 *   - Input and output must already be allocated on the device via cudaMalloc.
 *   - No host/device synchronization is performed here; the caller is
 *     responsible for synchronizing if needed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    int words_per_row = grid_dimensions >> 6;          // grid_dimensions / 64
    int total_words   = grid_dimensions * words_per_row;

    // words_per_row is a power of two because grid_dimensions is a power of two
    // and we divide by 64 (also power of two). We use this fact to convert
    // linear indices to (row, col) via shifts and masks.
    int words_log2 = 0;
    while ((1 << words_log2) < words_per_row)
        ++words_log2;
    int words_mask = words_per_row - 1;

    int blockSize = 256;
    int gridSize  = (total_words + blockSize - 1) / blockSize;

    game_of_life_kernel<<<gridSize, blockSize>>>(
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output),
        grid_dimensions,
        words_per_row,
        words_log2,
        words_mask,
        total_words
    );
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
// MEMORY_LAYOUT: ROWS
