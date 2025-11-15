#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

/*
 * CUDA implementation of one step of Conway's Game of Life on a square grid.
 *
 * Grid representation
 * -------------------
 * - The grid is grid_dimensions x grid_dimensions cells.
 * - Each cell is a single bit: 0 = dead, 1 = alive.
 * - The grid is stored in row-major order, bit-packed into std::uint64_t words.
 *   Each word contains 64 consecutive cells in the same row.
 *   Bit 0 (LSB) corresponds to the leftmost cell within the word,
 *   bit 63 (MSB) to the rightmost.
 *
 * Thread mapping
 * --------------
 * - Each CUDA thread processes exactly one 64-bit word.
 * - The kernel is launched with a 2D grid of thread blocks:
 *       x-dimension over word columns in a row,
 *       y-dimension over rows.
 *
 * Neighborhood handling
 * ---------------------
 * For a given word W at (row, col_word), we need up to 9 words:
 *   Row above:    n_left, n, n_right
 *   Current row:  c_left, c, c_right
 *   Row below:    s_left, s, s_right
 *
 * Border cells: any word outside the valid range is treated as 0 (all dead cells).
 *
 * Within each 64-bit word, we compute the 8 neighbor bitmasks:
 *   W  - left neighbors  (same row)
 *   E  - right neighbors (same row)
 *   N  - above
 *   S  - below
 *   NW - above-left
 *   NE - above-right
 *   SW - below-left
 *   SE - below-right
 *
 * To handle word-boundary neighbors (bit 0 and bit 63), we use shifts with
 * cross-word carry:
 *   - Left neighbor mask:
 *       L(x, left)  = (x << 1) | (left >> 63)
 *   - Right neighbor mask:
 *       R(x, right) = (x >> 1) | ((right & 1ULL) << 63)
 *
 * With bit 0 as the leftmost cell in a word:
 *   - For cell j, its left neighbor is at bit j-1 (or bit 63 of the left word).
 *   - For cell j, its right neighbor is at bit j+1 (or bit 0 of the right word).
 *
 * Neighbor counting (bit-parallel)
 * --------------------------------
 * We have 8 bitmasks (NW, N, NE, W, E, SW, S, SE). For each bit position,
 * we must count how many of these 8 masks have a '1' at that position.
 *
 * Instead of counting per cell, we use a bit-sliced counter:
 *   - Maintain three 64-bit words: count0, count1, count2.
 *   - For each bit position i:
 *       (count2[i], count1[i], count0[i]) is the 3-bit count of neighbors
 *       at that position, modulo 8.
 *
 * Since each cell has at most 8 neighbors, counts are in [0, 8]. Representing
 * the count modulo 8 is sufficient, because:
 *   - The Game of Life rules depend only on whether the count is 2 or 3.
 *   - 2 (010b) and 3 (011b) are distinct modulo 8.
 *   - Count 8 (1000b) becomes 000b modulo 8, which is fine because cells with
 *     8 neighbors are not supposed to survive or be born (they behave like 0).
 *
 * To add a neighbor mask 'm' to the counters, we perform a 3-bit increment
 * controlled by 'm' for each bit position:
 *   if m[i] == 1: count[i] = (count[i] + 1) mod 8
 *   else:         count[i] unchanged
 *
 * This is implemented using bitwise logic (no integer addition, so no cross-bit
 * carries) via:
 *   carry1 = count0 & m;
 *   count0 ^= m;
 *   carry2 = count1 & carry1;
 *   count1 ^= carry1;
 *   count2 ^= carry2;
 *
 * After processing all 8 neighbor masks, for each bit:
 *   - neighbor_count == 3  if  (count2 == 0, count1 == 1, count0 == 1)
 *   - neighbor_count == 2  if  (count2 == 0, count1 == 1, count0 == 0)
 *
 * Thus:
 *   neighbors_eq_3 =  count0 &  count1 & ~count2
 *   neighbors_eq_2 = ~count0 &  count1 & ~count2
 *
 * Game of Life transition
 * ------------------------
 * Let 'c' be the current word (bitmask of alive cells in this word).
 * The next state is:
 *   next = neighbors_eq_3 | (c & neighbors_eq_2)
 *        = born            | survived
 *
 * Boundary conditions
 * -------------------
 * All cells outside the grid are considered dead. We implement this by setting
 * neighbor words (n, s, left/right variants) to zero when the corresponding
 * row/column index is out of bounds, and by not injecting any carry from those
 * words during shift-with-carry operations.
 */

////////////////////////////////////////////////////////////////////////////////
// Device helper functions
////////////////////////////////////////////////////////////////////////////////

/**
 * Shift a word left by one bit, inserting bit 63 of 'left' into bit 0.
 *
 * This implements "left neighbor" mapping for bit-packed rows:
 *   - For bits 1..63: new_bit[j] = center_bit[j-1]
 *   - For bit 0:      new_bit[0] = left_bit[63]
 */
__device__ __forceinline__
std::uint64_t shift_left_with_carry(std::uint64_t center, std::uint64_t left)
{
    return (center << 1) | (left >> 63);
}

/**
 * Shift a word right by one bit, inserting bit 0 of 'right' into bit 63.
 *
 * This implements "right neighbor" mapping for bit-packed rows:
 *   - For bits 0..62: new_bit[j] = center_bit[j+1]
 *   - For bit 63:     new_bit[63] = right_bit[0]
 */
__device__ __forceinline__
std::uint64_t shift_right_with_carry(std::uint64_t center, std::uint64_t right)
{
    return (center >> 1) | ((right & 1ULL) << 63);
}

/**
 * Add one neighbor bitmask 'mask' to the bit-sliced counters (count2:count1:count0),
 * performing a modulo-8 increment at every bit position where 'mask' has a 1.
 *
 * For each bit position i:
 *   if mask[i] == 1:
 *        (count2[i], count1[i], count0[i]) := (count2[i], count1[i], count0[i]) + 1 (mod 8)
 *   else:
 *        unchanged
 *
 * Implemented entirely with bitwise operations, so there are no carries between
 * different bit positions.
 */
__device__ __forceinline__
void add_neighbor_mask(std::uint64_t mask,
                       std::uint64_t &count0,
                       std::uint64_t &count1,
                       std::uint64_t &count2)
{
    // First bit (LSB of the 3-bit counter)
    std::uint64_t carry1 = count0 & mask;  // positions where count0 and mask are both 1
    count0 ^= mask;                        // add mask to LSB (mod 2)

    // Second bit
    std::uint64_t carry2 = count1 & carry1; // carry from adding carry1 to count1
    count1 ^= carry1;                       // add carry1 to second bit (mod 2)

    // Third bit
    count2 ^= carry2;                      // add carry2 to third bit (mod 2), ignoring overflow
}

////////////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////////////

/**
 * Kernel that computes one Game of Life step on a bit-packed grid.
 *
 * Parameters:
 *   input        - pointer to input grid (device memory), bit-packed in uint64_t
 *   output       - pointer to output grid (device memory), bit-packed in uint64_t
 *   grid_dim     - number of cells per row/column (grid is grid_dim x grid_dim)
 *   words_per_row- number of 64-bit words per row (grid_dim / 64)
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,
                         int words_per_row)
{
    // Column index in units of 64-cell words
    int col_word = blockIdx.x * blockDim.x + threadIdx.x;
    // Row index (in cells)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= grid_dim || col_word >= words_per_row)
        return;

    // Row offset in number of 64-bit words
    std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row);
    const std::uint64_t* row_ptr = input + row_offset;

    // Current word (central)
    std::uint64_t c = row_ptr[col_word];

    // Neighbor words in the same row
    std::uint64_t c_left  = 0;
    std::uint64_t c_right = 0;
    if (col_word > 0)
        c_left = row_ptr[col_word - 1];
    if (col_word + 1 < words_per_row)
        c_right = row_ptr[col_word + 1];

    // Neighbor words in the row above
    std::uint64_t n      = 0;
    std::uint64_t n_left = 0;
    std::uint64_t n_right= 0;
    if (row > 0) {
        std::size_t row_above_offset = static_cast<std::size_t>(row - 1) * static_cast<std::size_t>(words_per_row);
        const std::uint64_t* row_above = input + row_above_offset;
        n = row_above[col_word];
        if (col_word > 0)
            n_left = row_above[col_word - 1];
        if (col_word + 1 < words_per_row)
            n_right = row_above[col_word + 1];
    }

    // Neighbor words in the row below
    std::uint64_t s      = 0;
    std::uint64_t s_left = 0;
    std::uint64_t s_right= 0;
    if (row + 1 < grid_dim) {
        std::size_t row_below_offset = static_cast<std::size_t>(row + 1) * static_cast<std::size_t>(words_per_row);
        const std::uint64_t* row_below = input + row_below_offset;
        s = row_below[col_word];
        if (col_word > 0)
            s_left = row_below[col_word - 1];
        if (col_word + 1 < words_per_row)
            s_right = row_below[col_word + 1];
    }

    // Compute the eight neighbor masks using shifts with cross-word carries.
    // Horizontal neighbors in current row
    std::uint64_t W  = shift_left_with_carry(c, c_left);   // left neighbors
    std::uint64_t E  = shift_right_with_carry(c, c_right); // right neighbors

    // Vertical neighbors
    std::uint64_t N  = n; // above
    std::uint64_t S  = s; // below

    // Diagonal neighbors
    std::uint64_t NW = shift_left_with_carry(n, n_left);    // above-left
    std::uint64_t NE = shift_right_with_carry(n, n_right);  // above-right
    std::uint64_t SW = shift_left_with_carry(s, s_left);    // below-left
    std::uint64_t SE = shift_right_with_carry(s, s_right);  // below-right

    // Bit-sliced neighbor count accumulators (modulo 8)
    std::uint64_t count0 = 0; // least significant bit
    std::uint64_t count1 = 0;
    std::uint64_t count2 = 0;

    // Add the eight neighbor masks to the counters
    add_neighbor_mask(NW, count0, count1, count2);
    add_neighbor_mask(N,  count0, count1, count2);
    add_neighbor_mask(NE, count0, count1, count2);
    add_neighbor_mask(W,  count0, count1, count2);
    add_neighbor_mask(E,  count0, count1, count2);
    add_neighbor_mask(SW, count0, count1, count2);
    add_neighbor_mask(S,  count0, count1, count2);
    add_neighbor_mask(SE, count0, count1, count2);

    // Determine where neighbor_count == 3 and neighbor_count == 2
    std::uint64_t neighbors_eq_3 =  count0 &  count1 & ~count2;       // 011b
    std::uint64_t neighbors_eq_2 = (~count0) &  count1 & ~count2;     // 010b

    // Apply Conway's Game of Life rules:
    // - A dead cell with exactly 3 neighbors becomes alive.
    // - A live cell with 2 or 3 neighbors survives.
    //   => new = (neighbors == 3) OR (alive & neighbors == 2)
    std::uint64_t next = neighbors_eq_3 | (c & neighbors_eq_2);

    // Store the result
    output[row_offset + static_cast<std::size_t>(col_word)] = next;
}

////////////////////////////////////////////////////////////////////////////////
// Host function
////////////////////////////////////////////////////////////////////////////////

/**
 * Host function that launches the CUDA kernel to compute one step of
 * Conway's Game of Life on a bit-packed grid.
 *
 * Parameters:
 *   input           - device pointer to input grid (bit-packed uint64_t)
 *   output          - device pointer to output grid (bit-packed uint64_t)
 *   grid_dimensions - number of cells per row/column (grid_dimensions x grid_dimensions)
 *
 * Notes:
 *   - 'input' and 'output' must be allocated with cudaMalloc and must not overlap.
 *   - No host-device synchronization is performed here; the caller is responsible
 *     for any necessary synchronization (e.g., cudaDeviceSynchronize).
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // Each row has grid_dimensions cells, packed into 64-bit words.
    // grid_dimensions is guaranteed to be a power of 2, so this is exact.
    int words_per_row = grid_dimensions >> 6; // divide by 64

    // Choose a reasonably sized block for modern NVIDIA GPUs.
    // 32 x 8 = 256 threads per block.
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;

    dim3 block_dim(BLOCK_X, BLOCK_Y);
    dim3 grid_dim(
        (words_per_row   + BLOCK_X - 1) / BLOCK_X,
        (grid_dimensions + BLOCK_Y - 1) / BLOCK_Y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output,
                                                 grid_dimensions,
                                                 words_per_row);
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
