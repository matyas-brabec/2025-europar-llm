#include <cstdint>
#include <cuda_runtime.h>

// Bring std::uint64_t into the global namespace for brevity.
using std::uint64_t;

/*
 * Bit-sliced neighbor-count accumulator.
 *
 * We maintain three 64-bit bitplanes (c0, c1, c2) that encode the neighbor
 * count for each cell modulo 8:
 *   count = c0*1 + c1*2 + c2*4  (per bit position)
 *
 * Adding a bitboard "bits" (each bit is 0 or 1) corresponds to adding 1 to
 * the count for every position where bits has a 1. This is done with a
 * 3-bit ripple-carry adder, truncated to 3 bits (i.e., modulo 8). This is
 * sufficient because the Game of Life only differentiates neighbor counts
 * of exactly 2 or 3; 8 neighbors (binary 1000) aliases to 0 modulo 8, and
 * both 0 and 8 cause the cell to be dead in the next generation.
 */
__device__ __forceinline__
void add_bitboard(uint64_t bits, uint64_t &c0, uint64_t &c1, uint64_t &c2)
{
    uint64_t t  = c0 & bits;  // carry from bit 0 to bit 1
    c0 ^= bits;               // new bit 0
    uint64_t t2 = c1 & t;     // carry from bit 1 to bit 2
    c1 ^= t;                  // new bit 1
    c2 ^= t2;                 // new bit 2 (carry into bit 2)
}

constexpr int GOL_BLOCK_DIM_X = 32;
constexpr int GOL_BLOCK_DIM_Y = 8;
constexpr int GOL_BLOCK_SIZE  = GOL_BLOCK_DIM_X * GOL_BLOCK_DIM_Y;

/*
 * CUDA kernel implementing one step of Conway's Game of Life on a bit-packed grid.
 *
 * Grid representation:
 *   - The universe is a grid_size x grid_size square grid of cells (grid_size is a power of 2).
 *   - Each row is stored as words_per_row 64-bit words.
 *   - Within each word, bit 0 is the leftmost cell, bit 63 is the rightmost cell.
 *
 * Each thread processes one 64-bit word (64 cells) at coordinates (row, colWord).
 *
 * For every word, we load up to 9 words from the input:
 *   - Previous row: left, center, right
 *   - Current row:  left, center, right
 *   - Next row:     left, center, right
 *
 * For border rows/columns, any word outside the grid is treated as 0 (all cells dead).
 *
 * From these 9 words we construct 8 bitboards representing the 8 neighbor directions:
 *   - up-left, up, up-right
 *   - left, right
 *   - down-left, down, down-right
 *
 * Then we accumulate neighbor counts modulo 8 in three bitplanes (c0, c1, c2).
 * Finally, for each bit we apply the Game of Life rules:
 *   - A live cell survives if it has exactly 2 or 3 neighbors.
 *   - A dead cell becomes live if it has exactly 3 neighbors.
 *
 * Using the bit-sliced representation:
 *   neighbors == 2 <=> c2=0, c1=1, c0=0  -> (~c2 & c1 & ~c0)
 *   neighbors == 3 <=> c2=0, c1=1, c0=1  -> (~c2 & c1 &  c0)
 *
 * New state bit = (neighbors == 3) OR (alive AND neighbors == 2).
 */
__launch_bounds__(GOL_BLOCK_SIZE)
__global__
void game_of_life_kernel(const uint64_t* __restrict__ input,
                         uint64_t* __restrict__ output,
                         int grid_size,
                         int words_per_row)
{
    // 2D thread coordinates in the word grid.
    int col = blockIdx.x * blockDim.x + threadIdx.x; // word index within row
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (row >= grid_size || col >= words_per_row)
        return;

    const int idx = row * words_per_row + col;

    // Flags indicating presence of neighbors inside the grid boundaries.
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_size);

    // Load center/left/right words for the current row.
    const uint64_t* row_ptr = input + row * words_per_row;
    uint64_t curC = row_ptr[col];
    uint64_t curL = has_left  ? row_ptr[col - 1] : 0ull;
    uint64_t curR = has_right ? row_ptr[col + 1] : 0ull;

    // Load center/left/right words for the previous row (if any).
    uint64_t prevC = 0ull, prevL = 0ull, prevR = 0ull;
    if (has_up) {
        const uint64_t* prev_row = input + (row - 1) * words_per_row;
        prevC = prev_row[col];
        prevL = has_left  ? prev_row[col - 1] : 0ull;
        prevR = has_right ? prev_row[col + 1] : 0ull;
    }

    // Load center/left/right words for the next row (if any).
    uint64_t nextC = 0ull, nextL = 0ull, nextR = 0ull;
    if (has_down) {
        const uint64_t* next_row = input + (row + 1) * words_per_row;
        nextC = next_row[col];
        nextL = has_left  ? next_row[col - 1] : 0ull;
        nextR = has_right ? next_row[col + 1] : 0ull;
    }

    // Construct neighbor bitboards.
    //
    // For horizontal shifts within a row, we have to bring in the boundary bit
    // from the adjacent word (left/right). Example for left neighbors:
    //
    //   mid_l = (curC << 1) | (curL >> 63);
    //
    // For bit position i (0-based) in curC:
    //   - (curC << 1) puts bit (i-1) from curC into position i (for i > 0),
    //     with bit 0 receiving 0.
    //   - (curL >> 63) provides bit 63 from the left word into position 0.
    //
    // This effectively builds, at each bit position, the value of the neighbor
    // to the immediate left. Similar logic applies to the right shift.

    // Neighbors from the row above.
    const uint64_t up_c = prevC;                                       // N
    const uint64_t up_l = (prevC << 1) | (prevL >> 63);                // NW
    const uint64_t up_r = (prevC >> 1) | (prevR << 63);                // NE

    // Neighbors from the same row (left & right).
    const uint64_t mid_l = (curC << 1) | (curL >> 63);                 // W
    const uint64_t mid_r = (curC >> 1) | (curR << 63);                 // E

    // Neighbors from the row below.
    const uint64_t down_c = nextC;                                     // S
    const uint64_t down_l = (nextC << 1) | (nextL >> 63);              // SW
    const uint64_t down_r = (nextC >> 1) | (nextR << 63);              // SE

    // Accumulate neighbor counts modulo 8 in bitplanes c0, c1, c2.
    uint64_t c0 = 0ull, c1 = 0ull, c2 = 0ull;

    add_bitboard(up_l,   c0, c1, c2);
    add_bitboard(up_c,   c0, c1, c2);
    add_bitboard(up_r,   c0, c1, c2);
    add_bitboard(mid_l,  c0, c1, c2);
    add_bitboard(mid_r,  c0, c1, c2);
    add_bitboard(down_l, c0, c1, c2);
    add_bitboard(down_c, c0, c1, c2);
    add_bitboard(down_r, c0, c1, c2);

    // Determine which cells have exactly 2 or 3 neighbors.
    //
    // neighbors == 2  -> binary 010 -> (c2 == 0, c1 == 1, c0 == 0)
    // neighbors == 3  -> binary 011 -> (c2 == 0, c1 == 1, c0 == 1)
    const uint64_t not_c2 = ~c2;
    const uint64_t eq2    = not_c2 &  c1 & ~c0;
    const uint64_t eq3    = not_c2 &  c1 &  c0;

    // Current live cells.
    const uint64_t alive  = curC;

    // Game of Life update rule:
    //   - A cell is alive in the next generation if:
    //       * it has 3 neighbors (birth or survival), OR
    //       * it is currently alive and has 2 neighbors (survival).
    const uint64_t next_word = eq3 | (alive & eq2);

    output[idx] = next_word;
}

/*
 * Host function to launch a single Game of Life step on the GPU.
 *
 * Parameters:
 *   - input:  Device pointer to the bit-packed input grid.
 *   - output: Device pointer to the bit-packed output grid.
 *   - grid_dimensions: Width/height of the square grid (power of 2, > 512).
 *
 * Notes:
 *   - The grid is assumed to be square and its dimension is a power of two,
 *     so it is guaranteed to be divisible by 64. Therefore, each row consists
 *     of exactly (grid_dimensions / 64) 64-bit words.
 *   - The function does not perform any synchronization or error checking;
 *     the caller is responsible for that if needed.
 */
void run_game_of_life(const uint64_t* input,
                      uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    dim3 blockDim(GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y);
    dim3 gridDim(
        (words_per_row   + blockDim.x - 1) / blockDim.x,
        (grid_dimensions + blockDim.y - 1) / blockDim.y
    );

    game_of_life_kernel<<<gridDim, blockDim>>>(input, output,
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
