#include <cstdint>
#include <cuda_runtime.h>

// Bring std::uint64_t into the global namespace for convenience.
using std::uint64_t;

/*
 * CUDA kernel implementing one step of Conway's Game of Life on a bit-packed grid.
 *
 * Each thread processes exactly one 64-bit word, which encodes 64 horizontally
 * consecutive cells in a single row. The grid is square with side length grid_dim,
 * and each row contains words_per_row 64-bit words.
 *
 * The input/output layout:
 *   - Row-major order.
 *   - Row r, word index w (0-based) is stored at input[r * words_per_row + w].
 *
 * Boundary handling:
 *   - Cells outside the grid are treated as dead (0).
 *   - For each word, we load up to 8 neighboring words (left/right, top/bottom
 *     and their diagonals). Missing neighbors (outside the grid) are treated as 0.
 *
 * Neighborhood computation:
 *   - For each thread's central word (wC), we build eight 64-bit masks:
 *       NW, N, NE, W, E, SW, S, SE
 *     such that for every bit position i (0..63), bit i in each mask
 *     corresponds to the respective neighbor of cell (row, col_bit=i).
 *
 *   - Cross-word neighbors for bit 0 and bit 63 are handled by incorporating
 *     bits from the left/right neighboring words (and their diagonal neighbors)
 *     into these masks.
 *
 * Per-cell update:
 *   - We iterate over all 64 bits in the word. For each bit, we:
 *       * Extract the 1-bit values of the eight neighbors from the masks.
 *       * Pack these eight bits into an 8-bit integer.
 *       * Use __popc() to count the number of alive neighbors (0..8).
 *       * Apply the Game of Life rules to determine the new state.
 *   - The bit loop is unrolled by the compiler (hinted by #pragma unroll).
 *
 * This approach keeps global memory accesses fully coalesced and leverages
 * fast bit operations and the __popc intrinsic for high performance.
 */
__global__ void game_of_life_kernel(const uint64_t* __restrict__ input,
                                    uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = words_per_row * grid_dim;
    if (idx >= total_words) {
        return;
    }

    // Compute row and column (word index within the row).
    int row = idx / words_per_row;
    int col = idx - row * words_per_row;

    // Flags for boundary checks.
    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < grid_dim);
    const bool has_left   = (col > 0);
    const bool has_right  = (col + 1 < words_per_row);

    // Load the central word (always valid).
    const uint64_t wC = input[idx];

    // Load neighbor words with boundary checks.
    const uint64_t wL  = has_left   ? input[idx - 1]                : 0ull;
    const uint64_t wR  = has_right  ? input[idx + 1]                : 0ull;
    const uint64_t wT  = has_top    ? input[idx - words_per_row]    : 0ull;
    const uint64_t wB  = has_bottom ? input[idx + words_per_row]    : 0ull;
    const uint64_t wTL = (has_top && has_left)
                         ? input[idx - words_per_row - 1]           : 0ull;
    const uint64_t wTR = (has_top && has_right)
                         ? input[idx - words_per_row + 1]           : 0ull;
    const uint64_t wBL = (has_bottom && has_left)
                         ? input[idx + words_per_row - 1]           : 0ull;
    const uint64_t wBR = (has_bottom && has_right)
                         ? input[idx + words_per_row + 1]           : 0ull;

    // Fast early exit: if the entire 3x3 block of words around this word is zero,
    // then no cell in this word can be born or survive (all neighbors are dead).
    if ((wC | wL | wR | wT | wB | wTL | wTR | wBL | wBR) == 0ull) {
        output[idx] = 0ull;
        return;
    }

    // Build 64-bit neighbor masks for each of the 8 directions.
    // Vertical neighbors (no horizontal shift needed).
    uint64_t N  = wT;
    uint64_t S  = wB;

    // Horizontal neighbors (within the same row) with cross-word handling.
    uint64_t W  = (wC << 1);
    uint64_t E  = (wC >> 1);

    // Diagonal neighbors (top and bottom rows shifted horizontally).
    uint64_t NW = (wT << 1);
    uint64_t NE = (wT >> 1);
    uint64_t SW = (wB << 1);
    uint64_t SE = (wB >> 1);

    // Inject cross-word bits for 0th and 63rd bit neighbors when applicable.
    if (has_left) {
        const uint64_t left_bit_C  = (wL  >> 63);               // For W mask bit 0.
        const uint64_t left_bit_TL = (wTL >> 63);               // For NW mask bit 0.
        const uint64_t left_bit_BL = (wBL >> 63);               // For SW mask bit 0.
        W  |= left_bit_C;
        NW |= left_bit_TL;
        SW |= left_bit_BL;
    }

    if (has_right) {
        const uint64_t right_bit_C  = (wR  & 1ull) << 63;       // For E mask bit 63.
        const uint64_t right_bit_TR = (wTR & 1ull) << 63;       // For NE mask bit 63.
        const uint64_t right_bit_BR = (wBR & 1ull) << 63;       // For SE mask bit 63.
        E  |= right_bit_C;
        NE |= right_bit_TR;
        SE |= right_bit_BR;
    }

    // Current word state and neighbor masks; we'll shift these right
    // as we iterate over bits from LSB (bit 0) to MSB (bit 63).
    uint64_t cur = wC;
    uint64_t nw  = NW;
    uint64_t n   = N;
    uint64_t ne  = NE;
    uint64_t w   = W;
    uint64_t e   = E;
    uint64_t sw  = SW;
    uint64_t s   = S;
    uint64_t se  = SE;

    uint64_t result = 0ull;

    // Process each of the 64 bits in this word.
    // For each bit position:
    //   - The LSB of each mask/register corresponds to that cell's neighbor/state.
    //   - We construct an 8-bit mask of neighbors and use __popc() to count them.
    //   - Apply Game of Life rules and set the corresponding bit in result.
#pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        const unsigned int cell = static_cast<unsigned int>(cur & 1ull);

        // Pack the eight neighbor bits into a single 8-bit mask.
        const unsigned int neighbor_mask =
            static_cast<unsigned int>(
                (nw & 1ull) |
                ((n  & 1ull) << 1) |
                ((ne & 1ull) << 2) |
                ((w  & 1ull) << 3) |
                ((e  & 1ull) << 4) |
                ((sw & 1ull) << 5) |
                ((s  & 1ull) << 6) |
                ((se & 1ull) << 7));

        // Count the alive neighbors using __popc (operates on 32-bit words).
        const unsigned int neighbors = __popc(neighbor_mask);

        // Game of Life rules (branchless):
        //   - A cell is born if it has exactly 3 neighbors.
        //   - A cell survives if it has exactly 2 neighbors and is currently alive.
        const unsigned int eq2 = (neighbors == 2u);
        const unsigned int eq3 = (neighbors == 3u);
        const unsigned int new_state = (eq3 | (eq2 & cell));  // 0 or 1

        // Set the bit in the result word.
        result |= (static_cast<uint64_t>(new_state) << bit);

        // Shift all registers right by one bit to move to the next cell.
        cur >>= 1;
        nw  >>= 1;
        n   >>= 1;
        ne  >>= 1;
        w   >>= 1;
        e   >>= 1;
        sw  >>= 1;
        s   >>= 1;
        se  >>= 1;
    }

    // Store the updated word.
    output[idx] = result;
}

/*
 * Host wrapper function.
 *
 * Executes one step of Conway's Game of Life on a square grid whose cells
 * are stored bit-packed in 64-bit words on the device.
 *
 * Parameters:
 *   - input           : Device pointer to the current grid state.
 *   - output          : Device pointer to the next grid state.
 *   - grid_dimensions : Width and height of the square grid (power of two,
 *                       > 512, and divisible by 64).
 *
 * Layout:
 *   - Each row has (grid_dimensions / 64) 64-bit words.
 *   - Both input and output are assumed to be allocated with cudaMalloc().
 *
 * This function only launches the kernel; any necessary synchronization
 * (e.g., cudaDeviceSynchronize) is the responsibility of the caller.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const int total_words   = words_per_row * grid_dimensions;

    const int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel. The pointers are already device pointers
    // allocated with cudaMalloc, so we pass them directly.
    game_of_life_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output),
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
