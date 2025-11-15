#include <cstdint>
#include <cuda_runtime.h>

/*
 * CUDA implementation of a single step of Conway's Game of Life on a
 * bit-packed square grid.
 *
 * - The grid is grid_dimensions x grid_dimensions cells.
 * - Each std::uint64_t encodes 64 consecutive cells in a row (bit 0 is the
 *   left-most cell in the word, bit 63 the right-most).
 * - Outside-grid cells are considered dead.
 * - Each CUDA thread processes exactly one 64-bit word (64 cells).
 *
 * Neighbor counting is done in a bit-parallel fashion:
 * - For each of the 8 neighbor directions we compute a 64-bit mask with
 *   1s where that neighbor is alive.
 * - We maintain a 4-bit counter per cell using four 64-bit words (c0, c1,
 *   c2, c3), updated for each neighbor bit-mask with a ripple-carry style
 *   increment implemented using only bitwise AND/XOR. This avoids scalar
 *   per-cell arithmetic and keeps all operations SIMD-within-a-register.
 *
 * The Life rules are applied using the final 4-bit neighbor count:
 *   - new cell = 1 iff (neighbors == 3) or (cell == 1 and neighbors == 2)
 *   expressed as:
 *      new = (neighbors == 3) | (alive & (neighbors == 2))
 *   where "neighbors == 2/3" are computed from (c3,c2,c1,c0).
 */

////////////////////////////////////////////////////////////////
// Device helper: increment 4-bit per-lane counters by a mask //
////////////////////////////////////////////////////////////////

__device__ __forceinline__
void add_neighbor_bits(std::uint64_t n,
                       std::uint64_t &c0,
                       std::uint64_t &c1,
                       std::uint64_t &c2,
                       std::uint64_t &c3)
{
    // For each bit lane where n has 1, add +1 to the 4-bit counter
    // encoded across (c3 c2 c1 c0) using ripple-carry logic.
    std::uint64_t m = n;

    std::uint64_t carry = m & c0;
    c0 ^= m;
    m = carry;

    carry = m & c1;
    c1 ^= m;
    m = carry;

    carry = m & c2;
    c2 ^= m;
    m = carry;

    c3 ^= m;
}

//////////////////////////////////////////////////////
// Kernel: one thread processes one 64-bit word     //
//////////////////////////////////////////////////////

__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dim,            // number of cells per row/column
                         int words_per_row,       // grid_dim / 64, power of two
                         unsigned int num_words,  // total number of 64-bit words
                         int row_shift_bits)      // log2(words_per_row)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    // Compute row and word-column index from flat word index.
    // Because words_per_row is a power of two, we can use shifts & masks.
    int row = static_cast<int>(idx >> row_shift_bits);
    int col = static_cast<int>(idx & (static_cast<unsigned int>(words_per_row) - 1));

    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim);
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    // Load center word
    const std::uint64_t cur = input[idx];

    // Load horizontal neighbors (same row)
    const std::uint64_t wl = has_left  ? input[idx - 1] : 0ull;
    const std::uint64_t wr = has_right ? input[idx + 1] : 0ull;

    // Load vertical and diagonal neighbors
    std::uint64_t up   = 0ull;
    std::uint64_t upL  = 0ull;
    std::uint64_t upR  = 0ull;
    std::uint64_t dn   = 0ull;
    std::uint64_t dnL  = 0ull;
    std::uint64_t dnR  = 0ull;

    if (has_up) {
        const unsigned int up_idx = idx - static_cast<unsigned int>(words_per_row);
        up  = input[up_idx];
        upL = has_left  ? input[up_idx - 1] : 0ull;
        upR = has_right ? input[up_idx + 1] : 0ull;
    }

    if (has_down) {
        const unsigned int dn_idx = idx + static_cast<unsigned int>(words_per_row);
        dn  = input[dn_idx];
        dnL = has_left  ? input[dn_idx - 1] : 0ull;
        dnR = has_right ? input[dn_idx + 1] : 0ull;
    }

    // Build 8 neighbor bitboards (one per direction) for this 64-cell chunk.
    // For horizontal and diagonal neighbors, cross-word bits are handled by
    // combining shifts within the word with bits from adjacent words.
    //
    // Left and right neighbors in the same row:
    const std::uint64_t nL  = (cur << 1) | (wl >> 63);              // left neighbors
    const std::uint64_t nR  = (cur >> 1) | (wr << 63);              // right neighbors

    // Vertical neighbors (no horizontal shift):
    const std::uint64_t nU  = up;                                   // up neighbors
    const std::uint64_t nD  = dn;                                   // down neighbors

    // Diagonal neighbors:
    const std::uint64_t nUL = (up << 1) | (upL >> 63);              // up-left
    const std::uint64_t nUR = (up >> 1) | (upR << 63);              // up-right
    const std::uint64_t nDL = (dn << 1) | (dnL >> 63);              // down-left
    const std::uint64_t nDR = (dn >> 1) | (dnR << 63);              // down-right

    // Accumulate neighbor counts using 4-bit per-lane counters.
    std::uint64_t c0 = 0ull;
    std::uint64_t c1 = 0ull;
    std::uint64_t c2 = 0ull;
    std::uint64_t c3 = 0ull;

    add_neighbor_bits(nUL, c0, c1, c2, c3);
    add_neighbor_bits(nU,  c0, c1, c2, c3);
    add_neighbor_bits(nUR, c0, c1, c2, c3);
    add_neighbor_bits(nL,  c0, c1, c2, c3);
    add_neighbor_bits(nR,  c0, c1, c2, c3);
    add_neighbor_bits(nDL, c0, c1, c2, c3);
    add_neighbor_bits(nD,  c0, c1, c2, c3);
    add_neighbor_bits(nDR, c0, c1, c2, c3);

    // Now (c3 c2 c1 c0) encodes the neighbor count per bit (0..8).
    // We need:
    //   - eq2: neighbors == 2  -> 0010b
    //   - eq3: neighbors == 3  -> 0011b
    //
    // eq2 = ~c3 & ~c2 &  c1 & ~c0
    // eq3 = ~c3 & ~c2 &  c1 &  c0
    const std::uint64_t not_c0 = ~c0;
    const std::uint64_t not_c2 = ~c2;
    const std::uint64_t not_c3 = ~c3;

    const std::uint64_t base_2_or_3 = not_c3 & not_c2 & c1;
    const std::uint64_t eq3 = base_2_or_3 & c0;
    const std::uint64_t eq2 = base_2_or_3 & not_c0;

    // Apply Conway's rules:
    //   new = (neighbors == 3) | (alive & (neighbors == 2))
    const std::uint64_t alive   = cur;
    const std::uint64_t survive = alive & eq2;
    const std::uint64_t born    = eq3;
    const std::uint64_t next    = survive | born;

    output[idx] = next;
}

///////////////////////////////////////////////////////////////
// Host function: launches one Game of Life update step      //
///////////////////////////////////////////////////////////////

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is a power of two and >= 512 by problem statement.
    // Each row has grid_dimensions cells, packed into 64-bit words.
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // words_per_row is also a power of two; compute log2(words_per_row)
    int row_shift_bits = 0;
    int tmp = words_per_row;
    while (tmp > 1) {
        tmp >>= 1;
        ++row_shift_bits;
    }

    const unsigned int num_words =
        static_cast<unsigned int>(grid_dimensions) *
        static_cast<unsigned int>(words_per_row);

    // Choose a block size that gives good occupancy on modern GPUs.
    const int block_size = 256;
    const unsigned int grid_size =
        (num_words + static_cast<unsigned int>(block_size) - 1u) /
        static_cast<unsigned int>(block_size);

    game_of_life_kernel<<<grid_size, block_size>>>(
        input, output,
        grid_dimensions,
        words_per_row,
        num_words,
        row_shift_bits
    );

    // Synchronization is intentionally omitted; the caller is responsible
    // for any required cudaDeviceSynchronize() or stream synchronization.
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
