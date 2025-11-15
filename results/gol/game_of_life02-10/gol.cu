#include <cstdint>
#include <cuda_runtime.h>

/*
 * Highly optimized CUDA implementation of one step of Conway's Game of Life
 * on a bit-packed grid. Each uint64_t encodes 64 consecutive cells in a row.
 *
 * Grid:
 *   - Square of size N x N, N is a power of 2, N > 512.
 *   - Each row is stored as N/64 64-bit words.
 *   - Bit 0 of a word corresponds to the left-most cell in that 64-cell block.
 *
 * Boundary conditions:
 *   - Cells outside the grid are treated as dead (0).
 *
 * Layout:
 *   - input[row * wordsPerRow + wordIndex] -> 64 cells of that row.
 *   - output has the same layout.
 *
 * Thread mapping:
 *   - 1D grid over all words: totalWords = N * (N/64)
 *   - Each thread processes one 64-bit word (64 cells) for its row.
 *
 * Neighbor counting:
 *   - For each word, we load up to 9 words (3x3 surrounding region: left/center/right
 *     for row-1, row, row+1).
 *   - From these we construct 8 64-bit masks for the 8 neighbor directions:
 *       top-left, top, top-right,
 *       left,           right,
 *       bottom-left, bottom, bottom-right.
 *   - For each bit position (cell) in the word, these masks provide the 8 neighbor
 *     bits in parallel.
 *   - Neighbor counts [0..8] for each bit position are computed using a bit-parallel
 *     binary accumulator (4 bit-planes) with ripple-carry addition of the 8 masks.
 *   - From the 4 bit-planes we derive masks for "exactly 2 neighbors" and
 *     "exactly 3 neighbors" (per bit position).
 *
 * Game of Life rule (per cell, per bit position):
 *   - next = 1 if (neighbors == 3) or (alive && neighbors == 2)
 *   - equivalently: next = eq3 | (eq2 & alive)
 *     where eq2 and eq3 are bit masks (bit=1 if that lane has 2 or 3 neighbors).
 */

namespace {

/*
 * Increment per-bit binary counter by 0 or 1, for 64 independent lanes in parallel.
 *
 * The neighbor counts are stored in four 64-bit planes (b3 b2 b1 b0), where each bit
 * position represents the binary count for that cell:
 *
 *    count = b0 + 2*b1 + 4*b2 + 8*b3   (per bit/lane)
 *
 * mask:  64-bit value whose bits are 0/1 indicating whether to add 1 in each lane.
 */
__device__ __forceinline__
void add_neighbor_mask(std::uint64_t mask,
                       std::uint64_t &b0,
                       std::uint64_t &b1,
                       std::uint64_t &b2,
                       std::uint64_t &b3)
{
    // Add "mask" (0 or 1 per bit) to the 4-bit counter (b3 b2 b1 b0) with ripple carry.
    // This is standard binary addition done in parallel across 64 lanes.
    std::uint64_t c = b0 & mask;
    b0 ^= mask;

    std::uint64_t c1 = b1 & c;
    b1 ^= c;

    std::uint64_t c2 = b2 & c1;
    b2 ^= c1;

    b3 ^= c2;
}

/*
 * CUDA kernel: perform one Game of Life step on a bit-packed square grid.
 *
 * input, output:
 *   - Device pointers to N x N grid packed into uint64_t words.
 *   - Each row has wordsPerRow = N / 64 words.
 *
 * n:
 *   - Grid dimension (N).
 *
 * wordsPerRow:
 *   - Number of 64-bit words per row (N/64), guaranteed to be a power of two.
 *
 * log2WordsPerRow:
 *   - log2(wordsPerRow); used to compute row and word index from the global word index.
 */
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int n,
                         int wordsPerRow,
                         int log2WordsPerRow)
{
    // Global index over all words in the grid.
    unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) *
        static_cast<unsigned long long>(blockDim.x) +
        static_cast<unsigned long long>(threadIdx.x);

    unsigned long long totalWords =
        static_cast<unsigned long long>(n) *
        static_cast<unsigned long long>(wordsPerRow);

    if (idx >= totalWords) {
        return;
    }

    // Derive row and word-in-row from idx.
    // Since wordsPerRow is a power of two, this is a shift and an AND.
    unsigned int row = static_cast<unsigned int>(idx >> log2WordsPerRow);
    unsigned int w   = static_cast<unsigned int>(idx & (static_cast<unsigned long long>(wordsPerRow) - 1u));

    // Pointer to the start of this row in the input array.
    const std::uint64_t* rowPtr = input + (static_cast<std::size_t>(row) * static_cast<std::size_t>(wordsPerRow));

    // Load center row words for this position.
    std::uint64_t midC = rowPtr[w];
    std::uint64_t midL = (w > 0u) ? rowPtr[w - 1u] : 0ull;
    std::uint64_t midR = (w + 1u < static_cast<unsigned int>(wordsPerRow)) ? rowPtr[w + 1u] : 0ull;

    // Load words from the row above (if any).
    std::uint64_t upC = 0ull, upL = 0ull, upR = 0ull;
    if (row > 0u) {
        const std::uint64_t* rowUpPtr = rowPtr - wordsPerRow;
        upC = rowUpPtr[w];
        if (w > 0u) {
            upL = rowUpPtr[w - 1u];
        }
        if (w + 1u < static_cast<unsigned int>(wordsPerRow)) {
            upR = rowUpPtr[w + 1u];
        }
    }

    // Load words from the row below (if any).
    std::uint64_t dnC = 0ull, dnL = 0ull, dnR = 0ull;
    if (row + 1u < static_cast<unsigned int>(n)) {
        const std::uint64_t* rowDnPtr = rowPtr + wordsPerRow;
        dnC = rowDnPtr[w];
        if (w > 0u) {
            dnL = rowDnPtr[w - 1u];
        }
        if (w + 1u < static_cast<unsigned int>(wordsPerRow)) {
            dnR = rowDnPtr[w + 1u];
        }
    }

    // Build 8 neighbor bit masks for this word.
    //
    // For each cell in this word (bit position j, 0..63), we want:
    //   - top-left:    (row-1, col-1)
    //   - top:         (row-1, col)
    //   - top-right:   (row-1, col+1)
    //   - left:        (row,   col-1)
    //   - right:       (row,   col+1)
    //   - bottom-left: (row+1, col-1)
    //   - bottom:      (row+1, col)
    //   - bottom-right:(row+1, col+1)
    //
    // Horizontal neighbors are constructed by shifting the center word and
    // incorporating a carry bit from the adjacent word via (<<1)|(prev>>63)
    // or (>>1)|(next<<63). At true grid boundaries we have zero-filled neighbors.

    std::uint64_t n0 = (upC << 1) | (upL >> 63);   // top-left
    std::uint64_t n1 = upC;                        // top
    std::uint64_t n2 = (upC >> 1) | (upR << 63);   // top-right

    std::uint64_t n3 = (midC << 1) | (midL >> 63); // left
    std::uint64_t n4 = (midC >> 1) | (midR << 63); // right

    std::uint64_t n5 = (dnC << 1) | (dnL >> 63);   // bottom-left
    std::uint64_t n6 = dnC;                        // bottom
    std::uint64_t n7 = (dnC >> 1) | (dnR << 63);   // bottom-right

    // Accumulate neighbor counts in four bit-planes (b3 b2 b1 b0).
    std::uint64_t b0 = 0ull;
    std::uint64_t b1 = 0ull;
    std::uint64_t b2 = 0ull;
    std::uint64_t b3 = 0ull;

    add_neighbor_mask(n0, b0, b1, b2, b3);
    add_neighbor_mask(n1, b0, b1, b2, b3);
    add_neighbor_mask(n2, b0, b1, b2, b3);
    add_neighbor_mask(n3, b0, b1, b2, b3);
    add_neighbor_mask(n4, b0, b1, b2, b3);
    add_neighbor_mask(n5, b0, b1, b2, b3);
    add_neighbor_mask(n6, b0, b1, b2, b3);
    add_neighbor_mask(n7, b0, b1, b2, b3);

    // Now neighbor_count per lane = b0 + 2*b1 + 4*b2 + 8*b3 (0..8).
    // We need masks for "exactly 2" and "exactly 3".
    //
    // Binary patterns:
    //   2 -> 0b0010: b3=0, b2=0, b1=1, b0=0
    //   3 -> 0b0011: b3=0, b2=0, b1=1, b0=1
    //
    // eq2 = (~b0) & b1 & (~b2) & (~b3)
    // eq3 =  b0  & b1 & (~b2) & (~b3)

    std::uint64_t not_b0 = ~b0;
    std::uint64_t not_b2 = ~b2;
    std::uint64_t not_b3 = ~b3;

    std::uint64_t eq2 = not_b0 & b1 & not_b2 & not_b3;
    std::uint64_t eq3 = b0 & b1 & not_b2 & not_b3;

    // Current alive cells for this word.
    std::uint64_t alive = midC;

    // Game of Life rule per bit:
    //   - A cell is alive in the next generation if:
    //       neighbors == 3   OR
    //       (alive && neighbors == 2)
    //
    //   => next = eq3 | (eq2 & alive)
    std::uint64_t next = eq3 | (eq2 & alive);

    // Store result at the same global word index.
    output[idx] = next;
}

} // anonymous namespace

/*
 * Host wrapper: run one Game of Life step on a bit-packed square grid.
 *
 * input:
 *   - Device pointer to input grid, bit-packed into std::uint64_t.
 *
 * output:
 *   - Device pointer to output grid, same layout as input.
 *
 * grid_dimensions:
 *   - N (width and height), a power of two, N > 512.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int n = grid_dimensions;
    const int BITS_PER_WORD = 64;

    // Number of 64-bit words per row. N is guaranteed to be a multiple of 64.
    const int wordsPerRow = n / BITS_PER_WORD;

    // Compute log2(wordsPerRow). wordsPerRow is a power of two.
    int log2WordsPerRow = 0;
    {
        int tmp = wordsPerRow;
        while ((tmp >>= 1) != 0) {
            ++log2WordsPerRow;
        }
    }

    // Total number of 64-bit words in the grid.
    unsigned long long totalWords =
        static_cast<unsigned long long>(n) *
        static_cast<unsigned long long>(wordsPerRow);

    // Choose a reasonable block size for modern GPUs.
    const int blockSize = 256;

    // Number of blocks needed (round up).
    unsigned long long gridSizeULL = (totalWords + blockSize - 1ull) / blockSize;

    // gridSizeULL is guaranteed to fit in 32 bits under the problem constraints.
    dim3 grid(static_cast<unsigned int>(gridSizeULL), 1, 1);
    dim3 block(blockSize, 1, 1);

    // Launch the kernel. Synchronization and error checking are done by the caller.
    game_of_life_kernel<<<grid, block>>>(input, output, n, wordsPerRow, log2WordsPerRow);
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
