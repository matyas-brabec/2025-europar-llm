// game_of_life.cu
//
// High-performance CUDA implementation of a single Conway's Game of Life step
// on a bit-packed grid using 64-bit words. Each thread processes one
// std::uint64_t, i.e., 64 cells in a row.
//
// Grid assumptions:
// - Square grid of size N x N, N is a power of 2 and N > 512.
// - Cells outside the grid are considered dead (0).
// - Grid is stored row-major, bit-packed:
//      - Each std::uint64_t stores 64 consecutive cells in a single row.
//      - Bit i (0 <= i < 64) of word w corresponds to column w*64 + i.
//
// Encoding:
// - 0 = dead cell
// - 1 = live cell
//
// Neighbor computation:
// - For a given word C at (row r, word-column c), we load up to 8 neighbor words:
//       UL, U, UR
//       L,  C, R
//       DL, D, DR
//   where indices outside the grid are treated as zeros.
// - We construct eight 64-bit neighbor bitfields (one for each direction) such
//   that for bit position j in the center word, the j-th bit of each neighbor
//   bitfield corresponds to that neighbor's state for the cell at (r, c*64 + j).
// - Horizontal neighbors that cross 64-bit word boundaries (bit 0 and bit 63)
//   are handled via bit shifts and the neighboring words.
//
// Neighbor count per bit (0..8) is accumulated in a bit-sliced fashion:
// - We maintain four 64-bit masks count0, count1, count2, count3 that represent
//   the 4-bit binary neighbor count per cell (LSB to MSB).
// - For each of the 8 neighbor bitfields, we perform a ripple-carry increment
//   of this 4-bit number for all 64 cells in parallel using bitwise XOR/AND.
//
// Life rules (per cell):
// - Let neighbors = number of live neighbors.
// - If cell is alive:
//       survive if neighbors == 2 or neighbors == 3
// - If cell is dead:
//       become alive if neighbors == 3
//
// Using bit-slice representation of neighbors:
// - neighbors == 2  <=>  count == 0b0010
// - neighbors == 3  <=>  count == 0b0011
//
// Next state mask per word:
//   next = (alive & (neighbors == 2)) | (neighbors == 3)

#include <cstdint>
#include <cuda_runtime.h>

// Bit-sliced increment of a 4-bit per-cell counter by a 1-bit per-cell addend.
// All 64 cells are processed in parallel using bitwise operations.
//
// Each cell's neighbor count is represented with four bitplanes:
//   count0: LSB
//   count1: next bit
//   count2: next bit
//   count3: MSB
//
// 'bits' is a 64-bit word whose j-th bit is 1 if the cell j receives +1 to its
// neighbor count from one particular neighbor direction.
//
// This function performs: count = count + bits (per cell, no overflow handling
// needed for this application since max count is 8 and 4 bits suffice).
__device__ __forceinline__
void add_neighbor_bits(std::uint64_t bits,
                       std::uint64_t &count0,
                       std::uint64_t &count1,
                       std::uint64_t &count2,
                       std::uint64_t &count3)
{
    // Ripple-carry addition: count += bits
    std::uint64_t carry0 = count0 & bits;
    count0 ^= bits;

    std::uint64_t carry1 = count1 & carry0;
    count1 ^= carry0;

    std::uint64_t carry2 = count2 & carry1;
    count2 ^= carry1;

    count3 ^= carry2;  // Any carry from bit2 goes into bit3.
}

// CUDA kernel: one thread processes one 64-bit word (64 cells).
// 'N' is the grid dimension (N x N).
// 'wordsPerRow' is N / 64 (number of 64-bit words per row).
// 'wordsShift' is log2(wordsPerRow) and is used to compute row index via shift.
__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int N,
                         int wordsPerRow,
                         int wordsShift)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWords = N * wordsPerRow;
    if (tid >= totalWords)
        return;

    // Compute row and word-column for this thread's word.
    // Since wordsPerRow is a power of two, we can use bit operations:
    int row = tid >> wordsShift;
    int col = tid & (wordsPerRow - 1);

    // Load center word.
    std::uint64_t C = input[tid];

    // Neighbor words; default to 0 (dead outside grid).
    std::uint64_t U  = 0, D  = 0;
    std::uint64_t L  = 0, R  = 0;
    std::uint64_t UL = 0, UR = 0;
    std::uint64_t DL = 0, DR = 0;

    // Vertical neighbors.
    if (row > 0) {
        int idxU = tid - wordsPerRow;
        U = input[idxU];
        if (col > 0) {
            UL = input[idxU - 1];
        }
        if (col + 1 < wordsPerRow) {
            UR = input[idxU + 1];
        }
    }
    if (row + 1 < N) {
        int idxD = tid + wordsPerRow;
        D = input[idxD];
        if (col > 0) {
            DL = input[idxD - 1];
        }
        if (col + 1 < wordsPerRow) {
            DR = input[idxD + 1];
        }
    }

    // Horizontal neighbors in the same row.
    if (col > 0) {
        L = input[tid - 1];
    }
    if (col + 1 < wordsPerRow) {
        R = input[tid + 1];
    }

    // Construct neighbor bitfields for all 64 cells in this word.
    //
    // For each bit position j in the center word:
    //   - above_left_bit(j)  = cell at (row-1, col-1, bit j-1)   (with word/bit wrap)
    //   - above_bit(j)       = cell at (row-1, col,   bit j)
    //   - above_right_bit(j) = cell at (row-1, col+1, bit j+1)
    //   etc.
    //
    // We align all neighbor bits so that for each neighbor direction, the j-th
    // bit represents the neighbor of the cell at bit j in the center word.
    //
    // Word-boundary handling:
    // - Shifts within a word (<<1, >>1) move neighbor bits horizontally.
    // - Bits that cross word boundaries are filled in from L, R, UL, UR, DL, DR.
    std::uint64_t above_left  = (U << 1) | (UL >> 63);
    std::uint64_t above       = U;
    std::uint64_t above_right = (U >> 1) | (UR << 63);

    std::uint64_t left        = (C << 1) | (L >> 63);
    std::uint64_t right       = (C >> 1) | (R << 63);

    std::uint64_t below_left  = (D << 1) | (DL >> 63);
    std::uint64_t below       = D;
    std::uint64_t below_right = (D >> 1) | (DR << 63);

    // Accumulate neighbor counts per cell using 4-bit bit-sliced counters.
    std::uint64_t count0 = 0;
    std::uint64_t count1 = 0;
    std::uint64_t count2 = 0;
    std::uint64_t count3 = 0;

    add_neighbor_bits(above_left,  count0, count1, count2, count3);
    add_neighbor_bits(above,       count0, count1, count2, count3);
    add_neighbor_bits(above_right, count0, count1, count2, count3);
    add_neighbor_bits(left,        count0, count1, count2, count3);
    add_neighbor_bits(right,       count0, count1, count2, count3);
    add_neighbor_bits(below_left,  count0, count1, count2, count3);
    add_neighbor_bits(below,       count0, count1, count2, count3);
    add_neighbor_bits(below_right, count0, count1, count2, count3);

    // neighbor count bits per cell are now:
    //   count = (count3 << 3) | (count2 << 2) | (count1 << 1) | count0
    //
    // We only need to detect counts == 2 or == 3.
    //
    // Binary:
    //   2 -> 0010 -> c3=0, c2=0, c1=1, c0=0
    //   3 -> 0011 -> c3=0, c2=0, c1=1, c0=1
    std::uint64_t not_c0 = ~count0;
    std::uint64_t not_c2 = ~count2;
    std::uint64_t not_c3 = ~count3;

    // neighbors == 2 mask
    std::uint64_t eq2 = not_c3 & not_c2 & count1 & not_c0;

    // neighbors == 3 mask
    std::uint64_t eq3 = not_c3 & not_c2 & count1 & count0;

    // Current cell states (1 = alive, 0 = dead) are in C.
    // Next state:
    //   survive_mask = C & (neighbors == 2)
    //   birth_mask   = neighbors == 3
    //   next         = survive_mask | birth_mask
    std::uint64_t survive_mask = C & eq2;
    std::uint64_t birth_mask   = eq3;
    std::uint64_t next         = survive_mask | birth_mask;

    output[tid] = next;
}

// Host function: perform one Game of Life step on a bit-packed grid.
// - input  : device pointer to N x N grid, bit-packed in std::uint64_t
// - output : device pointer where the next generation will be written
// - grid_dimensions : N (width == height), power of 2, N > 512
//
// Both input and output must be allocated with cudaMalloc and point to device
// memory regions of size (N * N / 8) bytes.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    int N = grid_dimensions;

    // Number of 64-bit words per row.
    int wordsPerRow = N >> 6;  // N / 64

    // Compute log2(wordsPerRow) on the host (wordsPerRow is guaranteed power-of-two).
    int wordsShift = 0;
    {
        int tmp = wordsPerRow;
        while (tmp > 1) {
            tmp >>= 1;
            ++wordsShift;
        }
    }

    int totalWords = N * wordsPerRow;

    // Launch configuration: one thread per 64-bit word.
    const int threadsPerBlock = 256;
    int blocks = (totalWords + threadsPerBlock - 1) / threadsPerBlock;

    game_of_life_kernel<<<blocks, threadsPerBlock>>>(input, output, N, wordsPerRow, wordsShift);
    // Any synchronization (cudaDeviceSynchronize, checking errors, etc.) is
    // handled by the caller as per the problem statement.
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
