// Conway's Game of Life - CUDA implementation using bit-packed 64-bit words.
//
// Each std::uint64_t encodes 64 horizontally consecutive cells in a row.
// Bit i (0 <= i < 64) of a word represents one cell: 1 = alive, 0 = dead.
//
// The grid is square with dimensions grid_dimensions x grid_dimensions
// (in cells). grid_dimensions is always a power-of-two >= 512 and a multiple
// of 64, so each row consists of grid_dimensions / 64 64-bit words.
//
// One CUDA thread processes one 64-bit word (i.e., 64 cells).
//
// Neighbor counts are computed in a fully bit-parallel manner:
//   - For each of the 64 bit positions we need the count (0..8) of alive
//     neighbors among the 8 surrounding cells.
//   - We do this using bit-sliced saturating counters: for each word we keep
//     three 64-bit masks (count0, count1, count2) representing a 3-bit
//     saturating count per cell (0..4, with 4 meaning ">=4").
//   - We add each of the 8 neighbor bitfields into this 3-bit counter using
//     simple bitwise operations (XOR and AND), effectively implementing
//     a bank of 64 independent 3-bit full adders in parallel.
//
// We do *not* use standard integer addition for the counts, because carries
// between bit positions would mix counts from different cells. Instead, the
// bit-sliced counters ensure that each bit position is updated independently.
//
// For the Game of Life rules we only need to know whether the neighbor count
// is exactly 2, exactly 3, or anything else; counts >= 4 can be treated
// identically and are encoded as value 4 in the saturating counter:
//   - 0 -> 000
//   - 1 -> 001
//   - 2 -> 010
//   - 3 -> 011
//   - 4 or more -> 100
//
// The update rule per cell is:
//   - If currently alive: survives iff neighbors == 2 or neighbors == 3.
//   - If currently dead: becomes alive iff neighbors == 3.
//
// Boundary handling:
//   - All cells outside the grid are treated as dead (0).
//   - This requires special treatment for the 0th and 63rd bits of each word,
//     where some neighbors reside in the left or right adjacent words.
//   - For bit 0 we also need bits from the three words to the left
//     (same row, row above, and row below).
//   - For bit 63 we need bits from the three words to the right.
//   - We obtain these via shifts plus injected edge bits from neighboring
//     words. Out-of-bound words (outside the grid) are simply treated as 0.
//
// No shared or texture memory is used; all reads are from global memory,
// and pointers are marked __restrict__ to enable better compiler optimization.

#include <cstdint>
#include <cuda_runtime.h>

// Short alias for 64-bit unsigned type.
using Uint64 = std::uint64_t;

// Device helper: add a neighbor bitfield into a 3-bit per-cell saturating counter.
//
// For each bit position i (0..63):
//   - bits[i] is either 0 or 1, indicating whether a neighbor is alive.
//   - (count2[i], count1[i], count0[i]) form a 3-bit integer in [0..4],
//     representing the number of neighbors seen so far for that cell,
//     with 4 meaning "4 or more" (saturated).
//
// This performs: count = min(count + bits, 4) per bit position, using fully
// parallel bitwise logic. The implementation is a bit-sliced 3-bit adder
// with saturation at 4.
__device__ __forceinline__
void add_neighbor_bits(Uint64 bits, Uint64 &count0, Uint64 &count1, Uint64 &count2)
{
    // Compute which positions are already saturated at 4 (binary 100).
    // is_four[i] = 1 iff (count2,count1,count0) == 100 at bit i.
    Uint64 is_four = count2 & ~count1 & ~count0;

    // Do not increment positions already at 4.
    Uint64 p = bits & ~is_four;

    // Add p (0 or 1) to the 3-bit value (count2,count1,count0) in bit-sliced form.
    // First bit (LSB).
    Uint64 carry0 = count0 & p;
    count0 ^= p;

    // Second bit.
    Uint64 carry1 = count1 & carry0;
    count1 ^= carry0;

    // Third bit; carry beyond this would correspond to going above 4,
    // but we have already masked out increments where the value is 4,
    // so the result stays in the range [0..4].
    count2 ^= carry1;
}

// CUDA kernel: computes one Game of Life step for a square grid.
//
// input  - pointer to input grid in bit-packed form (device memory).
// output - pointer to output grid in bit-packed form (device memory).
// grid_dim - width and height of the square grid in cells.
//            Must be a power of two and a multiple of 64.
//
// The grid is represented row-major, with grid_dim / 64 words per row.
// Each thread processes one 64-bit word (64 cells) at position (row, colWord).
__global__ void game_of_life_kernel(const Uint64* __restrict__ input,
                                    Uint64* __restrict__ output,
                                    int grid_dim)
{
    const int words_per_row = grid_dim >> 6; // grid_dim / 64

    // 2D thread index: (colWord, row).
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // word index along row
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (col >= words_per_row || row >= grid_dim)
        return;

    const Uint64* row_ptr = input + static_cast<std::size_t>(row) * words_per_row;
    const Uint64 center = row_ptr[col]; // current word (64 cells)

    // Load neighboring words where they exist; otherwise keep as 0.
    Uint64 cleft  = 0;
    Uint64 cright = 0;
    Uint64 nmid   = 0;
    Uint64 nleft  = 0;
    Uint64 nright = 0;
    Uint64 smid   = 0;
    Uint64 sleft  = 0;
    Uint64 sright = 0;

    // Left and right neighbors in the same row.
    if (col > 0) {
        cleft = row_ptr[col - 1];
    }
    if (col + 1 < words_per_row) {
        cright = row_ptr[col + 1];
    }

    // Row above (north).
    if (row > 0) {
        const Uint64* rowN = input + static_cast<std::size_t>(row - 1) * words_per_row;
        nmid = rowN[col];
        if (col > 0) {
            nleft = rowN[col - 1];
        }
        if (col + 1 < words_per_row) {
            nright = rowN[col + 1];
        }
    }

    // Row below (south).
    if (row + 1 < grid_dim) {
        const Uint64* rowS = input + static_cast<std::size_t>(row + 1) * words_per_row;
        smid = rowS[col];
        if (col > 0) {
            sleft = rowS[col - 1];
        }
        if (col + 1 < words_per_row) {
            sright = rowS[col + 1];
        }
    }

    // Construct bitfields for the 8 neighbor directions.
    //
    // For each direction D, the 64-bit word D has bit i = 1 if and only if
    // the neighbor in direction D of the cell at bit i in 'center' is alive.
    //
    // Vertical neighbors are straightforward: same bit position in row above/below.
    const Uint64 N = nmid;
    const Uint64 S = smid;

    // Horizontal neighbors:
    // - For bits 1..63, the west neighbor of bit i is bit i-1 of the same word.
    //   This is implemented by center << 1.
    // - For bit 0, the west neighbor is bit 63 of the word to the left.
    //   Implemented by injecting (cleft >> 63) into bit 0.
    const Uint64 W = (center << 1) | (cleft >> 63);

    // - For bits 0..62, the east neighbor of bit i is bit i+1 of the same word.
    //   This is implemented by center >> 1.
    // - For bit 63, the east neighbor is bit 0 of the word to the right.
    //   Implemented by injecting ((cright & 1) << 63) into bit 63.
    const Uint64 E = (center >> 1) | ((cright & Uint64{1}) << 63);

    // Diagonal neighbors: combine shifts with injected edge bits from
    // neighboring words above/below.
    const Uint64 NW = (nmid << 1) | (nleft >> 63);
    const Uint64 NE = (nmid >> 1) | ((nright & Uint64{1}) << 63);
    const Uint64 SW = (smid << 1) | (sleft >> 63);
    const Uint64 SE = (smid >> 1) | ((sright & Uint64{1}) << 63);

    // Bit-sliced neighbor counts: count2: MSB, count0: LSB.
    // Value per bit position is in [0..4], with 4 meaning ">=4".
    Uint64 count0 = 0;
    Uint64 count1 = 0;
    Uint64 count2 = 0;

    // Accumulate the 8 neighbor bitfields into the counts.
    add_neighbor_bits(N,  count0, count1, count2);
    add_neighbor_bits(S,  count0, count1, count2);
    add_neighbor_bits(W,  count0, count1, count2);
    add_neighbor_bits(E,  count0, count1, count2);
    add_neighbor_bits(NW, count0, count1, count2);
    add_neighbor_bits(NE, count0, count1, count2);
    add_neighbor_bits(SW, count0, count1, count2);
    add_neighbor_bits(SE, count0, count1, count2);

    // Decode neighbor count conditions needed for Game of Life:
    //   neighbors == 2 : (~count2 &  count1 & ~count0)
    //   neighbors == 3 : (~count2 &  count1 &  count0)
    //
    // Note that count2 = 1 corresponds to neighbors >= 4, so we explicitly
    // exclude those via the ~count2 term.
    const Uint64 not_c2 = ~count2;
    const Uint64 not_c0 = ~count0;

    const Uint64 mask_eq2 = not_c2 & count1 & not_c0;
    const Uint64 mask_eq3 = not_c2 & count1 & count0;

    const Uint64 alive = center;

    // Next state per bit:
    //   - Alive cell survives if neighbors == 2 or 3.
    //   - Dead cell becomes alive if neighbors == 3.
    //
    // Implemented as:
    //   next = (alive & mask_eq2) | mask_eq3;
    const Uint64 next = (alive & mask_eq2) | mask_eq3;

    output[static_cast<std::size_t>(row) * words_per_row + col] = next;
}

// Host function: execute one step of Conway's Game of Life on the GPU.
//
// input  - device pointer to the input grid (bit-packed std::uint64_t array).
// output - device pointer to the output grid (bit-packed std::uint64_t array).
// grid_dimensions - width/height of the square grid in cells (power of two,
//                   multiple of 64, >= 512).
//
// Both input and output must be allocated with cudaMalloc and sized to hold
// grid_dimensions * grid_dimensions cells, i.e. (grid_dimensions * grid_dimensions / 64)
// 64-bit words.
//
// This function launches the kernel but does not perform any host-device
// synchronization; the caller is responsible for that if needed.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // 2D launch configuration:
    //   - block.x spans words along the row (x direction)
    //   - block.y spans rows (y direction)
    //
    // block size of 256 threads (32x8) works well on modern NVIDIA data center GPUs.
    const dim3 block(32, 8, 1);
    const dim3 grid(
        (words_per_row    + block.x - 1) / block.x,
        (grid_dimensions  + block.y - 1) / block.y,
        1
    );

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
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
