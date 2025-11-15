// Conway's Game of Life, bit-packed CUDA implementation.
//
// Each cell is a single bit. The grid is square with side length N, where N is a power
// of two and N >= 512. Each row is stored as (N / 64) 64-bit words, each word
// encoding 64 consecutive cells in that row, with bit 0 the leftmost cell and bit 63
// the rightmost cell inside that 64-cell segment.
//
// Memory layout (row-major):
//   - words_per_row = N / 64
//   - total_words   = N * words_per_row
//   - word index w  corresponds to:
//        row = w / words_per_row
//        col = w % words_per_row
//
// Each CUDA thread processes exactly one 64-bit word (i.e., 64 cells).
//
// For each word, the kernel reads up to 9 words:
//   - current row:     left (L), center (C), right (R)
//   - row above:       uL, uC, uR
//   - row below:       dL, dC, dR
//
// Outside-grid cells are treated as dead by conditionally zeroing out-of-bound
// neighbor words. The bit 0 and bit 63 cells correctly see neighbors across word
// boundaries by combining L/R with the appropriate shifts.
//
// To compute neighbor counts, we construct 8 bitmasks representing the 8 neighbor
// directions for the 64 cells in the current word:
//
//   From row above:  NW (up_w), N (up_c), NE (up_e)
//   From same row:   W  (mid_w),        E  (mid_e)
//   From row below:  SW (dn_w), S (dn_c), SE (dn_e)
//
// For example, for the row above:
//   - up_c = uC
//   - up_w = (uC << 1) | (uL >> 63)
//   - up_e = (uC >> 1) | (uR << 63)
//
// The directions for the current row and below are analogous.
//
// We maintain three bitplanes (c0, c1, c2) that represent the neighbor count per
// cell modulo 8 in binary (LSB c0, then c1, then c2). We repeatedly "add 1" for
// each of the 8 neighbor masks by performing bitwise ripple-carry addition on
// these bitplanes. Because the true neighbor count is in [0, 8], representing the
// count modulo 8 is sufficient to determine whether the count equals 2 or 3:
//
//   - For counts 0..7, the 3-bit value is exact.
//   - For count 8, the 3-bit value wraps to 0, but in Game of Life any cell with
//     8 neighbors dies or stays dead, which is the same outcome as 0 neighbors.
//     We only care about "==2" and "==3", so this wraparound is safe.
//
// After accumulating neighbor counts, we derive bitmasks for "neighbors == 2"
// and "neighbors == 3" as:
//
//   eq2 = ~c2 &  c1 & ~c0  // binary 010
//   eq3 = ~c2 &  c1 &  c0  // binary 011
//
// The next state bits are then:
//
//   survive_mask = eq2 | eq3               // alive cell survives if 2 or 3 neighbors
//   birth_mask   = eq3                     // dead cell is born if exactly 3 neighbors
//   next = (cur & survive_mask) | (~cur & birth_mask)
//
// No shared memory or texture memory is used; all accesses are from global memory.
// All arithmetic on neighbor counts is done with 64-bit bitwise operations.

#include <cstdint>
#include <cuda_runtime.h>

// Add a single neighbor bitmask to the 3-bit-per-cell neighbor counter.
//
// The neighbor counter for each bit (cell) is stored as three 64-bit bitplanes:
//   c0 : least significant bit of the count
//   c1 : middle bit
//   c2 : most significant bit
//
// 'mask' has 1s where there is a neighbor contribution for that bit. We perform a
// bit-parallel addition of this 1-bit value (per cell) into the 3-bit counter:
//
//   (c2 c1 c0) += mask   (mod 8)
//
// This is done via ripple-carry addition using XOR and AND operations for all
// 64 cells at once. We ignore overflow beyond 3 bits (modulo 8 arithmetic).
static __device__ __forceinline__
void add_neighbor_mask(std::uint64_t &c0, std::uint64_t &c1, std::uint64_t &c2,
                       std::uint64_t mask)
{
    // Add mask to bitplane c0.
    std::uint64_t t0     = c0 ^ mask;      // new bit 0
    std::uint64_t carry0 = c0 & mask;      // carry into bit 1

    // Add carry0 to bitplane c1.
    std::uint64_t t1     = c1 ^ carry0;    // new bit 1
    std::uint64_t carry1 = c1 & carry0;    // carry into bit 2

    // Add carry1 to bitplane c2.
    std::uint64_t t2     = c2 ^ carry1;    // new bit 2
    // carry2 = c2 & carry1 would be an overflow beyond bit 2; we ignore it, which
    // effectively computes the sum modulo 8.

    c0 = t0;
    c1 = t1;
    c2 = t2;
}

// Kernel: compute one generation of Conway's Game of Life.
// Each thread processes one 64-bit word of the grid.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row,
                                    int words_per_row_log2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dim * words_per_row;

    if (tid >= total_words) {
        return;
    }

    // Compute row and column indices from the word index.
    // Since words_per_row is a power of two, we can avoid integer division/modulo:
    //   row = tid / words_per_row = tid >> words_per_row_log2
    //   col = tid % words_per_row = tid & (words_per_row - 1)
    int row = tid >> words_per_row_log2;
    int col = tid & (words_per_row - 1);

    const std::uint64_t cur = input[tid];

    // Determine boundary conditions.
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dim);

    const int row_stride = words_per_row;

    // Neighbor words in the current row.
    const std::uint64_t cL = has_left  ? input[tid - 1] : 0ull;
    const std::uint64_t cR = has_right ? input[tid + 1] : 0ull;

    // Neighbor words in the row above.
    const std::uint64_t uC = has_up ? input[tid - row_stride] : 0ull;
    const std::uint64_t uL = (has_up && has_left)
                             ? input[tid - row_stride - 1] : 0ull;
    const std::uint64_t uR = (has_up && has_right)
                             ? input[tid - row_stride + 1] : 0ull;

    // Neighbor words in the row below.
    const std::uint64_t dC = has_down ? input[tid + row_stride] : 0ull;
    const std::uint64_t dL = (has_down && has_left)
                             ? input[tid + row_stride - 1] : 0ull;
    const std::uint64_t dR = (has_down && has_right)
                             ? input[tid + row_stride + 1] : 0ull;

    // Initialize neighbor count bitplanes (modulo 8).
    std::uint64_t c0 = 0ull;  // least significant bit of neighbor count
    std::uint64_t c1 = 0ull;
    std::uint64_t c2 = 0ull;

    // Row above: contribute NW, N, NE neighbors for the current row.
    if (has_up) {
        // NW neighbors: cells one row up and one column left.
        // For each current cell (bit position j), its NW neighbor is at bit j-1 in uC,
        // plus possible contribution from uL's bit 63 for j = 0 when col > 0.
        const std::uint64_t up_w = (uC << 1) | (uL >> 63);

        // N neighbors: cells directly above; same bit positions as uC.
        const std::uint64_t up_c = uC;

        // NE neighbors: cells one row up and one column right.
        // For each current cell j, its NE neighbor is uC[j+1], plus dword crossover
        // from uR's bit 0 for j = 63 when col+1 < words_per_row.
        const std::uint64_t up_e = (uC >> 1) | (uR << 63);

        add_neighbor_mask(c0, c1, c2, up_w);
        add_neighbor_mask(c0, c1, c2, up_c);
        add_neighbor_mask(c0, c1, c2, up_e);
    }

    // Current row: contribute W and E neighbors (we must not include the cell itself).
    // W neighbors: for each cell j, its west neighbor is at position j-1.
    // mid_w[j] = cur[j-1] with word-boundary crossover from cL.
    const std::uint64_t mid_w = (cur << 1) | (cL >> 63);

    // E neighbors: for each cell j, its east neighbor is at position j+1.
    // mid_e[j] = cur[j+1] with word-boundary crossover from cR.
    const std::uint64_t mid_e = (cur >> 1) | (cR << 63);

    add_neighbor_mask(c0, c1, c2, mid_w);
    add_neighbor_mask(c0, c1, c2, mid_e);

    // Row below: contribute SW, S, SE neighbors for the current row.
    if (has_down) {
        const std::uint64_t dn_w = (dC << 1) | (dL >> 63);  // SW neighbors
        const std::uint64_t dn_c = dC;                      // S neighbors
        const std::uint64_t dn_e = (dC >> 1) | (dR << 63);  // SE neighbors

        add_neighbor_mask(c0, c1, c2, dn_w);
        add_neighbor_mask(c0, c1, c2, dn_c);
        add_neighbor_mask(c0, c1, c2, dn_e);
    }

    // Compute masks for neighbor count == 2 and neighbor count == 3.
    // We treat the neighbor count as 3-bit number (c2 c1 c0) modulo 8.
    //
    // Binary patterns:
    //   2 -> 010  -> ~c2 &  c1 & ~c0
    //   3 -> 011  -> ~c2 &  c1 &  c0
    //
    // Note: true count of 8 neighbors wraps to 0 (000) here, which is correct for
    // Game of Life decisions since we only care about counts equal to 2 or 3.
    const std::uint64_t not_c2 = ~c2;
    const std::uint64_t not_c0 = ~c0;

    const std::uint64_t eq2 = not_c2 & c1 & not_c0;
    const std::uint64_t eq3 = not_c2 & c1 & c0;

    const std::uint64_t survive_mask = eq2 | eq3;  // neighbors == 2 or 3
    const std::uint64_t birth_mask   = eq3;        // neighbors == 3

    // Apply Game of Life rules:
    //   - Any live cell with 2 or 3 neighbors survives.
    //   - Any live cell with fewer than 2 or more than 3 neighbors dies.
    //   - Any dead cell with exactly 3 neighbors becomes alive.
    const std::uint64_t next =
        (cur & survive_mask) | (~cur & birth_mask);

    output[tid] = next;
}

// Host function: run one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   - input           : device pointer to input grid (bit-packed as uint64_t words)
//   - output          : device pointer to output grid (bit-packed as uint64_t words)
//   - grid_dimensions : width and height of the (square) grid; power of 2, >= 512
//
// The caller is responsible for:
//   - allocating input/output using cudaMalloc
//   - copying data to/from the device if needed
//   - performing any required synchronization after this call
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Compute log2(words_per_row) on the host. words_per_row is guaranteed to be
    // a power of two because grid_dimensions is a power of two and we divide by 64.
    int words_per_row_log2 = 0;
    {
        int tmp = words_per_row;
        while (tmp > 1) {
            tmp >>= 1;
            ++words_per_row_log2;
        }
    }

    const int total_words = grid_dimensions * words_per_row;

    // Choose a reasonable block size; 256 threads per block is a common choice
    // that works well on modern NVIDIA GPUs.
    const int threads_per_block = 256;
    const int blocks = (total_words + threads_per_block - 1) / threads_per_block;

    game_of_life_kernel<<<blocks, threads_per_block>>>(
        input, output, grid_dimensions, words_per_row, words_per_row_log2
    );

    // No cudaDeviceSynchronize() here; the caller handles synchronization.
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
