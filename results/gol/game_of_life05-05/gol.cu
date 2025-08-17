#include <cuda_runtime.h>
#include <cstdint>

/*
Conway's Game of Life - Single-step kernel for a bit-packed grid.

Key characteristics:
- Each 64-bit word encodes 64 consecutive cells in a row (bit 0: leftmost, bit 63: rightmost).
- Each CUDA thread processes exactly one 64-bit word => no atomics and high throughput.
- Neighbor counts are computed using bit-parallel boolean arithmetic (carry-save adders) to
  update 64 cells at once, avoiding per-cell loops.
- Outside-grid cells are treated as dead; borders are handled via zero-fill of missing neighbors.
- No shared/texture memory: global memory with coalesced access and grid-stride loop.

The algorithm computes the 8-neighborhood count per bit using only bitwise logic:
1) For each of the three relevant rows (above, same, below), form three aligned bitsets:
   west (shifted with cross-word carry from the left word), center, east (shifted with carry from the right).
2) For each row, compute the sum of these three bitsets using a carry-save adder (CSA):
   - ones_row = x ^ y ^ z
   - twos_row = (x & y) | (x & z) | (y & z)
   Note that for the middle row (same row as the cell), the center term must be 0 because a
   cell is not its own neighbor.
3) Combine the three row sums across rows using another CSA layer. Let:
   - row ones: t1, m1, b1
   - row twos: t2, m2, b2
   We compute:
   - ones = t1 ^ m1 ^ b1
   - carry1 (2s from ones) = (t1 & m1) | (t1 & b1) | (m1 & b1)
   - twos_parity = t2 ^ m2 ^ b2
   - twos_carry  = (t2 & m2) | (t2 & b2) | (m2 & b2)
   The total per-bit neighbor count is:
     count = ones + 2*(carry1 + twos_parity) + 4*(twos_carry + (carry1 & twos_parity))
   We only need to test for equality to 2 or 3. This leads to:
     two_bit  = carry1 ^ twos_parity          (bit-1 of count)
     four_bit = twos_carry | (carry1 & twos_parity)  (bit-2 of count; if set, count >= 4)
   Then:
     eq2 = (~four_bit) &  two_bit & (~ones)
     eq3 = (~four_bit) &  two_bit &  ones
   And the next state is:
     next = eq3 | (alive & eq2)

Bit shifting with cross-word carry for west/east neighbors:
- West (neighbor to the left):  (center << 1) | (left >> 63)
- East (neighbor to the right): (center >> 1) | (right << 63)
For missing words (at boundaries), left/right words are zero, thus maintaining "outside is dead".

Grid indexing:
- Grid is square with dimension N = grid_dimensions, and words per row W = N / 64.
- Total number of 64-bit words is N * W. Thread idx maps to (row = idx / W, col = idx % W).
*/

static __forceinline__ __device__ uint64_t shift_west(uint64_t center, uint64_t left) {
    // For each bit position p, result[p] = center[p-1] with carry-in from left[63] to result[0].
    return (center << 1) | (left >> 63);
}

static __forceinline__ __device__ uint64_t shift_east(uint64_t center, uint64_t right) {
    // For each bit position p, result[p] = center[p+1] with carry-in from right[0] to result[63].
    return (center >> 1) | (right << 63);
}

static __forceinline__ __device__ void sum3(uint64_t a, uint64_t b, uint64_t c, uint64_t &ones, uint64_t &twos) {
    // Carry-save sum of three 1-bit-per-lane integers.
    // ones = a ^ b ^ c
    // twos = (a & b) | (a & c) | (b & c)
    ones = a ^ b ^ c;
    // Faster form for twos using associativity:
    // twos = (a & b) | (c & (a ^ b));
    twos = (a & b) | (c & (a ^ b));
}

__global__ void game_of_life_kernel(const uint64_t* __restrict__ in,
                                    uint64_t* __restrict__ out,
                                    int words_per_row,
                                    int height_rows)
{
    const size_t total_words = static_cast<size_t>(words_per_row) * static_cast<size_t>(height_rows);
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_words; idx += blockDim.x * gridDim.x) {

        const int row = static_cast<int>(idx / words_per_row);
        const int col = static_cast<int>(idx - static_cast<size_t>(row) * words_per_row);

        // Determine boundary presence
        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < height_rows);
        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < words_per_row);

        // Load the 3x3 neighborhood in 64-bit words with zero-fill at boundaries.
        // Top row (row-1)
        const uint64_t upL = (has_up && has_left)  ? in[idx - words_per_row - 1] : 0ull;
        const uint64_t upC = (has_up)              ? in[idx - words_per_row]     : 0ull;
        const uint64_t upR = (has_up && has_right) ? in[idx - words_per_row + 1] : 0ull;
        // Mid row (row)
        const uint64_t mdL = (has_left)  ? in[idx - 1] : 0ull;
        const uint64_t mdC = in[idx];
        const uint64_t mdR = (has_right) ? in[idx + 1] : 0ull;
        // Bottom row (row+1)
        const uint64_t dnL = (has_down && has_left)  ? in[idx + words_per_row - 1] : 0ull;
        const uint64_t dnC = (has_down)              ? in[idx + words_per_row]     : 0ull;
        const uint64_t dnR = (has_down && has_right) ? in[idx + words_per_row + 1] : 0ull;

        // Construct aligned neighbor bitsets for each of the three rows.
        // Top row: include west, center, east
        const uint64_t top_w = shift_west(upC, upL);
        const uint64_t top_c = upC;
        const uint64_t top_e = shift_east(upC, upR);
        uint64_t t1, t2;
        sum3(top_w, top_c, top_e, t1, t2);

        // Middle row: only left and right neighbors (exclude the cell itself)
        const uint64_t mid_w = shift_west(mdC, mdL);
        const uint64_t mid_e = shift_east(mdC, mdR);
        // sum3(mid_w, 0, mid_e, m1, m2) simplifies to:
        const uint64_t m1 = (mid_w ^ mid_e);
        const uint64_t m2 = (mid_w & mid_e);

        // Bottom row: include west, center, east
        const uint64_t bot_w = shift_west(dnC, dnL);
        const uint64_t bot_c = dnC;
        const uint64_t bot_e = shift_east(dnC, dnR);
        uint64_t b1, b2;
        sum3(bot_w, bot_c, bot_e, b1, b2);

        // Combine row sums across rows using carry-save logic.
        // ones from rows:
        const uint64_t ones = t1 ^ m1 ^ b1;
        const uint64_t carry1 = (t1 & m1) | (t1 & b1) | (m1 & b1); // contributes 2 each

        // twos across rows:
        const uint64_t twos_parity = t2 ^ m2 ^ b2; // parity of 2s
        const uint64_t twos_carry  = (t2 & m2) | (t2 & b2) | (m2 & b2); // contributes 4 each

        // Final bit-slices of neighbor count:
        const uint64_t bit2 = carry1 ^ twos_parity;            // 2's bit (value 2)
        const uint64_t bit4 = twos_carry | (carry1 & twos_parity); // 4's bit (value 4)

        // Apply Game of Life rules:
        // next = (neighbors == 3) | (alive & (neighbors == 2))
        const uint64_t alive = mdC;
        const uint64_t eq2 = (~bit4) &  bit2 & (~ones);
        const uint64_t eq3 = (~bit4) &  bit2 &  ones;

        const uint64_t next = eq3 | (alive & eq2);
        out[idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the number of cells per side (square grid), guaranteed to be a power of 2.
    // Number of 64-bit words per row:
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    const size_t total_words = static_cast<size_t>(grid_dimensions) * static_cast<size_t>(words_per_row);

    // CUDA launch configuration:
    // - Moderate block size to balance register pressure and occupancy.
    // - Grid-stride loop inside the kernel handles any total size.
    constexpr int block_size = 256;
    int max_blocks;
    cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMaxGridDimX, 0);
    // Clamp grid size to the device limit for safety:
    const int grid_size = static_cast<int>(
        (total_words + block_size - 1) / block_size <= static_cast<size_t>(max_blocks)
            ? (total_words + block_size - 1) / block_size
            : max_blocks
    );

    game_of_life_kernel<<<grid_size, block_size>>>(input, output, words_per_row, grid_dimensions);
    // Synchronization after kernel launch is handled by the caller as per the problem statement.
}