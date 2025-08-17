#include <cuda_runtime.h>
#include <cstdint>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  - Grid storage: Each 64-bit word encodes 64 consecutive cells in a row (1 = alive, 0 = dead).
  - One CUDA thread processes exactly one 64-bit word (64 cells) of the grid.
  - The computation is fully bit-parallel using boolean algebra and full-adder logic.
  - Outside-grid cells are treated as dead (zero), so boundary handling sets neighbor words to zero.
  - Shared/texture memory are not used; coalesced global loads are sufficient on modern GPUs.

  Neighbor counting via bitwise full adders:
    - For three inputs a, b, c representing the same bit position across three neighbor masks:
        sum1 = a XOR b XOR c                   // ones place of the partial sum
        carry1 = majority(a, b, c)             // twos place of the partial sum (a&b | b&c | a&c)
    - We compute partial sums for the three rows of neighbors (top triple, middle pair, bottom triple),
      then combine them with further adders to obtain the neighbor count bits:
         n1 = bit 0 (1s), n2 = bit 1 (2s), n4 = bit 2 (4s) of the neighbor count (0..8).
      The 8s bit is not required because the Game of Life decision depends only on counts 2 and 3.

  Horizontal (left/right) neighbors and diagonals require cross-word carry:
    - Left neighbors: (word << 1) | (leftWord >> 63)      // bring in bit 63 from the word on the left
    - Right neighbors: (word >> 1) | ((rightWord & 1) << 63) // bring in bit 0 from the word on the right
  This correctly handles the 0th and 63rd bits of each word without wrap-around.

  Game of Life rule (for each bit):
    - next = (neighbors == 3) | (alive & (neighbors == 2))
    - neighbors == 3 iff n4 == 0 and n2 == 1 and n1 == 1
    - neighbors == 2 iff n4 == 0 and n2 == 1 and n1 == 0
*/

static __device__ __forceinline__ std::uint64_t maj3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    // Majority (carry) function of three bitfields: true when at least two inputs are 1.
    return (a & b) | (b & c) | (a & c);
}

static __device__ __forceinline__ void add3_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                                std::uint64_t &sum, std::uint64_t &carry) {
    // Bitwise 3-input full adder across 64 lanes:
    // sum = a ^ b ^ c           (ones place)
    // carry = majority(a,b,c)   (twos place)
    sum = a ^ b ^ c;
    carry = maj3(a, b, c);
}

static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t w, std::uint64_t wl) {
    // Shift left by 1 with cross-word carry-in from left neighbor's bit 63.
    return (w << 1) | (wl >> 63);
}

static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t w, std::uint64_t wr) {
    // Shift right by 1 with cross-word carry-in from right neighbor's bit 0.
    return (w >> 1) | ((wr & 1ull) << 63);
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int grid_dim, int words_per_row) {
    const int total_words = grid_dim * words_per_row;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;

    // Compute 2D position: row and column (word index within row)
    const int row = tid / words_per_row;
    const int col = tid - row * words_per_row;

    // Indices for neighboring words
    const bool has_left   = (col > 0);
    const bool has_right  = (col + 1 < words_per_row);
    const bool has_top    = (row > 0);
    const bool has_bottom = (row + 1 < grid_dim);

    const int idx      = tid;
    const int idxL     = idx - 1;
    const int idxR     = idx + 1;
    const int idxT     = idx - words_per_row;
    const int idxB     = idx + words_per_row;
    const int idxTL    = idxT - 1;
    const int idxTR    = idxT + 1;
    const int idxBL    = idxB - 1;
    const int idxBR    = idxB + 1;

    // Load the 9 relevant words (center + 8 neighbors), substituting 0 when out of bounds.
    // Using simple branches here; only threads on boundaries will diverge.
    std::uint64_t center = input[idx];

    std::uint64_t left   = has_left  ? input[idxL]  : 0ull;
    std::uint64_t right  = has_right ? input[idxR]  : 0ull;

    std::uint64_t top        = has_top    ? input[idxT]  : 0ull;
    std::uint64_t bottom     = has_bottom ? input[idxB]  : 0ull;

    std::uint64_t top_left     = (has_top && has_left)     ? input[idxTL] : 0ull;
    std::uint64_t top_right    = (has_top && has_right)    ? input[idxTR] : 0ull;
    std::uint64_t bottom_left  = (has_bottom && has_left)  ? input[idxBL] : 0ull;
    std::uint64_t bottom_right = (has_bottom && has_right) ? input[idxBR] : 0ull;

    // Compute horizontally shifted masks with cross-word carries for top/mid/bottom rows.
    // These represent the diagonal and horizontal neighbor bitfields aligned to the center cell's bit position.
    const std::uint64_t tl = shl1_with_carry(top,    top_left);    // top-left neighbors
    const std::uint64_t t  = top;                                  // top neighbors
    const std::uint64_t tr = shr1_with_carry(top,    top_right);   // top-right neighbors

    const std::uint64_t l  = shl1_with_carry(center, left);        // left neighbors
    const std::uint64_t r  = shr1_with_carry(center, right);       // right neighbors

    const std::uint64_t bl = shl1_with_carry(bottom, bottom_left); // bottom-left neighbors
    const std::uint64_t b  = bottom;                               // bottom neighbors
    const std::uint64_t br = shr1_with_carry(bottom, bottom_right);// bottom-right neighbors

    // Stage 1: 3-input adders per row of neighbors.
    // Top triple (tl, t, tr)
    std::uint64_t s_top, c_top;
    add3_u64(tl, t, tr, s_top, c_top);

    // Middle pair (l, r) => same as add3 with third input 0
    const std::uint64_t s_mid = l ^ r;       // ones place
    const std::uint64_t c_mid = l & r;       // twos place

    // Bottom triple (bl, b, br)
    std::uint64_t s_bot, c_bot;
    add3_u64(bl, b, br, s_bot, c_bot);

    // Stage 2: Add the ones bits (s_top, s_mid, s_bot) -> n1 and carry into twos
    std::uint64_t n1, carry_ones_to_twos;
    add3_u64(s_top, s_mid, s_bot, n1, carry_ones_to_twos);  // n1 is the bit-0 of total neighbors

    // Stage 3: Combine twos contributions from:
    //   c_top, c_mid, c_bot, and carry_ones_to_twos (all are boolean bitfields)
    // We need n2 (bit-1) and n4 (bit-2) of the final neighbor count.
    // Sum four 1-bit values using pairwise adders:
    const std::uint64_t p1_sum   = c_top ^ c_mid;
    const std::uint64_t p1_carry = c_top & c_mid;

    const std::uint64_t p2_sum   = c_bot ^ carry_ones_to_twos;
    const std::uint64_t p2_carry = c_bot & carry_ones_to_twos;

    // Bit-1 (twos bit) is the parity of the four inputs:
    const std::uint64_t n2 = p1_sum ^ p2_sum;

    // Bit-2 (fours bit) is the parity of the number of carries when summing those four bits:
    // parity_of((p1_sum & p2_sum) + p1_carry + p2_carry)
    const std::uint64_t n4 = (p1_sum & p2_sum) ^ p1_carry ^ p2_carry;

    // Game of Life rule:
    // next = (neighbors == 3) | (alive & (neighbors == 2))
    // neighbors == 3 iff n4 == 0 and n2 == 1 and n1 == 1
    // neighbors == 2 iff n4 == 0 and n2 == 1 and n1 == 0
    const std::uint64_t not_n4 = ~n4;
    const std::uint64_t eq3 = (n2 & n1) & not_n4;
    const std::uint64_t eq2 = (n2 & ~n1) & not_n4;

    const std::uint64_t next = eq3 | (center & eq2);

    output[idx] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is the square grid width/height in cells (power of 2, >512).
    // Each 64-bit word stores 64 cells in a row.
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    const int total_words = words_per_row * grid_dimensions;

    // Launch configuration: one thread per 64-bit word.
    // 256 threads per block is a good default on A100/H100; adjust if desired.
    const int block_size = 256;
    const int grid_size = (total_words + block_size - 1) / block_size;

    game_of_life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row);
}