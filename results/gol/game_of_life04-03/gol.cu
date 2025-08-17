#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/*
  CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.

  Design choices and algorithmic details:

  - Each thread processes exactly one 64-bit word (64 cells in a row).
  - The grid is square with dimensions grid_dimensions x grid_dimensions (power of 2, >512).
  - The input and output grids are bit-packed row-major arrays of std::uint64_t.
  - Out-of-bounds (outside the grid) cells are treated as dead (0) — no wrapping.
  - We avoid shared/texture memory as per instructions; rely on global memory and L2/L1 caches.
  - Neighbor counting is done using bitwise "SIMD within a register" techniques:
      * Build eight 64-bit neighbor masks: N, S, E, W, NE, NW, SE, SW.
      * Compute horizontal sums per row using 3-input adders (bitwise half/full adders) to produce
        two bit-planes (ones and twos) for each row.
      * Vertically add those per-row sums to obtain the final lower-three bits (1,2,4 planes) of the
        neighbor counts (0..8). This avoids per-cell branching and explicit per-cell popcounts.
      * The next-state rule reduces to:
            next = birth | survive
            birth   = (count == 3)  = ones & twos & ~fours
            survive = alive & (count == 2 or 3) = alive & twos & ~fours
        where ones, twos, fours are the 1,2,4 bit-planes of the neighbor count per bit position.

  - Horizontal shifts with cross-word handling:
      * E = (cur >> 1) | (curR << 63)
      * W = (cur << 1) | (curL >> 63)
      * NE = (north >> 1) | (northR << 63), etc.
    Edge words (at col 0 or last col) use zero for missing neighbors.

  - Indexing:
      * width_words = grid_dimensions / 64 (also a power-of-two).
      * We pass width_log2 = log2(width_words) so row = idx >> width_log2, col = idx & (width_words - 1),
        avoiding integer division/modulo in device code.

  - Launch:
      * 1D grid, grid-stride loop to cover arbitrary sizes.
      * Thread block size 256 chosen as a good balance for register usage and occupancy on A100/H100.

  This implementation emphasizes bitwise parallelism and coalesced global access for high throughput.
*/

static inline int ilog2_u32_host(unsigned int x)
{
    // x is a power-of-two (>0), compute floor(log2(x)) portably.
    int r = 0;
    while ((1u << r) < x) ++r;
    return r;
}

__device__ __forceinline__ void sum3_bits_u64(const std::uint64_t a,
                                              const std::uint64_t b,
                                              const std::uint64_t c,
                                              std::uint64_t& ones,
                                              std::uint64_t& twos)
{
    // Per-bit sum of three 1-bit inputs yields:
    //   value = ones + 2*twos,
    // where:
    //   ones = a ^ b ^ c
    //   twos = (a&b) | (a&c) | (b&c)
    ones = a ^ b ^ c;
    twos = (a & b) | (a & c) | (b & c);
}

__global__ void life_kernel_bitpacked(const std::uint64_t* __restrict__ in,
                                      std::uint64_t* __restrict__ out,
                                      unsigned int width_words,
                                      unsigned int height_rows,
                                      unsigned int width_log2)
{
    const unsigned int width_mask = width_words - 1u;

    // Grid-stride loop to cover all words.
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (std::size_t)width_words * (std::size_t)height_rows;
         idx += (std::size_t)blockDim.x * (std::size_t)gridDim.x)
    {
        // Compute row and column in word coordinates using power-of-two properties.
        const unsigned int row = static_cast<unsigned int>(idx >> width_log2);
        const unsigned int col = static_cast<unsigned int>(idx & width_mask);

        // Load current word and its horizontal neighbors in the same row.
        const std::uint64_t cur  = in[idx];
        const std::uint64_t curL = (col > 0u) ? in[idx - 1] : 0ull;
        const std::uint64_t curR = (col + 1u < width_words) ? in[idx + 1] : 0ull;

        // Load words from the row above (north) and below (south), including their horizontal neighbors.
        std::uint64_t north  = 0ull, northL = 0ull, northR = 0ull;
        std::uint64_t south  = 0ull, southL = 0ull, southR = 0ull;

        if (row > 0u) {
            const std::size_t idxN = idx - width_words;
            north  = in[idxN];
            northL = (col > 0u) ? in[idxN - 1] : 0ull;
            northR = (col + 1u < width_words) ? in[idxN + 1] : 0ull;
        }
        if (row + 1u < height_rows) {
            const std::size_t idxS = idx + width_words;
            south  = in[idxS];
            southL = (col > 0u) ? in[idxS - 1] : 0ull;
            southR = (col + 1u < width_words) ? in[idxS + 1] : 0ull;
        }

        // Build directional neighbor masks with cross-word handling for bit 0 and bit 63.
        const std::uint64_t N  = north;
        const std::uint64_t S  = south;
        const std::uint64_t E  = (cur >> 1)   | (curR << 63);
        const std::uint64_t W  = (cur << 1)   | (curL >> 63);
        const std::uint64_t NE = (north >> 1) | (northR << 63);
        const std::uint64_t NW = (north << 1) | (northL >> 63);
        const std::uint64_t SE = (south >> 1) | (southR << 63);
        const std::uint64_t SW = (south << 1) | (southL >> 63);

        // Horizontal sums per row (N, C, S):
        // For N row: sum of (NW, N, NE)
        std::uint64_t n_ones, n_twos;
        sum3_bits_u64(NW, N, NE, n_ones, n_twos);

        // For current row: sum of (W, E). Treat as 3-input with third = 0 -> ones=W^E, twos=W&E
        const std::uint64_t c_ones = W ^ E;
        const std::uint64_t c_twos = W & E;

        // For S row: sum of (SW, S, SE)
        std::uint64_t s_ones, s_twos;
        sum3_bits_u64(SW, S, SE, s_ones, s_twos);

        // Vertical addition of the three row-sums to obtain the lower bits of neighbor count.
        // First add the ones bits across rows -> ones (bit0) and carry into twos.
        const std::uint64_t v_ones = n_ones ^ c_ones ^ s_ones;  // final ones bit (bit0)
        const std::uint64_t carry1 = (n_ones & c_ones) | (n_ones & s_ones) | (c_ones & s_ones); // carry into bit1

        // Now add the twos units (n_twos, c_twos, s_twos) and the carry1 into the twos place.
        const std::uint64_t twos_parity = n_twos ^ c_twos ^ s_twos;    // parity of twos units
        const std::uint64_t twos_pairs  = (n_twos & c_twos) | (n_twos & s_twos) | (c_twos & s_twos); // >=2 twos units
        const std::uint64_t v_twos      = twos_parity ^ carry1;        // final twos bit (bit1)
        const std::uint64_t carry2extra = twos_parity & carry1;        // extra carry to fours if both are 1
        const std::uint64_t v_fours     = twos_pairs | carry2extra;    // fours bit (bit2). Higher bits not needed for rules.

        // Apply Game of Life rules:
        const std::uint64_t alive   = cur;
        const std::uint64_t birth   = v_ones & v_twos & ~v_fours;    // exactly 3 neighbors
        const std::uint64_t survive = alive & v_twos & ~v_fours;     // exactly 2 or 3 neighbors
        const std::uint64_t next    = birth | survive;

        out[idx] = next;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // grid_dimensions is a power-of-two, > 512. Compute width in 64-bit words per row.
    const unsigned int height_rows = static_cast<unsigned int>(grid_dimensions);
    const unsigned int width_words = static_cast<unsigned int>(grid_dimensions) >> 6; // divide by 64

    // Compute log2(width_words) for fast row/col computation in the kernel.
    const unsigned int width_log2 = static_cast<unsigned int>(ilog2_u32_host(width_words));

    // Compute launch configuration.
    const std::size_t total_words = static_cast<std::size_t>(width_words) * static_cast<std::size_t>(height_rows);

    // Choose a reasonable block size for A100/H100; 256 is generally a good balance for this kernel.
    const int threads_per_block = 256;

    // Use enough blocks to cover the data; cap blocks to a large value and rely on grid-stride loop.
    // Modern GPUs support very large grid sizes, but we keep it robust.
    int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);
    const int max_blocks = 65535; // conservative cap for 1D grid on older HW; grid-stride loop handles the rest.
    if (blocks > max_blocks) blocks = max_blocks;

    life_kernel_bitpacked<<<blocks, threads_per_block>>>(input, output, width_words, height_rows, width_log2);
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
