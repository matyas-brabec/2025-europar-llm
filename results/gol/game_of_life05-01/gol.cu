#include <cstdint>
#include <cuda_runtime.h>

// CUDA kernel implementing one generation step of Conway's Game of Life on a bit-packed grid.
// Each thread processes one 64-bit word (64 cells) and updates the corresponding output word.
// The grid is square with size grid_dim x grid_dim cells, and its width is a power of two.
// Words are laid out row-major; each row has words_per_row = grid_dim / 64 words.
// Outside-grid cells are treated as dead (0).
// The algorithm uses bit-sliced counters to sum neighbor contributions for all 64 cells at once.
// This avoids cell-by-cell processing and eliminates atomics.
namespace {

static __device__ __forceinline__ void add_mask(uint64_t m, uint64_t& b0, uint64_t& b1, uint64_t& b2) {
    // Adds a 1-bit mask 'm' to the 3-bit bit-sliced counters (b2 b1 b0) per bit-position.
    // This is a ripple-carry add per bit (mod 8), but vectorized over 64 bit-positions in parallel.
    // b0 ^= m;
    // carry0 = old_b0 & m;
    // b1 ^= carry0;
    // carry1 = old_b1 & carry0;
    // b2 ^= carry1;    // (mod 8; overflow beyond b2 is ignored since max neighbors is 8)
    uint64_t c0 = b0 & m;
    b0 ^= m;
    uint64_t c1 = b1 & c0;
    b1 ^= c0;
    b2 ^= c1;
}

__global__ void gol_kernel_bitpacked(const uint64_t* __restrict__ in,
                                     uint64_t* __restrict__ out,
                                     int grid_dim,
                                     int words_per_row,
                                     unsigned int total_words)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;

    // Compute column within the row (word index within the row). words_per_row is power of 2.
    unsigned int col = tid & (static_cast<unsigned int>(words_per_row) - 1);

    // Boundary flags
    bool hasL = (col > 0);
    bool hasR = (col + 1u) < static_cast<unsigned int>(words_per_row);
    bool hasN = (tid >= static_cast<unsigned int>(words_per_row));
    bool hasS = (tid + static_cast<unsigned int>(words_per_row)) < total_words;

    // Base neighbor indices
    unsigned int idxL = tid - 1u;
    unsigned int idxR = tid + 1u;
    unsigned int idxN = tid - static_cast<unsigned int>(words_per_row);
    unsigned int idxS = tid + static_cast<unsigned int>(words_per_row);

    // Load the 3x3 neighborhood of words (center + left/right in current row, and the three above/below).
    // Missing neighbors outside the grid are treated as zero.
    uint64_t c  = in[tid];
    uint64_t l  = hasL ? in[idxL] : 0ull;
    uint64_t r  = hasR ? in[idxR] : 0ull;

    uint64_t nC = hasN ? in[idxN] : 0ull;
    uint64_t sC = hasS ? in[idxS] : 0ull;

    uint64_t nL = (hasN && hasL) ? in[idxN - 1u] : 0ull;
    uint64_t nR = (hasN && hasR) ? in[idxN + 1u] : 0ull;
    uint64_t sL = (hasS && hasL) ? in[idxS - 1u] : 0ull;
    uint64_t sR = (hasS && hasR) ? in[idxS + 1u] : 0ull;

    // Compute neighbor masks aligned to the current 64-bit word bit positions
    // Horizontal (same row)
    uint64_t W  = (c << 1) | (l >> 63);   // west neighbors
    uint64_t E  = (c >> 1) | (r << 63);   // east neighbors

    // Vertical (north/south, no horizontal offset)
    uint64_t N  = nC;
    uint64_t S  = sC;

    // Diagonals
    uint64_t NW = (nC << 1) | (nL >> 63);
    uint64_t NE = (nC >> 1) | (nR << 63);
    uint64_t SW = (sC << 1) | (sL >> 63);
    uint64_t SE = (sC >> 1) | (sR << 63);

    // Bit-sliced accumulation of 8 neighbor masks into three bitplanes b0 (LSB), b1, b2 (counts modulo 8)
    uint64_t b0 = 0ull, b1 = 0ull, b2 = 0ull;
    add_mask(NW, b0, b1, b2);
    add_mask(N,  b0, b1, b2);
    add_mask(NE, b0, b1, b2);
    add_mask(W,  b0, b1, b2);
    add_mask(E,  b0, b1, b2);
    add_mask(SW, b0, b1, b2);
    add_mask(S,  b0, b1, b2);
    add_mask(SE, b0, b1, b2);

    // Determine cells with exactly 2 or 3 neighbors using the bit-sliced counter (mod 8).
    // eq3: ~b2 & b1 & b0  (binary 011)
    // eq2: ~b2 & b1 & ~b0 (binary 010)
    uint64_t nb2 = ~b2;
    uint64_t eq3 = nb2 & b1 & b0;
    uint64_t eq2 = nb2 & b1 & (~b0);

    // Apply Conway's rules:
    // - Any cell with exactly 3 neighbors becomes alive.
    // - Any alive cell with exactly 2 neighbors stays alive.
    uint64_t next = eq3 | (c & eq2);

    out[tid] = next;
}

} // namespace

// Host function to run a single Game of Life step.
// input/output are device pointers allocated by cudaMalloc.
// grid_dimensions is the number of cells per row/column (grid is square), power of two.
// Each 64-bit word in input/output encodes 64 consecutive cells in a row (bit 0 is the least significant bit).
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Number of 64-bit words per row (grid_dimensions is guaranteed to be a power of two)
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    const unsigned int total_words = static_cast<unsigned int>(words_per_row) * static_cast<unsigned int>(grid_dimensions);

    // Launch parameters. Adjust block size for target GPU if needed.
    const int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1u) / block_size);

    gol_kernel_bitpacked<<<grid_size, block_size>>>(input, output, grid_dimensions, words_per_row, total_words);
}