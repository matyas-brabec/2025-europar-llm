#include <cstdint>
#include <cuda_runtime.h>

/*
  CUDA implementation of one iteration (generation) of Conway's Game of Life
  over a bit-packed square grid. Each std::uint64_t encodes 64 horizontal cells.

  Key points:
  - One CUDA thread processes one 64-bit word (i.e., 64 cells) to avoid atomics.
  - Grid is square with size N x N, where N is a power of two (> 512).
  - Cells outside the grid are treated as dead (zero); no wrap-around.
  - For each word, the kernel reads up to nine 64-bit words: the 3x3 neighborhood
    at word granularity (left/center/right x above/current/below rows).
  - Horizontal neighbor bits that cross 64-bit word boundaries are handled by
    OR-ing in the appropriate boundary bit from the adjacent word.
  - Neighbor counts (0..8) are accumulated using bit-sliced ripple-carry addition
    across 64 lanes in parallel (ones, twos, fours, eights bit-planes).
  - Next-state computation uses:
        new = (neighbors == 3) | (alive & (neighbors == 2))
    which is evaluated per bit using the bit-planes.

  Shared and texture memory are intentionally not used as they provide no benefit
  for this highly bitwise, bandwidth-bound workload with the given constraints.
*/

namespace {

using u64 = std::uint64_t;

// Accumulate one 1-bit-per-lane operand v into the running bit-sliced sum represented
// by (ones, twos, fours, eights). This is a ripple-carry adder operating independently
// on all 64 lanes using bitwise logic.
// - ones   holds the least significant bit (2^0) of the per-lane neighbor counts.
// - twos   holds the 2^1 bit.
// - fours  holds the 2^2 bit.
// - eights holds the 2^3 bit (set only when count reaches 8).
__device__ __forceinline__ void accum1(u64 v, u64& ones, u64& twos, u64& fours, u64& eights)
{
    // Add v to (ones, twos, fours), propagating carries; any carry out of fours sets eights.
    u64 t1 = ones ^ v;
    u64 c1 = ones & v;

    u64 t2 = twos ^ c1;
    u64 c2 = twos & c1;

    u64 t3 = fours ^ c2;
    u64 c3 = fours & c2;

    eights |= c3;
    ones = t1;
    twos = t2;
    fours = t3;
}

// Load helpers: safe fetch with boundary checks. Outside-grid words are treated as zero.
__device__ __forceinline__ u64 load_or_zero(const u64* __restrict__ base, std::size_t idx, bool in_bounds)
{
    return in_bounds ? base[idx] : u64(0);
}

// Compute horizontal West/East neighbor bitfields for a given triple (left, center, right) words.
// west_bits: for each bit i in center, contains the neighbor at i-1 (with cross-word handling).
// east_bits: for each bit i in center, contains the neighbor at i+1 (with cross-word handling).
__device__ __forceinline__ void horizontal_neighbors(u64 left, u64 center, u64 right, u64& west_bits, u64& east_bits)
{
    // West neighbor of bit i is original bit (i-1) in the row:
    //  - within-word: (center << 1) shifts bit i-1 into position i
    //  - cross-word for i == 0: bring in bit 63 of the left word into bit 0 via (left >> 63)
    west_bits = (center << 1) | (left >> 63);

    // East neighbor of bit i is original bit (i+1) in the row:
    //  - within-word: (center >> 1) shifts bit i+1 into position i
    //  - cross-word for i == 63: bring in bit 0 of the right word into bit 63 via (right << 63)
    east_bits = (center >> 1) | (right << 63);
}

// Kernel: compute one Game of Life step on a bit-packed grid.
// words_per_row: grid_dimensions / 64
// height: grid_dimensions
__global__ void gol_step_kernel(const u64* __restrict__ input,
                                u64* __restrict__ output,
                                int words_per_row,
                                int height)
{
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(height);
    const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_words) return;

    const int y = static_cast<int>(gid / words_per_row);
    const int x = static_cast<int>(gid % words_per_row);

    const std::size_t base = static_cast<std::size_t>(y) * static_cast<std::size_t>(words_per_row);

    // Fast interior-path flag: no boundary checks required when strictly inside the grid.
    const bool interior = (x > 0) && (x + 1 < words_per_row) && (y > 0) && (y + 1 < height);

    // Fetch the 3x3 neighborhood of 64-bit words around (y, x)
    u64 upL = 0, upC = 0, upR = 0;
    u64 midL = 0, midC = 0, midR = 0;
    u64 dnL = 0, dnC = 0, dnR = 0;

    if (interior) {
        const std::size_t base_up = base - static_cast<std::size_t>(words_per_row);
        const std::size_t base_dn = base + static_cast<std::size_t>(words_per_row);
        upL  = input[base_up + (x - 1)];
        upC  = input[base_up + x];
        upR  = input[base_up + (x + 1)];
        midL = input[base + (x - 1)];
        midC = input[base + x];
        midR = input[base + (x + 1)];
        dnL  = input[base_dn + (x - 1)];
        dnC  = input[base_dn + x];
        dnR  = input[base_dn + (x + 1)];
    } else {
        const bool has_up = (y > 0);
        const bool has_dn = (y + 1 < height);
        const bool has_left = (x > 0);
        const bool has_right = (x + 1 < words_per_row);

        const std::size_t base_up = has_up ? (base - static_cast<std::size_t>(words_per_row)) : 0;
        const std::size_t base_dn = has_dn ? (base + static_cast<std::size_t>(words_per_row)) : 0;

        upL  = load_or_zero(input, base_up + (x - 1), has_up && has_left);
        upC  = load_or_zero(input, base_up + x,       has_up);
        upR  = load_or_zero(input, base_up + (x + 1), has_up && has_right);

        midL = load_or_zero(input, base + (x - 1),    has_left);
        midC = input[base + x];
        midR = load_or_zero(input, base + (x + 1),    has_right);

        dnL  = load_or_zero(input, base_dn + (x - 1), has_dn && has_left);
        dnC  = load_or_zero(input, base_dn + x,       has_dn);
        dnR  = load_or_zero(input, base_dn + (x + 1), has_dn && has_right);
    }

    // Build the eight neighbor bitfields:
    // For rows above and below, include west, center, and east neighbors.
    // For the current row, include only west and east (exclude the center cell).
    u64 up_w, up_e, mid_w, mid_e, dn_w, dn_e;
    horizontal_neighbors(upL,  upC,  upR,  up_w,  up_e);
    horizontal_neighbors(midL, midC, midR, mid_w, mid_e);
    horizontal_neighbors(dnL,  dnC,  dnR,  dn_w,  dn_e);

    const u64 up_c  = upC;
    const u64 dn_c  = dnC;
    const u64 alive = midC;  // current cell states; used for survival rule

    // Accumulate neighbor counts (8 sources, exclude self 'midC'):
    u64 ones = 0, twos = 0, fours = 0, eights = 0;
    accum1(up_w,  ones, twos, fours, eights);
    accum1(up_c,  ones, twos, fours, eights);
    accum1(up_e,  ones, twos, fours, eights);
    accum1(mid_w, ones, twos, fours, eights);
    accum1(mid_e, ones, twos, fours, eights);
    accum1(dn_w,  ones, twos, fours, eights);
    accum1(dn_c,  ones, twos, fours, eights);
    accum1(dn_e,  ones, twos, fours, eights);

    // Compute masks for (neighbors == 2) and (neighbors == 3) using the bit-planes:
    // - eq2:  ones=0, twos=1, fours=0, eights=0
    // - eq3:  ones=1, twos=1, fours=0, eights=0
    const u64 not_ones   = ~ones;
    const u64 not_fours  = ~fours;
    const u64 not_eights = ~eights;

    const u64 eq2 = (not_ones & twos & not_fours & not_eights);
    const u64 eq3 = (ones     & twos & not_fours & not_eights);

    const u64 next = eq3 | (alive & eq2);

    output[gid] = next;
}

} // anonymous namespace

// Host entry point: perform one step of Conway's Game of Life over a bit-packed grid.
// - input:  device pointer to N*N/64 words (row-major, 64 cells per word).
// - output: device pointer to N*N/64 words for the next generation.
// - grid_dimensions: N (width == height), power-of-two greater than 512.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // divide by 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Choose a launch configuration that maps one thread per 64-bit word.
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);

    gol_step_kernel<<<blocks, threads_per_block>>>(input, output, words_per_row, grid_dimensions);
    // Synchronization is intentionally omitted; caller is responsible for it.
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
