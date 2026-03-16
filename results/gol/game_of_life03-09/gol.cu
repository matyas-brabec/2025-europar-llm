#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// High-performance one-step Conway kernel for a bit-packed square grid.
// - One thread owns one 64-bit word = 64 cells, so no atomics are needed.
// - Shared/texture memory is intentionally avoided. The only profitable cross-thread reuse here is
//   horizontal word reuse inside a warp, and warp shuffles provide that more cheaply than shared memory.
// - Input and output are assumed to be distinct device buffers.

namespace {

constexpr unsigned int kCellsPerWordLog2 = 6u;  // 64 cells per packed word.

// Use an explicit 64-bit type that is guaranteed to work with 64-bit warp shuffle overloads.
// The public API still uses std::uint64_t.
using word_t = unsigned long long;
static_assert(sizeof(word_t) == sizeof(std::uint64_t), "Expected 64-bit packed words.");

// Host-side helper: x is always a power of two under the problem constraints.
inline unsigned int log2_pow2_u32(unsigned int x) {
    unsigned int s = 0u;
    while (x > 1u) {
        x >>= 1u;
        ++s;
    }
    return s;
}

// Bitwise half/full adders. The compiler maps these boolean forms efficiently on modern NVIDIA GPUs.
__device__ __forceinline__ void half_adder(word_t a, word_t b, word_t& sum, word_t& carry) {
    sum = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ void full_adder(word_t a, word_t b, word_t c, word_t& sum, word_t& carry) {
    const word_t ab_xor = a ^ b;
    sum = ab_xor ^ c;
    carry = (a & b) | (c & ab_xor);
}

// Update one packed 64-cell word.
// Bit layout follows the prompt's natural mapping: bit i stores cell x+i within the row.
//
// Neighbor alignment for each bit position:
//   west  = (center << 1) | (left  >> 63)
//   east  = (center >> 1) | (right << 63)
// and likewise for the rows above and below.
//
// The 8 neighbor bitboards are reduced with a small carry-save tree so all 64 cells in the word are
// updated in parallel. If the final neighbor count is written as
//   count = b0 + 2*m
// then the Game of Life rule only cares about count == 2 or 3, i.e. m == 1. When m == 1, b0 tells
// 2 from 3, so
//   next = (m == 1) & (alive | b0).
__device__ __forceinline__ word_t evolve_word(word_t center,
                                              word_t left, word_t right,
                                              word_t above_left, word_t above, word_t above_right,
                                              word_t below_left, word_t below, word_t below_right) {
    word_t s0, c0;
    full_adder((above << 1) | (above_left >> 63),
               above,
               (above >> 1) | (above_right << 63),
               s0, c0);

    word_t s1, c1;
    full_adder((center << 1) | (left >> 63),
               (center >> 1) | (right << 63),
               (below << 1) | (below_left >> 63),
               s1, c1);

    word_t s2, c2;
    half_adder(below,
               (below >> 1) | (below_right << 63),
               s2, c2);

    word_t b0, sc;
    full_adder(s0, s1, s2, b0, sc);

    word_t t0, t1;
    full_adder(sc, c0, c1, t0, t1);

    const word_t count_is_2_or_3 = (t0 ^ c2) & ~t1;
    return count_is_2_or_3 & (center | b0);
}

template<int BX, int BY>
__global__ __launch_bounds__(256)
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         unsigned int grid_dim,
                         unsigned int words_per_row,
                         unsigned int tiles_per_row_mask,
                         unsigned int tiles_per_row_shift) {
    static_assert(BX == 16 || BX == 32, "BX must be 16 or 32 to match supported shuffle widths.");
    static_assert(BX * BY <= 256, "Block size must fit the launch-bounds contract.");

    // Flatten the logical 2D tile grid into blockIdx.x so very large grids do not depend on grid.y.
    const unsigned int tile_linear = blockIdx.x;
    const unsigned int tile_x = tile_linear & tiles_per_row_mask;
    const unsigned int tile_y = tile_linear >> tiles_per_row_shift;

    const unsigned int word_x = tile_x * BX + threadIdx.x;
    const unsigned int row = tile_y * BY + threadIdx.y;

    const std::size_t idx =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row) +
        static_cast<std::size_t>(word_x);

    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(words_per_row);
    const std::uint64_t* const p = input + idx;

    const unsigned int last_word = words_per_row - 1u;
    const unsigned int last_row = grid_dim - 1u;

    // Because words_per_row is a power of two and BX exactly tiles the row, true row edges always
    // coincide with subgroup edges. Only subgroup-edge lanes ever need special left/right handling.
    const bool has_left = word_x > 0u;
    const bool has_right = word_x < last_word;
    const bool has_up = row > 0u;
    const bool has_down = row < last_row;

    // Load only the center word from the three relevant rows.
    const word_t center = static_cast<word_t>(p[0]);

    const std::uint64_t* up_ptr = nullptr;
    const std::uint64_t* down_ptr = nullptr;

    word_t above = 0ull;
    word_t below = 0ull;

    if (has_up) {
        up_ptr = p - stride;
        above = static_cast<word_t>(up_ptr[0]);
    }
    if (has_down) {
        down_ptr = p + stride;
        below = static_cast<word_t>(down_ptr[0]);
    }

    // Horizontal neighbor words come from warp shuffles.
    // - BX == 32: one subgroup == one warp row segment.
    // - BX == 16: the width argument partitions a warp into two independent half-warps.
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    constexpr unsigned int kLastLane = static_cast<unsigned int>(BX - 1);

    word_t left = __shfl_up_sync(kFullWarpMask, center, 1, BX);
    word_t right = __shfl_down_sync(kFullWarpMask, center, 1, BX);

    word_t above_left = __shfl_up_sync(kFullWarpMask, above, 1, BX);
    word_t above_right = __shfl_down_sync(kFullWarpMask, above, 1, BX);

    word_t below_left = __shfl_up_sync(kFullWarpMask, below, 1, BX);
    word_t below_right = __shfl_down_sync(kFullWarpMask, below, 1, BX);

    const unsigned int lane = threadIdx.x;

    // Subgroup-edge lanes need halo words from global memory because shuffles do not cross subgroup
    // boundaries. This also naturally handles true grid boundaries by substituting zero.
    if (lane == 0u) {
        left = 0ull;
        above_left = 0ull;
        below_left = 0ull;

        if (has_left) {
            left = static_cast<word_t>(p[-1]);
            if (has_up) {
                above_left = static_cast<word_t>(up_ptr[-1]);
            }
            if (has_down) {
                below_left = static_cast<word_t>(down_ptr[-1]);
            }
        }
    }

    if (lane == kLastLane) {
        right = 0ull;
        above_right = 0ull;
        below_right = 0ull;

        if (has_right) {
            right = static_cast<word_t>(p[1]);
            if (has_up) {
                above_right = static_cast<word_t>(up_ptr[1]);
            }
            if (has_down) {
                below_right = static_cast<word_t>(down_ptr[1]);
            }
        }
    }

    const word_t next =
        evolve_word(center, left, right, above_left, above, above_right, below_left, below, below_right);

    output[idx] = static_cast<std::uint64_t>(next);
}

template<int BX, int BY>
inline void launch_game_of_life_kernel(const std::uint64_t* input,
                                       std::uint64_t* output,
                                       unsigned int grid_dim,
                                       unsigned int words_per_row) {
    const unsigned int tiles_per_row = words_per_row / BX;  // Exact: both are powers of two.
    const unsigned int tiles_per_col = grid_dim / BY;       // Exact: BY is also a power of two.
    const unsigned int tiles_per_row_shift = log2_pow2_u32(tiles_per_row);

    const std::uint64_t total_tiles =
        static_cast<std::uint64_t>(tiles_per_row) * static_cast<std::uint64_t>(tiles_per_col);

    game_of_life_kernel<BX, BY><<<static_cast<unsigned int>(total_tiles), dim3(BX, BY, 1)>>>(
        input,
        output,
        grid_dim,
        words_per_row,
        tiles_per_row - 1u,
        tiles_per_row_shift);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller owns synchronization and error handling; this function only launches the kernel.
    const unsigned int grid_dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = grid_dim >> kCellsPerWordLog2;

    // Smallest legal grid: 1024x1024 => 16 packed words per row.
    // Use 16-lane subgroups there, but only 8 rows per block so the launch still exposes enough blocks
    // to occupy a large GPU. All larger grids use the common 32-lane path.
    if (words_per_row == 16u) {
        launch_game_of_life_kernel<16, 8>(input, output, grid_dim, words_per_row);
    } else {
        launch_game_of_life_kernel<32, 8>(input, output, grid_dim, words_per_row);
    }
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
