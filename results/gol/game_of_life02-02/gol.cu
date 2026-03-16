#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Conway's Game of Life, one step, on a square bit-packed grid.
//
// Encoding:
//   - Row-major.
//   - Each 64-bit word stores 64 consecutive cells from one row.
//   - Bit k of a word is cell x = word_base + k.
//   - Cells outside the grid are dead.
//
// Performance strategy:
//   - One thread computes one 64-cell output word.
//   - Shared memory is intentionally not used: warp shuffles eliminate the hot horizontal
//     reuse, and the remaining accesses are regular read-only global loads.
//   - The update is fully bit-parallel: each thread computes 64 next-state cells at once
//     with a small fixed Boolean network instead of per-cell scalar work.
//   - A 1D grid is used so very tall domains do not hit CUDA's grid.y limit.

namespace {

using u64 = std::uint64_t;

constexpr int kThreadsPerBlock = 256;
constexpr unsigned int kFullMask = 0xFFFFFFFFu;

static_assert(sizeof(u64) == 8, "Expected 64-bit words.");

__device__ __forceinline__ u64 majority3(u64 a, u64 b, u64 c) {
    // Per bit position: 1 iff at least two of {a,b,c} are 1.
    return (a & b) | (c & (a ^ b));
}

__device__ __forceinline__ u64 exactly_one_of_four(u64 a, u64 b, u64 c, u64 d) {
    // Per bit position, test "exactly one of four 1-bit values is set".
    //
    // Group the four inputs as (a,b) and (c,d):
    //   - parity == 1 for populations 1 or 3
    //   - any population of 3 must fully occupy at least one pair
    // Therefore:
    //   exactly_one = parity & ~pair_has_two
    return (a ^ b ^ c ^ d) & ~((a & b) | (c & d));
}

__device__ __forceinline__ u64 shift_west(u64 center, u64 left_word) {
    // Bit k represents x = base + k, so the west neighbor (x-1) aligns with x by shifting
    // toward larger bit indices.
    return (center << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 shift_east(u64 center, u64 right_word) {
    // The east neighbor (x+1) aligns with x by shifting toward smaller bit indices.
    return (center >> 1) | (right_word << 63);
}

static inline unsigned int log2_pow2(unsigned int v) {
    // Host-side helper. Input is guaranteed to be a power of two.
    unsigned int s = 0;
    while (v > 1u) {
        v >>= 1;
        ++s;
    }
    return s;
}

template<int STRIP_WORDS>
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         int words_per_row,
                         int grid_dimensions,
                         unsigned int blocks_per_row_shift,
                         unsigned int blocks_per_row_mask) {
    static_assert(STRIP_WORDS == 16 || STRIP_WORDS == 32,
                  "Only 16-word and 32-word strips are used.");

    // 256-thread blocks:
    //   - STRIP_WORDS == 32 -> block = (32, 8), one warp per row strip
    //   - STRIP_WORDS == 16 -> block = (16, 16), one half-warp per row strip
    constexpr int BLOCK_ROWS = kThreadsPerBlock / STRIP_WORDS;
    constexpr int LAST_LANE = STRIP_WORDS - 1;

    // Flattened 2D tiling:
    //   block_col = low bits of blockIdx.x
    //   block_row = high bits of blockIdx.x
    // blocks_per_row is a power of two, so decode is just mask + shift.
    const unsigned int block_linear = blockIdx.x;
    const int block_col = static_cast<int>(block_linear & blocks_per_row_mask);
    const int block_row = static_cast<int>(block_linear >> blocks_per_row_shift);

    const int x = block_col * STRIP_WORDS + static_cast<int>(threadIdx.x);  // packed-word x
    const int y = block_row * BLOCK_ROWS + static_cast<int>(threadIdx.y);   // row y

    const std::size_t row_stride = static_cast<std::size_t>(words_per_row);
    const std::size_t idx =
        static_cast<std::size_t>(y) * row_stride + static_cast<std::size_t>(x);

    const int last_x = words_per_row - 1;
    const int last_y = grid_dimensions - 1;

    const bool has_top = (y != 0);
    const bool has_bottom = (y != last_y);
    const bool has_left = (x != 0);
    const bool has_right = (x != last_x);

    // Every thread loads only the center word of the top/middle/bottom rows.
    // Left/right neighboring words are normally obtained from adjacent lanes by shuffles.
    // Only strip boundaries (lane 0 / lane LAST_LANE) require an extra global load.
    const u64 mid_c = input[idx];
    const u64 top_c = has_top ? input[idx - row_stride] : 0ULL;
    const u64 bot_c = has_bottom ? input[idx + row_stride] : 0ULL;

    // Within a strip, threadIdx.x is the lane index.
    // For STRIP_WORDS == 16, the shuffle width partitions each warp into two independent
    // 16-lane groups, exactly matching the two row strips carried by that warp.
    const int lane = static_cast<int>(threadIdx.x);

    const u64 top_up = __shfl_up_sync(kFullMask, top_c, 1, STRIP_WORDS);
    const u64 top_dn = __shfl_down_sync(kFullMask, top_c, 1, STRIP_WORDS);
    const u64 mid_up = __shfl_up_sync(kFullMask, mid_c, 1, STRIP_WORDS);
    const u64 mid_dn = __shfl_down_sync(kFullMask, mid_c, 1, STRIP_WORDS);
    const u64 bot_up = __shfl_up_sync(kFullMask, bot_c, 1, STRIP_WORDS);
    const u64 bot_dn = __shfl_down_sync(kFullMask, bot_c, 1, STRIP_WORDS);

    // Top row: sum of west + center + east, encoded as t0 + 2*t1.
    u64 left_word = (lane != 0)
                        ? top_up
                        : ((has_top && has_left) ? input[idx - row_stride - 1] : 0ULL);
    u64 right_word = (lane != LAST_LANE)
                         ? top_dn
                         : ((has_top && has_right) ? input[idx - row_stride + 1] : 0ULL);

    u64 w = shift_west(top_c, left_word);
    u64 e = shift_east(top_c, right_word);
    const u64 t0 = w ^ top_c ^ e;
    const u64 t1 = majority3(w, top_c, e);

    // Middle row: only west + east contribute to the neighbor count.
    // Encode as m0 + 2*m1.
    left_word = (lane != 0) ? mid_up : (has_left ? input[idx - 1] : 0ULL);
    right_word = (lane != LAST_LANE) ? mid_dn : (has_right ? input[idx + 1] : 0ULL);

    w = shift_west(mid_c, left_word);
    e = shift_east(mid_c, right_word);
    const u64 m0 = w ^ e;
    const u64 m1 = w & e;

    // Bottom row: sum of west + center + east, encoded as b0 + 2*b1.
    left_word = (lane != 0)
                    ? bot_up
                    : ((has_bottom && has_left) ? input[idx + row_stride - 1] : 0ULL);
    right_word = (lane != LAST_LANE)
                     ? bot_dn
                     : ((has_bottom && has_right) ? input[idx + row_stride + 1] : 0ULL);

    w = shift_west(bot_c, left_word);
    e = shift_east(bot_c, right_word);
    const u64 b0 = w ^ bot_c ^ e;
    const u64 b1 = majority3(w, bot_c, e);

    // Neighbor-count decomposition:
    //
    //   top_count    = t0 + 2*t1
    //   mid_count    = m0 + 2*m1   (self excluded)
    //   bottom_count = b0 + 2*b1
    //
    //   total_neighbors = (t0 + m0 + b0) + 2*(t1 + m1 + b1)
    //
    // Let:
    //   s0 = low bit of (t0 + m0 + b0)
    //   c0 = carry   of (t0 + m0 + b0)
    //
    // Then:
    //   total_neighbors = s0 + 2*(c0 + t1 + m1 + b1)
    //
    // Counts 2 and 3 are exactly the cases where the upper term equals 1.
    // s0 distinguishes 2 from 3.
    const u64 s0 = t0 ^ m0 ^ b0;
    const u64 c0 = majority3(t0, m0, b0);
    const u64 upper_eq1 = exactly_one_of_four(c0, t1, m1, b1);

    // Conway rule:
    //   birth on 3
    //   survive on 2 or 3
    //
    // With the encoding above this becomes:
    //   upper_eq1 & (s0 | self)
    const u64 next = upper_eq1 & (s0 | mid_c);
    output[idx] = next;
}

}  // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // Fast path only:
    //   - input/output are device pointers from cudaMalloc
    //   - grid_dimensions is a power of two, > 512
    //   - input and output are distinct ping-pong buffers
    // The caller owns synchronization and error handling, so none is done here.

    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Because the grid width and height are powers of two, these tiles cover the domain exactly.
    // There are no bounds checks in the hot kernel.
    if (words_per_row == 16) {
        // Minimum legal case: 1024 cells per row -> 16 packed words.
        constexpr int STRIP_WORDS = 16;
        constexpr int BLOCK_ROWS = kThreadsPerBlock / STRIP_WORDS;  // 16
        constexpr unsigned int kBlocksPerRowShift = 0;
        constexpr unsigned int kBlocksPerRowMask = 0;

        const unsigned int total_blocks =
            static_cast<unsigned int>(grid_dimensions / BLOCK_ROWS);

        game_of_life_kernel<STRIP_WORDS>
            <<<total_blocks, dim3(STRIP_WORDS, BLOCK_ROWS, 1)>>>(
                input,
                output,
                words_per_row,
                grid_dimensions,
                kBlocksPerRowShift,
                kBlocksPerRowMask);
    } else {
        // All larger legal cases have at least 32 packed words per row.
        constexpr int STRIP_WORDS = 32;
        constexpr int BLOCK_ROWS = kThreadsPerBlock / STRIP_WORDS;  // 8

        const unsigned int blocks_per_row =
            static_cast<unsigned int>(words_per_row / STRIP_WORDS);
        const unsigned int blocks_per_row_shift = log2_pow2(blocks_per_row);
        const unsigned int blocks_per_row_mask = blocks_per_row - 1u;
        const unsigned int rows_of_blocks =
            static_cast<unsigned int>(grid_dimensions / BLOCK_ROWS);
        const unsigned int total_blocks = blocks_per_row * rows_of_blocks;

        game_of_life_kernel<STRIP_WORDS>
            <<<total_blocks, dim3(STRIP_WORDS, BLOCK_ROWS, 1)>>>(
                input,
                output,
                words_per_row,
                grid_dimensions,
                blocks_per_row_shift,
                blocks_per_row_mask);
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
