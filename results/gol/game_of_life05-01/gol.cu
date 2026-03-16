#include <cstdint>
#include <cuda_runtime.h>

namespace {

// High-throughput Conway's Game of Life step for a square, bit-packed board.
//
// Performance-focused design:
// - One CUDA thread updates one 64-bit word = 64 cells, so no atomics are needed.
// - All 64 cells in that word are updated simultaneously with SWAR/bit-sliced integer logic.
// - Shared memory and textures are intentionally avoided; the board is already compact, and
//   warp shuffles are sufficient to reuse adjacent words without extra synchronization.
//
// The special handling for bit 0 and bit 63 requested by the problem is implemented by
// shift_west() / shift_east(): bit 0 pulls its west neighbor from the word on the left,
// bit 63 pulls its east neighbor from the word on the right, and the same is done for the
// row above, the current row, and the row below.

using u64 = std::uint64_t;
using u32 = std::uint32_t;

constexpr int kWarpSize = 32;
constexpr u32 kWarpLaneMask = kWarpSize - 1u;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

// 256 threads = 8 warps per block. This is a strong fixed point for this kernel on
// recent NVIDIA data-center GPUs while keeping the launch geometry simple.
constexpr int kBlockShift = 8;
constexpr int kBlockSize = 1 << kBlockShift;

static_assert(kBlockSize == (1 << kBlockShift), "kBlockShift must match kBlockSize.");
static_assert((kBlockSize % kWarpSize) == 0, "kBlockSize must be a multiple of the warp size.");

// Read-only cached load. The input board is never modified by this kernel.
__device__ __forceinline__ u64 load_ro(const u64* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Align west/east neighbors to the current word's bit positions.
// For example, shift_west(center, left_word) returns a mask whose bit i is the cell immediately
// west of bit i in center. Bit 0 therefore comes from left_word bit 63; shift_east does the
// symmetric thing for bit 63 and the word on the right.
__device__ __forceinline__ u64 shift_west(u64 center, u64 left_word) {
    return (center << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 shift_east(u64 center, u64 right_word) {
    return (center >> 1) | (right_word << 63);
}

// Per-bit population count of two 1-bit masks. For every bit position i:
//   count_i = lo_i + 2 * hi_i
__device__ __forceinline__ void bitcount2(u64 a, u64 b, u64& lo, u64& hi) {
    lo = a ^ b;
    hi = a & b;
}

// Per-bit population count of three 1-bit masks. For every bit position i:
//   count_i = lo_i + 2 * hi_i
__device__ __forceinline__ void bitcount3(u64 a, u64 b, u64 c, u64& lo, u64& hi) {
    const u64 x = a ^ b;
    lo = x ^ c;
    hi = (a & b) | (x & c);
}

// Return a mask whose bit i is 1 iff exactly one of a_i, b_i, c_i, d_i is 1.
// This is cheaper than materializing the full 3-bit sum when all we need is the "sum == 1" test.
__device__ __forceinline__ u64 exactly_one_of_four(u64 a, u64 b, u64 c, u64 d) {
    const u64 u = a ^ b;
    const u64 v = c ^ d;
    const u64 bit0 = u ^ v;
    const u64 carries = (a & b) | (c & d) | (u & v);
    return bit0 & ~carries;
}

// Choose the adjacent 64-bit word for one row.
// The shuffle itself is executed unconditionally in the kernel so the *_sync mask is always valid.
// This helper only decides between:
//   - zero at the board edge,
//   - the shuffled word when the neighbor stays inside the same warp chunk of the row,
//   - one fallback global load when the row continues into the previous/next warp.
__device__ __forceinline__ u64 neighbor_left(u64 shuffled_left,
                                             const u64* row_center_ptr,
                                             bool has_neighbor_word,
                                             bool same_warp) {
    return !has_neighbor_word ? 0ULL : (same_warp ? shuffled_left : load_ro(row_center_ptr - 1));
}

__device__ __forceinline__ u64 neighbor_right(u64 shuffled_right,
                                              const u64* row_center_ptr,
                                              bool has_neighbor_word,
                                              bool same_warp) {
    return !has_neighbor_word ? 0ULL : (same_warp ? shuffled_right : load_ro(row_center_ptr + 1));
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         u32 words_per_row,
                         u32 word_mask,
                         int log2_words_per_row,
                         u32 last_row) {
    // run_game_of_life launches exactly one thread per 64-bit word, and total_words is chosen
    // so every block and warp is full. Therefore there is no tail path and the full warp mask is valid.
    const u32 idx = (static_cast<u32>(blockIdx.x) << kBlockShift) + static_cast<u32>(threadIdx.x);

    // words_per_row is a power of two, so word-in-row and row index are obtained with a mask and shift.
    const u32 word = idx & word_mask;
    const u32 row = idx >> log2_words_per_row;

    const bool has_up = row != 0u;
    const bool has_down = row != last_row;
    const bool has_left = word != 0u;
    const bool has_right = word != word_mask;

    // Because words_per_row is itself a power of two, a row either evenly divides a warp
    // or spans an integer number of whole warps. Therefore word&31 is sufficient to know
    // whether the adjacent word is already owned by a neighboring lane in this warp.
    const u32 warp_word = word & kWarpLaneMask;
    const bool same_warp_left = warp_word != 0u;
    const bool same_warp_right = warp_word != kWarpLaneMask;

    const u64* const center_ptr = input + idx;
    const u64* const up_ptr = has_up ? (center_ptr - words_per_row) : center_ptr;
    const u64* const down_ptr = has_down ? (center_ptr + words_per_row) : center_ptr;

    const u64 cur_word = load_ro(center_ptr);
    const u64 up_word = has_up ? load_ro(up_ptr) : 0ULL;
    const u64 down_word = has_down ? load_ro(down_ptr) : 0ULL;

    u64 shuffled_left;
    u64 shuffled_right;
    u64 west;
    u64 east;

    // Top row contribution: up-left, up, up-right.
    shuffled_left = __shfl_up_sync(kFullWarpMask, up_word, 1);
    shuffled_right = __shfl_down_sync(kFullWarpMask, up_word, 1);
    west = shift_west(up_word, neighbor_left(shuffled_left, up_ptr, has_up && has_left, same_warp_left));
    east = shift_east(up_word, neighbor_right(shuffled_right, up_ptr, has_up && has_right, same_warp_right));
    u64 top_lo, top_hi;
    bitcount3(west, up_word, east, top_lo, top_hi);

    // Middle row contribution: only west and east; the current word itself is not a neighbor.
    shuffled_left = __shfl_up_sync(kFullWarpMask, cur_word, 1);
    shuffled_right = __shfl_down_sync(kFullWarpMask, cur_word, 1);
    west = shift_west(cur_word, neighbor_left(shuffled_left, center_ptr, has_left, same_warp_left));
    east = shift_east(cur_word, neighbor_right(shuffled_right, center_ptr, has_right, same_warp_right));
    u64 mid_lo, mid_hi;
    bitcount2(west, east, mid_lo, mid_hi);

    // Bottom row contribution: down-left, down, down-right.
    shuffled_left = __shfl_up_sync(kFullWarpMask, down_word, 1);
    shuffled_right = __shfl_down_sync(kFullWarpMask, down_word, 1);
    west = shift_west(down_word, neighbor_left(shuffled_left, down_ptr, has_down && has_left, same_warp_left));
    east = shift_east(down_word, neighbor_right(shuffled_right, down_ptr, has_down && has_right, same_warp_right));
    u64 bot_lo, bot_hi;
    bitcount3(west, down_word, east, bot_lo, bot_hi);

    // Add the low bits of the three row counts:
    //   (top_lo + mid_lo + bot_lo) = sum_lsb + 2 * carry_to_high
    u64 sum_lsb, carry_to_high;
    bitcount3(top_lo, mid_lo, bot_lo, sum_lsb, carry_to_high);

    // The total neighbor count is:
    //   total = sum_lsb + 2 * (top_hi + mid_hi + bot_hi + carry_to_high)
    //
    // A cell is alive in the next generation only for totals 2 and 3.
    // Those are exactly the cases where the parenthesized group equals 1:
    //   total == 2 -> sum_lsb = 0
    //   total == 3 -> sum_lsb = 1
    //
    // Therefore:
    //   next = (group == 1) & (sum_lsb | current_cell)
    // which implements:
    //   - birth on 3 neighbors,
    //   - survival on 2 neighbors.
    const u64 high_is_one = exactly_one_of_four(top_hi, mid_hi, bot_hi, carry_to_high);
    output[idx] = high_is_one & (sum_lsb | cur_word);
}

// words_per_row is guaranteed to be a power of two.
inline int ilog2_pow2_u32(u32 v) {
    int log2 = 0;
    while ((v >>= 1u) != 0u) {
        ++log2;
    }
    return log2;
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller provides device pointers and owns all synchronization. This function only
    // computes launch parameters and submits one kernel for a single Conway step.
    //
    // Given the problem constraints:
    //   n = 2^k, k >= 10
    //   total_words = n * (n / 64) = n^2 / 64 = 2^(2k - 6)
    // so total_words is a power of two and therefore divisible by kBlockSize = 256.
    // That lets the kernel run with an exact one-thread-per-word launch and no tail handling.
    const u32 n = static_cast<u32>(grid_dimensions);
    const u32 words_per_row = n >> 6;
    const int log2_words_per_row = ilog2_pow2_u32(words_per_row);
    const std::uint64_t total_words = (static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n)) >> 6;
    const u32 num_blocks = static_cast<u32>(total_words >> kBlockShift);

    // Double-buffering is assumed: input and output must refer to different device buffers.
    game_of_life_kernel<<<num_blocks, kBlockSize>>>(
        input,
        output,
        words_per_row,
        words_per_row - 1u,
        log2_words_per_row,
        n - 1u);
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
