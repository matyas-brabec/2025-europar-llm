#include <cstdint>
#include <cuda_runtime.h>

// Bit-packed Conway's Game of Life.
//
// Design choices driven purely by performance on modern data-center GPUs:
// - One CUDA thread owns one 64-bit output word (64 cells), so no atomics are needed.
// - Shared/texture memory are intentionally avoided.
// - A per-bit __popc baseline is reasonable, but a bit-sliced 64-lane counter is faster here:
//   all 64 cells in a word are updated in parallel with bitwise logic.
// - Warp shuffles provide left/right neighboring words inside a warp (or half warp for the
//   smallest legal grid), so only subgroup-edge lanes need explicit halo handling.
// - Cells outside the grid are dead: halos are zero-filled, i.e. no toroidal wrap-around.
//
// Bit numbering follows the problem statement's boundary description:
// bit 0 is the leftmost cell in a 64-cell word, and bit 63 is the rightmost.

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

constexpr int kBlockSize = 256;          // 8 warps/block; good balance for A100/H100.
constexpr u32 kFullWarpMask = 0xFFFFFFFFu;

static_assert((kBlockSize & (kBlockSize - 1)) == 0, "Block size must be a power of two.");
static_assert((kBlockSize % 32) == 0, "Block size must be warp-aligned.");

// Add one 64-bit bitboard into a bit-sliced 3-bit counter.
// The counter is maintained modulo 8. That is sufficient because Game of Life only needs
// neighbor counts 2 and 3, and count 8 wraps to 0 modulo 8, which does not alias 2 or 3.
__device__ __forceinline__ void add_bitboard(u64 x, u64& ones, u64& twos, u64& fours) {
    u64 carry = ones & x;
    ones ^= x;
    x = carry;

    carry = twos & x;
    twos ^= x;
    x = carry;

    fours ^= x;
}

// 64-bit warp shuffle helpers built from native 32-bit shuffles.
template <int SUBGROUP_WIDTH>
__device__ __forceinline__ u64 shfl_up1_u64(u64 v) {
    u32 lo = static_cast<u32>(v);
    u32 hi = static_cast<u32>(v >> 32);
    lo = __shfl_up_sync(kFullWarpMask, lo, 1, SUBGROUP_WIDTH);
    hi = __shfl_up_sync(kFullWarpMask, hi, 1, SUBGROUP_WIDTH);
    return (static_cast<u64>(hi) << 32) | static_cast<u64>(lo);
}

template <int SUBGROUP_WIDTH>
__device__ __forceinline__ u64 shfl_down1_u64(u64 v) {
    u32 lo = static_cast<u32>(v);
    u32 hi = static_cast<u32>(v >> 32);
    lo = __shfl_down_sync(kFullWarpMask, lo, 1, SUBGROUP_WIDTH);
    hi = __shfl_down_sync(kFullWarpMask, hi, 1, SUBGROUP_WIDTH);
    return (static_cast<u64>(hi) << 32) | static_cast<u64>(lo);
}

// Fix subgroup-edge left/right words.
// SUBGROUP_WIDTH == 16:
//   The subgroup is exactly one full row (this only happens for 1024x1024 grids), so subgroup
//   edges are physical grid edges and the missing horizontal halos are always zero.
// SUBGROUP_WIDTH == 32:
//   A subgroup is one 32-word row segment. Interior subgroup edges fetch the missing halo words
//   from global memory; true row edges inject zeros.
template <int SUBGROUP_WIDTH>
__device__ __forceinline__ void apply_horizontal_halo(
    u64& left, u64& right,
    const u64* input,
    u64 base_index,
    bool load_left,
    bool load_right,
    u32 lane_in_group);

template <>
__device__ __forceinline__ void apply_horizontal_halo<16>(
    u64& left, u64& right,
    const u64* input,
    u64 base_index,
    bool load_left,
    bool load_right,
    u32 lane_in_group)
{
    (void)input;
    (void)base_index;
    (void)load_left;
    (void)load_right;

    if (lane_in_group == 0u) {
        left = 0ull;
    }
    if (lane_in_group == 15u) {
        right = 0ull;
    }
}

template <>
__device__ __forceinline__ void apply_horizontal_halo<32>(
    u64& left, u64& right,
    const u64* input,
    u64 base_index,
    bool load_left,
    bool load_right,
    u32 lane_in_group)
{
    if (lane_in_group == 0u) {
        left = load_left ? input[base_index - 1] : 0ull;
    }
    if (lane_in_group == 31u) {
        right = load_right ? input[base_index + 1] : 0ull;
    }
}

// Under the stated constraints:
// - grid_dimensions is a power of two and > 512
// - words_per_row = grid_dimensions / 64 is therefore either 16 or a multiple of 32
// - total_words = grid_dimensions * words_per_row is a power of two and divisible by kBlockSize
//
// The host launch therefore covers the grid exactly, and the kernel can omit a bounds check.
template <int SUBGROUP_WIDTH>
__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    u32 words_per_row,
    u32 words_per_row_log2,
    u32 last_row)
{
    static_assert(SUBGROUP_WIDTH == 16 || SUBGROUP_WIDTH == 32, "Unsupported subgroup width.");

    const u64 word_index =
        static_cast<u64>(blockIdx.x) * static_cast<u64>(kBlockSize) +
        static_cast<u64>(threadIdx.x);

    const u32 last_word     = words_per_row - 1u;
    const u32 wx            = static_cast<u32>(word_index) & last_word;
    const u32 row           = static_cast<u32>(word_index >> words_per_row_log2);
    const u32 lane_in_group = static_cast<u32>(threadIdx.x) & (SUBGROUP_WIDTH - 1);

    const bool has_up    = (row != 0u);
    const bool has_down  = (row != last_row);
    const bool has_left  = (wx  != 0u);
    const bool has_right = (wx  != last_word);

    const u64 row_stride = static_cast<u64>(words_per_row);
    const u64 up_index   = word_index - row_stride;
    const u64 down_index = word_index + row_stride;

    const u64 cC = input[word_index];

    u64 ones  = 0ull;
    u64 twos  = 0ull;
    u64 fours = 0ull;

    // North row: NW, N, NE
    {
        const u64 nC = has_up ? input[up_index] : 0ull;

        u64 nL = shfl_up1_u64<SUBGROUP_WIDTH>(nC);
        u64 nR = shfl_down1_u64<SUBGROUP_WIDTH>(nC);
        apply_horizontal_halo<SUBGROUP_WIDTH>(
            nL, nR, input, up_index,
            has_up && has_left,
            has_up && has_right,
            lane_in_group);

        add_bitboard((nC << 1) | (nL >> 63), ones, twos, fours);  // NW
        add_bitboard(nC,                         ones, twos, fours);  // N
        add_bitboard((nC >> 1) | (nR << 63), ones, twos, fours);  // NE
    }

    // Current row: W, E
    {
        u64 cL = shfl_up1_u64<SUBGROUP_WIDTH>(cC);
        u64 cR = shfl_down1_u64<SUBGROUP_WIDTH>(cC);
        apply_horizontal_halo<SUBGROUP_WIDTH>(
            cL, cR, input, word_index,
            has_left,
            has_right,
            lane_in_group);

        add_bitboard((cC << 1) | (cL >> 63), ones, twos, fours);  // W
        add_bitboard((cC >> 1) | (cR << 63), ones, twos, fours);  // E
    }

    // South row: SW, S, SE
    {
        const u64 sC = has_down ? input[down_index] : 0ull;

        u64 sL = shfl_up1_u64<SUBGROUP_WIDTH>(sC);
        u64 sR = shfl_down1_u64<SUBGROUP_WIDTH>(sC);
        apply_horizontal_halo<SUBGROUP_WIDTH>(
            sL, sR, input, down_index,
            has_down && has_left,
            has_down && has_right,
            lane_in_group);

        add_bitboard((sC << 1) | (sL >> 63), ones, twos, fours);  // SW
        add_bitboard(sC,                         ones, twos, fours);  // S
        add_bitboard((sC >> 1) | (sR << 63), ones, twos, fours);  // SE
    }

    // Classic Life rule from the bit-sliced counter:
    // - count == 2  -> twos=1, ones=0, fours=0  => cell survives iff it is already alive
    // - count == 3  -> twos=1, ones=1, fours=0  => cell is alive next generation
    //
    // This collapses to:
    //   next = twos & ~fours & (ones | alive)
    output[word_index] = twos & ~fours & (ones | cC);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Natural double-buffered usage is assumed: input and output are distinct device allocations.
    // No synchronization is performed here; the caller owns ordering and synchronization.

    const std::uint32_t n = static_cast<std::uint32_t>(grid_dimensions);
    const std::uint32_t words_per_row = n >> 6;  // 64 cells per 64-bit word.

    // words_per_row is a power of two by problem construction.
    std::uint32_t words_per_row_log2 = 0u;
    for (std::uint32_t t = words_per_row; t > 1u; t >>= 1u) {
        ++words_per_row_log2;
    }

    const std::uint64_t total_words =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(words_per_row);

    // total_words is a power of two and divisible by kBlockSize under the given constraints.
    const std::uint32_t blocks =
        static_cast<std::uint32_t>(total_words / static_cast<std::uint64_t>(kBlockSize));

    // words_per_row is either:
    // - 16  (the smallest legal grid: 1024x1024), where each half warp is one full row, or
    // - a multiple of 32, where each warp stays entirely inside a row segment.
    if (words_per_row == 16u) {
        game_of_life_kernel<16><<<blocks, kBlockSize>>>(
            input, output, words_per_row, words_per_row_log2, n - 1u);
    } else {
        game_of_life_kernel<32><<<blocks, kBlockSize>>>(
            input, output, words_per_row, words_per_row_log2, n - 1u);
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
