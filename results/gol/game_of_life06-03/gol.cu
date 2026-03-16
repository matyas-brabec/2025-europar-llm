#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// One-step Conway's Game of Life on a square, bit-packed grid.
//
// Storage layout:
//   - Each std::uint64_t stores 64 consecutive cells from one row.
//   - Bit 0 is the left edge of the packed 64-cell run.
//   - Bit 63 is the right edge of the packed 64-cell run.
//
// Performance strategy:
//   - One CUDA thread updates one 64-bit word, so 64 cells are processed at once.
//   - blockDim.x is fixed to one warp (32 threads). Each warp therefore covers 32 adjacent
//     packed words in x and can exchange x-neighbor words through warp shuffles instead of
//     reloading them from memory.
//   - Neighbor counts are computed with a bit-sliced full-adder tree. This adds the eight
//     per-cell neighbor bits for all 64 cells in parallel using only bitwise operations.
//   - No shared memory, no textures, and no atomics are needed.

namespace {

using u32 = unsigned int;

// CUDA shuffle intrinsics have native 64-bit overloads for unsigned long long on all supported
// host ABIs, so we use that as the working word type even though the external storage type is
// std::uint64_t.
using u64 = unsigned long long;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "Expected a 64-bit word type.");

constexpr int kWordBits       = 64;
constexpr int kWordBitShift   = 6;                 // log2(64)
constexpr int kWordMsbShift   = kWordBits - 1;     // 63

constexpr int kWarpWords      = 32;                // one warp spans 32 packed words in x
constexpr int kWarpWordShift  = 5;                 // log2(32)

constexpr int kBlockRows      = 4;                 // 4 warps per block is a good occupancy balance
constexpr int kBlockRowsShift = 2;                 // log2(4)
constexpr int kBlockThreads   = kWarpWords * kBlockRows;

static_assert(kWordBits  == (1 << kWordBitShift),   "Word geometry must stay power-of-two.");
static_assert(kWarpWords == (1 << kWarpWordShift),  "Warp geometry must stay power-of-two.");
static_assert(kBlockRows == (1 << kBlockRowsShift), "Block geometry must stay power-of-two.");

// Portable host-side log2 for guaranteed power-of-two inputs.
inline int log2_pow2(u32 x) {
    int shift = 0;
    while (x > 1u) {
        x >>= 1u;
        ++shift;
    }
    return shift;
}

__device__ __forceinline__ u64 majority(u64 a, u64 b, u64 c) {
    // Per-bit majority; on modern NVIDIA GPUs this maps well to LOP3.
    return (a & b) | (c & (a | b));
}

__device__ __forceinline__ void full_adder(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    // 3:2 compressor:
    //   sum   = low bit of a+b+c
    //   carry = high bit of a+b+c (weight 2)
    sum   = a ^ b ^ c;
    carry = majority(a, b, c);
}

__device__ __forceinline__ void half_adder(u64 a, u64 b, u64& sum, u64& carry) {
    // 2:2 compressor:
    //   sum   = low bit of a+b
    //   carry = high bit of a+b (weight 2)
    sum   = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ u64 evolve_packed_word(
    u64 nw, u64 n,  u64 ne,
    u64 w,  u64 self, u64 e,
    u64 sw, u64 s,  u64 se)
{
    // The 3x3 neighborhood of 64-bit words around the current word:
    //
    //   nw |  n | ne
    //   w  |self| e
    //   sw |  s | se
    //
    // Each bit position corresponds to one cell. The update below computes the next state for all
    // 64 cells in "self" simultaneously.
    //
    // Full-adder idea:
    //   For any three 64-bit inputs, xor gives the 1's bit of the per-bit sum and majority gives
    //   the 2's bit. We first compress the three *raw word columns*:
    //
    //     left  column: (nw, w,  sw)
    //     mid   column: (n,  self, s)
    //     right column: (ne, e,  se)
    //
    // The actual left/right neighbor columns seen by the target cells are then formed by shifting
    // the compressed middle column and injecting the edge bit from the compressed left/right raw
    // columns:
    //
    //   left  bits 1..63 <- mid bits 0..62
    //   left  bit 0      <- raw-left bit 63
    //
    //   right bits 0..62 <- mid bits 1..63
    //   right bit 63     <- raw-right bit 0
    //
    // This is exactly the required special handling for bit 0 / bit 63. The center neighbor
    // column is only (n, s); "self" must not count itself.

    u64 raw_left_sum,  raw_left_carry;
    u64 raw_mid_sum,   raw_mid_carry;
    u64 raw_right_sum, raw_right_carry;

    full_adder(nw, w,  sw, raw_left_sum,  raw_left_carry);
    full_adder(n,  self, s, raw_mid_sum,   raw_mid_carry);
    full_adder(ne, e,  se, raw_right_sum, raw_right_carry);

    const u64 left_sum   = (raw_mid_sum   << 1) | (raw_left_sum   >> kWordMsbShift);
    const u64 left_carry = (raw_mid_carry << 1) | (raw_left_carry >> kWordMsbShift);

    const u64 right_sum   = (raw_mid_sum   >> 1) | (raw_right_sum   << kWordMsbShift);
    const u64 right_carry = (raw_mid_carry >> 1) | (raw_right_carry << kWordMsbShift);

    u64 center_sum, center_carry;
    half_adder(n, s, center_sum, center_carry);

    // Combine the three neighbor columns into a 4-bit bit-sliced count:
    //   ones   = bit 0 of neighbor count
    //   twos   = bit 1 of neighbor count
    //   fours  = bit 2 of neighbor count
    //   eights = bit 3 of neighbor count
    u64 ones, twos_from_low_planes;
    u64 twos_from_high_planes, fours_from_high_planes;
    u64 twos, fours_from_cross;
    u64 fours, eights;

    full_adder(left_sum,   center_sum,   right_sum,   ones,                  twos_from_low_planes);
    full_adder(left_carry, center_carry, right_carry, twos_from_high_planes, fours_from_high_planes);
    half_adder(twos_from_low_planes,  twos_from_high_planes, twos,  fours_from_cross);
    half_adder(fours_from_high_planes, fours_from_cross,     fours, eights);

    // Conway rule:
    //   next = (count == 3) | (self & (count == 2))
    //
    // In the bit-sliced count:
    //   count is 2 or 3  <=> twos=1 and fours=eights=0
    //   among those, count is 3 OR the cell is already alive <=> (ones | self)
    const u64 count_is_2_or_3 = twos & ~(fours | eights);
    return count_is_2_or_3 & (ones | self);
}

__global__ __launch_bounds__(kBlockThreads)
void game_of_life_step_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    u32 grid_dimensions,
    int words_per_row_shift,
    int tiles_x_shift)
{
    // Block layout:
    //   - blockDim.x == 32, so each warp is one horizontal tile of 32 packed words.
    //   - blockDim.y == kBlockRows, so a block processes kBlockRows independent rows of tiles.
    //
    // Grid layout:
    //   We flatten the logical 2D launch grid into grid.x so we are not limited by grid.y for
    //   very large boards. Because the board size is a power of two, the inverse mapping from
    //   blockIdx.x to (row_tile, x_tile) uses only shifts and masks.

    const u32 words_per_row = 1u << words_per_row_shift;
    const u32 x_tile_mask   = (1u << tiles_x_shift) - 1u;

    const u32 block_linear = static_cast<u32>(blockIdx.x);
    const u32 x_tile       = block_linear & x_tile_mask;
    const u32 row_tile     = block_linear >> tiles_x_shift;

    const u32 word_x = (x_tile << kWarpWordShift) | static_cast<u32>(threadIdx.x);
    const u32 row    = (row_tile << kBlockRowsShift) | static_cast<u32>(threadIdx.y);

    const bool active = (row < grid_dimensions) && (word_x < words_per_row);

    // All lanes in the warp execute the ballot before any early return, so the resulting mask is
    // valid for subsequent shuffle operations even when only part of the warp is active (the only
    // practical partial case here is 16 words/row, i.e. grid_dimensions == 1024).
    const unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, active);
    if (!active) {
        return;
    }

    // Because words_per_row is a power of two and word_x < words_per_row, index computation can
    // use a shift/or instead of a multiply/add.
    const u32 index = (row << words_per_row_shift) | word_x;

    const bool has_top    = (row != 0u);
    const bool has_bottom = (row + 1u < grid_dimensions);
    const bool has_left   = (word_x != 0u);
    const bool has_right  = (word_x + 1u < words_per_row);

    // Every active thread loads only its own three center words.
    const u64 self  = input[index];
    const u64 north = has_top    ? input[index - words_per_row] : 0ull;
    const u64 south = has_bottom ? input[index + words_per_row] : 0ull;

    // Horizontal neighbors are exchanged within the warp. Only the two warp-edge lanes fall back
    // to one extra global load for x-1 / x+1 (or to zero at the grid boundary).
    const u64 sh_w  = __shfl_up_sync(active_mask,   self,  1);
    const u64 sh_nw = __shfl_up_sync(active_mask,   north, 1);
    const u64 sh_sw = __shfl_up_sync(active_mask,   south, 1);

    const u64 sh_e  = __shfl_down_sync(active_mask, self,  1);
    const u64 sh_ne = __shfl_down_sync(active_mask, north, 1);
    const u64 sh_se = __shfl_down_sync(active_mask, south, 1);

    const bool use_shuffle_left  = (static_cast<u32>(threadIdx.x) != 0u);
    const bool use_shuffle_right = (static_cast<u32>(threadIdx.x) + 1u < kWarpWords) && has_right;

    const u64 west =
        use_shuffle_left ? sh_w : (has_left ? input[index - 1u] : 0ull);
    const u64 north_west =
        use_shuffle_left ? sh_nw : ((has_top && has_left) ? input[index - words_per_row - 1u] : 0ull);
    const u64 south_west =
        use_shuffle_left ? sh_sw : ((has_bottom && has_left) ? input[index + words_per_row - 1u] : 0ull);

    const u64 east =
        use_shuffle_right ? sh_e : (has_right ? input[index + 1u] : 0ull);
    const u64 north_east =
        use_shuffle_right ? sh_ne : ((has_top && has_right) ? input[index - words_per_row + 1u] : 0ull);
    const u64 south_east =
        use_shuffle_right ? sh_se : ((has_bottom && has_right) ? input[index + words_per_row + 1u] : 0ull);

    output[index] = static_cast<std::uint64_t>(evolve_packed_word(
        north_west, north, north_east,
        west,       self,  east,
        south_west, south, south_east));
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The problem guarantees:
    //   - square dimensions
    //   - power-of-two size
    //   - input/output already allocated on the device
    //   - caller handles synchronization
    //
    // So the host side does only the minimal launch preparation needed for the optimized kernel.

    const u32 n = static_cast<u32>(grid_dimensions);

    // Because n is a power of two:
    //   words_per_row       = n / 64  = 2^(log2(n) - 6)
    //   x_tiles_per_row     = max(1, words_per_row / 32)
    // Both are therefore powers of two as well, so the kernel can recover row/column tile IDs
    // with shifts and masks only.
    const int grid_shift          = log2_pow2(n);
    const int words_per_row_shift = grid_shift - kWordBitShift;
    const int tiles_x_shift =
        (words_per_row_shift > kWarpWordShift) ? (words_per_row_shift - kWarpWordShift) : 0;

    // kBlockRows is also a power of two and n is guaranteed to be divisible by it.
    const std::size_t row_tiles  = static_cast<std::size_t>(n) >> kBlockRowsShift;
    const std::size_t block_count = row_tiles << tiles_x_shift;

    const dim3 block(kWarpWords, kBlockRows, 1);
    const dim3 grid(static_cast<u32>(block_count), 1, 1);

    // Intentionally no host-device synchronization and no error checking in this hot path.
    game_of_life_step_kernel<<<grid, block>>>(input, output, n, words_per_row_shift, tiles_x_shift);
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
