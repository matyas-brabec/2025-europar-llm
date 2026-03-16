#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// CUDA implementation of one Conway's Game of Life step on a bit-packed grid.
//
// Design choices for performance on modern data-center GPUs:
// - One thread owns one packed 64-bit output word, so stores are conflict-free and
//   no atomics are needed.
// - The grid stays in its packed representation end-to-end; no unpacking is done.
// - Shared / texture memory are intentionally avoided. Instead, each warp (or
//   half-warp for the 1024x1024 case) covers a contiguous row segment and exchanges
//   left/right center/north/south words via warp shuffles.
// - Neighbor counting uses carry-save / half-adder logic on 64-bit bit-planes, so
//   each thread updates 64 cells in parallel with pure bitwise operations.
// - Standard double buffering is assumed: input and output point to different device
//   allocations.

using u64 = std::uint64_t;

constexpr int kWordBits = 64;
constexpr int kWordBitShift = 6;       // log2(64)
constexpr int kBlockY8Log2 = 3;        // block.y = 8
constexpr int kBlockY16Log2 = 4;       // block.y = 16
constexpr unsigned kBlockY8 = 1u << kBlockY8Log2;
constexpr unsigned kBlockY16 = 1u << kBlockY16Log2;
constexpr unsigned kGridYLimit = 65535u;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert(sizeof(u64) * 8 == kWordBits, "Expected 64-bit words");

// Bit i in a packed word represents the cell at x = word_base + i.
// Bit 0 and bit 63 therefore need cross-word handling:
//   west  = (self << 1) | (left_word  >> 63)
//   east  = (self >> 1) | (right_word << 63)
// so the missing neighbor bit is pulled from the adjacent packed word.
static __device__ __forceinline__ u64 align_left(u64 word, u64 left_word) {
    return (word << 1) | (left_word >> (kWordBits - 1));
}

static __device__ __forceinline__ u64 align_right(u64 word, u64 right_word) {
    return (word >> 1) | (right_word << (kWordBits - 1));
}

// Carry-save adder for 64 independent bit positions at once:
//   a + b + c = sum + 2 * carry
// with:
//   sum   = a ^ b ^ c
//   carry = majority(a, b, c)
static __device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum = ab_xor ^ c;
    carry = (a & b) | (c & ab_xor);
}

static __device__ __forceinline__ void ha(u64 a, u64 b, u64& sum, u64& carry) {
    sum = a ^ b;
    carry = a & b;
}

// Consume the 3x3 packed-word neighborhood around the current word and compute the
// next 64-cell packed word.
static __device__ __forceinline__ u64 step_word(
    u64 north_left, u64 north, u64 north_right,
    u64 left,       u64 center, u64 right,
    u64 south_left, u64 south,  u64 south_right)
{
    // Compress the top and bottom neighbor triplets independently:
    //   row_sum + 2*row_carry = diag_left + vertical + diag_right
    u64 top_sum, top_carry;
    csa(align_left(north, north_left), north, align_right(north, north_right),
        top_sum, top_carry);

    u64 bottom_sum, bottom_carry;
    csa(align_left(south, south_left), south, align_right(south, south_right),
        bottom_sum, bottom_carry);

    // Left/right neighbors from the current row.
    u64 side_sum, side_carry;
    ha(align_left(center, left), align_right(center, right), side_sum, side_carry);

    // Combine the 1s planes.
    u64 b0, c01;
    csa(top_sum, bottom_sum, side_sum, b0, c01);

    // Combine the 2s planes from top/bottom and the carry generated above.
    u64 s2, c2;
    csa(c01, top_carry, bottom_carry, s2, c2);

    // Add the remaining 2s plane from left/right.
    u64 b1, c3;
    ha(s2, side_carry, b1, c3);

    // Final 4s / 8s planes.
    u64 b2, b3;
    ha(c2, c3, b2, b3);

    // Neighbor count per bit position:
    //   count = b0 + 2*b1 + 4*b2 + 8*b3
    //
    // Conway rule:
    //   next = (count == 3) | (center & (count == 2))
    //
    // This simplifies to:
    //   next = (b1 & ~(b2 | b3)) & (b0 | center)
    // because b1=1 and b2=b3=0 selects {2,3}, while b0 distinguishes 3 from 2.
    return (b1 & ~(b2 | b3)) & (b0 | center);
}

template <int TILE_WIDTH, int BLOCK_Y_LOG2>
static __device__ __forceinline__ void game_of_life_kernel_body(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    unsigned grid_dim,
    unsigned row_shift)
{
    static_assert(TILE_WIDTH == 16 || TILE_WIDTH == 32, "Unsupported tile width");
    static_assert(BLOCK_Y_LOG2 == 3 || BLOCK_Y_LOG2 == 4, "Unsupported block height");
    constexpr unsigned kTileLog2 = (TILE_WIDTH == 16 ? 4u : 5u);
    constexpr unsigned kLastLane = TILE_WIDTH - 1u;

    // Exact launch geometry:
    // - grid_dim is a power of two > 512, so it is divisible by both 8 and 16.
    // - words_per_row = grid_dim / 64 is also a power of two.
    // - For the 1024x1024 case, words_per_row == 16, so a 16-wide tile covers the
    //   entire packed row and width=16 shuffles partition each warp into two
    //   independent half-warps.
    // - For all larger valid grids, words_per_row is a multiple of 32, so a 32-wide
    //   tile maps exactly to one contiguous warp-sized row segment.
    //
    // The launch therefore covers only valid output words; there are no bounds
    // checks in the hot path.
    const unsigned y = (static_cast<unsigned>(blockIdx.y) << BLOCK_Y_LOG2) + threadIdx.y;
    const std::size_t row_stride = std::size_t{1} << row_shift;
    const std::size_t idx =
        (static_cast<std::size_t>(y) << row_shift) +
        (static_cast<std::size_t>(blockIdx.x) << kTileLog2) +
        static_cast<std::size_t>(threadIdx.x);

    const bool has_up = (y != 0u);
    const bool has_down = ((y + 1u) != grid_dim);
    const bool first_tile = (blockIdx.x == 0u);
    const bool last_tile = ((blockIdx.x + 1u) == gridDim.x);

    // Load the vertically aligned words. Outside the grid is treated as dead, so
    // missing top/bottom rows are represented by zero.
    const u64 center = input[idx];
    const u64 north = has_up ? input[idx - row_stride] : 0ull;
    const u64 south = has_down ? input[idx + row_stride] : 0ull;

    // Exchange horizontally adjacent center/north/south words within the warp.
    // width=TILE_WIDTH prevents the shuffle from crossing row boundaries.
    const u64 sh_left_center  = __shfl_up_sync  (kFullWarpMask, center, 1, TILE_WIDTH);
    const u64 sh_right_center = __shfl_down_sync(kFullWarpMask, center, 1, TILE_WIDTH);
    const u64 sh_left_north   = __shfl_up_sync  (kFullWarpMask, north,  1, TILE_WIDTH);
    const u64 sh_right_north  = __shfl_down_sync(kFullWarpMask, north,  1, TILE_WIDTH);
    const u64 sh_left_south   = __shfl_up_sync  (kFullWarpMask, south,  1, TILE_WIDTH);
    const u64 sh_right_south  = __shfl_down_sync(kFullWarpMask, south,  1, TILE_WIDTH);

    // Most lanes get left/right halo words from shuffles. Only the tile boundary
    // lanes need extra global loads. Missing left/right words at the outer grid
    // border are zero to honor the "outside is dead" rule.
    u64 left = sh_left_center;
    u64 north_left = sh_left_north;
    u64 south_left = sh_left_south;
    if (threadIdx.x == 0u) {
        left = first_tile ? 0ull : input[idx - 1];
        north_left = (has_up && !first_tile) ? input[idx - row_stride - 1] : 0ull;
        south_left = (has_down && !first_tile) ? input[idx + row_stride - 1] : 0ull;
    }

    u64 right = sh_right_center;
    u64 north_right = sh_right_north;
    u64 south_right = sh_right_south;
    if (threadIdx.x == kLastLane) {
        right = last_tile ? 0ull : input[idx + 1];
        north_right = (has_up && !last_tile) ? input[idx - row_stride + 1] : 0ull;
        south_right = (has_down && !last_tile) ? input[idx + row_stride + 1] : 0ull;
    }

    output[idx] = step_word(
        north_left, north, north_right,
        left, center, right,
        south_left, south, south_right);
}

__global__ __launch_bounds__(256)
void game_of_life_kernel32x8(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    unsigned grid_dim,
    unsigned row_shift)
{
    game_of_life_kernel_body<32, kBlockY8Log2>(input, output, grid_dim, row_shift);
}

__global__ __launch_bounds__(512)
void game_of_life_kernel32x16(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    unsigned grid_dim,
    unsigned row_shift)
{
    game_of_life_kernel_body<32, kBlockY16Log2>(input, output, grid_dim, row_shift);
}

__global__ __launch_bounds__(256)
void game_of_life_kernel16x16(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    unsigned grid_dim,
    unsigned row_shift)
{
    game_of_life_kernel_body<16, kBlockY16Log2>(input, output, grid_dim, row_shift);
}

// Host-side cold-path helper; a tiny loop is fine here.
inline unsigned log2_pow2(unsigned x) {
    unsigned s = 0;
    while (x > 1u) {
        x >>= 1;
        ++s;
    }
    return s;
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // The caller provides device pointers and handles synchronization. This function
    // intentionally performs only the launch, with no host-side synchronization or
    // launch-time error querying in the hot path.
    const unsigned grid_dim = static_cast<unsigned>(grid_dimensions);
    const unsigned words_per_row = grid_dim >> kWordBitShift;
    const unsigned row_shift = log2_pow2(words_per_row);

    if (words_per_row == 16u) {
        // 1024x1024 grid: exactly 16 packed words per row. A 16x16 launch keeps all
        // lanes productive and uses width=16 shuffles so each half-warp is isolated.
        const dim3 block(16u, kBlockY16, 1u);
        const dim3 grid(1u, grid_dim >> kBlockY16Log2, 1u);
        game_of_life_kernel16x16<<<grid, block>>>(input, output, grid_dim, row_shift);
    } else if ((grid_dim >> kBlockY8Log2) <= kGridYLimit) {
        // Preferred large-grid shape: one full warp per row segment, 256 threads/block.
        const dim3 block(32u, kBlockY8, 1u);
        const dim3 grid(words_per_row >> 5, grid_dim >> kBlockY8Log2, 1u);
        game_of_life_kernel32x8<<<grid, block>>>(input, output, grid_dim, row_shift);
    } else {
        // Very large grids that still fit on current H100/A100-class hardware can
        // exceed CUDA's 2D grid.y limit with 8 rows/block. Fall back to 16 rows/block
        // while preserving the same one-warp-per-row-segment mapping.
        const dim3 block(32u, kBlockY16, 1u);
        const dim3 grid(words_per_row >> 5, grid_dim >> kBlockY16Log2, 1u);
        game_of_life_kernel32x16<<<grid, block>>>(input, output, grid_dim, row_shift);
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
