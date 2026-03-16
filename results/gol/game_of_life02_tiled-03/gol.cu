#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// CUDA Game of Life for a bit-packed 8x8-tile layout.
//
// Assumed in-memory layout (the natural interpretation of "one uint64_t per 8x8 tile"):
//   * tiles are stored in row-major order
//   * within one 64-bit tile, bit index (8 * local_y + local_x) stores cell (local_x, local_y)
//     so every byte is one tile row
//
// Why this implementation is fast on modern NVIDIA GPUs:
//   * one thread updates one entire 8x8 tile = 64 cells at once
//   * neighbor counts are computed with bit-sliced boolean adders, so all 64 cells in the tile
//     are updated in parallel with plain integer logic
//   * along X, one warp spans exactly 32 adjacent tiles; west/east/diagonal neighbor tiles are
//     exchanged with warp shuffles, so the hot interior kernel only loads center/north/south
//     tiles from global memory for most threads
//   * Y boundaries are split into a tiny dedicated kernel, keeping the large interior kernel free
//     of per-thread Y-boundary checks
//
// Fast-path assumption:
//   * input and output are distinct device buffers (standard ping-pong update)

namespace {

using u64 = unsigned long long;
using index_t = std::size_t;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "u64 must match std::uint64_t size");

constexpr int kWarpTiles = 32;   // one warp covers 32 adjacent tiles along X
constexpr int kBlockRows = 2;    // two warps per CTA; enough CTA granularity even for 1024x1024 grids
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

constexpr u64 kCol0    = 0x0101010101010101ull; // x == 0 in every byte-row
constexpr u64 kCol7    = 0x8080808080808080ull; // x == 7 in every byte-row
constexpr u64 kNotCol0 = 0xFEFEFEFEFEFEFEFEull;
constexpr u64 kNotCol7 = 0x7F7F7F7F7F7F7F7Full;
constexpr u64 kByteMask = 0xFFull;

__device__ __forceinline__ u64 load_ro(const u64* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

// 3-input majority; the compiler maps this well to LOP3 on recent architectures.
__device__ __forceinline__ u64 majority(u64 a, u64 b, u64 c) {
    return (a & b) | (a & c) | (b & c);
}

// Count four 1-bit inputs at each bit position into three bit-planes:
//   ones  -> count bit 0
//   twos  -> count bit 1
//   fours -> count bit 2
// No carries propagate across bit positions: every bit position is an independent population count.
__device__ __forceinline__ void count4(
    u64 a, u64 b, u64 c, u64 d,
    u64& ones, u64& twos, u64& fours)
{
    const u64 xab = a ^ b;
    const u64 xcd = c ^ d;
    const u64 cab = a & b;
    const u64 ccd = c & d;

    ones = xab ^ xcd;
    const u64 carry = xab & xcd;
    twos = cab ^ ccd ^ carry;
    fours = majority(cab, ccd, carry);
}

// Compute the next-generation 8x8 tile from the 3x3 tile neighborhood.
//
// Each argument is an 8x8 bitboard (one bit per cell). The center tile is 'c'; the other
// arguments are the eight neighboring tiles. Missing tiles outside the grid are passed as zero.
//
// The core idea is to build eight "aligned neighbor" bitboards:
//   for example, in n_aligned, bit p equals the north neighbor of output cell p.
// Then a carry-save style boolean adder counts those eight aligned bitboards per bit position.
__device__ __forceinline__ u64 step_tile(
    u64 c,  u64 n,  u64 s,
    u64 w,  u64 e,
    u64 nw, u64 ne,
    u64 sw, u64 se)
{
    // Horizontal neighbors from the center row, with byte-local wrap suppressed.
    const u64 w_aligned = ((c & kNotCol7) << 1) | ((w & kCol7) >> 7);
    const u64 e_aligned = ((c & kNotCol0) >> 1) | ((e & kCol0) << 7);

    // Halo rows from the tiles above/below.
    const u64 n_row7 = n >> 56;         // bottom row of north tile -> low byte
    const u64 s_row0 = s & kByteMask;   // top row of south tile   -> low byte

    // Vertical neighbors from the center column.
    const u64 n_aligned = (c << 8) | n_row7;
    const u64 s_aligned = (c >> 8) | (s_row0 << 56);

    // First half of the 8-neighbor population count: N, S, W, E.
    u64 ones0, twos0, fours0;
    count4(n_aligned, s_aligned, w_aligned, e_aligned, ones0, twos0, fours0);

    // Diagonals reuse the already-aligned W/E bitboards and inject only one halo byte.
    const u64 nw_aligned =
        (w_aligned << 8) |
        (((n_row7 & 0x7Full) << 1) | (nw >> 63));

    const u64 ne_aligned =
        (e_aligned << 8) |
        (((n_row7 & 0xFEull) >> 1) | (((ne >> 56) & 0x1ull) << 7));

    const u64 sw_aligned =
        (w_aligned >> 8) |
        ((((s_row0 & 0x7Full) << 1) | ((sw & 0x80ull) >> 7)) << 56);

    const u64 se_aligned =
        (e_aligned >> 8) |
        ((((s_row0 & 0xFEull) >> 1) | ((se & 0x1ull) << 7)) << 56);

    // Second half of the 8-neighbor population count: NW, NE, SW, SE.
    u64 ones1, twos1, fours1;
    count4(nw_aligned, ne_aligned, sw_aligned, se_aligned, ones1, twos1, fours1);

    // Add the two 3-bit partial counts into a full 4-bit count (0..8).
    const u64 bit0 = ones0 ^ ones1;
    const u64 carry0 = ones0 & ones1;

    const u64 bit1 = twos0 ^ twos1 ^ carry0;
    const u64 carry1 = majority(twos0, twos1, carry0);

    const u64 bit2 = fours0 ^ fours1 ^ carry1;
    const u64 bit3 = majority(fours0, fours1, carry1);

    // Life rule:
    //   next = (count == 3) | (alive & (count == 2))
    //
    // With count bits [bit3 bit2 bit1 bit0]:
    //   count in {2,3}  <=>  bit1 && !bit2 && !bit3
    //   among {2,3}, count==3 is exactly bit0==1
    //
    // So the rule simplifies to:
    //   next = (count in {2,3}) & (bit0 | alive)
    const u64 two_or_three = bit1 & ~(bit2 | bit3);
    return two_or_three & (bit0 | c);
}

// Hot kernel: all tile rows except the first and last.
//
// Launch shape is fixed to blockDim.x == 32, blockDim.y == 2:
// each warp handles 32 adjacent tiles in one tile row, so west/east and diagonal neighbor tiles
// come from warp shuffles. Only lane 0 / lane 31 perform extra halo loads across warp boundaries.
__global__ __launch_bounds__(64)
void game_of_life_interior_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int tiles_per_dim)
{
    const int x = static_cast<int>(blockIdx.x) * kWarpTiles + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y) + 1;
    if (y >= tiles_per_dim - 1) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    const index_t pitch = static_cast<index_t>(tiles_per_dim);
    const index_t idx = static_cast<index_t>(y) * pitch + static_cast<index_t>(x);

    const u64 c = load_ro(input + idx);
    const u64 n = load_ro(input + idx - pitch);
    const u64 s = load_ro(input + idx + pitch);

    u64 w  = __shfl_up_sync(kFullWarpMask, c, 1);
    u64 e  = __shfl_down_sync(kFullWarpMask, c, 1);
    u64 nw = __shfl_up_sync(kFullWarpMask, n, 1);
    u64 ne = __shfl_down_sync(kFullWarpMask, n, 1);
    u64 sw = __shfl_up_sync(kFullWarpMask, s, 1);
    u64 se = __shfl_down_sync(kFullWarpMask, s, 1);

    if (lane == 0) {
        if (x > 0) {
            w  = load_ro(input + idx - 1);
            nw = load_ro(input + idx - pitch - 1);
            sw = load_ro(input + idx + pitch - 1);
        } else {
            w = 0ull;
            nw = 0ull;
            sw = 0ull;
        }
    } else if (lane == kWarpTiles - 1) {
        if (x + 1 < tiles_per_dim) {
            e  = load_ro(input + idx + 1);
            ne = load_ro(input + idx - pitch + 1);
            se = load_ro(input + idx + pitch + 1);
        } else {
            e = 0ull;
            ne = 0ull;
            se = 0ull;
        }
    }

    output[idx] = step_tile(c, n, s, w, e, nw, ne, sw, se);
}

// Tiny boundary kernel for the first and last tile rows.
//
// Left/right tile boundaries are already handled naturally by the warp-halo logic above.
// Splitting only the Y-boundaries keeps the much larger interior kernel branch-free in Y.
__global__ __launch_bounds__(64)
void game_of_life_top_bottom_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int tiles_per_dim)
{
    const int x = static_cast<int>(blockIdx.x) * kWarpTiles + static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const index_t pitch = static_cast<index_t>(tiles_per_dim);

    if (threadIdx.y == 0) {
        // Top tile row: north-side halo is dead.
        const index_t idx = static_cast<index_t>(x);
        const u64 c = load_ro(input + idx);
        const u64 s = load_ro(input + idx + pitch);

        u64 w  = __shfl_up_sync(kFullWarpMask, c, 1);
        u64 e  = __shfl_down_sync(kFullWarpMask, c, 1);
        u64 sw = __shfl_up_sync(kFullWarpMask, s, 1);
        u64 se = __shfl_down_sync(kFullWarpMask, s, 1);

        if (lane == 0) {
            if (x > 0) {
                w  = load_ro(input + idx - 1);
                sw = load_ro(input + idx + pitch - 1);
            } else {
                w = 0ull;
                sw = 0ull;
            }
        } else if (lane == kWarpTiles - 1) {
            if (x + 1 < tiles_per_dim) {
                e  = load_ro(input + idx + 1);
                se = load_ro(input + idx + pitch + 1);
            } else {
                e = 0ull;
                se = 0ull;
            }
        }

        output[idx] = step_tile(c, 0ull, s, w, e, 0ull, 0ull, sw, se);
    } else {
        // Bottom tile row: south-side halo is dead.
        const index_t idx = (pitch - 1) * pitch + static_cast<index_t>(x);
        const u64 c = load_ro(input + idx);
        const u64 n = load_ro(input + idx - pitch);

        u64 w  = __shfl_up_sync(kFullWarpMask, c, 1);
        u64 e  = __shfl_down_sync(kFullWarpMask, c, 1);
        u64 nw = __shfl_up_sync(kFullWarpMask, n, 1);
        u64 ne = __shfl_down_sync(kFullWarpMask, n, 1);

        if (lane == 0) {
            if (x > 0) {
                w  = load_ro(input + idx - 1);
                nw = load_ro(input + idx - pitch - 1);
            } else {
                w = 0ull;
                nw = 0ull;
            }
        } else if (lane == kWarpTiles - 1) {
            if (x + 1 < tiles_per_dim) {
                e  = load_ro(input + idx + 1);
                ne = load_ro(input + idx - pitch + 1);
            } else {
                e = 0ull;
                ne = 0ull;
            }
        }

        output[idx] = step_tile(c, n, 0ull, w, e, nw, ne, 0ull, 0ull);
    }
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Problem guarantees:
    //   * square grid
    //   * power-of-two side length
    //   * side length > 512
    // Therefore:
    //   tiles_per_dim = grid_dimensions / 8 is a power of two >= 128,
    //   hence divisible by 32 and perfect for one-warp-per-32-tiles mapping.
    //
    // No host synchronization is performed here; launch ordering/synchronization is owned
    // by the caller. The function operates directly on the original bit-packed tile format.
    const int tiles_per_dim = grid_dimensions >> 3;

    const auto* in  = reinterpret_cast<const u64*>(input);
    auto* out = reinterpret_cast<u64*>(output);

    const dim3 block(kWarpTiles, kBlockRows);

    // Interior: all tile rows except the first and last.
    const dim3 grid_interior(
        static_cast<unsigned>(tiles_per_dim / kWarpTiles),
        static_cast<unsigned>((tiles_per_dim - 2 + kBlockRows - 1) / kBlockRows),
        1u);

    game_of_life_interior_kernel<<<grid_interior, block>>>(in, out, tiles_per_dim);

    // Top and bottom tile rows.
    const dim3 grid_top_bottom(
        static_cast<unsigned>(tiles_per_dim / kWarpTiles),
        1u,
        1u);

    game_of_life_top_bottom_kernel<<<grid_top_bottom, block>>>(in, out, tiles_per_dim);
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
// MEMORY_LAYOUT: TILES
