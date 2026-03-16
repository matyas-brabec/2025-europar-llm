#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// Internal scalar aliases.
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The problem guarantees power-of-two dimensions > 512, so the number of 8x8 tiles
// per row/column is also a power of two and at least 128. We choose a block shape
// that maps exactly to that geometry:
//
// - blockDim.x == 32, so each warp spans one contiguous horizontal strip of 32 tiles.
//   This lets us obtain left/right neighboring tiles with warp shuffles instead of
//   extra global memory loads.
// - blockDim.y == 8, so each CTA contains 8 warps and still stays light enough for
//   high occupancy on A100/H100-class GPUs.
constexpr int kTileLog2 = 3;   // 8 cells per tile edge.
constexpr int kBlockX   = 32;  // Exactly one warp wide.
constexpr int kBlockY   = 8;
constexpr int kThreadsPerBlock = kBlockX * kBlockY;

static_assert(kBlockX == 32, "This kernel relies on blockDim.x being exactly one warp.");
constexpr u32 kLastLane = static_cast<u32>(kBlockX - 1);
constexpr u32 kFullWarpMask = 0xFFFFFFFFu;

// Intra-tile bit layout assumed by this implementation:
//
//   bit index = 8 * local_y + local_x
//
// So each byte is one tile row, row 0 lives in bits [0..7], row 7 in bits [56..63],
// and within each row-byte, column 0 is the least-significant bit.
//
// The 64-bit masks below therefore select tile borders directly.
constexpr u64 kCol0 = 0x0101010101010101ull;
constexpr u64 kCol7 = 0x8080808080808080ull;
constexpr u64 kRow0 = 0x00000000000000FFull;
constexpr u64 kRow7 = 0xFF00000000000000ull;

constexpr u64 kNotCol0 = 0xFEFEFEFEFEFEFEFEull;
constexpr u64 kNotCol7 = 0x7F7F7F7F7F7F7F7Full;

constexpr u64 kCornerNW = 0x8000000000000000ull;
constexpr u64 kCornerNE = 0x0100000000000000ull;
constexpr u64 kCornerSW = 0x0000000000000080ull;
constexpr u64 kCornerSE = 0x0000000000000001ull;

// Warp shuffle helpers for 64-bit values.
// Implemented via two 32-bit shuffles so the code is robust regardless of the exact
// host typedef behind std::uint64_t on the compilation platform.
__device__ __forceinline__ u64 shfl_up_1_u64(const u64 v) {
    const u32 lo = __shfl_up_sync(kFullWarpMask, static_cast<u32>(v), 1);
    const u32 hi = __shfl_up_sync(kFullWarpMask, static_cast<u32>(v >> 32), 1);
    return (static_cast<u64>(hi) << 32) | static_cast<u64>(lo);
}

__device__ __forceinline__ u64 shfl_down_1_u64(const u64 v) {
    const u32 lo = __shfl_down_sync(kFullWarpMask, static_cast<u32>(v), 1);
    const u32 hi = __shfl_down_sync(kFullWarpMask, static_cast<u32>(v >> 32), 1);
    return (static_cast<u64>(hi) << 32) | static_cast<u64>(lo);
}

// Bit-sliced full adder.
// For every bit position independently, this adds three 1-bit values and returns:
//   sum   = low bit of the 3-input sum
//   carry = high bit of the 3-input sum
//
// Because one 64-bit word represents 64 cells, this is effectively 64 independent
// full adders running in parallel.
__device__ __forceinline__ void full_adder(
    const u64 a, const u64 b, const u64 c, u64& sum, u64& carry) {
    const u64 t = a ^ b;
    sum   = t ^ c;
    carry = (a & b) | (c & t);
}

// One thread updates one 8x8 output tile.
//
// Key performance choices:
// - No shared memory: horizontal reuse is cheaper via warp shuffles, and the remaining
//   vertical reuse is handled well by the normal caches on modern datacenter GPUs.
// - No bounds checks for thread validity: the problem guarantees the launch geometry
//   divides the tiled grid exactly.
// - Only 3 compulsory global loads per interior thread (center/north/south tiles).
//   West/east neighbors of those rows are exchanged within the warp; only lane 0 and
//   lane 31 perform extra cross-warp/block fallback loads.
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const std::size_t tiles_per_dim) {

    const u32 lane = threadIdx.x;

    const std::size_t tx =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kBlockX) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t ty =
        static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(kBlockY) +
        static_cast<std::size_t>(threadIdx.y);

    const std::size_t idx  = ty * tiles_per_dim + tx;
    const std::size_t last = tiles_per_dim - 1;

    const bool has_north = (ty != 0);
    const bool has_south = (ty != last);
    const bool has_west  = (tx != 0);
    const bool has_east  = (tx != last);

    // Each thread loads only its own column from the previous/current/next tile row.
    const u64 c = input[idx];
    const u64 n = has_north ? input[idx - tiles_per_dim] : 0ull;
    const u64 s = has_south ? input[idx + tiles_per_dim] : 0ull;

    // Horizontal neighbors come from warp shuffles because each warp spans one
    // contiguous 32-tile strip. Only warp-edge lanes need true global fallback loads.
    u64 w  = shfl_up_1_u64(c);
    u64 e  = shfl_down_1_u64(c);
    u64 nw = shfl_up_1_u64(n);
    u64 ne = shfl_down_1_u64(n);
    u64 sw = shfl_up_1_u64(s);
    u64 se = shfl_down_1_u64(s);

    if (lane == 0) {
        w  = has_west ? input[idx - 1] : 0ull;
        nw = (has_north && has_west) ? input[idx - tiles_per_dim - 1] : 0ull;
        sw = (has_south && has_west) ? input[idx + tiles_per_dim - 1] : 0ull;
    }

    if (lane == kLastLane) {
        e  = has_east ? input[idx + 1] : 0ull;
        ne = (has_north && has_east) ? input[idx - tiles_per_dim + 1] : 0ull;
        se = (has_south && has_east) ? input[idx + tiles_per_dim + 1] : 0ull;
    }

    // Cache the border bits reused by multiple directional alignments.
    const u64 w_col7 = w & kCol7;
    const u64 e_col0 = e & kCol0;
    const u64 n_row7 = n & kRow7;
    const u64 s_row0 = s & kRow0;

    // Build the 8 aligned neighbor bitboards:
    // every bit position now represents "is the neighbor in that direction alive for
    // the current cell at this bit position?"
    //
    // Example: west has the cell immediately to the left of each current cell aligned
    // into the current cell's bit position, including cross-tile carry-in from the
    // western tile's column 7.
    u64 sum_a, carry_a;
    {
        const u64 west  = ((c << 1) & kNotCol0) | (w_col7 >> 7);
        const u64 east  = ((c >> 1) & kNotCol7) | (e_col0 << 7);
        const u64 north = (c << 8) | (n_row7 >> 56);
        full_adder(west, east, north, sum_a, carry_a);
    }

    u64 sum_b, carry_b;
    {
        const u64 south = (c >> 8) | (s_row0 << 56);

        const u64 northwest =
            ((c << 9) & kNotCol0) |
            ((n_row7 >> 55) & kNotCol0) |
            (w_col7 << 1) |
            ((nw & kCornerNW) >> 63);

        const u64 northeast =
            ((c << 7) & kNotCol7) |
            (n_row7 >> 57) |
            (e_col0 << 15) |
            ((ne & kCornerNE) >> 49);

        full_adder(south, northwest, northeast, sum_b, carry_b);
    }

    const u64 southwest =
        ((c >> 7) & kNotCol0) |
        (s_row0 << 57) |
        (w_col7 >> 15) |
        ((sw & kCornerSW) << 49);

    const u64 southeast =
        ((c >> 9) & kNotCol7) |
        ((s_row0 << 55) & kNotCol7) |
        (e_col0 >> 1) |
        ((se & kCornerSE) << 63);

    // Bit-sliced population count of the 8 neighbors.
    //
    // At this point we have 8 one-bit inputs per cell:
    //   {west, east, north, south, northwest, northeast, southwest, southeast}
    //
    // The adder tree below computes a 4-bit binary neighbor count for all 64 cells in
    // parallel. The final rule application is then:
    //   next = (count == 3) || (current && count == 2)
    //
    // Instead of materializing equality tests separately, we observe that:
    //   count in {2,3}  <=>  bit1 == 1 and bit2 == 0 and bit3 == 0
    // and the low bit decides between 2 and 3:
    //   bit0 == 0 -> count 2
    //   bit0 == 1 -> count 3
    const u64 partial_lo0 = sum_a ^ sum_b;
    const u64 partial_hi0 = sum_a & sum_b;
    const u64 partial_lo1 = southwest ^ southeast;
    const u64 partial_hi1 = southwest & southeast;

    const u64 bit0   = partial_lo0 ^ partial_lo1;
    const u64 carry0 = partial_lo0 & partial_lo1;

    u64 mid, high;
    full_adder(partial_hi0, partial_hi1, carry0, mid, high);

    u64 bit1, carry1;
    full_adder(mid, carry_a, carry_b, bit1, carry1);

    const u64 bit2 = carry1 ^ high;
    const u64 bit3 = carry1 & high;

    const u64 valid23 = bit1 & ~(bit2 | bit3);
    output[idx] = valid23 & (bit0 | c);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The hot path assumes standard ping-pong usage: input and output are distinct
    // device buffers. No synchronization is done here; the caller owns ordering.
    //
    // grid_dimensions is guaranteed to be a power of two > 512, so:
    //   tiles_per_dim = grid_dimensions / 8
    // is also a power of two and therefore divisible by both 32 and 8 exactly.
    // That lets the kernel omit per-thread bounds checks entirely.
    const std::size_t tiles_per_dim =
        static_cast<std::size_t>(grid_dimensions) >> kTileLog2;

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        static_cast<unsigned int>(tiles_per_dim / static_cast<std::size_t>(kBlockX)),
        static_cast<unsigned int>(tiles_per_dim / static_cast<std::size_t>(kBlockY)));

    game_of_life_kernel<<<grid, block>>>(input, output, tiles_per_dim);
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
