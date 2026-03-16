// High-throughput single-step Conway's Game of Life for a square grid stored as
// 8x8 bit tiles in device memory.
//
// Tile-local bit packing assumed by this kernel:
//   bits  0.. 7 : row 0, columns 0..7
//   bits  8..15 : row 1, columns 0..7
//   ...
//   bits 56..63 : row 7, columns 0..7
//
// In other words, bit index = row * 8 + col, the low byte is the tile's top row,
// and within each byte the LSB is the leftmost cell.
//
// Performance-oriented design choices:
//   - One thread updates one 8x8 tile = 64 cells at once.
//   - blockDim.x is fixed to one warp, so west/east neighbor tiles are exchanged
//     with warp shuffles instead of extra global loads.
//   - Neighbor counts are accumulated with bit-sliced carry-save adders, so all
//     64 cell counts in the tile are processed in parallel with integer bitwise ops.
//   - No shared or texture memory: for this access pattern on modern NVIDIA
//     data-center GPUs, global memory + warp shuffles is simpler and faster.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

using u64 = std::uint64_t;

constexpr int kBlockX = 32;      // exactly one warp in X
constexpr int kBlockY = 8;
constexpr int kBlockXShift = 5;  // log2(kBlockX)
constexpr int kBlockYShift = 3;  // log2(kBlockY)

static_assert(kBlockX == 32, "This kernel assumes blockDim.x == 32.");
static_assert((1 << kBlockXShift) == kBlockX, "kBlockXShift must match kBlockX.");
static_assert((1 << kBlockYShift) == kBlockY, "kBlockYShift must match kBlockY.");

constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

constexpr u64 kCol0      = 0x0101010101010101ULL;
constexpr u64 kCol7      = 0x8080808080808080ULL;
constexpr u64 kNotCol0   = 0xFEFEFEFEFEFEFEFEULL;
constexpr u64 kNotCol7   = 0x7F7F7F7F7F7F7F7FULL;
constexpr u64 kByteMask  = 0xFFULL;
constexpr u64 kByteNoLsb = 0xFEULL;
constexpr u64 kByteNoMsb = 0x7FULL;

// 64-bit shuffle wrappers. Casting through unsigned long long avoids depending on
// the exact host-side typedef behind std::uint64_t.
__device__ __forceinline__ u64 shfl_up_u64(u64 v, unsigned delta) {
    return static_cast<u64>(
        __shfl_up_sync(kFullWarpMask, static_cast<unsigned long long>(v), delta));
}

__device__ __forceinline__ u64 shfl_down_u64(u64 v, unsigned delta) {
    return static_cast<u64>(
        __shfl_down_sync(kFullWarpMask, static_cast<unsigned long long>(v), delta));
}

// Carry-save adder for bit-sliced popcount.
// For every bit position i:
//   sum[i]   = low bit of a[i] + b[i] + c[i]
//   carry[i] = high bit of a[i] + b[i] + c[i]
//
// Because each u64 contains 64 independent one-bit cell values, this is a
// 64-way parallel 3:2 compressor.
__device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

__global__ __launch_bounds__(kBlockX * kBlockY, 2)
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              int tiles_per_row) {
    // Each warp covers 32 consecutive tiles on one tile-row; each thread owns one tile.
    const int x = (static_cast<int>(blockIdx.x) << kBlockXShift) + static_cast<int>(threadIdx.x);
    const int y = (static_cast<int>(blockIdx.y) << kBlockYShift) + static_cast<int>(threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);

    const int last = tiles_per_row - 1;
    const bool has_north = (y != 0);
    const bool has_south = (y != last);

    const std::size_t row_stride = static_cast<std::size_t>(tiles_per_row);
    const std::size_t idx =
        static_cast<std::size_t>(y) * row_stride + static_cast<std::size_t>(x);

    // Direct loads: north / center / south tile words.
    // Missing rows beyond the grid are treated as zero, which exactly implements
    // the "outside cells are dead" boundary condition.
    const u64 n = has_north ? input[idx - row_stride] : 0ULL;
    const u64 c = input[idx];
    const u64 s = has_south ? input[idx + row_stride] : 0ULL;

    // West/east neighbor tiles are usually supplied by the adjacent lane in the warp.
    u64 w  = shfl_up_u64(c, 1);
    u64 e  = shfl_down_u64(c, 1);
    u64 nw = shfl_up_u64(n, 1);
    u64 ne = shfl_down_u64(n, 1);
    u64 sw = shfl_up_u64(s, 1);
    u64 se = shfl_down_u64(s, 1);

    // Shuffle exchange does not cross warp boundaries, so only lane 0 / lane 31
    // need explicit global loads. That keeps average memory traffic per tile low.
    if (lane == 0) {
        const bool has_west = (x != 0);
        w  = has_west ? input[idx - 1] : 0ULL;
        nw = (has_west && has_north) ? input[idx - row_stride - 1] : 0ULL;
        sw = (has_west && has_south) ? input[idx + row_stride - 1] : 0ULL;
    }

    if (lane == kBlockX - 1) {
        const bool has_east = (x != last);
        e  = has_east ? input[idx + 1] : 0ULL;
        ne = (has_east && has_north) ? input[idx - row_stride + 1] : 0ULL;
        se = (has_east && has_south) ? input[idx + row_stride + 1] : 0ULL;
    }

    // Pre-extract the only parts of neighbor tiles needed for cross-tile alignment.
    const u64 c_no_col7 = c & kNotCol7;
    const u64 c_no_col0 = c & kNotCol0;
    const u64 n_row7    = n >> 56;          // north tile row 7 moved into low byte
    const u64 s_row0    = s & kByteMask;    // south tile row 0
    const u64 w_col7    = w & kCol7;        // west tile column 7 in-place
    const u64 e_col0    = e & kCol0;        // east tile column 0 in-place

    // Build aligned neighbor bitboards. Each bit position now refers to the same
    // cell position inside the current tile.
    const u64 north_bits = (c << 8) | n_row7;
    const u64 south_bits = (c >> 8) | (s_row0 << 56);
    const u64 west_bits  = (c_no_col7 << 1) | (w_col7 >> 7);

    u64 ones_a, twos_a;
    csa(north_bits, south_bits, west_bits, ones_a, twos_a);

    const u64 east_bits =
        (c_no_col0 >> 1) | (e_col0 << 7);

    const u64 northwest_bits =
        (c_no_col7 << 9) |
        ((n_row7 & kByteNoMsb) << 1) |
        (w_col7 << 1) |
        (nw >> 63);

    const u64 northeast_bits =
        (c_no_col0 << 7) |
        ((n_row7 & kByteNoLsb) >> 1) |
        (e_col0 << 15) |
        (((ne >> 56) & 0x1ULL) << 7);

    u64 ones_b, twos_b;
    csa(east_bits, northwest_bits, northeast_bits, ones_b, twos_b);

    const u64 southwest_bits =
        (c_no_col7 >> 7) |
        ((s_row0 & kByteNoMsb) << 57) |
        (w_col7 >> 15) |
        (((sw >> 7) & 0x1ULL) << 56);

    const u64 southeast_bits =
        (c_no_col0 >> 9) |
        ((s_row0 & kByteNoLsb) << 55) |
        (e_col0 >> 1) |
        ((se & 0x1ULL) << 63);

    const u64 ones_c = southwest_bits ^ southeast_bits;
    const u64 twos_c = southwest_bits & southeast_bits;

    // Bit-sliced popcount of the eight neighbors.
    u64 ones, twos_d;
    csa(ones_a, ones_b, ones_c, ones, twos_d);

    u64 twos_e, fours_a;
    csa(twos_a, twos_b, twos_c, twos_e, fours_a);

    // Let count = ones + 2*K, where K = twos_a + twos_b + twos_c + twos_d.
    // Counts 2 and 3 are exactly the cases K == 1.
    //
    // twos_e ^ twos_d is parity(K).
    // fours_a is already 1 whenever K >= 3.
    // The remaining K != 1 case is K == 2, and parity(K) is then 0.
    // Therefore:
    //   ((twos_e ^ twos_d) & ~fours_a)  <=>  neighbor count is 2 or 3.
    const u64 count_2_or_3 = (twos_e ^ twos_d) & ~fours_a;

    // Final Life rule:
    //   count == 2 -> survive iff currently alive
    //   count == 3 -> alive regardless of current state
    //
    // Under count_2_or_3:
    //   count == 2 => ones == 0, so (ones | c) == c
    //   count == 3 => ones == 1, so (ones | c) == 1
    output[idx] = count_2_or_3 & (ones | c);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Out-of-place update: input and output are expected to be different device buffers.
    //
    // One 64-bit word stores one 8x8 tile, so the tile grid is (N/8) x (N/8).
    // The problem contract guarantees N is a power of 2 and N > 512, therefore:
    //   tiles_per_row = N / 8
    // is also a power of 2, and is divisible by both 32 and 8. That lets us launch
    // an exact grid without a per-thread x/y bounds check in the kernel body.
    const int tiles_per_row = grid_dimensions >> 3;

    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(static_cast<unsigned int>(tiles_per_row >> kBlockXShift),
                    static_cast<unsigned int>(tiles_per_row >> kBlockYShift),
                    1);

    // Caller controls synchronization and error handling policy; this function only
    // enqueues a single asynchronous kernel launch on the default stream.
    game_of_life_step_kernel<<<grid, block>>>(input, output, tiles_per_row);
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
