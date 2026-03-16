#include <cuda_runtime.h>
#include <cstdint>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

static_assert(sizeof(bool) == 1, "This implementation assumes 1-byte bool storage.");

constexpr u32 kFullWarpMask         = 0xFFFFFFFFu;
constexpr int kWarpWidthLog2        = 5;
constexpr int kWarpWidth            = 1 << kWarpWidthLog2;   // 32 cells, exactly one warp wide.
constexpr int kRowsPerWarpLog2      = 1;
constexpr int kRowsPerWarp          = 1 << kRowsPerWarpLog2; // 2 output rows per warp.
constexpr int kWarpsPerBlockLog2    = 4;
constexpr int kWarpsPerBlock        = 1 << kWarpsPerBlockLog2; // 16 warps/block.
constexpr int kTileRowsLog2         = kRowsPerWarpLog2 + kWarpsPerBlockLog2;
constexpr int kTileRows             = 1 << kTileRowsLog2;    // 32 rows/block.
constexpr u64 kMax32BitElementCount = u64{1} << 32;

// Adds a 1-bit mask into a bit-sliced 4-bit counter.
// For every bit position/cell independently, (b3:b2:b1:b0) stores the current
// neighbor count in binary. One call updates 32 counters in parallel.
__device__ __forceinline__ void add_mask_to_counter(
    const u32 x,
    u32& b0,
    u32& b1,
    u32& b2,
    u32& b3)
{
    const u32 c0 = b0 & x;
    b0 ^= x;
    const u32 c1 = b1 & c0;
    b1 ^= c0;
    const u32 c2 = b2 & c1;
    b2 ^= c1;
    b3 ^= c2;
}

// Computes the next-state bitmask for 32 cells at once.
//
// north/center/south are the packed 32-bit rows.
// *_lr packs the single horizontal halo bits for the corresponding row:
//   bit 0 -> cell immediately left  of the 32-cell segment
//   bit 1 -> cell immediately right of the 32-cell segment
//
// Outside-the-grid cells are represented by zero halo bits / zero halo rows.
__device__ __forceinline__ u32 compute_next_mask(
    const u32 north,
    const u32 center,
    const u32 south,
    const u32 north_lr,
    const u32 center_lr,
    const u32 south_lr)
{
    const u32 nw = (north << 1) | (north_lr & 1u);
    const u32 nn = north;
    const u32 ne = (north >> 1) | ((north_lr & 2u) << 30);

    const u32 ww = (center << 1) | (center_lr & 1u);
    const u32 ee = (center >> 1) | ((center_lr & 2u) << 30);

    const u32 sw = (south << 1) | (south_lr & 1u);
    const u32 ss = south;
    const u32 se = (south >> 1) | ((south_lr & 2u) << 30);

    u32 b0 = 0;
    u32 b1 = 0;
    u32 b2 = 0;
    u32 b3 = 0;

    add_mask_to_counter(nw, b0, b1, b2, b3);
    add_mask_to_counter(nn, b0, b1, b2, b3);
    add_mask_to_counter(ne, b0, b1, b2, b3);
    add_mask_to_counter(ww, b0, b1, b2, b3);
    add_mask_to_counter(ee, b0, b1, b2, b3);
    add_mask_to_counter(sw, b0, b1, b2, b3);
    add_mask_to_counter(ss, b0, b1, b2, b3);
    add_mask_to_counter(se, b0, b1, b2, b3);

    // Counts 2 and 3 share: b1 = 1 and b2 = b3 = 0.
    // Within that subset, b0 distinguishes:
    //   b0 = 0 -> count == 2  => survives only if currently alive
    //   b0 = 1 -> count == 3  => next state is alive regardless
    const u32 common_2_or_3 = (~(b2 | b3)) & b1;
    return common_2_or_3 & (center | b0);
}

// grid_dimensions is guaranteed to be a power of two, so log2 is trivial and
// computed once on the host. This is intentionally kept out of the hot kernel.
inline int log2_pow2(int x)
{
    unsigned int v = static_cast<unsigned int>(x);
    int l = 0;
    while (v > 1u) {
        v >>= 1;
        ++l;
    }
    return l;
}

// High-throughput single-step Game of Life kernel.
//
// Mapping:
// - blockDim = (32, 16)
// - blockDim.x is exactly one warp, so each threadIdx.y slice is a full warp.
// - Each warp loads two consecutive 32-cell row segments and turns them into two
//   32-bit masks with __ballot_sync.
// - 16 warps/block x 2 rows/warp = one 32x32 cell tile per block.
// - Shared memory holds packed row masks for the tile plus one halo row above
//   and below; horizontal halo cells are packed into 2-bit row_halo values.
// - The Life rule is evaluated with bit-sliced counters, so integer bitwise ops
//   update 32 cells at a time.
//
// The public API is bool-based, but the input is read as raw bytes. This keeps
// input loads on simple byte accesses while accepting any non-zero true encoding.
// Output is written as bool for full type correctness.
template <typename IndexT>
__global__ __launch_bounds__(kWarpWidth * kWarpsPerBlock)
void life_step_kernel(
    const unsigned char* __restrict__ input,
    bool* __restrict__ output,
    int dim_log2)
{
    __shared__ u32 row_masks[kTileRows + 2];
    __shared__ u32 row_halo[kTileRows + 2]; // bit0 = left halo, bit1 = right halo

    const int lane          = threadIdx.x; // 0..31
    const int warp_row_pair = threadIdx.y; // 0..15

    const bool has_left   = (blockIdx.x > 0);
    const bool has_right  = (blockIdx.x + 1 < gridDim.x);
    const bool has_top    = (blockIdx.y > 0);
    const bool has_bottom = (blockIdx.y + 1 < gridDim.y);

    // Exploit the power-of-two board dimension: all row addressing is done with shifts.
    const IndexT stride         = IndexT{1} << dim_log2;
    const IndexT x_base         = static_cast<IndexT>(blockIdx.x) << kWarpWidthLog2;
    const IndexT block_row_base = static_cast<IndexT>(blockIdx.y) << (kTileRowsLog2 + dim_log2);

    const int    local_row0  = warp_row_pair << kRowsPerWarpLog2; // 2 * warp_row_pair
    const int    shared_row0 = local_row0 + 1;
    const int    shared_row1 = shared_row0 + 1;
    const IndexT row0_base   = block_row_base + (static_cast<IndexT>(local_row0) << dim_log2) + x_base;
    const IndexT row1_base   = row0_base + stride;

    // Pack two consecutive rows handled by this warp.
    const u32 mask0 = __ballot_sync(kFullWarpMask, input[row0_base + lane] != 0);
    const u32 mask1 = __ballot_sync(kFullWarpMask, input[row1_base + lane] != 0);

    // Each warp publishes its two packed rows and their horizontal halo bits.
    u32 packed_center_lr = 0;
    if (lane == 0) {
        u32 center_lr0 = 0;
        u32 center_lr1 = 0;

        if (has_left) {
            center_lr0 |= static_cast<u32>(input[row0_base - 1] != 0);
            center_lr1 |= static_cast<u32>(input[row1_base - 1] != 0);
        }
        if (has_right) {
            center_lr0 |= static_cast<u32>(input[row0_base + kWarpWidth] != 0) << 1;
            center_lr1 |= static_cast<u32>(input[row1_base + kWarpWidth] != 0) << 1;
        }

        row_masks[shared_row0] = mask0;
        row_masks[shared_row1] = mask1;
        row_halo[shared_row0]  = center_lr0;
        row_halo[shared_row1]  = center_lr1;

        packed_center_lr = center_lr0 | (center_lr1 << 2);
    }
    packed_center_lr = __shfl_sync(kFullWarpMask, packed_center_lr, 0);

    const u32 center_lr0 = packed_center_lr & 0x3u;
    const u32 center_lr1 = (packed_center_lr >> 2) & 0x3u;

    // Top halo row for the block.
    if (warp_row_pair == 0) {
        if (has_top) {
            const IndexT top_base = row0_base - stride;
            const u32 top_mask = __ballot_sync(kFullWarpMask, input[top_base + lane] != 0);

            if (lane == 0) {
                u32 top_lr = 0;
                if (has_left) {
                    top_lr |= static_cast<u32>(input[top_base - 1] != 0);
                }
                if (has_right) {
                    top_lr |= static_cast<u32>(input[top_base + kWarpWidth] != 0) << 1;
                }
                row_masks[0] = top_mask;
                row_halo[0]  = top_lr;
            }
        } else if (lane == 0) {
            row_masks[0] = 0;
            row_halo[0]  = 0;
        }
    }

    // Bottom halo row for the block.
    if (warp_row_pair == kWarpsPerBlock - 1) {
        if (has_bottom) {
            const IndexT bottom_base = row1_base + stride;
            const u32 bottom_mask = __ballot_sync(kFullWarpMask, input[bottom_base + lane] != 0);

            if (lane == 0) {
                u32 bottom_lr = 0;
                if (has_left) {
                    bottom_lr |= static_cast<u32>(input[bottom_base - 1] != 0);
                }
                if (has_right) {
                    bottom_lr |= static_cast<u32>(input[bottom_base + kWarpWidth] != 0) << 1;
                }
                row_masks[kTileRows + 1] = bottom_mask;
                row_halo[kTileRows + 1]  = bottom_lr;
            }
        } else if (lane == 0) {
            row_masks[kTileRows + 1] = 0;
            row_halo[kTileRows + 1]  = 0;
        }
    }

    __syncthreads();

    // Reuse registers wherever possible:
    // - row0 uses north from shared, center=row0, south=row1
    // - row1 uses north=row0, center=row1, south from shared
    const u32 north0_mask = row_masks[shared_row0 - 1];
    const u32 north0_lr   = row_halo[shared_row0 - 1];
    const u32 south1_mask = row_masks[shared_row1 + 1];
    const u32 south1_lr   = row_halo[shared_row1 + 1];

    const u32 next0 = compute_next_mask(north0_mask, mask0, mask1, north0_lr, center_lr0, center_lr1);
    const u32 next1 = compute_next_mask(mask0, mask1, south1_mask, center_lr0, center_lr1, south1_lr);

    output[row0_base + lane] = ((next0 >> lane) & 1u) != 0u;
    output[row1_base + lane] = ((next1 >> lane) & 1u) != 0u;
}

} // namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // The problem guarantees a power-of-two square board larger than 512, so the
    // launch tiles the board exactly with 32x32-cell blocks and no partial work.
    const int dim_log2 = log2_pow2(grid_dimensions);

    const dim3 block(kWarpWidth, kWarpsPerBlock, 1);
    const dim3 grid(
        static_cast<unsigned int>(grid_dimensions) >> kWarpWidthLog2,
        static_cast<unsigned int>(grid_dimensions) >> kTileRowsLog2,
        1);

    const auto* input_bytes = reinterpret_cast<const unsigned char*>(input);

    // Most practical boards fit in <= 2^32 elements, so keep a fast 32-bit index
    // path. Very large boards still work via the 64-bit instantiation.
    const u64 cells =
        static_cast<u64>(static_cast<unsigned int>(grid_dimensions)) *
        static_cast<u64>(static_cast<unsigned int>(grid_dimensions));

    // Launch only; caller is responsible for any desired synchronization.
    if (cells <= kMax32BitElementCount) {
        life_step_kernel<u32><<<grid, block>>>(input_bytes, output, dim_log2);
    } else {
        life_step_kernel<u64><<<grid, block>>>(input_bytes, output, dim_log2);
    }
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// this label is used to identify the memory layout of the input and output arrays by the framework
// MEMORY_LAYOUT: BOOLS
