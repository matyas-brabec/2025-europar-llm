#include <cuda_runtime.h>
#include <cstdint>

/*
 * Optimized CUDA Conway's Game of Life for square, power-of-two boards allocated with cudaMalloc.
 *
 * Design choices tuned for modern data-center GPUs (A100/H100 class):
 *
 * 1) No shared memory:
 *    Each warp owns a horizontal stripe and walks multiple rows, keeping the 3-row rolling window
 *    in registers. Horizontal neighbors come from warp shuffles, so each input row is loaded once
 *    per tile and then reused directly from registers.
 *
 * 2) Explicit dead halo:
 *    Cells outside the grid are defined as dead. Top/bottom halos are represented by nullptr row
 *    pointers (uniform across the warp); left/right halos are explicit scalars carried only by the
 *    first/last lane of the warp.
 *
 * 3) Two kernels:
 *    - 32-wide, 1-cell-per-lane kernel: used only for the smallest legal board (1024x1024) so the
 *      GPU sees more independent warps.
 *    - 64-wide, 2-cells-per-lane kernel: used for larger boards. It halves shuffle/store overhead
 *      per output cell and uses aligned 16-bit loads/stores for adjacent cells.
 *
 * 4) 64-bit addressing:
 *    Even though grid_dimensions is an int, dim*dim can exceed 2^31 for large boards that still
 *    fit in GPU memory, so all row-stride arithmetic uses size_t.
 *
 * Assumptions intentionally exploited on the hot path:
 * - grid_dimensions is a power of 2 and > 512, so every tile dimension used here divides the board
 *   exactly and no partial-tile handling is needed.
 * - bool storage is byte-sized in CUDA device memory; values written are always 0 or 1.
 */

namespace {

using u8  = unsigned char;
using u16 = std::uint16_t;

static_assert(sizeof(bool) == 1, "This implementation assumes byte-sized bool storage.");

constexpr int      kWarpLanes            = 32;
constexpr int      kLastLane             = kWarpLanes - 1;
constexpr int      kRowsPerWarp          = 16;          // Vertical rolling window depth per warp.
constexpr int      kRowsPerWarpShift     = 4;           // log2(16)
constexpr int      kWarpsPerBlock        = 4;           // 128-thread blocks.
constexpr int      kBlockThreads         = kWarpLanes * kWarpsPerBlock;
constexpr int      kTileHeight           = kRowsPerWarp * kWarpsPerBlock;  // 64 rows per block.
constexpr int      kTileHeightShift      = 6;           // log2(64)
constexpr int      kScalarTileXShift     = 5;           // 32 columns per scalar-kernel block.
constexpr int      kPairedTileXShift     = 6;           // 64 columns per paired-kernel block.
constexpr int      kPairedKernelMinDim   = 2048;        // 1024x1024 uses the scalar kernel.
constexpr int      kMinBlocksPerSM       = 4;
constexpr unsigned kFullMask             = 0xffffffffu;

__device__ __forceinline__ u8 apply_life_rule(const int self, const int neighbors) {
    // self is already 0/1; branchless rule evaluation keeps the inner loop compact.
    return static_cast<u8>((neighbors == 3) | (self & (neighbors == 2)));
}

__device__ __forceinline__ u16 pack_two_bytes(const u8 lo, const u8 hi) {
    // NVIDIA GPUs are little-endian, so low/high byte order maps directly to cells [x, x+1].
    return static_cast<u16>(lo) | (static_cast<u16>(hi) << 8);
}

__device__ __forceinline__ u16 load_u16_aligned(const u8* ptr) {
    // The paired kernel always calls this with 2-byte aligned addresses:
    // - x is even by construction
    // - row stride is even because grid_dimensions is a power of two (> 512)
    u16 value;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(ptr));
    return value;
}

__device__ __forceinline__ void store_u16_aligned(u8* ptr, const u16 value) {
    asm volatile("st.global.u16 [%0], %1;" :: "l"(ptr), "h"(value));
}

__device__ __forceinline__ void load_scalar_row(
    const u8* row,
    const int x,
    const bool has_left,
    const bool has_right,
    const bool first_lane,
    const bool last_lane,
    int& center,
    int& left_halo,
    int& right_halo)
{
    center = 0;
    left_halo = 0;
    right_halo = 0;

    if (row != nullptr) {
        center = static_cast<int>(row[x]);
        if (first_lane && has_left) {
            left_halo = static_cast<int>(row[x - 1]);
        }
        if (last_lane && has_right) {
            right_halo = static_cast<int>(row[x + 1]);
        }
    }
}

__device__ __forceinline__ void load_paired_row(
    const u8* row,
    const int x0,
    const bool has_left,
    const bool has_right,
    const bool first_lane,
    const bool last_lane,
    int& a,
    int& b,
    int& left_halo,
    int& right_halo)
{
    a = 0;
    b = 0;
    left_halo = 0;
    right_halo = 0;

    if (row != nullptr) {
        const u16 pair = load_u16_aligned(row + x0);
        a = static_cast<int>(pair & 0x00ffu);
        b = static_cast<int>(pair >> 8);

        if (first_lane && has_left) {
            left_halo = static_cast<int>(row[x0 - 1]);
        }
        if (last_lane && has_right) {
            right_halo = static_cast<int>(row[x0 + 2]);
        }
    }
}

/*
 * Narrow kernel:
 * - One warp covers 32 columns x 16 rows.
 * - One block covers 32 columns x 64 rows.
 * - Chosen only for 1024x1024 so the GPU sees twice as many independent warps as the paired kernel.
 */
__global__ __launch_bounds__(kBlockThreads, kMinBlocksPerSM)
void game_of_life_kernel_scalar(const bool* __restrict__ input,
                                bool* __restrict__ output,
                                int dim)
{
    const u8* __restrict__ in  = reinterpret_cast<const u8*>(input);
    u8* __restrict__ out       = reinterpret_cast<u8*>(output);

    const int lane = static_cast<int>(threadIdx.x);  // 0..31, intentionally one full warp in X.
    const int warp = static_cast<int>(threadIdx.y);  // 0..3
    const bool first_lane = (lane == 0);
    const bool last_lane  = (lane == kLastLane);

    // Power-of-two dimensions let the compiler emit shifts for these tile coordinates.
    const int x      = (static_cast<int>(blockIdx.x) << kScalarTileXShift) + lane;
    const int y_base = (static_cast<int>(blockIdx.y) << kTileHeightShift) + (warp << kRowsPerWarpShift);

    const bool has_left  = (blockIdx.x != 0u);
    const bool has_right = (blockIdx.x + 1u != gridDim.x);

    const size_t stride = static_cast<size_t>(dim);  // 64-bit because dim*dim can exceed 2^31.
    const size_t base   = static_cast<size_t>(y_base) * stride;

    const u8* center_row = in + base;
    const u8* north_row  = (y_base != 0) ? (center_row - stride) : nullptr;
    const u8* south_row  = center_row + stride;  // Valid under the stated tile divisibility constraints.

    int n, nl, nr;
    int c, cl, cr;
    int s, sl, sr;

    load_scalar_row(north_row,  x, has_left, has_right, first_lane, last_lane, n, nl, nr);
    load_scalar_row(center_row, x, has_left, has_right, first_lane, last_lane, c, cl, cr);
    load_scalar_row(south_row,  x, has_left, has_right, first_lane, last_lane, s, sl, sr);

    u8* out_ptr = out + base + static_cast<size_t>(x);
    int next_south_y = y_base + 2;

    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        int nL = __shfl_up_sync  (kFullMask, n, 1);
        int nR = __shfl_down_sync(kFullMask, n, 1);
        int cL = __shfl_up_sync  (kFullMask, c, 1);
        int cR = __shfl_down_sync(kFullMask, c, 1);
        int sL = __shfl_up_sync  (kFullMask, s, 1);
        int sR = __shfl_down_sync(kFullMask, s, 1);

        if (first_lane) {
            nL = nl;
            cL = cl;
            sL = sl;
        }
        if (last_lane) {
            nR = nr;
            cR = cr;
            sR = sr;
        }

        const int north3    = nL + n + nR;
        const int south3    = sL + s + sR;
        const int center_lr = cL + cR;
        const int neighbors = north3 + south3 + center_lr;

        *out_ptr = apply_life_rule(c, neighbors);
        out_ptr += stride;

        if (i + 1 < kRowsPerWarp) {
            n  = c;  nl = cl;  nr = cr;
            c  = s;  cl = sl;  cr = sr;

            if (next_south_y < dim) {
                south_row += stride;
            } else {
                south_row = nullptr;
            }

            load_scalar_row(south_row, x, has_left, has_right, first_lane, last_lane, s, sl, sr);
            ++next_south_y;
        }
    }
}

/*
 * Wide kernel:
 * - One warp covers 64 columns x 16 rows.
 * - Each lane updates two horizontally adjacent cells per row.
 * - One block covers 64 columns x 64 rows.
 * - Used for >= 2048x2048 because it cuts shuffle/store overhead per output cell while keeping
 *   enough block-level parallelism on large boards.
 */
__global__ __launch_bounds__(kBlockThreads, kMinBlocksPerSM)
void game_of_life_kernel_paired(const bool* __restrict__ input,
                                bool* __restrict__ output,
                                int dim)
{
    const u8* __restrict__ in  = reinterpret_cast<const u8*>(input);
    u8* __restrict__ out       = reinterpret_cast<u8*>(output);

    const int lane = static_cast<int>(threadIdx.x);  // 0..31
    const int warp = static_cast<int>(threadIdx.y);  // 0..3
    const bool first_lane = (lane == 0);
    const bool last_lane  = (lane == kLastLane);

    // x0 is always even; each lane owns cells [x0, x0 + 1].
    const int x0     = (static_cast<int>(blockIdx.x) << kPairedTileXShift) + (lane << 1);
    const int y_base = (static_cast<int>(blockIdx.y) << kTileHeightShift) + (warp << kRowsPerWarpShift);

    const bool has_left  = (blockIdx.x != 0u);
    const bool has_right = (blockIdx.x + 1u != gridDim.x);

    const size_t stride = static_cast<size_t>(dim);
    const size_t base   = static_cast<size_t>(y_base) * stride;

    const u8* center_row = in + base;
    const u8* north_row  = (y_base != 0) ? (center_row - stride) : nullptr;
    const u8* south_row  = center_row + stride;

    int na, nb, nl, nr;
    int ca, cb, cl, cr;
    int sa, sb, sl, sr;

    load_paired_row(north_row,  x0, has_left, has_right, first_lane, last_lane, na, nb, nl, nr);
    load_paired_row(center_row, x0, has_left, has_right, first_lane, last_lane, ca, cb, cl, cr);
    load_paired_row(south_row,  x0, has_left, has_right, first_lane, last_lane, sa, sb, sl, sr);

    u8* out_ptr = out + base + static_cast<size_t>(x0);
    int next_south_y = y_base + 2;

    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        // For a pair [a, b], the only cross-lane horizontal dependencies are:
        // - previous lane's b (left neighbor of a)
        // - next lane's a     (right neighbor of b)
        int n_prev_b = __shfl_up_sync  (kFullMask, nb, 1);
        int n_next_a = __shfl_down_sync(kFullMask, na, 1);
        int c_prev_b = __shfl_up_sync  (kFullMask, cb, 1);
        int c_next_a = __shfl_down_sync(kFullMask, ca, 1);
        int s_prev_b = __shfl_up_sync  (kFullMask, sb, 1);
        int s_next_a = __shfl_down_sync(kFullMask, sa, 1);

        if (first_lane) {
            n_prev_b = nl;
            c_prev_b = cl;
            s_prev_b = sl;
        }
        if (last_lane) {
            n_next_a = nr;
            c_next_a = cr;
            s_next_a = sr;
        }

        // Two outputs share most of their north/south contributions.
        const int north_pair = na + nb;
        const int south_pair = sa + sb;
        const int shared_ns  = north_pair + south_pair;

        const int center_for_a = c_prev_b + cb;
        const int center_for_b = ca + c_next_a;

        const int neighbors_a = shared_ns + center_for_a + n_prev_b + s_prev_b;
        const int neighbors_b = shared_ns + center_for_b + n_next_a + s_next_a;

        const u8 next_a = apply_life_rule(ca, neighbors_a);
        const u8 next_b = apply_life_rule(cb, neighbors_b);

        store_u16_aligned(out_ptr, pack_two_bytes(next_a, next_b));
        out_ptr += stride;

        if (i + 1 < kRowsPerWarp) {
            na = ca;  nb = cb;  nl = cl;  nr = cr;
            ca = sa;  cb = sb;  cl = sl;  cr = sr;

            if (next_south_y < dim) {
                south_row += stride;
            } else {
                south_row = nullptr;
            }

            load_paired_row(south_row, x0, has_left, has_right, first_lane, last_lane, sa, sb, sl, sr);
            ++next_south_y;
        }
    }
}

}  // namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // The caller requested pure performance and handles synchronization externally.
    // Kernel launches therefore remain fully asynchronous.

    const dim3 block(kWarpLanes, kWarpsPerBlock, 1);

    // The problem guarantees a power-of-two side length > 512, so all launches tile the domain
    // exactly with shifts rather than ceil-divisions.
    if (grid_dimensions >= kPairedKernelMinDim) {
        const dim3 grid(static_cast<unsigned>(grid_dimensions >> kPairedTileXShift),
                        static_cast<unsigned>(grid_dimensions >> kTileHeightShift),
                        1);
        game_of_life_kernel_paired<<<grid, block>>>(input, output, grid_dimensions);
    } else {
        const dim3 grid(static_cast<unsigned>(grid_dimensions >> kScalarTileXShift),
                        static_cast<unsigned>(grid_dimensions >> kTileHeightShift),
                        1);
        game_of_life_kernel_scalar<<<grid, block>>>(input, output, grid_dimensions);
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
