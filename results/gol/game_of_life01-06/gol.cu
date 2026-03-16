#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace detail {

// Optimized CUDA Game of Life step for the exact API given.
//
// Rationale for this design:
// - The public API exposes bool grids in device memory, i.e. the natural external format is 1 byte/cell.
//   Repacking to a bit-compressed internal format can be worthwhile for multi-step simulations that stay
//   packed across generations, but for a single public step it adds extra kernels and extra global traffic.
//   This implementation therefore stays in the API-native byte layout.
// - The update is a 3x3 stencil and is memory-bandwidth bound. The kernel minimizes global traffic by
//   staging a tile in shared memory, doing one aligned 4-byte vector load/store per thread in X,
//   and computing two output rows per thread to amortize halo overhead in Y.
// - Cells outside the grid are dead. Edge blocks materialize this exactly by zero-filling halo rows/columns
//   in shared memory, so the hot update path has no per-cell bounds checks.
//
// Targeted for modern datacenter NVIDIA GPUs (A100/H100 class).

using byte = unsigned char;

static_assert(sizeof(bool) == 1, "This implementation relies on 1-byte bool storage.");

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;

// Each thread updates 4 horizontally adjacent cells.
constexpr int kCellsPerThreadXShift = 2;
constexpr int kCellsPerThreadX      = 1 << kCellsPerThreadXShift;  // 4

// Each thread updates 2 rows, separated by kThreadsY.
constexpr int kRowsPerThreadY = 2;

// A block covers 128x16 output cells:
//   32 threads in X * 4 cells/thread = 128 cells
//    8 threads in Y * 2 rows/thread  = 16 cells
constexpr int kTileXShift = 7;  // 128
constexpr int kTileYShift = 4;  // 16
constexpr int kTileX      = 1 << kTileXShift;
constexpr int kTileY      = 1 << kTileYShift;

static_assert(kTileX == kThreadsX * kCellsPerThreadX, "Tile/thread X mapping mismatch.");
static_assert(kTileY == kThreadsY * kRowsPerThreadY,  "Tile/thread Y mapping mismatch.");

// Shared-memory row layout (136 bytes total):
//   [0..2]     unused padding
//   [3]        left halo
//   [4..131]   128 in-tile cells
//   [132]      right halo
//   [133..135] unused padding
//
// Starting the tile body at offset 4 means each thread's 4-cell chunk is 4-byte aligned, which enables
// aligned uchar4 vector loads/stores and also gives warp accesses a bank-friendly 4-byte stride.
constexpr int kSharedXOffset = 4;
constexpr int kSharedStride  = kTileX + (2 * kSharedXOffset);  // 136 bytes
constexpr int kSharedRows    = kTileY + 2;                     // + top and bottom halo rows

static_assert((kSharedStride & 3) == 0, "Shared stride must stay 4-byte aligned.");

__device__ __forceinline__ uchar4 load_u8x4(const byte* p) {
    return *reinterpret_cast<const uchar4*>(p);
}

__device__ __forceinline__ void store_u8x4(byte* p, const uchar4 v) {
    *reinterpret_cast<uchar4*>(p) = v;
}

__device__ __forceinline__ uchar4 zero_u8x4() {
    return make_uchar4(0, 0, 0, 0);
}

// Compute 4 horizontally adjacent outputs from 3 shared-memory rows.
// A 6-column sliding window is cheaper than recomputing all 8 neighbors independently for each cell.
__device__ __forceinline__ uchar4 evolve4(const byte* __restrict__ up,
                                          const byte* __restrict__ mid,
                                          const byte* __restrict__ dn) {
    const uint32_t c0 = static_cast<uint32_t>(up[0]) + static_cast<uint32_t>(mid[0]) + static_cast<uint32_t>(dn[0]);
    const uint32_t c1 = static_cast<uint32_t>(up[1]) + static_cast<uint32_t>(mid[1]) + static_cast<uint32_t>(dn[1]);
    const uint32_t c2 = static_cast<uint32_t>(up[2]) + static_cast<uint32_t>(mid[2]) + static_cast<uint32_t>(dn[2]);
    const uint32_t c3 = static_cast<uint32_t>(up[3]) + static_cast<uint32_t>(mid[3]) + static_cast<uint32_t>(dn[3]);
    const uint32_t c4 = static_cast<uint32_t>(up[4]) + static_cast<uint32_t>(mid[4]) + static_cast<uint32_t>(dn[4]);
    const uint32_t c5 = static_cast<uint32_t>(up[5]) + static_cast<uint32_t>(mid[5]) + static_cast<uint32_t>(dn[5]);

    const uint32_t w0 = c0 + c1 + c2;
    const uint32_t w1 = w0 - c0 + c3;
    const uint32_t w2 = w1 - c1 + c4;
    const uint32_t w3 = w2 - c2 + c5;

    const uint32_t self0 = static_cast<uint32_t>(mid[1]);
    const uint32_t self1 = static_cast<uint32_t>(mid[2]);
    const uint32_t self2 = static_cast<uint32_t>(mid[3]);
    const uint32_t self3 = static_cast<uint32_t>(mid[4]);

    const uint32_t n0 = w0 - self0;
    const uint32_t n1 = w1 - self1;
    const uint32_t n2 = w2 - self2;
    const uint32_t n3 = w3 - self3;

    // Branchless Game of Life rule:
    // next = (neighbors == 3) || (alive && neighbors == 2)
    const unsigned char o0 = static_cast<unsigned char>((n0 == 3u) | ((self0 != 0u) & (n0 == 2u)));
    const unsigned char o1 = static_cast<unsigned char>((n1 == 3u) | ((self1 != 0u) & (n1 == 2u)));
    const unsigned char o2 = static_cast<unsigned char>((n2 == 3u) | ((self2 != 0u) & (n2 == 2u)));
    const unsigned char o3 = static_cast<unsigned char>((n3 == 3u) | ((self3 != 0u) & (n3 == 2u)));

    return make_uchar4(o0, o1, o2, o3);
}

__global__ __launch_bounds__(kThreadsX * kThreadsY)
void game_of_life_step_kernel(const byte* __restrict__ in,
                              byte* __restrict__ out,
                              int n) {
    __shared__ __align__(16) byte tile[kSharedRows * kSharedStride];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);

    const int last_bx = static_cast<int>(gridDim.x) - 1;
    const int last_by = static_cast<int>(gridDim.y) - 1;

    // Exact shifts are valid because the problem guarantees power-of-two dimensions and we choose power-of-two tiles.
    const int gx0 = (bx << kTileXShift) + (tx << kCellsPerThreadXShift);
    const int gy0 = (by << kTileYShift) + ty;

    const int shx      = kSharedXOffset + (tx << kCellsPerThreadXShift);
    const int sh_left  = shx - 1;
    const int sh_right = shx + kCellsPerThreadX;

    // Shared rows owned by this thread:
    //   sy0 in [1..8]   -> first  8 output rows of the block
    //   sy1 in [9..16]  -> second 8 output rows of the block
    // Shared row 0  is the top halo, row 17 is the bottom halo.
    const int sy0 = ty + 1;
    const int sy1 = sy0 + kThreadsY;

    const bool has_left   = (bx != 0);
    const bool has_right  = (bx != last_bx);
    const bool has_top    = (by != 0);
    const bool has_bottom = (by != last_by);

    const size_t n64 = static_cast<size_t>(n);
    const size_t g0  = static_cast<size_t>(gy0) * n64 + static_cast<size_t>(gx0);
    const size_t g1  = g0 + static_cast<size_t>(kThreadsY) * n64;  // second row handled by this thread

    byte* const s0 = tile + sy0 * kSharedStride;
    byte* const s1 = tile + sy1 * kSharedStride;

    // Load the two center rows handled by this thread.
    store_u8x4(s0 + shx, load_u8x4(in + g0));
    store_u8x4(s1 + shx, load_u8x4(in + g1));

    // Load left/right halo bytes for both rows.
    if (tx == 0) {
        s0[sh_left] = has_left ? in[g0 - 1] : 0;
        s1[sh_left] = has_left ? in[g1 - 1] : 0;
    }
    if (tx == kThreadsX - 1) {
        s0[sh_right] = has_right ? in[g0 + kCellsPerThreadX] : 0;
        s1[sh_right] = has_right ? in[g1 + kCellsPerThreadX] : 0;
    }

    // Top halo row: only the ty==0 warp needs to load it.
    if (ty == 0) {
        byte* const s_top = tile;
        if (has_top) {
            const size_t g_top = g0 - n64;
            store_u8x4(s_top + shx, load_u8x4(in + g_top));
            if (tx == 0) {
                s_top[sh_left] = has_left ? in[g_top - 1] : 0;
            }
            if (tx == kThreadsX - 1) {
                s_top[sh_right] = has_right ? in[g_top + kCellsPerThreadX] : 0;
            }
        } else {
            store_u8x4(s_top + shx, zero_u8x4());
            if (tx == 0) {
                s_top[sh_left] = 0;
            }
            if (tx == kThreadsX - 1) {
                s_top[sh_right] = 0;
            }
        }
    }

    // Bottom halo row: only the ty==7 warp needs to load it.
    if (ty == kThreadsY - 1) {
        byte* const s_bottom = tile + (kTileY + 1) * kSharedStride;
        if (has_bottom) {
            const size_t g_bottom = g1 + n64;
            store_u8x4(s_bottom + shx, load_u8x4(in + g_bottom));
            if (tx == 0) {
                s_bottom[sh_left] = has_left ? in[g_bottom - 1] : 0;
            }
            if (tx == kThreadsX - 1) {
                s_bottom[sh_right] = has_right ? in[g_bottom + kCellsPerThreadX] : 0;
            }
        } else {
            store_u8x4(s_bottom + shx, zero_u8x4());
            if (tx == 0) {
                s_bottom[sh_left] = 0;
            }
            if (tx == kThreadsX - 1) {
                s_bottom[sh_right] = 0;
            }
        }
    }

    __syncthreads();

    // All boundary handling is now encoded in the halo bytes, so the update path below is the same for
    // interior and edge cells: no per-cell bounds checks, just stencil arithmetic on shared memory.
    const byte* const up0  = tile + (sy0 - 1) * kSharedStride + sh_left;
    const byte* const mid0 = up0 + kSharedStride;
    const byte* const dn0  = mid0 + kSharedStride;

    const byte* const up1  = tile + (sy1 - 1) * kSharedStride + sh_left;
    const byte* const mid1 = up1 + kSharedStride;
    const byte* const dn1  = mid1 + kSharedStride;

    store_u8x4(out + g0, evolve4(up0, mid0, dn0));
    store_u8x4(out + g1, evolve4(up1, mid1, dn1));
}

}  // namespace detail

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Problem guarantees:
    // - square grid
    // - power-of-two dimensions
    // - dimensions > 512
    //
    // Therefore grid_dimensions is always divisible by both 128 and 16, so this launch covers the
    // full grid exactly with no partial blocks. The caller manages synchronization, so this function
    // only enqueues the kernel and performs no extra sync/error-checking work on the hot path.
    const dim3 block(detail::kThreadsX, detail::kThreadsY);
    const dim3 grid(static_cast<unsigned int>(grid_dimensions) >> detail::kTileXShift,
                    static_cast<unsigned int>(grid_dimensions) >> detail::kTileYShift);

    detail::game_of_life_step_kernel<<<grid, block>>>(
        reinterpret_cast<const detail::byte*>(input),
        reinterpret_cast<detail::byte*>(output),
        grid_dimensions);
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
