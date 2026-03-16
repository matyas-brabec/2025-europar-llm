#include <cuda_runtime.h>
#include <cstddef>

// High-performance one-step Conway's Game of Life for modern NVIDIA GPUs.
//
// Design notes:
// - blockDim.x == 32, so each warp owns one contiguous 32-cell row segment.
//   That makes left/right neighbors map naturally to lane +/- 1.
// - Each warp computes an 8-row strip, not just one row. The vertical
//   (north, center, south) values are rolled through registers, so after the
//   first row only one new load per thread is needed for each additional row.
// - The (north, center, south) triplet is packed into 3 bits. Two warp shuffles
//   exchange those packed triplets with the left/right lanes; a single popcount
//   then gives the 8-neighbor count.
// - Lane 0 and lane 31 keep a rolling halo column in registers for cross-warp
//   boundaries, so even the halo loads are amortized across the 8-row strip.
// - Shared memory is intentionally avoided: for byte-wide cell data it would
//   either suffer bank conflicts or require 32-bit expansion. Warp shuffles plus
//   rolling registers provide the same stencil reuse without synchronization.
// - The problem guarantees power-of-two dimensions > 512, so the launch geometry
//   exactly covers the grid and row-base computation can use shifts.

namespace {

constexpr int kBlockX               = 32;  // exactly one warp in X
constexpr int kBlockWarpsY          = 8;   // 8 independent warps per block
constexpr int kTileRowsPerWarp      = 8;   // each warp computes 8 consecutive rows
constexpr int kBlockRows            = kBlockWarpsY * kTileRowsPerWarp;  // 64 rows per block
constexpr int kBlockThreads         = kBlockX * kBlockWarpsY;

constexpr int kBlockXLog2           = 5;   // log2(32)
constexpr int kTileRowsPerWarpLog2  = 3;   // log2(8)
constexpr int kBlockRowsLog2        = 6;   // log2(64)

constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;
constexpr unsigned int kLastLane     = 31u;

static_assert(kBlockX == 32, "This kernel relies on one full warp covering 32 contiguous columns.");
static_assert((1 << kTileRowsPerWarpLog2) == kTileRowsPerWarp, "kTileRowsPerWarp must be a power of two.");
static_assert((1 << kBlockRowsLog2) == kBlockRows, "kBlockRows must be a power of two.");

// The input dimensions are guaranteed to be powers of two, so a simple shift-count
// computation is enough and avoids host-side dependencies on compiler-specific intrinsics.
inline int integer_log2_pow2(unsigned int v) {
    int log2 = 0;
    while (v > 1u) {
        v >>= 1u;
        ++log2;
    }
    return log2;
}

__device__ __forceinline__ unsigned int load_cell(const bool* __restrict__ grid, size_t idx) {
    // Typed bool loads guarantee canonical 0/1 semantics even if the caller's source
    // data did not originate from this kernel.
    return static_cast<unsigned int>(grid[idx]);
}

__device__ __forceinline__ unsigned int pack_triplet(unsigned int top,
                                                     unsigned int mid,
                                                     unsigned int bot) {
    // Pack three 0/1 values into bits [0,1,2].
    return top | (mid << 1) | (bot << 2);
}

// kInteriorY == true is the hot path used by almost all warps: their 8-row strip does
// not touch the top or bottom boundary, so all vertical boundary checks disappear.
// Only the first and last 8-row strips of the entire grid use kInteriorY == false.
template <bool kInteriorY>
__device__ __forceinline__ void process_warp_tile(const bool* __restrict__ input,
                                                  bool* __restrict__ output,
                                                  size_t idx,
                                                  unsigned int y,
                                                  unsigned int last,
                                                  size_t stride,
                                                  unsigned int lane,
                                                  bool use_left_edge,
                                                  bool use_right_edge) {
    // Rolling vertical window for the current column.
    unsigned int n;
    unsigned int c;
    unsigned int s;

    if (kInteriorY) {
        n = load_cell(input, idx - stride);
        c = load_cell(input, idx);
        s = load_cell(input, idx + stride);
    } else {
        n = (y != 0u)   ? load_cell(input, idx - stride) : 0u;
        c =               load_cell(input, idx);
        s = (y != last) ? load_cell(input, idx + stride) : 0u;
    }

    // Rolling halo window for the cross-warp neighbor column.
    // On lane 0 this is the left halo; on lane 31 this is the right halo.
    unsigned int edge_n = 0u;
    unsigned int edge_c = 0u;
    unsigned int edge_s = 0u;

    if (use_left_edge) {
        if (kInteriorY) {
            edge_n = load_cell(input, idx - stride - 1);
            edge_c = load_cell(input, idx - 1);
            edge_s = load_cell(input, idx + stride - 1);
        } else {
            edge_n = (y != 0u)   ? load_cell(input, idx - stride - 1) : 0u;
            edge_c =               load_cell(input, idx - 1);
            edge_s = (y != last) ? load_cell(input, idx + stride - 1) : 0u;
        }
    } else if (use_right_edge) {
        if (kInteriorY) {
            edge_n = load_cell(input, idx - stride + 1);
            edge_c = load_cell(input, idx + 1);
            edge_s = load_cell(input, idx + stride + 1);
        } else {
            edge_n = (y != 0u)   ? load_cell(input, idx - stride + 1) : 0u;
            edge_c =               load_cell(input, idx + 1);
            edge_s = (y != last) ? load_cell(input, idx + stride + 1) : 0u;
        }
    }

    #pragma unroll
    for (int iter = 0; iter < kTileRowsPerWarp; ++iter) {
        // Pack the vertical triplet for this thread and exchange the packed value
        // horizontally. That reduces six scalar shuffles down to two.
        const unsigned int packed = pack_triplet(n, c, s);

        unsigned int left_p  = __shfl_up_sync(kFullWarpMask, packed, 1);
        unsigned int right_p = __shfl_down_sync(kFullWarpMask, packed, 1);

        // Replace the undefined shuffle results on warp boundaries with the rolled halo.
        if (lane == 0u) {
            left_p = use_left_edge ? pack_triplet(edge_n, edge_c, edge_s) : 0u;
        } else if (lane == kLastLane) {
            right_p = use_right_edge ? pack_triplet(edge_n, edge_c, edge_s) : 0u;
        }

        // Bit layout for the popcount:
        //   left_p bits   -> NW, W,  SW
        //   n << 3        -> N
        //   s << 4        -> S
        //   right_p << 5  -> NE, E, SE
        // popcount(neighbor_bits) is therefore exactly the 8-neighbor count.
        const unsigned int neighbor_bits = left_p | (n << 3) | (s << 4) | (right_p << 5);
        const unsigned int neighbors = __popc(neighbor_bits);

        const unsigned int next =
            static_cast<unsigned int>(neighbors == 3u) |
            (c & static_cast<unsigned int>(neighbors == 2u));

        output[idx] = static_cast<bool>(next);

        if (iter != kTileRowsPerWarp - 1) {
            // Advance to the next output row in the strip.
            idx += stride;

            if (!kInteriorY) {
                ++y;
            }

            n = c;
            c = s;

            if (kInteriorY) {
                s = load_cell(input, idx + stride);

                if (use_left_edge) {
                    edge_n = edge_c;
                    edge_c = edge_s;
                    edge_s = load_cell(input, idx + stride - 1);
                } else if (use_right_edge) {
                    edge_n = edge_c;
                    edge_c = edge_s;
                    edge_s = load_cell(input, idx + stride + 1);
                }
            } else {
                const bool has_bottom = (y != last);
                s = has_bottom ? load_cell(input, idx + stride) : 0u;

                if (use_left_edge) {
                    edge_n = edge_c;
                    edge_c = edge_s;
                    edge_s = has_bottom ? load_cell(input, idx + stride - 1) : 0u;
                } else if (use_right_edge) {
                    edge_n = edge_c;
                    edge_c = edge_s;
                    edge_s = has_bottom ? load_cell(input, idx + stride + 1) : 0u;
                }
            }
        }
    }
}

__global__ __launch_bounds__(kBlockThreads, 4)
void game_of_life_step_kernel(const bool* __restrict__ input,
                              bool* __restrict__ output,
                              int grid_dimensions,
                              int grid_log2) {
    // blockDim = (32, 8):
    // - x dimension is exactly one warp
    // - y dimension is the number of independent warps in the block
    const unsigned int lane = threadIdx.x;
    const unsigned int x = (blockIdx.x << kBlockXLog2) + lane;
    const unsigned int start_y =
        (blockIdx.y << kBlockRowsLog2) + (threadIdx.y << kTileRowsPerWarpLog2);
    const unsigned int last = static_cast<unsigned int>(grid_dimensions - 1);

    const size_t stride = static_cast<size_t>(grid_dimensions);
    const size_t idx =
        (static_cast<size_t>(start_y) << grid_log2) + static_cast<size_t>(x);

    // Only lane 0 / lane 31 ever need explicit halo-column loads.
    const bool use_left_edge  = (lane == 0u)      && (x != 0u);
    const bool use_right_edge = (lane == kLastLane) && (x != last);

    // Almost all warps are interior in Y and can use the fully branch-free hot path.
    const bool interior_y_tile =
        (start_y != 0u) &&
        (start_y + static_cast<unsigned int>(kTileRowsPerWarp) <= last);

    if (interior_y_tile) {
        process_warp_tile<true>(input, output, idx, start_y, last, stride, lane,
                                use_left_edge, use_right_edge);
    } else {
        process_warp_tile<false>(input, output, idx, start_y, last, stride, lane,
                                 use_left_edge, use_right_edge);
    }
}

}  // namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Assumptions guaranteed by the prompt:
    // - input/output are device pointers from cudaMalloc
    // - grid is square, dimensions are a power of two, and > 512
    // - input and output are distinct ping-pong buffers
    //
    // No synchronization or error-querying is done here; the caller owns both.

    const unsigned int dim_u = static_cast<unsigned int>(grid_dimensions);
    const int grid_log2 = integer_log2_pow2(dim_u);

    const dim3 block(kBlockX, kBlockWarpsY);
    const dim3 grid(dim_u >> kBlockXLog2, dim_u >> kBlockRowsLog2);

    game_of_life_step_kernel<<<grid, block>>>(input, output, grid_dimensions, grid_log2);
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
