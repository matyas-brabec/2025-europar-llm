#include <cuda_runtime.h>

namespace {

/*
  Optimized one-step Conway's Game of Life for large square grids on modern NVIDIA GPUs.

  Performance-oriented design:
  - Split the problem into a large interior and a tiny boundary. The boundary is O(n),
    so it is handled by a separate simple kernel to keep the hot interior path free of
    per-cell boundary checks.
  - The interior kernel is warp-centric:
      * each warp covers 32 consecutive x positions,
      * horizontal neighbors are exchanged with warp shuffles,
      * each thread walks a short strip of rows in y and keeps a 3-row rolling window
        entirely in registers.
    This keeps global memory traffic close to the minimum while avoiding shared-memory
    tiling for byte cells (which would otherwise either bank-conflict or force a 4x
    expansion to 32-bit storage).
  - grid.y is capped and processed with a grid-stride loop, so very large grids do not
    launch millions of tiny blocks.

  Storage note:
  - The caller supplies bool buffers in device memory. CUDA device bool storage is 1 byte,
    so the buffers are safely aliased as bytes here and the kernels write canonical 0/1.
*/

static_assert(sizeof(bool) == 1, "This implementation assumes 1-byte bool storage.");

using cell_t = unsigned char;

constexpr int      kWarpSize        = 32;
constexpr int      kWarpsPerBlock   = 4;    // 128 threads/block keeps enough blocks even at 1024x1024.
constexpr int      kInteriorThreads = kWarpSize * kWarpsPerBlock;
constexpr int      kRowsPerStrip    = 16;   // Good reuse/occupancy balance on A100/H100 class GPUs.
constexpr int      kBoundaryThreads = 256;
constexpr int      kMaxGridY        = 128;  // Cap grid.y; the interior kernel grid-strides in y.
constexpr unsigned kFullMask        = 0xffffffffu;

__device__ __forceinline__ cell_t apply_life_rule(unsigned int neighbors, unsigned int self) {
    // neighbors excludes self and self is stored as 0/1.
    return static_cast<cell_t>((neighbors == 3u) | (self & (neighbors == 2u)));
}

__launch_bounds__(kInteriorThreads)
__global__ void life_interior_strip_kernel(const cell_t* __restrict__ in,
                                           cell_t* __restrict__ out,
                                           int n) {
    // Legal widths are powers of two > 512, so kInteriorThreads=128 divides every legal n.
    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid & (kWarpSize - 1);
    const bool lane0   = (lane == 0);
    const bool lane31  = (lane == (kWarpSize - 1));

    const int x        = static_cast<int>(blockIdx.x) * kInteriorThreads + tid;
    const int last     = n - 1;

    // Most threads are interior cells; only the global left/right border lanes suppress stores.
    const bool write_cell = (x > 0) && (x < last);

    // Only warp edge lanes need cross-warp halo columns from global memory.
    const bool need_left  = lane0  && (x > 0);
    const bool need_right = lane31 && (x < last);

    const size_t pitch    = static_cast<size_t>(n);
    const size_t x64      = static_cast<size_t>(x);
    const int    stride   = n;
    const int    stride2  = n + n;
    const int    y_stride = static_cast<int>(gridDim.y) * kRowsPerStrip;

    for (int y0 = static_cast<int>(blockIdx.y) * kRowsPerStrip + 1; y0 < last; y0 += y_stride) {
        const int remaining  = last - y0;
        const int valid_rows = (remaining < kRowsPerStrip) ? remaining : kRowsPerStrip;

        const size_t base = static_cast<size_t>(y0 - 1) * pitch + x64;

        // p points at row (y-1), current x. q points at output row y, current x.
        const cell_t* p = in  + base;
        cell_t*       q = out + base + pitch;

        // Rolling 3-row window in registers.
        unsigned int r0 = static_cast<unsigned int>(p[0]);
        unsigned int r1 = static_cast<unsigned int>(p[stride]);
        unsigned int r2 = static_cast<unsigned int>(p[stride2]);

        // Cached cross-warp halos for lane 0 / lane 31. They stay zero on the true grid borders.
        const cell_t* lp = nullptr;
        const cell_t* rp = nullptr;
        unsigned int  l0 = 0, l1 = 0, l2 = 0;
        unsigned int rr0 = 0, rr1 = 0, rr2 = 0;

        if (need_left) {
            lp = p - 1;
            l0 = static_cast<unsigned int>(lp[0]);
            l1 = static_cast<unsigned int>(lp[stride]);
            l2 = static_cast<unsigned int>(lp[stride2]);
        }
        if (need_right) {
            rp  = p + 1;
            rr0 = static_cast<unsigned int>(rp[0]);
            rr1 = static_cast<unsigned int>(rp[stride]);
            rr2 = static_cast<unsigned int>(rp[stride2]);
        }

        #pragma unroll
        for (int i = 0; i < kRowsPerStrip; ++i) {
            if (i < valid_rows) {
                // Horizontal neighbors come from the warp. Only the two warp-edge lanes override
                // with the cached cross-warp halo values loaded above.
                unsigned int up_l  = __shfl_up_sync  (kFullMask, r0, 1);
                unsigned int up_r  = __shfl_down_sync(kFullMask, r0, 1);
                unsigned int mid_l = __shfl_up_sync  (kFullMask, r1, 1);
                unsigned int mid_r = __shfl_down_sync(kFullMask, r1, 1);
                unsigned int dn_l  = __shfl_up_sync  (kFullMask, r2, 1);
                unsigned int dn_r  = __shfl_down_sync(kFullMask, r2, 1);

                if (lane0) {
                    up_l  = l0;
                    mid_l = l1;
                    dn_l  = l2;
                }
                if (lane31) {
                    up_r  = rr0;
                    mid_r = rr1;
                    dn_r  = rr2;
                }

                if (write_cell) {
                    const unsigned int neighbors =
                        (up_l + r0 + up_r) +
                        (mid_l + mid_r) +
                        (dn_l + r2 + dn_r);

                    q[0] = apply_life_rule(neighbors, r1);
                }
            }

            // Advance the rolling window when another valid row remains in this strip.
            if ((i + 1) < kRowsPerStrip && (i + 1) < valid_rows) {
                p += stride;
                q += stride;

                r0 = r1;
                r1 = r2;
                r2 = static_cast<unsigned int>(p[stride2]);

                if (need_left) {
                    lp += stride;
                    l0  = l1;
                    l1  = l2;
                    l2  = static_cast<unsigned int>(lp[stride2]);
                }
                if (need_right) {
                    rp  += stride;
                    rr0 = rr1;
                    rr1 = rr2;
                    rr2 = static_cast<unsigned int>(rp[stride2]);
                }
            }
        }
    }
}

__global__ void life_boundary_kernel(const cell_t* __restrict__ in,
                                     cell_t* __restrict__ out,
                                     int n) {
    // One thread per boundary cell. This is O(n), so simplicity is preferred over further tuning.
    const int t            = static_cast<int>(blockIdx.x) * kBoundaryThreads + static_cast<int>(threadIdx.x);
    const int border_count = (n << 2) - 4;
    if (t >= border_count) {
        return;
    }

    const int last            = n - 1;
    const int two_n           = n << 1;
    const int three_n_minus_2 = two_n + n - 2;

    int x, y;
    if (t < n) {
        x = t;
        y = 0;
    } else if (t < two_n) {
        x = t - n;
        y = last;
    } else if (t < three_n_minus_2) {
        x = 0;
        y = t - two_n + 1;   // Left column, excluding corners.
    } else {
        x = last;
        y = t - three_n_minus_2 + 1; // Right column, excluding corners.
    }

    const size_t pitch = static_cast<size_t>(n);
    const size_t idx   = static_cast<size_t>(y) * pitch + static_cast<size_t>(x);
    const int    stride = n;

    const cell_t* p = in + idx;

    const bool has_left  = (x > 0);
    const bool has_right = (x < last);
    const bool has_up    = (y > 0);
    const bool has_down  = (y < last);

    unsigned int neighbors = 0;

    if (has_up) {
        const cell_t* pu = p - stride;
        if (has_left)  neighbors += static_cast<unsigned int>(pu[-1]);
        neighbors += static_cast<unsigned int>(pu[0]);
        if (has_right) neighbors += static_cast<unsigned int>(pu[1]);
    }

    if (has_left)  neighbors += static_cast<unsigned int>(p[-1]);
    if (has_right) neighbors += static_cast<unsigned int>(p[1]);

    if (has_down) {
        const cell_t* pd = p + stride;
        if (has_left)  neighbors += static_cast<unsigned int>(pd[-1]);
        neighbors += static_cast<unsigned int>(pd[0]);
        if (has_right) neighbors += static_cast<unsigned int>(pd[1]);
    }

    out[idx] = apply_life_rule(neighbors, static_cast<unsigned int>(p[0]));
}

} // anonymous namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Standard double-buffered semantics are assumed: input and output must not alias.
    // The caller owns all synchronization; this function intentionally performs no sync
    // and does not inject host-side launch/error checks into the hot path.

    const cell_t* in  = reinterpret_cast<const cell_t*>(input);
    cell_t*       out = reinterpret_cast<cell_t*>(output);

    const int x_blocks = grid_dimensions / kInteriorThreads;

    // The interior kernel grid-strides in y, so grid.y is capped to avoid excessive block count
    // on very large grids while still keeping plenty of concurrency on A100/H100 class devices.
    const int total_strips = (grid_dimensions - 2 + kRowsPerStrip - 1) / kRowsPerStrip;
    const int y_blocks     = (total_strips < kMaxGridY) ? total_strips : kMaxGridY;

    life_interior_strip_kernel<<<dim3(static_cast<unsigned>(x_blocks), static_cast<unsigned>(y_blocks), 1),
                                 dim3(static_cast<unsigned>(kInteriorThreads), 1, 1)>>>(in, out, grid_dimensions);

    const int border_count   = (grid_dimensions << 2) - 4;
    const int boundary_blocks = (border_count + kBoundaryThreads - 1) / kBoundaryThreads;

    life_boundary_kernel<<<boundary_blocks, kBoundaryThreads>>>(in, out, grid_dimensions);
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
