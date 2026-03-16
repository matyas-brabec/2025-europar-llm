#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// CUDA device-side bool storage is 1 byte. Reinterpreting the buffers as bytes avoids
// repeated bool<->integer conversions in the hot path and still preserves 0/1 semantics.
static_assert(sizeof(bool) == 1, "This implementation assumes 1-byte bool storage.");

namespace {

using cell_t = unsigned char;
using u32 = std::uint32_t;

// One warp owns one 32-cell span along X. This lets us fetch left/right neighbors with
// warp shuffles instead of shared memory. Each warp then rolls through several consecutive
// rows, keeping the vertical 3-row stencil window entirely in registers.
//
// The grid size is guaranteed to be a power of two and > 512, so all chosen strip heights
// divide the problem exactly. That means:
//   - no partially active warps/blocks,
//   - no bounds checks on the output domain itself,
//   - only true grid-edge halo handling remains.
constexpr int kWarpWidth      = 32;
constexpr int kWarpsPerBlock  = 4;
constexpr u32 kFullWarpMask   = 0xFFFFFFFFu;

template <int TILE_ROWS>
__global__ __launch_bounds__(kWarpWidth * kWarpsPerBlock)
void game_of_life_step_kernel(const cell_t* __restrict__ input,
                              cell_t* __restrict__ output,
                              int dim) {
    constexpr int kRowsPerBlock = TILE_ROWS * kWarpsPerBlock;

    const int lane = threadIdx.x;  // 0..31
    const int warp = threadIdx.y;  // 0..kWarpsPerBlock-1

    const bool is_lane0     = (lane == 0);
    const bool is_lane_last = (lane == (kWarpWidth - 1));

    const int x  = (static_cast<int>(blockIdx.x) << 5) + lane;
    const int y0 = static_cast<int>(blockIdx.y) * kRowsPerBlock + warp * TILE_ROWS;

    const bool top_segment    = (y0 == 0);
    const bool bottom_segment = (y0 + TILE_ROWS == dim);

    // Block-level X-edge information. Only lane 0 / lane 31 ever need these halos.
    const bool has_left_halo  = (blockIdx.x != 0u);
    const bool has_right_halo = (blockIdx.x + 1u < gridDim.x);

    const std::size_t stride    = static_cast<std::size_t>(dim);
    const std::size_t start_idx = static_cast<std::size_t>(y0) * stride + static_cast<std::size_t>(x);

    // Vertical rolling window for the center column of this lane.
    u32 above = top_segment ? 0u : static_cast<u32>(input[start_idx - stride]);
    u32 curr  = static_cast<u32>(input[start_idx]);
    // Valid for all launched TILE_ROWS (8/16/32), because every strip starts at y0 <= dim - TILE_ROWS.
    u32 below = static_cast<u32>(input[start_idx + stride]);

    // Rolling halo state for the two boundary lanes of the warp.
    u32 above_w = 0u, curr_w = 0u, below_w = 0u;
    if (is_lane0 && has_left_halo) {
        const std::size_t left_idx = start_idx - 1u;
        above_w = top_segment ? 0u : static_cast<u32>(input[left_idx - stride]);
        curr_w  = static_cast<u32>(input[left_idx]);
        below_w = static_cast<u32>(input[left_idx + stride]);
    }

    u32 above_e = 0u, curr_e = 0u, below_e = 0u;
    if (is_lane_last && has_right_halo) {
        const std::size_t right_idx = start_idx + 1u;
        above_e = top_segment ? 0u : static_cast<u32>(input[right_idx - stride]);
        curr_e  = static_cast<u32>(input[right_idx]);
        below_e = static_cast<u32>(input[right_idx + stride]);
    }

    std::size_t out_idx  = start_idx;
    std::size_t next_idx = start_idx + (stride << 1);  // row y0 + 2

#pragma unroll
    for (int row = 0; row < TILE_ROWS; ++row) {
        // Horizontal neighbors inside the warp come from shuffles.
        u32 aw = __shfl_up_sync(kFullWarpMask, above, 1);
        u32 cw = __shfl_up_sync(kFullWarpMask, curr,  1);
        u32 bw = __shfl_up_sync(kFullWarpMask, below, 1);

        u32 ae = __shfl_down_sync(kFullWarpMask, above, 1);
        u32 ce = __shfl_down_sync(kFullWarpMask, curr,  1);
        u32 be = __shfl_down_sync(kFullWarpMask, below, 1);

        // Replace undefined shuffle results at the two warp edges with real halo values.
        aw = is_lane0     ? above_w : aw;
        cw = is_lane0     ? curr_w  : cw;
        bw = is_lane0     ? below_w : bw;

        ae = is_lane_last ? above_e : ae;
        ce = is_lane_last ? curr_e  : ce;
        be = is_lane_last ? below_e : be;

        const u32 neighbors = aw + above + ae +
                              cw        + ce +
                              bw + below + be;

        // Conway rule:
        //   live if exactly 3 neighbors, or if currently alive and exactly 2 neighbors.
        output[out_idx] = static_cast<cell_t>((neighbors == 3u) | ((neighbors == 2u) & curr));

        if (row + 1 < TILE_ROWS) {
            out_idx += stride;

            // Roll the center-column window.
            above = curr;
            curr  = below;

            // Only the very last strip touching the bottom border needs zero-padding.
            const bool next_valid = !bottom_segment || (row + 2 < TILE_ROWS);
            below = next_valid ? static_cast<u32>(input[next_idx]) : 0u;

            // Roll the warp-edge halo state in lockstep.
            if (is_lane0) {
                above_w = curr_w;
                curr_w  = below_w;
                below_w = (has_left_halo && next_valid) ? static_cast<u32>(input[next_idx - 1u]) : 0u;
            }

            if (is_lane_last) {
                above_e = curr_e;
                curr_e  = below_e;
                below_e = (has_right_halo && next_valid) ? static_cast<u32>(input[next_idx + 1u]) : 0u;
            }

            next_idx += stride;
        }
    }
}

template <int TILE_ROWS>
inline void launch_game_of_life(const cell_t* input, cell_t* output, int dim) {
    constexpr int kRowsPerBlock = TILE_ROWS * kWarpsPerBlock;

    const dim3 block(kWarpWidth, kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned int>(dim / kWarpWidth),
                    static_cast<unsigned int>(dim / kRowsPerBlock));

    game_of_life_step_kernel<TILE_ROWS><<<grid, block>>>(input, output, dim);
}

}  // namespace

/// @FIXED
/// extern "C" void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // The caller guarantees power-of-two dimensions > 512.
    // No synchronization here; the caller explicitly handles it.
    const cell_t* in  = reinterpret_cast<const cell_t*>(input);
    cell_t* out       = reinterpret_cast<cell_t*>(output);

    // Strip-height selection balances two effects:
    //   1) larger strips reduce top/bottom halo overhead,
    //   2) smaller strips create more total warps on the smallest legal boards.
    //
    // For 1024^2, TILE_ROWS=8 keeps enough independent warps in flight.
    // For larger boards, taller strips reduce memory traffic.
    if (grid_dimensions <= 1024) {
        launch_game_of_life<8>(in, out, grid_dimensions);
    } else if (grid_dimensions <= 2048) {
        launch_game_of_life<16>(in, out, grid_dimensions);
    } else {
        launch_game_of_life<32>(in, out, grid_dimensions);
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
