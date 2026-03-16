#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// Bit-packed Conway step for modern NVIDIA GPUs.
//
// Design choices:
// - One CUDA thread updates one 64-bit word, i.e. 64 cells at once.
// - Horizontal neighbor words are exchanged with warp shuffles instead of reloading
//   them from global memory. Only subgroup-edge lanes do extra global loads.
// - Per-cell neighbor counts are accumulated with a carry-save / full-adder tree:
//       top    = NW + N + NE
//       middle = W  + E
//       bottom = SW + S + SE
//   All of this happens bitwise across 64 cells in parallel.
// - Because the board dimensions are powers of two, the row width in words is also
//   a power of two. We therefore use two exact x-tilings:
//       * 32x4 blocks for rows with >= 32 words
//       * 16x8 blocks for the smallest legal case (16 words/row)
//   Both specializations are 128-thread CTAs, which is a good scheduling granularity
//   on A100/H100-class GPUs.
// - The kernel grid-strides in Y so the host can cap grid.y to an occupancy-shaped
//   value instead of launching one CTA per row-tile on extremely large boards.

using u64 = std::uint64_t;

constexpr int kThreadsPerBlock = 128;
constexpr int kBlockX32 = 32;
constexpr int kBlockY32 = 4;
constexpr int kBlockX16 = 16;
constexpr int kBlockY16 = 8;
constexpr int kMaxGridY = 65535;
constexpr int kResidentWaves = 2;

static_assert(kBlockX32 * kBlockY32 == kThreadsPerBlock,
              "32-wide specialization must remain a 128-thread CTA.");
static_assert(kBlockX16 * kBlockY16 == kThreadsPerBlock,
              "16-wide specialization must remain a 128-thread CTA.");

// Bit 0 is the first cell in the 64-cell run, so shifting left aligns the x-1
// neighbor with the current bit position, while shifting right aligns x+1.
__device__ __forceinline__ u64 align_west_neighbor(u64 center, u64 left_word) {
    return (center << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 align_east_neighbor(u64 center, u64 right_word) {
    return (center >> 1) | (right_word << 63);
}

// 3:2 compressor / full adder on 64 independent bit lanes.
// For three one-bit inputs a,b,c at every bit position:
//   sum   = a xor b xor c
//   carry = majority(a,b,c)
// so a + b + c = sum + 2*carry.
__device__ __forceinline__ void full_adder(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

template <int BLOCK_X, int BLOCK_Y>
__launch_bounds__(kThreadsPerBlock)
__global__ void game_of_life_kernel(const u64* __restrict__ input,
                                    u64* __restrict__ output,
                                    int words_per_row,
                                    int rows) {
    static_assert(BLOCK_X == 16 || BLOCK_X == 32,
                  "BLOCK_X must match the shuffle subgroup width.");
    static_assert(BLOCK_X * BLOCK_Y == kThreadsPerBlock,
                  "This kernel is tuned for 128-thread CTAs.");

    constexpr unsigned int kFullMask = 0xFFFFFFFFu;
    constexpr int kShuffleWidth = BLOCK_X;

    // X is always valid because run_game_of_life only launches exact x-tilings:
    // words_per_row is a power of two, and we choose BLOCK_X that divides it.
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + static_cast<int>(threadIdx.x);
    const bool first_word = (x == 0);
    const bool last_word  = (x == words_per_row - 1);

    // BLOCK_X is also the shuffle width:
    //   - BLOCK_X == 32: one row-segment per warp
    //   - BLOCK_X == 16: two independent row-segments per warp
    int y = static_cast<int>(blockIdx.y) * BLOCK_Y + static_cast<int>(threadIdx.y);
    const int y_stride = static_cast<int>(gridDim.y) * BLOCK_Y;

    const std::size_t stride = static_cast<std::size_t>(words_per_row);
    const std::size_t x_index = static_cast<std::size_t>(x);
    std::size_t row_base = static_cast<std::size_t>(y) * stride;
    const std::size_t row_base_stride = static_cast<std::size_t>(y_stride) * stride;

    for (;;) {
        const bool row_valid = (y < rows);

        // For the 16-wide specialization a warp contains two independent row-segments,
        // so we keep the whole warp alive until every subgroup has exhausted its rows.
        if (!__any_sync(kFullMask, row_valid)) {
            break;
        }

        const bool north_row_valid = row_valid && (y > 0);
        const bool south_row_valid = row_valid && ((y + 1) < rows);
        const std::size_t idx = row_base + x_index;

        // Load the three vertically aligned words (north/current/south).
        u64 north = 0ull;
        u64 center = 0ull;
        u64 south = 0ull;

        if (row_valid) {
            center = input[idx];
            if (north_row_valid) north = input[idx - stride];
            if (south_row_valid) south = input[idx + stride];
        }

        // Interior left/right words come from warp shuffles. Only subgroup-edge lanes
        // need one extra global load to cross a 16- or 32-word segment boundary.
        u64 north_left  = __shfl_up_sync(kFullMask, north, 1, kShuffleWidth);
        u64 center_left = __shfl_up_sync(kFullMask, center, 1, kShuffleWidth);
        u64 south_left  = __shfl_up_sync(kFullMask, south, 1, kShuffleWidth);

        u64 north_right  = __shfl_down_sync(kFullMask, north, 1, kShuffleWidth);
        u64 center_right = __shfl_down_sync(kFullMask, center, 1, kShuffleWidth);
        u64 south_right  = __shfl_down_sync(kFullMask, south, 1, kShuffleWidth);

        if (threadIdx.x == 0u) {
            if (row_valid && !first_word) {
                const std::size_t left_idx = idx - 1u;
                center_left = input[left_idx];
                north_left  = north_row_valid ? input[left_idx - stride] : 0ull;
                south_left  = south_row_valid ? input[left_idx + stride] : 0ull;
            } else {
                north_left = center_left = south_left = 0ull;
            }
        }

        if (threadIdx.x == static_cast<unsigned int>(BLOCK_X - 1)) {
            if (row_valid && !last_word) {
                const std::size_t right_idx = idx + 1u;
                center_right = input[right_idx];
                north_right  = north_row_valid ? input[right_idx - stride] : 0ull;
                south_right  = south_row_valid ? input[right_idx + stride] : 0ull;
            } else {
                north_right = center_right = south_right = 0ull;
            }
        }

        // Align horizontal neighbors for the current word's 64 cells.
        const u64 west = align_west_neighbor(center, center_left);
        const u64 east = align_east_neighbor(center, center_right);

        // Add north triplet, south triplet, and middle pair with full-adder logic.
        u64 top_sum, top_carry;
        full_adder(align_west_neighbor(north, north_left),
                   north,
                   align_east_neighbor(north, north_right),
                   top_sum,
                   top_carry);

        u64 bottom_sum, bottom_carry;
        full_adder(align_west_neighbor(south, south_left),
                   south,
                   align_east_neighbor(south, south_right),
                   bottom_sum,
                   bottom_carry);

        const u64 middle_sum   = west ^ east;
        const u64 middle_carry = west & east;

        // Count bit-planes:
        //   count = bit0 + 2*bit1 + 4*bit2 + 8*bit3
        // We only need bit0/bit1/bit2. bit3 can be ignored because count==8 also has
        // bit1==0, so it can never satisfy the Life predicates for 2 or 3 neighbors.
        u64 bit0, ones_carry;
        full_adder(top_sum, bottom_sum, middle_sum, bit0, ones_carry);

        u64 twos_sum, twos_carry;
        full_adder(top_carry, bottom_carry, middle_carry, twos_sum, twos_carry);

        const u64 bit1 = twos_sum ^ ones_carry;
        const u64 bit2 = twos_carry ^ (twos_sum & ones_carry);

        // Game of Life rule:
        //   next = (count == 3) | (alive & (count == 2))
        //        = bit1 & ~bit2 & (bit0 | alive)
        const u64 next = (bit1 & ~bit2) & (bit0 | center);

        if (row_valid) {
            output[idx] = next;
        }

        y += y_stride;
        row_base += row_base_stride;
    }
}

// Host-side launch shaping is not measured, so we use occupancy to cap grid.y.
// This keeps enough CTAs in flight while avoiding a pathological number of tiny
// CTAs for very tall boards. The kernel itself grid-strides over the remaining rows.
template <int BLOCK_X, int BLOCK_Y>
inline int choose_launch_grid_y(int grid_x, int total_tile_rows) {
    int device = 0;
    int sm_count = 1;
    int blocks_per_sm = 1;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        game_of_life_kernel<BLOCK_X, BLOCK_Y>,
        kThreadsPerBlock,
        0);

    if (sm_count < 1) sm_count = 1;
    if (blocks_per_sm < 1) blocks_per_sm = 1;

    int grid_y = (sm_count * blocks_per_sm * kResidentWaves + grid_x - 1) / grid_x;
    if (grid_y < 1) grid_y = 1;
    if (grid_y > total_tile_rows) grid_y = total_tile_rows;
    if (grid_y > kMaxGridY) grid_y = kMaxGridY;

    return grid_y;
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // input/output are assumed to be distinct device buffers (double buffering),
    // which allows the kernel to use __restrict__ safely.
    const int words_per_row = grid_dimensions >> 6;

    // Smallest legal board: 1024x1024 -> 16 words per row.
    // Use 16-wide row segments so no half-warps are wasted.
    if (words_per_row == kBlockX16) {
        const int grid_x = 1;
        const int total_tile_rows = grid_dimensions >> 3;  // exact: BLOCK_Y == 8
        const int grid_y = choose_launch_grid_y<kBlockX16, kBlockY16>(grid_x, total_tile_rows);

        game_of_life_kernel<kBlockX16, kBlockY16>
            <<<dim3(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(grid_y),
                    1u),
               dim3(static_cast<unsigned int>(kBlockX16),
                    static_cast<unsigned int>(kBlockY16),
                    1u)>>>(input, output, words_per_row, grid_dimensions);
        return;
    }

    // All larger legal boards have at least 32 words per row, and because the width
    // is a power of two the row width is an exact multiple of 32.
    const int grid_x = words_per_row >> 5;        // exact: BLOCK_X == 32
    const int total_tile_rows = grid_dimensions >> 2;  // exact: BLOCK_Y == 4
    const int grid_y = choose_launch_grid_y<kBlockX32, kBlockY32>(grid_x, total_tile_rows);

    game_of_life_kernel<kBlockX32, kBlockY32>
        <<<dim3(static_cast<unsigned int>(grid_x),
                static_cast<unsigned int>(grid_y),
                1u),
           dim3(static_cast<unsigned int>(kBlockX32),
                static_cast<unsigned int>(kBlockY32),
                1u)>>>(input, output, words_per_row, grid_dimensions);
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
