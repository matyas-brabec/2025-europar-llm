#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Optimized one-step Conway's Game of Life for bit-packed rows on modern NVIDIA GPUs.
//
// Key design choices:
// - One thread computes one 64-bit output word, i.e. 64 cells at once.
// - Data is assumed to use the natural LSB-first packing:
//     bit i of input[row * words_per_row + word] is cell x = 64 * word + i, y = row.
// - The current row's left/right neighboring words are obtained with warp shuffles.
//   Only subwarp-boundary lanes perform extra global loads for cross-warp or row-edge words.
// - No shared memory is used; for this access pattern, warp shuffles already remove most
//   redundant horizontal traffic with less overhead.
// - The launcher exploits the guaranteed power-of-two dimensions so the kernel has no x/y
//   bounds checks in the hot path.
//
// The implementation assumes the usual double-buffered Life step:
// input and output must be distinct device buffers.

namespace {

using u64   = std::uint64_t;
using usize = std::size_t;

constexpr unsigned int kFullWarpMask         = 0xFFFFFFFFu;
constexpr unsigned int kMaxPowerOfTwoGridY   = 32768u;  // Largest power of two <= CUDA's grid.y limit.

// Hardware lane id is needed because when blockDim.x < 32, a warp spans multiple rows;
// threadIdx.x alone is not enough to identify shuffle lanes correctly.
__device__ __forceinline__ unsigned int lane_id() {
    unsigned int lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// LSB-first packing:
//   west neighbor of bit x is input(x-1), which aligns onto bit x by shifting left.
//   east neighbor of bit x is input(x+1), which aligns onto bit x by shifting right.
__device__ __forceinline__ u64 align_west(const u64 word, const u64 left_word) {
    return (word << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 align_east(const u64 word, const u64 right_word) {
    return (word >> 1) | (right_word << 63);
}

// Bit-sliced adders operating independently on all 64 bit positions in parallel.
__device__ __forceinline__ void half_adder(const u64 a, const u64 b, u64& sum, u64& carry) {
    sum   = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ void full_adder(const u64 a, const u64 b, const u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

template <int SUBWARP_WIDTH>
__global__ __launch_bounds__(256)
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              int grid_dimensions,
                              int words_per_row) {
    static_assert(SUBWARP_WIDTH == 8 || SUBWARP_WIDTH == 16 || SUBWARP_WIDTH == 32,
                  "SUBWARP_WIDTH must be 8, 16, or 32.");

    // The launcher chooses exact power-of-two tiling:
    // - x is covered exactly by blockDim.x * gridDim.x
    // - y uses an exact power-of-two grid-stride loop
    // Therefore every thread in every warp/subwarp is valid for every loop iteration.
    const int word_x     = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int start_row  = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    const int row_stride = static_cast<int>(gridDim.y)  * static_cast<int>(blockDim.y);

    const usize pitch      = static_cast<usize>(words_per_row);
    const usize word_index = static_cast<usize>(word_x);
    usize idx              = static_cast<usize>(start_row) * pitch + word_index;
    const usize idx_stride = static_cast<usize>(row_stride) * pitch;

    const int  last_row       = grid_dimensions - 1;
    const bool has_left_word  = (word_x != 0);
    const bool has_right_word = (word_x != (words_per_row - 1));

    // For blockDim.x < 32, a warp contains multiple independent row segments.
    // Shuffle width keeps traffic inside one row segment:
    //   width 16 -> two rows per warp
    //   width  8 -> four rows per warp
    //   width 32 -> the common case, one row segment per warp
    const unsigned int lane             = lane_id();
    const unsigned int lane_in_subwarp  = lane & static_cast<unsigned int>(SUBWARP_WIDTH - 1);
    const bool subwarp_left_edge        = (lane_in_subwarp == 0u);
    const bool subwarp_right_edge       = (lane_in_subwarp == static_cast<unsigned int>(SUBWARP_WIDTH - 1));

    for (int row = start_row; row < grid_dimensions; row += row_stride, idx += idx_stride) {
        const bool has_north = (row != 0);
        const bool has_south = (row != last_row);

        // Load the three vertically adjacent center words.
        const u64 north   = has_north ? input[idx - pitch] : 0ULL;
        const u64 current = input[idx];
        const u64 south   = has_south ? input[idx + pitch] : 0ULL;

        // Horizontal neighbors mostly come from shuffles. Only subwarp-boundary lanes
        // need explicit global loads for cross-warp or row-edge words.
        u64 left_north    = __shfl_up_sync(kFullWarpMask, north,   1, SUBWARP_WIDTH);
        u64 right_north   = __shfl_down_sync(kFullWarpMask, north, 1, SUBWARP_WIDTH);
        u64 left_current  = __shfl_up_sync(kFullWarpMask, current,   1, SUBWARP_WIDTH);
        u64 right_current = __shfl_down_sync(kFullWarpMask, current, 1, SUBWARP_WIDTH);
        u64 left_south    = __shfl_up_sync(kFullWarpMask, south,   1, SUBWARP_WIDTH);
        u64 right_south   = __shfl_down_sync(kFullWarpMask, south, 1, SUBWARP_WIDTH);

        if (subwarp_left_edge) {
            if (has_left_word) {
                left_current = input[idx - 1];
                left_north   = has_north ? input[idx - pitch - 1] : 0ULL;
                left_south   = has_south ? input[idx + pitch - 1] : 0ULL;
            } else {
                left_current = 0ULL;
                left_north   = 0ULL;
                left_south   = 0ULL;
            }
        }

        if (subwarp_right_edge) {
            if (has_right_word) {
                right_current = input[idx + 1];
                right_north   = has_north ? input[idx - pitch + 1] : 0ULL;
                right_south   = has_south ? input[idx + pitch + 1] : 0ULL;
            } else {
                right_current = 0ULL;
                right_north   = 0ULL;
                right_south   = 0ULL;
            }
        }

        // Build the 8 aligned neighbor bitboards.
        // Current row contributes only W/E; the center cell itself is excluded from the neighbor count.
        u64 north_low, north_high;
        {
            const u64 north_west = align_west(north, left_north);
            const u64 north_east = align_east(north, right_north);
            full_adder(north_west, north, north_east, north_low, north_high);
        }

        u64 middle_low, middle_high;
        {
            const u64 west = align_west(current, left_current);
            const u64 east = align_east(current, right_current);
            half_adder(west, east, middle_low, middle_high);
        }

        u64 south_low, south_high;
        {
            const u64 south_west = align_west(south, left_south);
            const u64 south_east = align_east(south, right_south);
            full_adder(south_west, south, south_east, south_low, south_high);
        }

        // Add the three per-row counts:
        //   row count low bits  -> neighbor count bit 0 and carry into the 2's bit
        //   row count high bits -> partial 2's bit and carry into the 4's bit
        u64 ones_bit, carry_to_twos;
        full_adder(north_low, middle_low, south_low, ones_bit, carry_to_twos);

        u64 twos_from_row_highs, carry_to_fours;
        full_adder(north_high, middle_high, south_high, twos_from_row_highs, carry_to_fours);

        const u64 twos_bit = twos_from_row_highs ^ carry_to_twos;

        // twos_bit == 1 exactly for neighbor counts {2, 3, 6, 7}.
        // carry_to_fours rejects the remaining >=4 cases among those, i.e. {6, 7}.
        // Then:
        //   count == 3 -> ones_bit == 1, so the cell becomes alive regardless of current state
        //   count == 2 -> ones_bit == 0, so the cell survives only if it is currently alive
        output[idx] = twos_bit & ~carry_to_fours & (ones_bit | current);
    }
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6;

    // Smallest measured grid is 1024x1024 -> 16 packed words per row.
    // For that case, using 128-thread blocks increases block count and spreads work across more SMs.
    // Larger grids use 256-thread blocks.
    const int target_threads_per_block = (words_per_row <= 16) ? 128 : 256;

    // block_x is always a power-of-two factor of words_per_row, so x is tiled exactly.
    const int block_x = (words_per_row < target_threads_per_block) ? words_per_row : target_threads_per_block;
    const int block_y = target_threads_per_block / block_x;

    const unsigned int grid_x = static_cast<unsigned int>(words_per_row / block_x);

    // grid.y is capped at 2^15 (not 65535) because 2^15 is the largest power of two allowed by CUDA's grid.y limit.
    // Since the height is also a power of two, the resulting y-stride still partitions the domain exactly,
    // which keeps every warp/subwarp full and lets the kernel avoid bounds checks.
    const unsigned int row_block_count = static_cast<unsigned int>(grid_dimensions / block_y);
    const unsigned int grid_y = (row_block_count > kMaxPowerOfTwoGridY) ? kMaxPowerOfTwoGridY : row_block_count;

    const dim3 block(static_cast<unsigned int>(block_x),
                     static_cast<unsigned int>(block_y),
                     1u);

    const dim3 grid(grid_x, grid_y, 1u);

    // Width-32 shuffles are used for all normal wide-row cases.
    // Width-16/8 are needed only when a warp carries multiple independent rows.
    switch (block_x) {
        case 8:
            game_of_life_step_kernel<8><<<grid, block>>>(input, output, grid_dimensions, words_per_row);
            break;
        case 16:
            game_of_life_step_kernel<16><<<grid, block>>>(input, output, grid_dimensions, words_per_row);
            break;
        default:
            game_of_life_step_kernel<32><<<grid, block>>>(input, output, grid_dimensions, words_per_row);
            break;
    }
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
