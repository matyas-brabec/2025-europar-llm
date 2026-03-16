#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Optimized single-step Conway's Game of Life for a square, bit-packed board.
//
// Key implementation choices:
//
// 1. One CUDA thread owns one 64-bit word (64 cells), so every output word is
//    written exactly once and no atomics are needed.
//
// 2. One warp owns one horizontal tile of 32 consecutive words from a single row.
//    That mapping lets adjacent threads exchange left/right words with warp shuffles
//    instead of shared memory. Only warp-edge lanes fall back to an extra global load.
//
// 3. Outside-the-board cells are treated as dead by zero-filling missing neighbor
//    words at top/bottom/left/right boundaries.
//
// 4. Neighbor counts are computed bit-slicewise with carry-save adders, so each thread
//    updates all 64 cells in its word in parallel using integer logic only.
//
// 5. A 1D grid plus a grid-stride loop avoids the grid.y limit for very large boards
//    and avoids launching millions of tiny blocks on large inputs.

namespace gol_detail {

// Use a 64-bit unsigned type that is accepted by warp shuffle intrinsics across host ABIs.
using word_t = unsigned long long;
static_assert(sizeof(word_t) == sizeof(std::uint64_t), "Expected 64-bit words.");

constexpr int kWarpSize        = 32;
constexpr int kWarpShift       = 5;   // log2(32)
constexpr int kWordShift       = 6;   // log2(64)
constexpr int kWarpsPerBlock   = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kMaxBlocks = 2048;  // Enough queued work for A100/H100-class GPUs.

static_assert(kThreadsPerBlock <= 1024, "Block size exceeds CUDA limits.");
static_assert((kThreadsPerBlock % kWarpSize) == 0, "Block must contain whole warps.");

__device__ __forceinline__ word_t load_word(const std::uint64_t* __restrict__ ptr, std::size_t idx) {
    return static_cast<word_t>(ptr[idx]);
}

// After alignment:
//   align_west(center, left): bit i == original cell (i-1)
//   align_east(center, right): bit i == original cell (i+1)
// For bit 0 / bit 63, the carry-in comes from the adjacent word or zero at the board edge.
__device__ __forceinline__ word_t align_west(word_t center, word_t left) {
    return (center << 1) | (left >> 63);
}

__device__ __forceinline__ word_t align_east(word_t center, word_t right) {
    return (center >> 1) | (right << 63);
}

// Carry-save adder for bit-sliced population counting:
//   a + b + c = lo + 2*hi
__device__ __forceinline__ void csa(word_t& hi, word_t& lo, word_t a, word_t b, word_t c) {
    const word_t u = a ^ b;
    hi = (a & b) | (u & c);
    lo = u ^ c;
}

// Exactly one of four one-bit bit-planes is set.
// Inputs are paired so we only need one "any" term and one "pair" term per pair.
__device__ __forceinline__ word_t exact_one_of_four(word_t a, word_t b, word_t c, word_t d) {
    const word_t ab_any = a | b;
    const word_t cd_any = c | d;
    return (ab_any ^ cd_any) & ~((a & b) | (c & d));
}

__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                              std::uint64_t* __restrict__ output,
                              int grid_dimensions,
                              std::uint64_t total_tiles,
                              int words_per_row,
                              int tiles_shift,
                              unsigned int tiles_mask) {
    const unsigned int lane          = threadIdx.x & (kWarpSize - 1);
    const unsigned int warp_in_block = threadIdx.x >> kWarpShift;

    std::uint64_t warp_idx =
        static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(kWarpsPerBlock) +
        static_cast<std::uint64_t>(warp_in_block);

    const std::uint64_t warp_stride =
        static_cast<std::uint64_t>(gridDim.x) * static_cast<std::uint64_t>(kWarpsPerBlock);

    for (; warp_idx < total_tiles; warp_idx += warp_stride) {
        // One warp == one 32-word tile from one row.
        const int row    = static_cast<int>(warp_idx >> tiles_shift);
        const int tile_x = static_cast<int>(warp_idx & static_cast<std::uint64_t>(tiles_mask));
        const int word_x = (tile_x << kWarpShift) + static_cast<int>(lane);

        // The only possible partial tile is the smallest legal board: 1024x1024,
        // which has 16 words per row. Let the upper half-warp peel off here;
        // __activemask() below then shrinks naturally to the participating lanes.
        if (word_x >= words_per_row) {
            continue;
        }

        const std::size_t pitch = static_cast<std::size_t>(words_per_row);
        const std::size_t idx   = static_cast<std::size_t>(row) * pitch + static_cast<std::size_t>(word_x);

        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < grid_dimensions);
        const bool has_left  = (word_x > 0);
        const bool has_right = (word_x + 1 < words_per_row);

        // Each thread directly loads only the vertically aligned words.
        // Left/right words are fetched via shuffles whenever the neighbor is in the same warp tile.
        const word_t center = load_word(input, idx);
        const word_t north  = has_up   ? load_word(input, idx - pitch) : 0ULL;
        const word_t south  = has_down ? load_word(input, idx + pitch) : 0ULL;

        const unsigned int active = __activemask();

        // ---------------- Top row neighbors: NW, N, NE ----------------
        word_t tmp = __shfl_up_sync(active, north, 1);
        const word_t north_left_word =
            (lane != 0)
                ? tmp
                : ((has_up && has_left) ? load_word(input, idx - pitch - 1) : 0ULL);

        tmp = __shfl_down_sync(active, north, 1);
        const word_t north_right_word =
            (lane != kWarpSize - 1 && has_right)
                ? tmp
                : ((has_up && has_right) ? load_word(input, idx - pitch + 1) : 0ULL);

        word_t top1, top0;
        csa(top1, top0,
            align_west(north, north_left_word),
            north,
            align_east(north, north_right_word));

        // ---------------- Middle row neighbors: W, E ----------------
        // The center cell itself must not be counted.
        tmp = __shfl_up_sync(active, center, 1);
        const word_t left_word =
            (lane != 0)
                ? tmp
                : (has_left ? load_word(input, idx - 1) : 0ULL);

        tmp = __shfl_down_sync(active, center, 1);
        const word_t right_word =
            (lane != kWarpSize - 1 && has_right)
                ? tmp
                : (has_right ? load_word(input, idx + 1) : 0ULL);

        word_t mid1, mid0;
        csa(mid1, mid0,
            align_west(center, left_word),
            align_east(center, right_word),
            0ULL);

        // ---------------- Bottom row neighbors: SW, S, SE ----------------
        tmp = __shfl_up_sync(active, south, 1);
        const word_t south_left_word =
            (lane != 0)
                ? tmp
                : ((has_down && has_left) ? load_word(input, idx + pitch - 1) : 0ULL);

        tmp = __shfl_down_sync(active, south, 1);
        const word_t south_right_word =
            (lane != kWarpSize - 1 && has_right)
                ? tmp
                : ((has_down && has_right) ? load_word(input, idx + pitch + 1) : 0ULL);

        word_t bot1, bot0;
        csa(bot1, bot0,
            align_west(south, south_left_word),
            south,
            align_east(south, south_right_word));

        // Bit-sliced reduction:
        //
        //   top = top0 + 2*top1
        //   mid = mid0 + 2*mid1
        //   bot = bot0 + 2*bot1
        //
        // First add the low bits:
        //   top0 + mid0 + bot0 = sum0 + 2*carry0
        //
        // Therefore:
        //   count = sum0 + 2*(carry0 + top1 + mid1 + bot1)
        //
        // count is 2 or 3 iff exactly one of {carry0, top1, mid1, bot1} is set.
        // sum0 then distinguishes:
        //   sum0 = 0 -> count == 2
        //   sum0 = 1 -> count == 3
        word_t carry0, sum0;
        csa(carry0, sum0, top0, mid0, bot0);

        const word_t q_eq1 = exact_one_of_four(carry0, top1, mid1, bot1);
        const word_t eq3   = q_eq1 & sum0;
        const word_t eq2   = q_eq1 & ~sum0;

        // Conway rule:
        //   next = (count == 3) || (alive && count == 2)
        const word_t next = eq3 | (center & eq2);
        output[idx] = static_cast<std::uint64_t>(next);
    }
}

}  // namespace gol_detail

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is a power of two and > 512.
    // Each 64-bit word stores 64 consecutive cells from a row.
    const int words_per_row = grid_dimensions >> gol_detail::kWordShift;

    // One warp covers 32 words horizontally, so the number of tiles per row is:
    //   ceil(words_per_row / 32)
    // Under the problem constraints, this is always a power of two:
    //   16 -> 1, 32 -> 1, 64 -> 2, 128 -> 4, ...
    const int tiles_per_row = (words_per_row + (gol_detail::kWarpSize - 1)) >> gol_detail::kWarpShift;
    const unsigned int tiles_mask = static_cast<unsigned int>(tiles_per_row - 1);

    // Because tiles_per_row is a power of two, row/tile decomposition in the kernel
    // can use a shift and a mask instead of division and modulo.
    int tiles_shift = 0;
    for (int t = tiles_per_row; t > 1; t >>= 1) {
        ++tiles_shift;
    }

    const std::uint64_t total_tiles =
        static_cast<std::uint64_t>(grid_dimensions) * static_cast<std::uint64_t>(tiles_per_row);

    // Cap the launch size and let the kernel grid-stride over tiles.
    // This keeps enough queued work to saturate modern data-center GPUs while
    // avoiding the overhead of launching millions of tiny blocks on very large boards.
    const std::uint64_t blocks_needed =
        (total_tiles + static_cast<std::uint64_t>(gol_detail::kWarpsPerBlock) - 1ULL) /
        static_cast<std::uint64_t>(gol_detail::kWarpsPerBlock);

    const unsigned int grid_x =
        static_cast<unsigned int>(blocks_needed < gol_detail::kMaxBlocks
                                      ? blocks_needed
                                      : static_cast<std::uint64_t>(gol_detail::kMaxBlocks));

    // No explicit synchronization or error polling here; the caller owns stream/error handling.
    gol_detail::game_of_life_step_kernel<<<grid_x, gol_detail::kThreadsPerBlock>>>(
        input,
        output,
        grid_dimensions,
        total_tiles,
        words_per_row,
        tiles_shift,
        tiles_mask);
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
