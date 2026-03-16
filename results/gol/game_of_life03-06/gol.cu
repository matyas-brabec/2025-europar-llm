#include <cuda_runtime.h>
#include <cstdint>

namespace {

// One warp owns one 32-word horizontal segment from a single board row.
// This is intentional: it lets us reuse west/east neighbor words with warp
// shuffles instead of shared memory or redundant global loads.
using word_t = std::uint64_t;
using u32 = std::uint32_t;

constexpr u32 kWarpWidth     = 32u;
constexpr u32 kBlockX        = kWarpWidth;
constexpr u32 kBlockY        = 8u;
constexpr u32 kBlockThreads  = kBlockX * kBlockY;

// Roughly one million threads in flight at most (4096 * 256). Larger boards are
// covered with grid-stride loops. This is a good fixed launch target for A100/H100
// class GPUs without paying the overhead of launching a block per logical tile.
constexpr u32 kTargetBlocks  = 4096u;
constexpr u32 kMaxGridXTiles = 64u;

// Bit-sliced carry-save adder:
// for every bit position independently, a + b + c = sum + 2 * carry.
__device__ __forceinline__ void csa(
    const word_t a,
    const word_t b,
    const word_t c,
    word_t& carry,
    word_t& sum)
{
    const word_t u = a ^ b;
    carry = (a & b) | (u & c);
    sum   = u ^ c;
}

// One kernel step of Conway's Game of Life on a bit-packed board.
//
// Representation:
//   - Each 64-bit word stores 64 consecutive cells from one row.
//   - Bit value 1 means alive, 0 means dead.
//   - Bit 0 is treated as the leftmost cell inside the word, matching the prompt:
//       * left  neighbors => (word << 1) | (west_word >> 63)
//       * right neighbors => (word >> 1) | (east_word << 63)
//
// Mapping:
//   - One CUDA thread computes exactly one 64-bit output word.
//   - Each warp processes 32 consecutive words from a single row segment.
//   - West/east neighboring words are obtained with warp shuffles when they stay
//     inside the warp tile; only warp/tile boundaries fall back to direct global
//     loads. This captures the useful reuse without shared memory.
//
// Boundaries:
//   - Everything outside the board is dead, so border rows/columns inject zero words.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(
    const word_t* __restrict__ input,
    word_t* __restrict__ output,
    const u32 words_per_row,
    const u32 grid_dimensions)
{
    const u32 lane      = static_cast<u32>(threadIdx.x);
    const u32 row0      = static_cast<u32>(blockIdx.y) * kBlockY + static_cast<u32>(threadIdx.y);
    const u32 col0      = static_cast<u32>(blockIdx.x) * kBlockX + lane;
    const u32 row_stride = static_cast<u32>(gridDim.y) * kBlockY;
    const u32 col_stride = static_cast<u32>(gridDim.x) * kBlockX;

    for (u32 row = row0; row < grid_dimensions; row += row_stride) {
        const bool has_north = (row != 0u);
        const bool has_south = (row + 1u < grid_dimensions);

        const std::uint64_t row_base =
            static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(words_per_row);

        const word_t* const row_ptr       = input  + row_base;
        const word_t* const north_row_ptr = row_ptr - (has_north ? words_per_row : 0u);
        const word_t* const south_row_ptr = row_ptr + (has_south ? words_per_row : 0u);
        word_t* const       out_ptr       = output + row_base;

        for (u32 col = col0; col < words_per_row; col += col_stride) {
            const word_t self  = row_ptr[col];
            const word_t north = has_north ? north_row_ptr[col] : 0ull;
            const word_t south = has_south ? south_row_ptr[col] : 0ull;

            // This is full-mask almost always; __activemask() keeps the code correct
            // for the only partial-x case here (1024x1024 boards => 16 words/row).
            const unsigned mask = __activemask();

            // Fast path: neighboring words already loaded by adjacent lanes.
            word_t west_word       = __shfl_up_sync(mask,   self,  1);
            word_t east_word       = __shfl_down_sync(mask, self,  1);
            word_t north_west_word = __shfl_up_sync(mask,   north, 1);
            word_t north_east_word = __shfl_down_sync(mask, north, 1);
            word_t south_west_word = __shfl_up_sync(mask,   south, 1);
            word_t south_east_word = __shfl_down_sync(mask, south, 1);

            const bool has_global_west = (col != 0u);
            const bool has_global_east = (col + 1u < words_per_row);

            const bool has_warp_west =
                (lane != 0u) &&
                ((mask & (1u << (lane - 1u))) != 0u);

            const bool has_warp_east =
                (lane + 1u < kWarpWidth) &&
                ((mask & (1u << (lane + 1u))) != 0u);

            // Warp/tile boundaries: fetch the needed three words on the left/right
            // (north/current/south), or inject zeros at the true board boundary.
            if (!has_warp_west) {
                if (has_global_west) {
                    west_word       = row_ptr[col - 1u];
                    north_west_word = has_north ? north_row_ptr[col - 1u] : 0ull;
                    south_west_word = has_south ? south_row_ptr[col - 1u] : 0ull;
                } else {
                    west_word = north_west_word = south_west_word = 0ull;
                }
            }

            if (!has_warp_east) {
                if (has_global_east) {
                    east_word       = row_ptr[col + 1u];
                    north_east_word = has_north ? north_row_ptr[col + 1u] : 0ull;
                    south_east_word = has_south ? south_row_ptr[col + 1u] : 0ull;
                } else {
                    east_word = north_east_word = south_east_word = 0ull;
                }
            }

            // Align all eight neighbor directions to the current cell positions.
            const word_t north_left  = (north << 1) | (north_west_word >> 63);
            const word_t north_right = (north >> 1) | (north_east_word << 63);
            const word_t west        = (self  << 1) | (west_word       >> 63);
            const word_t east        = (self  >> 1) | (east_word       << 63);
            const word_t south_left  = (south << 1) | (south_west_word >> 63);
            const word_t south_right = (south >> 1) | (south_east_word << 63);

            // Count the 8 neighbors for all 64 cells in parallel with a bit-sliced
            // adder tree. We only need:
            //   - ones: low count bit
            //   - twos: second count bit
            //   - ge4 : whether the count is at least 4
            //
            // The Life rule:
            //   next = (count == 3) | (self & (count == 2))
            //        = twos & ~ge4 & (ones | self)
            word_t h0, l0;
            csa(north_left, north, north_right, h0, l0);

            word_t h1, l1;
            csa(south_left, south, south_right, h1, l1);

            const word_t h2 = west & east;
            const word_t l2 = west ^ east;

            word_t h3, ones;
            csa(l0, l1, l2, h3, ones);

            word_t h4, twos_a;
            csa(h0, h1, h2, h4, twos_a);

            // twos_a and h3 are both weight-2 planes:
            //   twos = low 2s bit
            //   twos_a & h3 = carry into the 4s plane
            const word_t twos = twos_a ^ h3;
            const word_t ge4  = h4 | (twos_a & h3);

            out_ptr[col] = twos & ~ge4 & (ones | self);
        }
    }
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Assumptions from the problem statement:
    //   - square board
    //   - grid_dimensions is a power of two
    //   - grid_dimensions > 512
    // Therefore:
    //   - words_per_row = grid_dimensions / 64 is exact
    //   - there are no partial words at the end of a row
    const std::uint32_t dim = static_cast<std::uint32_t>(grid_dimensions);
    const std::uint32_t words_per_row = dim >> 6;

    // Launch a capped number of logical tiles and let the kernel grid-stride over
    // larger boards. This keeps the launch geometry in a sweet spot for modern
    // data-center GPUs while still handling very large boards.
    const std::uint32_t tiles_x = (words_per_row + kBlockX - 1u) / kBlockX;
    const std::uint32_t tiles_y = (dim          + kBlockY - 1u) / kBlockY;

    const std::uint32_t grid_x = (tiles_x < kMaxGridXTiles) ? tiles_x : kMaxGridXTiles;
    const std::uint32_t grid_y_cap = kTargetBlocks / grid_x;
    const std::uint32_t grid_y = (tiles_y < grid_y_cap) ? tiles_y : grid_y_cap;

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(grid_x, grid_y);

    // The caller owns synchronization/stream semantics; this function only launches
    // one simulation step.
    game_of_life_kernel<<<grid, block>>>(input, output, words_per_row, dim);
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
