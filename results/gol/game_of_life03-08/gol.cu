#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

using u64 = std::uint64_t;

constexpr int kBlockSize = 256;
constexpr unsigned int kLaunchWaves = 4;

/*
  Bit-sliced three-input add primitives.

  For each bit position across the 64-bit word, xor3() returns the low bit of the
  population count of (a, b, c), and majority3() returns the high/carry bit.
  On modern NVIDIA GPUs, current CUDA compilers lower these boolean forms well
  (typically to LOP3-based code per 32-bit half).
*/
__device__ __forceinline__ u64 xor3(const u64 a, const u64 b, const u64 c) {
    return a ^ b ^ c;
}

__device__ __forceinline__ u64 majority3(const u64 a, const u64 b, const u64 c) {
    return (a & b) | (c & (a | b));
}

/*
  One thread computes one packed 64-cell word.

  A flat 1D grid-stride kernel is used for two reasons:
    1) it avoids the grid.y launch limit for very large boards,
    2) words_per_row is a power of two, so recovering (x, y) from a flat word index
       is just a mask + shift instead of division/modulo.

  Bit ordering assumption:
    - Bit i of a 64-bit word corresponds to column (word_x * 64 + i).
    - Therefore:
        west  neighbor alignment is a left shift  (<< 1),
        east  neighbor alignment is a right shift (>> 1).

  X-boundary optimization:
    - Because words_per_row is a power of two, the left/right word index inside a row
      can be wrapped with (x +/- 1) & row_mask.
    - That keeps left/right neighbor-word loads always in-bounds and branch-free.
    - The only incorrect data introduced by this wrap is the single carry bit imported
      from the wrapped word on the true left/right borders; that carry bit is explicitly
      zeroed with lsb/msb carry masks.

  Y-boundary handling:
    - Only the top and bottom rows need special handling.
    - Those rows are a tiny fraction of the total work, so simple branches are cheaper
      than masking six extra loaded words for every interior thread.

  Count construction:
    - Top row (NW, N, NE)      -> 2-bit count: top_hi:top_lo
    - Bottom row (SW, S, SE)   -> 2-bit count: bot_hi:bot_lo
    - Middle row side cells
      (W, E)                   -> 2-bit count: side_hi:side_lo

    First add top + bottom exactly to a 3-bit partial sum sum2:sum1:sum0.
    Then add W + E with a specialized 2-input adder.

    Any neighbor count >= 4 is collapsed into ge4 because the Life rule only needs
    to distinguish:
      - count == 2
      - count == 3
      - count >= 4 / count < 2
*/
__global__ __launch_bounds__(256)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         const std::size_t total_words,
                         const unsigned int words_per_row,
                         const unsigned int row_mask,
                         const unsigned int row_shift,
                         const unsigned int grid_dim) {
    const std::size_t thread_idx =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    const std::size_t pitch = static_cast<std::size_t>(words_per_row);
    const std::size_t row_mask_sz = static_cast<std::size_t>(row_mask);
    const unsigned int last_word_x = row_mask;
    const unsigned int last_row = grid_dim - 1u;

    for (std::size_t idx = thread_idx; idx < total_words; idx += stride) {
        // Recover word coordinates from the flat word index.
        const unsigned int x = static_cast<unsigned int>(idx) & row_mask;
        const unsigned int y = static_cast<unsigned int>(idx >> row_shift);

        // Row base has all low row-index bits cleared because words_per_row is a power of two.
        const std::size_t row_base = idx & ~row_mask_sz;

        // Wrap x inside the row; true border carries are zeroed by the masks below.
        const unsigned int left_x = (x - 1u) & row_mask;
        const unsigned int right_x = (x + 1u) & row_mask;
        const std::size_t left_word_idx = row_base | static_cast<std::size_t>(left_x);
        const std::size_t right_word_idx = row_base | static_cast<std::size_t>(right_x);

        // Only the injected cross-word carry bit must be suppressed at the actual borders.
        const u64 lsb_carry_mask = static_cast<u64>(x != 0u);                  // 0 or 1 at bit 0
        const u64 msb_carry_mask = static_cast<u64>(x != last_word_x) << 63;   // 0 or 1 at bit 63

        const u64 center = input[idx];

        // Middle row side neighbors (W, E). The center cell itself is excluded here.
        const u64 west =
            (center << 1) | ((input[left_word_idx] >> 63) & lsb_carry_mask);
        const u64 east =
            (center >> 1) | ((input[right_word_idx] << 63) & msb_carry_mask);

        const u64 side_lo = west ^ east;   // bit 0 of (W + E)
        const u64 side_hi = west & east;   // bit 1 of (W + E)

        // Top row (NW, N, NE) -> 2-bit count.
        u64 top_lo = 0ULL;
        u64 top_hi = 0ULL;
        if (y != 0u) {
            const std::size_t up_idx = idx - pitch;
            const u64 north = input[up_idx];
            const u64 north_west =
                (north << 1) | ((input[left_word_idx - pitch] >> 63) & lsb_carry_mask);
            const u64 north_east =
                (north >> 1) | ((input[right_word_idx - pitch] << 63) & msb_carry_mask);

            top_lo = xor3(north_west, north, north_east);
            top_hi = majority3(north_west, north, north_east);
        }

        // Bottom row (SW, S, SE) -> 2-bit count.
        u64 bottom_lo = 0ULL;
        u64 bottom_hi = 0ULL;
        if (y != last_row) {
            const std::size_t down_idx = idx + pitch;
            const u64 south = input[down_idx];
            const u64 south_west =
                (south << 1) | ((input[left_word_idx + pitch] >> 63) & lsb_carry_mask);
            const u64 south_east =
                (south >> 1) | ((input[right_word_idx + pitch] << 63) & msb_carry_mask);

            bottom_lo = xor3(south_west, south, south_east);
            bottom_hi = majority3(south_west, south, south_east);
        }

        // Exact addition of top + bottom -> 3-bit partial sum sum2:sum1:sum0.
        const u64 sum0 = top_lo ^ bottom_lo;
        const u64 carry0 = top_lo & bottom_lo;
        const u64 sum1 = xor3(top_hi, bottom_hi, carry0);
        const u64 sum2 = majority3(top_hi, bottom_hi, carry0);

        // Add the middle-row side count (W + E).
        // side_hi and (sum0 & side_lo) are mutually exclusive, so they merge into one carry.
        const u64 carry1 = side_hi | (sum0 & side_lo);
        const u64 count_lo = sum0 ^ side_lo;
        const u64 count_hi = sum1 ^ carry1;
        const u64 ge4 = sum2 | (sum1 & carry1);

        // Conway rule in bit-sliced form:
        //   next = (count == 3) | (alive & count == 2)
        //        = (~ge4) & count_hi & (count_lo | alive)
        const u64 two_or_three = count_hi & ~ge4;
        output[idx] = two_or_three & (count_lo | center);
    }
}

} // namespace

/*
  Executes one Game-of-Life step.

  Notes:
    - input/output are assumed to be distinct device buffers (double-buffered update).
    - No synchronization is performed here; the caller explicitly manages that.
    - Host-side launch configuration work is intentionally done here because only the
      device-side step performance matters.
*/
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    const unsigned int grid_dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = grid_dim >> 6; // grid_dimensions is a power of two > 512, so divisible by 64.
    const std::size_t total_words =
        static_cast<std::size_t>(grid_dim) * static_cast<std::size_t>(words_per_row);

    if (total_words == 0) {
        return;
    }

    // log2(words_per_row); words_per_row is also a power of two.
    unsigned int row_shift = 0;
    for (unsigned int w = words_per_row; w > 1u; w >>= 1u) {
        ++row_shift;
    }

    // The kernel uses no shared memory, so prefer L1 for the read-heavy access pattern.
    cudaFuncSetCacheConfig(game_of_life_kernel, cudaFuncCachePreferL1);

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, game_of_life_kernel, kBlockSize, 0);
    if (blocks_per_sm < 1) {
        blocks_per_sm = 1;
    }

    // Compact occupancy-sized launch + grid-stride loop:
    // avoids gigantic grids for very large boards while still keeping all SMs busy.
    const std::size_t max_blocks_needed =
        (total_words + static_cast<std::size_t>(kBlockSize) - 1u) /
        static_cast<std::size_t>(kBlockSize);

    std::size_t launch_blocks =
        static_cast<std::size_t>(sm_count) *
        static_cast<std::size_t>(blocks_per_sm) *
        static_cast<std::size_t>(kLaunchWaves);

    if (launch_blocks > max_blocks_needed) {
        launch_blocks = max_blocks_needed;
    }
    if (launch_blocks == 0) {
        launch_blocks = 1;
    }

    game_of_life_kernel<<<static_cast<unsigned int>(launch_blocks), kBlockSize>>>(
        input,
        output,
        total_words,
        words_per_row,
        words_per_row - 1u,
        row_shift,
        grid_dim);
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
