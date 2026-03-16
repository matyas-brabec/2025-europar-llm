#include <cuda_runtime.h>
#include <cstdint>

namespace {

// 64-bit packed cell word. Bit i corresponds to cell x = word_x * 64 + i in a row.
// With that convention:
//   left-neighbor-aligned  mask = (word << 1) | (left_word  >> 63)
//   right-neighbor-aligned mask = (word >> 1) | (right_word << 63)
// This is exactly the required special handling for bit 0 and bit 63.
using u64 = std::uint64_t;

// Fixed launch shape.
// For valid inputs:
//   grid_dimensions = 2^n, n >= 10
//   words_per_row   = grid_dimensions / 64 = 2^(n-6) >= 16
//   total_words     = grid_dimensions * words_per_row = 2^(2n-6)
// so total_words is always a multiple of 256, which lets the kernel avoid an
// idx < total_words guard in the hot path.
constexpr int kThreadsPerBlockLog2 = 8;
constexpr int kThreadsPerBlock = 1 << kThreadsPerBlockLog2;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert(sizeof(u64) == 8, "Expected 64-bit words.");
static_assert(sizeof(unsigned long long) == 8, "Expected 64-bit unsigned long long.");
static_assert(kThreadsPerBlock == 256, "Index computation below assumes 256 threads per block.");

// Bit-sliced half/full adders.
// Every bit position is an independent 1-bit lane, so one instruction sequence
// updates 64 cells at once. On modern NVIDIA GPUs these boolean forms map well
// to LOP3-heavy code.
__device__ __forceinline__ void half_adder(const u64 a, const u64 b, u64& sum, u64& carry) {
    sum = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ void full_adder(const u64 a, const u64 b, const u64 c,
                                           u64& sum, u64& carry) {
    sum = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// 64-bit shuffle helpers. Shared memory is intentionally avoided; horizontal
// neighbor words are exchanged with warp shuffles instead. Only subgroup-edge
// lanes need extra global loads for cross-subgroup words.
template<int WIDTH>
__device__ __forceinline__ u64 subgroup_shfl_up_1(const u64 v) {
    return static_cast<u64>(
        __shfl_up_sync(kFullWarpMask, static_cast<unsigned long long>(v), 1, WIDTH));
}

template<int WIDTH>
__device__ __forceinline__ u64 subgroup_shfl_down_1(const u64 v) {
    return static_cast<u64>(
        __shfl_down_sync(kFullWarpMask, static_cast<unsigned long long>(v), 1, WIDTH));
}

// Current row contribution: only left/right neighbors count; the center bit does not.
__device__ __forceinline__ void horizontal_pair_count(
    const u64 row, const u64 left_word, const u64 right_word, u64& sum, u64& carry) {
    const u64 left_neighbors  = (row << 1) | (left_word >> 63);
    const u64 right_neighbors = (row >> 1) | (right_word << 63);
    half_adder(left_neighbors, right_neighbors, sum, carry);
}

// Top/bottom row contribution: left/center/right are all neighbors.
__device__ __forceinline__ void horizontal_triplet_count(
    const u64 row, const u64 left_word, const u64 right_word, u64& sum, u64& carry) {
    const u64 left_neighbors  = (row << 1) | (left_word >> 63);
    const u64 right_neighbors = (row >> 1) | (right_word << 63);
    full_adder(left_neighbors, row, right_neighbors, sum, carry);
}

// SUBGROUP_WIDTH is 16 only for a 1024x1024 grid (16 packed words per row).
// For all larger valid grids it is 32, so each warp processes a 32-word segment
// of a row. In both cases the subgroup width matches the natural horizontal
// shuffle domain, which keeps cross-lane neighbor exchange cheap and correct.
template<int SUBGROUP_WIDTH>
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         const unsigned int words_per_row,
                         const u64 total_words) {
    static_assert(SUBGROUP_WIDTH == 16 || SUBGROUP_WIDTH == 32,
                  "Only 16- and 32-lane shuffle subgroups are supported.");

    // Exact 1-D mapping: one thread owns one 64-cell word.
    const u64 idx = (static_cast<u64>(blockIdx.x) << kThreadsPerBlockLog2) +
                    static_cast<u64>(threadIdx.x);

    // Because words_per_row is a power of two, x-within-row is a simple mask.
    const u64 stride   = static_cast<u64>(words_per_row);
    const u64 row_mask = stride - 1ULL;
    const u64 x        = idx & row_mask;

    const bool has_left   = x != 0ULL;
    const bool has_right  = x != row_mask;
    const bool has_top    = idx >= stride;
    const bool has_bottom = idx + stride < total_words;

    // Common-case global loads: only the vertically aligned words.
    // Relative to a naive 9-load stencil, the hot path is reduced to 3 loads/thread.
    const u64 center = input[idx];

    u64 top = 0ULL;
    u64 bottom = 0ULL;
    u64 top_idx = 0ULL;
    u64 bottom_idx = 0ULL;

    if (has_top) {
        top_idx = idx - stride;
        top = input[top_idx];
    }
    if (has_bottom) {
        bottom_idx = idx + stride;
        bottom = input[bottom_idx];
    }

    // Determine whether this lane sits at a shuffle subgroup boundary.
    // Only those lanes need the extra global loads for cross-subgroup words.
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int subgroup_mask = static_cast<unsigned int>(SUBGROUP_WIDTH - 1);
    const unsigned int lane_in_group = lane & subgroup_mask;
    const bool first_in_group = lane_in_group == 0U;
    const bool last_in_group  = lane_in_group == subgroup_mask;

    // Current row: get the word to the left/right.
    u64 left_word = subgroup_shfl_up_1<SUBGROUP_WIDTH>(center);
    u64 right_word = subgroup_shfl_down_1<SUBGROUP_WIDTH>(center);

    if (first_in_group) {
        left_word = has_left ? input[idx - 1ULL] : 0ULL;
    }
    if (last_in_group) {
        right_word = has_right ? input[idx + 1ULL] : 0ULL;
    }

    u64 mid_sum;
    u64 mid_carry;
    horizontal_pair_count(center, left_word, right_word, mid_sum, mid_carry);

    // Top row: same shuffle pattern, but boundary loads come from the row above.
    left_word = subgroup_shfl_up_1<SUBGROUP_WIDTH>(top);
    right_word = subgroup_shfl_down_1<SUBGROUP_WIDTH>(top);

    if (first_in_group) {
        left_word = (has_top && has_left) ? input[top_idx - 1ULL] : 0ULL;
    }
    if (last_in_group) {
        right_word = (has_top && has_right) ? input[top_idx + 1ULL] : 0ULL;
    }

    u64 top_sum;
    u64 top_carry;
    horizontal_triplet_count(top, left_word, right_word, top_sum, top_carry);

    // Bottom row: same again for the row below.
    left_word = subgroup_shfl_up_1<SUBGROUP_WIDTH>(bottom);
    right_word = subgroup_shfl_down_1<SUBGROUP_WIDTH>(bottom);

    if (first_in_group) {
        left_word = (has_bottom && has_left) ? input[bottom_idx - 1ULL] : 0ULL;
    }
    if (last_in_group) {
        right_word = (has_bottom && has_right) ? input[bottom_idx + 1ULL] : 0ULL;
    }

    u64 bottom_sum;
    u64 bottom_carry;
    horizontal_triplet_count(bottom, left_word, right_word, bottom_sum, bottom_carry);

    // Each row count is encoded as:
    //   row_count = row_sum + 2 * row_carry
    //
    // Add the three row counts in bit-sliced form:
    //   total_count = ones + 2 * twos + 4 * high
    // where `high` is intentionally saturated to "count >= 4", because the Life
    // rule only needs exact counts 2 and 3.
    u64 ones;
    u64 twos_from_row_lsb;
    full_adder(top_sum, mid_sum, bottom_sum, ones, twos_from_row_lsb);

    u64 twos_from_row_msb;
    u64 high_from_row_msb;
    full_adder(top_carry, mid_carry, bottom_carry,
               twos_from_row_msb, high_from_row_msb);

    u64 twos;
    u64 overflow_to_four_or_more;
    half_adder(twos_from_row_lsb, twos_from_row_msb, twos, overflow_to_four_or_more);

    const u64 high = high_from_row_msb | overflow_to_four_or_more;

    // Conway rule in bit-sliced form:
    //   count == 2 -> survive iff currently alive
    //   count == 3 -> always alive
    //
    // `twos & ~high` selects counts {2, 3}; `(ones | center)` turns:
    //   count 2 -> center
    //   count 3 -> 1
    output[idx] = twos & ~high & (ones | center);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Contract assumptions from the problem statement are taken as given:
    // - square grid
    // - grid_dimensions is a power of two and > 512
    // - input/output are device pointers from cudaMalloc
    // - input/output fit in device memory
    //
    // No host/device synchronization or launch error checking is done here;
    // the caller explicitly owns synchronization, and only steady-state speed matters.
    const unsigned int words_per_row = static_cast<unsigned int>(grid_dimensions >> 6);
    const u64 total_words =
        static_cast<u64>(grid_dimensions) * static_cast<u64>(words_per_row);

    // Exact launch: total_words is always a multiple of 256 for valid inputs.
    const unsigned int blocks =
        static_cast<unsigned int>(total_words >> kThreadsPerBlockLog2);

    // 1024x1024 -> 16 packed words per row, so half-warps are the natural shuffle domain.
    // Larger grids have at least 32 packed words per row, so full-warps are used.
    if (words_per_row == 16U) {
        game_of_life_kernel<16><<<blocks, kThreadsPerBlock>>>(
            input, output, words_per_row, total_words);
    } else {
        game_of_life_kernel<32><<<blocks, kThreadsPerBlock>>>(
            input, output, words_per_row, total_words);
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
