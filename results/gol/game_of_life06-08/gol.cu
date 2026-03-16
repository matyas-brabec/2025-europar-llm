#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// Bit-packed board word: one thread updates one 64-bit word = 64 cells.
using word_t = std::uint64_t;

// Bit layout convention used throughout this implementation:
//   bit k of a word stores the cell at x = word_base_x + k
// Therefore:
//   - the west neighbor of bit k is bit k-1
//   - the east neighbor of bit k is bit k+1
//
// To align neighbors with the current cell position, we shift the current row word and
// import one crossing bit from the adjacent word. This is exactly the "special handling"
// required for bit 0 / bit 63 at word boundaries.
__device__ __forceinline__ word_t align_west(word_t row_word, word_t left_word) {
    return (row_word << 1) | (left_word >> 63);
}

__device__ __forceinline__ word_t align_east(word_t row_word, word_t right_word) {
    return (row_word >> 1) | (right_word << 63);
}

// Bit-sliced full adder over 64 independent cell positions.
// For every bit position i:
//   a_i + b_i + c_i = sum_i + 2 * carry_i
//
// The expression is written in a form that modern NVIDIA compilers typically lower
// efficiently (often to LOP3-based logic on Ampere/Hopper).
__device__ __forceinline__ void add3(word_t a, word_t b, word_t c,
                                     word_t& sum, word_t& carry) {
    const word_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// kSubgroupWidth:
//   * 16 when a row contains only 16 packed words (the smallest legal board: 1024x1024)
//   * 32 for all larger legal boards
//
// Left/right neighbor words are exchanged with warp shuffles inside that subgroup.
// Only subgroup-edge lanes need fallback global loads, which removes most redundant
// left/right memory traffic without using shared memory.
template <int kSubgroupWidth>
__launch_bounds__(256, 2)
__global__ void game_of_life_kernel(const word_t* __restrict__ input,
                                    word_t* __restrict__ output,
                                    std::size_t words_per_row,
                                    std::size_t total_words) {
    static_assert(kSubgroupWidth == 16 || kSubgroupWidth == 32,
                  "Unsupported shuffle subgroup width.");
    static_assert((256 % kSubgroupWidth) == 0,
                  "Block size must be a multiple of the shuffle subgroup width.");

    const std::size_t x_mask        = words_per_row - 1;          // words_per_row is a power of two
    const std::size_t last_row_base = total_words - words_per_row;
    const std::size_t stride        = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    const unsigned subgroup_lane = static_cast<unsigned>(threadIdx.x) & (kSubgroupWidth - 1);
    const unsigned subgroup_last = kSubgroupWidth - 1;

    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += stride) {
        // Because the row width in packed words is a power of two, x can be extracted with
        // a mask instead of division/modulo. This is one of the key constraints we exploit.
        const std::size_t x        = idx & x_mask;
        const std::size_t row_base = idx - x;

        const bool has_left   = (x != 0);
        const bool has_right  = (x != x_mask);
        const bool has_top    = (row_base != 0);
        const bool has_bottom = (row_base != last_row_base);

        // These indices are only dereferenced when the corresponding has_* predicate is true.
        const std::size_t north_idx = has_top    ? idx - words_per_row : 0;
        const std::size_t south_idx = has_bottom ? idx + words_per_row : 0;

        // Core loads:
        //   center word for the current row
        //   matching word from the row above
        //   matching word from the row below
        //
        // Left/right words are obtained from shuffles whenever possible.
        const word_t center = input[idx];
        const word_t north  = has_top    ? input[north_idx] : 0ull;
        const word_t south  = has_bottom ? input[south_idx] : 0ull;

        // __activemask() keeps the code safe even if this kernel is later reused with a
        // different launch geometry or weaker input constraints.
        const unsigned shuffle_mask = __activemask();

        // Top-row contribution: NW + N + NE
        word_t top_lo, top_hi;
        {
            word_t north_left  = __shfl_up_sync  (shuffle_mask, north, 1, kSubgroupWidth);
            word_t north_right = __shfl_down_sync(shuffle_mask, north, 1, kSubgroupWidth);

            // Only subgroup-edge lanes need cross-subgroup or boundary fallback loads.
            if (subgroup_lane == 0) {
                north_left = (has_top && has_left) ? input[north_idx - 1] : 0ull;
            }
            if (subgroup_lane == subgroup_last) {
                north_right = (has_top && has_right) ? input[north_idx + 1] : 0ull;
            }

            const word_t nw = align_west(north, north_left);
            const word_t ne = align_east(north, north_right);
            add3(nw, north, ne, top_lo, top_hi);
        }

        // Middle-row contribution: W + E
        // The center cell itself is NOT part of the neighbor count.
        word_t mid_lo, mid_hi;
        {
            word_t left_word  = __shfl_up_sync  (shuffle_mask, center, 1, kSubgroupWidth);
            word_t right_word = __shfl_down_sync(shuffle_mask, center, 1, kSubgroupWidth);

            if (subgroup_lane == 0) {
                left_word = has_left ? input[idx - 1] : 0ull;
            }
            if (subgroup_lane == subgroup_last) {
                right_word = has_right ? input[idx + 1] : 0ull;
            }

            const word_t west = align_west(center, left_word);
            const word_t east = align_east(center, right_word);

            // Half-adder for two inputs:
            //   west + east = mid_lo + 2 * mid_hi
            mid_lo = west ^ east;
            mid_hi = west & east;
        }

        // Bottom-row contribution: SW + S + SE
        word_t bot_lo, bot_hi;
        {
            word_t south_left  = __shfl_up_sync  (shuffle_mask, south, 1, kSubgroupWidth);
            word_t south_right = __shfl_down_sync(shuffle_mask, south, 1, kSubgroupWidth);

            if (subgroup_lane == 0) {
                south_left = (has_bottom && has_left) ? input[south_idx - 1] : 0ull;
            }
            if (subgroup_lane == subgroup_last) {
                south_right = (has_bottom && has_right) ? input[south_idx + 1] : 0ull;
            }

            const word_t sw = align_west(south, south_left);
            const word_t se = align_east(south, south_right);
            add3(sw, south, se, bot_lo, bot_hi);
        }

        // Each row partial is encoded as:
        //   row_sum = row_lo + 2 * row_hi
        //
        // Combine the three row partials into a full 4-bit neighbor count:
        //   count = count0 + 2*count1 + 4*count2 + 8*count3
        //
        // This is still fully bit-sliced: each bit position represents one cell.
        word_t count0, carry2_from_low_bits;
        add3(top_lo, mid_lo, bot_lo, count0, carry2_from_low_bits);

        word_t tmp_count1, carry4_a;
        add3(carry2_from_low_bits, top_hi, mid_hi, tmp_count1, carry4_a);

        const word_t count1   = tmp_count1 ^ bot_hi;
        const word_t carry4_b = tmp_count1 & bot_hi;
        const word_t count2   = carry4_a ^ carry4_b;
        const word_t count3   = carry4_a & carry4_b;

        // Conway rule in terms of the bit-sliced count:
        //   birth/survival on exactly 3 neighbors
        //   survival only on exactly 2 neighbors
        //
        // count in {2,3} iff:
        //   count1 == 1 and count2 == 0 and count3 == 0
        //
        // Then:
        //   next = (count == 3) || (alive && count == 2)
        //        = (count1 & ~(count2 | count3)) & (count0 | alive)
        const word_t two_or_three = count1 & ~(count2 | count3);
        output[idx] = two_or_three & (center | count0);
    }
}

__host__ inline int get_current_device_sm_count() {
    // Cached per host thread to avoid repeated runtime queries when stepping many generations.
    thread_local int cached_device   = -1;
    thread_local int cached_sm_count = 0;

    int device = 0;
    cudaGetDevice(&device);

    if (device != cached_device) {
        cudaDeviceGetAttribute(&cached_sm_count, cudaDevAttrMultiProcessorCount, device);
        cached_device = device;
    }

    return cached_sm_count;
}

} // anonymous namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees:
    //   * square board
    //   * power-of-two dimension
    //   * dimension > 512
    //   * input/output are device buffers allocated by cudaMalloc
    //   * synchronization/error handling are managed externally
    //
    // This implementation directly consumes the provided bit-packed layout, so no host-side
    // or device-side format conversion is needed.

    constexpr int kBlockSize   = 256;
    constexpr int kBlocksPerSM = 8;

    const std::size_t words_per_row = static_cast<std::size_t>(grid_dimensions) >> 6;
    const std::size_t total_words   = words_per_row * static_cast<std::size_t>(grid_dimensions);

    if (total_words == 0) {
        return;
    }

    // Persistent-grid style launch:
    //   * enough blocks to fill the GPU well
    //   * grid-stride loop covers arbitrarily large boards
    //   * avoids 2D grid dimension limits on very large inputs
    const int sm_count = get_current_device_sm_count();
    const std::size_t desired_blocks =
        static_cast<std::size_t>((sm_count > 0) ? sm_count : 1) * kBlocksPerSM;
    const std::size_t max_blocks = (total_words + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(max_blocks < desired_blocks ? max_blocks : desired_blocks);

    if (words_per_row == 16) {
        // 1024x1024 cells => exactly 16 packed words per row.
        // Half-warps line up perfectly with rows, so use 16-lane shuffle subgroups.
        game_of_life_kernel<16><<<blocks, kBlockSize>>>(input, output, words_per_row, total_words);
    } else {
        // All larger legal boards have at least 32 packed words per row, and still a power-of-two
        // width, so full warps map cleanly to contiguous row segments.
        game_of_life_kernel<32><<<blocks, kBlockSize>>>(input, output, words_per_row, total_words);
    }

    // Intentionally no synchronization and no launch-error polling here:
    // the caller explicitly owns those responsibilities.
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
