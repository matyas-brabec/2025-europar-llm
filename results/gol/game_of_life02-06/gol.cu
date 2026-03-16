#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Optimized bit-packed Conway's Game of Life for modern NVIDIA GPUs.
//
// Representation:
//   - The grid is row-major.
//   - Within a row, cell x is stored in bit (x & 63) of word (x >> 6).
//   - One thread updates one 64-bit word, i.e. 64 cells at once.
//   - One warp updates 32 consecutive words of one row segment.
//
// Design choices:
//   - No shared memory: the prompt notes that it is unnecessary here.
//   - Horizontal word exchange uses warp shuffles; only segment-edge lanes need
//     one fallback global load.
//   - Neighbor counting stays fully bit-packed in registers using a short boolean
//     carry-save network; the kernel never expands cells into bytes or ints.
//   - The host wrapper does not synchronize; the caller is responsible for that.

namespace {

using word_t = unsigned long long;

constexpr int kCellsPerWordShift   = 6;  // 64 cells / word.
constexpr int kLastBit             = 63; // Highest bit position in a 64-bit word.
constexpr int kWarpWords           = 32; // One warp covers 32 consecutive words.
constexpr int kWarpShift           = 5;  // log2(32).
constexpr int kRowsPerBlock        = 8;  // 8 warps / block => 256 threads.
constexpr int kRowsPerBlockShift   = 3;  // log2(8).

static_assert(kWarpWords == (1 << kWarpShift), "kWarpWords must be a power of two.");
static_assert(kRowsPerBlock == (1 << kRowsPerBlockShift), "kRowsPerBlock must be a power of two.");

__device__ __forceinline__ void add3(const word_t a, const word_t b, const word_t c,
                                     word_t& sum, word_t& carry) {
    // Full-adder applied lane-wise to 64 independent 1-bit values:
    //   sum + 2*carry == a + b + c
    const word_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (c & ab_xor);
}

__device__ __forceinline__ word_t exact_one4(const word_t x, const word_t y,
                                             const word_t z, const word_t w) {
    // Return a bitmask with 1s exactly where x+y+z+w == 1, lane-wise.
    // Pairwise half-adders keep the dependency chain short.
    const word_t p = x ^ y;
    const word_t q = x & y;
    const word_t r = z ^ w;
    const word_t t = z & w;
    return (p ^ r) & ~(q | t);
}

__device__ __forceinline__ void adjacent_words(
    const unsigned active_mask,
    const unsigned lane,
    const int xw,
    const int words_per_row,
    const std::uint64_t* row_ptr,
    const word_t center,
    word_t& left,
    word_t& right) {
    // A warp covers 32 consecutive words of one row. In the common case the
    // neighboring word is already resident in the adjacent lane's register.
    // Only the first/last lane of a 32-word segment falls back to one extra load.
    const word_t shfl_left  = __shfl_up_sync(active_mask, center, 1);
    const word_t shfl_right = __shfl_down_sync(active_mask, center, 1);

    left = (xw != 0)
        ? ((lane != 0u) ? shfl_left : static_cast<word_t>(row_ptr[xw - 1]))
        : 0ull;

    right = (xw != words_per_row - 1)
        ? ((lane != static_cast<unsigned>(kWarpWords - 1))
            ? shfl_right
            : static_cast<word_t>(row_ptr[xw + 1]))
        : 0ull;
}

__global__ __launch_bounds__(kWarpWords * kRowsPerBlock)
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int grid_dimensions,
                         int words_per_row,
                         int words_log2,
                         unsigned int row_block_mask,
                         int row_blocks_log2) {
    // grid.x is a flattened 2-D launch space:
    //   low  bits -> row block (8 rows)
    //   high bits -> 32-word x-segment
    const unsigned linear_block = blockIdx.x;
    const unsigned row_block    = linear_block & row_block_mask;
    const unsigned segment      = linear_block >> row_blocks_log2;

    const unsigned lane         = threadIdx.x;
    const unsigned row_in_block = threadIdx.y;

    const int xw = static_cast<int>((segment << kWarpShift) | lane);
    if (xw >= words_per_row) {
        // Only possible for the smallest legal board: 1024x1024 => 16 words/row.
        return;
    }

    const int y = static_cast<int>((row_block << kRowsPerBlockShift) | row_in_block);
    const int last_row = grid_dimensions - 1;
    const bool has_top = (y != 0);
    const bool has_bottom = (y != last_row);

    const unsigned active_mask = __activemask();

    // Because the problem guarantees power-of-two dimensions, all row/word address
    // calculations reduce to shifts and bitwise ors.
    const std::size_t row_offset =
        (static_cast<std::size_t>(row_block) << (words_log2 + kRowsPerBlockShift)) |
        (static_cast<std::size_t>(row_in_block) << words_log2);

    const std::uint64_t* row_ptr = input + row_offset;
    std::uint64_t* out_row_ptr   = output + row_offset;

    const word_t center = static_cast<word_t>(row_ptr[xw]);

    // Compress the three neighbors from the row above into:
    //   top_count == s_top + 2*c_top
    word_t s_top = 0ull;
    word_t c_top = 0ull;
    if (has_top) {
        const std::uint64_t* up_row = row_ptr - words_per_row;
        const word_t up = static_cast<word_t>(up_row[xw]);

        word_t up_left, up_right;
        adjacent_words(active_mask, lane, xw, words_per_row, up_row, up, up_left, up_right);

        // Shift-and-stitch across 64-bit word boundaries so each bit position is
        // aligned with the cell updated by this thread.
        const word_t up_west = (up << 1) | (up_left >> kLastBit);
        const word_t up_east = (up >> 1) | (up_right << kLastBit);

        add3(up_west, up, up_east, s_top, c_top);
    }

    // Same compression for the row below.
    word_t s_bottom = 0ull;
    word_t c_bottom = 0ull;
    if (has_bottom) {
        const std::uint64_t* down_row = row_ptr + words_per_row;
        const word_t down = static_cast<word_t>(down_row[xw]);

        word_t down_left, down_right;
        adjacent_words(active_mask, lane, xw, words_per_row, down_row, down, down_left, down_right);

        const word_t down_west = (down << 1) | (down_left >> kLastBit);
        const word_t down_east = (down >> 1) | (down_right << kLastBit);

        add3(down_west, down, down_east, s_bottom, c_bottom);
    }

    // Current-row side neighbors.
    word_t center_left, center_right;
    adjacent_words(active_mask, lane, xw, words_per_row, row_ptr, center, center_left, center_right);

    const word_t west = (center << 1) | (center_left >> kLastBit);
    const word_t east = (center >> 1) | (center_right << kLastBit);

    // Neighbor count decomposition:
    //   top    = s_top    + 2*c_top
    //   bottom = s_bottom + 2*c_bottom
    //   top + bottom + west = s_mid + 2*c_mid
    //   total_neighbors = (s_mid + east) + 2*(carry_from_low + c_mid + c_top + c_bottom)
    //
    // Conway's rule needs only counts 2 and 3:
    //   - exactly one of the four "2-weight" contributors must be present
    //   - then low_bit==1 means count==3, while low_bit==0 means count==2 and the
    //     cell survives only if it is already alive.
    word_t s_mid, c_mid;
    add3(s_top, s_bottom, west, s_mid, c_mid);

    const word_t low_bit        = s_mid ^ east;
    const word_t carry_from_low = s_mid & east;
    const word_t twos_eq_1      = exact_one4(carry_from_low, c_mid, c_top, c_bottom);
    const word_t next           = twos_eq_1 & (low_bit | center);

    out_row_ptr[xw] = static_cast<std::uint64_t>(next);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees:
    //   - square grid
    //   - power-of-two dimension > 512
    //   - input/output are device pointers from cudaMalloc
    //   - input and output are distinct buffers
    //
    // The wrapper intentionally performs no validation and no synchronization.
    const int words_per_row = grid_dimensions >> kCellsPerWordShift;

    // One warp updates one 32-word segment. Since words_per_row is a power of two,
    // the segment count is either 1 (for 16 or 32 words/row) or another power of two.
    const unsigned int segments =
        static_cast<unsigned int>((words_per_row + kWarpWords - 1) >> kWarpShift);

    // Using grid.y for row blocks would hit the architectural 65535 limit for the
    // largest valid boards, so the launch space is flattened into grid.x instead.
    const unsigned int row_blocks =
        static_cast<unsigned int>(grid_dimensions >> kRowsPerBlockShift);

    int row_blocks_log2 = 0;
    while ((1u << row_blocks_log2) < row_blocks) {
        ++row_blocks_log2;
    }

    // row_blocks == words_per_row * 8, so:
    //   log2(words_per_row) == log2(row_blocks) - 3
    const int words_log2 = row_blocks_log2 - kRowsPerBlockShift;

    const std::uint64_t total_blocks_64 =
        static_cast<std::uint64_t>(segments) << row_blocks_log2;

    const dim3 block(kWarpWords, kRowsPerBlock, 1);
    const dim3 grid(static_cast<unsigned int>(total_blocks_64), 1, 1);

    game_of_life_kernel<<<grid, block>>>(
        input,
        output,
        grid_dimensions,
        words_per_row,
        words_log2,
        row_blocks - 1u,
        row_blocks_log2);
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
