#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

/*
Optimized single-step Conway's Game of Life for a square, bit-packed grid.

Key ideas used here for performance on modern datacenter GPUs:
  1. One CUDA thread updates one 64-bit word, i.e. 64 cells at once.
  2. Horizontal neighbor words are exchanged with warp/sub-warp shuffles instead of
     shared memory. Only the segment-edge lanes perform extra global halo loads.
  3. The eight neighbor counts are accumulated bit-slice-wise with carry-save adders,
     so all 64 cells in a word are updated in parallel with boolean operations.
  4. A flattened 1D grid of thread blocks is used to avoid gridDim.y limits on very
     large boards. Because the dimensions are powers of two, decoding the flattened
     block index is just a mask + shift.

Bit layout assumption (matches the prompt's "0th/63rd bit need special handling"):
  - Bit k in a word corresponds to column (word_index * 64 + k).
  - Therefore:
      west  neighbors = (word << 1) | (left_word  >> 63)
      east  neighbors = (word >> 1) | (right_word << 63)
  - Cells outside the board are treated as dead by zeroing missing rows/halos.
*/

namespace {

using u64 = std::uint64_t;

constexpr unsigned int kFullWarpMask   = 0xFFFFFFFFu;
constexpr int          kThreadsPerBlock = 256;

/* Build the "west" neighbor bitmask for the current 64-cell word. */
__device__ __forceinline__ u64 shift_left_with_carry(u64 word, u64 left_word) {
    return (word << 1) | (left_word >> 63);
}

/* Build the "east" neighbor bitmask for the current 64-cell word. */
__device__ __forceinline__ u64 shift_right_with_carry(u64 word, u64 right_word) {
    return (word >> 1) | (right_word << 63);
}

/*
Carry-save adder for three equally-weighted bitboards.
For every bit position independently:
  a + b + c = low + 2*high
where:
  low  = parity(a,b,c)
  high = majority(a,b,c)
*/
__device__ __forceinline__ void csa(u64& high, u64& low, u64 a, u64 b, u64 c) {
    low  = a ^ b ^ c;
    high = (a & b) | (a & c) | (b & c);
}

/*
Fetch the raw x-1 and x+1 words for one row.

`GROUP_WIDTH` is either 32 (one full row segment per warp) or 16 (two independent
row segments per warp). The shuffle width confines communication to that segment.
Only the first/last lane of each segment falls back to a global load that crosses
the segment boundary.
*/
template <int GROUP_WIDTH>
__device__ __forceinline__ void fetch_horizontal_neighbors(
    u64 word,
    const u64* __restrict__ cell_ptr,
    std::ptrdiff_t row_delta_words,
    bool row_exists,
    bool has_left_global,
    bool has_right_global,
    bool has_left_in_group,
    bool has_right_in_group,
    u64& left_word,
    u64& right_word)
{
    const u64 shfl_left  = __shfl_up_sync(kFullWarpMask,   word, 1, GROUP_WIDTH);
    const u64 shfl_right = __shfl_down_sync(kFullWarpMask, word, 1, GROUP_WIDTH);

    left_word =
        (row_exists && has_left_global)
            ? (has_left_in_group ? shfl_left : cell_ptr[row_delta_words - 1])
            : u64{0};

    right_word =
        (row_exists && has_right_global)
            ? (has_right_in_group ? shfl_right : cell_ptr[row_delta_words + 1])
            : u64{0};
}

template <int GROUP_WIDTH, int BLOCK_Y>
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    int words_per_row,
    int grid_dimensions,
    unsigned int blocks_per_row_mask,
    unsigned int blocks_per_row_shift)
{
    static_assert(GROUP_WIDTH * BLOCK_Y == kThreadsPerBlock,
                  "The tuned launch shape is expected to keep 256 threads per block.");

    constexpr int kGroupShift  = (GROUP_WIDTH == 16 ? 4 : 5);
    constexpr int kBlockYShift = (BLOCK_Y    == 16 ? 4 : 3);

    /*
    Flattened block decode:
      block_x = blockIdx.x % blocks_per_row
      block_y = blockIdx.x / blocks_per_row

    Because blocks_per_row is a power of two, the host passes its mask and shift so
    this decode is just bit operations.
    */
    const unsigned int linear_block = blockIdx.x;
    const int bx = static_cast<int>(linear_block & blocks_per_row_mask);
    const int by = static_cast<int>(linear_block >> blocks_per_row_shift);

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int x = (bx << kGroupShift)  | tx;
    const int y = (by << kBlockYShift) | ty;

    /*
    Exact tiling is guaranteed by the problem constraints:
      - grid_dimensions is a power of two,
      - words_per_row = grid_dimensions / 64 is a power of two,
      - launch shapes are chosen so block dimensions divide the board exactly.
    Hence no bounds checks are needed here, only physical board-edge checks.
    */
    const bool has_up            = (y != 0);
    const bool has_down          = (y != (grid_dimensions - 1));
    const bool has_left_global   = (x != 0);
    const bool has_right_global  = (x != (words_per_row - 1));
    const bool has_left_in_group = (tx != 0);
    const bool has_right_in_group= (tx != (GROUP_WIDTH - 1));

    const std::size_t idx =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(words_per_row) +
        static_cast<std::size_t>(x);

    const u64* const cell = input + idx;
    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(words_per_row);

    const u64 center = *cell;
    const u64 north  = has_up   ? cell[-stride] : u64{0};
    const u64 south  = has_down ? cell[ stride] : u64{0};

    u64 north_left_word, north_right_word;
    u64 center_left_word, center_right_word;
    u64 south_left_word, south_right_word;

    fetch_horizontal_neighbors<GROUP_WIDTH>(
        north, cell, -stride,
        has_up, has_left_global, has_right_global, has_left_in_group, has_right_in_group,
        north_left_word, north_right_word);

    fetch_horizontal_neighbors<GROUP_WIDTH>(
        center, cell, 0,
        true, has_left_global, has_right_global, has_left_in_group, has_right_in_group,
        center_left_word, center_right_word);

    fetch_horizontal_neighbors<GROUP_WIDTH>(
        south, cell, stride,
        has_down, has_left_global, has_right_global, has_left_in_group, has_right_in_group,
        south_left_word, south_right_word);

    /* The eight neighbor bitboards aligned to the current 64-cell lane group. */
    const u64 nw = shift_left_with_carry (north,  north_left_word);
    const u64 ne = shift_right_with_carry(north,  north_right_word);
    const u64 w  = shift_left_with_carry (center, center_left_word);
    const u64 e  = shift_right_with_carry(center, center_right_word);
    const u64 sw = shift_left_with_carry (south,  south_left_word);
    const u64 se = shift_right_with_carry(south,  south_right_word);

    /*
    Specialized 8-input bit-sliced adder tree.

    After this sequence:
      ones  = 1's bit of the neighbor count for each cell
      twos  = 2's bit
      fours = 4's bit

    We intentionally do NOT materialize the 8's bit, because Conway's rule only needs
    to know whether the neighbor count is 2 or 3:
      - For counts 0..8, (twos & ~fours) is true iff the count is 2 or 3.
      - Count 8 also has fours==0, but twos==0, so the missing 8's bit is irrelevant.
      - (ones | center) then maps {2,3} to the actual Life rule:
          count==2 -> survive only if center was already alive
          count==3 -> always alive
    */
    u64 ones  = nw ^ north;
    u64 twosA = nw & north;

    u64 twosB;
    csa(twosB, ones, ones, ne, w);

    u64 twos   = twosA ^ twosB;
    u64 foursA = twosA & twosB;

    u64 twosC;
    csa(twosC, ones, ones, e, sw);

    u64 twosD;
    csa(twosD, ones, ones, south, se);

    u64 foursB;
    csa(foursB, twos, twos, twosC, twosD);

    const u64 fours = foursA ^ foursB;
    output[idx] = twos & ~fours & (ones | center);
}

/* Host-side helper: blocks_per_row is guaranteed to be a power of two. */
inline unsigned int log2_power_of_two(unsigned int value) {
    unsigned int shift = 0;
    while (value > 1u) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

template <int GROUP_WIDTH, int BLOCK_Y>
inline void launch_game_of_life_step(
    const u64* input,
    u64* output,
    int grid_dimensions,
    int words_per_row)
{
    static_assert(GROUP_WIDTH * BLOCK_Y == kThreadsPerBlock,
                  "The tuned launch shape is expected to keep 256 threads per block.");

    const int blocks_per_row = words_per_row / GROUP_WIDTH;
    const int block_rows     = grid_dimensions / BLOCK_Y;

    const unsigned int blocks_per_row_mask  = static_cast<unsigned int>(blocks_per_row - 1);
    const unsigned int blocks_per_row_shift = log2_power_of_two(static_cast<unsigned int>(blocks_per_row));

    /*
    A 1D grid is used deliberately:
      total_blocks = blocks_per_row * block_rows

    This avoids any possible gridDim.y limit for very large boards while keeping the
    kernel-side decode cheap.
    */
    const unsigned int total_blocks = static_cast<unsigned int>(
        static_cast<std::size_t>(blocks_per_row) * static_cast<std::size_t>(block_rows));

    const dim3 block_dim(GROUP_WIDTH, BLOCK_Y, 1);
    const dim3 grid_dim(total_blocks, 1, 1);

    game_of_life_kernel<GROUP_WIDTH, BLOCK_Y><<<grid_dim, block_dim>>>(
        input,
        output,
        words_per_row,
        grid_dimensions,
        blocks_per_row_mask,
        blocks_per_row_shift);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    /*
    The board size is guaranteed to be a power of two greater than 512.
    Therefore words_per_row = grid_dimensions / 64 is also a power of two.

    Tuning choice:
      - 1024x1024 boards have exactly 16 words per row, so a 16-lane shuffle width
        avoids wasting half of each warp. One warp then carries two independent rows.
      - All larger boards use a 32-lane shuffle width, one row segment per warp.
    */
    const int words_per_row = grid_dimensions >> 6;

    if (words_per_row == 16) {
        launch_game_of_life_step<16, 16>(input, output, grid_dimensions, words_per_row);
    } else {
        launch_game_of_life_step<32, 8>(input, output, grid_dimensions, words_per_row);
    }

    /* No synchronization here; the caller explicitly owns synchronization. */
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
