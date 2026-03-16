#include <cuda_runtime.h>
#include <cstdint>

using u64 = std::uint64_t;

namespace
{
/*
    Performance-oriented Conway's Game of Life on a bit-packed grid.

    Key design choices:
    - One CUDA thread computes exactly one 64-bit word => 64 cells per thread.
    - No shared memory: for this access pattern on modern data-center GPUs,
      read-only global loads plus warp shuffles are simpler and typically faster.
    - The three center words needed by each thread (north/current/south) are loaded
      from global memory once. Left/right neighboring words are obtained mostly via
      warp shuffles because adjacent threads own adjacent words.
    - Neighbor counts are accumulated in a bit-sliced *saturating* counter:
          ones  : exact bit 0 of the count while count < 4
          twos  : exact bit 1 of the count while count < 4
          ge4   : "count >= 4" saturation flag
      This is sufficient because Game of Life only distinguishes:
          count == 2, count == 3, and everything else.
    - Bit 0 / bit 63 cross-word handling is performed by:
          left-aligned  neighbors: (word << 1) | (left_word  >> 63)
          right-aligned neighbors: (word >> 1) | (right_word << 63)
      This exactly covers the required upper-left / left / lower-left and
      upper-right / right / lower-right cases.
*/

constexpr int kBlockSize = 256;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert((kBlockSize % 32) == 0, "Block size must be a whole number of warps.");

/* Read-only load helper. The input grid is immutable for the duration of the step. */
__device__ __forceinline__ u64 load_ro(const u64* ptr)
{
    return __ldg(ptr);
}

/* Warp-local access to the previous / next thread's word. */
__device__ __forceinline__ u64 shfl_up_u64(const u64 value)
{
    return __shfl_up_sync(kFullWarpMask, value, 1);
}

__device__ __forceinline__ u64 shfl_down_u64(const u64 value)
{
    return __shfl_down_sync(kFullWarpMask, value, 1);
}

/*
    Add one aligned neighbor bitboard into the saturated bit-sliced counter.

    Per bit position, the state encoding is:
        000 -> 0 neighbors
        001 -> 1 neighbor
        010 -> 2 neighbors
        011 -> 3 neighbors
        100 -> 4 or more neighbors (saturated)

    Transition when x = 1:
        000 -> 001
        001 -> 010
        010 -> 011
        011 -> 100
        100 -> 100
*/
__device__ __forceinline__ void add_neighbor_saturated(
    const u64 x,
    u64& ones,
    u64& twos,
    u64& ge4)
{
    const u64 active = x & ~ge4;    // once count >= 4, further increments are irrelevant
    const u64 carry  = ones & active;

    ge4  |= twos & carry;           // only the 3 -> 4 transition sets the saturation flag
    ones ^= active;
    twos ^= carry;
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_step_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const unsigned int words_per_row,
    const unsigned int words_per_row_mask,
    const u64 last_row_start)
{
    /*
        Exact-launch kernel:
        - The host launches exactly one thread per packed 64-bit word.
        - Therefore there is no tail check here.

        Indexing notes:
        - words_per_row is a power of two, so x can be computed with a mask.
        - total word count can exceed 2^31 on large GPUs / grids, so indices are 64-bit.
    */
    const u64 row_stride = static_cast<u64>(words_per_row);
    const u64 idx =
        static_cast<u64>(blockIdx.x) * static_cast<u64>(kBlockSize) +
        static_cast<u64>(threadIdx.x);

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int x = static_cast<unsigned int>(idx & static_cast<u64>(words_per_row_mask));

    const bool has_up    = idx >= row_stride;
    const bool has_down  = idx < last_row_start;
    const bool has_left  = x != 0u;
    const bool has_right = x != words_per_row_mask;

    /*
        These are only dereferenced when the corresponding boundary predicate is true.
        north_idx intentionally wraps on the first row, but is never used there.
    */
    const u64 north_idx = idx - row_stride;
    const u64 south_idx = idx + row_stride;

    /* Center words for north/current/south rows. Missing rows are treated as all-dead. */
    const u64 current = load_ro(input + idx);
    const u64 north   = has_up   ? load_ro(input + north_idx) : u64{0};
    const u64 south   = has_down ? load_ro(input + south_idx) : u64{0};

    /* Saturated bit-sliced neighbor counter. */
    u64 ones = 0;
    u64 twos = 0;
    u64 ge4  = 0;

    /*
        North row contribution.
        The shuffles are unconditional so every lane in the warp executes the same warp
        intrinsic. x/has_left/has_right then decide whether the shuffled value is valid
        for this thread's row segment. This remains correct even for the smallest allowed
        case (words_per_row == 16), where a warp spans two rows.
    */
    const u64 north_from_left  = shfl_up_u64(north);
    const u64 north_from_right = shfl_down_u64(north);

    if (has_up)
    {
        u64 north_left_word  = 0;
        u64 north_right_word = 0;

        if (has_left)
        {
            north_left_word = (lane != 0u) ? north_from_left
                                           : load_ro(input + north_idx - 1u);
        }

        if (has_right)
        {
            north_right_word = (lane != 31u) ? north_from_right
                                             : load_ro(input + north_idx + 1u);
        }

        add_neighbor_saturated((north << 1) | (north_left_word >> 63), ones, twos, ge4);
        add_neighbor_saturated(north,                                      ones, twos, ge4);
        add_neighbor_saturated((north >> 1) | (north_right_word << 63),   ones, twos, ge4);
    }

    /* Current row contribution: only left/right neighbors, never the cell itself. */
    const u64 current_from_left  = shfl_up_u64(current);
    const u64 current_from_right = shfl_down_u64(current);

    u64 current_left_word  = 0;
    u64 current_right_word = 0;

    if (has_left)
    {
        current_left_word = (lane != 0u) ? current_from_left
                                         : load_ro(input + idx - 1u);
    }

    if (has_right)
    {
        current_right_word = (lane != 31u) ? current_from_right
                                           : load_ro(input + idx + 1u);
    }

    add_neighbor_saturated((current << 1) | (current_left_word >> 63),  ones, twos, ge4);
    add_neighbor_saturated((current >> 1) | (current_right_word << 63), ones, twos, ge4);

    /* South row contribution. */
    const u64 south_from_left  = shfl_up_u64(south);
    const u64 south_from_right = shfl_down_u64(south);

    if (has_down)
    {
        u64 south_left_word  = 0;
        u64 south_right_word = 0;

        if (has_left)
        {
            south_left_word = (lane != 0u) ? south_from_left
                                           : load_ro(input + south_idx - 1u);
        }

        if (has_right)
        {
            south_right_word = (lane != 31u) ? south_from_right
                                             : load_ro(input + south_idx + 1u);
        }

        add_neighbor_saturated((south << 1) | (south_left_word >> 63), ones, twos, ge4);
        add_neighbor_saturated(south,                                    ones, twos, ge4);
        add_neighbor_saturated((south >> 1) | (south_right_word << 63), ones, twos, ge4);
    }

    /*
        Game of Life rule:
            next = (neighbors == 3) || (alive && neighbors == 2)

        With the saturated counter:
            neighbors == 2  => twos=1, ones=0, ge4=0
            neighbors == 3  => twos=1, ones=1, ge4=0

        Therefore:
            next = twos & ~ge4 & (ones | alive)
    */
    output[idx] = twos & ~ge4 & (ones | current);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    /*
        The caller provides device pointers allocated by cudaMalloc and handles any
        synchronization. This function only launches the kernel.

        The implementation assumes the standard ping-pong update pattern:
        input and output should be distinct buffers.
    */

    const unsigned int words_per_row = static_cast<unsigned int>(grid_dimensions) >> 6;
    const u64 total_words =
        static_cast<u64>(grid_dimensions) * static_cast<u64>(words_per_row);

    /*
        Since grid_dimensions = 2^n with n >= 10:
            words_per_row = 2^(n-6)
            total_words   = 2^(2n-6)
        Hence total_words is a power of two and, in particular, divisible by 256.
        That lets us launch exactly one thread per packed word with no tail handling.
    */
    const u64 last_row_start = total_words - static_cast<u64>(words_per_row);
    const unsigned int num_blocks =
        static_cast<unsigned int>(total_words / static_cast<u64>(kBlockSize));

    game_of_life_step_kernel<<<num_blocks, kBlockSize>>>(
        input,
        output,
        words_per_row,
        words_per_row - 1u,
        last_row_start);
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
