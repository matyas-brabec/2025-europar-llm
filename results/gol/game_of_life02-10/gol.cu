#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/*
  High-throughput bit-packed Conway's Game of Life step for modern NVIDIA GPUs.

  Design choices:
  - One thread updates one 64-bit word, i.e. 64 cells at once.
  - blockDim.x is fixed to 32, so each warp covers one contiguous 32-word segment
    of a single grid row.
  - Each lane loads only the word directly above/current/below at its own x.
    Left/right neighboring words inside the segment are exchanged with warp
    shuffles; only segment-edge lanes need extra global loads.
  - The 8-neighbor population count is computed with a bit-sliced boolean adder
    network, so all 64 cells in the word are advanced in parallel with bitwise
    operations only.
  - Shared/texture memory are intentionally not used. After shuffle-based reuse of
    horizontal neighbors, additional staging only adds complexity and pressure.
*/

namespace {

using u64 = std::uint64_t;

constexpr int kWarpLanes = 32;
constexpr int kWarpShift = 5;          // log2(32)
constexpr int kBlockRows = 8;
constexpr int kBlockRowsShift = 3;     // log2(8)
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;
constexpr unsigned int kMaxGridY = 65535u;

static_assert(kWarpLanes == (1 << kWarpShift), "kWarpShift must match kWarpLanes");
static_assert(kBlockRows == (1 << kBlockRowsShift), "kBlockRowsShift must match kBlockRows");

// Read-only input load. On modern GPUs this lets the compiler pick the best
// read-only cache path for the input buffer.
__device__ __forceinline__ u64 load_ro(const u64* __restrict__ ptr) {
    return __ldg(ptr);
}

// Majority-of-three on bitboards. NVCC lowers this boolean form well on Ampere/Hopper.
__device__ __forceinline__ u64 majority3(const u64 a, const u64 b, const u64 c) {
    return (a & b) | (a & c) | (b & c);
}

// Each warp processes one 32-word span of one row. The center word for this lane
// is already in a register; this helper fetches horizontally adjacent words.
// Inside the span, shuffles provide neighbors. Only span-edge lanes perform the
// fallback global loads needed to cross warp-segment boundaries.
__device__ __forceinline__ void gather_lr(
    const u64 center,
    u64& left,
    u64& right,
    const unsigned active_mask,
    const u64* __restrict__ center_ptr,
    const bool has_left_word,
    const bool has_right_word,
    const int lane)
{
    left  = __shfl_up_sync(active_mask, center, 1);
    right = __shfl_down_sync(active_mask, center, 1);

    if (lane == 0) {
        left = has_left_word ? load_ro(center_ptr - 1) : 0ULL;
    }
    if (lane == kWarpLanes - 1 || !has_right_word) {
        right = has_right_word ? load_ro(center_ptr + 1) : 0ULL;
    }
}

// Bit i of a word encodes cell x = word_index * 64 + i.
// With that convention, west neighbors are formed with a left shift and east
// neighbors with a right shift. No tail mask is needed because the problem
// guarantees grid_dimensions is a power of two, hence a multiple of 64.
__global__ __launch_bounds__(kWarpLanes * kBlockRows)
void game_of_life_kernel(
    const u64* __restrict__ input,
    u64* __restrict__ output,
    const int grid_dimensions,
    const int row_shift)
{
    const std::size_t dim = static_cast<std::size_t>(grid_dimensions);
    const std::size_t words_per_row = std::size_t{1} << row_shift;

    const int lane = static_cast<int>(threadIdx.x);

    const std::size_t word =
        (static_cast<std::size_t>(blockIdx.x) << kWarpShift) +
        static_cast<std::size_t>(lane);

    // grid.y is capped at 65,535; grid.z extends the row dimension when needed.
    const std::size_t row_tile =
        static_cast<std::size_t>(blockIdx.z) * static_cast<std::size_t>(gridDim.y) +
        static_cast<std::size_t>(blockIdx.y);
    const std::size_t row =
        (row_tile << kBlockRowsShift) + static_cast<std::size_t>(threadIdx.y);

    const bool active = (row < dim) && (word < words_per_row);
    const unsigned active_mask = __ballot_sync(kFullWarpMask, active);
    if (!active) {
        return;
    }

    // words_per_row is a power of two, so row_base is a shift rather than a 64-bit multiply.
    const std::size_t row_base = row << row_shift;
    const std::size_t index = row_base + word;

    const bool has_prev = (row != 0);
    const bool has_next = ((row + 1) < dim);
    const bool has_left_word = (word != 0);
    const bool has_right_word = ((word + 1) < words_per_row);

    const u64* const center_ptr = input + index;
    const u64* const prev_ptr = has_prev ? (center_ptr - words_per_row) : nullptr;
    const u64* const next_ptr = has_next ? (center_ptr + words_per_row) : nullptr;

    u64 left, right;

    // Sum the three cells from the row above into a 2-bit bit-sliced number:
    // north = north_lo + 2*north_hi
    u64 north_lo = 0ULL;
    u64 north_hi = 0ULL;
    if (has_prev) {
        const u64 north = load_ro(prev_ptr);
        gather_lr(north, left, right, active_mask, prev_ptr, has_left_word, has_right_word, lane);

        const u64 nw = (north << 1) | (left >> 63);
        const u64 ne = (north >> 1) | (right << 63);

        north_lo = nw ^ north ^ ne;
        north_hi = majority3(nw, north, ne);
    }

    // Sum the two horizontal neighbors from the current row:
    // horiz = horiz_lo + 2*horiz_hi
    const u64 self = load_ro(center_ptr);
    gather_lr(self, left, right, active_mask, center_ptr, has_left_word, has_right_word, lane);

    const u64 west = (self << 1) | (left >> 63);
    const u64 east = (self >> 1) | (right << 63);

    const u64 horiz_lo = west ^ east;
    const u64 horiz_hi = west & east;

    // Sum the three cells from the row below into:
    // south = south_lo + 2*south_hi
    u64 south_lo = 0ULL;
    u64 south_hi = 0ULL;
    if (has_next) {
        const u64 south = load_ro(next_ptr);
        gather_lr(south, left, right, active_mask, next_ptr, has_left_word, has_right_word, lane);

        const u64 sw = (south << 1) | (left >> 63);
        const u64 se = (south >> 1) | (right << 63);

        south_lo = sw ^ south ^ se;
        south_hi = majority3(sw, south, se);
    }

    // Bit-sliced population count of the eight neighbors:
    //   north = north_lo + 2*north_hi
    //   horiz = horiz_lo + 2*horiz_hi
    //   south = south_lo + 2*south_hi
    const u64 count0 = north_lo ^ south_lo ^ horiz_lo;
    const u64 carry0 = majority3(north_lo, south_lo, horiz_lo);

    const u64 high_lo = north_hi ^ south_hi ^ horiz_hi;
    const u64 high_hi = majority3(north_hi, south_hi, horiz_hi);

    // count1 is the 2's bit of the final neighbor count.
    // count_ge_4 flags counts 4..8, which are all deaths in Life.
    const u64 count1 = high_lo ^ carry0;
    const u64 count_ge_4 = high_hi | (high_lo & carry0);

    // Conway rule in bit-sliced form:
    //   next = (count == 3) | (self & count == 2)
    //        = (~count_ge_4) & count1 & (count0 | self)
    const u64 count_2_or_3 = (~count_ge_4) & count1;
    output[index] = count_2_or_3 & (count0 | self);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Problem guarantees:
    // - square grid
    // - grid_dimensions is a power of two
    // - grid_dimensions > 512
    // - input/output are already device pointers from cudaMalloc
    //
    // The caller handles synchronization and error observation; this function
    // only enqueues the kernel.

    const std::size_t words_per_row = static_cast<std::size_t>(grid_dimensions) >> 6;
    const std::size_t word_tiles = (words_per_row + (kWarpLanes - 1)) >> kWarpShift;
    const std::size_t row_tiles = static_cast<std::size_t>(grid_dimensions) >> kBlockRowsShift;

    // words_per_row is a power of two, so row_shift = log2(words_per_row).
    // This host-side setup is not on the hot path.
    int row_shift = 0;
    for (std::size_t w = words_per_row; w > 1; w >>= 1) {
        ++row_shift;
    }

    const unsigned int grid_y =
        static_cast<unsigned int>(row_tiles > kMaxGridY ? kMaxGridY : row_tiles);
    const unsigned int grid_z =
        static_cast<unsigned int>(
            (row_tiles + static_cast<std::size_t>(grid_y) - 1) /
            static_cast<std::size_t>(grid_y));

    const dim3 block(kWarpLanes, kBlockRows, 1);
    const dim3 grid(static_cast<unsigned int>(word_tiles), grid_y, grid_z);

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions, row_shift);
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
