#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockThreads = 256;

// Use an intrinsic-friendly 64-bit register type for shuffles and bitwise arithmetic.
// The public API still uses std::uint64_t exactly as requested.
using u64 = unsigned long long;
static_assert(sizeof(u64) == sizeof(std::uint64_t), "Unexpected 64-bit type size");

// Plain cached global load. On modern data-center GPUs (A100/H100), regular read-only
// global loads are already well cached; the important optimization here is eliminating
// redundant neighbor loads with warp shuffles, not forcing a legacy __ldg path.
__device__ __forceinline__ u64 load64(const std::uint64_t* ptr) {
    return static_cast<u64>(*ptr);
}

// 3-input majority, bitwise and lane-parallel across all 64 cells in the word.
// For each bit position i, the result bit is 1 iff at least two of {a,b,c} have bit i set.
__device__ __forceinline__ u64 majority3(u64 a, u64 b, u64 c) {
    return (a & b) | (c & (a | b));
}

// 4-input exact-1 detector, again bitwise across all 64 cells.
// For each bit position i, the result bit is 1 iff exactly one of {a,b,c,d} has bit i set.
__device__ __forceinline__ u64 exactly_one_of_four(u64 a, u64 b, u64 c, u64 d) {
    const u64 ab_xor = a ^ b;
    const u64 cd_xor = c ^ d;
    return (ab_xor ^ cd_xor) & ~((a & b) | (c & d) | (ab_xor & cd_xor));
}

/*
  One thread updates one 64-bit word (64 cells).

  Performance-critical choices:
    - No shared memory: it only adds address arithmetic and synchronization here.
    - No texture memory: regular global loads plus cache are enough for this stencil.
    - Same-row neighbor words are exchanged with warp shuffles, so the common case loads
      only {center, north, south}. Only the first/last lane of each row fragment falls
      back to a global load for the left/right neighbor word.
    - The neighborhood count is computed with a bit-sliced adder network, not 64 scalar
      neighborhood extractions and 64 scalar popcounts. A scalar __popc-based approach is
      better than naive masking, but a bit-parallel adder is faster still because it updates
      all 64 cells in the word at once.

  Bit-sliced population count:
    top    = NW + N + NE  -> t0 + 2*t1
    middle =  W + E       -> m0 + 2*m1
    bottom = SW + S + SE  -> b0 + 2*b1

    t0 + m0 + b0 = l0 + 2*l1
    neighbors    = l0 + 2*(l1 + t1 + m1 + b1)

  Only counts 2 and 3 matter:
    - if (l1 + t1 + m1 + b1) != 1, the next state is 0
    - otherwise neighbors = 2 + l0
        l0 == 0 -> exactly 2 neighbors -> survive iff current cell is alive
        l0 == 1 -> exactly 3 neighbors -> cell is alive next generation

  Therefore:
    next = exact1(l1, t1, m1, b1) & (l0 | center)
*/
template <int SUBWARP>
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         unsigned int words_per_row,
                         unsigned int grid_dimensions,
                         unsigned int block_rows_total) {
    static_assert(SUBWARP == 8 || SUBWARP == 16 || SUBWARP == 32, "Unsupported subwarp width");
    static_assert(kBlockThreads % SUBWARP == 0, "Invalid block geometry");
    constexpr int ROWS_PER_BLOCK = kBlockThreads / SUBWARP;

    // A 3D launch is used only to bypass the 65535 limit on gridDim.y for very large boards.
    const std::size_t block_row =
        static_cast<std::size_t>(blockIdx.y) +
        static_cast<std::size_t>(blockIdx.z) * static_cast<std::size_t>(gridDim.y);
    if (block_row >= static_cast<std::size_t>(block_rows_total)) {
        return;
    }

    // Under the stated constraints, words_per_row is a power of two and SUBWARP divides it
    // exactly, so x is always in range for valid inputs and there is no x-bounds check here.
    const unsigned int x = blockIdx.x * SUBWARP + threadIdx.x;
    const unsigned int y = static_cast<unsigned int>(block_row) * ROWS_PER_BLOCK + threadIdx.y;
    if (y >= grid_dimensions) {
        return;
    }

    const std::size_t stride = static_cast<std::size_t>(words_per_row);
    const std::size_t idx = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);

    const bool row_left_edge = (x == 0);
    const bool row_right_edge = (x + 1u == words_per_row);
    const bool has_up = (y != 0);
    const bool has_down = (y + 1u < grid_dimensions);
    const bool segment_left_edge = (threadIdx.x == 0);
    const bool segment_right_edge = (threadIdx.x == SUBWARP - 1);

    const u64 center = load64(input + idx);

    u64 north = 0ull;
    u64 south = 0ull;
    if (has_up) {
        north = load64(input + idx - stride);
    }
    if (has_down) {
        south = load64(input + idx + stride);
    }

    // Using the active mask makes the kernel robust even if a future caller violates the
    // exact divisibility assumptions and creates partial warps in the last tile.
    const unsigned int active = __activemask();

    // Middle row: W + E
    u64 left_word = __shfl_up_sync(active, center, 1, SUBWARP);
    if (segment_left_edge) {
        left_word = row_left_edge ? 0ull : load64(input + idx - 1);
    }
    const u64 west = (center << 1) | (left_word >> 63);

    u64 right_word = __shfl_down_sync(active, center, 1, SUBWARP);
    if (segment_right_edge) {
        right_word = row_right_edge ? 0ull : load64(input + idx + 1);
    }
    const u64 east = (center >> 1) | (right_word << 63);

    const u64 m0 = west ^ east;
    const u64 m1 = west & east;

    // Top row: NW + N + NE
    left_word = __shfl_up_sync(active, north, 1, SUBWARP);
    if (segment_left_edge) {
        left_word = (has_up && !row_left_edge) ? load64(input + idx - stride - 1) : 0ull;
    }
    const u64 northwest = (north << 1) | (left_word >> 63);

    right_word = __shfl_down_sync(active, north, 1, SUBWARP);
    if (segment_right_edge) {
        right_word = (has_up && !row_right_edge) ? load64(input + idx - stride + 1) : 0ull;
    }
    const u64 northeast = (north >> 1) | (right_word << 63);

    const u64 t0 = northwest ^ north ^ northeast;
    const u64 t1 = majority3(northwest, north, northeast);

    // Bottom row: SW + S + SE
    left_word = __shfl_up_sync(active, south, 1, SUBWARP);
    if (segment_left_edge) {
        left_word = (has_down && !row_left_edge) ? load64(input + idx + stride - 1) : 0ull;
    }
    const u64 southwest = (south << 1) | (left_word >> 63);

    right_word = __shfl_down_sync(active, south, 1, SUBWARP);
    if (segment_right_edge) {
        right_word = (has_down && !row_right_edge) ? load64(input + idx + stride + 1) : 0ull;
    }
    const u64 southeast = (south >> 1) | (right_word << 63);

    const u64 b0 = southwest ^ south ^ southeast;
    const u64 b1 = majority3(southwest, south, southeast);

    // neighbors = l0 + 2*(l1 + t1 + m1 + b1)
    const u64 l0 = t0 ^ m0 ^ b0;
    const u64 l1 = majority3(t0, m0, b0);

    const u64 next = exactly_one_of_four(l1, t1, m1, b1) & (l0 | center);
    output[idx] = static_cast<std::uint64_t>(next);
}

template <int SUBWARP>
inline void launch_game_of_life(const std::uint64_t* input,
                                std::uint64_t* output,
                                unsigned int grid_dimensions) {
    constexpr unsigned int ROWS_PER_BLOCK = kBlockThreads / SUBWARP;
    constexpr unsigned int kMaxGridY = 65535u;

    const unsigned int words_per_row = grid_dimensions >> 6;
    const unsigned int block_rows_total = (grid_dimensions + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;

    // Choose a balanced (y,z) factorization so grid.y never exceeds 65535 while also avoiding
    // the pathological case where the last z-slice would contain almost entirely empty blocks.
    const unsigned int grid_z = (block_rows_total + kMaxGridY - 1u) / kMaxGridY;
    const unsigned int grid_y = (block_rows_total + grid_z - 1u) / grid_z;

    const dim3 block(static_cast<unsigned int>(SUBWARP), ROWS_PER_BLOCK, 1u);
    const dim3 grid(words_per_row / SUBWARP, grid_y, grid_z);

    game_of_life_kernel<SUBWARP><<<grid, block>>>(input, output, words_per_row, grid_dimensions, block_rows_total);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Per the prompt, the caller owns synchronization. This function only launches the kernel.
    // Input and output are assumed to be distinct device buffers (standard double-buffered Life).
    if (grid_dimensions <= 0) {
        return;
    }

    const unsigned int dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = dim >> 6;

    // Under the stated constraints (>512 and power of two), the only valid cases are:
    //   1024x1024 -> 16 words/row -> SUBWARP = 16
    //   >=2048x2048 -> >=32 words/row -> SUBWARP = 32
    // An 8-word case is kept so the code still works if reused for a 512x512 board.
    if (words_per_row == 8u) {
        launch_game_of_life<8>(input, output, dim);
    } else if (words_per_row == 16u) {
        launch_game_of_life<16>(input, output, dim);
    } else {
        launch_game_of_life<32>(input, output, dim);
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
