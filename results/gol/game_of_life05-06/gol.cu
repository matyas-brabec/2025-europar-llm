#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// High-performance, bit-parallel Conway's Game of Life.
//
// Design choices tuned for modern NVIDIA data-center GPUs:
//   - One CUDA thread updates one packed 64-bit word (64 cells), so no atomics are needed.
//   - No shared memory: the only horizontal cross-word dependency is one bit wide after a 1-bit shift,
//     so shuffles are cheaper and simpler.
//   - Eight neighbor masks are counted with a bit-sliced carry-save adder tree, processing all
//     64 cells in parallel instead of iterating cell-by-cell.
//   - A 1D grid avoids the 65535 limit on grid.y for very large boards that still fit in memory.

namespace {

using word_t = std::uint64_t;

constexpr unsigned int kFullMask = 0xffffffffu;
constexpr int kBlockY = 8;

static_assert(sizeof(word_t) == 8, "Expected 64-bit packed words.");

struct SumCarry {
    word_t sum;
    word_t carry;
};

__device__ __forceinline__ SumCarry csa3(const word_t a, const word_t b, const word_t c) {
    // Bit-sliced carry-save adder:
    //   a + b + c = sum + 2 * carry
    // independently for all 64 bit positions.
    const word_t ab_xor = a ^ b;
    return {ab_xor ^ c, (a & b) | (ab_xor & c)};
}

__device__ __forceinline__ word_t exactly_one_of_four(
    const word_t a, const word_t b, const word_t c, const word_t d) {
    // True per bit position iff exactly one of {a,b,c,d} is set there.
    const word_t p0 = a ^ b;
    const word_t p1 = c ^ d;
    return (p0 ^ p1) & ~((a & b) | (c & d) | (p0 & p1));
}

template <int TILE_X>
__device__ __forceinline__ void shifted_row_neighbors(
    const word_t row,
    const word_t* __restrict__ input,
    const std::size_t row_idx,
    const bool has_left_word,
    const bool has_right_word,
    const unsigned int lane,
    word_t& west,
    word_t& east) {
    static_assert(TILE_X == 16 || TILE_X == 32, "Unsupported tile width.");

    // Bit i represents the i-th cell inside this 64-cell word span.
    // To align the west neighbor (x-1) with x, the row shifts left.
    // To align the east neighbor (x+1) with x, the row shifts right.
    //
    // After the shift, only one cross-word bit is missing:
    //   west: bit 63 from the word to the left -> injected into bit 0
    //   east: bit 0  from the word to the right -> injected into bit 63
    //
    // Most threads obtain that bit from the adjacent thread via a 16- or 32-lane shuffle.
    // Only subgroup-edge threads fall back to one global load.
    unsigned int left_msb =
        __shfl_up_sync(kFullMask, static_cast<unsigned int>(row >> 63), 1, TILE_X);
    unsigned int right_lsb =
        __shfl_down_sync(kFullMask, static_cast<unsigned int>(row & word_t{1}), 1, TILE_X);

    if (lane == 0u) {
        left_msb = has_left_word ? static_cast<unsigned int>(input[row_idx - 1] >> 63) : 0u;
    }
    if (lane == static_cast<unsigned int>(TILE_X - 1)) {
        right_lsb = has_right_word ? static_cast<unsigned int>(input[row_idx + 1] & word_t{1}) : 0u;
    }

    west = (row << 1) | static_cast<word_t>(left_msb);
    east = (row >> 1) | (static_cast<word_t>(right_lsb) << 63);
}

template <int TILE_X>
__global__ __launch_bounds__(TILE_X * kBlockY)
void game_of_life_kernel(
    const word_t* __restrict__ input,
    word_t* __restrict__ output,
    const int words_per_row,
    const int grid_dimensions,
    const unsigned int tiles_per_row_mask,
    const int tiles_per_row_shift) {
    static_assert(TILE_X == 16 || TILE_X == 32, "Unsupported tile width.");

    const unsigned int words = static_cast<unsigned int>(words_per_row);
    const unsigned int board_dim = static_cast<unsigned int>(grid_dimensions);

    // The board width in packed words is a power of two, so the number of row tiles is also
    // a power of two. That lets us recover (tile_x, tile_y) from a single linear block index
    // using a cheap mask+shift instead of division/modulo.
    const unsigned int linear_block = blockIdx.x;
    const unsigned int tile_x = linear_block & tiles_per_row_mask;
    const unsigned int tile_y = linear_block >> tiles_per_row_shift;

    // blockDim.x == TILE_X, so threadIdx.x is already the lane inside the 16- or 32-thread row subgroup.
    const unsigned int x = tile_x * TILE_X + threadIdx.x;
    const unsigned int y = tile_y * kBlockY + threadIdx.y;
    const unsigned int lane = threadIdx.x;

    const std::size_t stride = static_cast<std::size_t>(words);
    const std::size_t idx = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);

    // Exact tiling is possible because the problem guarantees power-of-two dimensions.
    const bool has_n = (y != 0u);
    const bool has_s = (y + 1u != board_dim);
    const bool has_l = (x != 0u);
    const bool has_r = (x + 1u != words);

    const word_t center = input[idx];
    const word_t north_row = has_n ? input[idx - stride] : word_t{0};
    const word_t south_row = has_s ? input[idx + stride] : word_t{0};

    const std::size_t north_idx = has_n ? (idx - stride) : idx;
    const std::size_t south_idx = has_s ? (idx + stride) : idx;

    word_t west_c, east_c;
    word_t west_n, east_n;
    word_t west_s, east_s;

    // Build aligned neighbor masks for the three relevant rows:
    //   north: NW, N, NE
    //   center: W, E
    //   south: SW, S, SE
    shifted_row_neighbors<TILE_X>(center,    input, idx,       has_l,           has_r,           lane, west_c, east_c);
    shifted_row_neighbors<TILE_X>(north_row, input, north_idx, has_n && has_l,  has_n && has_r,  lane, west_n, east_n);
    shifted_row_neighbors<TILE_X>(south_row, input, south_idx, has_s && has_l,  has_s && has_r,  lane, west_s, east_s);

    // Carry-save reduction of the eight 1-bit neighbor masks.
    //
    // The arrangement below produces:
    //   neighbor_count = b0 + 2*q
    // where q is the population count of four carry masks.
    //
    // Conway's rule needs only count == 2 or count == 3, so we never materialize the full 0..8 count.
    // We only test q == 1:
    //   q == 1 and b0 == 0  -> exactly 2 neighbors
    //   q == 1 and b0 == 1  -> exactly 3 neighbors
    const SumCarry north_acc  = csa3(west_n, north_row, east_n);     // NW + N + NE
    const SumCarry south_acc  = csa3(west_s, south_row, east_s);     // SW + S + SE
    const SumCarry middle_acc = csa3(west_c, east_c, north_acc.sum); // W + E + north_sum

    const word_t b0 = middle_acc.sum ^ south_acc.sum;
    const word_t d1 = middle_acc.sum & south_acc.sum;
    const word_t q_eq_1 =
        exactly_one_of_four(north_acc.carry, middle_acc.carry, south_acc.carry, d1);

    // If q == 1 then neighbor_count == 2 + b0:
    //   b0 == 0 -> survive only if currently alive
    //   b0 == 1 -> birth/survival unconditionally
    output[idx] = q_eq_1 & (b0 | center);
}

inline int log2_pow2(unsigned int v) {
    // v is guaranteed to be a power of two.
    int shift = 0;
    while (v > 1u) {
        v >>= 1u;
        ++shift;
    }
    return shift;
}

template <int TILE_X>
inline void launch_game_of_life_kernel(
    const word_t* input,
    word_t* output,
    const int words_per_row,
    const int grid_dimensions) {
    static_assert(TILE_X == 16 || TILE_X == 32, "Unsupported tile width.");

    const unsigned int tiles_per_row = static_cast<unsigned int>(words_per_row / TILE_X);
    const unsigned int blocks_y = static_cast<unsigned int>(grid_dimensions / kBlockY);
    const unsigned int total_blocks = tiles_per_row * blocks_y;

    // Row-major layout in tile space.
    const unsigned int tiles_per_row_mask = tiles_per_row - 1u;
    const int tiles_per_row_shift = log2_pow2(tiles_per_row);

    const dim3 block(TILE_X, kBlockY);
    const dim3 grid(total_blocks);

    // input/output are assumed to be distinct device buffers (double buffering),
    // and synchronization is intentionally left to the caller.
    game_of_life_kernel<TILE_X><<<grid, block>>>(
        input,
        output,
        words_per_row,
        grid_dimensions,
        tiles_per_row_mask,
        tiles_per_row_shift);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6;

    // The only legal board with fewer than 32 packed words per row is 1024x1024,
    // which has exactly 16 packed words per row. Specializing that case avoids
    // half-empty warp rows and keeps shuffle subgroups full.
    if (words_per_row == 16) {
        launch_game_of_life_kernel<16>(input, output, words_per_row, grid_dimensions);
    } else {
        launch_game_of_life_kernel<32>(input, output, words_per_row, grid_dimensions);
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
