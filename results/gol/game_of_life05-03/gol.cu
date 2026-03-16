#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// Storage type mandated by the public interface.
using packed_t = std::uint64_t;

// Internal arithmetic/shuffle type. CUDA warp shuffle support is guaranteed for
// unsigned long long, and the conversion is a no-op on all supported platforms.
using word_t = unsigned long long;

static_assert(sizeof(packed_t) == sizeof(word_t), "Expected 64-bit packed words.");

// One warp in X is intentional: every warp owns 32 consecutive words from one row,
// so horizontal reuse can be handled with warp shuffles instead of shared memory.
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kBlockThreads = kBlockX * kBlockY;

static_assert(kBlockX == 32, "The shuffle path assumes exactly one warp along X.");

// Simple typed load/store helpers so the kernel can use the arithmetic/shuffle type
// without changing the externally visible API type.
__device__ __forceinline__ word_t load_word(const packed_t* ptr) {
    return static_cast<word_t>(*ptr);
}

__device__ __forceinline__ packed_t store_word(word_t v) {
    return static_cast<packed_t>(v);
}

// Bit i in the returned word becomes the western neighbor of bit i in `center`.
// This is exactly where the special handling for bit 0 happens: its west neighbor
// is imported from bit 63 of the word on the left.
__device__ __forceinline__ word_t align_west(word_t center, word_t left_word) {
    return (center << 1) | (left_word >> 63);
}

// Bit i in the returned word becomes the eastern neighbor of bit i in `center`.
// This is exactly where the special handling for bit 63 happens: its east neighbor
// is imported from bit 0 of the word on the right.
__device__ __forceinline__ word_t align_east(word_t center, word_t right_word) {
    return (center >> 1) | (right_word << 63);
}

// Bit-sliced carry for adding three 1-bit planes in parallel across all 64 cells.
__device__ __forceinline__ word_t majority3(word_t a, word_t b, word_t c) {
    return ((a ^ b) & c) | (a & b);
}

// Returns a 1-bit plane whose bit i is set iff exactly one of the four input
// bit-planes has bit i set.
__device__ __forceinline__ word_t exactly_one_of_four(word_t a, word_t b, word_t c, word_t d) {
    const word_t ab = a ^ b;
    const word_t cd = c ^ d;
    return (ab ^ cd) & ~((a & b) | (c & d) | (ab & cd));
}

// Compute the next generation for 64 cells at once.
//
// The 3x3 word neighborhood is supplied explicitly:
//
//   up_left   up_center   up_right
//   mid_left  mid_center  mid_right
//   dn_left   dn_center   dn_right
//
// The eight neighbor planes are first aligned so that bit i of each aligned word
// corresponds to the neighbor of cell i in `mid_center`.
//
// Then the 8-neighbor count is reduced with a tiny bit-sliced adder tree:
//
//   top    = NW + N + NE      -> (top1, top0)   in [0, 3]
//   middle = W + E            -> (mid1, mid0)   in [0, 2]
//   bottom = SW + S + SE      -> (bot1, bot0)   in [0, 3]
//
// Let:
//   low   = parity(top0, mid0, bot0)
//   carry = carry(top0, mid0, bot0)
//   H     = top1 + mid1 + bot1 + carry
//
// The Game of Life rule only needs to distinguish:
//   neighbors == 2  -> survive if currently alive
//   neighbors == 3  -> become/stay alive
//
// Because neighbors = low + 2*H, we only need H == 1.
// `low` then distinguishes 2 from 3.
// Therefore:
//
//   next = (H == 1) & (low | current)
//
// This updates all 64 cells with only Boolean logic, no per-cell loops, and no
// scalar popcount on individual bits.
__device__ __forceinline__ word_t life_step_word(
    word_t up_left,  word_t up_center,  word_t up_right,
    word_t mid_left, word_t mid_center, word_t mid_right,
    word_t dn_left,  word_t dn_center,  word_t dn_right) {

    const word_t nw = align_west(up_center, up_left);
    const word_t n  = up_center;
    const word_t ne = align_east(up_center, up_right);

    const word_t w  = align_west(mid_center, mid_left);
    const word_t e  = align_east(mid_center, mid_right);

    const word_t sw = align_west(dn_center, dn_left);
    const word_t s  = dn_center;
    const word_t se = align_east(dn_center, dn_right);

    const word_t top0 = nw ^ n ^ ne;
    const word_t top1 = majority3(nw, n, ne);

    const word_t mid0 = w ^ e;
    const word_t mid1 = w & e;

    const word_t bot0 = sw ^ s ^ se;
    const word_t bot1 = majority3(sw, s, se);

    const word_t low   = top0 ^ mid0 ^ bot0;
    const word_t carry = majority3(top0, mid0, bot0);

    const word_t high_eq_1 = exactly_one_of_four(top1, mid1, bot1, carry);

    return high_eq_1 & (low | mid_center);
}

// Kernel layout:
//   - blockDim.x == 32, so each warp covers 32 consecutive words from one row.
//   - blockDim.y > 1 to keep enough warps per block for occupancy.
//
// Main optimization:
//   Each thread loads only the three "center column" words:
//       row-1, row, row+1 at its own x position.
//   Left/right words are usually obtained from adjacent lanes via warp shuffles.
//   Only the two warp-edge lanes need cross-warp fallback loads.
//   This cuts the common-case global loads from 9 words/thread to 3 words/thread.
__global__ __launch_bounds__(kBlockThreads)
void game_of_life_kernel(const packed_t* __restrict__ input,
                         packed_t* __restrict__ output,
                         int words_per_row,
                         int grid_dimensions) {
    const int lane = static_cast<int>(threadIdx.x);
    const int col  = static_cast<int>(blockIdx.x) * kBlockX + lane;
    const int row  = static_cast<int>(blockIdx.y) * kBlockY + static_cast<int>(threadIdx.y);

    if (row >= grid_dimensions || col >= words_per_row) {
        return;
    }

    const int last_row = grid_dimensions - 1;
    const int last_col = words_per_row - 1;

    const bool has_up    = (row != 0);
    const bool has_down  = (row != last_row);
    const bool has_left  = (col != 0);
    const bool has_right = (col != last_col);

    const std::size_t idx =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(words_per_row) +
        static_cast<std::size_t>(col);

    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(words_per_row);
    const packed_t* const p = input + idx;

    word_t up_center = 0ull;
    word_t dn_center = 0ull;

    if (has_up) {
        up_center = load_word(p - stride);
    }

    const word_t mid_center = load_word(p);

    if (has_down) {
        dn_center = load_word(p + stride);
    }

    // The active mask is required because the last X block can be a partial warp
    // (for example, 16 words/row when grid_dimensions == 1024).
    const unsigned mask = __activemask();

    const word_t up_from_left  = __shfl_up_sync(mask,   up_center, 1);
    const word_t mid_from_left = __shfl_up_sync(mask,   mid_center, 1);
    const word_t dn_from_left  = __shfl_up_sync(mask,   dn_center, 1);

    const word_t up_from_right  = __shfl_down_sync(mask, up_center, 1);
    const word_t mid_from_right = __shfl_down_sync(mask, mid_center, 1);
    const word_t dn_from_right  = __shfl_down_sync(mask, dn_center, 1);

    // Default case: horizontal neighbors come from adjacent lanes.
    // Warp-edge lanes overwrite with direct cross-warp loads if those words exist.
    const bool left_in_warp  = (lane != 0);
    const bool right_in_warp = (lane != (kBlockX - 1)) && has_right;

    word_t up_left  = left_in_warp  ? up_from_left   : 0ull;
    word_t mid_left = left_in_warp  ? mid_from_left  : 0ull;
    word_t dn_left  = left_in_warp  ? dn_from_left   : 0ull;

    word_t up_right  = right_in_warp ? up_from_right  : 0ull;
    word_t mid_right = right_in_warp ? mid_from_right : 0ull;
    word_t dn_right  = right_in_warp ? dn_from_right  : 0ull;

    // Cross-warp fallback for the left word. Outside-grid cells stay zero.
    if (!left_in_warp && has_left) {
        mid_left = load_word(p - 1);
        if (has_up) {
            up_left = load_word(p - stride - 1);
        }
        if (has_down) {
            dn_left = load_word(p + stride - 1);
        }
    }

    // Cross-warp fallback for the right word. Outside-grid cells stay zero.
    if (!right_in_warp && has_right) {
        mid_right = load_word(p + 1);
        if (has_up) {
            up_right = load_word(p - stride + 1);
        }
        if (has_down) {
            dn_right = load_word(p + stride + 1);
        }
    }

    output[idx] = store_word(life_step_word(
        up_left,  up_center,  up_right,
        mid_left, mid_center, mid_right,
        dn_left,  dn_center,  dn_right));
}

}  // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The prompt guarantees a power-of-two square side length > 512, so the row
    // width in packed 64-bit words is always integral and at least 16.
    const int words_per_row = grid_dimensions >> 6;

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        static_cast<unsigned>((words_per_row   + kBlockX - 1) / kBlockX),
        static_cast<unsigned>((grid_dimensions + kBlockY - 1) / kBlockY));

    // Intentionally fire-and-forget: the caller owns synchronization and error
    // handling/timing policy.
    game_of_life_kernel<<<grid, block>>>(input, output, words_per_row, grid_dimensions);
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
