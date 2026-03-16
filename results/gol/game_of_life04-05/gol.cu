#include <cuda_runtime.h>
#include <cstdint>

// One-step Conway's Game of Life for a bit-packed square grid on the GPU.
// Row-major layout:
//   - each 64-bit word stores 64 consecutive cells from one row
//   - bit 0 is the first cell in the word
//   - bit 63 is the last cell in the word
//
// One thread updates exactly one 64-bit word, so no atomics are needed.

namespace {

using word_t = std::uint64_t;
using reg_t  = unsigned long long;

constexpr int kBlockSize = 256;
constexpr unsigned int kWarpSize = 32u;
constexpr unsigned int kWarpMask = 0xFFFFFFFFu;

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a multiple of the warp size.");
static_assert(sizeof(word_t) == sizeof(reg_t), "This implementation requires 64-bit words.");

__device__ __forceinline__ void csa(reg_t& carry, reg_t& sum,
                                    const reg_t a, const reg_t b, const reg_t c) {
    // Carry-save adder for three 1-bit-per-lane inputs:
    //   a + b + c = sum + 2 * carry
    const reg_t u = a ^ b;
    carry = (a & b) | (u & c);
    sum = u ^ c;
}

__global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const word_t* __restrict__ input,
                         word_t* __restrict__ output,
                         const unsigned int words_per_row,
                         const std::uint64_t last_row_start) {
    const std::uint64_t idx =
        static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(kBlockSize) +
        static_cast<std::uint64_t>(threadIdx.x);

    // words_per_row is a power of two, so x can be recovered with a mask instead of a modulo.
    const unsigned int last_x = words_per_row - 1u;
    const unsigned int x = static_cast<unsigned int>(idx) & last_x;
    const unsigned int lane = threadIdx.x & (kWarpSize - 1u);

    const bool has_up    = idx >= words_per_row;
    const bool has_down  = idx < last_row_start;
    const bool has_left  = x != 0u;
    const bool has_right = x != last_x;

    const word_t* const center = input + idx;

    // Shared memory is intentionally avoided. The profitable reuse here is horizontal reuse of
    // neighboring words inside a warp, and warp shuffles recover that reuse more cheaply than
    // staging tiles in shared memory.
    //
    // Every thread always loads only the vertically aligned words (above/current/below).
    // Immediate left/right words are taken from adjacent lanes when possible, with a single
    // fallback global load only at warp boundaries.
    const reg_t mc = static_cast<reg_t>(*center);

    reg_t uc = 0ull;
    reg_t dc = 0ull;
    if (has_up) {
        uc = static_cast<reg_t>(*(center - words_per_row));
    }
    if (has_down) {
        dc = static_cast<reg_t>(*(center + words_per_row));
    }

    const reg_t uc_l = __shfl_up_sync(kWarpMask, uc, 1);
    const reg_t mc_l = __shfl_up_sync(kWarpMask, mc, 1);
    const reg_t dc_l = __shfl_up_sync(kWarpMask, dc, 1);

    const reg_t uc_r = __shfl_down_sync(kWarpMask, uc, 1);
    const reg_t mc_r = __shfl_down_sync(kWarpMask, mc, 1);
    const reg_t dc_r = __shfl_down_sync(kWarpMask, dc, 1);

    // Missing neighbor words are zero, which directly implements "cells outside the grid are dead".
    reg_t ul = 0ull, ml = 0ull, dl = 0ull;
    if (has_left) {
        if (lane != 0u) {
            ul = uc_l;
            ml = mc_l;
            dl = dc_l;
        } else {
            ml = static_cast<reg_t>(*(center - 1));
            if (has_up) {
                ul = static_cast<reg_t>(*(center - words_per_row - 1u));
            }
            if (has_down) {
                dl = static_cast<reg_t>(*(center + words_per_row - 1u));
            }
        }
    }

    reg_t ur = 0ull, mr = 0ull, dr = 0ull;
    if (has_right) {
        if (lane != kWarpSize - 1u) {
            ur = uc_r;
            mr = mc_r;
            dr = dc_r;
        } else {
            mr = static_cast<reg_t>(*(center + 1));
            if (has_up) {
                ur = static_cast<reg_t>(*(center - words_per_row + 1u));
            }
            if (has_down) {
                dr = static_cast<reg_t>(*(center + words_per_row + 1u));
            }
        }
    }

    // Align neighbor bits so that bit i in every mask refers to the neighborhood of cell i.
    // This is where the special handling for bit 0 and bit 63 happens:
    //   - bit 0 pulls bit 63 from the word on the left   via (left_word  >> 63)
    //   - bit 63 pulls bit 0 from the word on the right via (right_word << 63)
    const reg_t up_left    = (uc << 1) | (ul >> 63);
    const reg_t up_right   = (uc >> 1) | (ur << 63);
    const reg_t mid_left   = (mc << 1) | (ml >> 63);
    const reg_t mid_right  = (mc >> 1) | (mr << 63);
    const reg_t down_left  = (dc << 1) | (dl >> 63);
    const reg_t down_right = (dc >> 1) | (dr << 63);

    // The prompt mentions __popc as the fast scalar way to evaluate a 3x3 neighborhood.
    // That is a good improvement over naive masking, but for the required "one thread owns one
    // 64-bit word" mapping, the fastest strategy on modern NVIDIA data-center GPUs is a bit-sliced
    // population count:
    //   - the eight neighbor bitboards are added with a carry-save adder tree
    //   - all 64 per-cell neighbor counts are computed in parallel
    reg_t h0, l0, h1, l1, h2, l2, h4, l4;
    csa(h0, l0, up_left,  uc,        up_right);
    csa(h1, l1, mid_left, mid_right, down_left);
    csa(h2, l2, l0,       l1,        dc);

    const reg_t ones = l2 ^ down_right;
    const reg_t h3   = l2 & down_right;

    csa(h4, l4, h0, h1, h2);

    const reg_t twos   = l4 ^ h3;
    const reg_t higher = h4 | (l4 & h3);

    // Life rule in bit-sliced form:
    //   count == 2 -> survive iff currently alive
    //   count == 3 -> always alive
    const reg_t next = twos & ~higher & (ones | mc);

    output[idx] = static_cast<word_t>(next);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const unsigned int dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = dim >> 6;
    const std::uint64_t total_words =
        static_cast<std::uint64_t>(dim) * static_cast<std::uint64_t>(words_per_row);
    const std::uint64_t last_row_start = total_words - words_per_row;

    // Given the stated constraints:
    //   - dim is a power of two and > 512
    //   - words_per_row is therefore a power of two
    //   - total_words = dim * words_per_row is a power of two >= 2^14
    //   - total_words is divisible by kBlockSize
    //
    // Because kBlockSize is also a multiple of 32, every launched warp is full, so the kernel can
    // use a fixed full-warp shuffle mask and omit a tail guard entirely.
    const unsigned int block_count = static_cast<unsigned int>(total_words / kBlockSize);

    // Intentionally enqueue only; synchronization and any error-handling policy are owned by the caller.
    game_of_life_kernel<<<block_count, kBlockSize>>>(input, output, words_per_row, last_row_start);
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
