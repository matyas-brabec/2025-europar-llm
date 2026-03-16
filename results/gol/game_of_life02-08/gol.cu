#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Use an explicit 64-bit unsigned type that CUDA's shuffle intrinsics accept directly.
// The public API still uses std::uint64_t; both have the same size/layout.
using u64 = unsigned long long;

static_assert(sizeof(u64) == sizeof(std::uint64_t), "u64 must match std::uint64_t in size");

// 256 threads/block = 8 warps/block.
// This is a good balance for the bitwise-heavy, register-only kernel on modern data-center GPUs.
constexpr int kBlockSize = 256;
constexpr unsigned int kWarpLanes = 32u;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

static_assert((kBlockSize % kWarpLanes) == 0, "Block size must be a multiple of warp size");

// Read-only global load helper.
// On modern NVIDIA GPUs regular global loads are already well cached, but __ldg still makes the
// read-only intent explicit and is harmless for this workload.
__device__ __forceinline__ u64 ro_load(const u64* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Layout convention used by this kernel:
//   bit k of a 64-bit word represents column (word_x * 64 + k).
// Therefore, after alignment:
//   west-neighbor bitboard = shift current word left by 1 and inject bit 63 from the word on the left
//   east-neighbor bitboard = shift current word right by 1 and inject bit 0  from the word on the right
__device__ __forceinline__ u64 align_west(const u64 word, const u64 left_word) {
    return (word << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 align_east(const u64 word, const u64 right_word) {
    return (word >> 1) | (right_word << 63);
}

// Carry-save adder for bitboards.
// For every bit position independently:
//   a + b + c = sum + 2 * carry
__device__ __forceinline__ void csa(const u64 a, const u64 b, const u64 c, u64& carry, u64& sum) {
    const u64 u = a ^ b;
    sum = u ^ c;
    carry = (a & b) | (u & c);
}

// One-thread-per-word kernel.
//
// Shared memory is intentionally not used here:
// horizontal reuse is handled with warp shuffles, while vertical reuse comes from the cache.
// This keeps the instruction stream short and avoids synchronization.
//
// Template specialization:
//   RowsFitInWarp == true  : words_per_row is 16 or 32. A whole row fits inside one shuffle group
//                            (width = words_per_row), so all valid horizontal neighbor words are
//                            obtained via shuffles and no cross-warp fallback loads are needed.
//   RowsFitInWarp == false : words_per_row is >= 64. Because words_per_row is a power of two, rows
//                            start on warp boundaries; only lane 0 / lane 31 need cross-warp
//                            fallback loads for left/right words.
//
// No bounds guard is needed:
// grid_dimensions is a power of two and > 512, so total_words = N * (N / 64) is also a power of two
// and always divisible by both 32 and kBlockSize (256). The launch therefore creates exactly one
// full thread per input word and every warp is full.
template <bool RowsFitInWarp>
__global__ __launch_bounds__(kBlockSize)
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              const u64 total_words,
                              const unsigned int words_per_row,
                              const unsigned int word_mask) {
    const u64 idx = static_cast<u64>(blockIdx.x) * static_cast<u64>(kBlockSize) +
                    static_cast<u64>(threadIdx.x);

    const u64 row_stride = static_cast<u64>(words_per_row);
    const u64 last_row_start = total_words - row_stride;

    const unsigned int lane = static_cast<unsigned int>(threadIdx.x) & (kWarpLanes - 1u);
    const unsigned int xword = static_cast<unsigned int>(idx) & word_mask;

    const bool has_north = idx >= row_stride;
    const bool has_south = idx < last_row_start;
    const bool has_west  = xword != 0u;
    const bool has_east  = xword != word_mask;

    // When a row fits in a warp, shuffles are restricted to the row width (16 or 32).
    // Otherwise, use the full warp width.
    const int shuffle_width = RowsFitInWarp ? static_cast<int>(words_per_row) : static_cast<int>(kWarpLanes);

    // Vertical center words: above, current, below.
    // Missing rows are treated as all-dead words.
    const u64 center = ro_load(input + idx);
    const u64 north  = has_north ? ro_load(input + idx - row_stride) : 0ull;
    const u64 south  = has_south ? ro_load(input + idx + row_stride) : 0ull;

    // First CSA stage: NW + N + NE
    u64 h0, l0;
    {
        const u64 north_left_shfl  = __shfl_up_sync(kFullWarpMask, north, 1, shuffle_width);
        const u64 north_right_shfl = __shfl_down_sync(kFullWarpMask, north, 1, shuffle_width);

        u64 north_left_word  = 0ull;
        u64 north_right_word = 0ull;

        if (RowsFitInWarp) {
            if (has_west) north_left_word = north_left_shfl;
            if (has_east) north_right_word = north_right_shfl;
        } else {
            if (has_west) {
                north_left_word = (lane != 0u)
                    ? north_left_shfl
                    : (has_north ? ro_load(input + idx - row_stride - 1ull) : 0ull);
            }
            if (has_east) {
                north_right_word = (lane != (kWarpLanes - 1u))
                    ? north_right_shfl
                    : (has_north ? ro_load(input + idx - row_stride + 1ull) : 0ull);
            }
        }

        csa(align_west(north, north_left_word),
            north,
            align_east(north, north_right_word),
            h0, l0);
    }

    // Second CSA stage: SW + S + SE
    u64 h1, l1;
    {
        const u64 south_left_shfl  = __shfl_up_sync(kFullWarpMask, south, 1, shuffle_width);
        const u64 south_right_shfl = __shfl_down_sync(kFullWarpMask, south, 1, shuffle_width);

        u64 south_left_word  = 0ull;
        u64 south_right_word = 0ull;

        if (RowsFitInWarp) {
            if (has_west) south_left_word = south_left_shfl;
            if (has_east) south_right_word = south_right_shfl;
        } else {
            if (has_west) {
                south_left_word = (lane != 0u)
                    ? south_left_shfl
                    : (has_south ? ro_load(input + idx + row_stride - 1ull) : 0ull);
            }
            if (has_east) {
                south_right_word = (lane != (kWarpLanes - 1u))
                    ? south_right_shfl
                    : (has_south ? ro_load(input + idx + row_stride + 1ull) : 0ull);
            }
        }

        csa(align_west(south, south_left_word),
            south,
            align_east(south, south_right_word),
            h1, l1);
    }

    // Third CSA stage: l0 + l1 + W
    // East is kept separate so it can be added as the final 1-bit input.
    u64 h2, l2, east;
    {
        const u64 center_left_shfl  = __shfl_up_sync(kFullWarpMask, center, 1, shuffle_width);
        const u64 center_right_shfl = __shfl_down_sync(kFullWarpMask, center, 1, shuffle_width);

        u64 center_left_word  = 0ull;
        u64 center_right_word = 0ull;

        if (RowsFitInWarp) {
            if (has_west) center_left_word = center_left_shfl;
            if (has_east) center_right_word = center_right_shfl;
        } else {
            if (has_west) {
                center_left_word = (lane != 0u)
                    ? center_left_shfl
                    : ro_load(input + idx - 1ull);
            }
            if (has_east) {
                center_right_word = (lane != (kWarpLanes - 1u))
                    ? center_right_shfl
                    : ro_load(input + idx + 1ull);
            }
        }

        const u64 west = align_west(center, center_left_word);
        east = align_east(center, center_right_word);

        csa(l0, l1, west, h2, l2);
    }

    // Final reduction.
    //
    // Let the 8-neighbor count be:
    //   count = NW + N + NE + W + E + SW + S + SE
    //
    // After the three CSA stages above:
    //   NW + N + NE = l0 + 2*h0
    //   SW + S + SE = l1 + 2*h1
    //   l0 + l1 + W = l2 + 2*h2
    //
    // So before adding E:
    //   count_without_E = l2 + 2*(h0 + h1 + h2)
    //
    // Add E explicitly:
    //   b0     = l2 ^ E       (count bit 0)
    //   carry0 = l2 & E       (a 1-bit carry into the "2s" place)
    //
    // Reduce h0 + h1 + h2:
    //   h0 + h1 + h2 = l3 + 2*h3
    //
    // Then:
    //   count = b0 + 2*(carry0 + l3 + 2*h3)
    //
    // Conway's rule only needs to know whether count is exactly 2 or 3.
    // That happens iff the higher-weight term equals exactly 1:
    //   carry0 + l3 + 2*h3 == 1
    // which is equivalent to:
    //   h3 == 0 and (l3 ^ carry0) == 1
    //
    // Finally:
    //   next = (count == 3) | (alive & (count == 2))
    //        = (b0 | alive) & high_is_one
    u64 h3, l3;
    csa(h0, h1, h2, h3, l3);

    const u64 b0 = l2 ^ east;
    const u64 carry0 = l2 & east;
    const u64 high_is_one = (l3 ^ carry0) & ~h3;

    output[idx] = (b0 | center) & high_is_one;
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The kernel operates directly on the bit-packed representation; no unpacking is needed.
    // The caller owns synchronization and error handling, so this function only launches the kernel.

    const unsigned int dim = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = dim >> 6;                 // grid_dimensions / 64
    const unsigned int word_mask = words_per_row - 1u;
    const u64 total_words = static_cast<u64>(dim) * static_cast<u64>(words_per_row);

    // Flatten the 2D word grid into a 1D launch so very large boards are not limited by gridDim.y.
    // Under the problem constraints total_words is always divisible by kBlockSize exactly.
    const unsigned int blocks = static_cast<unsigned int>(total_words / static_cast<u64>(kBlockSize));

    const u64* in = reinterpret_cast<const u64*>(input);
    u64* out = reinterpret_cast<u64*>(output);

    // Specialize the horizontal neighbor path:
    //   16 or 32 words/row  -> row fits inside one shuffle group
    //   >= 64 words/row     -> cross-warp fallback loads required at lane 0 / lane 31
    if (words_per_row <= kWarpLanes) {
        game_of_life_step_kernel<true><<<blocks, kBlockSize>>>(in, out, total_words, words_per_row, word_mask);
    } else {
        game_of_life_step_kernel<false><<<blocks, kBlockSize>>>(in, out, total_words, words_per_row, word_mask);
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
