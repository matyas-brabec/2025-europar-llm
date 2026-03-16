#include <cstdint>
#include <cuda_runtime.h>

using u64 = std::uint64_t;
using u32 = std::uint32_t;

constexpr int kBlockSize = 256;   // 8 warps per CTA.
constexpr u32 kWarpLastLane = 31u;

static_assert((kBlockSize & 31) == 0, "kBlockSize must be a multiple of the warp size (32).");

/*
  One-step Conway's Game of Life for a square, bit-packed board.

  Representation:
    - Row-major.
    - One 64-bit word stores 64 consecutive cells from the same row.
    - Bit 0 is the leftmost cell inside that 64-cell run; bit 63 is the rightmost.
    - All cells outside the board are dead.

  Performance-oriented choices:
    - One CUDA thread updates exactly one 64-bit word at a time, so there are no atomics
      and no write conflicts.
    - The row length in words is a power of two, so the column index is just a bit-mask.
    - Shared/texture memory is intentionally not used. The access pattern is regular, the
      hardware caches are effective, and warp shuffles are enough for the needed neighbor exchange.
    - A scalar rolling-window implementation using __popc/__popcll is a good baseline, but on
      modern data-center GPUs the fastest path for this packed representation is bit-sliced:
      the eight neighbor bitboards are summed with carry-save adders, so all 64 cells in the word
      are evaluated in parallel and the per-bit inner loop disappears entirely.
    - Each thread loads only three words unconditionally: center, north, south. Left/right and
      diagonal neighbor words are normally obtained from adjacent lanes via warp shuffles. Only
      warp-edge lanes fall back to explicit global loads.
*/

static __device__ __forceinline__ u64 shfl_up_u64(const unsigned int mask, const u64 value, const unsigned int delta) {
    return static_cast<u64>(__shfl_up_sync(mask, static_cast<unsigned long long>(value), static_cast<int>(delta)));
}

static __device__ __forceinline__ u64 shfl_down_u64(const unsigned int mask, const u64 value, const unsigned int delta) {
    return static_cast<u64>(__shfl_down_sync(mask, static_cast<unsigned long long>(value), static_cast<int>(delta)));
}

/*
  3:2 compressor (carry-save adder) on bit-planes.

  For every bit position independently:
      a + b + c = sum + 2 * carry

  "carry" here is the next count bit-plane, not a spatial carry to a neighboring cell.
*/
static __device__ __forceinline__ void csa3(const u64 a, const u64 b, const u64 c, u64& sum, u64& carry) {
    sum = a ^ b ^ c;
    carry = (a & b) | (c & (a | b));
}

static __global__ __launch_bounds__(kBlockSize)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         const u64 total_words,
                         const u64 words_per_row,
                         const u64 last_row_start,
                         const u32 words_per_row_mask) {
    const u64 global_thread =
        static_cast<u64>(blockIdx.x) * static_cast<u64>(blockDim.x) +
        static_cast<u64>(threadIdx.x);
    const u64 stride =
        static_cast<u64>(gridDim.x) * static_cast<u64>(blockDim.x);

    const u32 lane = static_cast<u32>(threadIdx.x) & kWarpLastLane;

    for (u64 idx = global_thread; idx < total_words; idx += stride) {
        // Only the column needs a modulo, and because the row length in words is a power of two
        // that modulo is just a mask. Top/bottom checks use the linear index directly.
        const u32 col = static_cast<u32>(idx) & words_per_row_mask;

        const bool has_up    = idx >= words_per_row;
        const bool has_down  = idx < last_row_start;
        const bool has_left  = col != 0u;
        const bool has_right = col != words_per_row_mask;

        const u64 center = input[idx];
        const u64 north  = has_up   ? input[idx - words_per_row] : 0ull;
        const u64 south  = has_down ? input[idx + words_per_row] : 0ull;

        // Align the six neighbor directions to the current word. After a 1-bit shift, only one
        // carried-in boundary bit from the adjacent word matters.
        u64 w  = center << 1;
        u64 e  = center >> 1;
        u64 nw = north  << 1;
        u64 ne = north  >> 1;
        u64 sw = south  << 1;
        u64 se = south  >> 1;

        const unsigned int active = __activemask();

        {
            // Special handling for bit 0: after shifting left, inject bit 63 from the word on the
            // left for the current, north, and south rows.
            const u64 center_from_left = shfl_up_u64(active, center, 1);
            const u64 north_from_left  = shfl_up_u64(active, north,  1);
            const u64 south_from_left  = shfl_up_u64(active, south,  1);

            if (has_left) {
                w |= ((lane != 0u) ? center_from_left : input[idx - 1]) >> 63;

                if (has_up) {
                    nw |= ((lane != 0u) ? north_from_left : input[idx - words_per_row - 1]) >> 63;
                }
                if (has_down) {
                    sw |= ((lane != 0u) ? south_from_left : input[idx + words_per_row - 1]) >> 63;
                }
            }
        }

        {
            // Special handling for bit 63: after shifting right, inject bit 0 from the word on the
            // right for the current, north, and south rows.
            const u64 center_from_right = shfl_down_u64(active, center, 1);
            const u64 north_from_right  = shfl_down_u64(active, north,  1);
            const u64 south_from_right  = shfl_down_u64(active, south,  1);

            if (has_right) {
                e |= (((lane != kWarpLastLane) ? center_from_right : input[idx + 1]) & 1ull) << 63;

                if (has_up) {
                    ne |= (((lane != kWarpLastLane) ? north_from_right : input[idx - words_per_row + 1]) & 1ull) << 63;
                }
                if (has_down) {
                    se |= (((lane != kWarpLastLane) ? south_from_right : input[idx + words_per_row + 1]) & 1ull) << 63;
                }
            }
        }

        // Sum the eight aligned neighbor bitboards with carry-save adders.
        u64 s0, c0;
        u64 s1, c1;
        u64 s2, c2;
        u64 s3, carry4;

        csa3(nw,    north, ne, s0, c0);
        csa3(w,     e,     sw, s1, c1);
        csa3(south, se,    s0, s2, c2);
        csa3(c0,    c1,    c2, s3, carry4);

        // count_bit0 and count_bit1 are the low two bits of the final 0..8 neighbor count.
        // The outstanding weight-4 plane (carry4) distinguishes {2,3} from the higher counts.
        // Therefore:
        //   count_is_2_or_3 = (~carry4) & count_bit1
        //
        // Life's rule:
        //   next = (count == 3) | (center && count == 2)
        //        = count_is_2_or_3 & (count_bit0 | center)
        const u64 count_bit0      = s1 ^ s2;
        const u64 carry_to_bit1   = s1 & s2;
        const u64 count_bit1      = s3 ^ carry_to_bit1;
        const u64 count_is_2_or_3 = (~carry4) & count_bit1;

        output[idx] = count_is_2_or_3 & (count_bit0 | center);
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The caller guarantees:
    //   - a valid power-of-two square board,
    //   - grid_dimensions > 512,
    //   - input/output are device pointers from cudaMalloc,
    //   - input and output are distinct buffers.
    //
    // No synchronization or launch-error polling is done here; the caller owns ordering and any
    // host-device synchronization. The kernel is grid-stride, so we launch only enough CTAs to
    // saturate the device according to the compiled kernel's actual residency.

    const u32 words_per_row_u32 = static_cast<u32>(grid_dimensions) >> 6;
    const u32 words_per_row_mask = words_per_row_u32 - 1u;
    const u64 words_per_row = static_cast<u64>(words_per_row_u32);

    const u64 total_words =
        static_cast<u64>(static_cast<u32>(grid_dimensions)) * words_per_row;
    const u64 last_row_start = total_words - words_per_row;

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    int active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        game_of_life_kernel,
        kBlockSize,
        0);
    if (active_blocks_per_sm < 1) {
        active_blocks_per_sm = 1;
    }

    u64 grid_blocks =
        (total_words + static_cast<u64>(kBlockSize) - 1ull) / static_cast<u64>(kBlockSize);

    const u64 resident_blocks =
        static_cast<u64>(sm_count) * static_cast<u64>(active_blocks_per_sm);

    if (grid_blocks > resident_blocks) {
        grid_blocks = resident_blocks;
    }

    game_of_life_kernel<<<static_cast<unsigned int>(grid_blocks), kBlockSize>>>(
        input,
        output,
        total_words,
        words_per_row,
        last_row_start,
        words_per_row_mask);
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
