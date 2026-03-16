#include <cstdint>
#include <cuda_runtime.h>

using u64 = std::uint64_t;
constexpr int kThreadsPerBlock = 256;

namespace {

// One thread owns one 64-bit word, exactly as requested.
// Layout is row-major: each row contains grid_dimensions / 64 words, and each word packs
// 64 consecutive cells from that row.
//
// A straightforward optimized baseline would iterate over the 64 bit positions in the word and
// use masked __popc/__popcll neighborhood counts. That is much better than scalar bit tests,
// but it still performs 64 neighborhood counts per thread. This kernel keeps the same
// thread-to-word mapping and instead uses bit-sliced boolean adders so that all 64 cells in
// the word are advanced in parallel.
//
// No shared memory or texture memory is used on purpose: the access pattern is regular,
// neighbor reuse is handled well by the hardware caches on modern GPUs, and avoiding explicit
// staging keeps the kernel lean. Every thread writes one unique output word, so no atomics
// are required.

// Majority bit for three 64-bit bitboards.
// On recent NVIDIA GPUs the compiler lowers this kind of boolean expression efficiently.
__device__ __forceinline__ u64 majority64(const u64 a, const u64 b, const u64 c) {
    return (a & b) | (a & c) | (b & c);
}

// Bit layout matches the prompt:
//   bit 0  is the left-most cell inside the 64-bit word,
//   bit 63 is the right-most cell.
//
// Therefore:
//   west-neighbor bits are aligned with (center << 1) | (left >> 63)
//   east-neighbor bits are aligned with (center >> 1) | (right << 63)
//
// This is the branch-free form of the special handling the prompt calls out for bit 0 and bit 63.
__device__ __forceinline__ u64 shift_west(const u64 center, const u64 left) {
    return (center << 1) | (left >> 63);
}

__device__ __forceinline__ u64 shift_east(const u64 center, const u64 right) {
    return (center >> 1) | (right << 63);
}

// Sum three 1-bit fields per lane.
// For each of the 64 bit lanes independently:
//   bit0 + 2*bit1 == a + b + c
__device__ __forceinline__ void count3(const u64 a, const u64 b, const u64 c,
                                       u64& bit0, u64& bit1) {
    bit0 = a ^ b ^ c;
    bit1 = majority64(a, b, c);
}

// Return a bitboard whose lane k is 1 iff exactly one of a/b/c/d has lane k set.
// We intentionally compute only this predicate rather than a full 4-input population count,
// because after splitting the 8-neighbor sum into low/high parts, Conway's rule only needs
// to know whether the "weight-2" part equals 1.
__device__ __forceinline__ u64 exactly_one_of_four(const u64 a, const u64 b,
                                                   const u64 c, const u64 d) {
    const u64 one01 = a ^ b;  // pair (a,b) has exactly one set
    const u64 two01 = a & b;  // pair (a,b) has exactly two set
    const u64 one23 = c ^ d;  // pair (c,d) has exactly one set
    const u64 two23 = c & d;  // pair (c,d) has exactly two set
    return (one01 ^ one23) & ~(two01 | two23);
}

// One Conway step on a bit-packed grid.
// Missing neighbors are injected as zero words, implementing the required
// "outside the grid is dead" boundary condition with no wrapping.
//
// For each 64-bit word, the 8-neighbor sum is decomposed into three strips:
//
//   top    = NW + N + NE  = top0 + 2*top1
//   middle = W  + E       = mid0 + 2*mid1
//   bottom = SW + S + SE  = bot0 + 2*bot1
//
// Then:
//
//   cnt0 + 2*carry1 = top0 + mid0 + bot0
//
// Let u = carry1 + top1 + mid1 + bot1.
// The full neighbor count is:
//
//   neighbor_count = cnt0 + 2*u
//
// Conway's rule only needs neighbor_count == 2 or neighbor_count == 3.
// That is equivalent to u == 1, with cnt0 selecting between 2 and 3:
//
//   next = (u == 1) & (cnt0 | center)
//
// - cnt0 == 0 => count 2, so the cell survives only if it is already alive.
// - cnt0 == 1 => count 3, so the cell is alive in the next generation regardless of current state.
__global__ __launch_bounds__(kThreadsPerBlock)
void game_of_life_step_kernel(const u64* __restrict__ input,
                              u64* __restrict__ output,
                              const u64 total_words,
                              const u64 row_stride,
                              const u64 col_mask,
                              const u64 last_row_start) {
    const u64 idx =
        static_cast<u64>(blockIdx.x) * static_cast<u64>(kThreadsPerBlock) + threadIdx.x;
    if (idx >= total_words) {
        return;
    }

    // Because the row length in words is a power of two, x-boundary detection is just a mask.
    const u64 col = idx & col_mask;
    const bool has_left = col != 0;
    const bool has_right = col != col_mask;

    // Top/bottom detection is just a range check on the linear word index.
    const bool has_up = idx >= row_stride;
    const bool has_down = idx < last_row_start;

    const u64 center = input[idx];
    const u64 left = has_left ? input[idx - 1] : 0ull;
    const u64 right = has_right ? input[idx + 1] : 0ull;

    const u64 west = shift_west(center, left);
    const u64 east = shift_east(center, right);
    const u64 mid0 = west ^ east;
    const u64 mid1 = west & east;

    u64 top0 = 0ull;
    u64 top1 = 0ull;
    if (has_up) {
        const u64 idx_up = idx - row_stride;
        const u64 up = input[idx_up];
        const u64 up_left = has_left ? input[idx_up - 1] : 0ull;
        const u64 up_right = has_right ? input[idx_up + 1] : 0ull;
        count3(shift_west(up, up_left), up, shift_east(up, up_right), top0, top1);
    }

    u64 bot0 = 0ull;
    u64 bot1 = 0ull;
    if (has_down) {
        const u64 idx_down = idx + row_stride;
        const u64 down = input[idx_down];
        const u64 down_left = has_left ? input[idx_down - 1] : 0ull;
        const u64 down_right = has_right ? input[idx_down + 1] : 0ull;
        count3(shift_west(down, down_left), down, shift_east(down, down_right), bot0, bot1);
    }

    u64 cnt0;
    u64 carry1;
    count3(top0, mid0, bot0, cnt0, carry1);

    const u64 u_eq_1 = exactly_one_of_four(carry1, top1, mid1, bot1);
    output[idx] = u_eq_1 & (cnt0 | center);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // input and output are device pointers allocated by cudaMalloc.
    // The launch is asynchronous; the caller is responsible for synchronization and error handling.
    // input and output are assumed not to alias.
    const u64 row_stride = static_cast<u64>(grid_dimensions) >> 6;  // words per row
    const u64 total_words = row_stride * static_cast<u64>(grid_dimensions);
    if (total_words == 0) {
        return;
    }

    const u64 col_mask = row_stride - 1;         // row_stride is a power of two
    const u64 last_row_start = total_words - row_stride;
    const unsigned int blocks =
        static_cast<unsigned int>((total_words + kThreadsPerBlock - 1) / kThreadsPerBlock);

    game_of_life_step_kernel<<<blocks, kThreadsPerBlock>>>(
        input, output, total_words, row_stride, col_mask, last_row_start);
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
