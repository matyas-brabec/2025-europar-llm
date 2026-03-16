#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// One thread updates one packed 64-cell word.
// One warp therefore updates 32 packed words = 2048 cells in a horizontal row tile.
using u64 = std::uint64_t;

constexpr unsigned kWarpSize          = 32;
constexpr unsigned kWarpTileShift     = 5;   // 2^5 = 32 packed words per warp tile.
constexpr unsigned kCellsPerWordShift = 6;   // 2^6 = 64 cells per packed word.
constexpr unsigned kWarpsPerBlock     = 4;   // 128-thread blocks: good distribution even for 1024x1024 boards.
constexpr unsigned kBlockThreadsX     = kWarpSize;
constexpr unsigned kBlockThreadsY     = kWarpsPerBlock;

// The kernel is written as a warp-granular grid-stride loop. 256k blocks is already far beyond
// what an A100/H100 needs for full saturation, while avoiding pathological multi-million-block
// launches on extremely large boards.
constexpr std::size_t kLaunchBlocksCap = std::size_t{1} << 18;

static_assert(kBlockThreadsX == kWarpSize, "Each block row must map to exactly one warp.");

// Host-side helper: the input is guaranteed to be a power of two, so log2 is just repeated shifts.
// This keeps the wrapper portable without relying on compiler-specific intrinsics.
inline unsigned log2_pow2(std::size_t x) {
    unsigned shift = 0;
    while (x > 1) {
        x >>= 1;
        ++shift;
    }
    return shift;
}

// On modern NVIDIA data-center GPUs, ordinary global loads from const __restrict__ memory are
// already well cached; explicit texture memory or shared-memory staging is unnecessary here.
__device__ __forceinline__ u64 load_ro(const u64* ptr) {
    return *ptr;
}

// Carry-save adder for bit-sliced accumulation.
// Every bit position is independent, so one 64-bit register represents 64 separate cells.
// a + b + c == sum + 2 * carry, evaluated independently for all 64 bit lanes.
// On Ampere/Hopper, ptxas typically lowers this boolean network to efficient LOP3 instructions.
__device__ __forceinline__ void csa(u64 a, u64 b, u64 c, u64& sum, u64& carry) {
    const u64 ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    carry = (a & b) | (ab_xor & c);
}

// For the natural packing used here, bit k in a word corresponds to x = 64*word + k.
// shift_west/east align horizontal neighbors for all 64 cells in parallel, bringing in the
// cross-word carry bit from the adjacent packed word.
__device__ __forceinline__ u64 shift_west(u64 center, u64 left_word) {
    return (center << 1) | (left_word >> 63);
}

__device__ __forceinline__ u64 shift_east(u64 center, u64 right_word) {
    return (center >> 1) | (right_word << 63);
}

// High-throughput Conway step kernel.
// Key design choices:
//   - No shared memory: each thread only loads the vertically aligned words (top/current/bottom).
//   - Horizontal reuse comes from warp shuffles, so the common case is 3 loads/thread instead of 9.
//   - One linear warp-tile index maps to (row, horizontal-tile) using only shifts/masks, made
//     possible by the power-of-two grid size.
__global__ void game_of_life_kernel(const u64* __restrict__ input,
                                    u64* __restrict__ output,
                                    std::size_t grid_dim,
                                    unsigned words_per_row_shift,
                                    unsigned tiles_per_row_shift) {
    constexpr unsigned kFullMask = 0xFFFFFFFFu;

    const std::size_t words_per_row     = std::size_t{1} << words_per_row_shift;
    const std::size_t tiles_per_row_mask =
        (std::size_t{1} << tiles_per_row_shift) - std::size_t{1};
    const std::size_t total_row_tiles   = grid_dim << tiles_per_row_shift;

    const unsigned lane         = threadIdx.x;
    const unsigned warp_in_block = threadIdx.y;

    const std::size_t first_warp_tile =
        static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const std::size_t warp_tile_stride =
        static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock;

    for (std::size_t warp_tile = first_warp_tile;
         warp_tile < total_row_tiles;
         warp_tile += warp_tile_stride) {

        // Because tiles_per_row is a power of two, row/tile decomposition is shift+mask only.
        const std::size_t row    = warp_tile >> tiles_per_row_shift;
        const std::size_t tile   = warp_tile & tiles_per_row_mask;
        const std::size_t word_x = (tile << kWarpTileShift) + lane;

        // Only the smallest legal board (1024x1024) produces a partial warp tile:
        // 1024 cells / 64 cells-per-word = 16 packed words per row.
        const unsigned active_mask = __ballot_sync(kFullMask, word_x < words_per_row);
        const bool lane_active = (active_mask & (1u << lane)) != 0u;

        if (lane_active) {
            const std::size_t row_base = row << words_per_row_shift;

            const u64* const row_ptr = input  + row_base;
            u64* const       out_ptr = output + row_base;

            // Outside-grid cells are dead, so missing rows/words contribute zeros.
            const bool has_top    = row != 0;
            const bool has_bottom = (row + 1) != grid_dim;
            const bool has_left   = word_x != 0;
            const bool has_right  = (word_x + 1) != words_per_row;

            const u64* const top_ptr = has_top    ? (row_ptr - words_per_row) : nullptr;
            const u64* const bot_ptr = has_bottom ? (row_ptr + words_per_row) : nullptr;

            // Center words for the three rows.
            const u64 top = has_top    ? load_ro(top_ptr + word_x) : u64{0};
            const u64 mid =               load_ro(row_ptr + word_x);
            const u64 bot = has_bottom ? load_ro(bot_ptr + word_x) : u64{0};

            // Horizontal neighbors usually come from adjacent lanes in the same warp tile.
            // Only tile boundaries require an extra global load.
            u64 top_l = __shfl_up_sync  (active_mask, top, 1);
            u64 top_r = __shfl_down_sync(active_mask, top, 1);
            u64 mid_l = __shfl_up_sync  (active_mask, mid, 1);
            u64 mid_r = __shfl_down_sync(active_mask, mid, 1);
            u64 bot_l = __shfl_up_sync  (active_mask, bot, 1);
            u64 bot_r = __shfl_down_sync(active_mask, bot, 1);

            if (lane == 0u) {
                mid_l = has_left ? load_ro(row_ptr + word_x - 1) : u64{0};
                top_l = (has_top    && has_left) ? load_ro(top_ptr + word_x - 1) : u64{0};
                bot_l = (has_bottom && has_left) ? load_ro(bot_ptr + word_x - 1) : u64{0};
            }

            if (lane == (kWarpSize - 1u) || !has_right) {
                mid_r = has_right ? load_ro(row_ptr + word_x + 1) : u64{0};
                top_r = (has_top    && has_right) ? load_ro(top_ptr + word_x + 1) : u64{0};
                bot_r = (has_bottom && has_right) ? load_ro(bot_ptr + word_x + 1) : u64{0};
            }

            // Build the 8 neighbor bitboards.
            // Scope-local temporaries keep register lifetimes short.
            u64 s_top, c_top;
            {
                const u64 west = shift_west(top, top_l);
                const u64 east = shift_east(top, top_r);
                csa(west, top, east, s_top, c_top);
            }

            u64 s_mid, c_mid;
            {
                const u64 west = shift_west(mid, mid_l);
                const u64 east = shift_east(mid, mid_r);
                s_mid = west ^ east;  // half adder of the two horizontal neighbors
                c_mid = west & east;
            }

            u64 s_bot, c_bot;
            {
                const u64 west = shift_west(bot, bot_l);
                const u64 east = shift_east(bot, bot_r);
                csa(west, bot, east, s_bot, c_bot);
            }

            // Reduce the 8 one-bit neighbor masks with a carry-save tree.
            //
            // After these two CSAs:
            //   neighbors = bit0 + 2*twos_a + 2*twos_b + 4*fours
            //
            // The final 2's-place is bit1 = twos_a ^ twos_b.
            // If twos_a and twos_b are both 1, the total is already >= 4, and bit1 becomes 0.
            // Therefore:
            //   count == 2 or 3  <=>  bit1 is 1 and fours is 0
            //
            // Finally, (bit0 | mid) encodes the Life rule:
            //   - count == 3 => bit0 == 1 => birth/survival
            //   - count == 2 => bit0 == 0 => survive only if currently alive (mid == 1)
            u64 bit0, twos_a;
            csa(s_top, s_mid, s_bot, bit0, twos_a);

            u64 twos_b, fours;
            csa(c_top, c_mid, c_bot, twos_b, fours);

            const u64 bit1 = twos_a ^ twos_b;
            out_ptr[word_x] = bit1 & ~fours & (bit0 | mid);
        }
    }
}

} // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The problem statement guarantees a power-of-two square board > 512.
    // A defensive early return keeps the wrapper well-defined for accidental misuse without
    // affecting the measured kernel path.
    if (grid_dimensions <= 0) {
        return;
    }

    const std::size_t grid_dim = static_cast<std::size_t>(grid_dimensions);

    // Since grid_dim is a power of two:
    //   words_per_row      = grid_dim / 64 = 2^(log2(grid_dim) - 6)
    //   tiles_per_row      = ceil(words_per_row / 32)
    //                      = max(1, words_per_row / 32)
    //                      = power of two as well
    //
    // That lets the kernel recover row/tile coordinates with shifts and masks only.
    const unsigned grid_dim_shift      = log2_pow2(grid_dim);
    const unsigned words_per_row_shift = grid_dim_shift - kCellsPerWordShift;
    const unsigned tiles_per_row_shift =
        (words_per_row_shift > kWarpTileShift)
            ? (words_per_row_shift - kWarpTileShift)
            : 0u;

    const std::size_t total_row_tiles = grid_dim << tiles_per_row_shift;
    const std::size_t blocks_needed =
        (total_row_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;

    // The kernel itself is grid-stride, so bounding the launch size is enough to keep very large
    // boards efficient without changing the steady-state work per tile.
    const std::size_t launch_blocks =
        (blocks_needed < kLaunchBlocksCap) ? blocks_needed : kLaunchBlocksCap;

    const dim3 block(kBlockThreadsX, kBlockThreadsY);
    game_of_life_kernel<<<static_cast<unsigned int>(launch_blocks), block, 0, 0>>>(
        input,
        output,
        grid_dim,
        words_per_row_shift,
        tiles_per_row_shift);
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
