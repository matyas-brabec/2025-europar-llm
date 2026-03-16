#include <cuda_runtime.h>
#include <cstddef>

namespace {

// This implementation intentionally exploits two prompt-specific facts:
//
// 1) grid_dimensions is always a power of two and > 512.
//    That means the grid is exactly divisible by 32, 256, 8, and 4, so all
//    kernel launch shapes tile the domain perfectly with no per-thread bounds
//    checks in the hot path.
//
// 2) Data transformations inside run_game_of_life are explicitly allowed and
//    are not measured.
//    We therefore pack the bool grid into 32-bit words (one bit per cell),
//    run the actual Game-of-Life step in that packed form, then unpack back to
//    bool. The packed step updates 32 cells per thread using only bitwise
//    logic, which is substantially faster than processing one byte per cell.
//
// Cells outside the grid are dead; the packed step injects zero-valued halo
// words/rows at the global boundaries.

using u32 = unsigned int;
static_assert(sizeof(u32) == 4, "This implementation relies on 32-bit words.");

constexpr u32 kWarpSize = 32u;
constexpr u32 kWordBits = 32u;
constexpr u32 kFullMask = 0xFFFFFFFFu;
constexpr u32 kLastLane = kWarpSize - 1u;

// Pack/unpack:
// - block = 256 x 4 = 1024 threads
// - one warp packs/unpacks one 32-cell word
// - each block covers 8 packed words across x and 4 rows across y
constexpr u32 kPackBlockX = 256u;
constexpr u32 kPackBlockY = 4u;
constexpr u32 kPackWordsPerBlockX = kPackBlockX / kWarpSize;

// Packed Life step:
// - block = 32 x 8 = 256 threads
// - one thread updates one packed word = 32 cells
// - each warp therefore updates one 1024-cell row segment
// - each block covers 32 packed words across x and 8 rows across y
constexpr u32 kStepBlockX = 32u;
constexpr u32 kStepBlockY = 8u;
constexpr u32 kStepWordsPerBlockX = kStepBlockX;

static_assert(kPackBlockX % kWarpSize == 0u, "Pack/unpack block must contain whole warps.");
static_assert(kStepBlockX == kWarpSize, "Step kernel assumes blockDim.x == warpSize.");
static_assert(kWordBits == kWarpSize, "Ballot packing assumes 32-bit words and 32-lane warps.");

struct PackedBufferCache {
    u32* a = nullptr;
    u32* b = nullptr;
    std::size_t capacity_words = 0;
};

// Per-host-thread cache so steady-state execution avoids cudaMalloc/cudaFree.
// Plain cudaMalloc/cudaFree are used instead of stream-ordered allocation
// because run_game_of_life has no stream parameter; this keeps behavior correct
// under both legacy and per-thread default-stream modes.
thread_local PackedBufferCache g_cache;

inline void ensure_packed_capacity(const std::size_t required_words) {
    if (g_cache.capacity_words >= required_words) {
        return;
    }

    if (g_cache.a != nullptr) {
        (void)cudaFree(g_cache.a);
        g_cache.a = nullptr;
    }
    if (g_cache.b != nullptr) {
        (void)cudaFree(g_cache.b);
        g_cache.b = nullptr;
    }
    g_cache.capacity_words = 0;

    (void)cudaMalloc(reinterpret_cast<void**>(&g_cache.a), required_words * sizeof(u32));
    (void)cudaMalloc(reinterpret_cast<void**>(&g_cache.b), required_words * sizeof(u32));
    g_cache.capacity_words = required_words;
}

// Adds one 1-bit neighbor mask into a four-bit per-cell counter represented as
// parallel bitplanes (ones/twos/fours/eights). Overflow beyond the eights plane
// is impossible because each Life neighborhood contains exactly eight cells.
__device__ __forceinline__ void add_mask_to_counter(
    const u32 v,
    u32& ones,
    u32& twos,
    u32& fours,
    u32& eights)
{
    u32 x = v;

    u32 carry = ones & x;
    ones ^= x;

    x = carry;
    carry = twos & x;
    twos ^= x;

    x = carry;
    carry = fours & x;
    fours ^= x;

    eights ^= carry;
}

// These shift/or forms map to SHF-style funnel shifts on modern NVIDIA GPUs.
__device__ __forceinline__ u32 shift_west(const u32 word, const u32 left_word) {
    return (word << 1) | (left_word >> 31);
}

__device__ __forceinline__ u32 shift_east(const u32 word, const u32 right_word) {
    return (word >> 1) | (right_word << 31);
}

__global__ __launch_bounds__(1024)
void pack_bool_grid_kernel(const unsigned char* __restrict__ input,
                           u32* __restrict__ packed) {
    const u32 lane = threadIdx.x & kLastLane;
    const u32 word_in_block = threadIdx.x / kWarpSize;
    const u32 word_x = static_cast<u32>(blockIdx.x) * kPackWordsPerBlockX + word_in_block;
    const u32 y = static_cast<u32>(blockIdx.y) * kPackBlockY + threadIdx.y;

    const u32 words_per_row = static_cast<u32>(gridDim.x) * kPackWordsPerBlockX;
    const u32 n = words_per_row * kWordBits;

    const std::size_t row_base = static_cast<std::size_t>(y) * static_cast<std::size_t>(n);
    const std::size_t cell_x =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kPackBlockX) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t cell_index = row_base + cell_x;

    const u32 mask = __ballot_sync(kFullMask, input[cell_index] != 0u);

    if (lane == 0u) {
        const u32 packed_row_base = y * words_per_row;
        packed[packed_row_base + word_x] = mask;
    }
}

__global__ __launch_bounds__(1024)
void unpack_bool_grid_kernel(const u32* __restrict__ packed,
                             unsigned char* __restrict__ output) {
    const u32 lane = threadIdx.x & kLastLane;
    const u32 word_in_block = threadIdx.x / kWarpSize;
    const u32 word_x = static_cast<u32>(blockIdx.x) * kPackWordsPerBlockX + word_in_block;
    const u32 y = static_cast<u32>(blockIdx.y) * kPackBlockY + threadIdx.y;

    const u32 words_per_row = static_cast<u32>(gridDim.x) * kPackWordsPerBlockX;
    const u32 n = words_per_row * kWordBits;

    const u32 packed_row_base = y * words_per_row;

    // One packed-word load per warp, then broadcast to all lanes.
    u32 word = 0u;
    if (lane == 0u) {
        word = __ldg(packed + packed_row_base + word_x);
    }
    word = __shfl_sync(kFullMask, word, 0);

    const std::size_t row_base = static_cast<std::size_t>(y) * static_cast<std::size_t>(n);
    const std::size_t cell_x =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kPackBlockX) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t cell_index = row_base + cell_x;

    output[cell_index] = static_cast<unsigned char>((word >> lane) & 1u);
}

__global__ __launch_bounds__(256)
void life_step_packed_kernel(const u32* __restrict__ input,
                             u32* __restrict__ output) {
    const u32 lane = threadIdx.x;  // blockDim.x is exactly one warp
    const u32 word_x =
        static_cast<u32>(blockIdx.x) * kStepWordsPerBlockX + lane;
    const u32 y =
        static_cast<u32>(blockIdx.y) * kStepBlockY + threadIdx.y;

    // Packed-row indices fit in 32 bits for the targeted A100/H100 memory
    // sizes, while the original bool-grid indices may not. The hot packed step
    // therefore stays on 32-bit addressing.
    const u32 words_per_row = static_cast<u32>(gridDim.x) * kStepWordsPerBlockX;
    const u32 row_base = y * words_per_row;

    const bool has_north = (blockIdx.y != 0u) || (threadIdx.y != 0u);
    const bool has_south = (blockIdx.y + 1u != gridDim.y) || (threadIdx.y + 1u != kStepBlockY);
    const bool has_left_block = (blockIdx.x != 0u);
    const bool has_right_block = (blockIdx.x + 1u != gridDim.x);

    const u32* row_c = input + row_base;
    const u32* row_n = has_north ? (row_c - words_per_row) : nullptr;
    const u32* row_s = has_south ? (row_c + words_per_row) : nullptr;

    // Three 32-bit loads update 32 cells; adjacent-row reuse is expected to hit
    // in L1/L2 because neighboring warps process consecutive rows.
    const u32 north = has_north ? __ldg(row_n + word_x) : 0u;
    const u32 current = __ldg(row_c + word_x);
    const u32 south = has_south ? __ldg(row_s + word_x) : 0u;

    // Neighboring packed words inside the warp come from shuffle exchange.
    // Only the two warp-edge lanes issue extra global loads for the horizontal halo.
    u32 north_left_word = __shfl_up_sync(kFullMask, north, 1);
    u32 current_left_word = __shfl_up_sync(kFullMask, current, 1);
    u32 south_left_word = __shfl_up_sync(kFullMask, south, 1);

    u32 north_right_word = __shfl_down_sync(kFullMask, north, 1);
    u32 current_right_word = __shfl_down_sync(kFullMask, current, 1);
    u32 south_right_word = __shfl_down_sync(kFullMask, south, 1);

    if (lane == 0u) {
        if (has_left_block) {
            const u32 left_word_x = word_x - 1u;
            north_left_word = has_north ? __ldg(row_n + left_word_x) : 0u;
            current_left_word = __ldg(row_c + left_word_x);
            south_left_word = has_south ? __ldg(row_s + left_word_x) : 0u;
        } else {
            north_left_word = 0u;
            current_left_word = 0u;
            south_left_word = 0u;
        }
    } else if (lane == kLastLane) {
        if (has_right_block) {
            const u32 right_word_x = word_x + 1u;
            north_right_word = has_north ? __ldg(row_n + right_word_x) : 0u;
            current_right_word = __ldg(row_c + right_word_x);
            south_right_word = has_south ? __ldg(row_s + right_word_x) : 0u;
        } else {
            north_right_word = 0u;
            current_right_word = 0u;
            south_right_word = 0u;
        }
    }

    // Align the three rows' west/east neighbors onto the current bit positions.
    const u32 north_west = shift_west(north, north_left_word);
    const u32 north_east = shift_east(north, north_right_word);
    const u32 west = shift_west(current, current_left_word);
    const u32 east = shift_east(current, current_right_word);
    const u32 south_west = shift_west(south, south_left_word);
    const u32 south_east = shift_east(south, south_right_word);

    // Sum the eight neighbors as parallel 4-bit counters.
    u32 ones = 0u;
    u32 twos = 0u;
    u32 fours = 0u;
    u32 eights = 0u;

    add_mask_to_counter(north_west, ones, twos, fours, eights);
    add_mask_to_counter(north,      ones, twos, fours, eights);
    add_mask_to_counter(north_east, ones, twos, fours, eights);
    add_mask_to_counter(west,       ones, twos, fours, eights);
    add_mask_to_counter(east,       ones, twos, fours, eights);
    add_mask_to_counter(south_west, ones, twos, fours, eights);
    add_mask_to_counter(south,      ones, twos, fours, eights);
    add_mask_to_counter(south_east, ones, twos, fours, eights);

    // Life rule in packed form:
    // - exactly 3 neighbors -> alive
    // - exactly 2 neighbors -> keep current state
    // Counts >= 4 are filtered by ~(fours | eights).
    const u32 next = (twos & ~(fours | eights)) & (ones | current);
    output[row_base + word_x] = next;
}

}  // namespace

/// @FIXED
/// extern "C" void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // The prompt guarantees:
    // - square grid
    // - power-of-two dimensions
    // - dimensions > 512
    //
    // Therefore:
    // - words_per_row is an integer,
    // - pack/unpack tiles divide the grid exactly,
    // - the packed step tile (1024 cells in x, 8 rows in y) also divides exactly.

    const u32 n = static_cast<u32>(grid_dimensions);
    const u32 words_per_row = n / kWordBits;
    const std::size_t packed_word_count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(words_per_row);

    ensure_packed_capacity(packed_word_count);

    const auto* input_bytes = reinterpret_cast<const unsigned char*>(input);
    auto* output_bytes = reinterpret_cast<unsigned char*>(output);

    const dim3 pack_block(kPackBlockX, kPackBlockY, 1u);
    const dim3 pack_grid(words_per_row / kPackWordsPerBlockX, n / kPackBlockY, 1u);

    const dim3 step_block(kStepBlockX, kStepBlockY, 1u);
    const dim3 step_grid(words_per_row / kStepWordsPerBlockX, n / kStepBlockY, 1u);

    // In steady state this function is just three launches:
    // pack -> packed Life step -> unpack.
    // No synchronization is performed here; the caller owns synchronization.
    pack_bool_grid_kernel<<<pack_grid, pack_block>>>(input_bytes, g_cache.a);
    life_step_packed_kernel<<<step_grid, step_block>>>(g_cache.a, g_cache.b);
    unpack_bool_grid_kernel<<<pack_grid, pack_block>>>(g_cache.b, output_bytes);
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// this label is used to identify the memory layout of the input and output arrays by the framework
// MEMORY_LAYOUT: BOOLS
