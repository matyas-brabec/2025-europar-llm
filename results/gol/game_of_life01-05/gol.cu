#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// Fast path overview:
// - The public API uses one byte per cell (`bool`), but the hot simulation kernel is far faster
//   when the grid is bit-packed.
// - We therefore pack 32 horizontal cells into one uint32_t. Each thread updates one packed word,
//   i.e. 32 cells at once via pure bitwise logic.
// - One warp spans 32 packed words (1024 cells in X). Horizontal neighbor exchange uses warp
//   shuffles; only lane 0 / lane 31 touch cross-segment halo words.
// - Each warp computes two consecutive output rows using rolling row registers. That cuts the
//   average number of packed-row loads from 3/row to 2/row without using shared memory.
// - Neighbor counts are accumulated with a carry-save adder tree, so every logical instruction
//   updates 32 independent cells in parallel.
//
// The problem statement explicitly allows internal data transformations inside run_game_of_life.
// We keep those transformations outside the hot kernel so the simulation step itself is optimized.
//
// A small tiled bool fallback kernel is also provided for robustness:
// - if the caller passes an unsupported tiny grid (< 1024), or
// - if a temporary packed workspace cannot be reserved.
// The packed kernel remains the intended fast path.

constexpr unsigned kFullMask = 0xFFFFFFFFu;

constexpr int kPackThreads         = 256;
constexpr int kPackWarpsPerBlock   = kPackThreads / 32;

constexpr int kRowsPerWarp         = 2;

constexpr int kFallbackBlockX      = 32;
constexpr int kFallbackBlockY      = 8;

// Thread-local cached workspace.
// The API does not expose a stream, so thread-local state is the safest way to reuse temporary
// storage without introducing cross-thread races. Reuse also avoids paying allocator overhead on
// steady-state repeated calls.
struct Workspace {
    uint32_t* data = nullptr;   // actual allocation size is 2 * capacity_words uint32_t's
    size_t    capacity_words = 0;
    int       device = -1;
};

thread_local Workspace g_workspace;
thread_local int g_sm_count = 0;
thread_local int g_sm_device = -1;

__host__ inline int log2_power_of_two(unsigned x) {
    // grid_dimensions is guaranteed to be a power of two, so this exact integer log2 is cheap.
    int r = 0;
    while (x > 1u) {
        x >>= 1;
        ++r;
    }
    return r;
}

__host__ inline int get_sm_count(int current_device) {
    if (g_sm_device != current_device) {
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, current_device);
        g_sm_device = current_device;
    }
    return g_sm_count;
}

__host__ inline bool ensure_workspace(size_t total_words, int current_device) {
    // Device switching is not the hot path, but correctness matters if the host thread changes
    // devices between calls.
    if (g_workspace.device != current_device) {
        if (g_workspace.data != nullptr && g_workspace.device >= 0) {
            const int old_device = g_workspace.device;
            cudaSetDevice(old_device);
            cudaFreeAsync(g_workspace.data, 0);
            cudaSetDevice(current_device);
        }
        g_workspace.data = nullptr;
        g_workspace.capacity_words = 0;
        g_workspace.device = current_device;
    }

    if (g_workspace.capacity_words < total_words) {
        if (g_workspace.data != nullptr) {
            cudaFreeAsync(g_workspace.data, 0);
            g_workspace.data = nullptr;
            g_workspace.capacity_words = 0;
        }

        if (cudaMallocAsync(reinterpret_cast<void**>(&g_workspace.data),
                            2 * total_words * sizeof(uint32_t),
                            0) != cudaSuccess) {
            g_workspace.data = nullptr;
            g_workspace.capacity_words = 0;
            return false;
        }

        g_workspace.capacity_words = total_words;
    }

    return g_workspace.data != nullptr;
}

__device__ __forceinline__ uint32_t load_packed_word(
    const uint32_t* __restrict__ grid,
    int row,
    int x_word,
    int n,
    int words_shift)
{
    // Packed row stride is a power of two, so row addressing is a shift, not a multiply.
    return (static_cast<unsigned>(row) < static_cast<unsigned>(n))
               ? __ldg(grid + ((static_cast<size_t>(row) << words_shift) +
                               static_cast<size_t>(x_word)))
               : 0u;
}

__device__ __forceinline__ unsigned char load_bool_cell(
    const bool* __restrict__ grid,
    int y,
    int x,
    int n)
{
    return (static_cast<unsigned>(y) < static_cast<unsigned>(n) &&
            static_cast<unsigned>(x) < static_cast<unsigned>(n))
               ? static_cast<unsigned char>(grid[static_cast<size_t>(y) * static_cast<size_t>(n) +
                                               static_cast<size_t>(x)])
               : 0u;
}

__device__ __forceinline__ void half_add(
    uint32_t a,
    uint32_t b,
    uint32_t& sum,
    uint32_t& carry)
{
    sum   = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ void full_add(
    uint32_t a,
    uint32_t b,
    uint32_t c,
    uint32_t& sum,
    uint32_t& carry)
{
    // sum   = a ^ b ^ c
    // carry = majority(a, b, c)
    // Written in a form that current NVCC lowers well to LOP3-based code on Ampere/Hopper.
    const uint32_t axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (axb & c);
}

__device__ __forceinline__ uint32_t evolve_word(
    uint32_t top,
    uint32_t mid,
    uint32_t bot,
    uint32_t top_left_halo,
    uint32_t top_right_halo,
    uint32_t mid_left_halo,
    uint32_t mid_right_halo,
    uint32_t bot_left_halo,
    uint32_t bot_right_halo,
    int lane)
{
    // Each input register holds 32 cells from one row.
    // Bit i corresponds to x = base_x + i. Ballot packing preserves exactly this layout.

    // Neighboring packed words in X come from warp shuffles.
    // Only the two warp-edge lanes substitute cross-segment halo words.
    uint32_t top_left_word  = __shfl_up_sync  (kFullMask, top, 1);
    uint32_t top_right_word = __shfl_down_sync(kFullMask, top, 1);
    uint32_t mid_left_word  = __shfl_up_sync  (kFullMask, mid, 1);
    uint32_t mid_right_word = __shfl_down_sync(kFullMask, mid, 1);
    uint32_t bot_left_word  = __shfl_up_sync  (kFullMask, bot, 1);
    uint32_t bot_right_word = __shfl_down_sync(kFullMask, bot, 1);

    if (lane == 0) {
        top_left_word = top_left_halo;
        mid_left_word = mid_left_halo;
        bot_left_word = bot_left_halo;
    }
    if (lane == 31) {
        top_right_word = top_right_halo;
        mid_right_word = mid_right_halo;
        bot_right_word = bot_right_halo;
    }

    // Construct the 8 aligned neighbor bitboards for all 32 cells in parallel.
    const uint32_t nw = (top << 1) | (top_left_word >> 31);
    const uint32_t nn = top;
    const uint32_t ne = (top >> 1) | (top_right_word << 31);

    const uint32_t ww = (mid << 1) | (mid_left_word >> 31);
    const uint32_t ee = (mid >> 1) | (mid_right_word << 31);

    const uint32_t sw = (bot << 1) | (bot_left_word >> 31);
    const uint32_t ss = bot;
    const uint32_t se = (bot >> 1) | (bot_right_word << 31);

    // Carry-save adder tree for eight 1-bit inputs:
    // count = ones + 2*twos + 4*fours + 8*eights
    uint32_t s0, c0, s1, c1, s2, c2;
    full_add(nw, nn, ne, s0, c0);
    full_add(ww, ee, sw, s1, c1);
    half_add(ss, se, s2, c2);

    uint32_t ones, c3;
    full_add(s0, s1, s2, ones, c3);

    uint32_t twos_partial, fours_partial;
    full_add(c0, c1, c2, twos_partial, fours_partial);

    uint32_t twos, carry_to_fours;
    half_add(twos_partial, c3, twos, carry_to_fours);

    uint32_t fours, eights;
    half_add(carry_to_fours, fours_partial, fours, eights);

    // Conway rule:
    // - survive on count == 2 if currently alive
    // - born on count == 3
    //
    // counts 2 and 3 are exactly the states where:
    //   twos == 1 and fours/eights == 0
    // Then `ones` distinguishes 3 from 2.
    const uint32_t can_live = twos & ~(fours | eights);
    return can_live & (ones | mid);
}

__global__ void pack_bool_to_words(
    const bool* __restrict__ input,
    uint32_t* __restrict__ output,
    int grid_shift,
    int words_shift,
    int words_mask,
    size_t total_words)
{
    // Grid-stride over packed words.
    // One warp packs one output uint32_t from 32 consecutive bool cells.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const size_t global_warp = static_cast<size_t>(blockIdx.x) * kPackWarpsPerBlock +
                               static_cast<size_t>(warp);
    const size_t total_warps = static_cast<size_t>(gridDim.x) * kPackWarpsPerBlock;

    for (size_t word_idx = global_warp; word_idx < total_words; word_idx += total_warps) {
        // Because words_per_row is a power of two, row decomposition is shift+mask.
        const int row  = static_cast<int>(word_idx >> words_shift);
        const int word = static_cast<int>(word_idx & static_cast<size_t>(words_mask));

        const size_t base = (static_cast<size_t>(row)  << grid_shift) +
                            (static_cast<size_t>(word) << 5);

        const bool alive = input[base + static_cast<size_t>(lane)];
        const uint32_t packed = __ballot_sync(kFullMask, alive);

        if (lane == 0) {
            output[word_idx] = packed;
        }
    }
}

__global__ void unpack_words_to_bool(
    const uint32_t* __restrict__ input,
    bool* __restrict__ output,
    int grid_shift,
    int words_shift,
    int words_mask,
    size_t total_words)
{
    // Inverse of pack_bool_to_words.
    // Lane 0 loads one packed word, broadcasts it across the warp, and each lane writes one bool.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const size_t global_warp = static_cast<size_t>(blockIdx.x) * kPackWarpsPerBlock +
                               static_cast<size_t>(warp);
    const size_t total_warps = static_cast<size_t>(gridDim.x) * kPackWarpsPerBlock;

    for (size_t word_idx = global_warp; word_idx < total_words; word_idx += total_warps) {
        uint32_t packed = 0;
        if (lane == 0) {
            packed = __ldg(input + word_idx);
        }
        packed = __shfl_sync(kFullMask, packed, 0);

        const int row  = static_cast<int>(word_idx >> words_shift);
        const int word = static_cast<int>(word_idx & static_cast<size_t>(words_mask));

        const size_t base = (static_cast<size_t>(row)  << grid_shift) +
                            (static_cast<size_t>(word) << 5);

        output[base + static_cast<size_t>(lane)] = ((packed >> lane) & 1u) != 0;
    }
}

__global__ void game_of_life_packed_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int n,
    int words_shift)
{
    // blockDim.x is always 32, so each warp is one contiguous X-segment of 32 packed words.
    // That is 32 * 32 = 1024 cells in X.
    const int segment = static_cast<int>(blockIdx.x);
    const int lane    = threadIdx.x;
    const int warp_y  = threadIdx.y;

    const int x_word = (segment << 5) + lane;
    const int y0     = ((static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y)) + warp_y)
                       << 1; // two rows per warp

    if (y0 >= n) {
        return;
    }

    const bool has_left_segment  = (segment != 0);
    const bool has_right_segment = (segment + 1 < static_cast<int>(gridDim.x));

    // Initial rolling window for rows y0-1, y0, y0+1.
    uint32_t top = load_packed_word(input, y0 - 1, x_word, n, words_shift);
    uint32_t mid = load_packed_word(input, y0 + 0, x_word, n, words_shift);
    uint32_t bot = load_packed_word(input, y0 + 1, x_word, n, words_shift);

    // Only edge lanes need true halo words from adjacent 32-word segments.
    uint32_t top_left_halo  = 0;
    uint32_t mid_left_halo  = 0;
    uint32_t bot_left_halo  = 0;
    uint32_t top_right_halo = 0;
    uint32_t mid_right_halo = 0;
    uint32_t bot_right_halo = 0;

    if (lane == 0 && has_left_segment) {
        top_left_halo = load_packed_word(input, y0 - 1, x_word - 1, n, words_shift);
        mid_left_halo = load_packed_word(input, y0 + 0, x_word - 1, n, words_shift);
        bot_left_halo = load_packed_word(input, y0 + 1, x_word - 1, n, words_shift);
    }

    if (lane == 31 && has_right_segment) {
        top_right_halo = load_packed_word(input, y0 - 1, x_word + 1, n, words_shift);
        mid_right_halo = load_packed_word(input, y0 + 0, x_word + 1, n, words_shift);
        bot_right_halo = load_packed_word(input, y0 + 1, x_word + 1, n, words_shift);
    }

    const size_t row_stride = static_cast<size_t>(1) << words_shift;
    const size_t out0 = (static_cast<size_t>(y0) << words_shift) + static_cast<size_t>(x_word);

    output[out0] = evolve_word(
        top, mid, bot,
        top_left_halo, top_right_halo,
        mid_left_halo, mid_right_halo,
        bot_left_halo, bot_right_halo,
        lane);

    // Roll the three-row window once to compute the next output row in the same warp.
    if (y0 + 1 < n) {
        top = mid;
        mid = bot;
        bot = load_packed_word(input, y0 + 2, x_word, n, words_shift);

        if (lane == 0) {
            top_left_halo = mid_left_halo;
            mid_left_halo = bot_left_halo;
            bot_left_halo = has_left_segment
                                ? load_packed_word(input, y0 + 2, x_word - 1, n, words_shift)
                                : 0u;
        }

        if (lane == 31) {
            top_right_halo = mid_right_halo;
            mid_right_halo = bot_right_halo;
            bot_right_halo = has_right_segment
                                 ? load_packed_word(input, y0 + 2, x_word + 1, n, words_shift)
                                 : 0u;
        }

        output[out0 + row_stride] = evolve_word(
            top, mid, bot,
            top_left_halo, top_right_halo,
            mid_left_halo, mid_right_halo,
            bot_left_halo, bot_right_halo,
            lane);
    }
}

__global__ void game_of_life_bool_fallback_kernel(
    const bool* __restrict__ input,
    bool* __restrict__ output,
    int n)
{
    // Robust fallback path only. The packed kernel above is the performance path.
    __shared__ unsigned char tile[kFallbackBlockY + 2][kFallbackBlockX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = static_cast<int>(blockIdx.x) * kFallbackBlockX + tx;
    const int y = static_cast<int>(blockIdx.y) * kFallbackBlockY + ty;

    const int sx = tx + 1;
    const int sy = ty + 1;

    tile[sy][sx] = load_bool_cell(input, y, x, n);

    if (tx == 0) {
        tile[sy][0] = load_bool_cell(input, y, x - 1, n);
    }
    if (tx == kFallbackBlockX - 1) {
        tile[sy][kFallbackBlockX + 1] = load_bool_cell(input, y, x + 1, n);
    }

    if (ty == 0) {
        tile[0][sx] = load_bool_cell(input, y - 1, x, n);
    }
    if (ty == kFallbackBlockY - 1) {
        tile[kFallbackBlockY + 1][sx] = load_bool_cell(input, y + 1, x, n);
    }

    if (tx == 0 && ty == 0) {
        tile[0][0] = load_bool_cell(input, y - 1, x - 1, n);
    }
    if (tx == kFallbackBlockX - 1 && ty == 0) {
        tile[0][kFallbackBlockX + 1] = load_bool_cell(input, y - 1, x + 1, n);
    }
    if (tx == 0 && ty == kFallbackBlockY - 1) {
        tile[kFallbackBlockY + 1][0] = load_bool_cell(input, y + 1, x - 1, n);
    }
    if (tx == kFallbackBlockX - 1 && ty == kFallbackBlockY - 1) {
        tile[kFallbackBlockY + 1][kFallbackBlockX + 1] =
            load_bool_cell(input, y + 1, x + 1, n);
    }

    __syncthreads();

    if (x < n && y < n) {
        const unsigned sum =
            tile[sy - 1][sx - 1] + tile[sy - 1][sx] + tile[sy - 1][sx + 1] +
            tile[sy    ][sx - 1]                    + tile[sy    ][sx + 1] +
            tile[sy + 1][sx - 1] + tile[sy + 1][sx] + tile[sy + 1][sx + 1];

        const bool alive = tile[sy][sx] != 0;
        output[static_cast<size_t>(y) * static_cast<size_t>(n) + static_cast<size_t>(x)] =
            (sum == 3u) || (alive && sum == 2u);
    }
}

} // anonymous namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    if (input == nullptr || output == nullptr || grid_dimensions <= 0) {
        return;
    }

    const int n = grid_dimensions;

    // The fast packed path assumes at least one full warp of packed words per row.
    // The prompt guarantees n > 512 and power-of-two, so n >= 1024 in the measured path.
    const bool use_fast_path = (n >= 1024);

    const int current_device = []() {
        int d = 0;
        cudaGetDevice(&d);
        return d;
    }();

    if (!use_fast_path) {
        dim3 block(kFallbackBlockX, kFallbackBlockY, 1);
        dim3 grid(static_cast<unsigned>((n + kFallbackBlockX - 1) / kFallbackBlockX),
                  static_cast<unsigned>((n + kFallbackBlockY - 1) / kFallbackBlockY),
                  1);
        game_of_life_bool_fallback_kernel<<<grid, block, 0, 0>>>(input, output, n);
        return;
    }

    // All addressing in the packed path is derived from shifts because n is a power of two.
    const int grid_shift   = log2_power_of_two(static_cast<unsigned>(n));
    const int words_shift  = grid_shift - 5;          // divide row length by 32 cells/word
    const int words_per_row = 1 << words_shift;
    const int words_mask    = words_per_row - 1;
    const int segments_per_row = words_per_row >> 5;  // 32 packed words per warp segment

    const size_t total_words = static_cast<size_t>(n) << words_shift;

    if (!ensure_workspace(total_words, current_device)) {
        // If temporary storage cannot be reserved, fall back to the direct bool kernel.
        dim3 block(kFallbackBlockX, kFallbackBlockY, 1);
        dim3 grid(static_cast<unsigned>((n + kFallbackBlockX - 1) / kFallbackBlockX),
                  static_cast<unsigned>((n + kFallbackBlockY - 1) / kFallbackBlockY),
                  1);
        game_of_life_bool_fallback_kernel<<<grid, block, 0, 0>>>(input, output, n);
        return;
    }

    // Touch only the compact 2*total_words region even if the cached buffer is larger.
    uint32_t* const packed_input  = g_workspace.data;
    uint32_t* const packed_output = g_workspace.data + total_words;

    // Pack/unpack are not the measured hot path, but they are still launched with enough work
    // to saturate the device. A modest block cap keeps launch overhead reasonable.
    const int sms = get_sm_count(current_device);
    const size_t blocks_needed = (total_words + kPackWarpsPerBlock - 1) / kPackWarpsPerBlock;
    const size_t blocks_cap = static_cast<size_t>((sms > 0) ? (sms * 8) : 1);
    int pack_blocks = static_cast<int>((blocks_needed < blocks_cap) ? blocks_needed : blocks_cap);
    if (pack_blocks < 1) {
        pack_blocks = 1;
    }

    pack_bool_to_words<<<pack_blocks, kPackThreads, 0, 0>>>(
        input,
        packed_input,
        grid_shift,
        words_shift,
        words_mask,
        total_words);

    // For the smallest measured grid (1024x1024), there is only one X-segment per row, so we use
    // fewer warps/block to keep the block count high enough to fill the GPU. Larger grids get
    // 8 warps/block for slightly better efficiency.
    const int warps_per_block = (segments_per_row == 1) ? 4 : 8;
    const int rows_per_block  = warps_per_block * kRowsPerWarp;

    dim3 hot_block(32, static_cast<unsigned>(warps_per_block), 1);
    dim3 hot_grid(static_cast<unsigned>(segments_per_row),
                  static_cast<unsigned>((n + rows_per_block - 1) / rows_per_block),
                  1);

    game_of_life_packed_kernel<<<hot_grid, hot_block, 0, 0>>>(
        packed_input,
        packed_output,
        n,
        words_shift);

    unpack_words_to_bool<<<pack_blocks, kPackThreads, 0, 0>>>(
        packed_output,
        output,
        grid_shift,
        words_shift,
        words_mask,
        total_words);

    // Intentionally no synchronization and no host-side error polling here:
    // the caller owns synchronization, and the task explicitly prioritizes kernel performance.
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
