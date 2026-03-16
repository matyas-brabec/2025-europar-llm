#include <cstdint>
#include <cuda_runtime.h>

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static_assert(sizeof(u32) == 4 && sizeof(u64) == 8, "Unexpected integer sizes.");

constexpr int kWarpSize = 32;
constexpr int kSubwordBits = 32;
constexpr int kSubwordsPerWarpSegment = 32;
constexpr int kCellsPerWarpSegment = kSubwordBits * kSubwordsPerWarpSegment;  // 1024 cells
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kMaxBlocks = 4096u;

__device__ __forceinline__ void carry_save_add(const u32 a,
                                               const u32 b,
                                               const u32 c,
                                               u32& lo,
                                               u32& hi) {
    // Bit-sliced 3:2 compressor:
    //   a + b + c = lo + 2*hi
    // for every bit position independently.
    const u32 ab = a ^ b;
    lo = ab ^ c;
    hi = (a & b) | (ab & c);
}

__device__ __forceinline__ u32 neighbor_from_left(const u32 left, const u32 center) {
    // Result bit i receives source bit i-1 from the logical row.
    // This is the "west neighbor aligned to the current cell" mask.
    // NVCC lowers this shift/or idiom to SHF-based code on modern GPUs.
    return (center << 1) | (left >> (kSubwordBits - 1));
}

__device__ __forceinline__ u32 neighbor_from_right(const u32 center, const u32 right) {
    // Result bit i receives source bit i+1 from the logical row.
    // This is the "east neighbor aligned to the current cell" mask.
    return (center >> 1) | (right << (kSubwordBits - 1));
}

__device__ __forceinline__ u32 evolve_subword(const u32 live,
                                              const u32 n_left,
                                              const u32 n_center,
                                              const u32 n_right,
                                              const u32 c_left,
                                              const u32 c_right,
                                              const u32 s_left,
                                              const u32 s_center,
                                              const u32 s_right) {
    // Exact 8-way bit-parallel population count using a compact carry-save tree.
    // The final bitplanes are:
    //   bit0 = 1s place
    //   bit1 = 2s place
    //   bit2 = 4s place
    //   bit3 = 8s place
    u32 p0, p1;
    u32 q0, q1;
    u32 r0, r1;

    carry_save_add(n_left, n_center, n_right, p0, p1);
    carry_save_add(c_left, c_right, s_left, q0, q1);
    carry_save_add(s_center, s_right, p0, r0, r1);

    const u32 bit0 = q0 ^ r0;
    const u32 carry0 = q0 & r0;

    u32 bit1_tmp, carry1;
    carry_save_add(p1, q1, r1, bit1_tmp, carry1);

    const u32 bit1 = bit1_tmp ^ carry0;
    const u32 carry2 = bit1_tmp & carry0;

    const u32 bit2 = carry1 ^ carry2;
    const u32 bit3 = carry1 & carry2;

    // Conway rule:
    //   alive next = (count == 3) || (live && count == 2)
    // count == 2 or 3 iff bit1=1 and bit2=bit3=0.
    const u32 count_is_2_or_3 = bit1 & ~(bit2 | bit3);
    return count_is_2_or_3 & (bit0 | live);
}

// Shared memory is intentionally not used.
// Each warp owns one contiguous row segment, keeps three row-aligned subwords in
// registers, uses warp shuffles for horizontal reuse, and only its two edge lanes
// need extra global loads.
template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * kWarpSize)
void game_of_life_kernel(const u32* __restrict__ input,
                         u32* __restrict__ output,
                         const u64 total_tasks,
                         const u32 subwords_per_row,
                         const u32 segments_per_row,
                         const u32 segments_per_row_mask) {
    constexpr unsigned int kFullMask = 0xFFFFFFFFu;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;

    u64 task = static_cast<u64>(blockIdx.x) * WarpsPerBlock + static_cast<u64>(warp_in_block);
    const u64 task_stride = static_cast<u64>(gridDim.x) * WarpsPerBlock;

    const u64 subwords_per_row_64 = static_cast<u64>(subwords_per_row);
    const u64 segments_per_row_64 = static_cast<u64>(segments_per_row);

    while (task < total_tasks) {
        // Tasks are warp-sized row segments in row-major order.
        // Because one task always spans exactly 32 subwords, its linear base
        // subword index is simply task * 32.
        const u32 segment = static_cast<u32>(task & static_cast<u64>(segments_per_row_mask));
        const u64 row_base = task * static_cast<u64>(kSubwordsPerWarpSegment);
        const u64 north_base = row_base - subwords_per_row_64;
        const u64 south_base = row_base + subwords_per_row_64;
        const u64 idx = row_base + static_cast<u64>(lane);

        const bool has_north = (task >= segments_per_row_64);
        const bool has_south = (task + segments_per_row_64 < total_tasks);
        const bool has_left = (segment != 0u);
        const bool has_right = (segment != segments_per_row_mask);

        const u32 center = input[idx];
        const u32 north = has_north ? input[north_base + static_cast<u64>(lane)] : 0u;
        const u32 south = has_south ? input[south_base + static_cast<u64>(lane)] : 0u;

        u32 center_left = __shfl_up_sync(kFullMask, center, 1, kWarpSize);
        u32 center_right = __shfl_down_sync(kFullMask, center, 1, kWarpSize);
        u32 north_left = __shfl_up_sync(kFullMask, north, 1, kWarpSize);
        u32 north_right = __shfl_down_sync(kFullMask, north, 1, kWarpSize);
        u32 south_left = __shfl_up_sync(kFullMask, south, 1, kWarpSize);
        u32 south_right = __shfl_down_sync(kFullMask, south, 1, kWarpSize);

        // Only the segment-edge lanes need fallback loads.
        if (lane == 0) {
            center_left = has_left ? input[row_base - 1u] : 0u;
            north_left = (has_north && has_left) ? input[north_base - 1u] : 0u;
            south_left = (has_south && has_left) ? input[south_base - 1u] : 0u;
        } else if (lane == kWarpSize - 1) {
            center_right = has_right ? input[row_base + kSubwordsPerWarpSegment] : 0u;
            north_right = (has_north && has_right) ? input[north_base + kSubwordsPerWarpSegment] : 0u;
            south_right = (has_south && has_right) ? input[south_base + kSubwordsPerWarpSegment] : 0u;
        }

        output[idx] = evolve_subword(
            center,
            neighbor_from_left(north_left, north), north, neighbor_from_right(north, north_right),
            neighbor_from_left(center_left, center), neighbor_from_right(center, center_right),
            neighbor_from_left(south_left, south), south, neighbor_from_right(south, south_right));

        task += task_stride;
    }
}

}  // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The incoming storage is already on the device and already bit-packed.
    // Reinterpreting it as 32-bit subwords preserves the exact in-memory layout on
    // NVIDIA GPUs and enables native-width bit logic and cross-subword shifts.
    const auto* input32 = reinterpret_cast<const u32*>(input);
    auto* output32 = reinterpret_cast<u32*>(output);

    const u32 grid_dim_u32 = static_cast<u32>(grid_dimensions);
    const u32 subwords_per_row = grid_dim_u32 >> 5;       // N / 32
    const u32 segments_per_row = grid_dim_u32 >> 10;      // N / 1024, guaranteed >= 1
    const u32 segments_per_row_mask = segments_per_row - 1u;
    const u64 total_tasks = static_cast<u64>(grid_dim_u32) * static_cast<u64>(segments_per_row);

    const u64 max_blocks_needed =
        (total_tasks + static_cast<u64>(kWarpsPerBlock) - 1u) / static_cast<u64>(kWarpsPerBlock);

    // Fixed grid cap chosen for modern data-center GPUs:
    // enough to oversubscribe A100/H100 class hardware by several waves while
    // keeping launch-side overhead minimal.
    const unsigned int blocks = static_cast<unsigned int>(
        (max_blocks_needed < static_cast<u64>(kMaxBlocks))
            ? max_blocks_needed
            : static_cast<u64>(kMaxBlocks));

    game_of_life_kernel<kWarpsPerBlock><<<blocks, kThreadsPerBlock>>>(
        input32,
        output32,
        total_tasks,
        subwords_per_row,
        segments_per_row,
        segments_per_row_mask);
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
