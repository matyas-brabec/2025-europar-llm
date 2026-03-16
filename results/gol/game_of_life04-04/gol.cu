#include <cstdint>
#include <cuda_runtime.h>

/*
    High-performance bit-packed Conway's Game of Life for modern NVIDIA GPUs.

    Design choices driven purely by kernel throughput:

    1) One logical work item produces one 64-bit output word (64 cells), so there are no atomics.

    2) No shared memory and no texture memory.
       For this stencil, the simplest high-performance path on A100/H100-class GPUs is:
         - load the center word from the three relevant rows,
         - obtain left/right neighbor words with warp shuffles,
         - only let tile-edge lanes fall back to direct global loads.

    3) Avoid 64 scalar neighbor counts per thread.
       Using __popc is already much better than a naive bit-by-bit masking loop, but the fastest path
       here is to count all 64 cells in parallel with a bit-sliced counter:
         - ones : low bit of the 0..3 neighbor count modulo 4
         - twos : second bit of the 0..3 neighbor count modulo 4
         - ge4  : latched once a cell reaches 4 or more neighbors
       Then:
         next = twos & (ones | current) & ~ge4
       which exactly encodes:
         - birth on 3 neighbors,
         - survival on 2 neighbors if currently alive,
         - death for <2 or >3 neighbors.

    4) The board size is a power of two and the input is bit-packed.
       We exploit that in two places:
         - row stride in words is a power of two, so row-major indexing uses a shift instead of a
           general multiply/divide,
         - the number of word tiles across a row is exact, so there are no partial x-tiles.

    5) Bit 0 and bit 63 cross-word handling:
         west neighbors: (center << 1) | (left  >> 63)
         east neighbors: (center >> 1) | (right << 63)
       This naturally imports the carry-in bit from the adjacent 64-bit word.
*/

namespace {

using u64 = std::uint64_t;

constexpr unsigned int kWarpWidth = 32u;
constexpr unsigned int kBlockRows = 8u;      // 32 x 8 = 256 threads per block
constexpr unsigned int kMaxBlocks = 2048u;   // enough to fully cover A100/H100-class GPUs
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

// Bit i of the returned value is the west neighbor for output bit i.
// Bit 0 imports bit 63 from the word on the left.
__device__ __forceinline__ u64 align_west(const u64 center, const u64 left) {
    return (center << 1) | (left >> 63);
}

// Bit i of the returned value is the east neighbor for output bit i.
// Bit 63 imports bit 0 from the word on the right.
__device__ __forceinline__ u64 align_east(const u64 center, const u64 right) {
    return (center >> 1) | (right << 63);
}

// Add one 64-bit neighbor bitboard into a bit-sliced counter.
//
// For every bit position independently:
//   - ones/twos hold the low two bits of the neighbor count modulo 4
//   - ge4 latches once the count reaches 4 or more
//
// This is effectively a carry chain for 1-bit increments, executed in parallel across all 64 cells.
__device__ __forceinline__ void accumulate_neighbor(const u64 neighbor,
                                                    u64& ones,
                                                    u64& twos,
                                                    u64& ge4) {
    const u64 carry_to_twos = ones & neighbor;
    ones ^= neighbor;
    ge4 |= twos & carry_to_twos;
    twos ^= carry_to_twos;
}

// Given the 3x3 word neighborhood around the current word:
//
//   n0 n1 n2
//   c0 c1 c2
//   s0 s1 s2
//
// compute the next-generation output word for c1.
//
// Note that c1 itself is not part of the neighbor count; it is only used in the final
// survival condition.
__device__ __forceinline__ u64 evolve_word(const u64 n0, const u64 n1, const u64 n2,
                                           const u64 c0, const u64 c1, const u64 c2,
                                           const u64 s0, const u64 s1, const u64 s2) {
    u64 ones = 0;
    u64 twos = 0;
    u64 ge4  = 0;

    // Cardinal neighbors.
    accumulate_neighbor(n1,                  ones, twos, ge4);
    accumulate_neighbor(s1,                  ones, twos, ge4);
    accumulate_neighbor(align_west(c1, c0),  ones, twos, ge4);
    accumulate_neighbor(align_east(c1, c2),  ones, twos, ge4);

    // Diagonal neighbors.
    accumulate_neighbor(align_west(n1, n0),  ones, twos, ge4);
    accumulate_neighbor(align_east(n1, n2),  ones, twos, ge4);
    accumulate_neighbor(align_west(s1, s0),  ones, twos, ge4);
    accumulate_neighbor(align_east(s1, s2),  ones, twos, ge4);

    // Conway rule application:
    //   count == 3 -> twos=1, ones=1, ge4=0
    //   count == 2 -> twos=1, ones=0, ge4=0, but only survives if c1 was alive
    return twos & (ones | c1) & ~ge4;
}

/*
    TileWidth == 32:
      - one warp handles one 32-word horizontal tile of one row

    TileWidth == 16:
      - one warp is split into two independent 16-lane subwarps via the shuffle width parameter,
        so the same warp handles two rows at once (the only narrow legal case is 16 words/row,
        i.e. a 1024x1024 board)

    grid.x covers the row exactly in TileWidth-word tiles.
    grid.y is capped, and the kernel grid-strides over rows.
*/
template <unsigned int TileWidth>
__global__ __launch_bounds__(256)
void game_of_life_shuffle_kernel(const u64* __restrict__ input,
                                 u64* __restrict__ output,
                                 const unsigned int rows,
                                 const unsigned int word_shift) {
    static_assert(TileWidth == 16u || TileWidth == 32u, "TileWidth must be 16 or 32.");

    constexpr unsigned int kGroupsPerWarp = kWarpWidth / TileWidth;

    // blockDim.x is always launched as 32.
    const unsigned int lane  = threadIdx.x & (TileWidth - 1u);
    const unsigned int group = threadIdx.x / TileWidth;

    // Exact x-tiling of the row; no partial tiles.
    const unsigned int col = blockIdx.x * TileWidth + lane;

    // In the 16-wide specialization, each warp contains two independent 16-lane groups.
    // Because the board height is a power of two, rows are even, so both groups enter/leave
    // the loop body together; using a full-warp mask is therefore valid.
    unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) * kGroupsPerWarp + group;
    const unsigned int row_step = blockDim.y * gridDim.y * kGroupsPerWarp;

    const size_t row_stride = size_t{1} << word_shift;

    for (; row < rows; row += row_step) {
        const size_t idx = (static_cast<size_t>(row) << word_shift) + static_cast<size_t>(col);

        const bool has_top    = row != 0u;
        const bool has_bottom = (row + 1u) < rows;

        // Left/right edge tests are tile/lane based. grid.x spans the row exactly.
        const bool has_left  = (lane != 0u) || (blockIdx.x != 0u);
        const bool has_right = (lane != (TileWidth - 1u)) || ((blockIdx.x + 1u) < gridDim.x);

        // Center column of the 3x3 word stencil.
        const u64 c1 = input[idx];

        size_t north = 0;
        size_t south = 0;
        u64 n1 = 0;
        u64 s1 = 0;

        if (has_top) {
            north = idx - row_stride;
            n1 = input[north];
        }

        if (has_bottom) {
            south = idx + row_stride;
            s1 = input[south];
        }

        // Left/right words usually come from neighboring lanes. Only tile-edge lanes need
        // direct global loads to bridge the shuffle boundary.
        u64 c0 = __shfl_up_sync  (kFullWarpMask, c1, 1, TileWidth);
        u64 c2 = __shfl_down_sync(kFullWarpMask, c1, 1, TileWidth);
        u64 n0 = __shfl_up_sync  (kFullWarpMask, n1, 1, TileWidth);
        u64 n2 = __shfl_down_sync(kFullWarpMask, n1, 1, TileWidth);
        u64 s0 = __shfl_up_sync  (kFullWarpMask, s1, 1, TileWidth);
        u64 s2 = __shfl_down_sync(kFullWarpMask, s1, 1, TileWidth);

        if (lane == 0u) {
            c0 = has_left ? input[idx - 1] : 0;
            n0 = (has_top    && has_left) ? input[north - 1] : 0;
            s0 = (has_bottom && has_left) ? input[south - 1] : 0;
        } else if (lane == (TileWidth - 1u)) {
            c2 = has_right ? input[idx + 1] : 0;
            n2 = (has_top    && has_right) ? input[north + 1] : 0;
            s2 = (has_bottom && has_right) ? input[south + 1] : 0;
        }

        output[idx] = evolve_word(n0, n1, n2, c0, c1, c2, s0, s1, s2);
    }
}

template <unsigned int TileWidth>
inline void launch_game_of_life_kernel(const u64* input,
                                       u64* output,
                                       const unsigned int rows,
                                       const unsigned int words_per_row,
                                       const unsigned int word_shift) {
    static_assert(TileWidth == 16u || TileWidth == 32u, "TileWidth must be 16 or 32.");

    constexpr unsigned int kGroupsPerWarp = kWarpWidth / TileWidth;

    // Exact x-coverage of the row. Because words_per_row is a power of two, this is also exact
    // for both TileWidth specializations.
    const unsigned int grid_x = words_per_row / TileWidth;

    // Keep the total block count around kMaxBlocks and let the kernel grid-stride over rows.
    // This avoids over-scheduling millions of tiny blocks while still providing enough work to
    // fully occupy modern data-center GPUs.
    const unsigned int logical_rows_per_block = kBlockRows * kGroupsPerWarp;
    const unsigned int needed_grid_y =
        (rows + logical_rows_per_block - 1u) / logical_rows_per_block;

    unsigned int max_grid_y = kMaxBlocks / grid_x;
    if (max_grid_y == 0u) {
        max_grid_y = 1u;
    }

    const unsigned int grid_y = (needed_grid_y < max_grid_y) ? needed_grid_y : max_grid_y;

    const dim3 block(kWarpWidth, kBlockRows);
    const dim3 grid(grid_x, grid_y);

    game_of_life_shuffle_kernel<TileWidth><<<grid, block>>>(input, output, rows, word_shift);
}

}  // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The problem statement guarantees:
    //   - square board
    //   - power-of-two dimensions
    //   - dimensions > 512
    //   - input/output are separate cudaMalloc allocations
    //
    // No synchronization is done here; the caller handles it.
    const unsigned int rows = static_cast<unsigned int>(grid_dimensions);
    if (rows == 0u) {
        return;
    }

    const unsigned int words_per_row = rows >> 6;  // 64 cells per 64-bit word

    // log2(words_per_row). words_per_row is guaranteed to be a power of two.
    unsigned int word_shift = 0u;
    for (unsigned int x = words_per_row; x > 1u; x >>= 1u) {
        ++word_shift;
    }

    // For 2048x2048 and larger boards, a 32-word warp tile is the natural mapping.
    // The only narrower legal case is 1024x1024 -> 16 words/row; the 16-wide specialization
    // handles that by splitting each warp into two independent 16-lane subwarps.
    if (words_per_row >= 32u) {
        launch_game_of_life_kernel<32u>(input, output, rows, words_per_row, word_shift);
    } else {
        launch_game_of_life_kernel<16u>(input, output, rows, words_per_row, word_shift);
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
