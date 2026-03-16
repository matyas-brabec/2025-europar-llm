#include <cuda_runtime.h>
#include <cstdint>

/*
 * Optimized one-step Conway's Game of Life for the problem's native storage format.
 *
 * Storage / bit layout assumed by this implementation:
 *   - The grid is stored as a row-major array of 8x8 tiles.
 *   - Each std::uint64_t stores one tile.
 *   - Bit (row * 8 + col) encodes the cell at local tile coordinate (row, col).
 *   - Row 0 lives in the least-significant byte, row 7 in the most-significant byte.
 *   - Within each byte, bit 0 is column 0 and bit 7 is column 7.
 *
 * This makes each 64-bit word a "byte-per-row" bitboard:
 *   - vertical movement is a shift by 8 bits
 *   - horizontal movement is a shift by 1 bit, plus a byte-edge fixup from the neighboring tile
 *
 * Kernel strategy:
 *   - One CUDA thread computes one 8x8 output tile (64 cells).
 *   - The thread loads the 3x3 neighborhood of input tiles.
 *   - It constructs the eight aligned neighbor bitboards (N, S, W, E, NW, NE, SW, SE).
 *   - Those eight 1-bit planes are summed with a carry-save adder tree, producing the exact
 *     4-bit neighbor count for all 64 cells in parallel.
 *   - The Conway rule is then applied with pure bitwise logic.
 *
 * Shared memory is intentionally not used. For this packed representation the working set per tile
 * is already tiny, neighboring threads naturally reuse overlapping tiles through L1/L2, and keeping
 * everything in registers avoids synchronization and extra complexity.
 */

namespace {
using u64 = std::uint64_t;

/*
 * 128 threads is a good fit here:
 *   - it is the largest power-of-two block size that always divides every legal tile-row length
 *     (tiles_per_dim is a power of two and, because grid_dimensions > 512, tiles_per_dim >= 128)
 *   - that lets run_game_of_life choose a grid size whose grid-stride step is an exact integer
 *     number of tile rows, preserving row alignment across loop iterations
 *   - it also keeps enough blocks available for smaller legal grids
 */
constexpr int kBlockThreads = 128;
constexpr int kQueuedOccupancyWaves = 8;

constexpr u64 kCol0Mask    = 0x0101010101010101ULL;
constexpr u64 kCol7Mask    = 0x8080808080808080ULL;
constexpr u64 kNotCol0Mask = 0xFEFEFEFEFEFEFEFEULL;
constexpr u64 kNotCol7Mask = 0x7F7F7F7F7F7F7F7FULL;

__device__ __forceinline__ u64 load_ro(const u64* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

/* Align the north/south neighbor rows to the current tile's bit positions. */
__device__ __forceinline__ u64 align_north(u64 center, u64 north) {
    return (center << 8) | (north >> 56);
}

__device__ __forceinline__ u64 align_south(u64 center, u64 south) {
    return (center >> 8) | (south << 56);
}

/* Shift within 8-bit rows and stitch the missing edge bit from the adjacent tile. */
__device__ __forceinline__ u64 shift_west(u64 center, u64 west) {
    return ((center & kNotCol7Mask) << 1) | ((west & kCol7Mask) >> 7);
}

__device__ __forceinline__ u64 shift_east(u64 center, u64 east) {
    return ((center & kNotCol0Mask) >> 1) | ((east & kCol0Mask) << 7);
}

/*
 * Three-input carry-save adder on 64 independent bit positions.
 * For each bit position:
 *   sum   = low bit of a+b+c
 *   carry = high bit of a+b+c
 */
__device__ __forceinline__ void csa(u64& carry, u64& sum, u64 a, u64 b, u64 c) {
    const u64 u = a ^ b;
    carry = (a & b) | (u & c);
    sum = u ^ c;
}

__global__ void game_of_life_kernel(const u64* __restrict__ input,
                                    u64* __restrict__ output,
                                    unsigned int tiles_per_dim,
                                    u64 num_tiles) {
    const u64 tid = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 stride = static_cast<u64>(blockDim.x) * gridDim.x;

    const u64 row_stride = static_cast<u64>(tiles_per_dim);
    const u64 south_limit = num_tiles - row_stride;
    const unsigned int x_mask = tiles_per_dim - 1u;

    /*
     * run_game_of_life chooses gridDim.x so that stride is always an integer number of tile rows.
     * Therefore a thread's x coordinate is invariant across grid-stride iterations and can be hoisted
     * out of the loop.
     */
    const unsigned int x = static_cast<unsigned int>(tid) & x_mask;
    const bool has_w_static = (x != 0u);
    const bool has_e_static = (x != x_mask);
    const bool x_is_interior = has_w_static & has_e_static;

    for (u64 idx = tid; idx < num_tiles; idx += stride) {
        const bool has_n = (idx >= row_stride);
        const bool has_s = (idx < south_limit);
        const bool interior = x_is_interior & has_n & has_s;

        const u64 west_idx  = idx - 1;
        const u64 east_idx  = idx + 1;
        const u64 north_idx = idx - row_stride;
        const u64 south_idx = idx + row_stride;

        const u64 tC = load_ro(input + idx);

        u64 tNW, tN, tNE;
        u64 tW,  tE;
        u64 tSW, tS, tSE;

        if (interior) {
            /* Fast path: the overwhelmingly common case. No boundary predicates required. */
            tNW = load_ro(input + north_idx - 1);
            tN  = load_ro(input + north_idx);
            tNE = load_ro(input + north_idx + 1);

            tW  = load_ro(input + west_idx);
            tE  = load_ro(input + east_idx);

            tSW = load_ro(input + south_idx - 1);
            tS  = load_ro(input + south_idx);
            tSE = load_ro(input + south_idx + 1);
        } else {
            /* Slow path: cells outside the grid are defined as dead, so missing tiles are zero. */
            const bool has_nw = has_n & has_w_static;
            const bool has_ne = has_n & has_e_static;
            const bool has_sw = has_s & has_w_static;
            const bool has_se = has_s & has_e_static;

            tN  = has_n       ? load_ro(input + north_idx)     : 0;
            tS  = has_s       ? load_ro(input + south_idx)     : 0;
            tW  = has_w_static ? load_ro(input + west_idx)     : 0;
            tE  = has_e_static ? load_ro(input + east_idx)     : 0;

            tNW = has_nw      ? load_ro(input + north_idx - 1) : 0;
            tNE = has_ne      ? load_ro(input + north_idx + 1) : 0;
            tSW = has_sw      ? load_ro(input + south_idx - 1) : 0;
            tSE = has_se      ? load_ro(input + south_idx + 1) : 0;
        }

        /* Build the eight aligned neighbor planes. */
        const u64 north_bits = align_north(tC, tN);
        const u64 south_bits = align_south(tC, tS);
        const u64 west_bits  = shift_west(tC, tW);
        const u64 east_bits  = shift_east(tC, tE);

        const u64 nw_bits = shift_west(north_bits, align_north(tW, tNW));
        const u64 ne_bits = shift_east(north_bits, align_north(tE, tNE));
        const u64 sw_bits = shift_west(south_bits, align_south(tW, tSW));
        const u64 se_bits = shift_east(south_bits, align_south(tE, tSE));

        /*
         * Exact population count of the eight neighbor planes with a carry-save adder tree.
         * Final bitboards:
         *   ones   -> count bit 0
         *   twos   -> count bit 1
         *   fours  -> count bit 2
         *   eights -> count bit 3
         */
        const u64 s01 = north_bits ^ south_bits;
        const u64 c01 = north_bits & south_bits;

        const u64 s23 = west_bits ^ east_bits;
        const u64 c23 = west_bits & east_bits;

        const u64 s45 = nw_bits ^ ne_bits;
        const u64 c45 = nw_bits & ne_bits;

        const u64 s67 = sw_bits ^ se_bits;
        const u64 c67 = sw_bits & se_bits;

        const u64 s0123 = s01 ^ s23;
        const u64 k0123 = s01 & s23;
        u64 c0123, twos0123;
        csa(c0123, twos0123, c01, c23, k0123);

        const u64 s4567 = s45 ^ s67;
        const u64 k4567 = s45 & s67;
        u64 c4567, twos4567;
        csa(c4567, twos4567, c45, c67, k4567);

        const u64 ones = s0123 ^ s4567;
        const u64 ones_to_twos = s0123 & s4567;

        u64 twos_to_fours, twos;
        csa(twos_to_fours, twos, twos0123, twos4567, ones_to_twos);

        u64 eights, fours;
        csa(eights, fours, c0123, c4567, twos_to_fours);

        /*
         * Conway rule in bit-sliced form:
         *   - count == 3 => birth or survival
         *   - count == 2 => survival only if the current cell is already alive
         *
         * twos=1 with no higher count bits means the count is 2 or 3.
         * (ones | tC) then selects:
         *   - count 3: ones=1 -> always alive next
         *   - count 2: ones=0 -> alive next only if current cell (tC) is alive
         */
        const u64 eligible_counts = twos & ~(fours | eights);
        output[idx] = eligible_counts & (ones | tC);
    }
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const unsigned int tiles_per_dim = static_cast<unsigned int>(grid_dimensions) >> 3;
    const std::uint64_t tiles_per_dim64 = static_cast<std::uint64_t>(tiles_per_dim);
    const std::uint64_t num_tiles = tiles_per_dim64 * tiles_per_dim64;

    /*
     * Host-side launch tuning is intentionally lightweight and cached per host thread.
     * Error handling / synchronization is deliberately left to the caller, as requested.
     */
    thread_local int cached_device = -1;
    thread_local int cached_max_blocks = 0;

    int device = 0;
    (void)cudaGetDevice(&device);

    if (device != cached_device) {
        int sm_count = 0;
        int active_blocks = 0;

        (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        /* No shared memory is used, so prefer L1 for the overlapping 3x3-tile reads. */
        (void)cudaFuncSetCacheConfig(game_of_life_kernel, cudaFuncCachePreferL1);

        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, game_of_life_kernel, kBlockThreads, 0);

        if (active_blocks < 1) {
            active_blocks = 1;
        }

        cached_max_blocks = sm_count * active_blocks * kQueuedOccupancyWaves;
        if (cached_max_blocks < 1) {
            cached_max_blocks = 1;
        }

        cached_device = device;
    }

    /*
     * num_tiles is exactly divisible by 128 under the stated problem constraints:
     *   tiles_per_dim is a power of two and >= 128, so num_tiles = tiles_per_dim^2 is a multiple of 128.
     */
    std::uint64_t launch_blocks = num_tiles / static_cast<std::uint64_t>(kBlockThreads);

    if (launch_blocks > static_cast<std::uint64_t>(cached_max_blocks)) {
        /*
         * Keep gridDim.x * blockDim.x equal to a whole-number multiple of the tile-row length.
         * Since one tile row contains tiles_per_dim tiles and each block contributes kBlockThreads threads,
         * that means gridDim.x must be a multiple of tiles_per_dim / kBlockThreads blocks.
         *
         * This preserves row alignment across grid-stride iterations:
         *   stride = gridDim.x * blockDim.x = integer_multiple * tiles_per_dim
         * so each thread keeps the same x coordinate on every loop trip, which in turn preserves
         * coalesced row-wise accesses and lets the kernel hoist x/left/right state out of the loop.
         */
        const std::uint64_t blocks_per_tile_row =
            tiles_per_dim64 / static_cast<std::uint64_t>(kBlockThreads);

        std::uint64_t capped_blocks =
            static_cast<std::uint64_t>(cached_max_blocks);
        capped_blocks -= capped_blocks % blocks_per_tile_row;

        if (capped_blocks == 0) {
            capped_blocks = blocks_per_tile_row;
        }

        launch_blocks = capped_blocks;
    }

    game_of_life_kernel<<<static_cast<int>(launch_blocks), kBlockThreads>>>(
        input, output, tiles_per_dim, num_tiles);
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
// MEMORY_LAYOUT: TILES
