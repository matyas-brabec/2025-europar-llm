#include <cstdint>
#include <cuda_runtime.h>

/*
  High-throughput Conway step for the exact packed layout described in the prompt.

  Assumed memory layout (the natural interpretation of "one uint64_t stores one 8x8 tile"):
    - Tiles are stored in row-major order across the whole grid.
    - Inside each 64-bit tile, bit (8 * local_y + local_x) is the cell at (local_y, local_x).
    - local_x == 0 is the least-significant bit of each byte, local_x == 7 the most-significant.

  Design choices tuned for modern data-center GPUs (A100/H100 class):
    - One thread updates one 8x8 tile, i.e. 64 cells at once in pure bit-parallel form.
    - blockDim.x is fixed to 32 so each warp spans 32 consecutive tiles of one tile row.
      West/east neighbor tiles are therefore exchanged with warp shuffles instead of extra
      global loads; each thread only loads its N/C/S tiles from memory, plus two boundary
      threads per warp perform the missing edge loads.
    - No shared memory / texture memory: for this representation they add complexity without
      helping enough; the reduced load count plus normal caches are sufficient.
    - Neighbor counts are accumulated with a SWAR half-adder tree over eight 64-bit neighbor
      bit-planes. Only the low three count bits are required:
        count == bit0 + 2*bit1 + 4*bit2   (mod 8)
      The only discarded overflow case is count==8 -> 0 mod 8, which is harmless because
      Conway's rules only care about counts 2 and 3.
    - Input and output are assumed to be distinct ping-pong buffers.
*/

namespace {

using u64 = std::uint64_t;

constexpr int kTileShift = 3;   // 8x8 tile
constexpr int kBlockX    = 32;  // exactly one warp in x
constexpr int kSmallBlockY = 4; // keeps enough blocks for the smallest legal grid (1024x1024)
constexpr int kLargeBlockY = 8; // slightly lower launch overhead for larger grids

static_assert(kBlockX == 32, "This kernel relies on one warp spanning exactly one tile stripe.");

constexpr u64 kCol0Mask    = 0x0101010101010101ULL; // local_x == 0 in every row byte
constexpr u64 kCol7Mask    = 0x8080808080808080ULL; // local_x == 7 in every row byte
constexpr u64 kNotCol0Mask = 0xFEFEFEFEFEFEFEFEULL;
constexpr u64 kNotCol7Mask = 0x7F7F7F7F7F7F7F7FULL;

__device__ __forceinline__ u64 shfl_up_1(const u64 v) {
    return static_cast<u64>(
        __shfl_up_sync(0xFFFFFFFFu, static_cast<unsigned long long>(v), 1, 32)
    );
}

__device__ __forceinline__ u64 shfl_down_1(const u64 v) {
    return static_cast<u64>(
        __shfl_down_sync(0xFFFFFFFFu, static_cast<unsigned long long>(v), 1, 32)
    );
}

__device__ __forceinline__ void half_add(const u64 a, const u64 b, u64& sum, u64& carry) {
    sum   = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ u64 majority3(const u64 a, const u64 b, const u64 c) {
    // 3-input majority; modern compilers map this kind of boolean expression to LOP3.
    return (a & b) | (c & (a | b));
}

template <int BLOCK_Y>
__global__ __launch_bounds__(kBlockX * BLOCK_Y)
void game_of_life_kernel(const u64* __restrict__ input,
                         u64* __restrict__ output,
                         int tiles_per_dim) {
    static_assert(BLOCK_Y == kSmallBlockY || BLOCK_Y == kLargeBlockY,
                  "Only tuned block heights are supported.");

    const int tx = static_cast<int>(blockIdx.x * kBlockX + threadIdx.x);
    const int ty = static_cast<int>(blockIdx.y * BLOCK_Y + threadIdx.y);
    const int idx = ty * tiles_per_dim + tx;

    const int last_tile = tiles_per_dim - 1;
    const bool has_n = (ty != 0);
    const bool has_s = (ty != last_tile);
    const bool has_w = (tx != 0);
    const bool has_e = (tx != last_tile);

    const unsigned lane = threadIdx.x;

    // Center column of the 3x3 tile neighborhood. Top/bottom boundaries are zeroed
    // because cells outside the grid are defined to be dead.
    const u64 c = input[idx];
    const u64 n = has_n ? input[idx - tiles_per_dim] : 0ULL;
    const u64 s = has_s ? input[idx + tiles_per_dim] : 0ULL;

    // Warp-local west/east tiles. Because blockDim.x == 32, each warp is exactly one
    // horizontal tile stripe. Interior lanes get W/E from shuffle; lane 0 / 31 patch
    // the missing values with a single extra global load each.
    u64 c_w = shfl_up_1(c);
    u64 c_e = shfl_down_1(c);
    u64 n_w = shfl_up_1(n);
    u64 n_e = shfl_down_1(n);
    u64 s_w = shfl_up_1(s);
    u64 s_e = shfl_down_1(s);

    if (lane == 0u) {
        c_w = has_w ? input[idx - 1] : 0ULL;
        n_w = (has_w && has_n) ? input[idx - tiles_per_dim - 1] : 0ULL;
        s_w = (has_w && has_s) ? input[idx + tiles_per_dim - 1] : 0ULL;
    }
    if (lane == 31u) {
        c_e = has_e ? input[idx + 1] : 0ULL;
        n_e = (has_e && has_n) ? input[idx - tiles_per_dim + 1] : 0ULL;
        s_e = (has_e && has_s) ? input[idx + tiles_per_dim + 1] : 0ULL;
    }

    // Align the eight neighbor directions to the current tile's 64 bit positions.
    //
    // north/south:
    //   shift by 8 bits (one row) and inject the bordering row from the adjacent tile.
    //
    // west/east:
    //   shift within each byte (one column) while masking row wrap, and inject the
    //   bordering column from the adjacent tile.
    //
    // diagonals:
    //   derived from aligned north/south plus the missing border column coming from the
    //   west/east or diagonal tiles.
    const u64 north = (c << 8) | (n >> 56);
    const u64 south = (c >> 8) | (s << 56);

    // Pair 0: north + south.
    u64 lo0, hi0;
    half_add(north, south, lo0, hi0);

    const u64 cw7 = c_w & kCol7Mask;
    const u64 ce0 = c_e & kCol0Mask;

    // Pair 1: west + east.
    u64 lo1, hi1;
    {
        const u64 west = ((c & kNotCol7Mask) << 1) | (cw7 >> 7);
        const u64 east = ((c & kNotCol0Mask) >> 1) | (ce0 << 7);
        half_add(west, east, lo1, hi1);
    }

    const u64 north_l = (north & kNotCol7Mask) << 1; // NW internal part
    const u64 north_r = (north & kNotCol0Mask) >> 1; // NE internal part
    const u64 south_l = (south & kNotCol7Mask) << 1; // SW internal part
    const u64 south_r = (south & kNotCol0Mask) >> 1; // SE internal part

    // Pair 2: northwest + northeast.
    u64 lo2, hi2;
    {
        const u64 northwest = north_l | (cw7 << 1) | ((n_w & kCol7Mask) >> 63);
        const u64 northeast = north_r | (ce0 << 15) | ((n_e & kCol0Mask) >> 49);
        half_add(northwest, northeast, lo2, hi2);
    }

    // Pair 3: southwest + southeast.
    u64 lo3, hi3;
    {
        const u64 southwest = south_l | (cw7 >> 15) | ((s_w & kCol7Mask) << 49);
        const u64 southeast = south_r | (ce0 >> 1) | ((s_e & kCol0Mask) << 63);
        half_add(southwest, southeast, lo3, hi3);
    }

    // Sum the four 2-bit partial sums:
    //   lo* are the weight-1 bits, hi* are the weight-2 bits.
    //
    // Final count bits are:
    //   count = count0 + 2*count1 + 4*count2   (mod 8)
    //
    // The discarded bit-3 overflow only represents count==8, which is irrelevant for
    // Conway's rule evaluation because only counts 2 and 3 can produce a live output.
    u64 s01, c01;
    u64 s23, c23;
    half_add(lo0, lo1, s01, c01);
    half_add(lo2, lo3, s23, c23);

    const u64 count0 = s01 ^ s23;
    const u64 carry0 = s01 & s23;
    const u64 mid1   = carry0 ^ c01 ^ c23;
    const u64 mid2   = majority3(carry0, c01, c23);

    u64 t01, u01;
    u64 t23, u23;
    half_add(hi0, hi1, t01, u01);
    half_add(hi2, hi3, t23, u23);

    const u64 q0 = t01 ^ t23;
    const u64 q1 = (t01 & t23) ^ u01 ^ u23;

    const u64 count1 = mid1 ^ q0;
    const u64 count2 = mid2 ^ q1 ^ (mid1 & q0);

    // Conway's rule in bit-sliced form:
    //   next = (count == 3) | (current & (count == 2))
    //
    // Using the count bit-planes:
    //   count==2 or 3  <=>  count1==1 and count2==0
    //   count==3       <=>  above and count0==1
    //
    // So:
    //   next = (count1 & ~count2) & (count0 | current)
    const u64 two_or_three = count1 & ~count2;
    output[idx] = two_or_three & (count0 | c);
}

template <int BLOCK_Y>
inline void launch_game_of_life_kernel(const u64* input, u64* output, int tiles_per_dim) {
    const dim3 block(kBlockX, BLOCK_Y, 1);
    const dim3 grid(
        static_cast<unsigned int>(tiles_per_dim / kBlockX),
        static_cast<unsigned int>(tiles_per_dim / BLOCK_Y),
        1
    );
    game_of_life_kernel<BLOCK_Y><<<grid, block>>>(input, output, tiles_per_dim);
}

} // namespace

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions) {
    // The prompt guarantees:
    //   - square grid
    //   - power-of-two dimensions
    //   - grid_dimensions > 512
    //
    // Therefore tiles_per_dim = grid_dimensions / 8 is also a power of two and is
    // always divisible by both 32 and the chosen BLOCK_Y, so the launch needs no tail
    // predicates and the kernel contains no output bounds checks.
    const int tiles_per_dim = grid_dimensions >> kTileShift;

    // Smallest legal grid is 1024x1024 => 128x128 tiles.
    // With BLOCK_Y=8 that would be only 64 blocks total, which underutilizes A100/H100.
    // Use a 4-row block there; otherwise use 8-row blocks to slightly reduce launch and
    // scheduling overhead.
    if (tiles_per_dim <= 128) {
        launch_game_of_life_kernel<kSmallBlockY>(input, output, tiles_per_dim);
    } else {
        launch_game_of_life_kernel<kLargeBlockY>(input, output, tiles_per_dim);
    }

    // Intentionally no synchronization and no error checking here:
    // the caller is responsible for synchronization, and avoiding extra host overhead
    // keeps the step launch as lean as possible.
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
