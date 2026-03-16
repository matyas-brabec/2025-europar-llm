#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// CUDA's shuffle intrinsics are guaranteed to support unsigned long long as a 64-bit lane value.
// std::uint64_t may alias either unsigned long or unsigned long long depending on platform, so we
// use unsigned long long internally and convert at the load/store boundaries.
using word_t = unsigned long long;

constexpr unsigned int kWarpSize   = 32u;
constexpr unsigned int kBlockShift = 8u;          // 2^8 = 256 threads/block
constexpr unsigned int kBlockSize  = 1u << kBlockShift;
constexpr unsigned int kFullMask   = 0xFFFFFFFFu;

static_assert(sizeof(word_t) == 8, "word_t must be 64-bit");
static_assert(sizeof(std::uint64_t) == 8, "std::uint64_t must be 64-bit");
static_assert(kBlockSize % kWarpSize == 0u, "Block size must be warp-aligned");

// Simple typed load helper. Kept separate only because the kernel uses word_t internally.
__device__ __forceinline__ word_t load_word(const std::uint64_t* ptr) {
    return static_cast<word_t>(*ptr);
}

// Bit-sliced 3:2 compressor / full adder.
// For every bit position independently:
//   a + b + c = sum + 2 * carry
// with:
//   sum   = a ^ b ^ c
//   carry = majority(a, b, c)
__device__ __forceinline__ void csa3(
    const word_t a,
    const word_t b,
    const word_t c,
    word_t& sum,
    word_t& carry)
{
    const word_t ab = a ^ b;
    sum   = ab ^ c;
    carry = (a & b) | (c & ab);
}

// Compute one output 64-cell word from the 3x3 word neighborhood around it.
//
// Bit layout assumption (matching the prompt's "0th" and "63rd" bit wording):
// - bit 0  = leftmost cell of the 64-cell chunk
// - bit 63 = rightmost cell of the 64-cell chunk
//
// Therefore:
// - shifting a word left aligns west-neighbor bits to the destination cells
// - shifting a word right aligns east-neighbor bits to the destination cells
// - the carry-in for bit 0 comes from bit 63 of the word to the left
// - the carry-in for bit 63 comes from bit 0 of the word to the right
__device__ __forceinline__ word_t evolve_word(
    const word_t center,
    const word_t left,
    const word_t right,
    const word_t up_left,
    const word_t up,
    const word_t up_right,
    const word_t down_left,
    const word_t down,
    const word_t down_right)
{
    // Align horizontal neighbors to the destination bit positions.
    const word_t west       = (center << 1) | (left >> 63);
    const word_t east       = (center >> 1) | (right << 63);
    const word_t north_west = (up << 1) | (up_left >> 63);
    const word_t north_east = (up >> 1) | (up_right << 63);
    const word_t south_west = (down << 1) | (down_left >> 63);
    const word_t south_east = (down >> 1) | (down_right << 63);

    // Compress each neighbor row horizontally:
    //   top = top0 + 2*top1   for {NW, N, NE}
    //   mid = mid0 + 2*mid1   for {W, E}
    //   bot = bot0 + 2*bot1   for {SW, S, SE}
    word_t top0, top1;
    csa3(north_west, up, north_east, top0, top1);

    const word_t mid0 = west ^ east;
    const word_t mid1 = west & east;

    word_t bot0, bot1;
    csa3(south_west, down, south_east, bot0, bot1);

    // Add the three 2-bit row counts in bit-sliced form.
    // Result is encoded as bit0/bit1/bit2/bit3 = 1/2/4/8 place of the neighbor count.
    word_t bit0, carry0;
    csa3(top0, mid0, bot0, bit0, carry0);

    word_t twos0, twos1;
    csa3(top1, mid1, bot1, twos0, twos1);

    const word_t bit1   = twos0 ^ carry0;
    const word_t carry1 = twos0 & carry0;
    const word_t bit2   = twos1 ^ carry1;
    const word_t bit3   = twos1 & carry1;

    // Conway update:
    // - survive on exactly 2 neighbors
    // - born on exactly 3 neighbors
    //
    // Using the bit-sliced count:
    //   count == 2 -> bit1=1, bit0=0, bit2=0, bit3=0
    //   count == 3 -> bit1=1, bit0=1, bit2=0, bit3=0
    //
    // This simplifies to:
    //   next = (center | bit0) & bit1 & ~(bit2 | bit3)
    return (center | bit0) & bit1 & ~(bit2 | bit3);
}

// Power-of-two integer log2 used on the host. The input is guaranteed to be a power of two.
inline unsigned int ilog2_pow2(unsigned int v) {
    unsigned int r = 0;
    while (v > 1u) {
        v >>= 1u;
        ++r;
    }
    return r;
}

// A 1D launch is deliberate:
// - it avoids the 2D grid.y <= 65535 limit for very large boards
// - words_per_row is a power of two, so x/y decode is just mask + shift
// - with a warp-aligned block size, warps naturally cover contiguous row segments
//
// Each thread owns exactly one 64-bit packed word, so no atomics are needed.
__global__ __launch_bounds__(256) void game_of_life_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    const unsigned int words_per_row,
    const unsigned int row_mask,
    const unsigned int last_row,
    const int row_shift)
{
    // Under the stated constraints:
    //   N = 2^m, words_per_row = N / 64 = 2^(m-6), total_words = 2^(2m-6)
    // and N > 512 => total_words is divisible by 256.
    //
    // So the host launches an exact multiple of kBlockSize threads and there is no tail block.
    const std::size_t idx = (static_cast<std::size_t>(blockIdx.x) << kBlockShift) + threadIdx.x;

    // Recover word coordinates with bit ops instead of integer div/mod.
    const unsigned int x = static_cast<unsigned int>(idx & row_mask);
    const unsigned int y = static_cast<unsigned int>(idx >> row_shift);

    const bool has_left  = (x != 0u);
    const bool has_right = (x != row_mask);
    const bool has_up    = (y != 0u);
    const bool has_down  = (y != last_row);

    // Precompute indices used by the boundary-path fallback loads.
    const std::size_t idx_left  = idx - 1u;
    const std::size_t idx_right = idx + 1u;
    const std::size_t idx_up    = idx - words_per_row;
    const std::size_t idx_down  = idx + words_per_row;

    // Load only the three vertically aligned words directly.
    // Horizontal neighbors are usually supplied by adjacent lanes via warp shuffles.
    const word_t center = load_word(input + idx);

    word_t up = 0ull;
    if (has_up) {
        up = load_word(input + idx_up);
    }

    word_t down = 0ull;
    if (has_down) {
        down = load_word(input + idx_down);
    }

    // Warp-local reuse of adjacent words is cheaper than shared memory here.
    //
    // For words_per_row >= 32, each warp covers one contiguous 32-word row segment.
    // For the only smaller legal case (words_per_row == 16), a warp spans two rows, but
    // the x-boundary predicates below ensure that the cross-row values at the 16-lane seam
    // are never consumed.
    const unsigned int lane          = threadIdx.x & (kWarpSize - 1u);
    const bool left_in_warp_segment  = (lane != 0u);
    const bool right_in_warp_segment = (lane != (kWarpSize - 1u));

    const word_t center_prev = __shfl_up_sync(kFullMask, center, 1);
    const word_t center_next = __shfl_down_sync(kFullMask, center, 1);
    const word_t up_prev     = __shfl_up_sync(kFullMask, up, 1);
    const word_t up_next     = __shfl_down_sync(kFullMask, up, 1);
    const word_t down_prev   = __shfl_up_sync(kFullMask, down, 1);
    const word_t down_next   = __shfl_down_sync(kFullMask, down, 1);

    // Outside-grid words are treated as zero. Only true row boundaries or warp-edge lanes need
    // fallback global loads; all other left/right words come from neighboring lanes.
    word_t left = 0ull;
    if (has_left) {
        left = left_in_warp_segment ? center_prev : load_word(input + idx_left);
    }

    word_t right = 0ull;
    if (has_right) {
        right = right_in_warp_segment ? center_next : load_word(input + idx_right);
    }

    word_t up_left = 0ull;
    if (has_up && has_left) {
        up_left = left_in_warp_segment ? up_prev : load_word(input + idx_up - 1u);
    }

    word_t up_right = 0ull;
    if (has_up && has_right) {
        up_right = right_in_warp_segment ? up_next : load_word(input + idx_up + 1u);
    }

    word_t down_left = 0ull;
    if (has_down && has_left) {
        down_left = left_in_warp_segment ? down_prev : load_word(input + idx_down - 1u);
    }

    word_t down_right = 0ull;
    if (has_down && has_right) {
        down_right = right_in_warp_segment ? down_next : load_word(input + idx_down + 1u);
    }

    const word_t next = evolve_word(
        center,
        left,
        right,
        up_left,
        up,
        up_right,
        down_left,
        down,
        down_right);

    output[idx] = static_cast<std::uint64_t>(next);
}

} // namespace

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // The problem statement guarantees:
    // - square board
    // - power-of-two dimension
    // - dimension > 512
    // - input/output are device pointers allocated with cudaMalloc
    //
    // Those assumptions are used directly for speed:
    // - no runtime validation
    // - no tail handling in the kernel
    // - no host-side synchronization or error polling here
    //
    // input and output are assumed to be distinct ping-pong buffers.
    const unsigned int dim           = static_cast<unsigned int>(grid_dimensions);
    const unsigned int words_per_row = dim >> 6;                 // 64 cells per packed word
    const unsigned int row_shift     = ilog2_pow2(words_per_row);
    const std::size_t total_words    = static_cast<std::size_t>(dim) * static_cast<std::size_t>(words_per_row);

    // Exact launch: total_words is always divisible by 256 under the stated constraints.
    const dim3 block(kBlockSize, 1u, 1u);
    const dim3 grid(static_cast<unsigned int>(total_words >> kBlockShift), 1u, 1u);

    game_of_life_kernel<<<grid, block>>>(
        input,
        output,
        words_per_row,
        words_per_row - 1u,
        dim - 1u,
        static_cast<int>(row_shift));
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
