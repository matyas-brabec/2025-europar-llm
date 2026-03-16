#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// The hot step kernel operates on 32 packed cells at a time.
// The prompt explicitly allows internal data transformations inside run_game_of_life,
// so we convert the caller's byte-per-cell bool grid into a 1-bit-per-cell layout,
// run the actual Life update there, then expand back to bool.
using word_t = unsigned int;

constexpr int kWordBits  = 32;
constexpr int kWordShift = 5;

// Transform kernels: simple, bandwidth-bound, warp-centric.
// One warp packs or unpacks one 32-cell group with fully coalesced byte accesses
// using ballot/shuffle. 256 threads = 8 warps per block.
constexpr int kTransformWarpsPerBlock = 8;
constexpr int kTransformThreads       = kTransformWarpsPerBlock * kWordBits;
constexpr int kTransformBlocksPerSM   = 8;

// Step kernel geometry.
// x is exactly one warp (32 packed words) so each warp covers a contiguous horizontal span.
// y is 16 rows, giving 512-thread blocks: good halo amortization without going to 1024 threads.
// Because grid_dimensions is a power of two > 512, all legal sizes divide these tile sizes exactly.
constexpr int kStepBlockWords = kWordBits;
constexpr int kStepBlockRows  = 16;

constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

static_assert(sizeof(word_t) == 4, "word_t must be 32 bits");
static_assert(kTransformThreads % kWordBits == 0, "Transform block size must be a whole number of warps");
static_assert(kStepBlockWords == kWordBits, "Step kernel assumes exactly one warp in x");

// Align the west neighbor of each bit to the current cell's bit position.
// For bit i, result bit i becomes original bit i-1; bit 0 is filled from the previous word's MSB.
// NVCC generally lowers this idiom to a single SHF instruction on modern GPUs.
__device__ __forceinline__ word_t shift_from_left(word_t center, word_t left_word) {
    return (center << 1) | (left_word >> (kWordBits - 1));
}

// Align the east neighbor of each bit to the current cell's bit position.
// For bit i, result bit i becomes original bit i+1; bit 31 is filled from the next word's LSB.
__device__ __forceinline__ word_t shift_from_right(word_t center, word_t right_word) {
    return (center >> 1) | (right_word << (kWordBits - 1));
}

__device__ __forceinline__ void half_adder(word_t a, word_t b, word_t& sum, word_t& carry) {
    sum   = a ^ b;
    carry = a & b;
}

__device__ __forceinline__ void full_adder(word_t a, word_t b, word_t c, word_t& sum, word_t& carry) {
    const word_t axb = a ^ b;
    sum   = axb ^ c;
    carry = (a & b) | (c & axb);
}

// Compute one packed Life update for 32 cells in parallel.
//
// The eight neighbor masks are bitboards: bit i in each mask corresponds to neighbor presence
// for cell i.  We count neighbors with a carry-save adder tree:
//
//   8 one-bit inputs -> binary count bits for each lane
//
// We do not materialize the full 4-bit count explicitly because the Life rule only needs
// to distinguish:
//   - count == 2
//   - count == 3
//   - count >= 4
//
// After the final half-adder:
//   bit0         = count's 1s bit
//   bit1         = count's 2s bit
//   (e2 | d2)    = count >= 4
//
// Life rule:
//   next = (count in {2,3}) && (alive || count == 3)
//        = bit1 && !(count >= 4) && (alive || bit0)
__device__ __forceinline__ word_t life_word(
    word_t alive,
    word_t nw, word_t n, word_t ne,
    word_t w,              word_t e,
    word_t sw, word_t s, word_t se)
{
    word_t s1, c1;
    word_t s2, c2;
    word_t s3, c3;
    full_adder(nw, n, ne, s1, c1);
    full_adder(w,  e, sw, s2, c2);
    half_adder(s, se, s3, c3);

    word_t bit0, d1;
    full_adder(s1, s2, s3, bit0, d1);

    word_t t2, d2;
    full_adder(c1, c2, c3, t2, d2);

    word_t bit1, e2;
    half_adder(t2, d1, bit1, e2);

    const word_t two_or_three = bit1 & ~(e2 | d2);
    return two_or_three & (bit0 | alive);
}

// Pack 32 consecutive bool cells into one 32-bit word.
//
// Because every row length is a multiple of 32, the flat bool array can be partitioned into
// 32-cell groups without ever crossing a row boundary.  That means the transform kernels can
// ignore 2D indexing entirely and just operate on a linear array of packed words.
__global__ __launch_bounds__(kTransformThreads)
void pack_bool_to_words_kernel(const bool* __restrict__ input,
                               word_t* __restrict__ packed,
                               int total_words)
{
    const int lane        = static_cast<int>(threadIdx.x) & (kWordBits - 1);
    const int warp_in_blk = static_cast<int>(threadIdx.x) >> kWordShift;
    const int warp_idx    = static_cast<int>(blockIdx.x) * kTransformWarpsPerBlock + warp_in_blk;
    const int warp_stride = static_cast<int>(gridDim.x) * kTransformWarpsPerBlock;

    for (int linear_word = warp_idx; linear_word < total_words; linear_word += warp_stride) {
        const std::size_t cell_base = static_cast<std::size_t>(linear_word) << kWordShift;
        const word_t bits = __ballot_sync(kFullWarpMask, input[cell_base + lane]);
        if (lane == 0) {
            packed[linear_word] = bits;
        }
    }
}

// Unpack one 32-bit word back into 32 bool cells.
// Lane 0 loads the packed word once, then broadcasts it to the warp with shuffle.
__global__ __launch_bounds__(kTransformThreads)
void unpack_words_to_bool_kernel(const word_t* __restrict__ packed,
                                 bool* __restrict__ output,
                                 int total_words)
{
    const int lane        = static_cast<int>(threadIdx.x) & (kWordBits - 1);
    const int warp_in_blk = static_cast<int>(threadIdx.x) >> kWordShift;
    const int warp_idx    = static_cast<int>(blockIdx.x) * kTransformWarpsPerBlock + warp_in_blk;
    const int warp_stride = static_cast<int>(gridDim.x) * kTransformWarpsPerBlock;

    for (int linear_word = warp_idx; linear_word < total_words; linear_word += warp_stride) {
        word_t bits = 0;
        if (lane == 0) {
            bits = packed[linear_word];
        }
        bits = __shfl_sync(kFullWarpMask, bits, 0);

        const std::size_t cell_base = static_cast<std::size_t>(linear_word) << kWordShift;
        output[cell_base + lane] = ((bits >> lane) & 1u) != 0u;
    }
}

// Actual measured hot kernel.
//
// Each thread updates one packed 32-cell word.
// A 32x16 block stages an 18x34 shared-memory tile:
//   (block_rows + 2 halo) x (block_words + 2 halo)
//
// This cuts global traffic dramatically: neighboring packed words and neighboring rows
// are reused from shared memory rather than reloaded per output word.  Dead-outside-grid
// semantics are handled by zero-filling the halo at global boundaries.
__global__ __launch_bounds__(kStepBlockWords * kStepBlockRows)
void gol_step_packed_kernel(const word_t* __restrict__ in,
                            word_t* __restrict__ out,
                            int words_per_row)
{
    __shared__ word_t tile[kStepBlockRows + 2][kStepBlockWords + 2];

    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int wx  = bx * kStepBlockWords + tx;
    const int row = by * kStepBlockRows  + ty;
    const int idx = row * words_per_row + wx;

    const int sx = tx + 1;
    const int sy = ty + 1;

    const bool has_left_block   = (bx != 0);
    const bool has_right_block  = (bx + 1 != static_cast<int>(gridDim.x));
    const bool has_top_block    = (by != 0);
    const bool has_bottom_block = (by + 1 != static_cast<int>(gridDim.y));

    // Center of the tile: always valid because the problem guarantees exact divisibility
    // by the chosen block shape.
    tile[sy][sx] = in[idx];

    // Left/right halo words.
    if (tx == 0) {
        tile[sy][0] = has_left_block ? in[idx - 1] : 0u;
    }
    if (tx == kStepBlockWords - 1) {
        tile[sy][kStepBlockWords + 1] = has_right_block ? in[idx + 1] : 0u;
    }

    // Top/bottom halo rows.
    if (ty == 0) {
        tile[0][sx] = has_top_block ? in[idx - words_per_row] : 0u;
    }
    if (ty == kStepBlockRows - 1) {
        tile[kStepBlockRows + 1][sx] = has_bottom_block ? in[idx + words_per_row] : 0u;
    }

    // Four corners of the halo.
    if (tx == 0 && ty == 0) {
        tile[0][0] = (has_left_block && has_top_block) ? in[idx - words_per_row - 1] : 0u;
    }
    if (tx == kStepBlockWords - 1 && ty == 0) {
        tile[0][kStepBlockWords + 1] =
            (has_right_block && has_top_block) ? in[idx - words_per_row + 1] : 0u;
    }
    if (tx == 0 && ty == kStepBlockRows - 1) {
        tile[kStepBlockRows + 1][0] =
            (has_left_block && has_bottom_block) ? in[idx + words_per_row - 1] : 0u;
    }
    if (tx == kStepBlockWords - 1 && ty == kStepBlockRows - 1) {
        tile[kStepBlockRows + 1][kStepBlockWords + 1] =
            (has_right_block && has_bottom_block) ? in[idx + words_per_row + 1] : 0u;
    }

    __syncthreads();

    const word_t north = tile[sy - 1][sx];
    const word_t self  = tile[sy][sx];
    const word_t south = tile[sy + 1][sx];

    const word_t nw = shift_from_left (north, tile[sy - 1][sx - 1]);
    const word_t n  = north;
    const word_t ne = shift_from_right(north, tile[sy - 1][sx + 1]);

    const word_t w  = shift_from_left (self,  tile[sy][sx - 1]);
    const word_t e  = shift_from_right(self,  tile[sy][sx + 1]);

    const word_t sw = shift_from_left (south, tile[sy + 1][sx - 1]);
    const word_t s  = south;
    const word_t se = shift_from_right(south, tile[sy + 1][sx + 1]);

    out[idx] = life_word(self, nw, n, ne, w, e, sw, s, se);
}

} // namespace

void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // The caller owns synchronization.  This function therefore stays fully asynchronous and
    // uses stream-ordered allocation/free on the default stream rather than any synchronizing path.
    cudaStream_t stream = 0;

    // Packed row length in 32-bit words.  Since grid_dimensions is a power of two > 512,
    // this is also a power of two and at least 32.
    const int words_per_row = grid_dimensions >> kWordShift;
    const int total_words   = grid_dimensions * words_per_row;
    const std::size_t packed_bytes = static_cast<std::size_t>(total_words) * sizeof(word_t);

    // One contiguous temp allocation holds both packed input and packed output.
    word_t* packed_storage = nullptr;
    cudaMallocAsync(&packed_storage, packed_bytes * 2, stream);

    word_t* packed_input  = packed_storage;
    word_t* packed_output = packed_storage + total_words;

    // Transform kernels are simple and bandwidth-bound.  A modest number of blocks per SM
    // is enough to saturate memory while avoiding silly launch geometry for huge boards.
    int device   = 0;
    int sm_count = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int max_useful_transform_blocks =
        (total_words + kTransformWarpsPerBlock - 1) / kTransformWarpsPerBlock;

    int transform_blocks = sm_count * kTransformBlocksPerSM;
    if (transform_blocks > max_useful_transform_blocks) transform_blocks = max_useful_transform_blocks;
    if (transform_blocks < 1) transform_blocks = 1;

    // bool -> packed bits
    pack_bool_to_words_kernel<<<transform_blocks, kTransformThreads, 0, stream>>>(
        input, packed_input, total_words);

    // Packed Life step
    const dim3 step_block(kStepBlockWords, kStepBlockRows);
    const dim3 step_grid(
        static_cast<unsigned>(words_per_row / kStepBlockWords),
        static_cast<unsigned>(grid_dimensions / kStepBlockRows));

    gol_step_packed_kernel<<<step_grid, step_block, 0, stream>>>(
        packed_input, packed_output, words_per_row);

    // packed bits -> bool
    unpack_words_to_bool_kernel<<<transform_blocks, kTransformThreads, 0, stream>>>(
        packed_output, output, total_words);

    // Safe without synchronization because the free is ordered after all prior work in the stream.
    cudaFreeAsync(packed_storage, stream);
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
