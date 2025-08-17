#include <cuda_runtime.h>
#include <cstdint>

// Conway's Game of Life, bit-packed (64 cells per 64-bit word).
// Each CUDA thread processes one 64-bit word (64 cells) and updates them in parallel using
// bitwise adders (SWAR). Neighbor counts are computed for all 64 cells simultaneously via
// trees of bitwise half/full adders, avoiding per-cell loops or atomics.
// 
// The grid is square (grid_dimensions x grid_dimensions), with rows packed into words:
//   words_per_row = grid_dimensions / 64
// The input/output arrays are sized accordingly and allocated on the device by the caller.
//
// Boundary handling: All cells outside the grid are dead. The 0th and 63rd bits within each word
// receive neighbor contributions from the adjacent words (left/right) in their respective rows,
// as well as from the rows above/below when computing diagonals. These cross-word contributions
// are handled explicitly by injecting the MSB/LSB bits from neighboring words into the shifts.
//
// No shared or texture memory is used; global memory loads are kept simple and coalesced by
// assigning one word per thread. H100/A100 have ample cache; redundant neighbor loads across
// threads are handled by the cache hierarchy efficiently.
//
// Performance strategy (summary):
// - Compute the 8 directional neighbor bitboards using word shifts plus cross-word bit injection.
// - Sum eight 1-bit bitboards into a 4-bit per-cell count using SWAR bitwise adders:
//     - Sum three directions from the above row into a 2-bit partial (a_lo, a_hi).
//     - Sum two directions from the same row (west/east) into a 2-bit partial (we_lo, we_hi).
//     - Sum three directions from the below row into a 2-bit partial (c_lo, c_hi).
//     - Add the two 2-bit partials (above + below) to create a 3-bit partial (ones, twos, fours).
//     - Add the same-row 2-bit partial to build the final 4-bit count (ones, twos, fours, eights).
// - Apply Life rules: next = (neighbors == 3) | (alive & (neighbors == 2)).
//   Equivalently, using the bit-sliced sum: 
//     neighbors == 3 -> ones & twos & ~fours & ~eights
//     neighbors == 2 -> ~ones & twos & ~fours & ~eights
//
// Notes:
// - This kernel executes one generation step. Host synchronization is delegated to the caller.
// - The grid dimension is guaranteed to be a power of 2 greater than 512, hence divisible by 64.

namespace {
    using u64 = std::uint64_t;

    // Sum of three 1-bit bitboards a + b + c -> (lo: LSB of sum, hi: MSB/carry)
    // For each bit position:
    //   lo = a ^ b ^ c
    //   hi = (a & b) | (a & c) | (b & c)
    static __device__ __forceinline__ void add3(u64 a, u64 b, u64 c, u64& lo, u64& hi) {
        u64 ab = a ^ b;
        lo = ab ^ c;
        hi = (a & b) | (a & c) | (b & c);
    }

    // Shift left by 1 and inject an external bit into bit 0 (LSB). in_bit must be 0 or 1.
    static __device__ __forceinline__ u64 shl1_in(u64 x, u64 in_bit) {
        return (x << 1) | (in_bit & 1ULL);
    }

    // Shift right by 1 and inject an external bit into bit 63 (MSB). in_bit must be 0 or 1.
    static __device__ __forceinline__ u64 shr1_in(u64 x, u64 in_bit) {
        return (x >> 1) | ((in_bit & 1ULL) << 63);
    }

    // CUDA kernel: one thread per 64-bit word.
    __global__ void life_step_kernel(const u64* __restrict__ in, u64* __restrict__ out, int grid_dimensions) {
        const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
        const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

        const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (tid >= total_words) return;

        const int row = static_cast<int>(tid / words_per_row);
        const int col = static_cast<int>(tid % words_per_row);

        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < words_per_row);
        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < grid_dimensions);

        const std::size_t idx = tid;

        // Load current row words
        const u64 Wc = in[idx];
        const u64 Wl = has_left  ? in[idx - 1] : 0ULL;
        const u64 Wr = has_right ? in[idx + 1] : 0ULL;

        // Load above row words (if any)
        u64 Ac = 0ULL, Al = 0ULL, Ar = 0ULL;
        if (has_up) {
            const std::size_t above_idx = idx - words_per_row;
            Ac = in[above_idx];
            if (has_left)  Al = in[above_idx - 1];
            if (has_right) Ar = in[above_idx + 1];
        }

        // Load below row words (if any)
        u64 Bc = 0ULL, Bl = 0ULL, Br = 0ULL;
        if (has_down) {
            const std::size_t below_idx = idx + words_per_row;
            Bc = in[below_idx];
            if (has_left)  Bl = in[below_idx - 1];
            if (has_right) Br = in[below_idx + 1];
        }

        // Current row horizontal neighbors (W and E), with cross-word bit injection at bit 0/63
        const u64 w_w = shl1_in(Wc, has_left ? (Wl >> 63) : 0ULL);                   // West contributions
        const u64 w_e = shr1_in(Wc, has_right ? (Wr & 1ULL) : 0ULL);                 // East contributions

        // Above row (NW, N, NE)
        u64 a_w = 0ULL, a_c = 0ULL, a_e = 0ULL;
        if (has_up) {
            a_c = Ac;                                                                 // North
            a_w = shl1_in(Ac, has_left ? (Al >> 63) : 0ULL);                          // North-West
            a_e = shr1_in(Ac, has_right ? (Ar & 1ULL) : 0ULL);                        // North-East
        }

        // Below row (SW, S, SE)
        u64 b_w = 0ULL, b_c = 0ULL, b_e = 0ULL;
        if (has_down) {
            b_c = Bc;                                                                 // South
            b_w = shl1_in(Bc, has_left ? (Bl >> 63) : 0ULL);                          // South-West
            b_e = shr1_in(Bc, has_right ? (Br & 1ULL) : 0ULL);                        // South-East
        }

        // Sum three contributions from above row (a_w, a_c, a_e) -> 2-bit partial (a_lo, a_hi)
        u64 a_lo, a_hi;
        add3(a_w, a_c, a_e, a_lo, a_hi);

        // Sum three contributions from below row (b_w, b_c, b_e) -> 2-bit partial (c_lo, c_hi)
        u64 c_lo, c_hi;
        add3(b_w, b_c, b_e, c_lo, c_hi);

        // Sum two contributions from current row (w_w, w_e) -> 2-bit partial (we_lo, we_hi)
        // For two 1-bit addends: lo = XOR, hi (carry) = AND
        const u64 we_lo = w_w ^ w_e;
        const u64 we_hi = w_w & w_e;

        // Add above and below 2-bit partials -> 3-bit partial (ones, twos, fours)
        // ones (bit0)
        const u64 ones_ac = a_lo ^ c_lo;
        const u64 carry01_ac = a_lo & c_lo;

        // twos (bit1) before final carry into fours
        const u64 tmp_ac = a_hi ^ c_hi;
        const u64 carry_from_hi_ac = a_hi & c_hi;

        const u64 twos_ac = tmp_ac ^ carry01_ac;
        const u64 carry12_ac = (tmp_ac & carry01_ac) | carry_from_hi_ac;

        // fours (bit2)
        const u64 fours_ac = carry12_ac;

        // Now add the same-row 2-bit partial (we_lo, we_hi) to the 3-bit partial (ones_ac, twos_ac, fours_ac)
        const u64 ones_sum = ones_ac ^ we_lo;
        const u64 carry01_sum = ones_ac & we_lo;

        const u64 tmp_we = twos_ac ^ we_hi;
        const u64 carry_twos_we = twos_ac & we_hi;

        const u64 twos_sum = tmp_we ^ carry01_sum;
        const u64 carry12_sum = (tmp_we & carry01_sum) | carry_twos_we;

        const u64 fours_sum = fours_ac ^ carry12_sum;
        const u64 eights_sum = fours_ac & carry12_sum;

        // Neighbor count bit-slices across 64 cells:
        //   ones_sum -> bit 0 of count
        //   twos_sum -> bit 1 of count
        //   fours_sum -> bit 2 of count
        //   eights_sum -> bit 3 of count (only nonzero for counts >= 8)
        //
        // Life rule: next = (neighbors == 3) | (alive & (neighbors == 2))
        const u64 alive = Wc;

        // neighbors == 3   -> ones=1, twos=1, fours=0, eights=0
        const u64 is3 = (ones_sum & twos_sum) & ~(fours_sum | eights_sum);

        // neighbors == 2   -> ones=0, twos=1, fours=0, eights=0
        const u64 is2 = (~ones_sum & twos_sum) & ~(fours_sum | eights_sum);

        out[idx] = is3 | (alive & is2);
    }
} // namespace

// Host API: run one step of Conway's Game of Life.
// - input:  device pointer to bit-packed input grid (1 bit per cell in row-major order).
// - output: device pointer to bit-packed output grid.
// - grid_dimensions: grid width and height (square), power of 2, > 512.
// Assumes input/output were allocated with cudaMalloc and are large enough.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Reasonable block size for memory-bound kernels on modern NVIDIA GPUs.
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total_words + block_size - 1) / block_size);

    life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
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
