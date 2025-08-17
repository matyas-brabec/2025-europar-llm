#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// Each 64-bit word encodes 64 consecutive cells in a row (bit 0 = first cell).
// Each CUDA thread processes exactly one 64-bit word (64 cells).
//
// Design highlights:
// - Memory layout: grid is a square of dimension N x N cells; each row has (N/64) 64-bit words.
// - Boundary handling: cells outside the grid are treated as dead; we guard all out-of-bounds loads.
// - Neighbor computation: use bit-parallel shifts with cross-word carry-in/out for bit 0 and 63.
// - Neighbor counting: use a carry-save adder (CSA) tree to sum the 8 neighbor bit-vectors.
//   For three bit-vectors a, b, c (each bit is 0/1 for a cell), CSA produces:
//     sum = a ^ b ^ c                 (bitwise XOR)
//     carry = (a & b) | (a & c) | (b & c)  (bitwise 'majority of three')
//   Summing 8 inputs with CSAs reduces to 4 bit-planes (b0..b3) of the per-cell neighbor count.
// - Life rule application: next = (count == 3) | (alive & (count == 2)) for each bit position.
//
// Note: Shared/texture memory is intentionally not used; simple global memory loads achieve high bandwidth
// due to coalescing when each thread processes one consecutive word.

static __device__ __forceinline__ void csa3_u64(std::uint64_t a,
                                                std::uint64_t b,
                                                std::uint64_t c,
                                                std::uint64_t& sum,
                                                std::uint64_t& carry)
{
    // Full adder for three 64-bit bit-vectors:
    // - sum = bitwise XOR of inputs
    // - carry = bitwise majority (1 if at least two inputs have a 1 at that bit)
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int words_per_row,
                                int height,
                                int row_shift,             // log2(words_per_row)
                                std::size_t total_words)   // total number of 64-bit words in the grid
{
    // Grid-stride loop over all words
    for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total_words;
         tid += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    {
        // Compute (row, col) of this word using power-of-two arithmetic
        const int col = static_cast<int>(tid & static_cast<std::size_t>(words_per_row - 1));
        const int row = static_cast<int>(tid >> row_shift);

        // Boundary flags
        const bool has_left  = (col > 0);
        const bool has_right = (col + 1 < words_per_row);
        const bool has_up    = (row > 0);
        const bool has_down  = (row + 1 < height);

        // Base indices for neighbor rows
        const std::size_t idx      = tid;
        const std::size_t idx_up   = idx - static_cast<std::size_t>(words_per_row);
        const std::size_t idx_down = idx + static_cast<std::size_t>(words_per_row);

        // Load center and neighbor words from global memory (guarded at boundaries).
        const std::uint64_t c  = in[idx];
        const std::uint64_t l  = has_left  ? in[idx - 1] : 0ULL;
        const std::uint64_t r  = has_right ? in[idx + 1] : 0ULL;

        const std::uint64_t n  = has_up    ? in[idx_up] : 0ULL;
        const std::uint64_t s  = has_down  ? in[idx_down] : 0ULL;

        const std::uint64_t nl = (has_up   && has_left)  ? in[idx_up - 1] : 0ULL;
        const std::uint64_t nr = (has_up   && has_right) ? in[idx_up + 1] : 0ULL;
        const std::uint64_t sl = (has_down && has_left)  ? in[idx_down - 1] : 0ULL;
        const std::uint64_t sr = (has_down && has_right) ? in[idx_down + 1] : 0ULL;

        // Build the eight neighbor bit-vectors with cross-word spill handling for bit 0 and 63:
        // - West/East within the same row from 'c' plus spill-in from 'l'/'r'.
        // - Diagonals from 'n'/'s' plus spill-in from 'nl'/'nr'/'sl'/'sr'.
        const std::uint64_t west  = (c << 1) | (l >> 63);
        const std::uint64_t east  = (c >> 1) | (r << 63);

        const std::uint64_t north = n;
        const std::uint64_t south = s;

        const std::uint64_t nw = (n << 1) | (nl >> 63);
        const std::uint64_t ne = (n >> 1) | (nr << 63);
        const std::uint64_t sw = (s << 1) | (sl >> 63);
        const std::uint64_t se = (s >> 1) | (sr << 63);

        // Carry-save adder (CSA) tree to sum 8 neighbor bit-vectors per bit position:
        // Group the 8 inputs into triples and reduce:
        //   (nw, north, ne) -> s1, c1
        //   (west, east, south) -> s2, c2
        //   (sw, se, 0) -> s3, c3
        // Then sum partial sums:
        //   (s1, s2, s3) -> s4, c4
        // The total sum per bit is: s4 + 2*(c4 + c1 + c2 + c3).
        // To get the full 4-bit count (0..8), we form the bit-planes:
        //   b0 = s4
        //   Let t = c1 + c2 + c3 + c4 (0..4). Compute t's bits (t0, t1, t2) using CSAs/half-adders:
        //     u, v = CSA(c1, c2, c3)  -> u = parity of (c1,c2,c3), v = carries
        //     t0 = u ^ c4
        //     carry1 = u & c4
        //     t1 = v ^ carry1
        //     t2 = v & carry1
        //   Therefore b1 = t0, b2 = t1, b3 = t2.
        std::uint64_t s1, c1;
        std::uint64_t s2, c2;
        std::uint64_t s3, c3;
        csa3_u64(nw, north, ne, s1, c1);
        csa3_u64(west, east, south, s2, c2);
        csa3_u64(sw, se, 0ULL, s3, c3);

        std::uint64_t s4, c4;
        csa3_u64(s1, s2, s3, s4, c4);

        std::uint64_t u, v;
        csa3_u64(c1, c2, c3, u, v);

        const std::uint64_t t0 = u ^ c4;
        const std::uint64_t carry1 = u & c4;
        const std::uint64_t t1 = v ^ carry1;
        const std::uint64_t t2 = v & carry1;

        // Bit-planes of neighbor count for each bit position [0..63]
        const std::uint64_t b0 = s4;   // LSB of count
        const std::uint64_t b1 = t0;
        const std::uint64_t b2 = t1;
        const std::uint64_t b3 = t2;   // MSB (only set when count == 8)

        // Masks for counts equal to 2 or 3:
        //   count == 2 -> b3=0, b2=0, b1=1, b0=0
        //   count == 3 -> b3=0, b2=0, b1=1, b0=1
        const std::uint64_t eq12_common = (~b3) & (~b2) & b1;
        const std::uint64_t eq2 = eq12_common & (~b0);
        const std::uint64_t eq3 = eq12_common & b0;

        // Apply Conway's Game of Life rule:
        // - Alive survives if neighbor count == 2 or 3
        // - Dead becomes alive if neighbor count == 3
        // Encode next state in the same bit-packed format.
        const std::uint64_t next = eq3 | (c & eq2);

        out[tid] = next;
    }
}

// Host function to launch the kernel for one generation step.
// - input: pointer to device memory holding the current grid state (bit-packed).
// - output: pointer to device memory for the next grid state (bit-packed).
// - grid_dimensions: width and height of the square grid in cells (power of two, > 512).
//
// Each CUDA thread processes one 64-bit word (64 cells).
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Number of 64-bit words per row (N / 64). N is a power of two >= 512, so divisible by 64.
    const int words_per_row = grid_dimensions >> 6;

    // Total number of words in the grid.
    const std::size_t total_words = static_cast<std::size_t>(words_per_row) * static_cast<std::size_t>(grid_dimensions);

    // Compute row_shift = log2(words_per_row) since words_per_row is power-of-two.
    int row_shift = 0;
    while ((1 << row_shift) < words_per_row) ++row_shift;

    // Choose a launch configuration. 256 threads per block is a good default for Ampere/Hopper.
    const int threads_per_block = 256;
    // Aim for enough blocks to cover the device; also cap to a large multiple to leverage grid-stride loops.
    int num_blocks = static_cast<int>((total_words + threads_per_block - 1) / threads_per_block);
    // Cap the grid size to a reasonable large number if desired (optional); here we let CUDA runtime handle it.

    gol_step_kernel<<<num_blocks, threads_per_block>>>(input, output,
                                                       words_per_row,
                                                       grid_dimensions,
                                                       row_shift,
                                                       total_words);
}