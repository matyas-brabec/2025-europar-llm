#include <cuda_runtime.h>
#include <cstdint>

/*
High-level idea:
- Each thread processes one 64-bit word (bit-packed 64 cells in a row).
- We compute the 8-neighbor sum for all 64 cells at once using bit-parallel operations.
- Horizontal neighbors across 64-bit word boundaries are handled by "shift with carry" using adjacent words.
- Neighbor sums are computed with a carry-save adder (CSA) tree to produce bitplanes for 1s, 2s, 4s, and 8s.
- New state bit for each cell is:
    alive_next = (neighbor_count == 3) | (alive_current & (neighbor_count == 2))
  which is derived directly from the bitplanes without per-cell loops.

Notes:
- All cells outside the grid are considered dead (zero), enforced in boundary handling.
- The grid is square with side length being a power of 2; width >= 512 and fits memory.
- No shared/texture memory is used; global memory accesses are coalesced for the primary word, with predictable strided accesses for neighboring rows.
*/

static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t center, std::uint64_t left) {
    // Logical shift-left by 1 with bit 0 filled from bit 63 of the word to the left.
    return (center << 1) | (left >> 63);
}

static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t center, std::uint64_t right) {
    // Logical shift-right by 1 with bit 63 filled from bit 0 of the word to the right.
    return (center >> 1) | (right << 63);
}

static __device__ __forceinline__ void CSA(std::uint64_t &carry, std::uint64_t &sum, std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    // Carry-save adder for three 1-bit operands per bit position.
    // Produces:
    //   sum   = a ^ b ^ c         (LSB of the per-bit sum)
    //   carry = (a&b) | ((a^b)&c) (carry to next bitplane)
    std::uint64_t u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (u & c);
}

__global__ void game_of_life_step_kernel(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words = words_per_row * grid_dimensions;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_words) return;

    // Compute 2D coordinates in the word grid.
    int row = tid / words_per_row;
    int col = tid - row * words_per_row; // faster than modulus

    // Boundary flags.
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dimensions);
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    // Load the 3x3 neighborhood of 64-bit words surrounding the current word.
    // Outside-grid words are treated as zero.
    // Current row
    std::uint64_t midL = has_left  ? input[(std::size_t)row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t midC =            input[(std::size_t)row * words_per_row +  col      ];
    std::uint64_t midR = has_right ? input[(std::size_t)row * words_per_row + (col + 1)] : 0ULL;

    // Row above
    std::uint64_t upL = 0ULL, upC = 0ULL, upR = 0ULL;
    if (has_up) {
        std::size_t upRowBase = (std::size_t)(row - 1) * words_per_row;
        upL = has_left  ? input[upRowBase + (col - 1)] : 0ULL;
        upC =            input[upRowBase +  col      ];
        upR = has_right ? input[upRowBase + (col + 1)] : 0ULL;
    }

    // Row below
    std::uint64_t dnL = 0ULL, dnC = 0ULL, dnR = 0ULL;
    if (has_down) {
        std::size_t dnRowBase = (std::size_t)(row + 1) * words_per_row;
        dnL = has_left  ? input[dnRowBase + (col - 1)] : 0ULL;
        dnC =            input[dnRowBase +  col      ];
        dnR = has_right ? input[dnRowBase + (col + 1)] : 0ULL;
    }

    // Build the 8 neighbor bitboards aligned to the current word's bit positions.
    // For each row, left neighbor is shl1_with_carry, right neighbor is shr1_with_carry, center is the row word itself.
    const std::uint64_t UL = shl1_with_carry(upC,  upL);
    const std::uint64_t U  = upC;
    const std::uint64_t UR = shr1_with_carry(upC,  upR);

    const std::uint64_t L  = shl1_with_carry(midC, midL);
    const std::uint64_t R  = shr1_with_carry(midC, midR);

    const std::uint64_t DL = shl1_with_carry(dnC,  dnL);
    const std::uint64_t D  = dnC;
    const std::uint64_t DR = shr1_with_carry(dnC,  dnR);

    // Carry-save adder tree to compute neighbor count bitplanes:
    // ones (1s place), twos (2s place), fours (4s place), eights (8s place)
    std::uint64_t h0, l0, h1, l1, h2, l2;
    CSA(h0, l0, UL, U,  UR);
    CSA(h1, l1, DL, D,  DR);
    CSA(h2, l2, L,  R,  0ULL);

    std::uint64_t h3, ones;
    CSA(h3, ones, l0, l1, l2);

    std::uint64_t h4, l4;
    CSA(h4, l4, h0, h1, h2);     // l4: partial 'twos', h4: carries to 'fours'

    std::uint64_t h5, twos;
    CSA(h5, twos, l4, h3, 0ULL); // twos: final 'twos', h5: carries to 'fours'

    const std::uint64_t fours  = h4 ^ h5; // sum of two carry sources to 4s plane
    const std::uint64_t eights = h4 & h5; // carry to 8s plane (only nonzero when neighbor count is 8)

    // Apply Game of Life rules:
    // - eq3: neighbor_count == 3 -> ones & twos & ~fours & ~eights
    // - eq2: neighbor_count == 2 -> ~ones & twos & ~fours & ~eights
    const std::uint64_t not_fours  = ~fours;
    const std::uint64_t not_eights = ~eights;

    const std::uint64_t eq3  = (ones & twos) & not_fours & not_eights;
    const std::uint64_t eq2  = (~ones & twos) & not_fours & not_eights;

    const std::uint64_t next = eq3 | (midC & eq2);

    // Store the next generation word.
    output[(std::size_t)row * words_per_row + col] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each thread processes one 64-bit word (64 cells). We launch enough threads
    // to cover all words in the grid. All synchronization is handled by the caller.
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64
    const int total_words = words_per_row * grid_dimensions;

    // Choose a block size that maps well to modern NVIDIA GPUs (A100/H100).
    // 256 threads per block tends to provide a good balance of occupancy and scheduling.
    const int block_size = 256;
    const int grid_size = (total_words + block_size - 1) / block_size;

    game_of_life_step_kernel<<<grid_size, block_size>>>(input, output, grid_dimensions);
}