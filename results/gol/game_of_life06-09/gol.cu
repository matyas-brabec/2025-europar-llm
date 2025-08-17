#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA Conway's Game of Life with bit-packed rows (1 bit per cell in uint64_t).

  Key design choices for performance:
  - Each thread updates exactly one 64-bit word (64 cells) to avoid atomics.
  - Bit-sliced full-adder (carry-save) logic is used to compute neighbor counts
    across 64 cells in parallel using only bitwise operations. We process the
    eight neighbor directions by first building horizontal partial sums per row,
    then vertically combining the three rows above/current/below.
  - Left/right neighbors do not wrap across 64-bit words. Thus, bits 0 and 63
    in each word require special handling to incorporate neighbor bits from the
    left/right adjacent words (and their above/below rows). We recompute those
    two edge bits exactly and overwrite them in the result.
  - Outside the grid, cells are considered dead (zero), so out-of-bounds
    neighbor words contribute zero.

  Notation in comments:
  - up = word from the row above (same column)
  - cur = word from the current row (same column)
  - dn = word from the row below (same column)
  - Shifts (<<, >>) move neighbor bits into alignment for each position within
    the same word. These shifts intentionally ignore cross-word neighbors. We
    correct bit positions 0 and 63 separately using adjacent words.
*/

/* Add three 1-bit-per-position operands in parallel using a carry-save adder:
   ones_out holds the sum bit (1's place), twos_out holds the carry bit (2's place).
   For bits a,b,c:
     ones = a ^ b ^ c
     twos = (a & b) | (b & c) | (a & c)
*/
static __forceinline__ __device__ void add3_u64(uint64_t a, uint64_t b, uint64_t c,
                                                uint64_t& ones_out, uint64_t& twos_out) {
    uint64_t t = a ^ b;
    ones_out = t ^ c;
    twos_out = (a & b) | (t & c);
}

/* Kernel: Each thread processes a single 64-bit word. */
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int grid_dim,          // number of cells per side (square grid)
                                    int words_per_row)     // number of 64-bit words per row
{
    // Map thread to (row, col-word) with a 2D launch: y -> row, x -> word index within row.
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || col >= words_per_row) return;

    const int idx = row * words_per_row + col;

    // Load center, above, and below words. Outside the grid contributes zero.
    const bool has_up = (row > 0);
    const bool has_dn = (row + 1 < grid_dim);

    const uint64_t cur = in[idx];
    const uint64_t up  = has_up ? in[idx - words_per_row] : 0ull;
    const uint64_t dn  = has_dn ? in[idx + words_per_row] : 0ull;

    // Build horizontal sums per row using carry-save adders.
    // For up/dn rows, neighbors are left, center, right (all three are neighbors).
    // For current row, neighbors are only left and right (center cell is not a neighbor).
    // Note: Shifts do not wrap across words; bit 0 and bit 63 will be fixed later.
    const uint64_t upL = (up >> 1);
    const uint64_t upC = up;
    const uint64_t upR = (up << 1);

    uint64_t up_ones, up_twos;
    add3_u64(upL, upC, upR, up_ones, up_twos);

    const uint64_t dnL = (dn >> 1);
    const uint64_t dnC = dn;
    const uint64_t dnR = (dn << 1);

    uint64_t dn_ones, dn_twos;
    add3_u64(dnL, dnC, dnR, dn_ones, dn_twos);

    const uint64_t curL = (cur >> 1);
    const uint64_t curR = (cur << 1);

    uint64_t cur_ones, cur_twos;
    add3_u64(curL, curR, 0ull, cur_ones, cur_twos); // sum of two neighbors (left/right)

    // Vertically combine the three rows' "ones" planes and "twos" planes.
    // Let:
    //   O  = parity of (up_ones + cur_ones + dn_ones)               -> ones1
    //   T2 = carry of  (up_ones + cur_ones + dn_ones) (weight 2)    -> twos1
    //   S  = parity of (up_twos + cur_twos + dn_twos)               -> ones2
    //   C  = carry of  (up_twos + cur_twos + dn_twos) (weight 4)    -> twos2
    //
    // Then the neighbor count per bit is:
    //   N = O + 2*(T2 + S) + 4*C
    //
    // From this, we derive masks for N==2 and N==3:
    //   N == 2  <=>  C==0, (T2+S)==1, O==0  => mask2 = ~C & ~(O) & (T2 xor S)
    //   N == 3  <=>  C==0, (T2+S)==1, O==1  => mask3 = ~C &    O  & (T2 xor S)
    uint64_t ones1, twos1;
    add3_u64(up_ones, cur_ones, dn_ones, ones1, twos1);

    uint64_t ones2, twos2;
    add3_u64(up_twos, cur_twos, dn_twos, ones2, twos2);

    const uint64_t X = twos1 ^ ones2;     // (T2 xor S)
    const uint64_t C = twos2;             // C (weight 4 carry)
    const uint64_t O = ones1;             // O

    const uint64_t mask_not_C = ~C;
    const uint64_t eq2_mask = mask_not_C & (~O) & X; // neighbor count == 2
    const uint64_t eq3_mask = mask_not_C &   O  & X; // neighbor count == 3

    // Base next-state for all 64 bits (bit 0 and 63 will be corrected below).
    // Alive in next gen if (count==3) OR (alive && count==2).
    uint64_t next = eq3_mask | (cur & eq2_mask);

    // Special handling for bit 0 and bit 63 to incorporate neighbors crossing 64-bit word boundaries.
    // For bit 0, also consider left-adjacent words (col-1) from up/cur/dn rows.
    // For bit 63, also consider right-adjacent words (col+1) from up/cur/dn rows.
    const bool has_left  = (col > 0);
    const bool has_right = (col + 1 < words_per_row);

    const uint64_t left_up   = (has_up   && has_left)  ? in[idx - words_per_row - 1] : 0ull;
    const uint64_t left_cur  = (has_left)              ? in[idx - 1]                 : 0ull;
    const uint64_t left_dn   = (has_dn   && has_left)  ? in[idx + words_per_row - 1] : 0ull;

    const uint64_t right_up  = (has_up   && has_right) ? in[idx - words_per_row + 1] : 0ull;
    const uint64_t right_cur = (has_right)             ? in[idx + 1]                 : 0ull;
    const uint64_t right_dn  = (has_dn   && has_right) ? in[idx + words_per_row + 1] : 0ull;

    // Compute neighbor count for bit 0 explicitly (8 neighbors):
    //   up-left:    bit63 of left_up
    //   up:         bit0  of up
    //   up-right:   bit1  of up
    //   left:       bit63 of left_cur
    //   right:      bit1  of cur
    //   down-left:  bit63 of left_dn
    //   down:       bit0  of dn
    //   down-right: bit1  of dn
    unsigned cnt0 = 0u;
    if (has_up) {
        cnt0 += static_cast<unsigned>(up & 1ull);
        cnt0 += static_cast<unsigned>((up >> 1) & 1ull);
        if (has_left) cnt0 += static_cast<unsigned>((left_up >> 63) & 1ull);
    }
    if (has_left) cnt0 += static_cast<unsigned>((left_cur >> 63) & 1ull);
    cnt0 += static_cast<unsigned>((cur >> 1) & 1ull);
    if (has_dn) {
        cnt0 += static_cast<unsigned>(dn & 1ull);
        cnt0 += static_cast<unsigned>((dn >> 1) & 1ull);
        if (has_left) cnt0 += static_cast<unsigned>((left_dn >> 63) & 1ull);
    }
    const unsigned alive0 = static_cast<unsigned>(cur & 1ull);
    const unsigned next0 = (cnt0 == 3u) | (alive0 & (cnt0 == 2u));

    // Compute neighbor count for bit 63 explicitly (8 neighbors):
    //   up-left:    bit62 of up
    //   up:         bit63 of up
    //   up-right:   bit0  of right_up
    //   left:       bit62 of cur
    //   right:      bit0  of right_cur
    //   down-left:  bit62 of dn
    //   down:       bit63 of dn
    //   down-right: bit0  of right_dn
    unsigned cnt63 = 0u;
    if (has_up) {
        cnt63 += static_cast<unsigned>((up >> 62) & 1ull);
        cnt63 += static_cast<unsigned>((up >> 63) & 1ull);
        if (has_right) cnt63 += static_cast<unsigned>(right_up & 1ull);
    }
    cnt63 += static_cast<unsigned>((cur >> 62) & 1ull);
    if (has_right) cnt63 += static_cast<unsigned>(right_cur & 1ull);
    if (has_dn) {
        cnt63 += static_cast<unsigned>((dn >> 62) & 1ull);
        cnt63 += static_cast<unsigned>((dn >> 63) & 1ull);
        if (has_right) cnt63 += static_cast<unsigned>(right_dn & 1ull);
    }
    const unsigned alive63 = static_cast<unsigned>((cur >> 63) & 1ull);
    const unsigned next63 = (cnt63 == 3u) | (alive63 & (cnt63 == 2u));

    // Overwrite bit 0 and bit 63 in the result with the corrected values.
    next = (next & ~1ull) | static_cast<uint64_t>(next0);
    next = (next & ~(1ull << 63)) | (static_cast<uint64_t>(next63) << 63);

    out[idx] = next;
}

/*
  Host entry point: executes one Game of Life step on a bit-packed square grid.

  Arguments:
  - input:  device pointer to input grid (bit-packed, uint64_t per 64 cells)
  - output: device pointer to output grid (same format)
  - grid_dimensions: width and height in cells (power of two; > 512)

  Notes:
  - input and output are assumed to be allocated via cudaMalloc by the caller.
  - This function performs no explicit synchronization; the caller handles it.
*/
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6; // grid_dimensions is power of 2, so exact divide by 64

    // Launch configuration:
    // - Use a 2D grid so each block row maps to a specific input row, ensuring
    //   threads in a block access contiguous memory for coalesced loads.
    // - blockDim.x tuned for throughput; no shared memory used.
    const int threads_per_block = 256;
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((words_per_row + threads_per_block - 1) / threads_per_block,
              grid_dimensions, 1);

    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
