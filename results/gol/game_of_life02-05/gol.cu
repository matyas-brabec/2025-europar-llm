#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one timestep of Conway's Game of Life on a square grid.
// The grid is bit-packed: each 64-bit word encodes 64 horizontal cells in a row.
// Bit i (0=LSB .. 63=MSB) corresponds to column (64*word_index + i).
// Cells outside the grid are treated as dead (0).
//
// This implementation computes neighbor counts using a bit-sliced (carry-save) adder
// network to efficiently derive masks for "exactly 2 neighbors" and "exactly 3 neighbors"
// per bit-lane without per-bit branching or shared memory. We only use global memory.
//
// For each 64-bit word in row r at column w, we load up to 9 words:
//   (r-1,w-1) (r-1,w) (r-1,w+1)
//   (r  ,w-1) (r  ,w) (r  ,w+1)
//   (r+1,w-1) (r+1,w) (r+1,w+1)
// This enables horizontal shifts with cross-word carry for the three involved rows.
// For boundary rows/columns, out-of-range loads are replaced by zeros.
//
// We compute the eight 1-bit masks of neighbor directions aligned to the center cell bit:
//   - Above-left  (A << 1 with carry from A_left MSB)
//   - Above       (A)
//   - Above-right (A >> 1 with carry from A_right LSB)
//   - Below-left  (B << 1 with carry from B_left MSB)
//   - Below       (B)
//   - Below-right (B >> 1 with carry from B_right LSB)
//   - Left        (C << 1 with carry from C_left MSB)   [current row]
//   - Right       (C >> 1 with carry from C_right LSB)  [current row]
// Note: The center (C) is not included in the neighbor sum.
//
// The carry-save adder (CSA) network reduces these 8 inputs to bit-planes of the
// neighbor count S per lane: we compute the least significant three bits (S0, S1, S2).
// This is sufficient to test S==2 or S==3 (S2 must be 0, and S1=1 with S0=0 or 1).
//
// Transition rule:
//   next = (S == 3) | (current & (S == 2))
//
// Thread mapping:
//   We launch a 2D grid: x-dimension covers 64-bit words per row, y-dimension covers rows.
//   Each thread computes one output 64-bit word.
//   No shared memory or texture memory is used; global loads are coalesced.

static __device__ __forceinline__ std::uint64_t shl1_with_carry(std::uint64_t x, std::uint64_t left_word) {
    // Shift left by 1 within the 64-bit word; bring in the MSB of the left neighbor word as new bit0.
    return (x << 1) | (left_word >> 63);
}

static __device__ __forceinline__ std::uint64_t shr1_with_carry(std::uint64_t x, std::uint64_t right_word) {
    // Shift right by 1 within the 64-bit word; bring in the LSB of the right neighbor word as new bit63.
    // Using (right_word << 63) implicitly grabs the bit0 and places it in bit63.
    return (x >> 1) | (right_word << 63);
}

static __device__ __forceinline__ void csa3(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                            std::uint64_t& sum, std::uint64_t& carry) {
    // Carry-Save Adder for three 1-bit operands per lane:
    // sum   = a ^ b ^ c                (LSB of the per-lane sum)
    // carry = majority(a,b,c)          (carry-out bit per lane, represents +2 at that lane)
    const std::uint64_t ab_xor = a ^ b;
    sum   = ab_xor ^ c;
    // carry = (a&b) | (a&c) | (b&c)
    const std::uint64_t ab_and = a & b;
    const std::uint64_t ac_and = a & c;
    const std::uint64_t bc_and = b & c;
    carry = ab_and | ac_and | bc_and;
}

__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ in,
                                    std::uint64_t* __restrict__ out,
                                    int N, int words_per_row) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;  // word index within the row
    const int r = blockIdx.y;                              // row index
    if (r >= N || w >= words_per_row) return;

    const int W = words_per_row;

    // Base indices for rows r-1, r, r+1
    const int idx_up    = (r > 0)        ? (r - 1) * W : 0;
    const int idx_mid   = r * W;
    const int idx_down  = (r + 1 < N)    ? (r + 1) * W : 0;

    // Load center words for the three rows.
    const std::uint64_t A = (r > 0)        ? in[idx_up  + w] : 0ull; // row above
    const std::uint64_t C =                  in[idx_mid + w];        // current row
    const std::uint64_t B = (r + 1 < N)    ? in[idx_down+ w] : 0ull; // row below

    // Load neighbor words used for cross-word carries in horizontal shifts.
    const bool has_left  = (w > 0);
    const bool has_right = (w + 1 < W);

    const std::uint64_t A_left  = (r > 0        && has_left ) ? in[idx_up  + (w - 1)] : 0ull;
    const std::uint64_t A_right = (r > 0        && has_right) ? in[idx_up  + (w + 1)] : 0ull;

    const std::uint64_t C_left  = (                 has_left ) ? in[idx_mid + (w - 1)] : 0ull;
    const std::uint64_t C_right = (                 has_right) ? in[idx_mid + (w + 1)] : 0ull;

    const std::uint64_t B_left  = (r + 1 < N && has_left ) ? in[idx_down + (w - 1)] : 0ull;
    const std::uint64_t B_right = (r + 1 < N && has_right) ? in[idx_down + (w + 1)] : 0ull;

    // Build neighbor direction masks aligned to the center bit of this word.
    // Above row contributions:
    const std::uint64_t n_aw = shl1_with_carry(A, A_left);
    const std::uint64_t n_ac = A;
    const std::uint64_t n_ae = shr1_with_carry(A, A_right);

    // Below row contributions:
    const std::uint64_t n_bw = shl1_with_carry(B, B_left);
    const std::uint64_t n_bc = B;
    const std::uint64_t n_be = shr1_with_carry(B, B_right);

    // Current row contributions (exclude center C itself):
    const std::uint64_t n_cw = shl1_with_carry(C, C_left);
    const std::uint64_t n_ce = shr1_with_carry(C, C_right);

    // Reduce the 8 neighbor masks to per-lane sum bits using a CSA (carry-save) network.
    // First layer: group inputs into three CSA3s.
    std::uint64_t s0_a, c0_a;
    std::uint64_t s0_b, c0_b;
    std::uint64_t s0_c, c0_c;

    csa3(n_aw, n_ac, n_ae, s0_a, c0_a);
    csa3(n_bw, n_bc, n_be, s0_b, c0_b);
    csa3(n_cw, n_ce, 0ull, s0_c, c0_c);

    // Second layer: sum the LSB sums to get final bit0 (S0) and a carry into weight-2 (c_s).
    std::uint64_t S0, c_s;
    csa3(s0_a, s0_b, s0_c, S0, c_s);  // S0 is the least significant bit of the neighbor count.

    // Sum the weight-2 carries from the first layer and the carry from LSB-sum:
    // t_sum: partial bit1 (weight-2) sum; t_carry: carry to weight-4.
    std::uint64_t t_sum, t_carry;
    csa3(c0_a, c0_b, c0_c, t_sum, t_carry);

    // Finalize bit1 (S1) and bit2 (S2)
    const std::uint64_t S1 = t_sum ^ c_s;
    const std::uint64_t w4_from_two = t_sum & c_s;   // extra carry into weight-4 from (t_sum + c_s)
    const std::uint64_t S2 = t_carry ^ w4_from_two;  // S2 is the 4's place bit; S3 (8's place) not needed for eq2/eq3

    // Build masks for exactly 2 or exactly 3 neighbors:
    // S == 2  => S2=0, S1=1, S0=0
    // S == 3  => S2=0, S1=1, S0=1
    const std::uint64_t not_S2 = ~S2;
    const std::uint64_t eq2 = not_S2 & S1 & (~S0);
    const std::uint64_t eq3 = not_S2 & S1 &   S0;

    // Apply Life rule: alive at next if (neighbors==3) or (alive && neighbors==2).
    const std::uint64_t next = eq3 | (C & eq2);

    out[idx_mid + w] = next;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions (N) is a power of two, >= 512, and divisible by 64.
    const int N = grid_dimensions;
    const int words_per_row = N / 64;

    // Configure a 2D launch: x spans words per row, y spans rows.
    // A 256-thread block tends to be a good balance on A100/H100 for memory-bound kernels.
    const int block_x = 256;
    dim3 block(block_x, 1, 1);
    dim3 grid((words_per_row + block.x - 1) / block.x, N, 1);

    game_of_life_kernel<<<grid, block>>>(input, output, N, words_per_row);
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
