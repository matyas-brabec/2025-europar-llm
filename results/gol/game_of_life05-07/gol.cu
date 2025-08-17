#include <cuda_runtime.h>
#include <cstdint>

// CUDA implementation of one step of Conway's Game of Life on a bit-packed grid.
// Each thread processes one 64-bit word (64 cells) and updates all 64 cells simultaneously.
// The grid is square with dimensions grid_dimensions x grid_dimensions (power of two).
// Memory layout: row-major, bit-packed, 64 cells per 64-bit word.
// Outside-the-grid cells are treated as dead (zero).
//
// Approach:
// - For each 64-bit word (center word C), load up to eight neighboring words:
//   - Left/right neighbors in the same row (C_L, C_R)
//   - Three words in the row above (A_L, A, A_R)
//   - Three words in the row below (B_L, B, B_R)
// - Construct eight aligned neighbor bitmaps relative to the current word's bit positions:
//   For each of the three rows (above, current, below), create bitmaps for left, mid, right neighbors,
//   except exclude the current cell itself (i.e., for the current row, only left and right).
//   Cross-word neighbors are handled by OR-ing in the carry bit from adjacent words.
// - Use a carry-save adder (CSA) network to count the eight neighbors per bit position without cross-bit carries.
//   This yields three bitplanes: ones (1s place), twos (2s place), and fours (4s place).
// - Apply Life rules using boolean logic on these bitplanes:
//     birth = (count == 3) = (~fours) & twos & ones
//     survive = (count == 2) = (~fours) & twos & (~ones)
//   next = birth | (self & survive)
//
// Notes:
// - Shared/texture memory is intentionally not used; global loads are coalesced by mapping threads along rows.
// - The CSA network efficiently computes neighbor counts without per-cell loops.
// - Boundary handling uses conditional loads that zero out-of-range neighbors (outside grid are dead).
// - The kernel expects words_per_row = grid_dimensions / 64.

namespace {
using u64 = std::uint64_t;

// Half-adder for three operands using bitwise logic.
// sum = a ^ b ^ c
// carry = (a & b) | (c & (a ^ b))
// This is a building block for carry-save addition across bitmaps.
static __device__ __forceinline__ void csa3(u64 a, u64 b, u64 c, u64 &sum, u64 &carry) {
    u64 u = a ^ b;
    sum   = u ^ c;
    carry = (a & b) | (c & u);
}

// Shift helpers that align neighbor cells into the current word's bit positions.
// We assume bit 0 is the leftmost cell in the 64-bit word.
// - shift_left_with_carry: aligns the left neighbor (col-1) to the current bit position.
//   For cross-word neighbors at bit 0, we inject bit 63 from the left neighbor word.
// - shift_right_with_carry: aligns the right neighbor (col+1) to the current bit position.
//   For cross-word neighbors at bit 63, we inject bit 0 from the right neighbor word.
static __device__ __forceinline__ u64 shift_left_with_carry(u64 center, u64 left_word) {
    return (center << 1) | (left_word >> 63);
}
static __device__ __forceinline__ u64 shift_right_with_carry(u64 center, u64 right_word) {
    return (center >> 1) | (right_word << 63);
}

__global__ void gol_step_kernel(const u64* __restrict__ input,
                                u64* __restrict__ output,
                                int grid_dimensions,
                                int words_per_row)
{
    // 2D launch: x across words within a row, y across rows
    int col_word = blockIdx.x * blockDim.x + threadIdx.x;
    int row      = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_word >= words_per_row || row >= grid_dimensions) return;

    // Flags for boundary handling (outside grid is dead)
    const bool has_left  = (col_word > 0);
    const bool has_right = (col_word + 1 < words_per_row);
    const bool has_up    = (row > 0);
    const bool has_down  = (row + 1 < grid_dimensions);

    // Base indices for current, above, and below rows
    const int idx_center_row = row * words_per_row;
    const int idx_up_row     = (has_up   ? (row - 1) : row) * words_per_row;   // safe index; will mask with has_up
    const int idx_down_row   = (has_down ? (row + 1) : row) * words_per_row;   // safe index; will mask with has_down

    // Load current row words
    const u64 C   = input[idx_center_row + col_word];
    const u64 C_L = has_left  ? input[idx_center_row + col_word - 1] : 0ull;
    const u64 C_R = has_right ? input[idx_center_row + col_word + 1] : 0ull;

    // Load above row words (zero if out of bounds)
    const u64 A   = has_up ? input[idx_up_row + col_word] : 0ull;
    const u64 A_L = (has_up && has_left)  ? input[idx_up_row + col_word - 1] : 0ull;
    const u64 A_R = (has_up && has_right) ? input[idx_up_row + col_word + 1] : 0ull;

    // Load below row words (zero if out of bounds)
    const u64 B   = has_down ? input[idx_down_row + col_word] : 0ull;
    const u64 B_L = (has_down && has_left)  ? input[idx_down_row + col_word - 1] : 0ull;
    const u64 B_R = (has_down && has_right) ? input[idx_down_row + col_word + 1] : 0ull;

    // Construct eight aligned neighbor bitmaps relative to the current word:
    // Above row neighbors
    const u64 aL = shift_left_with_carry(A, A_L);   // above-left
    const u64 aM = A;                               // above
    const u64 aR = shift_right_with_carry(A, A_R);  // above-right

    // Current row neighbors (exclude the center cell itself)
    const u64 cL = shift_left_with_carry(C, C_L);   // left
    const u64 cR = shift_right_with_carry(C, C_R);  // right

    // Below row neighbors
    const u64 bL = shift_left_with_carry(B, B_L);   // below-left
    const u64 bM = B;                               // below
    const u64 bR = shift_right_with_carry(B, B_R);  // below-right

    // Carry-save adder (CSA) network to count the eight neighbors per bit without cross-bit carries.
    // First stage: group into triples (except current row which has two; treat missing third as zero).
    u64 sA, cA; csa3(aL, aM, aR, sA, cA);           // above row: 3 operands -> sA (1s), cA (2s)
    u64 sB, cB; csa3(bL, bM, bR, sB, cB);           // below row: 3 operands -> sB (1s), cB (2s)
    const u64 sC = cL ^ cR;                         // current row: 2 operands -> sC (1s)
    const u64 cC = cL & cR;                         // current row: 2 operands -> cC (2s)

    // Second stage: combine ones (weight 1) and carries (weight 2) separately.
    u64 sD, cD; csa3(sA, sB, sC, sD, cD);           // ones -> sD (1s), cD (2s)
    u64 sE, cE; csa3(cA, cB, cC, sE, cE);           // twos -> sE (2s), cE (4s)

    // Third stage: combine remaining twos (weight 2). Missing third operand is zero.
    const u64 sF = sE ^ cD;                         // twos -> sF (2s)
    const u64 cF = sE & cD;                         // carry into fours -> cF (4s)

    // Bitplanes for the neighbor count: ones (1s), twos (2s), fours (4s).
    const u64 ones  = sD;
    const u64 twos  = sF;
    // Accurate fours bitplane is XOR of the two four-weight carries (parity). We do not need the 8s bitplane.
    const u64 fours = cE ^ cF;

    // Life rules:
    // - Birth: exactly 3 neighbors -> (~fours) & twos & ones
    // - Survival: exactly 2 neighbors -> (~fours) & twos & (~ones)
    const u64 birth   = (~fours) & twos & ones;
    const u64 survive = (~fours) & twos & (~ones);

    // Next state = birth | (self & survive)
    const u64 next = birth | (C & survive);

    output[idx_center_row + col_word] = next;
}

} // namespace

// Host entry point: executes one generation step.
// input:  bit-packed input grid (grid_dimensions x grid_dimensions), row-major, 64 cells per 64-bit word.
// output: bit-packed output grid in the same layout.
// grid_dimensions: width/height of the square grid, power of two, > 512.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // Number of 64-bit words per row (grid_dimensions is multiple of 64 by problem constraints).
    const int words_per_row = grid_dimensions >> 6;

    // Launch configuration:
    // - Threads distributed along X dimension across words in a row for coalesced memory access.
    // - Y dimension processes multiple rows per block to limit grid.y and improve occupancy.
    constexpr int BLOCK_X = 256;
    constexpr int BLOCK_Y = 4;   // 256*4=1024 threads per block, good for H100/A100

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((words_per_row + BLOCK_X - 1) / BLOCK_X,
              (grid_dimensions + BLOCK_Y - 1) / BLOCK_Y,
              1);

    gol_step_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
