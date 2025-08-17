#include <cuda_runtime.h>
#include <cstdint>

// High-performance, branch-minimal CUDA implementation of a single step of Conway's Game of Life
// operating on a bit-packed grid (64 cells per 64-bit word).
//
// Key implementation notes:
// - The grid is square with side length N (power of two, >512). Each row has W = N/64 64-bit words.
// - Each CUDA thread processes one 64-bit word (i.e., 64 cells) using bit-sliced logic.
// - Neighbor counts are computed with carry-less (bitwise) adders to avoid cross-bit carries.
// - Boundary conditions (outside the grid) are treated as dead cells (zeros), handled via conditional loads.
// - No shared/texture memory is used; coalesced global memory loads and L1/L2 caching provide good performance.
// - We compute three bit-planes of the neighbor sum: ones, twos, fours. This is sufficient to test "sum == 2" and "sum == 3".
// - The next state per bit is:
//     next = (sum == 3) | (alive & (sum == 2))
//          = (ones & twos & ~fours) | (alive & ~ones & twos & ~fours)

// Funnel-like shifts across 64-bit word boundaries to align horizontal neighbors into the current word's bit positions.
// For a given word "center", inject MSB/LSB from the adjacent words "left" or "right" to model cross-word neighbors.
__device__ __forceinline__ std::uint64_t shift_left_inject(std::uint64_t center, std::uint64_t left) {
    // Shift left by 1: bit i receives original bit (i-1). Inject left word's MSB into bit 0.
    return (center << 1) | (left >> 63);
}
__device__ __forceinline__ std::uint64_t shift_right_inject(std::uint64_t center, std::uint64_t right) {
    // Shift right by 1: bit i receives original bit (i+1). Inject right word's LSB into bit 63.
    return (center >> 1) | (right << 63);
}

// Add three 64-bit bit-vectors (carry-less per bit-lane).
// Returns pair (sum_mod2, carry), representing per-bit sums: a + b + c = sum_mod2 + 2*carry
struct Sum2 {
    std::uint64_t s; // bitwise sum modulo 2
    std::uint64_t c; // bitwise carry (1 if at least two inputs were 1)
};
__device__ __forceinline__ Sum2 add3_bitwise(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    std::uint64_t ab_x = a ^ b;
    std::uint64_t ab_c = a & b;
    Sum2 out;
    out.s = ab_x ^ c;
    out.c = (ab_x & c) | ab_c;
    return out;
}

// CUDA kernel: compute one Game of Life step for a bit-packed grid.
__global__ void gol_step_kernel(const std::uint64_t* __restrict__ in,
                                std::uint64_t* __restrict__ out,
                                int N, int W, int W_shift, std::uint64_t total_words)
{
    // Grid-stride loop so we can choose an arbitrary grid size independent of problem size.
    std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint64_t stride = blockDim.x * (std::uint64_t)gridDim.x;

    while (tid < total_words) {
        // Compute row (y) and word-in-row (xw) without division/modulo since W is a power of two.
        std::uint64_t y  = tid >> W_shift;
        std::uint64_t xw = tid & (std::uint64_t)(W - 1);

        const bool has_up    = (y > 0);
        const bool has_down  = (y + 1u < (std::uint64_t)N);
        const bool has_left  = (xw > 0);
        const bool has_right = (xw + 1u < (std::uint64_t)W);

        const std::uint64_t base = y << W_shift;

        // Center row words
        const std::uint64_t mC = in[base + xw];
        const std::uint64_t mL = has_left  ? in[base + xw - 1u] : 0ull;
        const std::uint64_t mR = has_right ? in[base + xw + 1u] : 0ull;

        // Upper row words (or zeros if beyond boundary)
        const std::uint64_t ubase = has_up ? (base - (std::uint64_t)W) : 0ull;
        const std::uint64_t uC = has_up ? in[ubase + xw] : 0ull;
        const std::uint64_t uL = (has_up && has_left)  ? in[ubase + xw - 1u] : 0ull;
        const std::uint64_t uR = (has_up && has_right) ? in[ubase + xw + 1u] : 0ull;

        // Lower row words (or zeros if beyond boundary)
        const std::uint64_t dbase = has_down ? (base + (std::uint64_t)W) : 0ull;
        const std::uint64_t dC = has_down ? in[dbase + xw] : 0ull;
        const std::uint64_t dL = (has_down && has_left)  ? in[dbase + xw - 1u] : 0ull;
        const std::uint64_t dR = (has_down && has_right) ? in[dbase + xw + 1u] : 0ull;

        // Align the eight neighbor contributions into the current word's bit positions.
        const std::uint64_t u_left  = shift_left_inject(uC, uL);   // up-left
        const std::uint64_t u_right = shift_right_inject(uC, uR);  // up-right
        const std::uint64_t m_left  = shift_left_inject(mC, mL);   // left
        const std::uint64_t m_right = shift_right_inject(mC, mR);  // right
        const std::uint64_t d_left  = shift_left_inject(dC, dL);   // down-left
        const std::uint64_t d_right = shift_right_inject(dC, dR);  // down-right
        const std::uint64_t u_mid   = uC;                          // up
        const std::uint64_t d_mid   = dC;                          // down

        // Sum the three columns (left, center, right) using carry-less adders:
        // Column-wise triplets: (u_left, u_mid, u_right), (m_left, -, m_right), (d_left, d_mid, d_right)
        // For the middle row we only have two neighbors (left, right): emulate add3 with c=0.
        const Sum2 top = add3_bitwise(u_left, u_mid, u_right);   // top.s (ones), top.c (twos)
        const Sum2 mid = { m_left ^ m_right, m_left & m_right }; // mid.s (ones), mid.c (twos)
        const Sum2 bot = add3_bitwise(d_left, d_mid, d_right);   // bot.s (ones), bot.c (twos)

        // Combine the three column sums into overall ones/twos/fours bit-planes.
        // First, add the ones planes of top/mid/bot (this generates a carry into the twos plane).
        const std::uint64_t ones_sum  = top.s ^ mid.s ^ bot.s;
        const std::uint64_t ones_carry = (top.s & mid.s) | ((top.s ^ mid.s) & bot.s);

        // Then, add the twos planes of top/mid/bot.
        const std::uint64_t twos_partial  = top.c ^ mid.c ^ bot.c;
        const std::uint64_t twos_carry_pm = (top.c & mid.c) | ((top.c ^ mid.c) & bot.c);

        // Finally, incorporate the carry from the ones addition into the twos plane,
        // producing the final twos and fours bit-planes.
        const std::uint64_t twos  = twos_partial ^ ones_carry;
        const std::uint64_t fours = twos_carry_pm | (twos_partial & ones_carry);

        // Apply Game of Life rules:
        // - Birth: cell becomes alive if neighbor sum == 3  => ones && twos && !fours
        // - Survival: alive cell stays alive if sum == 2    => !ones && twos && !fours
        const std::uint64_t not_fours = ~fours;
        const std::uint64_t births    = ones_sum & twos & not_fours;
        const std::uint64_t survives  = (~ones_sum) & twos & not_fours & mC;

        out[base + xw] = births | survives;

        tid += stride;
    }
}

// Host function to run one step of the Game of Life.
// input:  bit-packed grid (64 cells per 64-bit word), row-major order
// output: next state in the same bit-packed format
// grid_dimensions: number of cells per row/column (N), power-of-two > 512
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int N = grid_dimensions;
    const int W = N >> 6; // words per row (N / 64). N is a power of two, so this is exact.

    // Compute log2(W) for fast row/column computation in the kernel.
    int W_shift = 0;
    // W is guaranteed to be a power of two; compute shift via loop (executed once on host).
    while ((1 << W_shift) < W) ++W_shift;

    const std::uint64_t total_words = (std::uint64_t)N * (std::uint64_t)W;

    // Configure kernel launch. Use grid-stride loop inside the kernel for flexibility.
    constexpr int TPB = 256;
    int blocks = (int)((total_words + TPB - 1) / TPB);
    // Cap the number of blocks to a large but reasonable number if desired, but typically not necessary.
    // The caller is responsible for synchronization; we just launch the kernel.
    gol_step_kernel<<<blocks, TPB>>>(input, output, N, W, W_shift, total_words);
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
