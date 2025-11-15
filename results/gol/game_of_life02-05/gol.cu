// CUDA implementation of a single Conway's Game of Life step on a bit-packed grid.
//
// Each cell is encoded as a single bit in a std::uint64_t word.
// A word stores 64 consecutive cells within the same row.
// The grid is square, with dimensions being a power of two > 512, and divisible by 64.
//
// This implementation processes 64 cells per thread using bitwise operations and
// a bit-sliced adder network to compute neighbor counts without unpacking bits,
// targeting high performance on modern NVIDIA data-center GPUs (e.g., A100/H100).

#include <cuda_runtime.h>
#include <cstdint>

// Small helper structs used for bit-sliced multi-bit per-cell arithmetic.
// For each bit position i (0..63), the value represented is:
//   Bits2:  v(i) = b0[i] + 2*b1[i]
//   Bits3:  v(i) = b0[i] + 2*b1[i] + 4*b2[i]
//   Bits4:  v(i) = b0[i] + 2*b1[i] + 4*b2[i] + 8*b3[i]
struct Bits2 {
    std::uint64_t b0;
    std::uint64_t b1;
};

struct Bits3 {
    std::uint64_t b0;
    std::uint64_t b1;
    std::uint64_t b2;
};

struct Bits4 {
    std::uint64_t b0;
    std::uint64_t b1;
    std::uint64_t b2;
    std::uint64_t b3;
};

// Shift a 64-bit word one bit to the "west" (towards more-significant bit indices),
// pulling in the MSB of the left neighbor word.
// For bit i (0..63) of the result, value = cell at (x-1) in the original grid.
__device__ __forceinline__ std::uint64_t shift_left_with_neighbor(std::uint64_t x,
                                                                  std::uint64_t left_neighbor)
{
    // (left_neighbor >> 63) contributes its MSB to bit 0 of the result.
    return (x << 1) | (left_neighbor >> 63);
}

// Shift a 64-bit word one bit to the "east" (towards less-significant bit indices),
// pulling in the LSB of the right neighbor word.
// For bit i (0..63) of the result, value = cell at (x+1) in the original grid.
__device__ __forceinline__ std::uint64_t shift_right_with_neighbor(std::uint64_t x,
                                                                   std::uint64_t right_neighbor)
{
    // (right_neighbor << 63) contributes its LSB to bit 63 of the result.
    return (x >> 1) | (right_neighbor << 63);
}

// Add two 1-bit-per-cell fields (a and b). Resulting per-cell sum ranges 0..2.
// Implemented as a half-adder: sum0 = a XOR b, sum1 = a AND b.
__device__ __forceinline__ Bits2 add1_1(std::uint64_t a, std::uint64_t b)
{
    Bits2 r;
    r.b0 = a ^ b;   // LSB of sum
    r.b1 = a & b;   // carry into bit1
    return r;
}

// Add two 2-bit-per-cell fields (x and y). Resulting per-cell sum ranges 0..4.
// Full-adder style, but entirely bit-sliced so no carries propagate between cells.
__device__ __forceinline__ Bits3 add2_2(const Bits2& x, const Bits2& y)
{
    Bits3 r;

    // Add bit 0 of x and y.
    std::uint64_t s0 = x.b0 ^ y.b0;
    std::uint64_t c1 = x.b0 & y.b0; // carry into bit1

    // Add bit 1 of x and y, plus carry c1.
    std::uint64_t t1  = x.b1 ^ y.b1;
    std::uint64_t c2a = x.b1 & y.b1;
    std::uint64_t s1  = t1 ^ c1;
    std::uint64_t c2b = t1 & c1;
    std::uint64_t s2  = c2a | c2b; // bit2 (value 4) of the sum

    r.b0 = s0;
    r.b1 = s1;
    r.b2 = s2;
    return r;
}

// Add two 3-bit-per-cell fields (x and y). Resulting per-cell sum ranges 0..8.
// Again, implemented as a fully bit-sliced adder.
__device__ __forceinline__ Bits4 add3_3(const Bits3& x, const Bits3& y)
{
    Bits4 r;

    // Add bit 0 of x and y.
    std::uint64_t s0 = x.b0 ^ y.b0;
    std::uint64_t c1 = x.b0 & y.b0; // carry into bit1

    // Add bit 1 of x and y, plus carry c1.
    std::uint64_t t1  = x.b1 ^ y.b1;
    std::uint64_t c2a = x.b1 & y.b1;
    std::uint64_t s1  = t1 ^ c1;
    std::uint64_t c2b = t1 & c1;
    std::uint64_t c2  = c2a | c2b; // carry into bit2

    // Add bit 2 of x and y, plus carry c2.
    std::uint64_t t2  = x.b2 ^ y.b2;
    std::uint64_t c3a = x.b2 & y.b2;
    std::uint64_t s2  = t2 ^ c2;
    std::uint64_t c3b = t2 & c2;
    std::uint64_t s3  = c3a | c3b; // bit3 of the final sum

    r.b0 = s0;
    r.b1 = s1;
    r.b2 = s2;
    r.b3 = s3;
    return r;
}

// CUDA kernel: compute one Game of Life step.
// - input:  bit-packed current generation (device pointer)
// - output: bit-packed next generation  (device pointer)
// - grid_dim: number of rows and columns (square grid)
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    const int words_per_row = grid_dim >> 6; // grid_dim is a power of 2 and >= 512, so divisible by 64.

    // 2D thread coordinates in "word space":
    //   x = index of 64-bit word within a row (0 .. words_per_row-1)
    //   y = row index (0 .. grid_dim-1)
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= words_per_row || y >= grid_dim)
        return;

    const int idx = y * words_per_row + x;

    // Load the center word (current row, current word).
    const std::uint64_t center = input[idx];

    // Neighboring words in 3x3 word block around (y, x).
    // For interior cells (not near borders), we can load without bounds checks.
    std::uint64_t mL, mR;
    std::uint64_t tL, t, tR;
    std::uint64_t bL, b, bR;

    const bool interior =
        (x > 0) && (x + 1 < words_per_row) &&
        (y > 0) && (y + 1 < grid_dim);

    if (interior) {
        // Fast path for interior cells: directly index neighbors.
        const int top_idx    = idx - words_per_row;
        const int bottom_idx = idx + words_per_row;

        mL = input[idx - 1];
        mR = input[idx + 1];

        tL = input[top_idx - 1];
        t  = input[top_idx];
        tR = input[top_idx + 1];

        bL = input[bottom_idx - 1];
        b  = input[bottom_idx];
        bR = input[bottom_idx + 1];
    } else {
        // Boundary path: treat all out-of-range positions as dead (0).
        // This enforces the rule that cells outside the grid are always dead.

        // Same row neighbors.
        mL = (x > 0)              ? input[idx - 1] : std::uint64_t(0);
        mR = (x + 1 < words_per_row) ? input[idx + 1] : std::uint64_t(0);

        // Top row neighbors.
        if (y > 0) {
            const int top_idx = idx - words_per_row;
            t  = input[top_idx];
            tL = (x > 0)              ? input[top_idx - 1] : std::uint64_t(0);
            tR = (x + 1 < words_per_row) ? input[top_idx + 1] : std::uint64_t(0);
        } else {
            tL = t = tR = std::uint64_t(0);
        }

        // Bottom row neighbors.
        if (y + 1 < grid_dim) {
            const int bottom_idx = idx + words_per_row;
            b  = input[bottom_idx];
            bL = (x > 0)              ? input[bottom_idx - 1] : std::uint64_t(0);
            bR = (x + 1 < words_per_row) ? input[bottom_idx + 1] : std::uint64_t(0);
        } else {
            bL = b = bR = std::uint64_t(0);
        }
    }

    // Construct the 8 per-cell neighbor bitmasks using shifts and neighboring words.
    // For each bit position i in [0,63]:
    //   north[i]  = cell at (y-1, x_bit)
    //   south[i]  = cell at (y+1, x_bit)
    //   west[i]   = cell at (y,   x_bit-1)
    //   east[i]   = cell at (y,   x_bit+1)
    //   nw[i]     = cell at (y-1, x_bit-1)
    //   ne[i]     = cell at (y-1, x_bit+1)
    //   sw[i]     = cell at (y+1, x_bit-1)
    //   se[i]     = cell at (y+1, x_bit+1)
    //
    // Out-of-range neighbor positions (outside the global grid) are treated as 0
    // by virtue of zero-valued boundary words.
    const std::uint64_t north = t;
    const std::uint64_t south = b;

    const std::uint64_t west  = shift_left_with_neighbor(center, mL);
    const std::uint64_t east  = shift_right_with_neighbor(center, mR);

    const std::uint64_t nw = shift_left_with_neighbor(t, tL);
    const std::uint64_t ne = shift_right_with_neighbor(t, tR);
    const std::uint64_t sw = shift_left_with_neighbor(b, bL);
    const std::uint64_t se = shift_right_with_neighbor(b, bR);

    // Bit-sliced neighbor count over the 8 masks: north, south, east, west, ne, nw, se, sw.
    // We sum them using a small adder tree:
    //   1) Pairwise add 1-bit fields -> four Bits2
    //   2) Pairwise add Bits2 -> two Bits3
    //   3) Add the two Bits3 -> Bits4 (per-cell count in [0..8])
    const Bits2 s01 = add1_1(north, south);
    const Bits2 s23 = add1_1(east,  west);
    const Bits2 s45 = add1_1(ne,    nw);
    const Bits2 s67 = add1_1(se,    sw);

    const Bits3 s0123 = add2_2(s01, s23);
    const Bits3 s4567 = add2_2(s45, s67);

    const Bits4 sum = add3_3(s0123, s4567);

    // sum.b0..b3 now hold the per-cell neighbor count in binary:
    //   count = sum.b0 * 1 + sum.b1 * 2 + sum.b2 * 4 + sum.b3 * 8
    //
    // Game of Life rule (B3/S23):
    //   - A live cell survives if it has 2 or 3 neighbors.
    //   - A dead cell becomes alive if it has exactly 3 neighbors.
    //
    // We only need to know where count == 2 or count == 3.
    //
    // count == 2 -> binary 0010 -> (b3,b2,b1,b0) = (0,0,1,0)
    // count == 3 -> binary 0011 -> (b3,b2,b1,b0) = (0,0,1,1)
    const std::uint64_t not_b3 = ~sum.b3;
    const std::uint64_t not_b2 = ~sum.b2;
    const std::uint64_t not_b0 = ~sum.b0;

    const std::uint64_t eq2 = not_b3 & not_b2 &  sum.b1 &  not_b0;
    const std::uint64_t eq3 = not_b3 & not_b2 &  sum.b1 &  sum.b0;

    // Current cell state (1 = alive, 0 = dead).
    const std::uint64_t alive = center;

    // Next generation per cell:
    //   next = (alive && (count == 2 || count == 3)) || (!alive && count == 3)
    //        = (alive && eq2) || eq3
    const std::uint64_t next = (alive & eq2) | eq3;

    // Store result.
    output[idx] = next;
}

// Host wrapper:
// Launches the CUDA kernel to perform one Game of Life step on the device.
// - input, output: device pointers allocated with cudaMalloc, sized for grid_dimensions^2 cells,
//   bit-packed as std::uint64_t (64 cells per word).
// - grid_dimensions: width/height of the square grid (power of 2, > 512).
//
// No host-device synchronization is performed here; the caller is responsible for it.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int words_per_row = grid_dimensions >> 6;

    // Use a 2D grid of thread blocks over (word-columns, rows).
    // 32x8 threads per block generally maps well to contemporary GPUs.
    dim3 block_dim(32, 8);
    dim3 grid_dim(
        (words_per_row    + block_dim.x - 1) / block_dim.x,
        (grid_dimensions  + block_dim.y - 1) / block_dim.y
    );

    game_of_life_kernel<<<grid_dim, block_dim>>>(input, output, grid_dimensions);
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
