#include <cuda_runtime.h>
#include <cstdint>

// Conway's Game of Life - bit-packed CUDA implementation for 64 cells per 64-bit word.
// Each thread processes one 64-bit word (64 cells) for a given row and word index.
// The neighbors are computed purely with bitwise operations (no shared/texture memory).
// Borders are treated as dead: cells outside the grid contribute zero.

// 3:2 compressor (carry-save adder) to add three 1-bit-per-cell bitboards without cross-bit carries.
// For each bit position i, it computes:
//   sum[i]   = a[i] XOR b[i] XOR c[i]          (LSB of the local sum)
//   carry[i] = majority(a[i], b[i], c[i])      (carry into the next significance bit for that position)
// The total per-position sum is sum + 2*carry.
static __device__ __forceinline__ void csa_u64(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                                               std::uint64_t &sum, std::uint64_t &carry) {
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// Kernel: compute one generation update.
// in/out: bit-packed grids; N = grid width/height; wordsPerRow = N / 64.
__global__ void life_kernel_bitpacked(const std::uint64_t* __restrict__ in,
                                      std::uint64_t* __restrict__ out,
                                      int N, int wordsPerRow) {
    const int wx = blockIdx.x * blockDim.x + threadIdx.x; // word index in row
    const int ry = blockIdx.y * blockDim.y + threadIdx.y; // row index
    if (wx >= wordsPerRow || ry >= N) return;

    const bool hasLeftWord  = (wx > 0);
    const bool hasRightWord = (wx + 1 < wordsPerRow);
    const bool hasUpRow     = (ry > 0);
    const bool hasDownRow   = (ry + 1 < N);

    const std::size_t rowOffset    = static_cast<std::size_t>(ry) * static_cast<std::size_t>(wordsPerRow);
    const std::size_t idx          = rowOffset + static_cast<std::size_t>(wx);
    const std::size_t rowUpOffset  = hasUpRow   ? (static_cast<std::size_t>(ry - 1) * static_cast<std::size_t>(wordsPerRow))  : 0;
    const std::size_t rowDnOffset  = hasDownRow ? (static_cast<std::size_t>(ry + 1) * static_cast<std::size_t>(wordsPerRow))  : 0;

    // Load current word and its horizontal neighbors within the same row (handling boundaries as zero).
    const std::uint64_t cur = in[idx];
    const std::uint64_t curL = hasLeftWord  ? in[idx - 1] : 0ull;
    const std::uint64_t curR = hasRightWord ? in[idx + 1] : 0ull;

    // Load words from the row above.
    const std::uint64_t upC = hasUpRow ? in[rowUpOffset + wx] : 0ull;
    const std::uint64_t upL = (hasUpRow && hasLeftWord ) ? in[rowUpOffset + wx - 1] : 0ull;
    const std::uint64_t upR = (hasUpRow && hasRightWord) ? in[rowUpOffset + wx + 1] : 0ull;

    // Load words from the row below.
    const std::uint64_t dnC = hasDownRow ? in[rowDnOffset + wx] : 0ull;
    const std::uint64_t dnL = (hasDownRow && hasLeftWord ) ? in[rowDnOffset + wx - 1] : 0ull;
    const std::uint64_t dnR = (hasDownRow && hasRightWord) ? in[rowDnOffset + wx + 1] : 0ull;

    // Compute neighbor bitboards aligned to the current word's bit positions.
    // For each bit position p (0..63), the following bitboards have bit p set if the corresponding neighbor is alive:
    //   W: west neighbor (same row, col-1)
    //   E: east neighbor (same row, col+1)
    //   N: north neighbor (row-1, same col)
    //   S: south neighbor (row+1, same col)
    //   NW, NE, SW, SE: diagonal neighbors
    //
    // Shifts across 64-bit word boundaries are handled via adjacent words (curL, curR, upL/upR, dnL/dnR).
    const std::uint64_t W  = (cur << 1) | (curL >> 63);
    const std::uint64_t E  = (cur >> 1) | (curR << 63);
    const std::uint64_t N  = upC;
    const std::uint64_t S  = dnC;
    const std::uint64_t NW = (upC << 1) | (upL >> 63);
    const std::uint64_t NE = (upC >> 1) | (upR << 63);
    const std::uint64_t SW = (dnC << 1) | (dnL >> 63);
    const std::uint64_t SE = (dnC >> 1) | (dnR << 63);

    // Sum the 8 one-bit-per-cell neighbor bitboards using a carry-save adder tree to produce 3 bitplanes:
    //   b0: 1's place (LSB), b1: 2's place, b2: 4's place of the neighbor count (range 0..8).
    // We group inputs into triples to avoid cross-bit carries. The derivation ensures correctness for all bits.
    std::uint64_t s1, c1; csa_u64(W,  E,  N,  s1, c1); // s1 + 2*c1 = W + E + N
    std::uint64_t s2, c2; csa_u64(S,  NW, NE, s2, c2); // s2 + 2*c2 = S + NW + NE
    std::uint64_t s3, c3; csa_u64(SW, SE,  0, s3, c3); // s3 + 2*c3 = SW + SE

    // Combine the three partial sums for the LSB plane.
    std::uint64_t b0, cS; csa_u64(s1, s2, s3, b0, cS); // b0 + 2*cS = s1 + s2 + s3

    // Now we must add the four carry planes (cS, c1, c2, c3) to obtain the higher bitplanes b1 and b2.
    // Let T = cS + c1 + c2 + c3 (0..4). We compute T in binary:
    //   t0 = LSB of T, t1 = next bit of T, t2 = MSB of T (only set when T == 4).
    const std::uint64_t u  = cS ^ c1;
    const std::uint64_t v  = cS & c1;
    const std::uint64_t u2 = c2 ^ c3;
    const std::uint64_t v2 = c2 & c3;

    const std::uint64_t t0 = u ^ u2;                       // LSB of T
    const std::uint64_t t1 = (u & u2) ^ v ^ v2;            // next bit of T
    // const std::uint64_t t2 = v & v2;                    // MSB of T (only nonzero when T==4, i.e., count >= 8)

    const std::uint64_t b1 = t0;                           // 2's place
    const std::uint64_t b2 = t1;                           // 4's place

    // Apply Game of Life rules using the neighbor count bitplanes and current state.
    // births: cells with exactly 3 neighbors -> (~b2) & b1 & b0
    // survive: alive cells with exactly 2 neighbors -> alive & (~b2) & b1 & (~b0)
    const std::uint64_t alive   = cur;
    const std::uint64_t eq3     = (~b2) & b1 & b0;
    const std::uint64_t eq2     = (~b2) & b1 & (~b0);
    const std::uint64_t nextGen = eq3 | (alive & eq2);

    out[idx] = nextGen;
}

// Host function to launch the kernel for one generation step.
// input:  bit-packed input grid, one std::uint64_t per 64 cells along a row (cudaMalloc'ed).
// output: bit-packed output grid, same layout (cudaMalloc'ed).
// grid_dimensions: N (width == height), must be power of 2 > 512. All off-grid cells are treated as dead.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    const int N = grid_dimensions;
    const int wordsPerRow = N >> 6; // N / 64; N guaranteed multiple of 64 for powers of two >= 512

    // Use a 2D launch configuration to map threads to (row, word) pairs.
    // 256 threads/block is a good default for A100/H100. We favor x-dimension for coalesced accesses.
    const dim3 block(32, 8, 1);
    const dim3 grid((wordsPerRow + block.x - 1) / block.x,
                    (N           + block.y - 1) / block.y,
                    1);

    life_kernel_bitpacked<<<grid, block>>>(input, output, N, wordsPerRow);
    // Caller is responsible for synchronization and error checking if needed.
}