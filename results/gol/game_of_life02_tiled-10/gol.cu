#include <cuda_runtime.h>
#include <cstdint>

// Conway’s Game of Life on a bit-packed 8x8-tile grid.
// Each std::uint64_t encodes an 8×8 tile, with bit index b = y*8 + x (row-major within the tile),
// where y in [0,7] is the row (top to bottom), and x in [0,7] is the column (left to right).
// The least significant bit (bit 0) corresponds to the tile cell at (y=0, x=0).
//
// This implementation assigns one CUDA thread per 8×8 tile. Each thread loads the 3×3 neighborhood
// of tiles around its tile (9 words total; out-of-bounds neighbors are treated as 0), then computes
// the next state for its 64 cells using bit-parallel operations. We avoid using shared or texture memory,
// adhering to the problem's guidance, and rely on simple global memory loads and register-heavy computation.
//
// The per-tile update uses a row-by-row bit-parallel algorithm. For each of the 8 rows within the tile,
// it constructs the 3 relevant rows (up, current, down) across the 3 horizontal tiles (west, center, east)
// with correct edge handling across tile boundaries. It computes the neighbor count modulo 2 (ones)
// and the number of pairs (twos, i.e., floor(count/2) for each row triple) using only bitwise operations.
// Then, vertical combination across the three row triples produces, for each bit/column in the row,
// the conditions "neighbors == 2" and "neighbors == 3" exactly, without ever forming full multi-bit counts.
// Finally, the Game of Life rule is applied:
//   next = (neighbors == 3) | (alive & (neighbors == 2))
//
// Boundary conditions: all cells outside the grid are dead (0).
// Tiling: The grid is tiles_per_dim = grid_dimensions / 8 per side. The tiles are stored in row-major order:
// tile index = tile_y * tiles_per_dim + tile_x.

static __device__ __forceinline__ void unpack_rows(std::uint64_t tile, std::uint8_t rows[8]) {
    // Extract 8 rows (each 8-bit) from a 64-bit tile.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        rows[i] = static_cast<std::uint8_t>(tile >> (i * 8));
    }
}

// majority of three bitmasks (per-bit majority), used for carry-from-ones across rows.
static __device__ __forceinline__ std::uint8_t majority3(std::uint8_t a, std::uint8_t b, std::uint8_t c) {
    // (a & b) | (a & c) | (b & c)
    return static_cast<std::uint8_t>((a & b) | (a & c) | (b & c));
}

// Compute mask of positions where exactly one of the four bitmasks has a 1.
// This supports testing whether the "units-of-2" sum across rows equals exactly 1 for each bit position.
static __device__ __forceinline__ std::uint8_t exactly_one_of4(std::uint8_t a, std::uint8_t b, std::uint8_t c, std::uint8_t d) {
    std::uint8_t parity = static_cast<std::uint8_t>(a ^ b ^ c ^ d);
    // at least two among (a,b,c,d): any pairwise AND
    std::uint8_t ab = static_cast<std::uint8_t>(a & b);
    std::uint8_t ac = static_cast<std::uint8_t>(a & c);
    std::uint8_t ad = static_cast<std::uint8_t>(a & d);
    std::uint8_t bc = static_cast<std::uint8_t>(b & c);
    std::uint8_t bd = static_cast<std::uint8_t>(b & d);
    std::uint8_t cd = static_cast<std::uint8_t>(c & d);
    std::uint8_t anyPair = static_cast<std::uint8_t>(ab | ac | ad | bc | bd | cd);
    return static_cast<std::uint8_t>(parity & static_cast<std::uint8_t>(~anyPair));
}

// Shift a row left by one within an 8-bit lane and bring-in the bit from the neighbor row's MSB as bit0.
// This computes "west neighbor" contributions respecting tile boundaries across tiles.
static __device__ __forceinline__ std::uint8_t left_of_row(std::uint8_t row, std::uint8_t neighbor_w_row) {
    // ((row << 1) & 0xFE) | (neighbor_w_row >> 7)
    return static_cast<std::uint8_t>(static_cast<std::uint8_t>((row << 1) & 0xFEu) | static_cast<std::uint8_t>(neighbor_w_row >> 7));
}

// Shift a row right by one within an 8-bit lane and bring-in the bit from the neighbor row's LSB as bit7.
// This computes "east neighbor" contributions respecting tile boundaries across tiles.
static __device__ __forceinline__ std::uint8_t right_of_row(std::uint8_t row, std::uint8_t neighbor_e_row) {
    // ((row >> 1) & 0x7F) | ((neighbor_e_row & 1) << 7)
    return static_cast<std::uint8_t>(static_cast<std::uint8_t>((row >> 1) & 0x7Fu) | static_cast<std::uint8_t>((neighbor_e_row & 1u) << 7));
}

__global__ void life_step_kernel_tile8x8(const std::uint64_t* __restrict__ input,
                                         std::uint64_t* __restrict__ output,
                                         int tiles_per_dim) {
    const std::uint64_t total_tiles = static_cast<std::uint64_t>(tiles_per_dim) * static_cast<std::uint64_t>(tiles_per_dim);
    const std::uint64_t threads = static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(gridDim.x);
    std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);

    while (tid < total_tiles) {
        int ty = static_cast<int>(tid / tiles_per_dim);
        int tx = static_cast<int>(tid % tiles_per_dim);

        // Neighbor availability flags
        const bool hasN = (ty > 0);
        const bool hasS = (ty + 1 < tiles_per_dim);
        const bool hasW = (tx > 0);
        const bool hasE = (tx + 1 < tiles_per_dim);

        const std::uint64_t idxC = static_cast<std::uint64_t>(ty) * tiles_per_dim + tx;

        // Load 3x3 tile neighborhood; out-of-bounds => 0 (dead)
        // Using simple conditional loads; on modern GPUs, divergence here is minimal compared to total work.
        const std::uint64_t tileC  = input[idxC];

        const std::uint64_t tileN  = hasN ? input[idxC - tiles_per_dim] : 0ull;
        const std::uint64_t tileS  = hasS ? input[idxC + tiles_per_dim] : 0ull;
        const std::uint64_t tileW  = hasW ? input[idxC - 1] : 0ull;
        const std::uint64_t tileE  = hasE ? input[idxC + 1] : 0ull;

        const std::uint64_t tileNW = (hasN && hasW) ? input[idxC - tiles_per_dim - 1] : 0ull;
        const std::uint64_t tileNE = (hasN && hasE) ? input[idxC - tiles_per_dim + 1] : 0ull;
        const std::uint64_t tileSW = (hasS && hasW) ? input[idxC + tiles_per_dim - 1] : 0ull;
        const std::uint64_t tileSE = (hasS && hasE) ? input[idxC + tiles_per_dim + 1] : 0ull;

        // Unpack rows (8 bytes per tile). This saves repeated shifts in the inner loop.
        std::uint8_t rC[8], rN[8], rS[8], rW[8], rE[8], rNW[8], rNE[8], rSW[8], rSE[8];
        unpack_rows(tileC,  rC);
        unpack_rows(tileN,  rN);
        unpack_rows(tileS,  rS);
        unpack_rows(tileW,  rW);
        unpack_rows(tileE,  rE);
        unpack_rows(tileNW, rNW);
        unpack_rows(tileNE, rNE);
        unpack_rows(tileSW, rSW);
        unpack_rows(tileSE, rSE);

        // Compute next state for the 8x8 tile row-by-row, assembling the result into a 64-bit word.
        std::uint64_t result = 0ull;

        #pragma unroll
        for (int y = 0; y < 8; ++y) {
            // Rows for current, up, and down relative to this row y, with boundary fixups across tiles.
            const std::uint8_t cur = rC[y];
            const std::uint8_t up  = (y > 0) ? rC[y - 1] : rN[7];
            const std::uint8_t dn  = (y < 7) ? rC[y + 1] : rS[0];

            // Neighbor rows horizontally: west/east for the same y, and also for up/dn when needed.
            const std::uint8_t rowW  = rW[y];
            const std::uint8_t rowE  = rE[y];

            const std::uint8_t upW   = (y > 0) ? rW[y - 1] : rNW[7];
            const std::uint8_t upE   = (y > 0) ? rE[y - 1] : rNE[7];

            const std::uint8_t dnW   = (y < 7) ? rW[y + 1] : rSW[0];
            const std::uint8_t dnE   = (y < 7) ? rE[y + 1] : rSE[0];

            // Horizontal neighbor masks for each of the three row triples:
            // For up and dn rows: include center row (vertical neighbors N and S).
            // For mid row: exclude the center cell (only W and E neighbors), to avoid counting itself.
            const std::uint8_t L_up  = left_of_row(up, upW);
            const std::uint8_t R_up  = right_of_row(up, upE);

            const std::uint8_t L_mid = left_of_row(cur, rowW);
            const std::uint8_t R_mid = right_of_row(cur, rowE);

            const std::uint8_t L_dn  = left_of_row(dn, dnW);
            const std::uint8_t R_dn  = right_of_row(dn, dnE);

            // Row triple reductions into ones (bit0) and twos (bit1) per position:
            // ones = parity of (L, C, R)
            // twos = majority(L, C, R) => 1 if at least two among (L,C,R) are 1
            const std::uint8_t ones_up  = static_cast<std::uint8_t>(L_up ^ up ^ R_up);
            const std::uint8_t twos_up  = majority3(L_up, up, R_up);

            const std::uint8_t ones_mid = static_cast<std::uint8_t>(L_mid ^ R_mid);     // center excluded
            const std::uint8_t twos_mid = static_cast<std::uint8_t>(L_mid & R_mid);     // majority(L,0,R) == L & R

            const std::uint8_t ones_dn  = static_cast<std::uint8_t>(L_dn ^ dn ^ R_dn);
            const std::uint8_t twos_dn  = majority3(L_dn, dn, R_dn);

            // Vertical reduction across the three rows:
            // ones_sum is parity of ones across rows (bit0 of neighbor count).
            const std::uint8_t ones_sum = static_cast<std::uint8_t>(ones_up ^ ones_mid ^ ones_dn);

            // c01 is carry from ones across rows into the weight-2 plane: majority of (ones_up, ones_mid, ones_dn).
            const std::uint8_t c01 = majority3(ones_up, ones_mid, ones_dn);

            // units-of-2 count (per position) equals c01 + twos_up + twos_mid + twos_dn.
            // We need to know when this equals exactly 1 to identify counts 2 or 3:
            const std::uint8_t exactly_one2 = exactly_one_of4(c01, twos_up, twos_mid, twos_dn);

            // neighbors == 2: ones_sum == 0 AND exactly_one2 == 1
            const std::uint8_t eq2 = static_cast<std::uint8_t>(static_cast<std::uint8_t>(~ones_sum) & exactly_one2);

            // neighbors == 3: ones_sum == 1 AND exactly_one2 == 1
            const std::uint8_t eq3 = static_cast<std::uint8_t>(ones_sum & exactly_one2);

            // Apply Game of Life rule: next = (neighbors == 3) | (alive & (neighbors == 2))
            const std::uint8_t next_row = static_cast<std::uint8_t>(eq3 | static_cast<std::uint8_t>(cur & eq2));

            // Place this 8-bit row result back into the 64-bit tile word.
            result |= (static_cast<std::uint64_t>(next_row) << (y * 8));
        }

        output[idxC] = result;
        tid += threads;
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    // grid_dimensions is a power of two, >= 512, and divisible by 8 by problem statement.
    const int tiles_per_dim = grid_dimensions >> 3;
    const std::uint64_t total_tiles = static_cast<std::uint64_t>(tiles_per_dim) * static_cast<std::uint64_t>(tiles_per_dim);

    // Choose a launch configuration that maps one thread per tile, grid-stride inside kernel for scalability.
    const int block_size = 256;
    const int grid_size = static_cast<int>((total_tiles + block_size - 1) / block_size);

    life_step_kernel_tile8x8<<<grid_size, block_size>>>(input, output, tiles_per_dim);
    // No synchronization here; the caller handles host-device synchronization as per the requirements.
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
// MEMORY_LAYOUT: TILES
