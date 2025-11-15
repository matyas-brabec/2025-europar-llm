// Conway's Game of Life - CUDA implementation using bit-packed 64-bit words.
//
// Each std::uint64_t encodes 64 consecutive cells in a row (bit 0 is the
// least-significant bit, bit 63 the most-significant). The grid is square,
// with size `grid_dimensions x grid_dimensions`, where `grid_dimensions` is
// a power of two greater than 512, and always divisible by 64.
//
// This implementation computes one simulation step entirely on the GPU.
// It assigns one thread per 64-bit word (i.e., per group of 64 horizontal
// cells). For each such word, it:
//   1. Loads up to 9 words from global memory (current word plus its
//      neighbors in the 3x3 word neighborhood).
//   2. Constructs 8 64-bit masks representing the alive neighbors in each
//      of the 8 directions (N, NE, E, SE, S, SW, W, NW).
//   3. Uses a bit-sliced counter (4 bitplanes: ones, twos, fours, eights)
//      to count neighbors per bit position (0..8).
//   4. Applies the Game of Life rules using bitwise logic to obtain the
//      next state of all 64 cells in that word.
//
// All cells outside the grid are treated as dead. No shared or texture
// memory is used; only global memory and registers are employed.
//
// The core kernel is `game_of_life_step_kernel`, and the host function
// `run_game_of_life` launches the kernel exactly once (one time step).
//
// Note: Any host-device synchronization is left to the caller of
// `run_game_of_life` as required.

#include <cstdint>
#include <cuda_runtime.h>

// Small helper alias for readability.
using u64 = std::uint64_t;

// Compute the "west" neighbor mask for a word.
// For each cell in `center`, its west neighbor is either:
//   - the bit immediately to its left in `center`, or
//   - for bit 0, bit 63 of `left_word`.
// `left_word` must be 0 at the left boundary to enforce dead cells outside.
static __device__ __forceinline__
u64 shift_west(u64 center, u64 left_word)
{
    // (center << 1) shifts bits towards more significant positions;
    // the least significant bit is filled with 0.
    // (left_word >> 63) extracts its MSB and moves it into bit 0.
    return (center << 1) | (left_word >> 63);
}

// Compute the "east" neighbor mask for a word.
// For each cell in `center`, its east neighbor is either:
//   - the bit immediately to its right in `center`, or
//   - for bit 63, bit 0 of `right_word`.
// `right_word` must be 0 at the right boundary to enforce dead cells outside.
static __device__ __forceinline__
u64 shift_east(u64 center, u64 right_word)
{
    // (center >> 1) shifts bits towards less significant positions;
    // the most significant bit is filled with 0.
    // (right_word << 63) moves its LSB into bit 63.
    return (center >> 1) | (right_word << 63);
}

// Bit-sliced addition of a 1-bit mask into a 4-bit (0..8) counter per bit.
//
// The neighbor count per cell is represented using 4 bitplanes:
//   - ones  : bit 0 of the per-cell counter
//   - twos  : bit 1
//   - fours : bit 2
//   - eights: bit 3
//
// Each bit in 'mask' is either 0 or 1 and we conceptually add it to the
// 4-bit counter for that bit position. The operations below implement
// per-bit addition with no carries between bit positions (everything
// uses AND/XOR/OR, which are bitwise).
static __device__ __forceinline__
void add_mask_into_counter(u64 mask, u64 &ones, u64 &twos, u64 &fours, u64 &eights)
{
    // First add to the ones bitplane.
    u64 carry = ones & mask;
    ones ^= mask;

    // Propagate carry into the twos bitplane.
    u64 carry2 = twos & carry;
    twos ^= carry;

    // Propagate carry into the fours bitplane.
    u64 carry3 = fours & carry2;
    fours ^= carry2;

    // Any carry out of fours sets the eights bitplane.
    eights |= carry3;
}

// CUDA kernel: compute one Game of Life step for a bit-packed grid.
//
// Parameters:
//   input          - pointer to device memory, bit-packed grid, N x N cells.
//   output         - pointer to device memory, same layout, to store next state.
//   grid_dim       - N, the width and height of the square grid.
//   words_per_row  - number of 64-bit words per row (grid_dim / 64).
//
// One thread processes one 64-bit word (64 horizontal cells).
__global__ void game_of_life_step_kernel(const u64* __restrict__ input,
                                         u64* __restrict__ output,
                                         int grid_dim,
                                         std::size_t words_per_row)
{
    // Compute word coordinates within the grid in terms of word indices.
    const std::size_t word_x = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t y      = static_cast<std::size_t>(blockIdx.y) * blockDim.y +
                               static_cast<std::size_t>(threadIdx.y);

    if (word_x >= words_per_row || y >= static_cast<std::size_t>(grid_dim))
        return;

    const std::size_t idx = y * words_per_row + word_x;

    // Load current word and its horizontal neighbors in the same row.
    const u64 center = input[idx];
    const bool has_left  = (word_x > 0);
    const bool has_right = (word_x + 1 < words_per_row);

    const u64 left  = has_left  ? input[idx - 1] : 0ull;
    const u64 right = has_right ? input[idx + 1] : 0ull;

    // Load words from row above and below (and their horizontal neighbors).
    const bool has_up    = (y > 0);
    const bool has_down  = (y + 1 < static_cast<std::size_t>(grid_dim));

    const std::size_t row_stride = words_per_row;

    u64 up        = 0ull;
    u64 up_left   = 0ull;
    u64 up_right  = 0ull;
    u64 down      = 0ull;
    u64 down_left = 0ull;
    u64 down_right= 0ull;

    if (has_up) {
        const std::size_t up_idx = idx - row_stride;
        up = input[up_idx];
        if (has_left)
            up_left = input[up_idx - 1];
        if (has_right)
            up_right = input[up_idx + 1];
    }

    if (has_down) {
        const std::size_t down_idx = idx + row_stride;
        down = input[down_idx];
        if (has_left)
            down_left = input[down_idx - 1];
        if (has_right)
            down_right = input[down_idx + 1];
    }

    // Construct neighbor masks in each of the 8 directions.
    // N and S are just the words directly above and below.
    const u64 maskN  = up;
    const u64 maskS  = down;

    // E and W from current row (with cross-word shifts).
    const u64 maskW  = shift_west(center, left);
    const u64 maskE  = shift_east(center, right);

    // For NW, NE, SW, SE we shift the appropriate row up/down with neighbors.
    const u64 maskNW = shift_west(up,   up_left);
    const u64 maskNE = shift_east(up,   up_right);
    const u64 maskSW = shift_west(down, down_left);
    const u64 maskSE = shift_east(down, down_right);

    // Bit-sliced neighbor count accumulator (0..8).
    u64 ones   = 0ull;  // bit 0 of count
    u64 twos   = 0ull;  // bit 1 of count
    u64 fours  = 0ull;  // bit 2 of count
    u64 eights = 0ull;  // bit 3 of count

    // Add all eight neighbor masks into the per-bit counter.
    add_mask_into_counter(maskN,  ones, twos, fours, eights);
    add_mask_into_counter(maskNE, ones, twos, fours, eights);
    add_mask_into_counter(maskE,  ones, twos, fours, eights);
    add_mask_into_counter(maskSE, ones, twos, fours, eights);
    add_mask_into_counter(maskS,  ones, twos, fours, eights);
    add_mask_into_counter(maskSW, ones, twos, fours, eights);
    add_mask_into_counter(maskW,  ones, twos, fours, eights);
    add_mask_into_counter(maskNW, ones, twos, fours, eights);

    // Now we have the neighbor count in binary per bit:
    //   count = ones (bit 0) + 2*twos (bit 1) + 4*fours (bit 2) + 8*eights (bit 3).
    //
    // We need masks for:
    //   - eq2: count == 2 (binary 0010)
    //   - eq3: count == 3 (binary 0011)
    //
    // The conditions per bit are:
    //   eq2:  !ones & twos & !fours & !eights
    //   eq3:   ones & twos & !fours & !eights
    const u64 not_ones   = ~ones;
    const u64 not_fours  = ~fours;
    const u64 not_eights = ~eights;

    const u64 eq2 = not_ones & twos & not_fours & not_eights;
    const u64 eq3 = ones     & twos & not_fours & not_eights;

    // Current alive cells.
    const u64 alive = center;

    // Apply Game of Life rules:
    //   - Any alive cell with 2 or 3 neighbors survives.
    //   - Any dead cell with exactly 3 neighbors becomes alive.
    //
    // Masks:
    //   survive = alive & (eq2 | eq3)
    //   born    = ~alive & eq3
    const u64 survive = alive & (eq2 | eq3);
    const u64 born    = (~alive) & eq3;
    const u64 next    = survive | born;

    // Store result.
    output[idx] = next;
}

// Host function to run one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   input           - device pointer to bit-packed input grid.
//   output          - device pointer to bit-packed output grid.
//   grid_dimensions - side length N of the square grid (N x N), power of 2.
//
// Both `input` and `output` must be allocated with cudaMalloc; no
// host-device synchronization or memory copies are performed here.
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int    dim           = grid_dimensions;
    const size_t words_per_row = static_cast<size_t>(dim) >> 6;  // dim / 64

    // Configure a 2D grid where:
    //   - x dimension covers the words within a row,
    //   - y dimension covers the rows.
    //
    // Block size is chosen to give a reasonable number of threads per block
    // and good memory coalescing along the x direction.
    dim3 blockDim(32, 8, 1);  // 256 threads per block

    dim3 gridDim(
        static_cast<unsigned int>((words_per_row + blockDim.x - 1) / blockDim.x),
        static_cast<unsigned int>((static_cast<size_t>(dim) + blockDim.y - 1) / blockDim.y),
        1
    );

    // Launch kernel asynchronously. The caller is responsible for
    // performing any necessary synchronization.
    game_of_life_step_kernel<<<gridDim, blockDim>>>(input, output, dim, words_per_row);
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
