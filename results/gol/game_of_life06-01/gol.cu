/******************************************************************************/
/* CUDA implementation of one Game‐of‐Life generation.
 *
 * The grid is square with grid_dimensions cells per side.
 * Cells are bit–packed into std::uint64_t’s such that one word holds 64 cells.
 * We assume that the grid dimensions are a power‐of–2 and that grid_dimensions 
 * is a multiple of 64.
 *
 * Bit–packing convention:
 *   In each 64–bit word the 0th bit (i.e. the least–significant bit) represents 
 *   the leftmost cell and the 63rd bit (most–significant bit) represents the 
 *   rightmost cell.
 *
 * To obtain the eight neighbor masks for each word (for north, south, west, east 
 * and the four diagonals) we use bit–shifts.  In particular, note that:
 *
 *   - To shift the pattern “left” (i.e. to obtain the west–neighbor contribution)
 *     we use a left–shift (<< 1); to shift “right” (east–neighbor) we use a right–shift (>> 1).
 *
 *   - When a neighbor word in the same row is available (to the left or right of our
 *     word), we must extract the missing boundary bit and merge it in.
 *     For the west neighbor, if this thread’s word is not the leftmost word in its row,
 *     the missing left (west) bit comes from the previous word’s rightmost bit:
 *         extra = (prev_word >> 63) & 1.
 *     For the east neighbor, if not rightmost, the missing right (east) bit comes from 
 *         extra = (next_word << 63)
 *
 *   - For the north and south rows, analogous logic is applied.
 *
 * To compute the neighbor count for 64 cells concurrently we “bit–slice” a 4–bit sum 
 * (which can hold values 0–8) into four 64–bit accumulators (sum0, sum1, sum2, sum3).  
 * Each accumulator’s jth bit holds one bit of the 4–bit number for cell j.
 *
 * We then add (using full–adder logic without inter–bit carries between independent cells)
 * the contribution of each of the eight neighboring directions.
 *
 * Finally, we compute the next cell state using the Game–of–Life rules:
 *   A cell becomes alive if it has exactly 3 live neighbors, or if it is already alive 
 *   and has exactly 2 live neighbors.
 *
 * The new state is computed per–bit as:
 *      new_cell = (neighbor_count == 3) OR (center & (neighbor_count == 2)).
 *
 * The equality tests (==3 and ==2) are computed “bit–parallel” from the bit–sliced sum:
 *   neighbor count == 3  <==>   (sum3==0) & (sum2==0) & (sum1==1) & (sum0==1)
 *   neighbor count == 2  <==>   (sum3==0) & (sum2==0) & (sum1==1) & (sum0==0)
 *
 * All required neighboring words (from the current row, the one above and the one below)
 * are loaded from global memory. Boundary conditions assume that all cells outside 
 * the grid are dead.
 *
 * Each CUDA thread processes one std::uint64_t word (i.e. 64 cells).
 *
 * The kernel is optimized for modern NVIDIA GPUs (H100, A100) compiled with the latest
 * CUDA toolkit.
 *
 * Author: Experienced CUDA programmer
 */
/******************************************************************************/

#include <cstdint>
#include <cuda_runtime.h>

// __device__ inline function to add one 1–bit value (mask m) into a bit–sliced 4–bit counter.
// The 4–bit counter is stored in (s3,s2,s1,s0); for each cell j the counter is:
//    count[j] = ( (s3>>j)&1)*8 + ((s2>>j)&1)*4 + ((s1>>j)&1)*2 + ((s0>>j)&1)
__device__ inline void add_bit(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3, uint64_t m) {
    // First full–adder stage: add m to the current least–significant bit.
    uint64_t t0 = s0 ^ m;         // sum bit = XOR of s0 and m.
    uint64_t carry = s0 & m;      // carry bit for this stage.
    // Second stage.
    uint64_t t1 = s1 ^ carry;
    carry = s1 & carry;
    // Third stage.
    uint64_t t2 = s2 ^ carry;
    carry = s2 & carry;
    // Fourth stage.
    uint64_t t3 = s3 ^ carry;
    // Update the bit–sliced counters.
    s0 = t0;
    s1 = t1;
    s2 = t2;
    s3 = t3;
}

// CUDA kernel to compute one Game-of-Life generation.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim)
{
    // Each row has grid_dim cells; each word holds 64 cells.
    int words_per_row = grid_dim >> 6;  // grid_dim / 64

    // Compute thread's word coordinates.
    int col = blockIdx.x * blockDim.x + threadIdx.x; // word index within row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // row index.
    if (row >= grid_dim || col >= words_per_row)
        return;

    int idx = row * words_per_row + col;
    // Load the "center" word.
    std::uint64_t center = input[idx];

    // Compute neighbor contributions from the same row.
    // Using our bit–packing convention: in each word the 0th bit is the leftmost cell 
    // and the 63rd bit is the rightmost cell.
    // For same row, the west neighbor is obtained by shifting left (<< 1). For cells that 
    // cross word boundaries, we incorporate the missing bit from the previous word.
    std::uint64_t w = (center << 1) |
         ((col > 0) ? ((input[row * words_per_row + (col - 1)] >> 63) & 1ULL) : 0ULL);
    // Similarly, the east neighbor is obtained by shifting right (>> 1) and merging from next word.
    std::uint64_t e = (center >> 1) |
         ((col < words_per_row - 1) ? (input[row * words_per_row + (col + 1)] << 63) : 0ULL);

    // Initialize neighbor masks for north and south directions.
    std::uint64_t N = 0, S = 0, nw = 0, ne = 0, sw = 0, se = 0;

    if (row > 0) {
        N = input[(row - 1) * words_per_row + col];
        nw = (N << 1) |
             ((col > 0) ? ((input[(row - 1) * words_per_row + (col - 1)] >> 63) & 1ULL) : 0ULL);
        ne = (N >> 1) |
             ((col < words_per_row - 1) ? (input[(row - 1) * words_per_row + (col + 1)] << 63) : 0ULL);
    }
    if (row < grid_dim - 1) {
        S = input[(row + 1) * words_per_row + col];
        sw = (S << 1) |
             ((col > 0) ? ((input[(row + 1) * words_per_row + (col - 1)] >> 63) & 1ULL) : 0ULL);
        se = (S >> 1) |
             ((col < words_per_row - 1) ? (input[(row + 1) * words_per_row + (col + 1)] << 63) : 0ULL);
    }

    // Also, N and S (directly above and below) are neighbor contributions.
    // Note: In the same row, we do not include the center word itself.

    // Bit–sliced 4–bit counters for the neighbor count; each of these 64 lanes holds one bit.
    std::uint64_t sum0 = 0;
    std::uint64_t sum1 = 0;
    std::uint64_t sum2 = 0;
    std::uint64_t sum3 = 0;

    // Add contributions from each neighbor direction.
    // The eight neighbors (if out–of–bounds, the corresponding mask is 0).
    add_bit(sum0, sum1, sum2, sum3, nw);
    add_bit(sum0, sum1, sum2, sum3, N);
    add_bit(sum0, sum1, sum2, sum3, ne);
    add_bit(sum0, sum1, sum2, sum3, w);
    add_bit(sum0, sum1, sum2, sum3, e);
    add_bit(sum0, sum1, sum2, sum3, sw);
    add_bit(sum0, sum1, sum2, sum3, S);
    add_bit(sum0, sum1, sum2, sum3, se);

    // Compute bit–masks for which neighbor count equals 3 or equals 2.
    // For count == 3, in binary 3 = 0011: that is (sum3==0) & (sum2==0) & (sum1==1) & (sum0==1).
    std::uint64_t eq3 = (~sum3) & (~sum2) & sum1 & sum0;
    // For count == 2, binary 2 = 0010: (sum3==0) & (sum2==0) & (sum1==1) & (sum0==0).
    std::uint64_t eq2 = (~sum3) & (~sum2) & sum1 & (~sum0);

    // Game–of–Life update:
    // A cell becomes live if it has exactly 3 neighbors, or if it is already live and has exactly 2.
    std::uint64_t next = eq3 | (center & eq2);

    // Write the result back to global memory.
    output[idx] = next;
}

// Host function to invoke one Game-of-Life step.
// 'input' and 'output' are device pointers to bit–packed grids (allocated via cudaMalloc).
// grid_dimensions is the number of cells on a side.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute number of 64–bit words per row.
    int words_per_row = grid_dimensions >> 6;  // grid_dimensions / 64

    // Choose a block size tuned for performance on modern GPUs.
    // Here we choose 32 threads in x and 8 threads in y.
    dim3 block(32, 8);
    // Compute grid dimensions so that each thread handles one 64–cell word.
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
    // (Synchronization is handled by the caller.)
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
