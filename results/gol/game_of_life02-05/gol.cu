#include <cstdint>
#include <cuda_runtime.h>

// __global__ kernel implementing one Game of Life step on a bit‐packed grid.
// Each thread is responsible for computing one 64‐bit word (i.e. 64 contiguous cells)
// of the output grid. The grid is square with dimensions grid_dimensions x grid_dimensions,
// and each row is stored as grid_dimensions/64 64‐bit words.
// The kernel implements the Game of Life rules by first precomputing, for each word,
// bit‐aligned neighbor masks for the upper, current, and lower rows. Boundaries are handled
// by treating cells outside the grid as 0 (dead).
//
__global__ void game_of_life_kernel(const std::uint64_t* input,
                                    std::uint64_t* output,
                                    int grid_dimensions)
{
    // Compute number of 64‐bit words per row.
    int nwords = grid_dimensions >> 6;  // equivalent to grid_dimensions / 64

    // "col" indexes the word in the row, "row" indexes the grid row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dimensions || col >= nwords)
        return;

    int index = row * nwords + col;

    // Flags indicating whether neighbors in the vertical direction exist.
    bool has_above = (row > 0);
    bool has_below = (row < grid_dimensions - 1);

    // --- Compute neighbor masks for the row above ---
    // If row above is available, load the neighboring words; otherwise, use 0.
    std::uint64_t ab_center = has_above ? input[(row - 1) * nwords + col] : 0;
    std::uint64_t ab_left_raw = (has_above && col > 0) ? input[(row - 1) * nwords + col - 1] : 0;
    std::uint64_t ab_right_raw = (has_above && col < nwords - 1) ? input[(row - 1) * nwords + col + 1] : 0;
    // For a cell in the current word, its upper-left neighbor comes from either:
    //   - if the cell is not the first bit, then from ab_center (shifted left by 1),
    //   - if the cell is the first bit, then from the left word (bit 63)
    // Combining both cases via bit‐parallel shifts:
    std::uint64_t above_left_mask = (ab_center << 1) |
                                    ((has_above && col > 0) ? ((ab_left_raw >> 63) & 1ULL) : 0ULL);
    // The direct upper neighbor is simply ab_center.
    std::uint64_t above_center_mask = ab_center;
    // For the upper-right neighbor, for cell at bit i:
    //   - if i < 63, the neighbor is ab_center shifted right by 1,
    //   - if i==63, then it is taken from the adjacent right word, bit 0.
    std::uint64_t above_right_mask = (ab_center >> 1) |
                                     ((has_above && col < nwords - 1) ? ((ab_right_raw & 1ULL) << 63) : 0ULL);

    // --- Compute neighbor masks for the current row ---
    // Load the current word.
    std::uint64_t cur_center = input[row * nwords + col];
    std::uint64_t cur_left_raw = (col > 0) ? input[row * nwords + col - 1] : 0;
    std::uint64_t cur_right_raw = (col < nwords - 1) ? input[row * nwords + col + 1] : 0;
    // For left neighbor in the current row:
    std::uint64_t left_mask = (cur_center << 1) |
                              ((col > 0) ? ((cur_left_raw >> 63) & 1ULL) : 0ULL);
    // For right neighbor in the current row:
    std::uint64_t right_mask = (cur_center >> 1) |
                               ((col < nwords - 1) ? ((cur_right_raw & 1ULL) << 63) : 0ULL);

    // --- Compute neighbor masks for the row below ---
    std::uint64_t bl_center = has_below ? input[(row + 1) * nwords + col] : 0;
    std::uint64_t bl_left_raw = (has_below && col > 0) ? input[(row + 1) * nwords + col - 1] : 0;
    std::uint64_t bl_right_raw = (has_below && col < nwords - 1) ? input[(row + 1) * nwords + col + 1] : 0;
    std::uint64_t below_left_mask = (bl_center << 1) |
                                    ((has_below && col > 0) ? ((bl_left_raw >> 63) & 1ULL) : 0ULL);
    std::uint64_t below_center_mask = bl_center;
    std::uint64_t below_right_mask = (bl_center >> 1) |
                                     ((has_below && col < nwords - 1) ? ((bl_right_raw & 1ULL) << 63) : 0ULL);

    // The eight neighbor masks for the 64 bits in the current word are now precomputed.
    // Each mask's bit i corresponds to the state (0 or 1) of that neighbor for the cell at bit i.
    //
    // Now loop over all 64 bit positions (cells) in the word, count the number of
    // live neighbors, and apply Game of Life rules:
    // - A cell becomes alive if it has exactly 3 neighbors.
    // - A live cell remains alive if it has 2 neighbors.
    std::uint64_t result = 0;
    for (int i = 0; i < 64; i++)
    {
        int count = ((above_left_mask >> i) & 1) +
                    ((above_center_mask >> i) & 1) +
                    ((above_right_mask >> i) & 1) +
                    ((left_mask >> i) & 1) +
                    ((right_mask >> i) & 1) +
                    ((below_left_mask >> i) & 1) +
                    ((below_center_mask >> i) & 1) +
                    ((below_right_mask >> i) & 1);

        int cell = (cur_center >> i) & 1;
        // A cell is alive if it exactly has 3 live neighbors, or if it is currently live and has 2 neighbors.
        int new_cell = (count == 3) || (cell && (count == 2));
        result |= (static_cast<std::uint64_t>(new_cell) << i);
    }

    // Write the computed 64-bit word into the output grid.
    output[index] = result;
}

// run_game_of_life executes one step of Conway's Game of Life by launching a CUDA kernel.
// The grids are bit-packed so that each std::uint64_t word represents 64 consecutive cells in a row.
// The grid is square with dimensions grid_dimensions x grid_dimensions.
// It is assumed that both input and output device pointers have been allocated with cudaMalloc.
// Host-device synchronization is assumed to be handled externally.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int nwords = grid_dimensions >> 6;  // equal to grid_dimensions / 64

    // Configure block and grid dimensions.
    // We choose a 2D block of size 16x16. The x-dimension indexes words in a row,
    // and the y-dimension indexes the rows.
    dim3 block(16, 16);
    dim3 grid((nwords + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);

    // Note: Synchronization (cudaDeviceSynchronize, etc.) is handled by the caller.
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
