#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel that computes one step of Conway's Game of Life on a bit‐packed grid.
// Each thread processes one 64‐bit word representing 64 consecutive cells in one row.
// The grid is represented as a square grid of cells with dimensions grid_dimensions x grid_dimensions,
// where each row contains grid_dimensions/64 words. Cells outside of the grid are considered dead.
__global__ void game_of_life_kernel(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Each 64‐bit word encodes 64 consecutive cells; therefore, number of words per row:
    int words_per_row = grid_dimensions >> 6; // equivalent to grid_dimensions / 64

    // Compute the thread's word coordinates: row index (r) and column index (c) in the bit‐packed grid.
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= grid_dimensions || c >= words_per_row)
        return; // Out‐of-bound threads do nothing

    // Load the nine neighboring words. For cells at the boundary in any direction,
    // missing neighbors are treated as all-zero (dead).
    //
    // The nine words correspond to:
    //  - Row above: upper_left, upper, upper_right
    //  - Same row:   left,      center, right
    //  - Row below:  lower_left, lower, lower_right
    //
    // Within each word, cells are indexed from 0 to 63, where the 0th cell is the leftmost
    // and the 63rd cell is the rightmost in that 64-bit segment.
    std::uint64_t upper_left  = 0, upper     = 0, upper_right  = 0;
    std::uint64_t left        = 0, center    = 0, right        = 0;
    std::uint64_t lower_left  = 0, lower     = 0, lower_right  = 0;

    // Row above: if available
    if (r > 0)
    {
        int idx = (r - 1) * words_per_row;
        upper = input[idx + c];
        if (c > 0)
            upper_left = input[idx + c - 1];
        if (c < words_per_row - 1)
            upper_right = input[idx + c + 1];
    }
    // The current row: always available
    {
        int idx = r * words_per_row;
        center = input[idx + c];
        if (c > 0)
            left = input[idx + c - 1];
        if (c < words_per_row - 1)
            right = input[idx + c + 1];
    }
    // Row below: if available
    if (r < grid_dimensions - 1)
    {
        int idx = (r + 1) * words_per_row;
        lower = input[idx + c];
        if (c > 0)
            lower_left = input[idx + c - 1];
        if (c < words_per_row - 1)
            lower_right = input[idx + c + 1];
    }

    // Process each of the 64 cells in the current word.
    // For each bit index, compute the sum of live neighbors and apply the Game of Life rules.
    std::uint64_t next_word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit)
    {
        int neighbors = 0;

        // Row above:
        // For the left neighbor in the row above, if this is the first bit (bit==0)
        // then fetch bit 63 from the upper_left word; otherwise, fetch bit (bit-1) from the upper word.
        int up_left = (bit == 0) ? int((upper_left >> 63) & 1ULL) : int((upper >> (bit - 1)) & 1ULL);
        int up_center = int((upper >> bit) & 1ULL);
        // For the right neighbor in the row above, if this is the last bit (bit==63)
        // then fetch bit 0 from the upper_right word; otherwise, fetch bit (bit+1) from the upper word.
        int up_right = (bit == 63) ? int((upper_right >> 0) & 1ULL) : int((upper >> (bit + 1)) & 1ULL);
        neighbors += up_left + up_center + up_right;

        // Same row (excluding the center cell itself):
        int left_val = (bit == 0) ? int((left >> 63) & 1ULL) : int((center >> (bit - 1)) & 1ULL);
        int right_val = (bit == 63) ? int((right >> 0) & 1ULL) : int((center >> (bit + 1)) & 1ULL);
        neighbors += left_val + right_val;

        // Row below:
        int low_left = (bit == 0) ? int((lower_left >> 63) & 1ULL) : int((lower >> (bit - 1)) & 1ULL);
        int low_center = int((lower >> bit) & 1ULL);
        int low_right = (bit == 63) ? int((lower_right >> 0) & 1ULL) : int((lower >> (bit + 1)) & 1ULL);
        neighbors += low_left + low_center + low_right;

        // Extract the current cell's value from the center word.
        int cell = int((center >> bit) & 1ULL);
        // Apply Conway's Game of Life rules:
        //   - A live cell with 2 or 3 live neighbors survives.
        //   - A dead cell with exactly 3 live neighbors becomes alive.
        int new_cell = ((neighbors == 3) || ((cell == 1) && (neighbors == 2))) ? 1 : 0;
        // Set the computed new cell's value at position 'bit' in the next_word.
        next_word |= (std::uint64_t(new_cell) << bit);
    }

    // Write out the new 64-bit word in the output grid.
    output[r * words_per_row + c] = next_word;
}

// Host function that launches one step of the Game of Life simulation.
// The input and output grids are bit-packed arrays allocated in GPU memory via cudaMalloc.
// The grid dimensions specify the width and height of the square grid, and are assumed to be
// a power of 2 and at least 512.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Calculate the number of 64-bit words per row.
    int words_per_row = grid_dimensions >> 6; // grid_dimensions / 64

    // Choose a 2D thread block configuration.
    // Each thread processes one 64-bit word.
    dim3 blockDim(16, 16);
    // Compute grid dimensions in terms of words:
    //   - The x-dimension is the number of words per row.
    //   - The y-dimension is the number of rows (which is grid_dimensions).
    dim3 gridDim((words_per_row + blockDim.x - 1) / blockDim.x,
                 (grid_dimensions + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, grid_dimensions);
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
