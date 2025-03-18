#include <cstdint>
#include <cuda_runtime.h>

// This CUDA kernel performs one step of Conway's Game of Life on a 2D grid
// where the grid is stored in a bit‐packed format. Each 64‐bit word encodes 64 cells
// of a single row. Threads are mapped such that each thread processes one 64-bit word.
// The kernel reads the 3×3 neighborhood of cells (taking care of special wrapping
// at the boundaries between adjacent words in the same row) and applies the standard
// Game of Life rules to compute the new state of each cell.
// Note: Cells outside the grid are treated as dead.
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dimensions,     // number of rows/columns in cells
                                    int words_per_row)       // number of 64-bit words per row (grid_dimensions/64)
{
    // Determine the position within the grid of 64-bit words.
    // 'row' is the row index (each row contains grid_dimensions cells).
    // 'col' is the word index within the row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread is within valid boundaries.
    if (row >= grid_dimensions || col >= words_per_row)
        return;

    // Compute the linear index for the current word.
    int idx = row * words_per_row + col;

    // Load the current 64-bit word (representing 64 cells) for the current row.
    std::uint64_t cur = input[idx];

    // For the current row, load neighboring words if available.
    // For bit positions at the left (bit0) and right (bit63) special handling is needed.
    std::uint64_t left  = (col > 0) ? input[row * words_per_row + col - 1] : 0ULL;
    std::uint64_t right = (col < words_per_row - 1) ? input[row * words_per_row + col + 1] : 0ULL;

    // For the top (row-1) row: load the center and left/right words if the row exists;
    // otherwise treat as zero (all dead).
    std::uint64_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_idx = (row - 1) * words_per_row + col;
        top = input[top_idx];
        top_left  = (col > 0) ? input[(row - 1) * words_per_row + col - 1] : 0ULL;
        top_right = (col < words_per_row - 1) ? input[(row - 1) * words_per_row + col + 1] : 0ULL;
    }

    // For the bottom (row+1) row: similarly load words if available.
    std::uint64_t bottom = 0, bottom_left = 0, bottom_right = 0;
    if (row < grid_dimensions - 1) {
        int bottom_idx = (row + 1) * words_per_row + col;
        bottom = input[bottom_idx];
        bottom_left  = (col > 0) ? input[(row + 1) * words_per_row + col - 1] : 0ULL;
        bottom_right = (col < words_per_row - 1) ? input[(row + 1) * words_per_row + col + 1] : 0ULL;
    }

    // Compute the new state for the 64 cells in the current word.
    // We will loop over each bit (cell) and compute the number of live neighbors.
    std::uint64_t res = 0ULL;
    for (int bit = 0; bit < 64; bit++) {
        int count = 0;  // will count the live neighbors around the current cell

        // For the top row of neighbors (if available).
        if (row > 0) {
            // Top-left neighbor:
            if (bit == 0)
                count += (top_left >> 63) & 1ULL;
            else
                count += (top >> (bit - 1)) & 1ULL;

            // Top-center neighbor.
            count += (top >> bit) & 1ULL;

            // Top-right neighbor:
            if (bit == 63)
                count += (top_right >> 0) & 1ULL;
            else
                count += (top >> (bit + 1)) & 1ULL;
        }

        // For the current row (neighbors in the same row, excluding the cell itself).
        // Left neighbor:
        if (bit == 0)
            count += (left >> 63) & 1ULL;
        else
            count += (cur >> (bit - 1)) & 1ULL;

        // Right neighbor:
        if (bit == 63)
            count += (right >> 0) & 1ULL;
        else
            count += (cur >> (bit + 1)) & 1ULL;

        // For the bottom row of neighbors (if available).
        if (row < grid_dimensions - 1) {
            // Bottom-left neighbor:
            if (bit == 0)
                count += (bottom_left >> 63) & 1ULL;
            else
                count += (bottom >> (bit - 1)) & 1ULL;

            // Bottom-center neighbor.
            count += (bottom >> bit) & 1ULL;

            // Bottom-right neighbor:
            if (bit == 63)
                count += (bottom_right >> 0) & 1ULL;
            else
                count += (bottom >> (bit + 1)) & 1ULL;
        }

        // Extract the current cell state (1 for alive, 0 for dead)
        int cell = (cur >> bit) & 1ULL;

        // Apply Game of Life rules:
        // - A dead cell with exactly 3 live neighbors becomes alive.
        // - A live cell with 2 or 3 live neighbors survives.
        // Otherwise the cell dies or remains dead.
        int newState = (count == 3) || (cell && (count == 2));

        // Pack the resulting bit into the output.
        res |= (static_cast<std::uint64_t>(newState) << bit);
    }

    // Store the computed word into the output grid.
    output[idx] = res;
}

// Host function that launches one step of the Game of Life simulation.
// It takes the input grid and computes the next generation into the output grid.
// The grid is bit-packed so that each std::uint64_t holds 64 cells in a row.
// The grid is square with dimensions grid_dimensions x grid_dimensions.
// Note: It is assumed that grid_dimensions is a power of 2 and a multiple of 64.
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64-bit words per row.
    int words_per_row = grid_dimensions / 64;

    // Choose a 2D thread block that balances occupancy and work distribution.
    // Here, we choose a block of 16x16 threads.
    dim3 block(16, 16);

    // Compute the grid dimensions in terms of 64-bit words.
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. All necessary host-device synchronization is assumed
    // to be handled by the caller.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions, words_per_row);
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
