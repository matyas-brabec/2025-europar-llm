You are an experienced CUDA programmer trying to provide the most optimized code for the given task and the target hardware.
Assume that the target hardware is a modern NVIDIA GPU for data centers such as the H100 or A100, and the code is compiled using the latest CUDA toolkit and the latest compatible host compiler.
The output should contain only the requested source code. The code must not contain any placeholders or unimplemented parts. All high-level overviews and specific explanations should be included in code comments. Do not output any non-source text.

Write a CUDA implementation of Conway’s Game of Life.

Given a 2D grid of cells, each cell can be in one of two states: dead (0) or alive (1). A cell’s neighborhood comprises eight adjacent cells (horizontal, vertical, and diagonal neighbors). The next generation of the grid is computed based on the following rules:
- Any alive cell with fewer than 2 alive neighbors dies (underpopulation).
- Any alive cell with 2 or 3 alive neighbors survives.
- Any alive cell with more than 3 alive neighbors dies (overpopulation).
- Any dead cell with exactly 3 alive neighbors becomes alive (reproduction).

Assume that all cells outside the grid are dead, this is important for handling boundary cells. Additionally, the grid dimensions are always a power of 2, greater than 512, and small enough to ensure both the input and output fit within GPU memory.

Besides the CUDA kernel, implement a `run_game_of_life` function that calls the kernel. The performance of the kernel and the simulation itself is the only concern. Any necessary data transformations can be performed inside this function, as they will not be measured. Any host-device synchronization is handled by the caller of `run_game_of_life`.

The function `run_game_of_life` executes one step of Conway’s Game of Life using a CUDA kernel.

- `input`: Pointer to the input grid, where each cell is represented as `true` (alive) or `false` (dead).
- `output`: Pointer to the output grid, storing the next state.
- `grid_dimensions`: The width/height of the square grid (always a power of 2).

```cpp
void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
```

Assume the input and output grids are duly allocated by `cudaMalloc`.

---

```c++
// Includes necessary CUDA runtime header.
#include <cuda_runtime.h>

// Define the block size used for tiling the grid in shared memory.
// A block of 32x32 threads is chosen for optimal occupancy on modern GPUs.
#define BLOCK_SIZE 32

// CUDA kernel implementing one step of Conway's Game of Life.
// Each thread computes the next state of one cell by reading from a shared memory tile.
// Shared memory is used to reduce global memory accesses by loading a tile with a 1-cell halo border.
__global__ void game_of_life_kernel(const bool* input, bool* output, int grid_dim) {
    // Allocate shared memory tile with an extra border (halo) of 1 cell in each direction.
    // The tile size is (BLOCK_SIZE + 2) x (BLOCK_SIZE + 2).
    __shared__ bool tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Compute global x and y indices for the cell processed by this thread.
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Compute the indices within the shared memory tile.
    // The central cell is stored at position (threadIdx.y + 1, threadIdx.x + 1).
    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // Load the central cell from global memory into shared memory.
    // If the global index is out-of-bound, treat the cell as dead.
    if (x < grid_dim && y < grid_dim)
        tile[local_y][local_x] = input[y * grid_dim + x];
    else
        tile[local_y][local_x] = false;

    // Load halo cells from global memory into shared memory.
    // Left halo.
    if (threadIdx.x == 0) {
        int nx = x - 1;
        if (nx >= 0 && y < grid_dim)
            tile[local_y][0] = input[y * grid_dim + nx];
        else
            tile[local_y][0] = false;
    }
    // Right halo.
    if (threadIdx.x == BLOCK_SIZE - 1) {
        int nx = x + 1;
        if (nx < grid_dim && y < grid_dim)
            tile[local_y][BLOCK_SIZE + 1] = input[y * grid_dim + nx];
        else
            tile[local_y][BLOCK_SIZE + 1] = false;
    }
    // Top halo.
    if (threadIdx.y == 0) {
        int ny = y - 1;
        if (ny >= 0 && x < grid_dim)
            tile[0][local_x] = input[ny * grid_dim + x];
        else
            tile[0][local_x] = false;
    }
    // Bottom halo.
    if (threadIdx.y == BLOCK_SIZE - 1) {
        int ny = y + 1;
        if (ny < grid_dim && x < grid_dim)
            tile[BLOCK_SIZE + 1][local_x] = input[ny * grid_dim + x];
        else
            tile[BLOCK_SIZE + 1][local_x] = false;
    }
    // Top-left corner.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int nx = x - 1;
        int ny = y - 1;
        if (nx >= 0 && ny >= 0)
            tile[0][0] = input[ny * grid_dim + nx];
        else
            tile[0][0] = false;
    }
    // Top-right corner.
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        int nx = x + 1;
        int ny = y - 1;
        if (nx < grid_dim && ny >= 0)
            tile[0][BLOCK_SIZE + 1] = input[ny * grid_dim + nx];
        else
            tile[0][BLOCK_SIZE + 1] = false;
    }
    // Bottom-left corner.
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        int nx = x - 1;
        int ny = y + 1;
        if (nx >= 0 && ny < grid_dim)
            tile[BLOCK_SIZE + 1][0] = input[ny * grid_dim + nx];
        else
            tile[BLOCK_SIZE + 1][0] = false;
    }
    // Bottom-right corner.
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        int nx = x + 1;
        int ny = y + 1;
        if (nx < grid_dim && ny < grid_dim)
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = input[ny * grid_dim + nx];
        else
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = false;
    }

    // Ensure all shared memory loads complete before computation.
    __syncthreads();

    // Only compute if the thread's global indices are within grid bounds.
    if (x < grid_dim && y < grid_dim) {
        // Count alive neighbors using the shared memory tile.
        // Sum the eight neighboring cells (all cells in the 3x3 neighborhood except the center).
        int alive_neighbors = 0;
        alive_neighbors += tile[local_y - 1][local_x - 1]; // Top-left
        alive_neighbors += tile[local_y - 1][local_x];     // Top
        alive_neighbors += tile[local_y - 1][local_x + 1];   // Top-right
        alive_neighbors += tile[local_y][local_x - 1];       // Left
        alive_neighbors += tile[local_y][local_x + 1];       // Right
        alive_neighbors += tile[local_y + 1][local_x - 1];   // Bottom-left
        alive_neighbors += tile[local_y + 1][local_x];       // Bottom
        alive_neighbors += tile[local_y + 1][local_x + 1];   // Bottom-right

        // Apply Conway's Game of Life rules.
        bool current = tile[local_y][local_x];
        bool next_state = false;
        if (current) {
            // An alive cell survives if it has 2 or 3 alive neighbors.
            if (alive_neighbors == 2 || alive_neighbors == 3)
                next_state = true;
        } else {
            // A dead cell becomes alive if it has exactly 3 alive neighbors.
            if (alive_neighbors == 3)
                next_state = true;
        }

        // Write the computed state to the output grid in global memory.
        output[y * grid_dim + x] = next_state;
    }
}

// Host function to execute one step of Conway's Game of Life.
// This function sets up the kernel launch configuration and calls the kernel.
// The input and output grids are assumed to be allocated on the device using cudaMalloc.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Define block dimensions.
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Calculate grid dimensions, ensuring coverage of the entire grid.
    dim3 grid((grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_dimensions + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch the CUDA kernel to compute the next generation.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
}
```

---

Avoid the use of shared or texture memory, as these have proven unnecessary and only add complexity.

---

```c++
#include <cuda_runtime.h>

// Define the block size for kernel execution.
// A block size of 32x32 threads is typically optimal on modern NVIDIA GPUs.
#define BLOCK_SIZE 32

// CUDA kernel implementing one step of Conway's Game of Life without using shared or texture memory.
// Each thread processes a single cell by reading its eight neighbors directly from global memory.
__global__ void game_of_life_kernel(const bool* input, bool* output, int grid_dim) {
    // Calculate the global coordinates for the cell handled by this thread.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the bounds of the grid.
    if (x >= grid_dim || y >= grid_dim)
        return;

    // Count the number of alive neighbors.
    // Neighbors include the eight adjacent cells (horizontal, vertical, and diagonal).
    int alive_neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        int neighbor_y = y + dy;
        // Skip out-of-bound rows.
        if (neighbor_y < 0 || neighbor_y >= grid_dim)
            continue;
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_x = x + dx;
            // Skip out-of-bound columns.
            if (neighbor_x < 0 || neighbor_x >= grid_dim)
                continue;
            // Skip the cell itself.
            if (dx == 0 && dy == 0)
                continue;
            // Each neighbor contributes 1 if alive, 0 if dead.
            alive_neighbors += input[neighbor_y * grid_dim + neighbor_x];
        }
    }

    // Retrieve the current state of the cell.
    bool current = input[y * grid_dim + x];
    bool next_state = false;

    // Apply the rules of Conway's Game of Life:
    // - Any alive cell with 2 or 3 alive neighbors survives.
    // - Any dead cell with exactly 3 alive neighbors becomes alive.
    // - Otherwise, the cell dies or remains dead.
    if (current) {
        next_state = (alive_neighbors == 2 || alive_neighbors == 3);
    } else {
        next_state = (alive_neighbors == 3);
    }

    // Write the computed next state back to global memory.
    output[y * grid_dim + x] = next_state;
}

// Host function to launch the CUDA kernel for one step of Conway's Game of Life.
// This function calculates the grid and block dimensions and launches the kernel.
// The input and output pointers are assumed to point to device memory allocated using cudaMalloc.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    // Define the block dimensions.
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Calculate the grid dimensions to cover the entire grid.
    dim3 grid((grid_dimensions + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Any necessary host-device synchronization is assumed to be handled externally.
    game_of_life_kernel<<<grid, block>>>(input, output, grid_dimensions);
}
```

---

Optimize the previously generated kernel by changing the grid representation to a bitwise encoding, where each cell is encoded as one bit.

--- 

```c++
#include <cuda_runtime.h>
#include <stdint.h>

// In this implementation, the grid is represented in a bitwise encoding where each 32-bit word
// packs 32 cells. The grid is square with dimensions 'grid_dimensions'.
// Each row consists of (grid_dimensions / 32) unsigned ints.
// Bit ordering: bit 0 corresponds to the leftmost cell in the 32-bit word,
// and bit 31 corresponds to the rightmost cell.

// Define block dimensions for kernel execution. Each thread processes one 32-bit word (32 cells).
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// CUDA kernel that computes one generation of Conway's Game of Life on a bit-packed grid.
// Each thread handles one 32-bit word (i.e. 32 cells) from a given row.
__global__ void game_of_life_bit_kernel(const unsigned int* input, unsigned int* output, int grid_dimensions) {
    // Calculate number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // equivalent to grid_dimensions / 32

    // Compute the row index and the word index within that row for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check that we are within grid bounds.
    if (row >= grid_dimensions || word_idx >= words_per_row)
        return;

    // Compute the index into the input/output arrays.
    int index = row * words_per_row + word_idx;

    // Load the 32-bit word for the current row.
    unsigned int cur = input[index];

    // Load adjacent words for the current row to handle horizontal neighbors.
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load words from the row above.
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Load words from the row below.
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dimensions - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Initialize the output word that will store the next state for 32 cells.
    unsigned int out_word = 0;

    // Process each bit in the 32-bit word.
    for (int bit = 0; bit < 32; bit++) {
        // Calculate the absolute column index for this bit.
        int col = (word_idx << 5) + bit;  // word_idx * 32 + bit
        // Ensure we do not process bits beyond grid_dimensions.
        if (col >= grid_dimensions)
            break;

        // Extract the current cell state (0 for dead, 1 for alive).
        int cell = (cur >> bit) & 1;
        int neighbors = 0;

        // Define a lambda to extract a bit from a word with automatic handling of word boundaries.
        // If the requested bit index is -1, it fetches the rightmost bit from the left adjacent word.
        // If the index is 32, it fetches the leftmost bit from the right adjacent word.
        auto getBit = [](unsigned int word, unsigned int left_word, unsigned int right_word, int b) -> int {
            if (b < 0) {
                // For b == -1, get bit 31 from the left word.
                return (left_word >> 31) & 1;
            } else if (b >= 32) {
                // For b == 32, get bit 0 from the right word.
                return (right_word >> 0) & 1;
            } else {
                return (word >> b) & 1;
            }
        };

        // Count neighbors from the top row (if it exists).
        if (row > 0) {
            neighbors += getBit(top, top_left, top_right, bit - 1);
            neighbors += getBit(top, top_left, top_right, bit);
            neighbors += getBit(top, top_left, top_right, bit + 1);
        }

        // Count horizontal neighbors from the current row (excluding the cell itself).
        neighbors += getBit(cur, cur_left, cur_right, bit - 1);
        neighbors += getBit(cur, cur_left, cur_right, bit + 1);

        // Count neighbors from the bottom row (if it exists).
        if (row < grid_dimensions - 1) {
            neighbors += getBit(bot, bot_left, bot_right, bit - 1);
            neighbors += getBit(bot, bot_left, bot_right, bit);
            neighbors += getBit(bot, bot_left, bot_right, bit + 1);
        }

        // Apply Conway's Game of Life rules:
        // - An alive cell survives if it has 2 or 3 alive neighbors.
        // - A dead cell becomes alive if it has exactly 3 alive neighbors.
        int next_cell = 0;
        if (cell) {
            next_cell = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            next_cell = (neighbors == 3) ? 1 : 0;
        }

        // Set the bit in the output word if the next state is alive.
        out_word |= (next_cell << bit);
    }

    // Write the computed output word back to global memory.
    output[index] = out_word;
}

// Host function that launches one simulation step of Conway's Game of Life on a bit-packed grid.
// The grid is represented as an array of unsigned ints (each storing 32 cells).
// 'grid_dimensions' is the width/height of the square grid.
void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // grid_dimensions / 32

    // Set up the execution configuration.
    // The grid dimension in x corresponds to the number of words per row,
    // and in y corresponds to the number of rows.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Any required host-device synchronization is handled externally.
    game_of_life_bit_kernel<<<grid, block>>>(input, output, grid_dimensions);
}

```

---

Instead of performing individual checks for neighbors, utilize the `__popc` intrinsic function to count the number of set bits in a word, which can significantly improve performance.

---

```c++
#include <cuda_runtime.h>
#include <stdint.h>

// In this implementation the grid is stored in a bit-packed format,
// where each unsigned int packs 32 cells (each cell is 1 bit).
// The grid is square with dimensions 'grid_dimensions' (which is a power of 2).
// Each row consists of (grid_dimensions / 32) unsigned ints.
// For each cell, the eight neighbors are fetched from adjacent words as needed,
// and the __popc intrinsic is used to count the number of set bits (alive neighbors)
// in an 8-bit mask built for each cell.

// Define block dimensions; each thread processes one 32-bit word (32 cells)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// CUDA kernel: computes one generation of Conway's Game of Life on a bit-packed grid
// without using shared or texture memory. Each thread handles a 32-bit word.
__global__ void game_of_life_bit_kernel(const unsigned int* input, unsigned int* output, int grid_dim) {
    // Compute number of 32-bit words per row.
    int words_per_row = grid_dim >> 5;  // equivalent to grid_dim / 32

    // Determine the current row and word index.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check grid bounds.
    if (row >= grid_dim || word_idx >= words_per_row)
        return;

    // Calculate the linear index for the current word.
    int index = row * words_per_row + word_idx;

    // Load the current word and its horizontal neighbors.
    unsigned int cur = input[index];
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load the words from the row above (if available).
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Load the words from the row below (if available).
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Prepare the output word.
    unsigned int out_word = 0;

    // Process each bit in the current 32-bit word.
    // Each bit corresponds to one cell.
    for (int bit = 0; bit < 32; bit++) {
        // Compute the absolute column index for the current cell.
        int col = (word_idx << 5) + bit;  // word_idx * 32 + bit
        if (col >= grid_dim)
            break;

        // Build an 8-bit mask for the 8 neighbors of this cell.
        // Bit positions in the mask (0 to 7) are assigned as follows:
        // 0: top-left, 1: top, 2: top-right,
        // 3: left,             4: right,
        // 5: bottom-left, 6: bottom, 7: bottom-right.
        unsigned int neighbor_mask = 0;

        // Top row neighbors.
        if (row > 0) {
            int nb;
            // Top-left neighbor.
            if (bit == 0)
                nb = (top_left >> 31) & 1;
            else
                nb = (top >> (bit - 1)) & 1;
            neighbor_mask |= nb; // assign to bit 0

            // Top neighbor.
            nb = (top >> bit) & 1;
            neighbor_mask |= (nb << 1); // assign to bit 1

            // Top-right neighbor.
            if (bit == 31)
                nb = (top_right >> 0) & 1;
            else
                nb = (top >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 2); // assign to bit 2
        }

        // Current row neighbors.
        {
            int nb;
            // Left neighbor.
            if (bit == 0)
                nb = (cur_left >> 31) & 1;
            else
                nb = (cur >> (bit - 1)) & 1;
            neighbor_mask |= (nb << 3); // assign to bit 3

            // Right neighbor.
            if (bit == 31)
                nb = (cur_right >> 0) & 1;
            else
                nb = (cur >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 4); // assign to bit 4
        }

        // Bottom row neighbors.
        if (row < grid_dim - 1) {
            int nb;
            // Bottom-left neighbor.
            if (bit == 0)
                nb = (bot_left >> 31) & 1;
            else
                nb = (bot >> (bit - 1)) & 1;
            neighbor_mask |= (nb << 5); // assign to bit 5

            // Bottom neighbor.
            nb = (bot >> bit) & 1;
            neighbor_mask |= (nb << 6); // assign to bit 6

            // Bottom-right neighbor.
            if (bit == 31)
                nb = (bot_right >> 0) & 1;
            else
                nb = (bot >> (bit + 1)) & 1;
            neighbor_mask |= (nb << 7); // assign to bit 7
        }

        // Use the __popc intrinsic to count the number of alive neighbors.
        int count = __popc(neighbor_mask);

        // Retrieve the current cell's state.
        int cell = (cur >> bit) & 1;
        int next_state = 0;

        // Apply Conway's Game of Life rules:
        // - An alive cell survives if it has 2 or 3 alive neighbors.
        // - A dead cell becomes alive if it has exactly 3 alive neighbors.
        if (cell)
            next_state = (count == 2 || count == 3) ? 1 : 0;
        else
            next_state = (count == 3) ? 1 : 0;

        // Set the corresponding bit in the output word.
        out_word |= (next_state << bit);
    }

    // Write the computed 32-bit word to the output grid.
    output[index] = out_word;
}

// Host function that launches one simulation step of Conway's Game of Life
// on a bit-packed grid. The grid is represented as an array of unsigned ints,
// where each bit represents a cell.
void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5;  // grid_dimensions / 32

    // Set up the execution configuration: grid dimensions in x correspond to the
    // number of words per row, and in y correspond to the number of rows.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled externally.
    game_of_life_bit_kernel<<<grid, block>>>(input, output, grid_dimensions);
}

```

---

Computing one cell at a time is computationally expensive. Can you devise a method to process multiple cells simultaneously to improve performance? Note that we are using a row-based bitwise encoding of the grid.

---

```c++
#include <cuda_runtime.h>
#include <stdint.h>

// In this optimized version, the grid is stored in a bit‐packed row‐based encoding,
// where each 32‐bit unsigned int packs 32 cells (one bit per cell).
// Each row consists of (grid_dimensions / 32) unsigned ints.
// To improve performance, each thread processes an entire 32‐bit word,
// but instead of iterating over each cell (32 iterations),
// the kernel processes 2 cells at a time (16 iterations per word).
// For each 2‐cell group, a 4‐bit “window” is extracted from each of the three rows
// (top, current, and bottom). This 4‐bit window covers the two cells plus one extra
// neighbor on each side. For each cell within the group, its 3×3 neighborhood
// is contained in 9 bits from the corresponding window (the two overlapping windows
// for the two cells are shifted by one bit). The __popc intrinsic is then used to
// count the alive neighbors in the 9‐bit mask.
// The new state for each cell is computed according to the standard rules:
//   - An alive cell survives if it has 2 or 3 alive neighbors.
//   - A dead cell becomes alive if it has exactly 3 alive neighbors.
// All cells outside the grid are considered dead.

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Inline helper to extract a 4-bit window from a row, given the current word and its horizontal neighbors.
// 'word'      : the 32-bit word for the row.
// 'left_word' : the word immediately to the left (or 0 if none).
// 'right_word': the word immediately to the right (or 0 if none).
// 'base'      : the starting bit position in 'word' for the 2-cell group (must be in [0, 31]).
// The desired window covers bits [base-1, base+2] (4 bits total), taking bits from adjacent words if needed.
__device__ inline unsigned int extract_window(unsigned int word, unsigned int left_word, unsigned int right_word, int base) {
    unsigned int window = 0;
    // Bit position -1 (left neighbor of first cell in the group)
    if (base == 0) {
        // Use the least-significant bit of the left word’s most-significant position.
        window |= ((left_word >> 31) & 1U) << 0;
    } else {
        window |= ((word >> (base - 1)) & 1U) << 0;
    }
    // Bits for positions corresponding to the two cells and the right neighbor of the group.
    // These come from the current word if they are in bounds.
    // Bit at position 0 in the group window corresponds to cell at index 'base'
    if (base < 32)
        window |= ((word >> base) & 1U) << 1;
    else
        window |= 0U << 1;
    // Next cell in the group.
    if (base + 1 < 32)
        window |= ((word >> (base + 1)) & 1U) << 2;
    else
        window |= 0U << 2;
    // Bit for the neighbor to the right of the group (cell at index base+2).
    if (base + 2 < 32)
        window |= ((word >> (base + 2)) & 1U) << 3;
    else {
        // If out-of-bound, take from right_word (bit 0)
        window |= ((right_word >> 0) & 1U) << 3;
    }
    return window; // window is in range [0,15]
}

// CUDA kernel that computes one generation of Conway's Game of Life on a bit‐packed grid.
// Each thread processes one 32‐bit word (32 cells) of the grid.
// Instead of processing one cell at a time, the kernel processes groups of 2 cells simultaneously
// (16 groups per word), using 9‐bit neighborhoods and __popc to count alive neighbors.
__global__ void game_of_life_bit_2cells_kernel(const unsigned int* input, unsigned int* output, int grid_dim) {
    // Number of 32-bit words per row.
    int words_per_row = grid_dim >> 5;  // grid_dim / 32

    // Compute the row index and the word index within that row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;

    // Calculate the index for the current 32-bit word.
    int index = row * words_per_row + word_idx;

    // Load the current word and its horizontal neighbors.
    unsigned int cur = input[index];
    unsigned int cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    unsigned int cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;

    // Load the corresponding words from the row above.
    unsigned int top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    // Load the corresponding words from the row below.
    unsigned int bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }

    // Prepare the output word.
    unsigned int out_word = 0;

    // Process the 32 cells in groups of 2 cells (16 groups per word).
    // 'base' is the starting bit index for the 2-cell group within the 32-bit word.
    for (int group = 0; group < 16; group++) {
        int base = group * 2;

        // Extract 4-bit windows from top, current, and bottom rows.
        unsigned int top_window   = extract_window(top, top_left, top_right, base);
        unsigned int cur_window   = extract_window(cur, cur_left, cur_right, base);
        unsigned int bot_window   = extract_window(bot, bot_left, bot_right, base);

        // For the two cells in the group, their 3x3 neighborhoods are taken from overlapping bits:
        // For cell 0 in the group, the neighborhood uses bits [0,1,2] from each window.
        // For cell 1 in the group, the neighborhood uses bits [1,2,3] from each window.
        // Build 9-bit masks for each cell.
        unsigned int mask0 = 0;
        unsigned int mask1 = 0;

        // Top row contributions.
        mask0 |= (top_window & 0x7) << 0;         // bits 0..2 for cell0
        mask1 |= ((top_window >> 1) & 0x7) << 0;    // bits 1..3 for cell1
        // Current row contributions (skip the center cell).
        // For cell0, take bit 0 and bit 2.
        mask0 |= ((cur_window >> 0) & 1) << 3;
        mask0 |= ((cur_window >> 2) & 1) << 4;
        // For cell1, take bit 1 and bit 3.
        mask1 |= ((cur_window >> 1) & 1) << 3;
        mask1 |= ((cur_window >> 3) & 1) << 4;
        // Bottom row contributions.
        mask0 |= (bot_window & 0x7) << 5;         // bits 0..2 for cell0
        mask1 |= ((bot_window >> 1) & 0x7) << 5;    // bits 1..3 for cell1

        // Count alive neighbors using __popc on the 9-bit masks.
        // __popc operates on 32-bit integers, so the 9-bit mask is in the lower bits.
        int count0 = __popc(mask0);
        int count1 = __popc(mask1);

        // The current state of the cells is in cur_window:
        // For cell0: bit 1, for cell1: bit 2.
        int cell0 = (cur_window >> 1) & 1;
        int cell1 = (cur_window >> 2) & 1;

        // Compute next state using Game of Life rules.
        // Alive cell survives if neighbor count is 2 or 3.
        // Dead cell becomes alive if neighbor count is exactly 3.
        int next0 = (cell0 && (count0 == 2 || count0 == 3)) || (!cell0 && (count0 == 3));
        int next1 = (cell1 && (count1 == 2 || count1 == 3)) || (!cell1 && (count1 == 3));

        // Set the corresponding bits in the output word.
        out_word |= (next0 << base) | (next1 << (base + 1));
    }

    // Write the computed 32-bit word to global memory.
    output[index] = out_word;
}

// Host function to execute one generation of Conway's Game of Life on a bit‐packed grid,
// using the optimized kernel that processes 2 cells simultaneously per thread.
// 'input' and 'output' are pointers to device memory (allocated via cudaMalloc),
// and 'grid_dimensions' is the width/height of the square grid.
void run_game_of_life(const unsigned int* input, unsigned int* output, int grid_dimensions) {
    // Calculate the number of 32-bit words per row.
    int words_per_row = grid_dimensions >> 5; // grid_dimensions / 32

    // Define block dimensions.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    // Calculate grid dimensions (number of blocks) to cover the entire grid.
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);

    // Launch the kernel. Host-device synchronization is assumed to be handled externally.
    game_of_life_bit_2cells_kernel<<<grid, block>>>(input, output, grid_dimensions);
}

```

---

Your solution is correct, but could you try implementing a different approach? To efficiently compute the neighbor count, apply full adder logic to add the bits from adjacent neighbor words concurrently. That is, for three neighbor words, compute the sum as the XOR of the three inputs and the carry as the majority function, and then combine these results across groups of neighbors to obtain the total count for each bit position in parallel. This allows you to process all cells in the word simultaneously using simple bitwise operations.

---

```c++
#include <cuda_runtime.h>
#include <stdint.h>

// This implementation uses a row‐based bitwise encoding with 16‐bit words,
// where each word packs 16 cells (one bit per cell). To compute the neighbor
// count for each cell, we apply full adder logic in parallel on the bit
// columns. Each neighbor (from one of the 8 directions) is first converted into
// a “digit” whose value is either 0 or 1. These 1‐bit digits are then added
// using bit‐sliced full adder logic so that each cell’s neighbor sum (ranging from 0 to 8)
// is represented by a 4‐bit number (stored in 4 separate bitmasks, one per bit).
//
// The grid accesses for neighbor cells account for horizontal boundaries by
// fetching bits from adjacent words when needed. Vertical boundaries are handled
// by checking the row index (cells outside the grid are considered dead).
//
// After computing the neighbor count for each cell, the Game of Life rules are applied:
//   - For an alive cell (bit = 1): it survives if the neighbor count is 2 or 3.
//   - For a dead cell (bit = 0): it becomes alive if the neighbor count is exactly 3.
//
// The final result is repacked into a 16‐bit word for each block of 16 cells.


// Structure to hold a “digit” in 4 bits per cell.
// For each cell (each bit position in a 16‐bit word), the neighbor count is represented as:
//   value = d0 + 2*d1 + 4*d2 + 8*d3, where each dX is a bitmask (16 bits) holding the Xth bit
// for each cell independently.
struct Digit {
    uint16_t r0;
    uint16_t r1;
    uint16_t r2;
    uint16_t r3;
};

// Convert a 1‐bit value (0 or 1 for each cell) into a 4‐bit digit representation.
__device__ inline Digit make_digit(uint16_t bit) {
    Digit d;
    d.r0 = bit;  // the value is in the least‐significant bit position
    d.r1 = 0;
    d.r2 = 0;
    d.r3 = 0;
    return d;
}

// Add two digits using full adder logic per cell (i.e. per bit column).
// Each addition is performed independently without propagating carry across cell boundaries.
__device__ inline Digit add_digit(const Digit &A, const Digit &B) {
    Digit R;
    // Add the LSBs.
    uint16_t s0 = A.r0 ^ B.r0;
    uint16_t c0 = A.r0 & B.r0;
    // Add the next bits along with the carry from the previous column.
    uint16_t s1 = A.r1 ^ B.r1 ^ c0;
    uint16_t c1 = (A.r1 & B.r1) | (A.r1 & c0) | (B.r1 & c0);
    // Continue for the third bit.
    uint16_t s2 = A.r2 ^ B.r2 ^ c1;
    uint16_t c2 = (A.r2 & B.r2) | (A.r2 & c1) | (B.r2 & c1);
    // And the fourth bit.
    uint16_t s3 = A.r3 ^ B.r3 ^ c2;
    // The final carry is ignored (maximum sum is 8, fitting in 4 bits).
    R.r0 = s0;
    R.r1 = s1;
    R.r2 = s2;
    R.r3 = s3;
    return R;
}

// Add eight digits using pairwise full-adder additions.
__device__ inline Digit add_multiple(Digit a, Digit b, Digit c, Digit d,
                                      Digit e, Digit f, Digit g, Digit h) {
    Digit t0 = add_digit(a, b);
    Digit t1 = add_digit(c, d);
    Digit t2 = add_digit(e, f);
    Digit t3 = add_digit(g, h);
    Digit u0 = add_digit(t0, t1);
    Digit u1 = add_digit(t2, t3);
    Digit sum = add_digit(u0, u1);
    return sum;
}

// Extract the 4-bit neighbor count for a given cell (bit position) from the digit.
__device__ inline int extract_digit(const Digit &d, int bit_index) {
    int b0 = (d.r0 >> bit_index) & 1;
    int b1 = (d.r1 >> bit_index) & 1;
    int b2 = (d.r2 >> bit_index) & 1;
    int b3 = (d.r3 >> bit_index) & 1;
    return b0 + (b1 << 1) + (b2 << 2) + (b3 << 3);
}

// Block dimensions for kernel execution.
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

// CUDA kernel that computes one generation of Conway's Game of Life using full adder logic
// to sum neighbor bits concurrently. The grid is stored in 16-bit words (16 cells per word).
__global__ void game_of_life_bit_adder_kernel(const uint16_t* input, uint16_t* output, int grid_dim) {
    // Compute the number of 16-bit words per row.
    int words_per_row = grid_dim / 16;
    
    // Determine the row and word index for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;
    
    // Linear index into the grid.
    int index = row * words_per_row + word_idx;
    
    // Load the current word (16 cells) for this row.
    uint16_t cur = input[index];
    
    // Fetch horizontal neighbors from the same row.
    uint16_t cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    uint16_t cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;
    
    // Fetch words from the row above.
    uint16_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Fetch words from the row below.
    uint16_t bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Compute neighbor bit vectors for each of the 8 directions.
    // For horizontal shifts, a left shift (<< 1) gives the left neighbor bits,
    // and a right shift (>> 1) gives the right neighbor bits.
    // Boundary bits are fixed by fetching the appropriate bit from adjacent words.
    uint16_t n_top_left  = (word_idx > 0) ? ((top_left << 1) | ((top_left >> 15) & 1)) : (top << 1);
    uint16_t n_top_center = top;
    uint16_t n_top_right = (word_idx < words_per_row - 1) ? ((top >> 1) | ((top_right & 1) << 15)) : (top >> 1);
    
    uint16_t n_left  = (word_idx > 0) ? ((cur_left << 1) | ((cur_left >> 15) & 1)) : (cur << 1);
    uint16_t n_right = (word_idx < words_per_row - 1) ? ((cur >> 1) | ((cur_right & 1) << 15)) : (cur >> 1);
    
    uint16_t n_bot_left  = (word_idx > 0) ? ((bot_left << 1) | ((bot_left >> 15) & 1)) : (bot << 1);
    uint16_t n_bot_center = bot;
    uint16_t n_bot_right = (word_idx < words_per_row - 1) ? ((bot >> 1) | ((bot_right & 1) << 15)) : (bot >> 1);
    
    // Convert each neighbor contribution (16-bit, one bit per cell) into a 4-bit digit.
    Digit d_top_left    = make_digit(n_top_left);
    Digit d_top_center  = make_digit(n_top_center);
    Digit d_top_right   = make_digit(n_top_right);
    Digit d_left        = make_digit(n_left);
    Digit d_right       = make_digit(n_right);
    Digit d_bot_left    = make_digit(n_bot_left);
    Digit d_bot_center  = make_digit(n_bot_center);
    Digit d_bot_right   = make_digit(n_bot_right);
    
    // Sum all 8 neighbor contributions using full adder logic.
    Digit neighbor_sum = add_multiple(d_top_left, d_top_center, d_top_right,
                                      d_left, d_right, d_bot_left, d_bot_center, d_bot_right);
    
    // Determine the new state for each cell according to the Game of Life rules.
    // For each cell (each bit position in the 16-bit word), extract the neighbor count
    // (a 4-bit number) and apply:
    //   - If the cell is alive: survive if count == 2 or count == 3.
    //   - If the cell is dead: become alive if count == 3.
    uint16_t new_word = 0;
    for (int i = 0; i < 16; i++) {
        int count = extract_digit(neighbor_sum, i);
        int cell = (cur >> i) & 1;
        int new_state = 0;
        if (cell)
            new_state = (count == 2 || count == 3) ? 1 : 0;
        else
            new_state = (count == 3) ? 1 : 0;
        new_word |= (new_state << i);
    }
    
    // Write the computed new state for the 16 cells back to global memory.
    output[index] = new_word;
}

// Host function to execute one simulation step of Conway's Game of Life using the
// full adder (bitwise) approach on a bit-packed grid represented with 16-bit words.
// 'grid_dimensions' is the number of cells per row (must be divisible by 16).
void run_game_of_life(const uint16_t* input, uint16_t* output, int grid_dimensions) {
    // Calculate the number of 16-bit words per row.
    int words_per_row = grid_dimensions / 16;
    
    // Define block dimensions.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);
    
    // Launch the kernel. Any necessary host-device synchronization is assumed to be handled externally.
    game_of_life_bit_adder_kernel<<<grid, block>>>(input, output, grid_dimensions);
}

```

---

The code compiles and runs, but the output is incorrect.

---

```c++
#include <cuda_runtime.h>
#include <stdint.h>

// In this implementation the grid is stored in a 16‐bit word per block of 16 cells.
// Bit ordering: bit 0 is the leftmost cell, bit 15 is the rightmost cell.
// For each cell, its 8 neighbors (top‐left, top, top‐right, left, right, bottom‐left, bottom, bottom‐right)
// are extracted from the corresponding neighbor words with proper shifts (using 0 for out‐of‐bounds).
// To compute the neighbor count per cell in parallel, we represent each one‐bit neighbor
// as a “digit” (in a bit‐sliced form) and sum eight such digits using full‐adder logic.
// The final 4‐bit result per cell (stored across 4 bitmasks) gives the neighbor count (0–8).
// Game of Life rules are then applied to produce the next state.

struct Digit {
    uint16_t r0;
    uint16_t r1;
    uint16_t r2;
    uint16_t r3;
};

// Convert a 16-bit mask (each bit 0 or 1) into a digit (bit-sliced 4-bit number per cell).
__device__ inline Digit make_digit(uint16_t bit) {
    Digit d;
    d.r0 = bit;  // the least-significant slice holds the value (0 or 1)
    d.r1 = 0;
    d.r2 = 0;
    d.r3 = 0;
    return d;
}

// Add two digits using bitwise full-adder logic per cell (each cell is one bit in the 16-bit mask).
// Since each digit is only 0 or 1 initially (and sums up to 8), no cross-cell carry occurs.
__device__ inline Digit add_digit(const Digit &A, const Digit &B) {
    Digit R;
    uint16_t s0 = A.r0 ^ B.r0;          // sum of LSBs
    uint16_t c0 = A.r0 & B.r0;          // carry from LSBs
    uint16_t s1 = A.r1 ^ B.r1 ^ c0;
    uint16_t c1 = (A.r1 & B.r1) | (A.r1 & c0) | (B.r1 & c0);
    uint16_t s2 = A.r2 ^ B.r2 ^ c1;
    uint16_t c2 = (A.r2 & B.r2) | (A.r2 & c1) | (B.r2 & c1);
    uint16_t s3 = A.r3 ^ B.r3 ^ c2;
    // Final carry is ignored (maximum sum is 8, fitting in 4 bits).
    R.r0 = s0;
    R.r1 = s1;
    R.r2 = s2;
    R.r3 = s3;
    return R;
}

// Sum eight digits by pairwise additions.
__device__ inline Digit add_multiple(Digit a, Digit b, Digit c, Digit d,
                                      Digit e, Digit f, Digit g, Digit h) {
    Digit t0 = add_digit(a, b);
    Digit t1 = add_digit(c, d);
    Digit t2 = add_digit(e, f);
    Digit t3 = add_digit(g, h);
    Digit u0 = add_digit(t0, t1);
    Digit u1 = add_digit(t2, t3);
    Digit sum = add_digit(u0, u1);
    return sum;
}

// Extract the 4-bit neighbor count for a given cell (bit position) from the digit.
__device__ inline int extract_digit(const Digit &d, int bit_index) {
    int b0 = (d.r0 >> bit_index) & 1;
    int b1 = (d.r1 >> bit_index) & 1;
    int b2 = (d.r2 >> bit_index) & 1;
    int b3 = (d.r3 >> bit_index) & 1;
    return b0 + (b1 << 1) + (b2 << 2) + (b3 << 3);
}

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

// CUDA kernel implementing one Game of Life generation using full-adder logic on a 16-bit packed grid.
// Each thread processes one 16-bit word representing 16 cells.
__global__ void game_of_life_bit_adder_kernel(const uint16_t* input, uint16_t* output, int grid_dim) {
    // Compute the number of 16-bit words per row.
    int words_per_row = grid_dim / 16;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= grid_dim || word_idx >= words_per_row)
        return;
    
    int index = row * words_per_row + word_idx;
    uint16_t cur = input[index];
    
    // Fetch horizontal neighbors from the current row.
    uint16_t cur_left  = (word_idx > 0) ? input[row * words_per_row + (word_idx - 1)] : 0;
    uint16_t cur_right = (word_idx < words_per_row - 1) ? input[row * words_per_row + (word_idx + 1)] : 0;
    
    // Fetch vertical neighbor rows.
    uint16_t top = 0, top_left = 0, top_right = 0;
    if (row > 0) {
        int top_index = (row - 1) * words_per_row + word_idx;
        top = input[top_index];
        top_left  = (word_idx > 0) ? input[(row - 1) * words_per_row + (word_idx - 1)] : 0;
        top_right = (word_idx < words_per_row - 1) ? input[(row - 1) * words_per_row + (word_idx + 1)] : 0;
    }
    uint16_t bot = 0, bot_left = 0, bot_right = 0;
    if (row < grid_dim - 1) {
        int bot_index = (row + 1) * words_per_row + word_idx;
        bot = input[bot_index];
        bot_left  = (word_idx > 0) ? input[(row + 1) * words_per_row + (word_idx - 1)] : 0;
        bot_right = (word_idx < words_per_row - 1) ? input[(row + 1) * words_per_row + (word_idx + 1)] : 0;
    }
    
    // Compute neighbor masks with corrected shifts.
    // For left neighbors: each cell's left neighbor is in the same row, one position to the left.
    // Compute by shifting the current word left by 1; for cell 0, use previous word's rightmost bit if available.
    uint16_t n_left = cur << 1;
    if (word_idx > 0) {
        uint16_t extra = (cur_left >> 15) & 1;
        n_left = (n_left & 0xFFFE) | extra;  // clear bit 0 and set it to extra
    }
    // For right neighbors: shift the current word right by 1; for cell 15, use next word's leftmost bit if available.
    uint16_t n_right = cur >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = cur_right & 1;
        n_right = (n_right & 0x7FFF) | (extra << 15);  // clear bit 15 and set it to extra
    }
    
    // Top neighbors (no horizontal shift).
    uint16_t n_top_center = top;
    // Top-left: shift top row left by 1; for cell 0, use top_left's rightmost bit.
    uint16_t n_top_left = top << 1;
    if (word_idx > 0) {
        uint16_t extra = (top_left >> 15) & 1;
        n_top_left = (n_top_left & 0xFFFE) | extra;
    }
    // Top-right: shift top row right by 1; for cell 15, use top_right's leftmost bit.
    uint16_t n_top_right = top >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = top_right & 1;
        n_top_right = (n_top_right & 0x7FFF) | (extra << 15);
    }
    
    // Bottom neighbors.
    uint16_t n_bot_center = bot;
    uint16_t n_bot_left = bot << 1;
    if (word_idx > 0) {
        uint16_t extra = (bot_left >> 15) & 1;
        n_bot_left = (n_bot_left & 0xFFFE) | extra;
    }
    uint16_t n_bot_right = bot >> 1;
    if (word_idx < words_per_row - 1) {
        uint16_t extra = bot_right & 1;
        n_bot_right = (n_bot_right & 0x7FFF) | (extra << 15);
    }
    
    // Convert each neighbor mask into a digit.
    Digit d_top_left    = make_digit(n_top_left);
    Digit d_top_center  = make_digit(n_top_center);
    Digit d_top_right   = make_digit(n_top_right);
    Digit d_left        = make_digit(n_left);
    Digit d_right       = make_digit(n_right);
    Digit d_bot_left    = make_digit(n_bot_left);
    Digit d_bot_center  = make_digit(n_bot_center);
    Digit d_bot_right   = make_digit(n_bot_right);
    
    // Sum all 8 neighbor contributions using full-adder logic.
    Digit neighbor_sum = add_multiple(d_top_left, d_top_center, d_top_right,
                                      d_left, d_right, d_bot_left, d_bot_center, d_bot_right);
    
    // Compute next state for each cell (each bit in the 16-bit word) based on Game of Life rules.
    uint16_t new_word = 0;
    for (int i = 0; i < 16; i++) {
        int count = extract_digit(neighbor_sum, i);
        int cell = (cur >> i) & 1;
        int new_state = 0;
        if (cell)
            new_state = (count == 2 || count == 3) ? 1 : 0;
        else
            new_state = (count == 3) ? 1 : 0;
        new_word |= (new_state << i);
    }
    
    output[index] = new_word;
}

// Host function to launch one generation of Conway's Game of Life using the full-adder method.
// 'input' and 'output' are pointers to device memory (allocated via cudaMalloc).
// 'grid_dimensions' is the number of cells per row (must be divisible by 16).
void run_game_of_life(const uint16_t* input, uint16_t* output, int grid_dimensions) {
    int words_per_row = grid_dimensions / 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((words_per_row + block.x - 1) / block.x,
              (grid_dimensions + block.y - 1) / block.y);
    game_of_life_bit_adder_kernel<<<grid, block>>>(input, output, grid_dimensions);
}

```
