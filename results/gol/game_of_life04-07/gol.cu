#include <cuda_runtime.h>
#include <cstdint>

// In this implementation we assume that each grid row is represented by (grid_dimensions/64) uint64_t words.
// Each word packs 64 cells. We adopt the following convention for a 64‐bit word:
//   • Logical cell indices within a word: 0 ... 63, where index 0 is the left‐most cell and 63 is the right‐most.
//   • In the C/C++ bit–representation of the word, bit position (63 - i) holds the state of the cell with logical index i.
//     (That is, the MSB holds cell0, and the LSB holds cell63.)
//
// For each uint64_t word, one CUDA thread computes the next generation state for the 64 cells
// contained in that word. To determine the 8–cell neighborhood of each cell, we need to fetch the
// adjacent words from the same row (neighbors for right/left internal cells) and from the rows above and below.
// For cells on the boundaries of a word the neighbor cell may belong to an adjacent word; if that adjacent word
// is not available (i.e. at the grid border) we assume those cells are dead.
// 
// The Game of Life rule is applied as follows for each cell:
//   • If a cell is alive and it has 2 or 3 live neighbors, it remains live; otherwise it dies.
//   • If a cell is dead and it has exactly 3 live neighbors, it becomes live; otherwise it remains dead.
//
// To maximize performance we use simple bit–mask extractions and loop over the 64 cells in each word (the loop is unrolled).
// Although it may be possible to compute the neighbor sum in a fully bit–parallel manner, handling the boundary cells
// (bit 0 and bit 63) requires “rotated” operations across adjacent words; this implementation avoids extra complexity
// by iterating 64 times in each thread. The __popc intrinsic was considered for fast bit–counting but its use is best suited
// to whole–word population counts rather than per–cell neighbor sums, so here we compute per–cell counts by combining eight 1–bit accesses.

 
// __device__ inline helper to extract the cell value from a word at logical cell index 'i' (0 = leftmost, 63 = rightmost).
// The cell is stored in bit position (63 - i).
__device__ __forceinline__ int get_cell(uint64_t word, int i) {
    return (int)((word >> (63 - i)) & 1ULL);
}

// The CUDA kernel computes one Game of Life step for one uint64_t word (64 cells)
// in the bit-packed grid. Grid dimensions in terms of cells and words are communicated by grid_dim (cells per side)
// and words_per_row ( = grid_dim / 64).
__global__ void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint64_t* __restrict__ output,
                                    int grid_dim,
                                    int words_per_row)
{
    // Each thread handles one word.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = grid_dim * words_per_row;
    if (idx >= total_words)
        return;
        
    // Determine 2D word coordinates.
    int row = idx / words_per_row;
    int col = idx % words_per_row;
    
    // Load current word and its 8 neighbors (if they exist, else use 0).
    // For a given cell, neighbors come from the same row (left/right), 
    // the row above (top-left, top, top-right) and the row below (bottom-left, bottom, bottom-right).
    std::uint64_t current   = input[row * words_per_row + col];
    std::uint64_t left      = (col > 0)             ? input[row * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t right     = (col < words_per_row-1) ? input[row * words_per_row + (col + 1)] : 0ULL;
    
    std::uint64_t top       = (row > 0)             ? input[(row - 1) * words_per_row + col] : 0ULL;
    std::uint64_t bottom    = (row < grid_dim-1)      ? input[(row + 1) * words_per_row + col] : 0ULL;
    
    std::uint64_t top_left     = ((row > 0) && (col > 0))             ? input[(row - 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t top_right    = ((row > 0) && (col < words_per_row-1)) ? input[(row - 1) * words_per_row + (col + 1)] : 0ULL;
    std::uint64_t bottom_left  = ((row < grid_dim-1) && (col > 0))      ? input[(row + 1) * words_per_row + (col - 1)] : 0ULL;
    std::uint64_t bottom_right = ((row < grid_dim-1) && (col < words_per_row-1)) ? input[(row + 1) * words_per_row + (col + 1)] : 0ULL;
    
    std::uint64_t out_word = 0ULL;
    
    // Process each of the 64 bits (cells) in this word.
    // Loop is unrolled to help the compiler optimize.
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        int live_neighbors = 0;
        
        // --- Same row neighbors ---
        // Left neighbor in same row:
        if (i > 0) {
            // Neighbor within the same word: cell (i-1) from 'current'.
            live_neighbors += get_cell(current, i - 1);
        } else {
            // i == 0 (leftmost cell); neighbor comes from the previous word in same row, if available.
            // In the adjacent (left) word, the rightmost cell (cell index 63) is the neighbor.
            if (col > 0)
                live_neighbors += get_cell(left, 63);
        }
        // Right neighbor in same row:
        if (i < 63) {
            live_neighbors += get_cell(current, i + 1);
        } else {
            // i == 63 (rightmost cell); neighbor from next word in same row (cell index 0).
            if (col < words_per_row - 1)
                live_neighbors += get_cell(right, 0);
        }
        
        // --- Top row neighbors (if row exists) ---
        if (row > 0) {
            // Top-left neighbor:
            if (col > 0) {
                if (i > 0)
                    live_neighbors += get_cell(top_left, i - 1);
                else
                    live_neighbors += get_cell(top_left, 63);
            }
            // Top neighbor (same column): always aligned.
            live_neighbors += get_cell(top, i);
            // Top-right neighbor:
            if (col < words_per_row - 1) {
                if (i < 63)
                    live_neighbors += get_cell(top_right, i + 1);
                else
                    live_neighbors += get_cell(top_right, 0);
            }
        }
        
        // --- Bottom row neighbors (if row exists) ---
        if (row < grid_dim - 1) {
            // Bottom-left neighbor:
            if (col > 0) {
                if (i > 0)
                    live_neighbors += get_cell(bottom_left, i - 1);
                else
                    live_neighbors += get_cell(bottom_left, 63);
            }
            // Bottom neighbor:
            live_neighbors += get_cell(bottom, i);
            // Bottom-right neighbor:
            if (col < words_per_row - 1) {
                if (i < 63)
                    live_neighbors += get_cell(bottom_right, i + 1);
                else
                    live_neighbors += get_cell(bottom_right, 0);
            }
        }
        
        // Retrieve the current cell state.
        int cell_state = get_cell(current, i);
        // Apply Game of Life rules:
        // A cell becomes live if it has exactly 3 neighbors or if it is live and has exactly 2 neighbors.
        int new_state = ((live_neighbors == 3) || (cell_state && (live_neighbors == 2))) ? 1 : 0;
        
        // Set the computed bit in the output word at the proper bit position.
        out_word |= ((std::uint64_t)new_state << (63 - i));
    }
    
    // Write the computed 64–cell word to the output grid.
    output[row * words_per_row + col] = out_word;
}
  
// Host–side function performing one generation of Conway's Game of Life.
// 'input' and 'output' point to bit–packed grids pre–allocated on the GPU using cudaMalloc.
// 'grid_dimensions' is the width/height of the square grid (in cells) and is always a power of 2.
/// @FIXED: extern "C" void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions)
{
    // Compute the number of 64–bit words per row.
    int words_per_row = grid_dimensions >> 6; // equivalent to grid_dimensions/64

    // Total number of words in the grid.
    int total_words = grid_dimensions * words_per_row;
    
    // Launch configuration: each thread works on one word.
    int threadsPerBlock = 256;
    int blocks = (total_words + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the CUDA kernel.
    game_of_life_kernel<<<blocks, threadsPerBlock>>>(input, output, grid_dimensions, words_per_row);
    
    // Note: Host-device synchronization (e.g. cudaDeviceSynchronize()) is assumed to be handled by the caller.
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
