#include "gol.cuh"
#include <cstdint>
#include <cuda_runtime.h>

__device__ inline uint64_t get_cell(const uint64_t* grid, int x, int y, int grid_dimensions) {
    if (x < 0 || x >= grid_dimensions || y < 0 || y >= grid_dimensions) {
        return 0;
    }
    
    constexpr int cells_per_uint64 = 64;
    int index = y * grid_dimensions + x;
    int word_index = index / cells_per_uint64;
    int bit_index = index % cells_per_uint64;
    
    return (grid[word_index] >> bit_index) & 1ULL;
}

__global__ void gol_kernel(const uint64_t* input, uint64_t* output, int grid_dimensions) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= grid_dimensions || y >= grid_dimensions) {
        return;
    }

    int neighbors = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            neighbors += get_cell(input, x + dx, y + dy, grid_dimensions);
        }
    }
    
    uint64_t current_state = get_cell(input, x, y, grid_dimensions);
    uint64_t new_state = (neighbors == 3 || (current_state && neighbors == 2)) ? 1 : 0;
    
    if (new_state) {
        int index = y * grid_dimensions + x;
        int word_index = index / 64;
        int bit_index = index % 64;
        
        unsigned int* output32 = reinterpret_cast<unsigned int*>(output);
        int which_uint32 = (bit_index >= 32) ? 1 : 0;
        int bit_pos_in_uint32 = bit_index % 32;
        
        atomicOr(&output32[word_index * 2 + which_uint32], 1U << bit_pos_in_uint32);
    }
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    int total_cells = grid_dimensions * grid_dimensions;
    int num_words = (total_cells + 63) / 64;
    
    cudaMemset(output, 0, sizeof(uint64_t) * num_words);

    dim3 blockSize(32, 8);
    dim3 gridSize((grid_dimensions + blockSize.x - 1) / blockSize.x, 
                  (grid_dimensions + blockSize.y - 1) / blockSize.y);

    gol_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// MEMORY_LAYOUT: ROWS
