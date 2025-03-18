#include "gol.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define INDEX(x, y, dim) ((y) * (dim) + (x))

__global__ void gol_kernel(const bool* input, bool* output, int dim) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim || y >= dim) return;

    int count = 0;
    
    int x_start = (x == 0) ? 0 : -1;
    int x_end = (x == dim - 1) ? 0 : 1;
    int y_start = (y == 0) ? 0 : -1;
    int y_end = (y == dim - 1) ? 0 : 1;
    
    for (int dx = x_start; dx <= x_end; dx++) {
        for (int dy = y_start; dy <= y_end; dy++) {
            
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            count += input[INDEX(nx, ny, dim)];
        }
    }

    int current = input[INDEX(x, y, dim)];
    int new_state = (count == 3 || (current && count == 2)) ? 1 : 0;

    output[INDEX(x, y, dim)] = new_state;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    dim3 blockSize(32, 32);
    dim3 gridSize((grid_dimensions) / 32, (grid_dimensions) / 32);

    gol_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// MEMORY_LAYOUT: BOOLS
