#include "gol.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define INDEX(x, y, dim) ((y) * (dim) + (x))
#define WORD_TYPE uint64_t

__device__ WORD_TYPE load_word(const WORD_TYPE* input, int x_word, int y_word, int x_dim, int y_dim) {
    if (x_word < 0 || x_word >= x_dim || y_word < 0 || y_word >= y_dim) {
        return 0;
    }
    return input[y_word * x_dim + x_word];
}

__global__ void gol_kernel(const WORD_TYPE* input, WORD_TYPE* output, int dim) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x_dim = dim / (sizeof(uint64_t) * 8);
    
    if (x >= x_dim || y >= dim) return;

    WORD_TYPE tl, tc, tr;
    WORD_TYPE cl, cc, cr;
    WORD_TYPE bl, bc, br;

    tl = load_word(input, x - 1, y - 1, x_dim, dim);
    tc = load_word(input, x,     y - 1, x_dim, dim);
    tr = load_word(input, x + 1, y - 1, x_dim, dim);
    cl = load_word(input, x - 1, y,     x_dim, dim);
    cc = load_word(input, x,     y,     x_dim, dim);
    cr = load_word(input, x + 1, y,     x_dim, dim);
    bl = load_word(input, x - 1, y + 1, x_dim, dim);
    bc = load_word(input, x,     y + 1, x_dim, dim);
    br = load_word(input, x + 1, y + 1, x_dim, dim);

    WORD_TYPE result = 0;

    for (int bit = 1; bit < sizeof(uint64_t) * 8 - 1; bit++) {
        int count = 0;
        
        count += (tc >> (bit - 1)) & 1;
        count += (tc >> bit) & 1;
        count += (tc >> (bit + 1)) & 1;

        count += (cc >> (bit - 1)) & 1;
        count += (cc >> (bit + 1)) & 1;
        
        count += (bc >> (bit - 1)) & 1;
        count += (bc >> bit) & 1;
        count += (bc >> (bit + 1)) & 1;

        WORD_TYPE current = (cc >> bit) & 1;
        WORD_TYPE new_state = (count == 3 || (current && count == 2)) ? 1 : 0;

        result |= new_state << bit;
    }

    // 0th bit
    {
        int count = 0;
        
        count += (tl >> 63) & 1;
        count += (cl >> 63) & 1;
        count += (bl >> 63) & 1;

        count += (tc) & 1;
        count += (tc >> 1) & 1;

        count += (cc >> 1) & 1;

        count += (bc) & 1;
        count += (bc >> 1) & 1;
        
        WORD_TYPE current = cc & 1;
        
        WORD_TYPE new_state = (count == 3 || (current && count == 2)) ? 1 : 0;

        result |= new_state;
    }

    // 63rd bit
    {
        int count = 0;
        
        count += tr & 1;
        count += cr & 1;
        count += br & 1;

        count += (tc >> 63) & 1;
        count += (tc >> 62) & 1;

        count += (cc >> 62) & 1;

        count += (bc >> 63) & 1;
        count += (bc >> 62) & 1;

        WORD_TYPE current = (cc >> 63) & 1;
        WORD_TYPE new_state = (count == 3 || (current && count == 2)) ? 1 : 0;
        result |= new_state << 63;
    }

    output[y * x_dim + x] = result;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    int x_dim = grid_dimensions / (sizeof(uint64_t) * 8);
    int y_dim = grid_dimensions;

    dim3 blockSize(32, 8);
    dim3 gridSize((x_dim + blockSize.x - 1) / blockSize.x, (y_dim + blockSize.y - 1) / blockSize.y);

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
