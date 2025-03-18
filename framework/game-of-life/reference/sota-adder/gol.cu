#include "gol.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define INDEX(x, y, dim) ((y) * (dim) + (x))
#define WORD_TYPE uint64_t
#define BITS 64

__device__ WORD_TYPE load_word(const WORD_TYPE* input, int x_word, int y_word, int x_dim, int y_dim) {
    if (x_word < 0 || x_word >= x_dim || y_word < 0 || y_word >= y_dim) {
        return 0;
    }
    return input[y_word * x_dim + x_word];
}

__device__ WORD_TYPE compute_center_word_adder(
    WORD_TYPE A, WORD_TYPE B, WORD_TYPE C, 
    WORD_TYPE H, WORD_TYPE I, WORD_TYPE D,
    WORD_TYPE G, WORD_TYPE F, WORD_TYPE E) {

    const WORD_TYPE AB_1 = A & B;
    const WORD_TYPE AB_0 = A ^ B;
    const WORD_TYPE CD_1 = C & D;
    const WORD_TYPE CD_0 = C ^ D;
    const WORD_TYPE EF_1 = E & F;
    const WORD_TYPE EF_0 = E ^ F;
    const WORD_TYPE GH_1 = G & H;
    const WORD_TYPE GH_0 = G ^ H;
    const WORD_TYPE AD_0 = AB_0 ^ CD_0;
    const WORD_TYPE AD_1 = AB_1 ^ CD_1 ^ (AB_0 & CD_0);
    const WORD_TYPE AD_2 = AB_1 & CD_1;
    const WORD_TYPE EH_0 = EF_0 ^ GH_0;
    const WORD_TYPE EH_1 = EF_1 ^ GH_1 ^ (EF_0 & GH_0);
    const WORD_TYPE EH_2 = EF_1 & GH_1;
    const WORD_TYPE AH_0 = AD_0 ^ EH_0;
    const WORD_TYPE X = AD_0 & EH_0;
    const WORD_TYPE Y = AD_1 ^ EH_1;
    const WORD_TYPE AH_1 = X ^ Y;
    const WORD_TYPE AH_23 = AD_2 | EH_2 | (AD_1 & EH_1) | (X & Y);
    const WORD_TYPE Z = ~AH_23 & AH_1;
    const WORD_TYPE I_2 = ~AH_0 & Z;
    const WORD_TYPE I_3 = AH_0 & Z;

    return (I & I_2) | I_3;
}

__device__  WORD_TYPE compute_center_word(
    WORD_TYPE lt, WORD_TYPE ct, WORD_TYPE rt, 
    WORD_TYPE lc, WORD_TYPE cc, WORD_TYPE rc,
    WORD_TYPE lb, WORD_TYPE cb, WORD_TYPE rb) {

    const WORD_TYPE A = (lc << 1) | (lt >> (BITS - 1));
    const WORD_TYPE B = (cc << 1) | (ct >> (BITS - 1));
    const WORD_TYPE C = (rc << 1) | (rt >> (BITS - 1));
    const WORD_TYPE D = rc;
    const WORD_TYPE E = (rc >> 1) | (rb << (BITS - 1));
    const WORD_TYPE F = (cc >> 1) | (cb << (BITS - 1));
    const WORD_TYPE G = (lc >> 1) | (lb << (BITS - 1));
    const WORD_TYPE H = lc;
    const WORD_TYPE I = cc;

    return compute_center_word_adder(A, B, C, H, I, D, G, F, E);
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

    // WARNING: function is implemented for columns not rows
    //   ==> we need to transpose the input
    WORD_TYPE result = compute_center_word(
        tl, cl, bl,
        tc, cc, bc,
        tr, cr, br);

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
    (void) input;
    (void) output;
    (void) grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// MEMORY_LAYOUT: ROWS
