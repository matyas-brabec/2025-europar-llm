#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

static_assert(sizeof(bool) == 1, "This implementation assumes bool is 1 byte.");

// Tunable block dimensions. These are chosen to give good occupancy and memory coalescing
// on modern NVIDIA data-center GPUs (A100/H100).
#ifndef GOL_BLOCK_DIM_X
#define GOL_BLOCK_DIM_X 32
#endif

#ifndef GOL_BLOCK_DIM_Y
#define GOL_BLOCK_DIM_Y 16
#endif

// CUDA kernel implementing one step of Conway's Game of Life.
// Each thread updates one cell. The grid is square (N x N), where N is a power of 2 > 512.
// The grid is divided into 2D blocks of size GOL_BLOCK_DIM_X x GOL_BLOCK_DIM_Y.
// Shared memory tiling is used to minimize global memory traffic: each block loads a
// (GOL_BLOCK_DIM_X+2) x (GOL_BLOCK_DIM_Y+2) tile, including a 1-cell halo on all sides.
//
// Boundary handling: cells outside the grid are considered dead (0). This is enforced
// when loading halo values into shared memory.
__global__ __launch_bounds__(GOL_BLOCK_DIM_X * GOL_BLOCK_DIM_Y, 2)
void game_of_life_kernel(const unsigned char* __restrict__ input,
                         unsigned char* __restrict__ output,
                         int N)
{
    // Shared memory tile:
    // - X dimension: [0 .. GOL_BLOCK_DIM_X+1]
    // - Y dimension: [0 .. GOL_BLOCK_DIM_Y+1]
    // Indices 1..GOL_BLOCK_DIM_X and 1..GOL_BLOCK_DIM_Y store the "interior" cells for this block.
    // Index 0 and GOL_BLOCK_DIM_X+1 / GOL_BLOCK_DIM_Y+1 are the halo (neighbor) cells.
    //
    // Using 32-bit ints for shared cells avoids shared-memory bank conflicts on modern GPUs.
    __shared__ unsigned int tile[GOL_BLOCK_DIM_Y + 2][GOL_BLOCK_DIM_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * GOL_BLOCK_DIM_X + tx;
    const int y = blockIdx.y * GOL_BLOCK_DIM_Y + ty;

    const int sx = tx + 1;  // Shared memory X index (shifted by +1 for halo)
    const int sy = ty + 1;  // Shared memory Y index

    const size_t pitch = static_cast<size_t>(N);
    const size_t center_idx = static_cast<size_t>(y) * pitch + static_cast<size_t>(x);

    // Load center cell for this thread into shared memory.
    // All (x, y) are guaranteed to be valid because N is a power of 2 > 512 and the block
    // dimensions divide N (32 and 16 are powers of 2), so there are no partial blocks.
    unsigned int center = static_cast<unsigned int>(input[center_idx]);
    tile[sy][sx] = center;

    // Load left halo (one extra column at shared-memory X index 0).
    if (tx == 0)
    {
        unsigned int left = 0u;
        if (x > 0)
        {
            left = static_cast<unsigned int>(input[center_idx - 1]);
        }
        tile[sy][0] = left;
    }

    // Load right halo (one extra column at shared-memory X index GOL_BLOCK_DIM_X + 1).
    if (tx == GOL_BLOCK_DIM_X - 1)
    {
        unsigned int right = 0u;
        if (x + 1 < N)
        {
            right = static_cast<unsigned int>(input[center_idx + 1]);
        }
        tile[sy][GOL_BLOCK_DIM_X + 1] = right;
    }

    // Load top halo (one extra row at shared-memory Y index 0).
    if (ty == 0)
    {
        unsigned int top = 0u;
        if (y > 0)
        {
            const size_t top_idx = center_idx - pitch;
            top = static_cast<unsigned int>(input[top_idx]);
        }
        tile[0][sx] = top;
    }

    // Load bottom halo (one extra row at shared-memory Y index GOL_BLOCK_DIM_Y + 1).
    if (ty == GOL_BLOCK_DIM_Y - 1)
    {
        unsigned int bottom = 0u;
        if (y + 1 < N)
        {
            const size_t bottom_idx = center_idx + pitch;
            bottom = static_cast<unsigned int>(input[bottom_idx]);
        }
        tile[GOL_BLOCK_DIM_Y + 1][sx] = bottom;
    }

    // Load the four corner halo cells. Only four threads per block execute these branches:
    // top-left, top-right, bottom-left, bottom-right of the block.
    if (tx == 0 && ty == 0)
    {
        unsigned int val = 0u;
        if (x > 0 && y > 0)
        {
            const size_t idx = center_idx - pitch - 1;
            val = static_cast<unsigned int>(input[idx]);
        }
        tile[0][0] = val;
    }

    if (tx == GOL_BLOCK_DIM_X - 1 && ty == 0)
    {
        unsigned int val = 0u;
        if (x + 1 < N && y > 0)
        {
            const size_t idx = center_idx - pitch + 1;
            val = static_cast<unsigned int>(input[idx]);
        }
        tile[0][GOL_BLOCK_DIM_X + 1] = val;
    }

    if (tx == 0 && ty == GOL_BLOCK_DIM_Y - 1)
    {
        unsigned int val = 0u;
        if (x > 0 && y + 1 < N)
        {
            const size_t idx = center_idx + pitch - 1;
            val = static_cast<unsigned int>(input[idx]);
        }
        tile[GOL_BLOCK_DIM_Y + 1][0] = val;
    }

    if (tx == GOL_BLOCK_DIM_X - 1 && ty == GOL_BLOCK_DIM_Y - 1)
    {
        unsigned int val = 0u;
        if (x + 1 < N && y + 1 < N)
        {
            const size_t idx = center_idx + pitch + 1;
            val = static_cast<unsigned int>(input[idx]);
        }
        tile[GOL_BLOCK_DIM_Y + 1][GOL_BLOCK_DIM_X + 1] = val;
    }

    // Ensure all shared-memory loads are visible before computing neighbor sums.
    __syncthreads();

    // Compute the sum of the eight neighbors from shared memory.
    // We do not include the center cell in the sum.
    const unsigned int n00 = tile[sy - 1][sx - 1];
    const unsigned int n01 = tile[sy - 1][sx    ];
    const unsigned int n02 = tile[sy - 1][sx + 1];
    const unsigned int n10 = tile[sy    ][sx - 1];
    const unsigned int n12 = tile[sy    ][sx + 1];
    const unsigned int n20 = tile[sy + 1][sx - 1];
    const unsigned int n21 = tile[sy + 1][sx    ];
    const unsigned int n22 = tile[sy + 1][sx + 1];

    const unsigned int neighbor_sum =
        n00 + n01 + n02 +
        n10 +        n12 +
        n20 + n21 + n22;

    // Conway's Game of Life update rule:
    // - Any live cell with 2 or 3 live neighbors survives.
    // - Any dead cell with exactly 3 live neighbors becomes alive.
    // - All other cells die or stay dead.
    //
    // Expressed in boolean algebra:
    //   new_state = (neighbor_sum == 3) || (center == 1 && neighbor_sum == 2)
    //
    // We implement this using integer arithmetic to avoid branches:
    //   new_state = (neighbor_sum == 3) | (center & (neighbor_sum == 2))
    const unsigned int is_three = (neighbor_sum == 3);
    const unsigned int is_two   = (neighbor_sum == 2);
    const unsigned int new_state = (is_three | (center & is_two));

    // Store result back to global memory. Values are 0 or 1.
    output[center_idx] = static_cast<unsigned char>(new_state);
}

// Host function that runs one step of Conway's Game of Life on the GPU.
//
// Parameters:
//   input            - device pointer to the input grid of size (grid_dimensions^2),
//                      with each cell stored as a bool (false = dead, true = alive).
//   output           - device pointer to the output grid (same layout as input).
//   grid_dimensions  - width/height of the square grid; always a power of 2 > 512.
//
// The input and output arrays must be allocated with cudaMalloc by the caller.
// No host-device synchronization is performed here; the caller is responsible for
// synchronizing as needed.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    const int N = grid_dimensions;

    // Reinterpret bool pointers as unsigned char pointers. In CUDA, bool is 1 byte,
    // and using unsigned char gives better-defined behavior for arithmetic and
    // memory operations. This is safe because unsigned char can alias any object.
    const unsigned char* d_in  = reinterpret_cast<const unsigned char*>(input);
    unsigned char*       d_out = reinterpret_cast<unsigned char*>(output);

    // Because grid_dimensions is always a power of 2 > 512 and the block dimensions
    // (32 x 16) are powers of 2, N is guaranteed to be divisible by both 32 and 16.
    // Therefore, we can use exact division here with no partial edge blocks.
    dim3 block(GOL_BLOCK_DIM_X, GOL_BLOCK_DIM_Y, 1);
    dim3 grid(static_cast<unsigned int>(N / GOL_BLOCK_DIM_X),
              static_cast<unsigned int>(N / GOL_BLOCK_DIM_Y),
              1);

    game_of_life_kernel<<<grid, block>>>(d_in, d_out, N);
    // No cudaDeviceSynchronize() here; caller handles synchronization if needed.
}

// The content below was not generated by GPT; it was added to ensure the framework can compile the code.


#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

void initialize_internal_data_structures(int grid_dimensions) {
    (void)grid_dimensions;
}

// this label is used to identify the memory layout of the input and output arrays by the framework
// MEMORY_LAYOUT: BOOLS
