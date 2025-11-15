#include <cuda_runtime.h>
#include <cstdint>

/*
 * CUDA implementation of Conway's Game of Life on a square grid.
 *
 * Input/output format:
 * - The logical grid is grid_dimensions x grid_dimensions cells.
 * - Each cell is 0 (dead) or 1 (alive).
 * - On the device, the external API exposes the grid in bit-packed form:
 *     * A std::uint64_t encodes an 8x8 tile of cells.
 *     * Tiles are laid out in row-major order:
 *         tile_index = tile_y * (grid_dimensions / 8) + tile_x
 *       where (tile_x, tile_y) is the tile's coordinate.
 *     * Within a tile, bits are laid out in row-major order:
 *         bit_index = local_y * 8 + local_x
 *       where (local_x, local_y) are coordinates in [0, 7].
 *
 * Implementation strategy:
 * - For simplicity and performance on modern GPUs, the actual
 *   computation of the next generation is done on a dense
 *   1-byte-per-cell representation:
 *     * 0 byte => dead cell
 *     * 1 byte => alive cell
 * - We perform three kernels:
 *     1) unpack_tiles_kernel: bit-packed tiles -> dense byte grid
 *     2) game_of_life_step_kernel: one Game of Life step on the byte grid
 *     3) pack_tiles_kernel: dense byte grid -> bit-packed tiles
 *
 * Notes:
 * - Shared and texture memory are not used, since global memory access
 *   with a well-shaped grid/block configuration is sufficient on H100/A100.
 * - Any data transformation (unpack/pack) overhead is assumed acceptable
 *   per the problem statement.
 * - The caller is responsible for host-device synchronization;
 *   this code does not call cudaDeviceSynchronize.
 */


/*
 * Kernel 1: Unpack bit-packed 8x8 tiles into a dense 1-byte-per-cell grid.
 */
__global__ void unpack_tiles_kernel(const std::uint64_t* __restrict__ input,
                                    std::uint8_t* __restrict__ dense_grid,
                                    int grid_dim)
{
    const int tiles_per_row = grid_dim / 8;
    const int num_tiles     = tiles_per_row * tiles_per_row;

    int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_index >= num_tiles) return;

    // Determine tile coordinates in the grid of tiles.
    int tile_y = tile_index / tiles_per_row;
    int tile_x = tile_index % tiles_per_row;

    // Base coordinates of this tile in the cell grid.
    const int base_x = tile_x * 8;
    const int base_y = tile_y * 8;

    std::uint64_t word = input[tile_index];

    // Unpack 8x8 cells from the 64-bit word into dense_grid.
    // bit_index = local_y * 8 + local_x.
#pragma unroll
    for (int local_y = 0; local_y < 8; ++local_y) {
        // Extract the 8 bits for this row as a byte.
        std::uint8_t row_bits = static_cast<std::uint8_t>(word >> (local_y * 8));
        const int global_y    = base_y + local_y;
        const int row_offset  = global_y * grid_dim + base_x;

#pragma unroll
        for (int local_x = 0; local_x < 8; ++local_x) {
            std::uint8_t bit = (row_bits >> local_x) & 0x1u;
            dense_grid[row_offset + local_x] = bit;
        }
    }
}


/*
 * Kernel 2: Compute one Game of Life step on a dense 1-byte-per-cell grid.
 *
 * - grid_in:  current generation, values 0 or 1
 * - grid_out: next generation,    values 0 or 1
 * - grid_dim: width/height of the square grid
 *
 * Every thread updates exactly one cell.
 *
 * Boundary handling:
 * - Cells outside the grid are logically dead, i.e., neighbor accesses
 *   beyond the border contribute 0.
 * - For performance, interior cells (1 <= x < dim-1, 1 <= y < dim-1)
 *   use a fast path with no bounds checks.
 *   Boundary cells use a separate path with conditionals.
 */
__global__ void game_of_life_step_kernel(const std::uint8_t* __restrict__ grid_in,
                                         std::uint8_t* __restrict__ grid_out,
                                         int grid_dim)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_dim || y >= grid_dim) {
        return;
    }

    const int idx = y * grid_dim + x;

    std::uint8_t neighbors = 0;

    // Fast path: interior cells (no boundary checks needed)
    if (x > 0 && x < grid_dim - 1 && y > 0 && y < grid_dim - 1) {
        const int row_stride = grid_dim;
        const std::uint8_t* p = grid_in + idx;

        // Manually unrolled 3x3 neighborhood sum excluding center.
        neighbors =
            p[-row_stride - 1] + p[-row_stride] + p[-row_stride + 1] +
            p[-1]                               + p[+1] +
            p[+row_stride - 1] + p[+row_stride] + p[+row_stride + 1];
    } else {
        // Boundary path: explicitly clamp neighbor accesses to grid bounds.
        const int dim = grid_dim;

        // Row above
        if (y > 0) {
            const int base = (y - 1) * dim;
            if (x > 0) neighbors += grid_in[base + x - 1];
            neighbors += grid_in[base + x];
            if (x + 1 < dim) neighbors += grid_in[base + x + 1];
        }

        // Same row
        const int base = y * dim;
        if (x > 0) neighbors += grid_in[base + x - 1];
        if (x + 1 < dim) neighbors += grid_in[base + x + 1];

        // Row below
        if (y + 1 < dim) {
            const int base2 = (y + 1) * dim;
            if (x > 0) neighbors += grid_in[base2 + x - 1];
            neighbors += grid_in[base2 + x];
            if (x + 1 < dim) neighbors += grid_in[base2 + x + 1];
        }
    }

    const std::uint8_t cell = grid_in[idx];

    // Apply Game of Life rules:
    // - live cell survives iff it has 2 or 3 neighbors
    // - dead cell becomes alive iff it has exactly 3 neighbors
    std::uint8_t next_state =
        (neighbors == 3) | (cell & (neighbors == 2));

    grid_out[idx] = next_state;
}


/*
 * Kernel 3: Pack a dense 1-byte-per-cell grid back into 8x8 bit-packed tiles.
 */
__global__ void pack_tiles_kernel(const std::uint8_t* __restrict__ dense_grid,
                                  std::uint64_t* __restrict__ output,
                                  int grid_dim)
{
    const int tiles_per_row = grid_dim / 8;
    const int num_tiles     = tiles_per_row * tiles_per_row;

    int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_index >= num_tiles) return;

    int tile_y = tile_index / tiles_per_row;
    int tile_x = tile_index % tiles_per_row;

    const int base_x = tile_x * 8;
    const int base_y = tile_y * 8;

    std::uint64_t word = 0;

#pragma unroll
    for (int local_y = 0; local_y < 8; ++local_y) {
        const int global_y   = base_y + local_y;
        const int row_offset = global_y * grid_dim + base_x;

#pragma unroll
        for (int local_x = 0; local_x < 8; ++local_x) {
            const int idx = row_offset + local_x;
            std::uint8_t cell = dense_grid[idx] & 0x1u;

            const int bit_index = local_y * 8 + local_x;
            word |= (static_cast<std::uint64_t>(cell) << bit_index);
        }
    }

    output[tile_index] = word;
}


/*
 * Public API:
 *
 * Execute one step of Conway's Game of Life on a square grid using CUDA.
 *
 * Parameters:
 * - input:  Device pointer to bit-packed input grid.
 * - output: Device pointer to bit-packed output grid.
 * - grid_dimensions: Width/height of the square grid (power of 2).
 *
 * Requirements:
 * - input and output must be allocated with cudaMalloc.
 * - grid_dimensions is a power of 2, > 512, and a multiple of 8.
 *
 * Behavior:
 * - This function:
 *     1) Allocates temporary dense grids on the device.
 *     2) Unpacks the bit-packed input into a dense byte grid.
 *     3) Runs one Game of Life step on the dense grid.
 *     4) Packs the result back into bit-packed tiles in output.
 *     5) Frees the temporary dense grids.
 * - No host-device synchronization (cudaDeviceSynchronize) is performed.
 */
void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    const int dim        = grid_dimensions;
    const int num_cells  = dim * dim;
    const int tiles_per_row = dim / 8;
    const int num_tiles  = tiles_per_row * tiles_per_row;

    // Allocate temporary dense (1-byte-per-cell) grids on the device.
    std::uint8_t* d_dense_in  = nullptr;
    std::uint8_t* d_dense_out = nullptr;

    cudaMalloc(&d_dense_in,  static_cast<std::size_t>(num_cells) * sizeof(std::uint8_t));
    cudaMalloc(&d_dense_out, static_cast<std::size_t>(num_cells) * sizeof(std::uint8_t));

    // 1) Unpack bit-packed tiles to dense grid.
    {
        const int threads_per_block = 256;
        const int blocks = (num_tiles + threads_per_block - 1) / threads_per_block;
        unpack_tiles_kernel<<<blocks, threads_per_block>>>(input, d_dense_in, dim);
    }

    // 2) Run one Game of Life step on dense grid.
    {
        // Use a 2D launch configuration that favors coalesced accesses.
        dim3 block_dim(32, 8);  // 256 threads per block
        dim3 grid_dim_2d((dim + block_dim.x - 1) / block_dim.x,
                         (dim + block_dim.y - 1) / block_dim.y);

        game_of_life_step_kernel<<<grid_dim_2d, block_dim>>>(d_dense_in, d_dense_out, dim);
    }

    // 3) Pack dense grid back into bit-packed tiles.
    {
        const int threads_per_block = 256;
        const int blocks = (num_tiles + threads_per_block - 1) / threads_per_block;
        pack_tiles_kernel<<<blocks, threads_per_block>>>(d_dense_out, output, dim);
    }

    // Free temporary dense grids.
    cudaFree(d_dense_in);
    cudaFree(d_dense_out);
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
// MEMORY_LAYOUT: TILES
