// CUDA implementation of a single-step Conway's Game of Life on a square grid,
// where the grid is bit-packed into 8×8 tiles stored in std::uint64_t values.
//
// Layout assumptions:
// - The full grid is grid_dimensions × grid_dimensions cells.
// - grid_dimensions is a power of 2, >= 512, and divisible by 8.
// - The grid is partitioned into tiles of size 8×8 cells.
// - Each tile is stored in one std::uint64_t (8 bytes).
// - Tiles are stored in row-major order: tile (tileY, tileX) is at index
//   tileY * tilesPerDim + tileX, where tilesPerDim = grid_dimensions / 8.
// - Within a tile, bits are laid out in row-major order with LSB first:
//   bit index = localY * 8 + localX, where localX, localY ∈ [0,7].
//   - localY = 0 is the top row of the tile
//   - localX = 0 is the leftmost column of the tile
//   - Bit 0 (LSB) is cell (0,0) of the tile, bit 63 is cell (7,7).
//
// Kernel strategy:
// - One thread processes one 8×8 tile.
// - Each thread loads a 3×3 neighborhood of tiles (up to 9 uint64_t values).
//   Tiles that would fall outside the grid are treated as all-dead (value 0).
// - For every cell in the central tile (64 cells), the thread counts the
//   number of alive neighbors by looking up bits in the 3×3 tile neighborhood.
// - The next state is computed using Conway's rules:
//     - Alive cell with <2 or >3 neighbors dies.
//     - Alive cell with 2 or 3 neighbors survives.
//     - Dead cell with exactly 3 neighbors becomes alive.
// - The result for the 8×8 tile is written back as one uint64_t.
//
// Performance considerations:
// - The kernel performs only coalesced global loads/stores and uses registers
//   for all intermediate data. No shared or texture memory is used.
// - Loops over tile coordinates are small and fully unrolled to reduce
//   branching and loop overhead.
// - Neighbor tiles at grid boundaries are zero-filled at load time, so the
//   neighbor-counting logic does not need to handle global boundary checks.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Device kernel
////////////////////////////////////////////////////////////////////////////////

__global__
void game_of_life_kernel(const std::uint64_t* __restrict__ input,
                         std::uint64_t* __restrict__ output,
                         int tilesPerDim)
{
    // Compute tile coordinates for this thread.
    const int tileX = blockIdx.x * blockDim.x + threadIdx.x;
    const int tileY = blockIdx.y * blockDim.y + threadIdx.y;

    if (tileX >= tilesPerDim || tileY >= tilesPerDim) {
        return;
    }

    const std::size_t tileIndex =
        static_cast<std::size_t>(tileY) * static_cast<std::size_t>(tilesPerDim)
        + static_cast<std::size_t>(tileX);

    // Load the 3×3 neighborhood of tiles surrounding the current tile.
    //
    // tiles[ (dy+1)*3 + (dx+1) ] corresponds to the tile at relative
    // offset (dy, dx), where dy, dx ∈ {-1, 0, 1}.
    //
    // Example:
    //   tiles[4] = tiles[(0+1)*3 + (0+1)] = center tile
    //   tiles[1] = tiles[(-1+1)*3 + (0+1)] = tile above
    //   tiles[7] = tiles[(1+1)*3 + (0+1)] = tile below
    //
    // Tiles outside the grid are treated as all-dead (0).
    std::uint64_t tiles[9];

    for (int dy = -1; dy <= 1; ++dy) {
        const int nty = tileY + dy;
        const bool yInBounds = (nty >= 0 && nty < tilesPerDim);

        for (int dx = -1; dx <= 1; ++dx) {
            const int ntx = tileX + dx;

            std::uint64_t val = 0;
            if (yInBounds && ntx >= 0 && ntx < tilesPerDim) {
                const std::size_t nIndex =
                    static_cast<std::size_t>(nty) * static_cast<std::size_t>(tilesPerDim)
                    + static_cast<std::size_t>(ntx);
                val = input[nIndex];
            }

            tiles[(dy + 1) * 3 + (dx + 1)] = val;
        }
    }

    const std::uint64_t centerTile = tiles[4]; // central tile at offset (0,0)

    std::uint64_t nextTile = 0;

    // Iterate over the 8×8 cells within the tile and compute the next state.
    // All loops are unrolled to maximize ILP and reduce control overhead.
    #pragma unroll
    for (int localY = 0; localY < 8; ++localY) {
        const int rowBase = localY * 8; // bit index base for this row

        #pragma unroll
        for (int localX = 0; localX < 8; ++localX) {
            const int selfBitIndex = rowBase + localX;

            const int selfAlive =
                static_cast<int>((centerTile >> selfBitIndex) & 1ull);

            int aliveNeighbors = 0;

            // Sum up alive neighbors over the 3×3 neighborhood.
            #pragma unroll
            for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                const int ny = localY + offsetY;

                // Determine which neighbor tile row this vertical offset falls
                // into and the local Y coordinate inside that tile.
                //
                // ny ∈ [-1, 8]
                // - If ny < 0      -> tileDY = -1, localNY = 7
                // - If 0 <= ny <=7 -> tileDY =  0, localNY = ny
                // - If ny > 7      -> tileDY =  1, localNY = 0
                const int tileDY = (ny < 0) ? -1 : (ny > 7 ? 1 : 0);
                const int localNY = ny & 7; // wrap to [0,7] (since ny is in [-1,8])

                #pragma unroll
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    // Skip the cell itself.
                    if (offsetX == 0 && offsetY == 0) {
                        continue;
                    }

                    const int nx = localX + offsetX;

                    // Determine which neighbor tile column this horizontal offset
                    // falls into and the local X coordinate inside that tile.
                    //
                    // nx ∈ [-1, 8]
                    // - If nx < 0      -> tileDX = -1, localNX = 7
                    // - If 0 <= nx <=7 -> tileDX =  0, localNX = nx
                    // - If nx > 7      -> tileDX =  1, localNX = 0
                    const int tileDX = (nx < 0) ? -1 : (nx > 7 ? 1 : 0);
                    const int localNX = nx & 7; // wrap to [0,7]

                    const int tileIdx = (tileDY + 1) * 3 + (tileDX + 1);
                    const std::uint64_t neighborTile = tiles[tileIdx];

                    const int neighborBitIndex = (localNY << 3) + localNX;

                    aliveNeighbors += static_cast<int>(
                        (neighborTile >> neighborBitIndex) & 1ull
                    );
                }
            }

            // Apply Conway's Game of Life rules:
            //
            // - Any alive cell with fewer than 2 alive neighbors dies.
            // - Any alive cell with 2 or 3 alive neighbors survives.
            // - Any alive cell with more than 3 alive neighbors dies.
            // - Any dead cell with exactly 3 alive neighbors becomes alive.
            //
            // aliveNeighbors ∈ [0,8]
            // selfAlive is 0 or 1.

            if (aliveNeighbors == 3 ||
                (selfAlive && aliveNeighbors == 2)) {
                nextTile |= (std::uint64_t{1} << selfBitIndex);
            }
        }
    }

    output[tileIndex] = nextTile;
}

////////////////////////////////////////////////////////////////////////////////
// Host entry point
////////////////////////////////////////////////////////////////////////////////

void run_game_of_life(const std::uint64_t* input,
                      std::uint64_t* output,
                      int grid_dimensions)
{
    // grid_dimensions is the number of cells per side; each tile is 8×8 cells.
    const int tilesPerDim = grid_dimensions >> 3; // grid_dimensions / 8

    // Thread block configuration: 16×16 threads per block.
    // Each thread handles one 8×8 tile, so the logical tile grid is
    // covered by the CUDA grid formed below.
    const dim3 blockDim(16, 16);
    const dim3 gridDim(
        (tilesPerDim + blockDim.x - 1) / blockDim.x,
        (tilesPerDim + blockDim.y - 1) / blockDim.y
    );

    // Launch kernel. Host-device synchronization and error checking
    // are assumed to be handled by the caller.
    game_of_life_kernel<<<gridDim, blockDim>>>(input, output, tilesPerDim);
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
