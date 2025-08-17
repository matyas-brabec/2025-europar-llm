#include <cuda_runtime.h>
#include <stdint.h>

/*
  High-performance CUDA implementation of one step of Conway's Game of Life.

  Design:
  - Thread-block tiling with 2D tiles and a 1-cell halo in shared memory.
    Each block of size BX x BY computes a BY x BX tile of the output grid.
    We allocate shared memory of size (BY+2) x (BX+2) to stage the tile plus halos.
  - Cooperative halo loading: only threads at the borders of the block load halos,
    and specific corner threads load the four corner halo cells. This avoids races.
  - Boundary handling: Cells outside the global grid are assumed dead (0). Halo
    loads check bounds and insert 0 if out-of-range. For interior blocks (not near
    the global boundary), we skip bounds checks entirely to reduce instruction count.
  - Data representation: Input and output are bool arrays in device memory. We
    reinterpret them as uint8_t for efficient arithmetic and coalesced memory ops.
  - Update rule: next = (sum == 3) | (center & (sum == 2)), using integer logic to
    avoid branches.
  - Block configuration: Default is BX=32, BY=16 (512 threads/block). Grid dimensions
    are assumed to be powers of two > 512, so they are divisible by these block sizes.
    However, we still handle arbitrary sizes with ceil division and out-of-range checks.

  Notes for modern NVIDIA GPUs (A100/H100):
  - Shared memory footprints are tiny, keeping high occupancy.
  - Global accesses for the tile interior are fully coalesced. Halos are thin borders.
  - The interior fast-path avoids bounds checks for most blocks.
*/

static_assert(sizeof(bool) == 1, "This implementation assumes bool is 1 byte.");

template<int BX, int BY>
__global__ void game_of_life_kernel(const uint8_t* __restrict__ in,
                                    uint8_t* __restrict__ out,
                                    int N)
{
    // Thread coordinates within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global coordinates this thread is responsible for
    const int x = blockIdx.x * BX + tx;
    const int y = blockIdx.y * BY + ty;

    // Shared memory tile with 1-cell halo on all sides
    __shared__ uint8_t tile[BY + 2][BX + 2];

    // Interior fast-path: if this block is completely inside the domain (no global boundary contact),
    // we can skip all bound checks for halo loads.
    const bool interior_block =
        (blockIdx.x > 0) && (blockIdx.x < gridDim.x - 1) &&
        (blockIdx.y > 0) && (blockIdx.y < gridDim.y - 1);

    if (interior_block) {
        // Interior cell (always valid)
        tile[ty + 1][tx + 1] = in[y * N + x];

        // Halos: load only where needed to avoid redundant work; corners by designated threads.
        if (tx == 0)           tile[ty + 1][0]       = in[y * N + (x - 1)];     // left
        if (tx == BX - 1)      tile[ty + 1][BX + 1]  = in[y * N + (x + 1)];     // right
        if (ty == 0)           tile[0][tx + 1]       = in[(y - 1) * N + x];     // top
        if (ty == BY - 1)      tile[BY + 1][tx + 1]  = in[(y + 1) * N + x];     // bottom

        if (tx == 0 && ty == 0)                       tile[0][0]         = in[(y - 1) * N + (x - 1)]; // top-left
        if (tx == BX - 1 && ty == 0)                  tile[0][BX + 1]    = in[(y - 1) * N + (x + 1)]; // top-right
        if (tx == 0 && ty == BY - 1)                  tile[BY + 1][0]    = in[(y + 1) * N + (x - 1)]; // bottom-left
        if (tx == BX - 1 && ty == BY - 1)             tile[BY + 1][BX + 1] = in[(y + 1) * N + (x + 1)]; // bottom-right
    } else {
        // Generic path with boundary checks to enforce "outside are dead (0)"
        // Interior element
        uint8_t center = 0;
        if (x < N && y < N) center = in[y * N + x];
        tile[ty + 1][tx + 1] = center;

        // Left halo
        if (tx == 0) {
            uint8_t v = 0;
            const int gx = x - 1;
            if (gx >= 0 && y < N) v = in[y * N + gx];
            tile[ty + 1][0] = v;
        }
        // Right halo
        if (tx == BX - 1) {
            uint8_t v = 0;
            const int gx = x + 1;
            if (gx < N && y < N) v = in[y * N + gx];
            tile[ty + 1][BX + 1] = v;
        }
        // Top halo
        if (ty == 0) {
            uint8_t v = 0;
            const int gy = y - 1;
            if (x < N && gy >= 0) v = in[gy * N + x];
            tile[0][tx + 1] = v;
        }
        // Bottom halo
        if (ty == BY - 1) {
            uint8_t v = 0;
            const int gy = y + 1;
            if (x < N && gy < N) v = in[gy * N + x];
            tile[BY + 1][tx + 1] = v;
        }
        // Corner halos
        if (tx == 0 && ty == 0) {
            uint8_t v = 0;
            const int gx = x - 1, gy = y - 1;
            if (gx >= 0 && gy >= 0) v = in[gy * N + gx];
            tile[0][0] = v;
        }
        if (tx == BX - 1 && ty == 0) {
            uint8_t v = 0;
            const int gx = x + 1, gy = y - 1;
            if (gx < N && gy >= 0) v = in[gy * N + gx];
            tile[0][BX + 1] = v;
        }
        if (tx == 0 && ty == BY - 1) {
            uint8_t v = 0;
            const int gx = x - 1, gy = y + 1;
            if (gx >= 0 && gy < N) v = in[gy * N + gx];
            tile[BY + 1][0] = v;
        }
        if (tx == BX - 1 && ty == BY - 1) {
            uint8_t v = 0;
            const int gx = x + 1, gy = y + 1;
            if (gx < N && gy < N) v = in[gy * N + gx];
            tile[BY + 1][BX + 1] = v;
        }
    }

    __syncthreads();

    // Compute the next state for valid coordinates only (threads outside NxN do nothing)
    if (x < N && y < N) {
        // Flattened access to shared tile to reduce indexing overhead
        const int pitch = BX + 2;
        const int s_idx = (ty + 1) * pitch + (tx + 1);
        const uint8_t* s = &tile[0][0];

        // Sum 8 neighbors from the halo-padded shared tile
        int sum =
            s[s_idx - pitch - 1] + s[s_idx - pitch] + s[s_idx - pitch + 1] +
            s[s_idx - 1]                        +          s[s_idx + 1]     +
            s[s_idx + pitch - 1] + s[s_idx + pitch] + s[s_idx + pitch + 1];

        const uint8_t center = s[s_idx];

        // Game of Life rule: survive if exactly 3 neighbors,
        // or if already alive and exactly 2 neighbors.
        // Use bitwise ops to avoid branches.
        uint8_t next = ((sum == 3) | ((sum == 2) & (center != 0))) ? 1u : 0u;

        out[y * N + x] = next;
    }
}

void run_game_of_life(const bool* input, bool* output, int grid_dimensions)
{
    // Reinterpret bool arrays as bytes for efficient arithmetic and memory ops.
    const uint8_t* in  = reinterpret_cast<const uint8_t*>(input);
    uint8_t* out       = reinterpret_cast<uint8_t*>(output);

    const int N = grid_dimensions;

    // Tuned block size for modern GPUs; ensures coalesced access along X.
    constexpr int BX = 32;
    constexpr int BY = 16;

    // Grid dimensions; using ceil division for generality (though N is a power of 2 > 512)
    dim3 block(BX, BY);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY);

    // Launch the kernel; any host-device synchronization is handled by the caller.
    game_of_life_kernel<BX, BY><<<grid, block>>>(in, out, N);
}