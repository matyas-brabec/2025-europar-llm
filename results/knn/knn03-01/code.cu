#include <cuda_runtime.h>
#include <utility>
#include <math_constants.h>

// Optimized k-NN for 2D points using one warp per query.
// - Data is processed in tiles cached in shared memory by the whole block.
// - Each warp maintains a private top-k (sorted ascending by distance) in shared memory.
// - Candidates are inserted cooperatively by the warp to avoid races.
// - k is assumed to be a power of two in [32, 1024].

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable hyper-parameters chosen for modern data-center GPUs (A100/H100).
// - WARPS_PER_BLOCK: number of queries processed concurrently per block.
// - TILE_POINTS: number of data points cached per tile.
constexpr int WARPS_PER_BLOCK     = 8;                    // 8 warps per block = 256 threads
constexpr int THREADS_PER_BLOCK   = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_POINTS         = 4096;                 // 4096 points per tile => 32 KiB tile (float2)

static __device__ __forceinline__ float sq_distance_2d(const float2 a, const float2 b) {
    // Squared Euclidean distance between 2D points.
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return fmaf(dx, dx, dy * dy);
}

// Lower-bound position in ascending array 'arr' of length 'n' for value 'x'.
// Returns the first index i such that arr[i] >= x.
// For arrays filled with +inf initially, this will return 0 for any finite x.
static __device__ __forceinline__ int lower_bound_asc(const float* arr, int n, float x) {
    int l = 0, r = n;
#pragma unroll 1
    while (l < r) {
        int m = (l + r) >> 1;
        float v = arr[m];
        if (x <= v) {
            r = m;
        } else {
            l = m + 1;
        }
    }
    return l;
}

// Cooperative warp insertion of multiple candidates into a shared, ascending top-k array.
// Each lane provides at most one candidate (dist, index). The function will:
// - Iteratively select one of the lanes whose candidate qualifies (dist < current worst),
// - Compute the insertion position,
// - Shift the tail [pos, k-2] to the right by one cooperatively across the warp,
// - Insert the candidate at position 'pos',
// - Repeat until no remaining qualifying candidates.
// Notes:
// - top_dists/top_idx: pointers to the warp's private top-k arrays in shared memory.
// - k: size of the top-k.
// - dist, idx: the calling lane's candidate. Lanes without candidate can pass dist=+inf or rely on the active flag.
static __device__ __forceinline__ void warp_cooperative_insert(float* __restrict__ top_dists,
                                                               int*   __restrict__ top_idx,
                                                               int k,
                                                               float dist,
                                                               int idx,
                                                               unsigned full_mask)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    // 'active' indicates this lane still has a candidate to try inserting.
    // First-level filter: must be better than the current worst.
    bool active = (dist < top_dists[k - 1]);

    // Process insertions one by one, prioritizing lower lane indices among qualifiers.
#pragma unroll 1
    while (true) {
        // Re-evaluate qualification against potentially updated worst value.
        bool qualifies = active && (dist < top_dists[k - 1]);
        unsigned mask = __ballot_sync(full_mask, qualifies);
        if (mask == 0)
            break;

        int leader_lane = __ffs(mask) - 1;

        // Broadcast the leader's candidate (dist, idx) to the whole warp.
        float cand_dist = __shfl_sync(full_mask, dist, leader_lane);
        int   cand_idx  = __shfl_sync(full_mask, idx,  leader_lane);

        // Leader computes insertion position (lower_bound) in the ascending array.
        int pos = 0;
        if (lane == leader_lane) {
            pos = lower_bound_asc(top_dists, k, cand_dist);
        }
        pos = __shfl_sync(full_mask, pos, leader_lane);

        // Cooperative right-shift by 1: move [pos .. k-2] -> [pos+1 .. k-1].
        // Process from high to low indices to avoid overwrite hazards.
        __syncwarp(full_mask);
        for (int j = (k - 2) - lane; j >= pos; j -= WARP_SIZE) {
            top_dists[j + 1] = top_dists[j];
            top_idx[j + 1]   = top_idx[j];
        }
        __syncwarp(full_mask);

        // Single lane writes the new element; any lane can do it, choose lane 0.
        if (lane == 0) {
            top_dists[pos] = cand_dist;
            top_idx[pos]   = cand_idx;
        }
        __syncwarp(full_mask);

        // The leader's candidate has been inserted; mark it consumed.
        if (lane == leader_lane) {
            active = false;
        }
    }
}

// CUDA kernel: one warp (32 threads) computes the k nearest neighbors for one query point.
__global__ void knn_kernel_warp_per_query(const float2* __restrict__ query,
                                          int query_count,
                                          const float2* __restrict__ data,
                                          int data_count,
                                          std::pair<int, float>* __restrict__ result,
                                          int k)
{
    extern __shared__ unsigned char smem_raw[];

    // Shared memory layout:
    // [0 .. TILE_POINTS*sizeof(float2)) : shared cache of data tile
    // Next: WARPS_PER_BLOCK*k floats for top-k distances of each warp
    // Next: WARPS_PER_BLOCK*k ints for top-k indices of each warp
    float2* tile_points = reinterpret_cast<float2*>(smem_raw);
    float*  all_top_dists = reinterpret_cast<float*>(smem_raw + TILE_POINTS * sizeof(float2));
    int*    all_top_idx   = reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(all_top_dists) + WARPS_PER_BLOCK * (size_t)k * sizeof(float));

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid >> 5; // warp id within block [0 .. WARPS_PER_BLOCK-1]
    const unsigned FULL_MASK = 0xffffffffu;

    // Determine which query this warp is responsible for.
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool warp_active = (warp_global < query_count);

    // Pointers to the warp's private top-k buffers in shared memory.
    float* warp_top_dists = all_top_dists + warp * (size_t)k;
    int*   warp_top_idx   = all_top_idx   + warp * (size_t)k;

    // Load query point for this warp and broadcast to all lanes.
    float2 q = make_float2(0.0f, 0.0f);
    if (warp_active && lane == 0) {
        q = query[warp_global];
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Initialize the warp's top-k arrays (ascending order; fill with +inf distances and -1 indices).
    if (warp_active) {
#pragma unroll 1
        for (int i = lane; i < k; i += WARP_SIZE) {
            warp_top_dists[i] = CUDART_INF_F;
            warp_top_idx[i]   = -1;
        }
    }
    __syncwarp(FULL_MASK); // ensure a consistent initial state per warp

    // Iterate over data in tiles cached into shared memory.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        const int tile_count = min(TILE_POINTS, data_count - tile_base);

        // Cooperative load of the tile into shared memory by the whole block.
        for (int i = tid; i < tile_count; i += blockDim.x) {
            tile_points[i] = data[tile_base + i];
        }
        __syncthreads();

        // Each active warp processes the tile, each lane strides by 32 within the tile.
        if (warp_active) {
#pragma unroll 1
            for (int j = lane; j < tile_count; j += WARP_SIZE) {
                float2 p = tile_points[j];
                float dist = sq_distance_2d(q, p);
                int   idx  = tile_base + j;

                // Cooperative insertion across the warp for candidates that beat the current worst.
                // Each lane participates to avoid divergence around warp-level synchronization.
                warp_cooperative_insert(warp_top_dists, warp_top_idx, k, dist, idx, FULL_MASK);
            }
        }

        __syncthreads(); // ensure all warps finished using the current tile before reloading
    }

    // Write results for this warp's query as (index, distance) pairs in ascending order of distance.
    if (warp_active) {
        const int out_base = warp_global * k;
#pragma unroll 1
        for (int i = lane; i < k; i += WARP_SIZE) {
            // std::pair is a simple POD; directly assign to its fields.
            result[out_base + i].first  = warp_top_idx[i];
            result[out_base + i].second = warp_top_dists[i];
        }
    }
}

// Public C++ interface. Allocations are assumed to be done via cudaMalloc.
// This function configures and launches the kernel with carefully chosen parameters.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Kernel launch configuration.
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    // One warp per query. Each block processes WARPS_PER_BLOCK queries.
    int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (blocks <= 0) blocks = 1;
    dim3 grid(blocks, 1, 1);

    // Dynamic shared memory size per block:
    // - TILE_POINTS float2 for the cached tile
    // - WARPS_PER_BLOCK * k floats for distances
    // - WARPS_PER_BLOCK * k ints for indices
    size_t shmem_bytes = TILE_POINTS * sizeof(float2)
                       + WARPS_PER_BLOCK * (size_t)k * sizeof(float)
                       + WARPS_PER_BLOCK * (size_t)k * sizeof(int);

    // Opt-in larger shared memory carveout and dynamic shared memory limit where supported.
    // These calls are best-effort; ignore errors if unsupported (e.g., on older devices).
    cudaFuncSetAttribute(knn_kernel_warp_per_query,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);
    cudaFuncSetAttribute(knn_kernel_warp_per_query,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem_bytes);

    // Launch the kernel.
    knn_kernel_warp_per_query<<<grid, block, shmem_bytes>>>(query, query_count, data, data_count, result, k);
}