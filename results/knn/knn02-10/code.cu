// Optimized CUDA implementation of k-NN for 2D points (float2) using one warp per query.
//
// Key design choices:
// - Each query is processed by a single warp (32 threads).
// - All warps in a block share a tile of data points cached in shared memory.
// - Each warp maintains a private top-k list (indices + distances) in shared memory.
// - Within a warp, all 32 lanes compute distances in parallel; lane 0 owns and updates
//   the shared top-k buffer, consuming candidates from all lanes via warp shuffles.
// - Distances use squared Euclidean (L2) distance: (dx*dx + dy*dy).
//
// Assumptions from the problem statement:
// - data_count >= k.
// - k is a power of two between 32 and 1024 (inclusive).
// - query_count and data_count are large enough to benefit from GPU parallelism.
// - query, data, and result pointers refer to device memory allocated with cudaMalloc.
//

#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Number of threads in a warp on NVIDIA GPUs.
static constexpr int WARP_SIZE = 32;

// Number of warps per thread block.
// 4 warps * 32 threads = 128 threads per block.
// This choice, together with TILE_SIZE=512, keeps the shared memory per block
// low enough (~36 KB for k=1024) to allow at least two blocks per SM.
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// Number of data points cached per tile in shared memory.
// With TILE_SIZE=512, shared memory per block:
//   tile: 512 * sizeof(float2) = 4 KB
//   top-k: WARPS_PER_BLOCK * k * (sizeof(int) + sizeof(float))
//   For k=1024: 4 * 1024 * 8 = 32 KB
//   Total ≈ 36 KB (fits comfortably in 96 KB SMEM with 2 blocks/SM).
static constexpr int TILE_SIZE = 512;

// Device helper: squared Euclidean distance between two 2D float2 points.
__device__ __forceinline__
float squared_distance(const float2 &a, const float2 &b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Initialize top-k arrays for a single query (one warp).
// - indices[0..k-1] set to -1
// - dists[0..k-1] set to FLT_MAX
__device__ __forceinline__
void knn_init_topk(int k, int *indices, float *dists) {
    for (int i = 0; i < k; ++i) {
        indices[i] = -1;
        dists[i] = FLT_MAX;
    }
}

// Insert a candidate (cand_idx, cand_dist) into the top-k list if it is closer
// than the current k-th neighbor.
// The arrays are maintained in ascending order of distance:
//   dists[0] is the smallest distance (nearest neighbor),
//   dists[k-1] is the largest distance (current worst among the k best).
//
// Algorithm:
// 1. If cand_dist >= dists[k-1], the candidate is worse than the current worst;
//    skip it.
// 2. Otherwise, shift larger elements to the right to make space for the new
//    candidate, keeping the arrays sorted.
__device__ __forceinline__
void knn_insert_topk(int k, int *indices, float *dists, int cand_idx, float cand_dist) {
    // Fast rejection based on the current worst distance.
    if (cand_dist >= dists[k - 1]) {
        return;
    }

    int pos = k - 1;
    // Shift elements with distance > cand_dist one position to the right.
    while (pos > 0 && cand_dist < dists[pos - 1]) {
        dists[pos]   = dists[pos - 1];
        indices[pos] = indices[pos - 1];
        --pos;
    }

    // Insert new candidate at the found position.
    dists[pos]   = cand_dist;
    indices[pos] = cand_idx;
}

// Kernel implementing k-NN for 2D points with one warp per query.
//
// Each block processes WARPS_PER_BLOCK queries (one warp per query).
// All warps in the block share a tile of data points in shared memory.
// Within each warp, lane 0 owns the warp's top-k buffer in shared memory;
// all lanes compute distances and then provide candidates to lane 0 via shuffles.
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result,
                int k) {
    // Dynamic shared memory layout:
    // [ 0                                ... TILE_SIZE-1 ]   -> float2 tile_points[]
    // [ TILE_SIZE                        ... ]               -> int   topk_indices_all[WARPS_PER_BLOCK * k]
    // [ TILE_SIZE + WARPS_PER_BLOCK * k  ... ]               -> float topk_dists_all[WARPS_PER_BLOCK * k]
    extern __shared__ unsigned char smem[];
    float2 *tile_points = reinterpret_cast<float2*>(smem);
    int    *topk_indices_all = reinterpret_cast<int*>(tile_points + TILE_SIZE);
    float  *topk_dists_all   = reinterpret_cast<float*>(topk_indices_all + WARPS_PER_BLOCK * k);

    const int lane_id          = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x >> 5;  // threadIdx.x / WARP_SIZE
    const int warp_global_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    const bool valid_warp = (warp_global_id < query_count);

    // Each warp gets its own slice of the shared top-k buffers.
    int   *warp_indices = topk_indices_all + warp_id_in_block * k;
    float *warp_dists   = topk_dists_all   + warp_id_in_block * k;

    // Load query point and initialize top-k for this warp.
    float2 q;
    if (valid_warp) {
        if (lane_id == 0) {
            q = query[warp_global_id];
            knn_init_topk(k, warp_indices, warp_dists);
        }
        // Broadcast query to all lanes in the warp.
        q.x = __shfl_sync(0xFFFFFFFFu, q.x, 0);
        q.y = __shfl_sync(0xFFFFFFFFu, q.y, 0);
    }

    // Ensure that tile_points is safe to use (even though currently unused)
    // before entering the main loop; also keeps barrier counts balanced.
    __syncthreads();

    // Process all data points in tiles cached in shared memory.
    for (int base = 0; base < data_count; base += TILE_SIZE) {
        const int remaining  = data_count - base;
        const int tile_size  = (remaining > TILE_SIZE) ? TILE_SIZE : remaining;

        // All threads in the block cooperatively load the next tile of data.
        // This ensures coalesced reads from global memory.
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {
            tile_points[t] = data[base + t];
        }

        // Make sure the entire tile is loaded before any warp uses it.
        __syncthreads();

        // Each valid warp processes the entire tile against its query.
        if (valid_warp) {
            // For each chunk of up to WARP_SIZE points in the tile:
            for (int tile_offset = 0; tile_offset < tile_size; tile_offset += WARP_SIZE) {
                const int t              = tile_offset + lane_id;
                const int global_data_ix = base + t;

                float dist = FLT_MAX;
                int   idx  = -1;

                if (t < tile_size) {
                    const float2 p = tile_points[t];
                    dist = squared_distance(q, p);
                    idx  = global_data_ix;
                }

                // Now each lane in the warp has one candidate (idx, dist).
                // Lane 0 will consume all 32 per chunk and update the warp's
                // top-k list stored in shared memory.
                const unsigned mask = 0xFFFFFFFFu;
                for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
                    const int   cand_idx  = __shfl_sync(mask, idx,  src_lane);
                    const float cand_dist = __shfl_sync(mask, dist, src_lane);

                    if (lane_id == 0 && cand_idx >= 0) {
                        knn_insert_topk(k, warp_indices, warp_dists, cand_idx, cand_dist);
                    }
                }
            }
        }

        // Synchronize before the tile is overwritten in the next iteration.
        __syncthreads();
    }

    // After scanning all data, write the k nearest neighbors for this query.
    // The top-k buffers are maintained in ascending order of distance.
    if (valid_warp && lane_id == 0) {
        const int out_base = warp_global_id * k;
        for (int i = 0; i < k; ++i) {
            result[out_base + i].first  = warp_indices[i];
            result[out_base + i].second = warp_dists[i];
        }
    }
}

// Host function: prepares and launches the kernel.
// All pointers are assumed to point to device memory.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    const dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    const int warps_per_block = WARPS_PER_BLOCK;
    const int num_blocks      = (query_count + warps_per_block - 1) / warps_per_block;
    const dim3 grid_dim(num_blocks, 1, 1);

    // Compute required dynamic shared memory size:
    // - TILE_SIZE float2 elements for the data tile.
    // - WARPS_PER_BLOCK * k ints and floats for per-warp top-k buffers.
    const size_t shared_bytes =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * (sizeof(int) + sizeof(float));

    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(query,
                                                      query_count,
                                                      data,
                                                      data_count,
                                                      result,
                                                      k);

    // Optional error checking or cudaDeviceSynchronize() can be performed by the caller.
}