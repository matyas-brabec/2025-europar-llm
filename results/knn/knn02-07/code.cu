#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// This implementation assigns one CUDA warp (32 threads) to each query point.
// The warp cooperatively scans all data points in 2D space, keeping a private
// top-k list of nearest neighbors for its query in shared memory.
//
// Key features:
// - Warp-per-query: each warp handles one query, all 32 threads share the work.
// - Data tiling: data points are processed in tiles cached in shared memory.
// - Top-k storage per query: for each query/warp we keep an ordered list of
//   indices and distances of its k nearest neighbors in shared memory.
// - Warp-level communication: shuffles and ballots are used so only candidates
//   that beat the current worst neighbor are inserted into the top-k list.
//
// Top-k details:
// - For each query, we maintain a sorted array (ascending by distance) of size k.
//   The worst (largest) distance is always at position k - 1.
// - For a new candidate (idx, dist), we first compare dist against the current
//   worst distance. If it is not better, we skip it.
// - If it is better, we binary-search for its insertion position and shift the
//   tail of the array to make room. Because new insertions are rare compared to
//   distance computations, this remains efficient even for k up to 1024.
//
// Constraints matched from the problem:
// - Exactly one warp (32 threads) per query.
// - No extra device memory allocations; only shared memory is used.
// - k is a power of two between 32 and 1024 (inclusive).
// - Data are processed iteratively in batches cached in shared memory.

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;    // 4 warps * 32 threads = 128 threads per block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_K             = 1024; // maximum supported k
constexpr int TILE_POINTS       = 1024; // number of data points cached per tile in shared memory

// Helper to insert a candidate into the warp-local top-k array.
// This is called only by lane 0 of each warp.
// top_dists: ascending sorted array of size k.
// top_idx  : corresponding indices.
// k        : actual k (<= MAX_K).
// idx/dist : new candidate.
// worst_dist (in/out): current worst distance for this warp (top_dists[k-1]).
__device__ __forceinline__
void insert_topk_lane0(float* top_dists, int* top_idx, int k,
                       int idx, float dist, float &worst_dist)
{
    // Precondition: dist < worst_dist
    // Binary search to find insertion position in ascending array.
    int left = 0;
    int right = k - 1;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (dist < top_dists[mid])
            right = mid;
        else
            left = mid + 1;
    }
    int pos = left;

    // Shift elements [pos, k-2] one step to the right; drop the old worst at k-1.
    for (int t = k - 1; t > pos; --t) {
        top_dists[t] = top_dists[t - 1];
        top_idx[t]   = top_idx[t - 1];
    }

    top_dists[pos] = dist;
    top_idx[pos]   = idx;
    worst_dist     = top_dists[k - 1];
}

__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    // Shared memory layout:
    // - s_data: tile of data points reused by all warps in the block.
    // - s_top_dists/s_top_indices: per-warp top-k buffers.
    __shared__ float2 s_data[TILE_POINTS];
    __shared__ float  s_top_dists[WARPS_PER_BLOCK * MAX_K];
    __shared__ int    s_top_indices[WARPS_PER_BLOCK * MAX_K];

    const int lane           = threadIdx.x & (WARP_SIZE - 1); // lane id in warp [0,31]
    const int warp_in_block  = threadIdx.x >> 5;              // warp id in block [0, WARPS_PER_BLOCK-1]
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    // One warp processes exactly one query; if there are more warps than queries, excess warps return.
    if (global_warp_id >= query_count)
        return;

    // Pointers to this warp's top-k buffers in shared memory.
    float* top_dists = &s_top_dists[warp_in_block * MAX_K];
    int*   top_idx   = &s_top_indices[warp_in_block * MAX_K];

    // Initialize top-k to +inf distance and invalid index.
    // Each lane initializes a strided subset of the k entries.
    for (int i = lane; i < k; i += WARP_SIZE) {
        top_dists[i] = FLT_MAX;
        top_idx[i]   = -1;
    }
    __syncwarp();

    // Load query point and broadcast to all lanes in the warp.
    float qx = 0.0f, qy = 0.0f;
    if (lane == 0) {
        float2 q = query[global_warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // Current worst distance for this warp's top-k list.
    float worst_dist = FLT_MAX;
    worst_dist = __shfl_sync(0xFFFFFFFFu, worst_dist, 0);

    // Process the dataset in tiles cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int remaining  = data_count - tile_start;
        int tile_size  = (remaining < TILE_POINTS) ? remaining : TILE_POINTS;

        // Block-wide load of tile into shared memory (coalesced).
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            s_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // Each warp processes all points in the tile.
        // Each lane handles a strided subset of the tile's points.
        for (int j = lane; j < tile_size; j += WARP_SIZE) {
            float2 p = s_data[j];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = dx * dx + dy * dy;
            int   data_idx = tile_start + j;

            // Check if this candidate can possibly enter the top-k for this warp.
            float current_worst = __shfl_sync(0xFFFFFFFFu, worst_dist, 0);
            bool  is_better     = (dist < current_worst);

            // Build a mask of lanes with candidates better than current worst.
            unsigned int mask = __ballot_sync(0xFFFFFFFFu, is_better);

            // Process better candidates one by one in lane 0 using warp shuffles.
            // This avoids races on the shared top-k arrays and ensures correctness.
            while (mask) {
                int cand_lane = __ffs(mask) - 1; // index of least significant set bit [0,31]

                // Fetch candidate's distance and index from the owning lane.
                float cand_dist = __shfl_sync(0xFFFFFFFFu, dist,     cand_lane);
                int   cand_idx  = __shfl_sync(0xFFFFFFFFu, data_idx, cand_lane);

                if (lane == 0) {
                    // Re-check against the (possibly updated) worst_dist.
                    if (cand_dist < worst_dist) {
                        insert_topk_lane0(top_dists, top_idx, k, cand_idx, cand_dist, worst_dist);
                    }
                }

                // Broadcast updated worst_dist to all lanes in the warp.
                worst_dist = __shfl_sync(0xFFFFFFFFu, worst_dist, 0);

                // Clear the processed lane's bit and continue.
                mask &= (mask - 1);
            }
        }

        __syncthreads(); // Ensure all warps are done with this tile before loading the next one.
    }

    // Write out the final top-k results for this query.
    // top_dists[0..k-1] are ascending; top_idx[] hold corresponding data indices.
    // Each lane writes a strided subset of results.
    for (int i = lane; i < k; i += WARP_SIZE) {
        int out_index = global_warp_id * k + i;
        result[out_index].first  = top_idx[i];
        result[out_index].second = top_dists[i];
    }
}

// Host interface as requested.
// Assumes query, data, and result pointers are already allocated on the device
// with cudaMalloc, and that data_count >= k and k is a power of two in [32,1024].
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K)
        return;

    dim3 block_dim(THREADS_PER_BLOCK);
    int  num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(num_blocks);

    knn_kernel<<<grid_dim, block_dim>>>(query, query_count, data, data_count, result, k);
    // Synchronization and error checking are left to the caller if desired.
}