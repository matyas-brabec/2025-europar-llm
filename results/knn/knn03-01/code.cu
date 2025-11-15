#include <cuda_runtime.h>
#include <utility>

// -------------------------------------------------------------------------------------------------
// Optimized k-NN (k nearest neighbors) for 2D points on modern NVIDIA GPUs (e.g., A100/H100).
//
// Design:
//   * Each warp (32 threads) processes one query point.
//   * All data points are processed in tiles. Each tile of points is cooperatively loaded into
//     shared memory by the entire block.
//   * After loading a tile, each warp uses the cached tile to compute distances from its assigned
//     query point.
//   * Per query (per warp), we maintain an intermediate list of the current k best candidates
//     (indices + squared distances). This list is stored in shared memory, in a region private to
//     each warp.
//   * Within a warp, all lanes compute candidate distances in parallel. Lane 0 collects the
//     candidate distances/indices via warp shuffles and updates the warp-private top-k list.
//   * The top-k list is kept as an unsorted array along with the current worst (maximum) distance.
//     Updates only scan the list when a new candidate enters the top-k.
//   * After all tiles are processed, lane 0 of each warp selection-sorts its k candidates by
//     distance and writes the ordered k neighbors to the result array.
//
// Assumptions and constraints:
//   * k is a power of two in [32, 1024] (inclusive).
//   * data_count >= k.
//   * query_count and data_count are large enough to benefit from GPU parallelism.
//   * No device-side dynamic allocations; only registers and shared memory are used.
//   * Input pointers (query, data, result) are device pointers obtained via cudaMalloc.
// -------------------------------------------------------------------------------------------------

// Tunable launch configuration and tiling parameters.
constexpr int WARP_SIZE           = 32;
constexpr int THREADS_PER_BLOCK   = 256;   // 8 warps per block.
constexpr int DATA_TILE_SIZE      = 4096;  // Number of data points per shared-memory tile.

// -------------------------------------------------------------------------------------------------
// Device helper: Insert a candidate (dist, idx) into the current warp's top-k list.
//
// The top-k list is stored in shared memory arrays best_dists[0..k-1], best_indices[0..k-1].
// The list is not kept sorted. We track:
//   * current_size:  number of valid entries in the list (<= k).
//   * worst_dist:    current maximum distance in the list.
//   * worst_pos:     index of the current maximum distance.
//
// Behavior:
//   * While current_size < k, all candidates are inserted, and we update worst_dist/worst_pos
//     incrementally.
//   * Once the list is full (current_size == k):
//       - Only candidates with dist < worst_dist are inserted.
//       - On insertion, we overwrite best_dists[worst_pos], best_indices[worst_pos] and then
//         rescan the entire list to find the new worst_dist and worst_pos.
//
// This function is intended to be called by lane 0 only.
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__
void knn_insert_candidate(float dist,
                          int   idx,
                          float* best_dists,
                          int*   best_indices,
                          int    k,
                          int&   current_size,
                          float& worst_dist,
                          int&   worst_pos)
{
    if (current_size < k)
    {
        // Still filling the list: accept every candidate.
        best_dists[current_size]   = dist;
        best_indices[current_size] = idx;

        if (current_size == 0 || dist > worst_dist)
        {
            worst_dist = dist;
            worst_pos  = current_size;
        }

        ++current_size;
    }
    else
    {
        // List is full: only accept if this candidate improves the top-k.
        if (dist >= worst_dist)
        {
            // Not better than the current worst; ignore.
            return;
        }

        // Replace the worst entry.
        best_dists[worst_pos]   = dist;
        best_indices[worst_pos] = idx;

        // Recompute the new worst entry.
        float new_worst_dist = best_dists[0];
        int   new_worst_pos  = 0;

        for (int i = 1; i < k; ++i)
        {
            float v = best_dists[i];
            if (v > new_worst_dist)
            {
                new_worst_dist = v;
                new_worst_pos  = i;
            }
        }

        worst_dist = new_worst_dist;
        worst_pos  = new_worst_pos;
    }
}

// -------------------------------------------------------------------------------------------------
// CUDA kernel: Compute k-nearest neighbors for 2D points using warp-per-query and shared-memory
// tiling of data points.
//
// Each warp processes exactly one query. The mapping is:
//   global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE)
// Query index is global_warp_id.
//
// Shared memory layout per block (dynamic shared memory):
//   [0..DATA_TILE_SIZE-1]           : float2 tile_points[DATA_TILE_SIZE]
//   [DATA_TILE_SIZE..]              : float best_dists[warpsPerBlock * k]
//   [DATA_TILE_SIZE + ...]          : int   best_indices[warpsPerBlock * k]
// -------------------------------------------------------------------------------------------------
__global__ void knn_kernel(const float2* __restrict__ query,
                           int                            query_count,
                           const float2* __restrict__ data,
                           int                            data_count,
                           int                            k,
                           std::pair<int, float>* __restrict__ result)
{
    extern __shared__ unsigned char smem[];
    float2* tile_points = reinterpret_cast<float2*>(smem);

    // Compute number of warps per block at runtime (blockDim.x is multiple of WARP_SIZE).
    const int warpsPerBlock = blockDim.x / WARP_SIZE;

    // Pointers to warp-private top-k buffers in shared memory.
    float* shared_best_dists =
        reinterpret_cast<float*>(tile_points + DATA_TILE_SIZE);
    int*   shared_best_indices =
        reinterpret_cast<int*>(shared_best_dists + warpsPerBlock * k);

    const int lane_id      = threadIdx.x & (WARP_SIZE - 1);   // 0..31
    const int warp_id      = threadIdx.x >> 5;                // warp index within block
    const int global_warp  = blockIdx.x * warpsPerBlock + warp_id;
    const unsigned full_mask = 0xFFFFFFFFu;

    if (global_warp >= query_count)
    {
        return;
    }

    // Warp-private sections of the shared top-k lists.
    float* best_dists   = shared_best_dists   + warp_id * k;
    int*   best_indices = shared_best_indices + warp_id * k;

    // Load the query point for this warp.
    float2 q;
    if (lane_id == 0)
    {
        q = query[global_warp];
    }
    // Broadcast query coordinates to all lanes in the warp.
    q.x = __shfl_sync(full_mask, q.x, 0);
    q.y = __shfl_sync(full_mask, q.y, 0);

    // Initialize top-k structures (only lane 0 needs to write; others just synchronize).
    if (lane_id == 0)
    {
        for (int i = 0; i < k; ++i)
        {
            best_dists[i]   = CUDART_INF_F;
            best_indices[i] = -1;
        }
    }
    __syncwarp(full_mask);

    // Lane-0-private metadata for the top-k list.
    int   current_size = 0;
    float worst_dist   = -CUDART_INF_F;  // Sentinel value; updated as entries are inserted.
    int   worst_pos    = -1;

    // Process the data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE_SIZE)
    {
        // Size of this tile (may be partial at the end).
        int remaining = data_count - tile_start;
        int tile_size = (remaining > DATA_TILE_SIZE) ? DATA_TILE_SIZE : remaining;

        // Load tile of data points into shared memory.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x)
        {
            tile_points[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // Each warp processes all points in the tile.
        // We advance in strides of WARP_SIZE so that each lane handles one point per iteration.
        for (int base = 0; base < tile_size; base += WARP_SIZE)
        {
            int   point_idx = base + lane_id;
            float dist      = CUDART_INF_F;
            int   index     = -1;

            if (point_idx < tile_size)
            {
                float2 p  = tile_points[point_idx];
                float dx  = p.x - q.x;
                float dy  = p.y - q.y;
                dist      = dx * dx + dy * dy;  // Squared Euclidean distance.
                index     = tile_start + point_idx;
            }

            // Determine which lanes hold valid candidates in this iteration.
            unsigned active_mask = __ballot_sync(full_mask, point_idx < tile_size);

            // Lane 0 collects all active candidates from the warp using shuffles and
            // updates the top-k list.
            if (lane_id == 0)
            {
                // Iterate over all possible lanes; process only those that are active.
                for (int src = 0; src < WARP_SIZE; ++src)
                {
                    if (active_mask & (1u << src))
                    {
                        float cand_dist = __shfl_sync(active_mask, dist, src);
                        int   cand_idx  = __shfl_sync(active_mask, index, src);

                        knn_insert_candidate(cand_dist,
                                             cand_idx,
                                             best_dists,
                                             best_indices,
                                             k,
                                             current_size,
                                             worst_dist,
                                             worst_pos);
                    }
                }
            }

            // Ensure lane 0 finishes updating the top-k list before the next batch in this tile.
            __syncwarp(full_mask);
        }

        // Synchronize all threads before loading the next tile into shared memory.
        __syncthreads();
    }

    // After all tiles have been processed, lane 0 of each warp owns an unsorted top-k list.
    // Sort it in ascending order of distance (selection sort).
    if (lane_id == 0)
    {
        for (int i = 0; i < k - 1; ++i)
        {
            int   min_pos  = i;
            float min_dist = best_dists[i];

            for (int j = i + 1; j < k; ++j)
            {
                float d = best_dists[j];
                if (d < min_dist)
                {
                    min_dist = d;
                    min_pos  = j;
                }
            }

            if (min_pos != i)
            {
                // Swap distances.
                float tmp_dist          = best_dists[i];
                best_dists[i]           = best_dists[min_pos];
                best_dists[min_pos]     = tmp_dist;

                // Swap indices.
                int tmp_idx             = best_indices[i];
                best_indices[i]         = best_indices[min_pos];
                best_indices[min_pos]   = tmp_idx;
            }
        }

        // Write the sorted k nearest neighbors for this query to global memory.
        int out_base = global_warp * k;
        for (int j = 0; j < k; ++j)
        {
            result[out_base + j].first  = best_indices[j];
            result[out_base + j].second = best_dists[j];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Host wrapper: run_knn
//
// Launches the CUDA kernel with a warp-per-query mapping and shared-memory tiling. This function
// assumes `query`, `data`, and `result` are device pointers obtained via cudaMalloc.
//
// Parameters:
//   * query       : device pointer to query points (float2), size query_count.
//   * query_count : number of query points.
//   * data        : device pointer to data points (float2), size data_count.
//   * data_count  : number of data points.
//   * result      : device pointer to output array of size (query_count * k) of std::pair<int,float>.
//   * k           : number of nearest neighbors to find per query (power of two, 32..1024).
// -------------------------------------------------------------------------------------------------
void run_knn(const float2* query,
             int           query_count,
             const float2* data,
             int           data_count,
             std::pair<int, float>* result,
             int           k)
{
    // Number of warps (queries) per block.
    const int warpsPerBlock = THREADS_PER_BLOCK / WARP_SIZE;

    // Grid size: one warp per query.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size:
    //   * DATA_TILE_SIZE float2 points
    //   * warpsPerBlock * k floats for distances
    //   * warpsPerBlock * k ints for indices
    size_t shared_bytes =
        static_cast<size_t>(DATA_TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(float) +
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(int);

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(numBlocks);

    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        k,
        result);
}
