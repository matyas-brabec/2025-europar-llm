// CUDA KNN implementation for 2D points.
// 
// Each query is processed by a single warp (32 threads). The data points
// are processed in tiles that are cached in shared memory. All threads in
// the block collaborate to load each tile. Then each warp computes distances
// from its query to all points in the tile and updates a private K-NN list.
// 
// For each warp:
//   - Lane 0 maintains a sorted (ascending) list of the K best neighbors
//     in shared memory (indices + squared distances).
//   - All 32 lanes compute distances in parallel.
//   - Distances for one "chunk" of 32 points are written into a per-warp
//     candidate buffer in shared memory.
//   - After a warp-level sync, lane 0 sequentially inserts these 32
//     candidates into the sorted list using an insertion-based K-selection.
//   - Because data_count >= k, after scanning all data points the list
//     contains K elements sorted by ascending distance.
// 
// The final K-NN list for each query is written back to the result array,
// where result[i * k + j] holds the j-th nearest neighbor for query i.
//
// This implementation is optimized for modern data-center GPUs (A100/H100)
// and assumes large data_count and query_count values to amortize overhead.

#include <cuda_runtime.h>
#include <utility>
#include <float.h>

// Tunable parameters. These choices balance shared memory usage and occupancy
// on modern GPUs (A100/H100).
static constexpr int WARP_SIZE        = 32;
static constexpr int WARPS_PER_BLOCK  = 4;    // 4 warps -> 128 threads per block
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS      = 4096; // Number of data points per shared-memory tile

// Device helper: insert a candidate into a sorted ascending list of length up to k.
// - top_dist / top_idx: arrays of current best neighbors, sorted by distance ascending
// - filled: number of valid entries currently in the list (0 <= filled <= k)
// - k: maximum size of the list
//
// If filled < k: the candidate is always inserted and filled is incremented.
// If filled == k: the candidate is inserted only if it is better than the current worst
// (which is at position k-1). Insertion is done by shifting larger entries to the right.
__device__ __forceinline__
void insert_candidate_sorted_asc(float dist, int idx,
                                 float *top_dist, int *top_idx,
                                 int &filled, int k)
{
    if (filled < k)
    {
        // There is still space: insert and keep array sorted ascending.
        int pos = filled;
        ++filled;

        // Shift elements larger than 'dist' to the right.
        while (pos > 0 && dist < top_dist[pos - 1])
        {
            top_dist[pos] = top_dist[pos - 1];
            top_idx[pos]  = top_idx[pos - 1];
            --pos;
        }

        top_dist[pos] = dist;
        top_idx[pos]  = idx;
    }
    else if (dist < top_dist[k - 1])
    {
        // List is full and candidate is better than the current worst (largest distance).
        int pos = k - 1;

        // Shift elements larger than 'dist' to the right.
        while (pos > 0 && dist < top_dist[pos - 1])
        {
            top_dist[pos] = top_dist[pos - 1];
            top_idx[pos]  = top_idx[pos - 1];
            --pos;
        }

        top_dist[pos] = dist;
        top_idx[pos]  = idx;
    }
}

// Kernel: each warp processes one query point.
__global__
void knn_kernel(const float2 * __restrict__ query,  int query_count,
                const float2 * __restrict__ data,   int data_count,
                std::pair<int, float> * __restrict__ result,
                int k)
{
    // Dynamic shared memory layout:
    // [ float2 tile_points[TILE_POINTS] ]
    // [ float  top_dist[WARPS_PER_BLOCK * k] ]
    // [ int    top_idx [WARPS_PER_BLOCK * k] ]
    // [ float  cand_dist[WARPS_PER_BLOCK * WARP_SIZE] ]
    // [ int    cand_idx [WARPS_PER_BLOCK * WARP_SIZE] ]
    extern __shared__ unsigned char shared_mem[];
    float2 *s_points = reinterpret_cast<float2*>(shared_mem);

    float *s_top_dist = reinterpret_cast<float*>(
        s_points + TILE_POINTS);

    int *s_top_idx = reinterpret_cast<int*>(
        s_top_dist + WARPS_PER_BLOCK * k);

    float *s_cand_dist = reinterpret_cast<float*>(
        s_top_idx + WARPS_PER_BLOCK * k);

    int *s_cand_idx = reinterpret_cast<int*>(
        s_cand_dist + WARPS_PER_BLOCK * WARP_SIZE);

    const int thread_id      = threadIdx.x;
    const int lane_id        = thread_id & (WARP_SIZE - 1);   // thread's lane within its warp
    const int warp_in_block  = thread_id >> 5;                // warp index within block
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;

    // Each warp corresponds to one query. Some warps in the last block may be inactive.
    const bool warp_active = (global_warp_id < query_count);

    // Per-warp pointers into shared memory for the K-NN list.
    float *warp_top_dist = s_top_dist + warp_in_block * k;
    int   *warp_top_idx  = s_top_idx  + warp_in_block * k;

    // Per-warp candidate buffer for one chunk of WARP_SIZE data points.
    float *warp_cand_dist = s_cand_dist + warp_in_block * WARP_SIZE;
    int   *warp_cand_idx  = s_cand_idx  + warp_in_block * WARP_SIZE;

    // Load the query point for this warp and broadcast it to all lanes.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active)
    {
        if (lane_id == 0)
        {
            float2 q = query[global_warp_id];
            qx = q.x;
            qy = q.y;
        }
        // Broadcast query coordinates from lane 0 to all lanes in the warp.
        unsigned mask = __activemask();
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Number of neighbors currently stored for this query (only meaningful in lane 0).
    int filled = 0;

    // Process the data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS)
    {
        const int tile_size = min(TILE_POINTS, data_count - tile_start);

        // All threads in the block cooperatively load the tile into shared memory.
        for (int i = thread_id; i < tile_size; i += blockDim.x)
        {
            s_points[i] = data[tile_start + i];
        }

        // Ensure the tile is fully loaded before any warp uses it.
        __syncthreads();

        if (warp_active)
        {
            // Each warp traverses the tile in chunks of WARP_SIZE points.
            for (int base = 0; base < tile_size; base += WARP_SIZE)
            {
                const int j = base + lane_id;

                // Compute distance to the query for this lane's point in the chunk.
                float dist = FLT_MAX;
                int   idx  = -1;
                if (j < tile_size)
                {
                    float2 p = s_points[j];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist = dx * dx + dy * dy;  // squared Euclidean distance
                    idx  = tile_start + j;     // global index of the data point
                }

                // Store candidate into per-warp candidate buffer.
                warp_cand_dist[lane_id] = dist;
                warp_cand_idx [lane_id] = idx;

                // Ensure all lanes have written their candidate before lane 0 reads them.
                __syncwarp();

                // Lane 0 inserts candidates into the K-NN list for this query.
                if (lane_id == 0)
                {
                    for (int t = 0; t < WARP_SIZE; ++t)
                    {
                        const int   cand_idx  = warp_cand_idx[t];
                        const float cand_dist = warp_cand_dist[t];

                        // cand_idx < 0 means this lane produced no valid candidate
                        // (outside tile bounds for the last chunk).
                        if (cand_idx < 0)
                            continue;

                        insert_candidate_sorted_asc(cand_dist, cand_idx,
                                                    warp_top_dist, warp_top_idx,
                                                    filled, k);
                    }
                }

                // Synchronize the warp before reusing the candidate buffer.
                __syncwarp();
            }
        }

        // Ensure no warp is still using the tile before it is overwritten.
        __syncthreads();
    }

    // Write final K-NN results for this query back to global memory.
    if (warp_active && lane_id == 0)
    {
        const int out_base = global_warp_id * k;
        for (int i = 0; i < k; ++i)
        {
            result[out_base + i].first  = warp_top_idx[i];
            result[out_base + i].second = warp_top_dist[i];
        }
    }
}

// Host entry point.
// 
// query       : device pointer to array of float2 of size query_count
// data        : device pointer to array of float2 of size data_count
// result      : device pointer to array of std::pair<int,float> of size query_count * k
// k           : number of neighbors, power-of-two in [32, 1024]
// 
// Assumptions:
//   - All pointers refer to device memory allocated with cudaMalloc.
//   - data_count >= k.
//   - query_count and data_count are large enough to justify GPU execution.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Grid configuration: one warp per query.
    const int threads_per_block = THREADS_PER_BLOCK;
    const int warps_per_block   = WARPS_PER_BLOCK;

    const int num_warps = (query_count + 0); // each warp processes at most one query
    const int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    dim3 block_dim(threads_per_block);
    dim3 grid_dim(num_blocks);

    // Dynamic shared memory size computation. We allocate only what we need given k.
    const size_t shared_points =
        static_cast<size_t>(TILE_POINTS) * sizeof(float2);

    const size_t shared_topk =
        static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) *
        (sizeof(float) + sizeof(int));

    const size_t shared_candidates =
        static_cast<size_t>(warps_per_block) * static_cast<size_t>(WARP_SIZE) *
        (sizeof(float) + sizeof(int));

    const size_t shared_bytes = shared_points + shared_topk + shared_candidates;

    // Opt-in for enough dynamic shared memory on devices that support >48KB (A100/H100).
    int device = 0;
    cudaGetDevice(&device);

    int max_shared_optin = 0;
    cudaDeviceGetAttribute(&max_shared_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);

    // Request at most the device's opt-in limit (should be >= shared_bytes on A100/H100).
    int shared_opt = static_cast<int>(shared_bytes);
    if (max_shared_optin > 0 && shared_opt > max_shared_optin)
        shared_opt = max_shared_optin;

    cudaFuncSetAttribute(knn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_opt);

    // Launch the kernel.
    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        query, query_count,
        data,  data_count,
        result, k);
}