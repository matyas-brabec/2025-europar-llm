#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// This implementation assigns one warp (32 threads) to each query point.
// The data points are processed in tiles that are cached in shared memory.
// Within each warp, threads collaboratively maintain a list of the k nearest
// neighbors for their query. The intermediate result (indices and distances)
// is stored per-warp in shared memory.
//
// Distance metric: squared Euclidean distance in 2D.
//
// Main design points:
// - 1 warp per query, many warps per block.
// - The entire block cooperatively loads tiles of data into shared memory.
// - Each warp iterates over the shared-memory tile, computing distances to
//   its query and updating its private top-k list.
// - The top-k list is stored unsorted during accumulation and then sorted
//   (ascending by distance) cooperatively within the warp at the end.
// - Updates of the top-k list use warp-wide reductions (via __shfl_sync)
//   so that multiple threads participate in maintaining the intermediate
//   result (e.g., finding the current worst element).

// Tunable parameters.
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;     // 8 warps * 32 threads = 256 threads per block.
constexpr int BLOCK_THREADS     = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_SIZE         = 2048;  // Number of data points cached per block in shared memory.

// Warp-wide helper: find maximum value and its index in an array of length k.
// Each thread processes a strided subset of the array, then the warp reduces
// to the global maximum. The result (maxVal, maxIdx) is broadcast to all lanes.
__device__ __forceinline__
void warp_find_max(const float* __restrict__ arr,
                   int k,
                   float &maxVal,
                   int &maxIdx)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    float localMax = -FLT_MAX;
    int   localIdx = -1;

    // Strided scan over [0, k)
    for (int i = lane; i < k; i += WARP_SIZE) {
        float v = arr[i];
        if (v > localMax) {
            localMax = v;
            localIdx = i;
        }
    }

    // Warp-wide reduction to find maximum value and its index.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float otherMax = __shfl_down_sync(FULL_MASK, localMax, offset);
        int   otherIdx = __shfl_down_sync(FULL_MASK, localIdx, offset);
        if (otherMax > localMax) {
            localMax = otherMax;
            localIdx = otherIdx;
        }
    }

    // Broadcast the result from lane 0 to all lanes.
    maxVal = __shfl_sync(FULL_MASK, localMax, 0);
    maxIdx = __shfl_sync(FULL_MASK, localIdx, 0);
}

// Warp-wide helper: find minimum value and its index in arr[offset..k-1].
// Used for final selection-sort of the top-k list so that neighbors are
// ordered by ascending distance.
__device__ __forceinline__
void warp_find_min_from(const float* __restrict__ arr,
                        int k,
                        int offset,
                        float &minVal,
                        int &minIdx)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    float localMin = FLT_MAX;
    int   localIdx = -1;

    // Strided scan over [offset, k)
    for (int i = lane + offset; i < k; i += WARP_SIZE) {
        float v = arr[i];
        if (v < localMin) {
            localMin = v;
            localIdx = i;
        }
    }

    // Warp-wide reduction to find minimum value and its index.
    for (int offsetShfl = WARP_SIZE / 2; offsetShfl > 0; offsetShfl >>= 1) {
        float otherMin = __shfl_down_sync(FULL_MASK, localMin, offsetShfl);
        int   otherIdx = __shfl_down_sync(FULL_MASK, localIdx, offsetShfl);
        if (otherMin < localMin) {
            localMin = otherMin;
            localIdx = otherIdx;
        }
    }

    // Broadcast result to all lanes.
    minVal = __shfl_sync(FULL_MASK, localMin, 0);
    minIdx = __shfl_sync(FULL_MASK, localIdx, 0);
}

// Kernel implementing k-NN with one warp per query.
// query   : array of float2(query_count)
// data    : array of float2(data_count)
// result  : array of std::pair<int,float>(query_count * k)
// k       : number of nearest neighbors (power of two, 32..1024)
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    extern __shared__ unsigned char smem[];

    // Layout of shared memory:
    // - First: TILE_SIZE float2's for cached data points (shared by all warps).
    // - Then:  WARPS_PER_BLOCK * k floats for per-warp distances.
    // - Then:  WARPS_PER_BLOCK * k ints   for per-warp indices.
    float2* s_data = reinterpret_cast<float2*>(smem);
    float*  s_dist_base = reinterpret_cast<float*>(s_data + TILE_SIZE);
    int*    s_idx_base  = reinterpret_cast<int*>(s_dist_base + WARPS_PER_BLOCK * k);

    const int thread_id = threadIdx.x;
    const int lane      = thread_id & (WARP_SIZE - 1);
    const int warp_id   = thread_id / WARP_SIZE;

    // Warp-private top-k arrays in shared memory.
    float* warp_top_dist = s_dist_base + warp_id * k;
    int*   warp_top_idx  = s_idx_base  + warp_id * k;

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool warp_active = (query_idx < query_count);

    // Initialize the per-warp top-k lists.
    // All entries start with distance = +INF and index = -1.
    if (warp_active) {
        for (int i = lane; i < k; i += WARP_SIZE) {
            warp_top_dist[i] = FLT_MAX;
            warp_top_idx[i]  = -1;
        }
    }

    // We must ensure all writes to shared memory are visible before any warp
    // starts reading or updating the per-warp top-k buffers.
    __syncthreads();

    float worst_dist = FLT_MAX;
    int   worst_pos  = 0;

    // Find initial worst (maximum) distance among the k entries (all INF).
    // This also establishes a consistent worst_dist/worst_pos across the warp.
    if (warp_active) {
        warp_find_max(warp_top_dist, k, worst_dist, worst_pos);
    }

    // Preload the query point for this warp into registers.
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }

    // Tile-based iteration over all data points.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Block-wide load of current tile of data points into shared memory.
        // All threads in the block participate to maximize memory throughput.
        for (int i = thread_id; i < tile_size; i += BLOCK_THREADS) {
            s_data[i] = data[tile_start + i];
        }

        // Ensure the entire tile is loaded before any warp begins processing.
        __syncthreads();

        if (warp_active) {
            const unsigned FULL_MASK = 0xffffffffu;

            // Process the tile in groups of WARP_SIZE points.
            int num_groups = (tile_size + WARP_SIZE - 1) / WARP_SIZE;

            for (int group = 0; group < num_groups; ++group) {
                int idx_in_tile = group * WARP_SIZE + lane;

                float cand_dist = FLT_MAX;
                int   cand_idx  = -1;

                if (idx_in_tile < tile_size) {
                    float2 d = s_data[idx_in_tile];
                    float dx = d.x - qx;
                    float dy = d.y - qy;
                    cand_dist = dx * dx + dy * dy;
                    cand_idx  = tile_start + idx_in_tile;  // Global data index.
                }

                // Now we have up to 32 candidates (one per lane) in this group.
                // We update the per-warp top-k list with these candidates,
                // inserting any that are better than the current worst.
                //
                // We do this sequentially over lanes, but each insertion uses
                // 32 threads cooperatively (warp_find_max) so that updating
                // the intermediate result exploits warp-level parallelism.
                for (int source_lane = 0; source_lane < WARP_SIZE; ++source_lane) {
                    // Broadcast candidate from lane 'source_lane' to all lanes.
                    float cd = __shfl_sync(FULL_MASK, cand_dist, source_lane);
                    int   ci = __shfl_sync(FULL_MASK, cand_idx,  source_lane);

                    // Skip invalid candidates (out-of-bounds in a partial group).
                    if (ci < 0)
                        continue;

                    // Single-lane write to the worst position (to avoid races).
                    int do_insert = 0;
                    if (lane == 0 && cd < worst_dist) {
                        warp_top_dist[worst_pos] = cd;
                        warp_top_idx[worst_pos]  = ci;
                        do_insert = 1;
                    }

                    // Broadcast whether we actually inserted this candidate.
                    int inserted = __shfl_sync(FULL_MASK, do_insert, 0);

                    if (inserted) {
                        // Ensure the update to warp_top_dist/warp_top_idx is visible
                        // before recomputing the new worst element.
                        __syncwarp(FULL_MASK);
                        warp_find_max(warp_top_dist, k, worst_dist, worst_pos);
                        __syncwarp(FULL_MASK);
                    }
                }
            }
        }

        // All warps must finish using this tile before it can be overwritten.
        __syncthreads();
    }

    // After processing all tiles, the warp has k nearest neighbors in
    // warp_top_dist / warp_top_idx, but in arbitrary order.
    // We cooperatively perform a selection sort (ascending by distance) so
    // that result[i*k + j] is truly the j-th nearest neighbor.
    if (warp_active) {
        for (int out_pos = 0; out_pos < k; ++out_pos) {
            float min_val;
            int   min_idx;

            // Find minimum distance and its index in [out_pos, k).
            warp_find_min_from(warp_top_dist, k, out_pos, min_val, min_idx);

            if (lane == 0 && min_idx >= 0 && min_idx != out_pos) {
                // Swap positions out_pos and min_idx.
                float tmp_d = warp_top_dist[out_pos];
                int   tmp_i = warp_top_idx[out_pos];

                warp_top_dist[out_pos] = warp_top_dist[min_idx];
                warp_top_idx[out_pos]  = warp_top_idx[min_idx];

                warp_top_dist[min_idx] = tmp_d;
                warp_top_idx[min_idx]  = tmp_i;
            }

            __syncwarp(0xffffffffu);
        }

        // Write the sorted k nearest neighbors for this query to global memory.
        for (int i = lane; i < k; i += WARP_SIZE) {
            int out_idx = query_idx * k + i;
            result[out_idx].first  = warp_top_idx[i];
            result[out_idx].second = warp_top_dist[i];
        }
    }
}

// Host-side wrapper with the requested interface.
// Assumptions:
// - query, data, and result are device pointers allocated with cudaMalloc.
// - data_count >= k, and k is a power of two between 32 and 1024 inclusive.
// - query_count and data_count are large enough to benefit from GPU execution.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Compute grid configuration: one warp per query.
    int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Shared memory size: data tile + per-warp top-k (dist + idx).
    size_t shared_mem_bytes =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * k * (sizeof(float) + sizeof(int));

    knn_kernel<<<num_blocks, BLOCK_THREADS, shared_mem_bytes>>>(
        query, query_count, data, data_count, result, k);

    // In a production environment you might want to handle errors and/or
    // omit device synchronization here for better overlap; for simplicity,
    // we synchronize to ensure completion when run_knn returns.
    cudaDeviceSynchronize();
}