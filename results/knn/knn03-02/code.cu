// Optimized CUDA k-NN implementation for 2D points with warp-per-query mapping.
// Each query is handled by a single warp (32 threads). The data set is processed
// in tiles cached in shared memory. Each lane keeps a small local top-k' list
// (in registers), and a final warp-wide multi-way merge produces the global top-k.
//
// Assumptions / design decisions:
// - Target GPU: modern NVIDIA data-center GPU (A100/H100).
// - k is a power of two, 32 <= k <= 1024.
// - data_count >= k.
// - Each block contains WARPS_PER_BLOCK warps; each warp processes one query.
// - Data is processed in tiles of TILE_SIZE points stored in shared memory.
// - For each query, we do a brute-force scan over all data points.
// - Distance metric is squared Euclidean distance (L2 norm squared).
//
// Complexity per query:
// - Distance evaluations: O(data_count).
// - Local top-k' maintenance per lane: O((k/32) * data_count / 32) updates.
// - Final warp-wide multi-way merge over at most k candidates: O(k log 32).

#include <cuda_runtime.h>
#include <utility>   // std::pair
#include <cfloat>    // FLT_MAX

// Tunable kernel parameters
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

// Number of data points per shared-memory tile.
// TILE_SIZE * sizeof(float2) must fit into available shared memory.
constexpr int TILE_SIZE = 1024;

// Maximum per-lane local storage for intermediate neighbors.
// For k in [32, 1024], local_k = ceil(k / WARP_SIZE) <= 32.
constexpr int LOCAL_K_MAX = 32;

// ---------------------------------------------------------------------------
// Device helper: Insert a candidate (dist, idx) into a per-thread local top-k'
// list of length local_k, stored sorted by ascending distance.
//
// bestDist[0]..bestDist[local_k-1] are kept in ascending order:
//   bestDist[0]   : smallest distance in this lane's local list
//   bestDist[local_k-1] : largest (worst) distance in this lane's local list
//
// On insertion:
// - If candDist is worse than or equal to the current worst (>= bestDist[local_k-1]),
//   it is discarded.
// - Otherwise, it is inserted into its correct position (insertion sort step),
//   and the former worst element is dropped.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void insert_local_topk(float candDist,
                       int   candIdx,
                       float *bestDist,  // size LOCAL_K_MAX, only first local_k used
                       int   *bestIdx,   // size LOCAL_K_MAX, only first local_k used
                       int   local_k)
{
    // Reject if candidate is not better than the current worst
    if (candDist >= bestDist[local_k - 1]) {
        return;
    }

    // Insertion sort step into ascending-ordered array
    int j = local_k - 1;
    while (j > 0 && bestDist[j - 1] > candDist) {
        bestDist[j] = bestDist[j - 1];
        bestIdx[j]  = bestIdx[j - 1];
        --j;
    }
    bestDist[j] = candDist;
    bestIdx[j]  = candIdx;
}

// ---------------------------------------------------------------------------
// CUDA kernel: k-NN for 2D points.
// Each warp processes a single query point and scans the entire data set.
// Data are loaded into shared memory in tiles for better cache locality.
// Per-lane local top-k' lists are merged at the end to form the query's top-k.
// ---------------------------------------------------------------------------
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           int k,
                           std::pair<int, float> * __restrict__ result)
{
    // Shared-memory buffer for a tile of data points
    extern __shared__ float2 s_data[];

    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid & (WARP_SIZE - 1);

    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active        = (global_warp_id < query_count);
    const int query_idx      = global_warp_id;

    // Per-lane local list length: ceil(k / WARP_SIZE), bounded by LOCAL_K_MAX
    const int local_k   = (k + WARP_SIZE - 1) / WARP_SIZE;
    const unsigned full_mask = 0xffffffffu;

    // Per-lane local top-k' storage (in registers)
    float local_best_dist[LOCAL_K_MAX];
    int   local_best_idx[LOCAL_K_MAX];

    float2 q;

    if (active) {
        // Initialize this lane's local top-k' list
        #pragma unroll
        for (int i = 0; i < LOCAL_K_MAX; ++i) {
            if (i < local_k) {
                local_best_dist[i] = FLT_MAX;
                local_best_idx[i]  = -1;
            }
        }

        // Load the query point for this warp and broadcast to all lanes
        if (lane_id == 0) {
            q = query[query_idx];
        }
        q.x = __shfl_sync(full_mask, q.x, 0);
        q.y = __shfl_sync(full_mask, q.y, 0);
    }

    // Iterate over the data set in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) {
            tile_size = TILE_SIZE;
        }

        // Load the current tile into shared memory using all threads in the block
        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active) {
            // Each warp computes distances from its query to the cached tile points
            for (int i = lane_id; i < tile_size; i += WARP_SIZE) {
                float2 p = s_data[i];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float dist = dx * dx + dy * dy;
                int   idx  = tile_start + i;

                insert_local_topk(dist, idx, local_best_dist, local_best_idx, local_k);
            }
        }

        // Synchronize before loading the next tile to avoid data hazards
        __syncthreads();
    }

    if (active) {
        // Final multi-way merge:
        // We have 32 lanes, each with a sorted local list of size local_k.
        // Total candidates per query = WARP_SIZE * local_k >= k.
        // We perform k iterations of a warp-wide argmin to select neighbors
        // in ascending order of distance and write them directly to global memory.

        const int out_base = query_idx * k;
        int pos = 0; // this lane's current position in its local list

        for (int out_rank = 0; out_rank < k; ++out_rank) {
            // Candidate for this lane: next element from its local list, or FLT_MAX if exhausted
            float cand_dist = (pos < local_k) ? local_best_dist[pos] : FLT_MAX;
            int   cand_idx  = (pos < local_k) ? local_best_idx[pos]  : -1;

            // Warp-wide argmin reduction on (cand_dist, cand_idx, lane_id)
            float best_val  = cand_dist;
            int   best_idx  = cand_idx;
            int   best_lane = lane_id;

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other_val  = __shfl_down_sync(full_mask, best_val,  offset);
                int   other_idx  = __shfl_down_sync(full_mask, best_idx,  offset);
                int   other_lane = __shfl_down_sync(full_mask, best_lane, offset);

                if (other_val < best_val) {
                    best_val  = other_val;
                    best_idx  = other_idx;
                    best_lane = other_lane;
                }
            }

            // Broadcast the best candidate from lane 0 to all lanes
            best_val  = __shfl_sync(full_mask, best_val,  0);
            best_idx  = __shfl_sync(full_mask, best_idx,  0);
            best_lane = __shfl_sync(full_mask, best_lane, 0);

            // Lane 0 writes the selected neighbor for this query to global memory
            if (lane_id == 0) {
                result[out_base + out_rank].first  = best_idx;
                result[out_base + out_rank].second = best_val;
            }

            // The lane that contributed the best element advances its local pointer
            if (lane_id == best_lane) {
                ++pos;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side entry point.
// query      : pointer to device array of float2, size query_count
// query_count: number of query points
// data       : pointer to device array of float2, size data_count
// data_count : number of data points
// result     : pointer to device array of std::pair<int,float>, size query_count * k
// k          : number of nearest neighbors per query (power of 2, 32 <= k <= 1024)
// ---------------------------------------------------------------------------
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

    // Grid configuration: one warp per query
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    // Shared memory size: one tile of TILE_SIZE float2 points
    const size_t shared_mem_bytes = TILE_SIZE * sizeof(float2);

    knn_kernel<<<grid, block, shared_mem_bytes>>>(query,
                                                  query_count,
                                                  data,
                                                  data_count,
                                                  k,
                                                  result);
}