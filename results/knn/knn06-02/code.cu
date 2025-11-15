// Optimized CUDA k-NN (k nearest neighbors) for 2D points with one warp per query.
// Target: modern NVIDIA data-center GPUs (A100/H100).
//
// The design follows the specification:
//  - One warp (32 threads) processes one query.
//  - All data points are processed in batches cached in shared memory.
//  - For each query (warp), we maintain:
//      * A private intermediate result of k nearest neighbors (indices + distances).
//      * A shared-memory candidate buffer of size k (indices + distances).
//      * A shared integer indicating the number of candidates currently stored.
//      * A max_distance variable tracking the distance of the k-th nearest neighbor.
//  - New distances lower than max_distance (or while we still have <k neighbors)
//    are inserted into the candidate buffer using atomicAdd to obtain positions.
//  - When the candidate buffer is full (before inserting a new candidate), we
//    merge it with the intermediate result using warp-cooperative selection.
//  - After processing all batches, we merge any remaining candidates.
//  - Final results for each query are written to result[i * k + j] as
//    std::pair<int,float>(index, distance), using squared Euclidean distances.

#include <cuda_runtime.h>
#include <utility>
#include <algorithm>

// Constants for warp and block configuration.
constexpr int WARP_SIZE        = 32;  // Fixed warp size on NVIDIA GPUs.
constexpr int WARPS_PER_BLOCK  = 4;   // 4 warps (128 threads) per block.

// Internal struct used for distances and indices in shared memory.
struct Neighbor {
    float dist;
    int   index;
};

// Device function that merges the current intermediate result and the candidate
// buffer for a single warp (i.e., a single query).
//
// It selects up to k smallest distances from the union of:
//   - warp_results[0..result_count-1]
//   - warp_cands[0..candidate_count-1]
// and stores them in warp_results (sorted by increasing distance).
//
// All 32 threads of the warp participate cooperatively using warp shuffles.
// Synchronization is done with __syncwarp() only (no __syncthreads()).
//
// Parameters are pointers to per-warp data structures stored in shared memory.
__device__ __forceinline__
void warp_merge_knn(
    Neighbor *warp_results,        // [k] current best neighbors (per warp)
    Neighbor *warp_cands,          // [k] candidate buffer (per warp)
    Neighbor *warp_merged,         // [k] temporary buffer for merged result (per warp)
    int      *warp_result_count,   // current number of valid entries in warp_results
    int      *warp_candidate_count,// current number of valid entries in warp_cands
    float    *warp_max_dist,       // distance of k-th neighbor (or INF if <k)
    int       k                    // requested number of neighbors
) {
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane_id        = threadIdx.x & (WARP_SIZE - 1);

    int result_count    = *warp_result_count;
    int candidate_count = *warp_candidate_count;

    // If there is nothing to merge, return early.
    if (result_count == 0 && candidate_count == 0) {
        return;
    }

    // Total available neighbors and the number of neighbors we will keep.
    int total      = result_count + candidate_count;
    int new_count  = (total < k) ? total : k;

    // Repeatedly select the global minimum among all remaining entries across
    // result and candidate buffers. Each iteration picks one neighbor to go
    // into warp_merged[sel], guaranteeing sorted order by distance.
    for (int sel = 0; sel < new_count; ++sel) {
        float best_dist         = CUDART_INF_F;
        int   best_index        = -1;
        int   best_from_cand    = 0;   // 0 => from results, 1 => from candidates

        // Each lane scans a subset of warp_results[0..result_count-1].
        for (int idx = lane_id; idx < result_count; idx += WARP_SIZE) {
            float d = warp_results[idx].dist;
            if (d < best_dist) {
                best_dist      = d;
                best_index     = idx;
                best_from_cand = 0;
            }
        }

        // Each lane also scans a subset of warp_cands[0..candidate_count-1].
        for (int idx = lane_id; idx < candidate_count; idx += WARP_SIZE) {
            float d = warp_cands[idx].dist;
            if (d < best_dist) {
                best_dist      = d;
                best_index     = idx;
                best_from_cand = 1;
            }
        }

        // Warp-wide reduction to find global minimum distance and its location.
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_dist  = __shfl_down_sync(FULL_MASK, best_dist,      offset);
            int   other_index = __shfl_down_sync(FULL_MASK, best_index,     offset);
            int   other_from  = __shfl_down_sync(FULL_MASK, best_from_cand, offset);

            if (other_dist < best_dist) {
                best_dist      = other_dist;
                best_index     = other_index;
                best_from_cand = other_from;
            }
        }

        // Lane 0 writes the selected neighbor into warp_merged[sel] and marks
        // the original entry as used by setting its distance to INF.
        if (lane_id == 0) {
            Neighbor best_neighbor;
            if (best_index >= 0) {
                if (best_from_cand) {
                    best_neighbor = warp_cands[best_index];
                    warp_cands[best_index].dist = CUDART_INF_F;
                } else {
                    best_neighbor = warp_results[best_index];
                    warp_results[best_index].dist = CUDART_INF_F;
                }
            } else {
                best_neighbor.dist  = CUDART_INF_F;
                best_neighbor.index = -1;
            }
            warp_merged[sel] = best_neighbor;
        }

        __syncwarp(FULL_MASK);
    }

    // Copy merged result back into warp_results and reset remaining entries.
    for (int idx = lane_id; idx < new_count; idx += WARP_SIZE) {
        warp_results[idx] = warp_merged[idx];
    }
    for (int idx = lane_id + new_count; idx < k; idx += WARP_SIZE) {
        warp_results[idx].dist  = CUDART_INF_F;
        warp_results[idx].index = -1;
    }

    // Update counts and max distance in a warp-synchronous manner.
    if (lane_id == 0) {
        *warp_result_count    = new_count;
        *warp_candidate_count = 0;
        if (new_count == k) {
            *warp_max_dist = warp_results[k - 1].dist;   // sorted: k-th neighbor
        } else {
            *warp_max_dist = CUDART_INF_F;               // still fewer than k
        }
    }
    __syncwarp(FULL_MASK);
}

// Kernel that computes k-NN for 2D points using one warp per query.
__global__
void knn_kernel(
    const float2 * __restrict__ query,   // [query_count]
    int                           query_count,
    const float2 * __restrict__ data,    // [data_count]
    int                           data_count,
    int                           k,
    std::pair<int,float> * __restrict__ result, // [query_count * k]
    int                           tile_points   // number of data points per shared-memory tile
) {
    extern __shared__ unsigned char smem[];

    const int warp_size        = WARP_SIZE;
    const int warps_per_block  = blockDim.x / warp_size;
    const int thread_id        = threadIdx.x;
    const int warp_id_in_block = thread_id / warp_size;
    const int lane_id          = thread_id & (warp_size - 1);

    const int global_warp_id   = blockIdx.x * warps_per_block + warp_id_in_block;

    // Each warp handles at most one query.
    const bool warp_active = (global_warp_id < query_count);

    // Shared memory layout:
    //   [0 .. tile_points-1]                 -> float2 data tile
    //   [tile_points .. + WARPS_PER_BLOCK*k] -> Neighbor results per warp
    //   [... + WARPS_PER_BLOCK*k]           -> Neighbor candidates per warp
    //   [... + WARPS_PER_BLOCK*k]           -> Neighbor merged per warp
    //   [... + WARPS_PER_BLOCK]             -> int candidate_counts per warp
    //   [... + WARPS_PER_BLOCK]             -> int result_counts per warp
    //   [... + WARPS_PER_BLOCK]             -> float max_dists per warp
    unsigned char *ptr = smem;

    float2  *s_data_tile         = reinterpret_cast<float2*>(ptr);
    ptr += static_cast<size_t>(tile_points) * sizeof(float2);

    Neighbor *s_results          = reinterpret_cast<Neighbor*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * k * sizeof(Neighbor);

    Neighbor *s_candidates       = reinterpret_cast<Neighbor*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * k * sizeof(Neighbor);

    Neighbor *s_merged           = reinterpret_cast<Neighbor*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * k * sizeof(Neighbor);

    int     *s_candidate_counts  = reinterpret_cast<int*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * sizeof(int);

    int     *s_result_counts     = reinterpret_cast<int*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * sizeof(int);

    float   *s_max_dists         = reinterpret_cast<float*>(ptr);
    // End of shared layout.

    // Pointers to per-warp data structures.
    Neighbor *warp_results        = s_results          + warp_id_in_block * k;
    Neighbor *warp_candidates     = s_candidates       + warp_id_in_block * k;
    Neighbor *warp_merged         = s_merged           + warp_id_in_block * k;
    int      *warp_candidate_cnt  = s_candidate_counts + warp_id_in_block;
    int      *warp_result_cnt     = s_result_counts    + warp_id_in_block;
    float    *warp_max_dist       = s_max_dists        + warp_id_in_block;

    // Initialize per-warp state for active warps.
    if (warp_active) {
        if (lane_id == 0) {
            *warp_candidate_cnt = 0;
            *warp_result_cnt    = 0;
            *warp_max_dist      = CUDART_INF_F;
        }
        // Initialize distances of the intermediate result to INF.
        for (int idx = lane_id; idx < k; idx += warp_size) {
            warp_results[idx].dist  = CUDART_INF_F;
            warp_results[idx].index = -1;
        }
        __syncwarp(0xffffffffu);
    }

    // Broadcast query point within the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active && lane_id == 0) {
        float2 q = query[global_warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xffffffffu, qx, 0);
    qy = __shfl_sync(0xffffffffu, qy, 0);

    // Process the data points in tiles loaded into shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int tile_size = tile_points;
        if (tile_start + tile_size > data_count) {
            tile_size = data_count - tile_start;
        }

        // Cooperative loading of the tile into shared memory by the whole block.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            s_data_tile[idx] = data[tile_start + idx];
        }
        __syncthreads();  // Block-wide barrier for shared tile.

        // Each active warp processes the cached tile.
        if (warp_active) {
            for (int local_idx = lane_id; local_idx < tile_size; local_idx += warp_size) {
                float2 p = s_data_tile[local_idx];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = dx * dx + dy * dy;  // squared Euclidean distance

                int   curr_result_cnt = *warp_result_cnt;
                float curr_max_dist   = *warp_max_dist;

                // Accept candidate if we still have fewer than k neighbors,
                // or if this distance is below max_distance.
                bool is_candidate = (curr_result_cnt < k) || (dist < curr_max_dist);

                unsigned int mask = __ballot_sync(0xffffffffu, is_candidate);

                // Process each candidate in the warp one by one.
                while (mask) {
                    int src_lane = __ffs(mask) - 1;

                    float cand_dist        = __shfl_sync(0xffffffffu, dist,       src_lane);
                    int   cand_local_idx   = __shfl_sync(0xffffffffu, local_idx,  src_lane);
                    int   cand_global_idx  = tile_start + cand_local_idx;

                    // If the candidate buffer is full (k entries), we merge it
                    // into the intermediate result before inserting the new one.
                    int merge_flag = 0;
                    if (lane_id == 0) {
                        if (*warp_candidate_cnt >= k) {
                            merge_flag = 1;
                        }
                    }
                    merge_flag = __shfl_sync(0xffffffffu, merge_flag, 0);

                    if (merge_flag) {
                        warp_merge_knn(
                            warp_results,
                            warp_candidates,
                            warp_merged,
                            warp_result_cnt,
                            warp_candidate_cnt,
                            warp_max_dist,
                            k
                        );
                    }

                    // Insert the new candidate into the buffer (lane 0 only),
                    // using atomicAdd to determine its position.
                    if (lane_id == 0) {
                        int pos = atomicAdd(warp_candidate_cnt, 1);
                        // We guarantee pos < k because we merged when full.
                        warp_candidates[pos].dist  = cand_dist;
                        warp_candidates[pos].index = cand_global_idx;
                    }
                    __syncwarp(0xffffffffu);

                    mask &= mask - 1;  // Clear the processed bit.
                }
            }
        }

        __syncthreads();  // Ensure all warps are done before reusing the tile.
    }

    // After processing all batches, merge any remaining candidates.
    if (warp_active) {
        int merge_flag = 0;
        if (lane_id == 0 && *warp_candidate_cnt > 0) {
            merge_flag = 1;
        }
        merge_flag = __shfl_sync(0xffffffffu, merge_flag, 0);

        if (merge_flag) {
            warp_merge_knn(
                warp_results,
                warp_candidates,
                warp_merged,
                warp_result_cnt,
                warp_candidate_cnt,
                warp_max_dist,
                k
            );
        }

        int final_count = *warp_result_cnt;

        // Write final k nearest neighbors to global memory.
        // For safety, if final_count < k (should not happen when data_count >= k),
        // remaining entries are filled with (-1, INF).
        for (int idx = lane_id; idx < k; idx += warp_size) {
            int out_idx = global_warp_id * k + idx;
            if (idx < final_count) {
                result[out_idx].first  = warp_results[idx].index;
                result[out_idx].second = warp_results[idx].dist;
            } else {
                result[out_idx].first  = -1;
                result[out_idx].second = CUDART_INF_F;
            }
        }
    }
}

// Host function that configures and launches the kernel.
//
// query   : device pointer to array of float2 query points  [query_count]
// data    : device pointer to array of float2 data points   [data_count]
// result  : device pointer to array of std::pair<int,float> [query_count * k]
// k       : number of nearest neighbors (power of two, 32..1024)
void run_knn(
    const float2 *query,
    int           query_count,
    const float2 *data,
    int           data_count,
    std::pair<int,float> *result,
    int           k
) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Device properties for shared memory configuration.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Maximum shared memory per block (opt-in if available).
    size_t max_smem = (prop.sharedMemPerBlockOptin > 0)
                      ? prop.sharedMemPerBlockOptin
                      : prop.sharedMemPerBlock;

    // Compute shared memory usage per block:
    //
    // Per warp:
    //   3 * k * sizeof(Neighbor)   // results, candidates, merged
    // Per block (WARPS_PER_BLOCK warps):
    //   warp_neighbors_bytes * WARPS_PER_BLOCK
    // Scalars:
    //   WARPS_PER_BLOCK * (2 * sizeof(int) + sizeof(float))
    //
    // Plus tile_points * sizeof(float2) for data tiles.

    size_t neighbor_bytes_per_warp  = static_cast<size_t>(3) * k * sizeof(Neighbor);
    size_t neighbor_bytes_per_block = neighbor_bytes_per_warp * WARPS_PER_BLOCK;
    size_t scalar_bytes_per_block   = WARPS_PER_BLOCK * (2 * sizeof(int) + sizeof(float));
    size_t base_shared              = neighbor_bytes_per_block + scalar_bytes_per_block;

    // Determine how many data points we can fit into a tile given the shared
    // memory constraints. Use up to 2048 points per tile when possible.
    size_t remaining_smem = (max_smem > base_shared) ? (max_smem - base_shared) : 0;
    int max_tile_points   = static_cast<int>(remaining_smem / sizeof(float2));
    if (max_tile_points < 1) {
        max_tile_points = 1;  // Fallback, though this should not happen on A100/H100.
    }
    int tile_points = std::min(max_tile_points, 2048);

    size_t shared_bytes = base_shared + static_cast<size_t>(tile_points) * sizeof(float2);

    // Allow the kernel to use up to the device's opt-in shared memory.
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(max_smem)
    );

    // One warp per query.
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    const int total_warps       = query_count;
    const int blocks            = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(blocks);
    dim3 block(threads_per_block);

    knn_kernel<<<grid, block, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        k,
        result,
        tile_points
    );

    // Synchronize to ensure completion before returning to the caller.
    cudaDeviceSynchronize();
}