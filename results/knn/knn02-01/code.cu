#include <cuda_runtime.h>
#include <utility>

// -----------------------------------------------------------------------------
// Configuration constants
// -----------------------------------------------------------------------------

// One warp (32 threads) processes a single query.
static constexpr int WARP_SIZE         = 32;
static constexpr int WARPS_PER_BLOCK   = 4;                      // 4 warps per block => 128 threads
static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
static constexpr int TILE_SIZE         = 2048;                   // Number of data points cached in shared memory per tile

// -----------------------------------------------------------------------------
// CUDA kernel implementing k-NN for 2D points.
// Each query is processed by a single warp (32 threads).
//
// Algorithm (per query / warp):
//  1. A warp loads its query point.
//  2. The block iterates over the data set in tiles of TILE_SIZE points.
//     Each tile is loaded into shared memory cooperatively by all threads.
//  3. For each tile, the warp processes its points in chunks of 32:
//       - Each lane computes distance to one data point from the tile.
//       - Lane 0 gathers all 32 distances via warp shuffles and updates a
//         shared-memory top-k list for the query using a streaming selection:
//            * Until k elements are filled, all candidates are inserted.
//            * Once filled, only candidates better than the current worst
//              are inserted, and the new worst is recomputed by scanning k.
//  4. After all data points have been processed, the warp's unsorted top-k
//     list (distance + index) is sorted in ascending order using an in-place
//     bitonic sort in shared memory driven by 32 threads in the warp.
//  5. The sorted neighbors are written to the result array for that query.
// -----------------------------------------------------------------------------

__global__ void knn_kernel_2d(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int k)
{
    extern __shared__ unsigned char shared_raw[];

    // Layout of shared memory:
    // [0 .. TILE_SIZE-1]         : tile of data points (float2)
    // [TILE_SIZE .. + WARPS_PER_BLOCK*k-1] : neighbor indices (int) per warp
    // [.. + WARPS_PER_BLOCK*k ..]          : neighbor distances (float) per warp
    float2* sData = reinterpret_cast<float2*>(shared_raw);
    int* sNeighborIdxAll = reinterpret_cast<int*>(sData + TILE_SIZE);
    float* sNeighborDistAll = reinterpret_cast<float*>(sNeighborIdxAll + WARPS_PER_BLOCK * k);

    const int lane_id        = threadIdx.x & (WARP_SIZE - 1);   // Lane index within warp
    const int warp_id_in_blk = threadIdx.x >> 5;                // Warp index within block (0..WARPS_PER_BLOCK-1)
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_blk; // One warp per query

    const bool warp_active = (global_warp_id < query_count);

    // Per-warp pointers into shared memory for neighbor lists
    int* warpNeighborIdx   = sNeighborIdxAll   + warp_id_in_blk * k;
    float* warpNeighborDist = sNeighborDistAll + warp_id_in_blk * k;

    // Load the query point for this warp
    float2 q;
    if (warp_active) {
        q = query[global_warp_id];
    }

    // State for streaming top-k, used only by lane 0 of active warps.
    int   neighbor_count = 0; // Number of neighbors currently stored (<= k)
    float worst_dist     = 0.0f; // Distance of the worst (largest) neighbor currently in list
    int   worst_pos      = 0;    // Position of the worst neighbor in the list

    if (warp_active && lane_id == 0) {
        neighbor_count = 0;
        // No need to initialize warpNeighborDist/warpNeighborIdx here;
        // elements are filled sequentially as candidates are accepted.
    }

    // -------------------------------------------------------------------------
    // Iterate over the data points in tiles cached in shared memory.
    // -------------------------------------------------------------------------
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        const int tile_size = (tile_start + TILE_SIZE <= data_count)
                                ? TILE_SIZE
                                : (data_count - tile_start);

        // Load tile of data points into shared memory (all threads cooperate).
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sData[i] = data[tile_start + i];
        }

        // Ensure the entire tile is loaded before any warp starts using it.
        __syncthreads();

        // Each active warp now processes the tile.
        if (warp_active) {
            const int num_warp_iters = (tile_size + WARP_SIZE - 1) / WARP_SIZE;

            for (int it = 0; it < num_warp_iters; ++it) {
                const int data_idx_in_tile = it * WARP_SIZE + lane_id;
                const int global_data_idx  = tile_start + data_idx_in_tile;

                // Compute distance for this lane's candidate, or mark invalid if out-of-range.
                float dist = 0.0f;
                int idx    = -1;

                if (data_idx_in_tile < tile_size) {
                    const float2 p = sData[data_idx_in_tile];
                    const float dx = p.x - q.x;
                    const float dy = p.y - q.y;
                    dist = dx * dx + dy * dy;  // squared L2 distance
                    idx  = global_data_idx;
                } else {
                    // No valid point for this lane in this iteration.
                    dist = 0.0f;
                    idx  = -1;
                }

                // Lane 0 gathers candidate distances and indices using warp shuffles
                // and updates the top-k neighbor list.
                const unsigned full_mask = 0xFFFFFFFFu;

                // All lanes participate in the shuffles; only lane 0 updates the list.
                for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
                    const float cand_dist = __shfl_sync(full_mask, dist, src_lane);
                    const int   cand_idx  = __shfl_sync(full_mask, idx, src_lane);

                    if (lane_id == 0 && cand_idx >= 0) {
                        // Streaming top-k insertion:
                        //  - While neighbor_count < k: append candidate.
                        //  - Once neighbor_count == k: only accept if better than worst_dist,
                        //    replacing the current worst and recomputing worst_dist/worst_pos.

                        if (neighbor_count < k) {
                            // Append candidate
                            warpNeighborDist[neighbor_count] = cand_dist;
                            warpNeighborIdx[neighbor_count]  = cand_idx;
                            ++neighbor_count;

                            // If we've just filled k entries, compute the initial worst.
                            if (neighbor_count == k) {
                                worst_pos  = 0;
                                worst_dist = warpNeighborDist[0];
                                for (int i = 1; i < k; ++i) {
                                    const float d = warpNeighborDist[i];
                                    if (d > worst_dist) {
                                        worst_dist = d;
                                        worst_pos  = i;
                                    }
                                }
                            }
                        } else {
                            // neighbor_count == k
                            if (cand_dist < worst_dist) {
                                // Replace current worst with candidate
                                warpNeighborDist[worst_pos] = cand_dist;
                                warpNeighborIdx[worst_pos]  = cand_idx;

                                // Recompute new worst entry
                                worst_pos  = 0;
                                worst_dist = warpNeighborDist[0];
                                for (int i = 1; i < k; ++i) {
                                    const float d = warpNeighborDist[i];
                                    if (d > worst_dist) {
                                        worst_dist = d;
                                        worst_pos  = i;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ensure all warps are done using this tile before it is overwritten.
        __syncthreads();
    }

    // At this point, each active warp has exactly k neighbors in its list:
    // warpNeighborIdx[0..k-1], warpNeighborDist[0..k-1] (unsorted).

    if (warp_active) {
        // ---------------------------------------------------------------------
        // Sort the k neighbors by ascending distance using in-place bitonic sort.
        // The sort operates entirely within the warp, using shared memory for
        // storage and 32 threads cooperatively performing compare-exchange steps.
        // ---------------------------------------------------------------------
        // Bitonic sort network from CUDA samples, adapted to use only 32 threads
        // for arbitrary power-of-two k (32 <= k <= 1024).
        for (int size_stage = 2; size_stage <= k; size_stage <<= 1) {
            for (int stride = size_stage >> 1; stride > 0; stride >>= 1) {
                __syncwarp();

                for (int i = lane_id; i < k; i += WARP_SIZE) {
                    int j = i ^ stride;
                    if (j > i) {
                        const bool ascending = ((i & size_stage) == 0);

                        float di = warpNeighborDist[i];
                        float dj = warpNeighborDist[j];
                        int   ii = warpNeighborIdx[i];
                        int   ij = warpNeighborIdx[j];

                        if (ascending) {
                            if (di > dj) {
                                // Swap to enforce ascending order
                                warpNeighborDist[i] = dj;
                                warpNeighborDist[j] = di;
                                warpNeighborIdx[i]  = ij;
                                warpNeighborIdx[j]  = ii;
                            }
                        } else {
                            if (di < dj) {
                                // Swap to enforce descending order in this segment
                                warpNeighborDist[i] = dj;
                                warpNeighborDist[j] = di;
                                warpNeighborIdx[i]  = ij;
                                warpNeighborIdx[j]  = ii;
                            }
                        }
                    }
                }
            }
        }
        __syncwarp();

        // ---------------------------------------------------------------------
        // Write sorted neighbors to global memory.
        // For query q = global_warp_id, its neighbors are stored in
        // result[q * k + 0 .. q * k + (k - 1)].
        // ---------------------------------------------------------------------
        const int base_out = global_warp_id * k;
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            const int out_idx = base_out + i;
            result[out_idx].first  = warpNeighborIdx[i];
            result[out_idx].second = warpNeighborDist[i];
        }
    }
}

// -----------------------------------------------------------------------------
// Host-side interface
// -----------------------------------------------------------------------------

// run_knn:
//   Launches the CUDA kernel to compute k-nearest neighbors for 2D points.
//   Inputs:
//     - query      : device pointer to query points (float2)
//     - query_count: number of queries
//     - data       : device pointer to data points (float2)
//     - data_count : number of data points
//     - result     : device pointer to result array of size query_count * k
//                    (each element is std::pair<int, float>)
//     - k          : number of neighbors (power of two, 32 <= k <= 1024)
void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // One warp per query; each block has WARPS_PER_BLOCK warps.
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Shared memory size:
    //  - TILE_SIZE float2 points
    //  - WARPS_PER_BLOCK * k neighbor indices (int)
    //  - WARPS_PER_BLOCK * k neighbor distances (float)
    const size_t shared_mem_size =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * (sizeof(int) + sizeof(float));

    knn_kernel_2d<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    // Synchronize to ensure results are ready when function returns.
    cudaDeviceSynchronize();
}