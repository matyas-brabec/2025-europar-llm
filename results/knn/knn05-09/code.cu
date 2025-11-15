#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

/*
 * Optimized k-NN (k nearest neighbors) for 2D points on modern NVIDIA GPUs.
 *
 * Design highlights:
 * - Each query is processed by a single warp (32 threads).
 * - For each query (warp):
 *     - A private intermediate top-k list is stored in shared memory.
 *     - A private candidate buffer of size k is stored in shared memory.
 * - The input dataset is processed in tiles. Each tile of data points is
 *   loaded into shared memory by the whole block.
 * - Each active warp:
 *     - Computes distances from its query to points in the current tile.
 *     - Adds points that are closer than the current k-th neighbor
 *       (worst distance in the intermediate result) to its candidate buffer.
 *     - When the candidate buffer is nearly full, it is merged with the
 *       intermediate top-k result using a warp-level bitonic sort over
 *       the 2k elements (previous top-k + candidates).
 * - After the last tile, any remaining candidates are merged.
 * - Distances are squared Euclidean (no sqrt).
 *
 * Memory layout in shared memory per block:
 *   [Tile data (float2) : TILE_SIZE]
 *   [Intermediate top-k buffers for WARPS_PER_BLOCK warps: WARPS_PER_BLOCK * MAX_K Neighbor]
 *   [Candidate buffers for WARPS_PER_BLOCK warps:          WARPS_PER_BLOCK * MAX_K Neighbor]
 *
 * Here, Neighbor is { float dist; int idx; }.
 */

struct Neighbor {
    float dist;
    int   idx;
};

// Tunable parameters for target GPU (H100/A100)
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;       // 4 warps -> 128 threads per block
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_SIZE         = 4096;    // points per tile loaded into shared memory
constexpr int MAX_K             = 1024;    // maximum supported k (given by problem statement)

/*
 * Warp-level merge of:
 *   - current intermediate top-k list (best[0..k-1])
 *   - candidate buffer (cand[0..k-1])
 *
 * Both best[] and cand[] reside in shared memory, and only the first k entries
 * are used. Any unused entries must contain "infinite" distance (FLT_MAX).
 *
 * The merge:
 *   - Treats best[0..k-1] and cand[0..k-1] as a virtual array of length 2k.
 *   - Performs a warp-level bitonic sort on that array (ascending by distance).
 *   - After sorting, the first k positions (virtual indices 0..k-1) correspond
 *     to the smallest k distances; these reside in best[0..k-1].
 *   - Updates worstDist = best[k-1].dist.
 *   - Resets cand[0..k-1] to (INF, -1) so the candidate buffer is empty.
 *
 * All threads in the warp participate; synchronization is done via __syncwarp().
 */
__device__ __forceinline__
void warp_merge_topk(Neighbor* best,
                     Neighbor* cand,
                     int       k,
                     float    &worstDist)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);

    const int n = 2 * k;  // virtual length = k (best) + k (cand)

    // Bitonic sort network over the virtual array of length n.
    // Virtual index range [0, n):
    //   0..k-1   -> best[0..k-1]
    //   k..2k-1  -> cand[0..k-1]
    for (int kStage = 2; kStage <= n; kStage <<= 1) {
        for (int j = kStage >> 1; j > 0; j >>= 1) {
            // Each thread processes indices in [0, n) with stride WARP_SIZE.
            for (int idx = laneId; idx < n; idx += WARP_SIZE) {
                int ixj = idx ^ j;
                if (ixj > idx && ixj < n) {
                    // Direction: up = true -> ascending, false -> descending within subsequence
                    bool up = ((idx & kStage) == 0);

                    Neighbor a = (idx < k) ? best[idx] : cand[idx - k];
                    Neighbor b = (ixj < k) ? best[ixj] : cand[ixj - k];

                    // Compare and swap depending on direction
                    bool doSwap = up ? (a.dist > b.dist) : (a.dist < b.dist);
                    if (doSwap) {
                        if (idx < k) best[idx]     = b;
                        else         cand[idx - k] = b;

                        if (ixj < k) best[ixj]     = a;
                        else         cand[ixj - k] = a;
                    }
                }
            }
            __syncwarp();
        }
    }

    // After sort, best[0..k-1] contain the k smallest distances of all 2k values.
    // Update worstDist to the k-th (last) element's distance.
    if (laneId == 0) {
        worstDist = best[k - 1].dist;
    }
    worstDist = __shfl_sync(FULL_MASK, worstDist, 0);

    // Reset candidate buffer (cand[0..k-1]) so it is empty.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        cand[i].dist = FLT_MAX;
        cand[i].idx  = -1;
    }
    __syncwarp();
}

/*
 * Kernel: compute k nearest neighbors for query points.
 *
 * Each warp (32 threads) processes one query:
 *   - warpIdInBlock: which warp inside the block
 *   - globalWarpId : which query this warp is assigned to
 *
 * Block-wide:
 *   - Tiles of TILE_SIZE data points are loaded into shared memory.
 *
 * Warp-wide (for each active warp):
 *   - For each tile:
 *       - For each data point in the tile, each lane processes a subset
 *         of the tile indices (striding by WARP_SIZE).
 *       - Each lane computes squared distance to its query.
 *       - If the distance is smaller than the current worst distance in
 *         the intermediate top-k list, it is added to the candidate buffer.
 *       - When the candidate buffer is close to full (within WARP_SIZE slots),
 *         the warp merges the candidates with the intermediate top-k.
 *   - After all tiles, remaining candidates are merged.
 *   - The final top-k list is written to global memory.
 */
__global__ void knn_kernel(const float2* __restrict__ query,
                           int                         query_count,
                           const float2* __restrict__ data,
                           int                         data_count,
                           std::pair<int, float>* __restrict__ result,
                           int                         k)
{
    extern __shared__ unsigned char shared_mem[];

    // Layout: [tile data][best arrays][candidate arrays]
    float2*  s_data = reinterpret_cast<float2*>(shared_mem);
    Neighbor* s_best = reinterpret_cast<Neighbor*>(s_data + TILE_SIZE);
    Neighbor* s_cand = s_best + WARPS_PER_BLOCK * MAX_K;

    const unsigned FULL_MASK = 0xffffffffu;

    const int threadId      = threadIdx.x;
    const int warpIdInBlock = threadId / WARP_SIZE;
    const int laneId        = threadId & (WARP_SIZE - 1);
    const int globalWarpId  = blockIdx.x * WARPS_PER_BLOCK + warpIdInBlock;

    // Determine if this warp is assigned a real query
    const bool warpActive = (globalWarpId < query_count);

    // Pointers to this warp's regions in shared memory
    Neighbor* warpBest = s_best + warpIdInBlock * MAX_K;
    Neighbor* warpCand = s_cand + warpIdInBlock * MAX_K;

    float qx = 0.0f;
    float qy = 0.0f;

    // Initialize per-warp state for active warps
    if (warpActive) {
        // Load query point once per warp and broadcast via shuffle
        if (laneId == 0) {
            float2 q = query[globalWarpId];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);

        // Initialize intermediate top-k and candidate buffers
        for (int i = laneId; i < k; i += WARP_SIZE) {
            warpBest[i].dist = FLT_MAX;
            warpBest[i].idx  = -1;
            warpCand[i].dist = FLT_MAX;
            warpCand[i].idx  = -1;
        }
        __syncwarp();
    }

    // Worst distance among current top-k (same for all lanes in the warp)
    float worstDist = FLT_MAX;
    // Number of valid entries in candidate buffer (same for all lanes)
    int candCount = 0;

    // Iterate over the dataset in tiles
    for (int tileBase = 0; tileBase < data_count; tileBase += TILE_SIZE) {
        int tileSize = data_count - tileBase;
        if (tileSize > TILE_SIZE) {
            tileSize = TILE_SIZE;
        }

        // Block-wide load of tile data into shared memory
        for (int idx = threadId; idx < tileSize; idx += blockDim.x) {
            s_data[idx] = data[tileBase + idx];
        }
        __syncthreads();

        // Process this tile for the current warp's query
        if (warpActive) {
            for (int localIdx = laneId; localIdx < tileSize; localIdx += WARP_SIZE) {
                // If candidate buffer is close to full, merge it with the intermediate result
                if (candCount >= k - WARP_SIZE && candCount > 0) {
                    warp_merge_topk(warpBest, warpCand, k, worstDist);
                    candCount = 0;
                }

                float2 p = s_data[localIdx];
                float dx = p.x - qx;
                float dy = p.y - qy;
                // Squared Euclidean distance
                float dist = fmaf(dy, dy, dx * dx);

                bool isCandidate = (dist < worstDist);

                // Warp-wide aggregation of candidate insertions
                unsigned int mask = __ballot_sync(FULL_MASK, isCandidate);
                int numNew = __popc(mask);

                if (numNew > 0) {
                    // baseIdx is the starting index in the candidate buffer
                    int baseIdx = candCount;

                    // Rank of this lane among the active candidate lanes
                    unsigned int laneMask = (laneId == 0) ? 0u : ((1u << laneId) - 1u);
                    int offset = __popc(mask & laneMask);

                    if (isCandidate) {
                        int globalDataIdx = tileBase + localIdx;
                        warpCand[baseIdx + offset].dist = dist;
                        warpCand[baseIdx + offset].idx  = globalDataIdx;
                    }

                    // Update candCount in lane 0 and broadcast
                    if (laneId == 0) {
                        candCount += numNew;
                    }
                    candCount = __shfl_sync(FULL_MASK, candCount, 0);
                }
            }
        }

        __syncthreads();
    }

    // Final merge of remaining candidates for active warps
    if (warpActive) {
        if (candCount > 0) {
            warp_merge_topk(warpBest, warpCand, k, worstDist);
            candCount = 0;
        }

        // Write the final top-k results for this query to global memory
        const int queryIdx = globalWarpId;
        const int outBase  = queryIdx * k;

        for (int i = laneId; i < k; i += WARP_SIZE) {
            Neighbor nb = warpBest[i];
            result[outBase + i].first  = nb.idx;
            result[outBase + i].second = nb.dist;
        }
    }
}

/*
 * Host wrapper:
 *   - Launches knn_kernel with appropriate grid/block configuration.
 *   - Uses dynamic shared memory sized for the maximum k (MAX_K).
 *
 * Assumptions:
 *   - k is a power of two, 32 <= k <= 1024 (MAX_K).
 *   - query, data, and result point to device memory allocated via cudaMalloc.
 */
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K) {
        return;
    }

    dim3 blockDim(THREADS_PER_BLOCK);
    int  numWarps = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(numWarps);

    // Shared memory size per block:
    //   TILE_SIZE float2 points +
    //   WARPS_PER_BLOCK * MAX_K Neighbor for intermediate top-k +
    //   WARPS_PER_BLOCK * MAX_K Neighbor for candidate buffers
    size_t sharedBytes =
        sizeof(float2) * TILE_SIZE +
        sizeof(Neighbor) * MAX_K * WARPS_PER_BLOCK * 2;

    knn_kernel<<<gridDim, blockDim, sharedBytes>>>(query,
                                                   query_count,
                                                   data,
                                                   data_count,
                                                   result,
                                                   k);
}