#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// -----------------------------------------------------------------------------
// Optimized CUDA k-NN (2D, squared Euclidean) implementation.
//
// Key design points:
//   * One warp (32 threads) computes k-NN for a single query point.
//   * Block size: 8 warps (256 threads).
//   * All data points are processed in tiles cached in shared memory.
//   * For each query/warp we maintain:
//       - A sorted "intermediate" result of size k in shared memory.
//       - A candidate buffer of size k in shared memory.
//       - A shared candidate counter updated with atomicAdd.
//       - A max_distance (distance of k-th neighbor) used for filtering.
//   * When candidate buffer is full (or at the end), we:
//       0. (Invariant) Intermediate array is sorted ascending.
//       1. Sort the candidate buffer with a bitonic sort (ascending).
//       2. Merge candidate and intermediate into a bitonic array by
//          taking min(intermediate[i], candidate[k-1-i]) element-wise.
//       3. Bitonic-sort the merged array to restore sorted intermediate.
//   * Bitonic sort and merge per query are done within a warp,
//     using shared memory and warp-synchronous programming.
// -----------------------------------------------------------------------------

// Tunable constants.
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_K             = 1024;   // per problem statement
constexpr int DATA_TILE_SIZE    = 1024;   // number of data points cached per block in shared memory

// Struct for internal neighbor representation (distance + index).
struct Neighbor {
    float dist;
    int   idx;
};

// Shared memory:
//   * dataTile: cached data points for the block
//   * inter[warp][k]: per-warp (per-query) intermediate result (sorted)
//   * cand[warp][k]:  per-warp (per-query) candidate buffer
//   * candCount[warp]: number of used entries in cand[warp]
__shared__ float2   s_dataTile[DATA_TILE_SIZE];
__shared__ Neighbor s_intermediate[WARPS_PER_BLOCK][MAX_K];
__shared__ Neighbor s_candidates [WARPS_PER_BLOCK][MAX_K];
__shared__ int      s_candidateCount[WARPS_PER_BLOCK];

// -----------------------------------------------------------------------------
// Warp-level bitonic sort for Neighbor arrays, ascending by dist.
// n must be a power of two and <= MAX_K.
//
// Parallelization:
//   * Each thread in the warp processes indices i = laneId, i += WARP_SIZE.
//   * __syncwarp is used to synchronize threads between compare-exchange steps.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void bitonicSortWarp(Neighbor *arr, int n, unsigned laneId)
{
    // Standard bitonic sort network translated to CUDA and parallelized over a warp.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = laneId; i < n; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    Neighbor a = arr[i];
                    Neighbor b = arr[ixj];
                    bool shouldSwap = (ascending ? (a.dist > b.dist) : (a.dist < b.dist));
                    if (shouldSwap) {
                        arr[i]  = b;
                        arr[ixj] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// -----------------------------------------------------------------------------
// Flush candidate buffer into intermediate result for a single warp/query.
//
// Steps:
//   0. (Assumption) s_intermediate[warp] is sorted ascending (invariant).
//   1. Pad candidate buffer [candCount..k-1] with (INF, -1) if not full.
//   2. Bitonic sort candidate buffer ascending.
//   3. Merge candidate and intermediate:
//        merged[i] = min( intermediate[i], candidates[k-1-i] )
//      producing a bitonic sequence stored back into intermediate.
//   4. Bitonic sort intermediate to restore ascending order.
//
// Parameters:
//   inter     - pointer to this warp's intermediate array (size MAX_K)
//   cand      - pointer to this warp's candidate array (size MAX_K)
//   k         - actual k (<= MAX_K, power of two)
//   candCount - number of valid candidates currently stored (<= k)
//   laneId    - lane index within warp
// -----------------------------------------------------------------------------
__device__ __forceinline__
void flushCandidateBuffer(Neighbor *inter, Neighbor *cand,
                          int k, int candCount, unsigned laneId)
{
    // 1. Pad candidate buffer with "infinite" distances if not full.
    for (int i = laneId + candCount; i < k; i += WARP_SIZE) {
        cand[i].dist = FLT_MAX;
        cand[i].idx  = -1;
    }
    __syncwarp();

    // 2. Sort candidate buffer ascending by distance.
    bitonicSortWarp(cand, k, laneId);
    __syncwarp();

    // 3. Merge candidate and intermediate into a bitonic sequence in 'inter'.
    //    merged[i] = min( inter[i], cand[k-1-i] )
    for (int i = laneId; i < k; i += WARP_SIZE) {
        Neighbor a = inter[i];
        Neighbor b = cand[k - 1 - i];
        inter[i] = (a.dist <= b.dist) ? a : b;
    }
    __syncwarp();

    // 4. Sort merged result (in inter) in ascending order.
    bitonicSortWarp(inter, k, laneId);
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Core k-NN kernel.
//
// Each warp handles a single query point:
//   - Loads query point into registers.
//   - Iterates over data points in tiles cached in shared memory.
//   - For each tile, each thread computes distances for multiple data points,
//     applying a warp-cooperative insertion into the candidate buffer.
//   - When candidate buffer is full, flushCandidateBuffer is called.
//   - After processing all tiles, if candidate buffer is non-empty, it is
//     flushed one last time.
//   - Final intermediate result is written to the global 'result' array.
// -----------------------------------------------------------------------------
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result,
                int k)
{
    const unsigned laneId       = threadIdx.x & (WARP_SIZE - 1);
    const unsigned warpIdInBlock = threadIdx.x / WARP_SIZE;
    const unsigned globalWarpId  = blockIdx.x * WARPS_PER_BLOCK + warpIdInBlock;

    if (globalWarpId >= static_cast<unsigned>(query_count)) {
        return;
    }

    Neighbor *inter = s_intermediate[warpIdInBlock];
    Neighbor *cand  = s_candidates[warpIdInBlock];
    int      &candCount = s_candidateCount[warpIdInBlock];

    // Initialize intermediate result for this query:
    //   distances = INF, indices = -1; sorted ascending by construction.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        inter[i].dist = FLT_MAX;
        inter[i].idx  = -1;
    }
    if (laneId == 0) {
        candCount = 0;
    }
    __syncwarp();

    // Initial max_distance (distance of k-th nearest neighbor in intermediate).
    // Since all are INF, initial max_distance is INF.
    float maxDist = FLT_MAX;

    // Load query point for this warp into registers.
    float2 q = query[globalWarpId];

    // Process data points in tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += DATA_TILE_SIZE) {
        int tileCount = data_count - tileStart;
        if (tileCount > DATA_TILE_SIZE) {
            tileCount = DATA_TILE_SIZE;
        }

        // Load current tile of data points into shared memory.
        // All threads in block cooperate.
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            s_dataTile[i] = data[tileStart + i];
        }
        __syncthreads();

        // Each warp processes all data points in this tile.
        // We iterate over tile in chunks of WARP_SIZE so that each thread
        // handles a unique data point in each iteration.
        const unsigned FULL_MASK = 0xFFFFFFFFu;

        for (int base = 0; base < tileCount; base += WARP_SIZE) {
            int idxInTile = base + laneId;
            bool valid    = (idxInTile < tileCount);

            float dist  = 0.0f;
            int   index = tileStart + idxInTile;

            if (valid) {
                float2 p = s_dataTile[idxInTile];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                // Squared Euclidean distance.
                dist = dx * dx + dy * dy;
            }

            // Candidate if valid and closer than current maxDist.
            bool isCandidate = valid && (dist < maxDist);
            bool pending     = isCandidate;

            // Warp-cooperative insertion into candidate buffer, with capacity k.
            while (true) {
                unsigned mask = __ballot_sync(FULL_MASK, pending);
                if (mask == 0) {
                    break; // no pending candidates left in this warp for this group
                }

                int pendingCount = __popc(mask);

                // Load current candidate count from shared memory and broadcast.
                int candCountCur = 0;
                if (laneId == 0) {
                    candCountCur = candCount;
                }
                candCountCur = __shfl_sync(FULL_MASK, candCountCur, 0);
                int avail = k - candCountCur;

                if (avail == 0) {
                    // Candidate buffer full: flush into intermediate.
                    flushCandidateBuffer(inter, cand, k, candCountCur, laneId);

                    // Reset candidate count and update maxDist = k-th neighbor distance.
                    if (laneId == 0) {
                        candCount = 0;
                    }
                    float newMaxDist = 0.0f;
                    if (laneId == 0) {
                        newMaxDist = inter[k - 1].dist;
                    }
                    newMaxDist = __shfl_sync(FULL_MASK, newMaxDist, 0);
                    maxDist = newMaxDist;

                    // Try again with same pending candidates and new buffer state.
                    continue;
                }

                int numInsert = (pendingCount < avail) ? pendingCount : avail;

                int basePos = 0;
                if (laneId == 0) {
                    // Use atomicAdd (as required) to reserve positions in candidate buffer.
                    basePos = atomicAdd(&candCount, numInsert);
                }
                basePos  = __shfl_sync(FULL_MASK, basePos, 0);
                numInsert = __shfl_sync(FULL_MASK, numInsert, 0);

                // Rank of this lane within the pending mask.
                int rank = __popc(mask & ((1u << laneId) - 1u));

                bool willInsert = pending && (rank < numInsert);
                if (willInsert) {
                    cand[basePos + rank].dist = dist;
                    cand[basePos + rank].idx  = index;
                    pending = false;
                }

                // If we filled the buffer exactly, flush it immediately.
                if (numInsert == avail) {
                    // After insertion, candCountCur + numInsert == k.
                    flushCandidateBuffer(inter, cand, k, candCountCur + numInsert, laneId);

                    if (laneId == 0) {
                        candCount = 0;
                    }
                    float newMaxDist = 0.0f;
                    if (laneId == 0) {
                        newMaxDist = inter[k - 1].dist;
                    }
                    newMaxDist = __shfl_sync(FULL_MASK, newMaxDist, 0);
                    maxDist = newMaxDist;
                }
                // Loop continues if any lanes still have pending candidates.
            }
        }

        __syncthreads(); // ensure all warps done before loading next tile
    }

    // After last tile: if candidate buffer is not empty, flush it.
    int finalCandCount = 0;
    if (laneId == 0) {
        finalCandCount = candCount;
    }
    finalCandCount = __shfl_sync(0xFFFFFFFFu, finalCandCount, 0);

    if (finalCandCount > 0) {
        flushCandidateBuffer(inter, cand, k, finalCandCount, laneId);
        if (laneId == 0) {
            candCount = 0;
        }
    }
    __syncwarp();

    // Write final k nearest neighbors for this query to global memory.
    // inter[0..k-1] is sorted ascending by distance.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        int outIdx = globalWarpId * k + i;
        result[outIdx].first  = inter[i].idx;
        result[outIdx].second = inter[i].dist;
    }
}

// -----------------------------------------------------------------------------
// Host interface as requested.
//
// query       - device pointer to query points (float2), size query_count
// query_count - number of query points
// data        - device pointer to data points (float2), size data_count
// data_count  - number of data points
// result      - device pointer to output array of size query_count * k
// k           - number of nearest neighbors (power of two, 32..1024)
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K) {
        return;
    }

    dim3 blockDim(THREADS_PER_BLOCK);
    int warpsPerBlock = WARPS_PER_BLOCK;
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    dim3 gridDim(numBlocks);

    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result, k);

    // In production code, it is good practice to check for errors, e.g.:
    // cudaDeviceSynchronize();
    // cudaGetLastError();
}