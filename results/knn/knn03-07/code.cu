#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

/*
 * Optimized k-NN (k-nearest neighbors) for 2D points on NVIDIA GPUs.
 *
 * Design overview
 * ---------------
 * - One warp (32 threads) is responsible for computing the k nearest neighbors
 *   of a single query point.
 * - Each thread block contains WARPS_PER_BLOCK warps; thus each block processes
 *   WARPS_PER_BLOCK queries in parallel.
 * - The data set is processed in tiles. Each tile is cooperatively loaded into
 *   shared memory by all threads in the block, and then all warps in the block
 *   reuse that tile for their own query point.
 * - For each query, the warp maintains a streaming top-k structure:
 *     * The global top-k list (k distances + indices) is distributed across
 *       the 32 threads: each thread holds k/32 entries (k is guaranteed to be
 *       a power of two between 32 and 1024, so k/32 is an integer in [1, 32]).
 *     * The list is initialized with "empty" entries and filled by streaming
 *       over all data points. Once the list is full, candidates are only
 *       inserted if they are better than the current worst element.
 *     * To decide whether a candidate qualifies, the warp uses an approximate
 *       threshold (an upper bound on the current worst distance) for a fast
 *       reject, and a warp-wide reduction to find the exact worst when needed.
 * - Distances are computed in parallel:
 *     * For each tile, the warp processes points in batches of 32: each lane
 *       computes the distance to one point, then the 32 candidates are
 *       sequentially inserted into the warp's shared top-k structure using
 *       warp shuffles.
 * - After all data points are processed:
 *     * Each warp writes its distributed top-k list into a contiguous region
 *       of shared memory.
 *     * Lane 0 of each warp performs a simple selection sort on its k entries
 *       (O(k^2), but k <= 1024 and done once per query).
 *     * All lanes then write the sorted results to global memory in a
 *       coalesced manner.
 *
 * Notes
 * -----
 * - No additional device memory is allocated; only registers and shared memory
 *   are used.
 * - k must be a power of two between 32 and 1024 (inclusive) and data_count >= k.
 * - The result is sorted in ascending order of distance for each query.
 */

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;    // Number of warps (queries) per block
constexpr int BLOCK_THREADS     = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_SIZE         = 1024; // Number of data points cached per tile
constexpr int MAX_K             = 1024; // Maximum supported k
constexpr int MAX_LOCAL_K       = MAX_K / WARP_SIZE; // 1024 / 32 = 32

// Warp-synchronous insertion of a candidate into the distributed top-k list.
__device__ __forceinline__
void warp_insert_candidate(
    float candidateDist,
    int   candidateIndex,
    float *best_dists,     // per-thread array of size MAX_LOCAL_K
    int   *best_indices,   // per-thread array of size MAX_LOCAL_K
    int   k,
    int   local_k_per_lane,
    int   laneId,
    unsigned int warpMask,
    int   &fillCount,
    float &approxMax)
{
    // Fast path while we have not yet filled k entries:
    // we simply fill slots sequentially without any comparisons.
    int fc = __shfl_sync(warpMask, fillCount, 0);
    if (fc < k) {
        int insertPos   = fc;                     // 0 .. k-1
        int ownerLane   = insertPos & (WARP_SIZE - 1); // insertPos % 32
        int ownerOffset = insertPos >> 5;         // insertPos / 32

        if (laneId == ownerLane) {
            best_dists[ownerOffset]   = candidateDist;
            best_indices[ownerOffset] = candidateIndex;
        }

        if (laneId == 0) {
            fillCount = fc + 1;
            // We do not set approxMax here; it will be set when we first
            // compute the exact worst element during a heavy-path insertion.
        }
        return;
    }

    // At this point, we have already inserted k elements; we maintain the top-k.
    // First, apply a fast reject using an approximate threshold.
    float am = __shfl_sync(warpMask, approxMax, 0);
    if (candidateDist >= am) {
        // Candidate cannot be better than the current worst; discard.
        return;
    }

    // Heavy path: recompute the exact worst element across the k entries.

    // Each lane scans its own local_k_per_lane entries to find its local maximum.
    float localMax       = -CUDART_INF_F;
    int   localMaxOffset = 0;

    for (int i = 0; i < local_k_per_lane; ++i) {
        float v = best_dists[i];
        if (v > localMax) {
            localMax       = v;
            localMaxOffset = i;
        }
    }

    // Encode the local maximum slot as a global slot index in [0, k-1].
    int localMaxSlot = (localMaxOffset << 5) + laneId; // offset * 32 + laneId

    float maxDist = localMax;
    int   maxSlot = localMaxSlot;

    // Warp-wide reduction to find the global maximum distance and its slot.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float otherDist = __shfl_down_sync(warpMask, maxDist, offset);
        int   otherSlot = __shfl_down_sync(warpMask, maxSlot, offset);
        if (otherDist > maxDist) {
            maxDist = otherDist;
            maxSlot = otherSlot;
        }
    }

    // Broadcast the global maximum distance and slot to all lanes.
    float globalMaxDist = __shfl_sync(warpMask, maxDist, 0);
    int   globalMaxSlot = __shfl_sync(warpMask, maxSlot, 0);

    // Update approximate threshold in lane 0; it is an upper bound on
    // the current worst distance (it may be slightly larger after replacement,
    // which is still safe for pruning).
    if (laneId == 0) {
        approxMax = globalMaxDist;
    }

    // If the candidate is not better than the exact worst, discard it.
    if (candidateDist >= globalMaxDist) {
        return;
    }

    // Replace the global worst element with the candidate.
    int ownerLane   = globalMaxSlot & (WARP_SIZE - 1);
    int ownerOffset = globalMaxSlot >> 5;

    if (laneId == ownerLane && ownerOffset < local_k_per_lane) {
        best_dists[ownerOffset]   = candidateDist;
        best_indices[ownerOffset] = candidateIndex;
    }
}

// CUDA kernel implementing k-NN for 2D points.
__global__
void knn_kernel_2d(
    const float2 *__restrict__ query,
    int                      query_count,
    const float2 *__restrict__ data,
    int                      data_count,
    int                      k,
    std::pair<int, float> *__restrict__ result)
{
    // Shared memory:
    // - sh_data:  tile of data points reused by all warps in the block.
    // - sh_top_dists / sh_top_indices: per-warp buffers for the final top-k
    //   results (used for sorting and writing out).
    __shared__ float2 sh_data[TILE_SIZE];
    __shared__ float  sh_top_dists[WARPS_PER_BLOCK * MAX_K];
    __shared__ int    sh_top_indices[WARPS_PER_BLOCK * MAX_K];

    const int tid    = threadIdx.x;
    const int laneId = tid & (WARP_SIZE - 1); // 0..31
    const int warpId = tid >> 5;             // 0..(WARPS_PER_BLOCK-1)

    const int globalWarpId = blockIdx.x * WARPS_PER_BLOCK + warpId;
    const int queryId      = globalWarpId;   // one warp per query

    // Some warps in the last block may not correspond to a valid query.
    const bool warpActive = (queryId < query_count);

    // Load the query point (or a dummy value for inactive warps).
    float2 q;
    if (warpActive) {
        q = query[queryId];
    } else {
        q.x = 0.0f;
        q.y = 0.0f;
    }

    // k is guaranteed to be a power of two between 32 and 1024; thus
    // local_k_per_lane is an integer in [1, 32].
    const int local_k_per_lane = k / WARP_SIZE;

    // Per-thread portion of the distributed top-k list.
    float best_dists[MAX_LOCAL_K];
    int   best_indices[MAX_LOCAL_K];

    // Initialize local top-k storage with "infinite" distances.
    for (int i = 0; i < MAX_LOCAL_K; ++i) {
        best_dists[i]   = CUDART_INF_F;
        best_indices[i] = -1;
    }

    // Shared warp-wide state (stored per thread, but lane 0 acts as master).
    int   fillCount = 0;           // number of entries actually filled, up to k
    float approxMax = CUDART_INF_F; // approximate upper bound on current worst

    const unsigned int warpMask = __activemask();

    // Process the data set in tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        const int remaining = data_count - tileStart;
        const int tileSize  = (remaining < TILE_SIZE) ? remaining : TILE_SIZE;

        // Cooperative load of the tile into shared memory.
        for (int i = tid; i < tileSize; i += blockDim.x) {
            sh_data[i] = data[tileStart + i];
        }
        __syncthreads();

        // Process the tile in batches of 32 points per warp.
        for (int base = 0; base < tileSize; base += WARP_SIZE) {
            const int j = base + laneId;

            float dist;
            int   dataIndex;

            if (j < tileSize && warpActive) {
                const float2 dp = sh_data[j];
                const float dx  = dp.x - q.x;
                const float dy  = dp.y - q.y;
                // Squared Euclidean distance (using FMA where possible).
                dist      = fmaf(dy, dy, dx * dx);
                dataIndex = tileStart + j;
            } else {
                // Inactive or out-of-range lanes generate dummy candidates
                // that will be ignored.
                dist      = CUDART_INF_F;
                dataIndex = -1;
            }

            // Broadcast and insert the 32 candidates of this batch one by one.
            for (int srcLane = 0; srcLane < WARP_SIZE; ++srcLane) {
                float cdist  = __shfl_sync(warpMask, dist, srcLane);
                int   cindex = __shfl_sync(warpMask, dataIndex, srcLane);

                if (cindex < 0) {
                    // Dummy candidate; skip.
                    continue;
                }

                // All threads in the warp cooperatively update the top-k list
                // with the candidate (cdist, cindex).
                warp_insert_candidate(
                    cdist,
                    cindex,
                    best_dists,
                    best_indices,
                    k,
                    local_k_per_lane,
                    laneId,
                    warpMask,
                    fillCount,
                    approxMax
                );
            }
        }

        __syncthreads();
    }

    // Each warp writes its distributed top-k list into a contiguous region
    // of shared memory, then sorts it and writes it out to global memory.

    const int warpBase = warpId * MAX_K;

    // Scatter from per-thread storage to shared, assembling k entries per warp.
    for (int offset = 0; offset < local_k_per_lane; ++offset) {
        const int globalSlot = (offset << 5) + laneId; // offset * 32 + laneId
        if (globalSlot < k) {
            sh_top_dists[warpBase + globalSlot]   = best_dists[offset];
            sh_top_indices[warpBase + globalSlot] = best_indices[offset];
        }
    }

    __syncthreads();

    // Lane 0 of each active warp performs a simple selection sort on its k
    // entries in shared memory to order neighbors by increasing distance.
    if (warpActive && laneId == 0) {
        float *dists = &sh_top_dists[warpBase];
        int   *idxs  = &sh_top_indices[warpBase];

        for (int i = 0; i < k - 1; ++i) {
            int   minPos = i;
            float minVal = dists[i];

            for (int j = i + 1; j < k; ++j) {
                float v = dists[j];
                if (v < minVal) {
                    minVal = v;
                    minPos = j;
                }
            }

            if (minPos != i) {
                float tmpd = dists[i];
                dists[i]   = dists[minPos];
                dists[minPos] = tmpd;

                int tmpi   = idxs[i];
                idxs[i]    = idxs[minPos];
                idxs[minPos] = tmpi;
            }
        }
    }

    __syncthreads();

    // All lanes of each active warp write the sorted results to global memory.
    if (warpActive) {
        float *dists = &sh_top_dists[warpBase];
        int   *idxs  = &sh_top_indices[warpBase];

        for (int j = laneId; j < k; j += WARP_SIZE) {
            const int outPos = queryId * k + j;
            result[outPos].first  = idxs[j];
            result[outPos].second = dists[j];
        }
    }
}

// Host-side interface as specified.
void run_knn(
    const float2 *query,
    int           query_count,
    const float2 *data,
    int           data_count,
    std::pair<int, float> *result,
    int           k)
{
    // Assumptions based on problem statement:
    // - data_count >= k
    // - k is a power of two between 32 and 1024 (inclusive).
    // - query, data, and result are device pointers (allocated with cudaMalloc).

    const dim3 blockDim(BLOCK_THREADS);
    const int  numWarps  = query_count; // one warp per query
    const int  gridX     = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const dim3 gridDim(gridX);

    knn_kernel_2d<<<gridDim, blockDim>>>(
        query,
        query_count,
        data,
        data_count,
        k,
        result
    );
}
