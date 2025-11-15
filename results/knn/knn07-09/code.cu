#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// ---------------------------------------------------------------------------
// Optimized k-NN kernel for 2D points.
// 
// Design highlights:
//
// * Each query is processed by exactly one warp (32 threads).
// * A thread block contains WARPS_PER_BLOCK warps and processes that many
//   queries concurrently.
// * The data points are processed in tiles and cached in shared memory.
//   All threads in the block cooperatively load each tile.
// * Each warp keeps:
//      - A private "intermediate result" (top-k neighbors) in the global
//        result[] array assigned to that query. This array is always kept
//        sorted in ascending order of distance.
//      - A private candidate buffer of size k in shared memory
//        (distances + indices) plus a shared candidate counter, updated via
//        atomicAdd as specified.
// * For each tile, each warp computes distances from its query to all data
//   points in that tile, filters them by a per-query max_distance (distance
//   of the current k-th neighbor), and inserts qualifying points into the
//   candidate buffer.
// * If the candidate buffer is full and new candidates need to be inserted,
//   the buffer is merged with the intermediate result:
//      0) Invariant: intermediate result is sorted ascending.
//      1) Sort candidate buffer ascending with a serial Bitonic Sort.
//      2) Merge buffer and intermediate result into a bitonic sequence:
//           merged[i] = min( intermediate[i], buffer[k-1-i] ).
//      3) Sort merged sequence ascending via Bitonic Sort, obtaining an
//         updated intermediate result (still of size k).
//   The k-th neighbor distance is then stored into max_distance.
// * Warp-level cooperation:
//      - Warp shuffles (__shfl_sync) are used to broadcast scalars (e.g. the
//        query coordinates, candidate buffer base index, candidate_count,
//        max_distance) from lane 0 to the rest of the warp.
//      - Warp barriers (__syncwarp) are used inside the per-warp merge
//        routine to guarantee consistent views of shared state.
// * Candidate insertion uses a warp-wide ballot/prefix scheme so that only
//   one atomicAdd per warp is required per "insertion round", yet every lane
//   capable of inserting a candidate gets its own slot in the shared buffer.
// * The Bitonic Sort used here is a straightforward serial reference
//   implementation, as allowed by the problem statement. It runs in O(k log^2 k)
//   time and is executed only by lane 0 of each warp, which is acceptable
//   because k <= 1024 and merges are relatively infrequent.
//
// Assumptions:
// * data_count >= k.
// * k is a power of two in [32, 1024].
// * query, data, and result are all device pointers allocated with cudaMalloc.
// * Typical usage has many queries and millions of data points.
// ---------------------------------------------------------------------------

constexpr int WARP_SIZE        = 32;
constexpr int MAX_K            = 1024;   // maximum allowed k
constexpr int WARPS_PER_BLOCK  = 8;      // one query per warp
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_SIZE        = THREADS_PER_BLOCK; // number of data points per tile

using Pair = std::pair<int, float>;

// ---------------------------------------------------------------------------
// Serial Bitonic Sort on arrays 'dist' (keys) and 'idx' (payload).
// The length n must be a power of two (here n == k).
// This implementation is executed by a single thread (lane 0).
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bitonic_sort_inplace(float* dist, int* idx, int n)
{
    // Reference-like implementation as per pseudocode in the problem statement.
    for (int k = 2; k <= n; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < n; ++i)
            {
                int l = i ^ j;
                if (l > i)
                {
                    bool ascending  = ((i & k) == 0);
                    bool shouldSwap = (dist[i] > dist[l]);
                    if (!ascending)
                    {
                        // Reverse comparison when sorting descending.
                        shouldSwap = !shouldSwap;
                    }
                    if (shouldSwap)
                    {
                        float td = dist[i];
                        dist[i]  = dist[l];
                        dist[l]  = td;

                        int ti = idx[i];
                        idx[i] = idx[l];
                        idx[l] = ti;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flush and merge a warp's candidate buffer with its intermediate result.
//
// candDist/candIdx : shared-memory arrays of size at least k for this warp.
// candCount        : shared-memory counter for this warp's candidate buffer.
// k                : number of neighbors.
// result           : global array of pairs; result[queryIndex * k .. +k-1]
//                    is the intermediate result for this query, sorted
//                    ascending by distance.
// queryIndex       : index of the current query handled by this warp.
// max_distance     : per-warp max_distance (distance of the k-th neighbor).
// warpMask         : active mask for this warp (here always 0xffffffff).
//
// Steps:
//  1) Pad the candidate buffer (if partially filled) with FLT_MAX distances
//     so that Bitonic Sort can work on exactly k elements.
//  2) Sort buffer ascending via Bitonic Sort.
//  3) Merge buffer and intermediate result into a length-k bitonic sequence:
//         merged[i] = min( best[i], buffer[k-1-i] ), by distance.
//     The merged sequence is kept in candDist/candIdx.
//  4) Sort this merged sequence ascending via Bitonic Sort.
//  5) Write the merged, sorted result back to result[] for this query.
//  6) Update max_distance to the distance of the k-th neighbor.
//  7) Reset the candidate counter to 0.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void flush_and_merge(float* candDist,
                     int*   candIdx,
                     int*   candCount,
                     int    k,
                     Pair*  result,
                     int    queryIndex,
                     float& max_distance,
                     unsigned warpMask)
{
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Only lane 0 performs the serial work; all lanes then synchronize.
    if (lane == 0)
    {
        int count = *candCount;

        if (count > 0)
        {
            // Pad unused slots with "infinite" distance so that we always sort k elements.
            for (int i = count; i < k; ++i)
            {
                candDist[i] = FLT_MAX;
                candIdx[i]  = -1;
            }

            // 1) Sort candidate buffer ascending.
            bitonic_sort_inplace(candDist, candIdx, k);

            // 2) Merge buffer and existing intermediate result into a bitonic sequence.
            int base = queryIndex * k;
            for (int i = 0; i < k; ++i)
            {
                float bestDist = result[base + i].second;
                int   bestIdx  = result[base + i].first;

                float bufDist  = candDist[k - 1 - i];
                int   bufIdx   = candIdx[k - 1 - i];

                if (bestDist <= bufDist)
                {
                    // Take from intermediate result.
                    candDist[i] = bestDist;
                    candIdx[i]  = bestIdx;
                }
                else
                {
                    // Take from buffer.
                    candDist[i] = bufDist;
                    candIdx[i]  = bufIdx;
                }
            }

            // 3) Sort merged bitonic sequence ascending.
            bitonic_sort_inplace(candDist, candIdx, k);

            // 4) Write merged top-k back to intermediate result.
            for (int i = 0; i < k; ++i)
            {
                result[base + i].first  = candIdx[i];
                result[base + i].second = candDist[i];
            }

            // 5) Update max_distance to the distance of the k-th nearest neighbor.
            max_distance = candDist[k - 1];

            // 6) Reset candidate counter.
            *candCount = 0;
        }
    }

    // Broadcast updated max_distance from lane 0 to the entire warp.
    max_distance = __shfl_sync(warpMask, max_distance, 0);

    // Ensure all lanes see the updated candidate_count and buffer state.
    __syncwarp(warpMask);
}

// ---------------------------------------------------------------------------
// GPU kernel: each warp processes one query and computes its k nearest
// neighbors among all data points.
// ---------------------------------------------------------------------------
__global__
void knn_kernel(const float2* __restrict__ query,
                int                           query_count,
                const float2* __restrict__ data,
                int                           data_count,
                Pair* __restrict__            result,
                int                           k)
{
    // Dynamic shared memory layout:
    //   [0                              .. TILE_SIZE * sizeof(float2)                  ) : data tile (float2)
    //   [TILE_SIZE * sizeof(float2)     .. + WARPS_PER_BLOCK*MAX_K*sizeof(float)       ) : candidate distances
    //   [prev + WARPS_PER_BLOCK*MAX_K*sizeof(float) .. + WARPS_PER_BLOCK*MAX_K*sizeof(int)) : candidate indices
    //   [prev + WARPS_PER_BLOCK*MAX_K*sizeof(int)   .. + WARPS_PER_BLOCK*sizeof(int)   ) : candidate counts
    extern __shared__ unsigned char smem[];

    unsigned char* ptr = smem;

    // Data tile shared by entire block.
    float2* sData = reinterpret_cast<float2*>(ptr);
    ptr += TILE_SIZE * sizeof(float2);

    // Candidate buffers per warp (distances then indices).
    float* sCandDistBase = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(float);

    int* sCandIdxBase = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(int);

    // Candidate counts per warp.
    int* sCandCount = reinterpret_cast<int*>(ptr);

    const int tid       = threadIdx.x;
    const int lane      = tid & (WARP_SIZE - 1);
    const int warpLocal = tid / WARP_SIZE;
    const int warpGlobal = blockIdx.x * WARPS_PER_BLOCK + warpLocal;
    const unsigned warpMask = 0xFFFFFFFFu;

    // Pointers to this warp's candidate buffer and counter.
    float* candDist = sCandDistBase + warpLocal * MAX_K;
    int*   candIdx  = sCandIdxBase  + warpLocal * MAX_K;
    int*   candCount = sCandCount   + warpLocal;

    // Initialize candidate count for all warps.
    if (lane == 0)
    {
        *candCount = 0;
    }

    // Determine if this warp is assigned a valid query.
    const bool warpValid = (warpGlobal < query_count);

    // Per-warp max_distance: distance of the current k-th nearest neighbor.
    // Initialize to infinity so that all points are initially considered.
    float max_distance = FLT_MAX;

    // Initialize intermediate result (result[warpGlobal * k .. +k-1]) for valid warps.
    if (warpValid && lane == 0)
    {
        const int base = warpGlobal * k;
        for (int i = 0; i < k; ++i)
        {
            result[base + i].first  = -1;
            result[base + i].second = FLT_MAX;
        }
    }
    __syncwarp(warpMask);

    // Load query coordinates into registers and broadcast within warp.
    float2 q;
    if (warpValid)
    {
        if (lane == 0)
        {
            q = query[warpGlobal];
        }
        q.x = __shfl_sync(warpMask, q.x, 0);
        q.y = __shfl_sync(warpMask, q.y, 0);
    }

    // Process all data points in tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE)
    {
        int tileSize = TILE_SIZE;
        if (tileStart + tileSize > data_count)
        {
            tileSize = data_count - tileStart;
        }

        // Load tile of data points into shared memory.
        if (tid < tileSize)
        {
            sData[tid] = data[tileStart + tid];
        }

        __syncthreads(); // ensure tile is fully loaded

        // Every warp processes all points in the current tile.
        const int numIters = (tileSize + WARP_SIZE - 1) / WARP_SIZE;

        for (int it = 0; it < numIters; ++it)
        {
            const int localIndex = it * WARP_SIZE + lane;

            float dist     = 0.0f;
            int   data_idx = -1;
            bool  pending  = false;

            // Compute distance for this lane's data point, if any and if warp is valid.
            if (warpValid && localIndex < tileSize)
            {
                const float2 p = sData[localIndex];
                const float dx = p.x - q.x;
                const float dy = p.y - q.y;
                dist           = dx * dx + dy * dy;
                data_idx       = tileStart + localIndex;

                // Filter by current k-th neighbor distance.
                pending = (dist < max_distance);
            }

            // Warp-synchronous candidate insertion loop.
            // This loop may call flush_and_merge() when the candidate buffer fills up,
            // and will retry insertion for any still-pending candidates afterward.
            while (__any_sync(warpMask, pending))
            {
                // Mask of lanes that still have a pending candidate.
                unsigned pendingMask = __ballot_sync(warpMask, pending);
                if (pendingMask == 0u)
                {
                    break;
                }

                // Current number of candidates in buffer.
                int curCount = 0;
                if (lane == 0)
                {
                    curCount = *candCount;
                }
                curCount = __shfl_sync(warpMask, curCount, 0);

                const int freeSlots = k - curCount;

                if (freeSlots == 0)
                {
                    // Buffer is full; merge candidates with the intermediate result.
                    // Only meaningful for valid warps; for invalid warps, pending is always false.
                    if (warpValid)
                    {
                        flush_and_merge(candDist,
                                        candIdx,
                                        candCount,
                                        k,
                                        result,
                                        warpGlobal,
                                        max_distance,
                                        warpMask);
                    }

                    // Re-evaluate this candidate's eligibility with updated max_distance.
                    if (pending)
                    {
                        pending = (dist < max_distance);
                    }

                    continue;
                }

                // Number of lanes currently pending in this round.
                const int totalPending = __popc(pendingMask);
                int allowed = totalPending;

                // Only 'allowed' pending candidates can be inserted in this round.
                if (lane == 0 && allowed > freeSlots)
                {
                    allowed = freeSlots;
                }
                allowed = __shfl_sync(warpMask, allowed, 0);

                // Compute rank of this lane among pending lanes (0-based).
                const int myRank =
                    __popc(pendingMask & ((1u << lane) - 1));

                if (pending && myRank < allowed)
                {
                    // This lane will insert its candidate in this round.
                    int base = 0;
                    if (lane == 0)
                    {
                        // Atomic add to determine starting index for this batch.
                        base = atomicAdd(candCount, allowed);
                    }
                    base = __shfl_sync(warpMask, base, 0);

                    const int pos = base + myRank;
                    candDist[pos] = dist;
                    candIdx[pos]  = data_idx;

                    // Candidate successfully inserted.
                    pending = false;
                }

                // Lanes that are still 'pending' (myRank >= allowed) will loop again.
            }
        }

        __syncthreads(); // ensure all warps finished with this tile before loading next
    }

    // After processing all data points, flush any remaining candidates.
    if (warpValid)
    {
        flush_and_merge(candDist,
                        candIdx,
                        candCount,
                        k,
                        result,
                        warpGlobal,
                        max_distance,
                        warpMask);
    }
}

// ---------------------------------------------------------------------------
// Host interface.
// 
// query       : device pointer to query_count float2 points.
// data        : device pointer to data_count float2 points.
// result      : device pointer to query_count * k pairs (index, distance).
// query_count : number of query points.
// data_count  : number of data points.
// k           : number of neighbors (power of two, 32 <= k <= 1024).
//
// This function launches the knn_kernel with a configuration chosen for
// modern data center GPUs (such as A100/H100) and uses dynamic shared memory
// sized for the maximum supported k (MAX_K).
// ---------------------------------------------------------------------------
void run_knn(const float2* query,
             int           query_count,
             const float2* data,
             int           data_count,
             Pair*         result,
             int           k)
{
    // Guard against degenerate cases; nothing to do.
    if (query_count <= 0 || data_count <= 0 || k <= 0)
    {
        return;
    }

    // Clamp k to MAX_K for safety (problem guarantees k <= MAX_K).
    if (k > MAX_K)
    {
        k = MAX_K;
    }

    const dim3 blockDim(THREADS_PER_BLOCK);
    const int  warpsPerBlock = WARPS_PER_BLOCK;
    const dim3 gridDim( (query_count + warpsPerBlock - 1) / warpsPerBlock );

    // Dynamic shared memory size (in bytes), sized for MAX_K.
    const size_t sharedBytes =
        TILE_SIZE * sizeof(float2) +                 // sData
        WARPS_PER_BLOCK * MAX_K * sizeof(float) +    // sCandDistBase
        WARPS_PER_BLOCK * MAX_K * sizeof(int) +      // sCandIdxBase
        WARPS_PER_BLOCK * sizeof(int);               // sCandCount

    knn_kernel<<<gridDim, blockDim, sharedBytes>>>(query,
                                                   query_count,
                                                   data,
                                                   data_count,
                                                   result,
                                                   k);

    // Synchronize to ensure completion before the caller uses 'result'.
    cudaDeviceSynchronize();
}