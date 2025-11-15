#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// The kernel implements batched k-NN search for 2D points using one warp (32 threads)
// per query. Each warp maintains its own intermediate top-k result distributed across
// threads and a shared-memory buffer of size k for candidate neighbors.
//
// High-level algorithm per query (warp):
//   - Maintain distributed arrays of k best distances and indices (intermediate result).
//   - Maintain a shared-memory buffer of up to k candidate neighbors.
//   - Process the dataset in tiles cached in shared memory.
//   - For each tile, each lane computes distances to a subset of points.
//   - If a point is closer than the current k-th neighbor (worst distance), it is added
//     to the candidate buffer (warp-synchronously).
//   - When the candidate buffer would overflow, it is merged into the intermediate
//     result using all threads in the warp; the worst distance is updated.
//   - After all tiles, any remaining candidates are merged.
//   - Finally, the intermediate result is sorted (via repeated warp-wide selection of
//     the minimum) and written to the output array as std::pair<int,float>.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters for the target hardware (A100/H100).
// We use 4 warps (128 threads) per block and a data tile of 4096 points.
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int MAX_K = 1024;              // upper bound given in the problem
static constexpr int MAX_SLOTS_PER_LANE = MAX_K / WARP_SIZE; // 32
static constexpr int TILE_SIZE = 4096;

// Merge the candidate buffer (in shared memory) for a single warp into its
// intermediate top-k result (distributed across warp lanes).
//
// Each lane holds `slots` elements of the top-k arrays in registers:
//   topDist[0..slots-1], topIdx[0..slots-1].
//
// The candidate buffer for this warp is candDistWarp[0..bufCount-1] and
// candIdxWarp[0..bufCount-1] in shared memory.
//
// The algorithm processes candidates sequentially. For each candidate that passes
// the current "worst distance" threshold, all lanes cooperate to find the
// current worst element in the top-k and replace it if the candidate is better.
// After all candidates are processed, the worst distance is recomputed.
__device__ __forceinline__
void merge_candidate_buffer_warp(int laneId,
                                 float *topDist,
                                 int   *topIdx,
                                 int    slots,
                                 float *candDistWarp,
                                 int   *candIdxWarp,
                                 int    bufCount,
                                 int    k,
                                 float &worstDist)
{
    const unsigned FULL_MASK = 0xffffffffu;

    // Process each candidate sequentially, using all 32 lanes cooperatively.
    for (int j = 0; j < bufCount; ++j) {
        // Load candidate (lane 0 loads from shared memory, then broadcast).
        float candD = 0.0f;
        int   candI = -1;
        if (laneId == 0) {
            candD = candDistWarp[j];
            candI = candIdxWarp[j];
        }
        candD = __shfl_sync(FULL_MASK, candD, 0);
        candI = __shfl_sync(FULL_MASK, candI, 0);

        // Filter by current threshold; if not better, skip.
        if (candD >= worstDist) {
            continue;
        }

        // Find the current worst (maximum distance) among the top-k entries.
        float localMax = topDist[0];
        int   localSlot = 0;
        for (int s = 1; s < slots; ++s) {
            float v = topDist[s];
            if (v > localMax) {
                localMax  = v;
                localSlot = s;
            }
        }

        // Warp-wide reduction to find global maximum distance and its location
        // (lane index and slot index within that lane).
        float maxDist = localMax;
        int   maxLane = laneId;
        int   maxSlot = localSlot;

        // Binary tree reduction within the warp.
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float otherDist = __shfl_down_sync(FULL_MASK, maxDist, offset);
            int   otherLane = __shfl_down_sync(FULL_MASK, maxLane, offset);
            int   otherSlot = __shfl_down_sync(FULL_MASK, maxSlot, offset);

            if (otherDist > maxDist) {
                maxDist = otherDist;
                maxLane = otherLane;
                maxSlot = otherSlot;
            }
        }

        // Broadcast the global worst element information.
        maxDist = __shfl_sync(FULL_MASK, maxDist, 0);
        maxLane = __shfl_sync(FULL_MASK, maxLane, 0);
        maxSlot = __shfl_sync(FULL_MASK, maxSlot, 0);

        // If candidate is better than current worst, replace it.
        if (candD < maxDist) {
            if (laneId == maxLane) {
                topDist[maxSlot] = candD;
                topIdx[maxSlot]  = candI;
            }
        }
        __syncwarp(FULL_MASK);
    }

    // Recompute the true worst distance among the updated top-k entries.
    float localMax2 = topDist[0];
    for (int s = 1; s < slots; ++s) {
        float v = topDist[s];
        if (v > localMax2) {
            localMax2 = v;
        }
    }

    float globalMax = localMax2;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, globalMax, offset);
        if (other > globalMax) {
            globalMax = other;
        }
    }
    globalMax = __shfl_sync(FULL_MASK, globalMax, 0);

    // Update the shared threshold (same in all lanes).
    worstDist = globalMax;
}

// After all data points have been processed and the intermediate top-k result
// is available in distributed arrays (topDist/topIdx), produce a sorted list
// of neighbors for this query (ascending by distance) and write them into
// the result array.
//
// We implement a warp-wide selection sort: for each output position 0..k-1,
// all lanes cooperate to find the global minimum among the remaining entries,
// write it to output, and mark it as used by setting its distance to FLT_MAX.
__device__ __forceinline__
void write_sorted_results_warp(int laneId,
                               float *topDist,
                               int   *topIdx,
                               int    slots,
                               int    k,
                               std::pair<int, float> *resultBase)
{
    const unsigned FULL_MASK = 0xffffffffu;

    for (int outPos = 0; outPos < k; ++outPos) {
        // Find local minimum in this lane.
        float localMin  = topDist[0];
        int   localSlot = 0;
        for (int s = 1; s < slots; ++s) {
            float v = topDist[s];
            if (v < localMin) {
                localMin  = v;
                localSlot = s;
            }
        }

        // Warp-wide reduction to find global minimum (distance + location).
        float minDist = localMin;
        int   minLane = laneId;
        int   minSlot = localSlot;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float otherDist = __shfl_down_sync(FULL_MASK, minDist, offset);
            int   otherLane = __shfl_down_sync(FULL_MASK, minLane, offset);
            int   otherSlot = __shfl_down_sync(FULL_MASK, minSlot, offset);

            if (otherDist < minDist) {
                minDist = otherDist;
                minLane = otherLane;
                minSlot = otherSlot;
            }
        }

        // Broadcast the global minimum information.
        minDist = __shfl_sync(FULL_MASK, minDist, 0);
        minLane = __shfl_sync(FULL_MASK, minLane, 0);
        minSlot = __shfl_sync(FULL_MASK, minSlot, 0);

        // Lane that owns the minimum provides its index.
        int minIndexLane = 0;
        if (laneId == minLane) {
            minIndexLane = topIdx[minSlot];
        }
        int minIndex = __shfl_sync(FULL_MASK, minIndexLane, minLane);

        // Lane 0 writes the result for this position.
        if (laneId == 0) {
            resultBase[outPos].first  = minIndex;
            resultBase[outPos].second = minDist;
        }

        // Mark the selected element as used by setting its distance to +inf.
        if (laneId == minLane) {
            topDist[minSlot] = FLT_MAX;
        }

        __syncwarp(FULL_MASK);
    }
}

// Main KNN kernel: one warp per query.
template <int WarpsPerBlock, int TileSize, int MaxK>
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> * __restrict__ result,
                           int k)
{
    const unsigned FULL_MASK = 0xffffffffu;

    // Identify warp and lane within the block and grid.
    int threadId      = threadIdx.x;
    int warpIdInBlock = threadId / WARP_SIZE;
    int laneId        = threadId & (WARP_SIZE - 1);
    int globalWarpId  = blockIdx.x * WarpsPerBlock + warpIdInBlock;

    if (globalWarpId >= query_count)
        return;

    // Number of top-k elements held by each lane (k is always a power of two and
    // between 32 and 1024, so this division is exact and >= 1).
    int slots = k / WARP_SIZE;

    // Dynamic shared memory layout:
    //   [0 .. TileSize-1]                    : float2 sharedData[]
    //   [TileSize .. TileSize + WarpsPerBlock*MaxK - 1] : int candIdx[]
    //   [.. + WarpsPerBlock*MaxK - 1]        : float candDist[]
    extern __shared__ unsigned char sharedRaw[];
    float2 *sharedData = reinterpret_cast<float2 *>(sharedRaw);

    int *candIdxAll = reinterpret_cast<int *>(
        sharedData + TileSize);
    float *candDistAll = reinterpret_cast<float *>(
        candIdxAll + WarpsPerBlock * MaxK);

    // Pointers to candidate buffer for this warp.
    int   warpBufBase = warpIdInBlock * MaxK;
    int  *candIdxWarp = candIdxAll  + warpBufBase;
    float *candDistWarp = candDistAll + warpBufBase;

    // Per-lane intermediate top-k arrays in registers.
    float topDist[MAX_SLOTS_PER_LANE];
    int   topIdx [MAX_SLOTS_PER_LANE];

    // Initialize intermediate result with "infinite" distances.
    for (int i = 0; i < slots; ++i) {
        topDist[i] = FLT_MAX;
        topIdx[i]  = -1;
    }

    // Current k-th nearest neighbor distance (same in all lanes).
    float worstDist = FLT_MAX;

    // Candidate buffer count (only lane 0 holds the authoritative value).
    int candidateCount = 0;
    if (laneId != 0) {
        candidateCount = 0; // value ignored; lane 0 is the source when shuffling.
    }

    // Load this warp's query point; lane 0 reads from global memory, then broadcast.
    float qx = 0.0f, qy = 0.0f;
    if (laneId == 0) {
        float2 q = query[globalWarpId];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Pointer to this query's result segment.
    std::pair<int, float> *resultBase = result + static_cast<size_t>(globalWarpId) * k;

    // Process the dataset in tiles cached in shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TileSize) {
        int tileSize = TileSize;
        if (tileStart + tileSize > data_count) {
            tileSize = data_count - tileStart;
        }

        // Block-wide cooperative load of the current tile.
        for (int idx = threadIdx.x; idx < tileSize; idx += blockDim.x) {
            sharedData[idx] = data[tileStart + idx];
        }
        __syncthreads();

        // Each warp processes all points in the tile; lanes stride over the tile.
        for (int idx = laneId; idx < tileSize; idx += WARP_SIZE) {
            float2 p = sharedData[idx];

            float dx = qx - p.x;
            float dy = qy - p.y;
            float dist = dx * dx + dy * dy;
            int   dataIdx = tileStart + idx;

            // Check against current worst distance threshold.
            bool isCandidate = (dist < worstDist);

            // Warp-wide mask of lanes that produced a candidate.
            unsigned mask = __ballot_sync(FULL_MASK, isCandidate);
            int num = __popc(mask);
            if (num == 0)
                continue;

            // Lane 0 checks if the buffer would overflow and triggers a merge if needed.
            int needFlush = 0;
            int bufCount  = __shfl_sync(FULL_MASK, candidateCount, 0);
            if (laneId == 0) {
                if (bufCount + num > k) {
                    needFlush = 1;
                }
            }
            needFlush = __shfl_sync(FULL_MASK, needFlush, 0);

            if (needFlush) {
                bufCount = __shfl_sync(FULL_MASK, candidateCount, 0);
                merge_candidate_buffer_warp(laneId,
                                            topDist,
                                            topIdx,
                                            slots,
                                            candDistWarp,
                                            candIdxWarp,
                                            bufCount,
                                            k,
                                            worstDist);
                if (laneId == 0) {
                    candidateCount = 0;
                }
                __syncwarp(FULL_MASK);
                bufCount = 0;
            }

            // Reserve space in the candidate buffer for this batch of candidates.
            int baseIdx = 0;
            if (laneId == 0) {
                baseIdx = candidateCount;
                candidateCount += num;
            }
            baseIdx = __shfl_sync(FULL_MASK, baseIdx, 0);

            // Each lane with a candidate writes it into its assigned slot in the buffer.
            if (isCandidate) {
                unsigned maskBefore = mask & ((1u << laneId) - 1u);
                int offset = __popc(maskBefore);
                int bufIdx = baseIdx + offset;

                candDistWarp[bufIdx] = dist;
                candIdxWarp[bufIdx]  = dataIdx;
            }
        }

        __syncthreads(); // Ensure all warps are done with this tile before reusing sharedData.
    }

    // After the last tile, merge any remaining candidates.
    int finalBufCount = __shfl_sync(FULL_MASK, candidateCount, 0);
    if (finalBufCount > 0) {
        merge_candidate_buffer_warp(laneId,
                                    topDist,
                                    topIdx,
                                    slots,
                                    candDistWarp,
                                    candIdxWarp,
                                    finalBufCount,
                                    k,
                                    worstDist);
        if (laneId == 0) {
            candidateCount = 0;
        }
        __syncwarp(FULL_MASK);
    }

    // Finally, sort the intermediate result and write it to global memory.
    write_sorted_results_warp(laneId,
                              topDist,
                              topIdx,
                              slots,
                              k,
                              resultBase);
}

// Host-side wrapper that configures and launches the kernel.
// Assumes that `query`, `data`, and `result` are device pointers allocated
// with cudaMalloc, and that data_count >= k and k is a power of two
// between 32 and 1024 inclusive.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Basic parameter checks (optional, can be omitted in production).
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K)
        return;
    if ((k & (WARP_SIZE - 1)) != 0)
        return; // k must be a multiple of warp size (guaranteed by problem).

    dim3 blockDim(THREADS_PER_BLOCK);
    int numWarps = query_count;
    int gridDim  = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Compute required shared memory size.
    size_t sharedBytes =
        sizeof(float2) * TILE_SIZE +
        sizeof(int)    * WARPS_PER_BLOCK * MAX_K +
        sizeof(float)  * WARPS_PER_BLOCK * MAX_K;

    knn_kernel<WARPS_PER_BLOCK, TILE_SIZE, MAX_K>
        <<<gridDim, blockDim, sharedBytes>>>(query,
                                             query_count,
                                             data,
                                             data_count,
                                             result,
                                             k);

    // Synchronize to ensure completion before returning.
    cudaDeviceSynchronize();
}