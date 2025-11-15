#include <cuda_runtime.h>
#include <utility>

/*
 * Optimized k-NN kernel (squared Euclidean distance in 2D) for modern NVIDIA GPUs.
 *
 * Requirements implemented:
 *  - Each query is processed by a single warp (32 threads).
 *  - Intermediate k-NN result is kept in registers, distributed so that
 *    each thread stores k/32 consecutive neighbors.
 *  - For each query/warp, a candidate buffer of size k (indices + distances)
 *    is stored in shared memory, plus a per-warp candidate counter.
 *  - Input data points are processed in batches cached into shared memory.
 *  - For each batch, each warp computes distances to its query, filters by
 *    current max_distance, and adds closer points to its candidate buffer.
 *    Warp ballot is used to count and position new candidates.
 *  - When the candidate buffer becomes full (k elements), it is merged with
 *    the intermediate result using the specified sequence:
 *      0. Intermediate result is always sorted ascending (invariant).
 *      1. Swap content of buffer and intermediate such that buffer is in
 *         registers (and intermediate is in shared memory).
 *      2. Sort the buffer in registers using Bitonic Sort (ascending).
 *      3. Merge buffer and intermediate into a bitonic sequence by taking
 *         element-wise minima of buffer[i] and intermediate[k - 1 - i].
 *      4. Sort this bitonic sequence with Bitonic Sort to obtain an updated
 *         sorted intermediate result.
 *  - After all data is processed, any non-empty candidate buffer is merged
 *    once more with the intermediate result (with padding).
 *  - Bitonic Sort uses warp shuffle instructions for cross-thread exchanges.
 *    Since each thread stores consecutive elements, exchanged elements always
 *    have the same index within each thread's register array. Within-thread
 *    exchanges are simple register swaps.
 */

static constexpr int WARPS_PER_BLOCK        = 8;    // 8 warps (256 threads) per block
static constexpr int THREADS_PER_BLOCK      = WARPS_PER_BLOCK * 32;
static constexpr int MAX_K                  = 1024; // max k (power of two, 32..1024)
static constexpr int MAX_ITEMS_PER_THREAD   = MAX_K / 32; // max k/32 = 32
static constexpr int DATA_TILE_SIZE         = 1024; // number of data points cached per block

// Convenience alias for the result type.
using PairIF = std::pair<int, float>;

// Full warp mask.
#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

// Warp-wide function to obtain the k-th element (index k-1) from a distributed array.
// The array is stored so that each lane holds itemsPerThread consecutive elements.
template <int MAX_ITEMS_PER_THREAD>
__device__ inline float warpGetKthValue(const float (&values)[MAX_ITEMS_PER_THREAD],
                                        int itemsPerThread, int k)
{
    int laneId   = threadIdx.x & 31;
    int index    = k - 1;                     // global index of k-th nearest neighbor
    int ownerLane = index / itemsPerThread;   // lane holding that element
    int localSlot = index % itemsPerThread;   // index within that lane's register array
    float v = values[localSlot];
    // Broadcast from ownerLane to all lanes.
    return __shfl_sync(FULL_MASK, v, ownerLane);
}

/*
 * Warp-level Bitonic Sort for k elements distributed across the warp.
 *
 * Layout:
 *  - k is a power of two in [32, MAX_K].
 *  - Each warp has 32 lanes; each lane stores itemsPerThread = k / 32
 *    consecutive elements in its register arrays 'values' and 'indices'.
 *
 * Algorithm:
 *  - Follows the classical bitonic sorting network for k elements.
 *  - For strides smaller than itemsPerThread, comparators operate within
 *    each thread (intra-thread), using register swaps.
 *  - For strides greater or equal to itemsPerThread, comparators operate
 *    across threads (inter-thread). In this case the partner index differs
 *    only in the more significant bits (lane id), while the localSlot
 *    (element index within the thread) is identical. This allows us to
 *    use __shfl_xor_sync on the per-slot values.
 */
template <int MAX_ITEMS_PER_THREAD, int MAX_K>
__device__ inline void warpBitonicSort(float (&values)[MAX_ITEMS_PER_THREAD],
                                       int   (&indices)[MAX_ITEMS_PER_THREAD],
                                       int k,
                                       unsigned int activeMask)
{
    int laneId         = threadIdx.x & 31;
    int itemsPerThread = k >> 5; // k / 32, guaranteed power-of-two
    // log2(itemsPerThread), itemsPerThread >= 1
    int shift = __ffs(itemsPerThread) - 1;

    // Outer loops over bitonic sequence size (size) and comparison stride (stride).
    for (int size = 2; size <= k; size <<= 1)
    {
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            if (stride < itemsPerThread)
            {
                // Intra-thread comparators: indices differ only in localSlot.
                // We process each pair once (p > s) and update both entries.
                for (int s = 0; s < itemsPerThread; ++s)
                {
                    int p = s ^ stride;
                    if (p <= s) continue; // each pair only once

                    int i0 = laneId * itemsPerThread + s;
                    // i1 = i0 ^ stride, but we don't need it explicitly.
                    bool ascending = ((i0 & size) == 0);

                    float v0 = values[s];
                    float v1 = values[p];
                    int   id0 = indices[s];
                    int   id1 = indices[p];

                    float minV, maxV;
                    int   minId, maxId;
                    if (v0 < v1) {
                        minV = v0; minId = id0;
                        maxV = v1; maxId = id1;
                    } else {
                        minV = v1; minId = id1;
                        maxV = v0; maxId = id0;
                    }

                    if (ascending) {
                        // Put smaller at lower index, larger at higher index.
                        values[s] = minV; indices[s] = minId;
                        values[p] = maxV; indices[p] = maxId;
                    } else {
                        // Descending sequence.
                        values[s] = maxV; indices[s] = maxId;
                        values[p] = minV; indices[p] = minId;
                    }
                }
            }
            else
            {
                // Inter-thread comparators: indices differ in lane bits only.
                // For a given localSlot s, the partner lane is:
                //   partnerLane = laneId ^ (stride / itemsPerThread)
                // and both lanes work on the same localSlot s.
                int partnerMask = stride >> shift; // power-of-two (>=1)

                for (int s = 0; s < itemsPerThread; ++s)
                {
                    int i      = laneId * itemsPerThread + s;
                    int partner = i ^ stride;

                    // Both indices in a pair share the same 'size' bit,
                    // hence the same direction (ascending/descending).
                    bool ascending = ((i & size) == 0);
                    bool isLow     = ((i & stride) == 0); // "lower" index in the pair

                    float v   = values[s];
                    int   id  = indices[s];

                    // Partner's value and index from the lane with id laneId ^ partnerMask.
                    float ov  = __shfl_xor_sync(activeMask, v,  partnerMask, 32);
                    int   oid = __shfl_xor_sync(activeMask, id, partnerMask, 32);

                    float minV, maxV;
                    int   minId, maxId;
                    if (v < ov) {
                        minV = v;   minId = id;
                        maxV = ov;  maxId = oid;
                    } else {
                        minV = ov;  minId = oid;
                        maxV = v;   maxId = id;
                    }

                    if (ascending) {
                        // Ascending sequence: lower index gets min, higher gets max.
                        if (isLow) {
                            values[s] = minV; indices[s] = minId;
                        } else {
                            values[s] = maxV; indices[s] = maxId;
                        }
                    } else {
                        // Descending sequence: lower index gets max, higher gets min.
                        if (isLow) {
                            values[s] = maxV; indices[s] = maxId;
                        } else {
                            values[s] = minV; indices[s] = minId;
                        }
                    }
                }
            }

            __syncwarp(activeMask);
        }
    }
}

/*
 * Merge a full candidate buffer (k elements) with the current intermediate
 * result for one warp/query.
 *
 * Parameters:
 *  - warpIdInBlock: warp index within the block.
 *  - bestDist/bestIdx: intermediate k-NN result stored in registers, sorted ascending.
 *  - maxDistance: reference to the current max_distance variable (updated here).
 *  - sharedCandDist/sharedCandIdx: per-warp candidate buffer in shared memory.
 *  - k: total number of neighbors (power of two between 32 and MAX_K).
 *  - itemsPerThread: k / 32, number of elements per lane.
 *
 * Steps implemented:
 *  1. Swap contents of buffer and intermediate result so that the buffer
 *     (candidates) is in registers (bufDist/bufIdx) and the intermediate
 *     result is in shared memory.
 *  2. Sort the buffer (bufDist/bufIdx) with bitonic sort (ascending).
 *  3. Merge buffer and intermediate by taking, for each global index i,
 *     min(buffer[i], intermediate[k-1-i]) into bestDist/bestIdx. The result
 *     is a bitonic sequence containing the best k elements of both.
 *  4. Sort bestDist/bestIdx (bitonic sequence) ascending with bitonic sort.
 *     Update maxDistance to the last element (k-th nearest).
 */
template <int WARPS_PER_BLOCK_T, int MAX_ITEMS_PER_THREAD_T, int MAX_K_T>
__device__ inline void merge_full_buffer(int warpIdInBlock,
                                         float (&bestDist)[MAX_ITEMS_PER_THREAD_T],
                                         int   (&bestIdx)[MAX_ITEMS_PER_THREAD_T],
                                         float &maxDistance,
                                         float sharedCandDist[WARPS_PER_BLOCK_T][MAX_K_T],
                                         int   sharedCandIdx [WARPS_PER_BLOCK_T][MAX_K_T],
                                         int   k,
                                         int   itemsPerThread)
{
    int laneId = threadIdx.x & 31;

    float bufDist[MAX_ITEMS_PER_THREAD_T];
    int   bufIdx [MAX_ITEMS_PER_THREAD_T];

    int base = laneId * itemsPerThread;

    // Step 1: swap buffer (shared) and intermediate (registers).
    #pragma unroll
    for (int i = 0; i < MAX_ITEMS_PER_THREAD_T; ++i)
    {
        if (i < itemsPerThread)
        {
            int idx = base + i;
            bufDist[i] = sharedCandDist[warpIdInBlock][idx];
            bufIdx[i]  = sharedCandIdx [warpIdInBlock][idx];

            sharedCandDist[warpIdInBlock][idx] = bestDist[i];
            sharedCandIdx [warpIdInBlock][idx] = bestIdx[i];
        }
    }

    __syncwarp(FULL_MASK);

    // Step 2: sort buffer in registers (ascending).
    warpBitonicSort<MAX_ITEMS_PER_THREAD_T, MAX_K_T>(bufDist, bufIdx, k, FULL_MASK);

    __syncwarp(FULL_MASK);

    // Step 3: merge buffer and intermediate into a bitonic sequence.
    #pragma unroll
    for (int i = 0; i < MAX_ITEMS_PER_THREAD_T; ++i)
    {
        if (i < itemsPerThread)
        {
            int idx1 = base + i;
            int idx2 = k - 1 - idx1; // mirrored index

            float candD = bufDist[i];
            int   candI = bufIdx[i];

            float intermD = sharedCandDist[warpIdInBlock][idx2];
            int   intermI = sharedCandIdx [warpIdInBlock][idx2];

            if (candD <= intermD) {
                bestDist[i] = candD;
                bestIdx[i]  = candI;
            } else {
                bestDist[i] = intermD;
                bestIdx[i]  = intermI;
            }
        }
    }

    __syncwarp(FULL_MASK);

    // Step 4: sort the merged bitonic sequence (ascending).
    warpBitonicSort<MAX_ITEMS_PER_THREAD_T, MAX_K_T>(bestDist, bestIdx, k, FULL_MASK);

    __syncwarp(FULL_MASK);

    // Update maxDistance to distance of k-th nearest neighbor.
    maxDistance = warpGetKthValue<MAX_ITEMS_PER_THREAD_T>(bestDist, itemsPerThread, k);
}

/*
 * Main k-NN kernel.
 *
 * Each warp processes exactly one query point.
 *  - The warp index (within the grid) identifies the query index.
 *  - The block cooperatively loads batches of data points into shared memory.
 *  - Each warp computes squared distances from its query to all data points,
 *    filters by maxDistance, and fills its candidate buffer.
 *  - When the candidate buffer is full, it is merged with the intermediate
 *    result using merge_full_buffer().
 */
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           PairIF * __restrict__ result,
                           int k)
{
    // Shared memory: cached data tile and per-warp candidate buffers.
    __shared__ float2 s_data[DATA_TILE_SIZE];
    __shared__ int    s_candIdx [WARPS_PER_BLOCK][MAX_K];
    __shared__ float  s_candDist[WARPS_PER_BLOCK][MAX_K];

    const int warpIdInBlock = threadIdx.x >> 5;  // warp index inside block
    const int laneId        = threadIdx.x & 31;  // lane index in warp
    const int globalWarpId  = blockIdx.x * WARPS_PER_BLOCK + warpIdInBlock;
    const int queryIdx      = globalWarpId;

    const bool activeWarp   = (queryIdx < query_count);
    const int  itemsPerThread = k >> 5; // k / 32; guaranteed power-of-two

    // Per-warp intermediate result: stored in registers, k/32 elements per lane.
    float bestDist[MAX_ITEMS_PER_THREAD];
    int   bestIdx [MAX_ITEMS_PER_THREAD];

    float maxDistance = CUDART_INF_F; // distance of k-th nearest neighbor
    int   bufCount    = 0;            // number of candidates currently in buffer

    float qx = 0.0f, qy = 0.0f;

    if (activeWarp)
    {
        // Initialize intermediate result with +inf and invalid indices.
        #pragma unroll
        for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i)
        {
            if (i < itemsPerThread)
            {
                bestDist[i] = CUDART_INF_F;
                bestIdx[i]  = -1;
            }
        }

        // Load this warp's query point and broadcast its coordinates.
        if (laneId == 0)
        {
            float2 q = query[queryIdx];
            qx = q.x;
            qy = q.y;
        }
    }

    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Process data in tiles.
    for (int base = 0; base < data_count; base += DATA_TILE_SIZE)
    {
        int tileSize = data_count - base;
        if (tileSize > DATA_TILE_SIZE) tileSize = DATA_TILE_SIZE;

        // Load data tile into shared memory cooperatively.
        for (int idx = threadIdx.x; idx < tileSize; idx += blockDim.x)
        {
            s_data[idx] = data[base + idx];
        }

        __syncthreads();

        if (activeWarp)
        {
            // Each lane processes a subset of the tile. The union of all lanes'
            // subsets covers the whole tile.
            for (int tIdx = laneId; tIdx < tileSize; tIdx += 32)
            {
                float2 p = s_data[tIdx];
                float dx = qx - p.x;
                float dy = qy - p.y;
                float dist2 = dx * dx + dy * dy;

                // Filter using current maxDistance (distance of k-th neighbor).
                bool isCandidate = (dist2 < maxDistance);

                unsigned int mask = __ballot_sync(FULL_MASK, isCandidate);

                // Process all candidate lanes for this data point, possibly in
                // multiple iterations if the buffer is almost full.
                while (mask)
                {
                    int freeSpace = k - bufCount;

                    if (freeSpace == 0)
                    {
                        // Candidate buffer is full: merge with intermediate result.
                        merge_full_buffer<WARPS_PER_BLOCK, MAX_ITEMS_PER_THREAD, MAX_K>(
                            warpIdInBlock,
                            bestDist,
                            bestIdx,
                            maxDistance,
                            s_candDist,
                            s_candIdx,
                            k,
                            itemsPerThread);

                        bufCount = 0;
                        freeSpace = k;
                    }

                    int totalPending = __popc(mask);
                    int insertCount  = (totalPending < freeSpace) ? totalPending : freeSpace;

                    if (insertCount == 0)
                        break;

                    bool pending = ((mask >> laneId) & 1u);
                    unsigned int prefixMask = mask & ((1u << laneId) - 1u);
                    int rank = __popc(prefixMask);

                    // Insert first 'insertCount' candidates from this mask into
                    // the candidate buffer, contiguously.
                    if (pending && rank < insertCount)
                    {
                        int indexInBuffer = bufCount + rank;
                        if (indexInBuffer < k)
                        {
                            s_candDist[warpIdInBlock][indexInBuffer] = dist2;
                            s_candIdx [warpIdInBlock][indexInBuffer] = base + tIdx;
                        }
                    }

                    // Update buffer count (warp-synchronous).
                    if (laneId == 0)
                    {
                        bufCount += insertCount;
                    }
                    bufCount = __shfl_sync(FULL_MASK, bufCount, 0);

                    // Remove inserted lanes from the mask.
                    unsigned int insertedMask = __ballot_sync(FULL_MASK, pending && (rank < insertCount));
                    mask &= ~insertedMask;
                }

                // If the buffer has become full exactly after processing this
                // data point, merge it immediately.
                if (bufCount == k)
                {
                    merge_full_buffer<WARPS_PER_BLOCK, MAX_ITEMS_PER_THREAD, MAX_K>(
                        warpIdInBlock,
                        bestDist,
                        bestIdx,
                        maxDistance,
                        s_candDist,
                        s_candIdx,
                        k,
                        itemsPerThread);

                    bufCount = 0;
                }
            }
        }

        __syncthreads();
    }

    if (activeWarp)
    {
        // After all data is processed, merge any remaining candidates in the
        // buffer (if bufCount > 0) with the intermediate result.
        if (bufCount > 0)
        {
            // Pad unused buffer entries with +inf so that the buffer logically
            // contains exactly k elements. These padded entries will not affect
            // the final k nearest neighbors.
            for (int idx = laneId; idx < k; idx += 32)
            {
                if (idx >= bufCount)
                {
                    s_candDist[warpIdInBlock][idx] = CUDART_INF_F;
                    s_candIdx [warpIdInBlock][idx] = -1;
                }
            }
            __syncwarp(FULL_MASK);

            merge_full_buffer<WARPS_PER_BLOCK, MAX_ITEMS_PER_THREAD, MAX_K>(
                warpIdInBlock,
                bestDist,
                bestIdx,
                maxDistance,
                s_candDist,
                s_candIdx,
                k,
                itemsPerThread);
        }

        // Write final k nearest neighbors for this query to global memory.
        int baseOut = queryIdx * k + laneId * itemsPerThread;
        #pragma unroll
        for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i)
        {
            if (i < itemsPerThread)
            {
                int outIdx = baseOut + i;
                result[outIdx].first  = bestIdx[i];
                result[outIdx].second = bestDist[i];
            }
        }
    }
}

/*
 * Host interface.
 *
 * The input pointers (query, data, result) must be device pointers allocated
 * with cudaMalloc. This function launches the k-NN kernel. No additional
 * device memory is allocated here or in the kernel.
 *
 * Parameters:
 *  - query       : device pointer to array of float2, size query_count.
 *  - query_count : number of query points.
 *  - data        : device pointer to array of float2, size data_count.
 *  - data_count  : number of data points.
 *  - result      : device pointer to array of PairIF, size query_count * k.
 *  - k           : number of nearest neighbors per query (power of two,
 *                  32 <= k <= 1024).
 */
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             PairIF *result,
             int k)
{
    dim3 blockDim(THREADS_PER_BLOCK);
    int numBlocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(numBlocks > 0 ? numBlocks : 1);

    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result, k);
}