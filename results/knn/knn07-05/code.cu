#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// Constants for kernel configuration.
// One warp (32 threads) processes one query.
constexpr int WARP_SIZE          = 32;
constexpr int WARPS_PER_BLOCK    = 4;      // 4 warps per block -> 128 threads per block
constexpr int THREADS_PER_BLOCK  = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_SIZE          = 1024;   // Number of data points cached per block in shared memory

using PairIF = std::pair<int, float>;

// ---------------------------------------------------------------------------
// Bitonic sort on (distance, index) pairs, in-place, serial on a single thread.
// Input size n must be a power of two (k is guaranteed to be).
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bitonic_sort_pairs(float *dist, int *idx, int n)
{
    for (int k = 2; k <= n; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = 0; i < n; ++i)
            {
                int l = i ^ j;
                if (l > i)
                {
                    bool ascending = ((i & k) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    bool do_swap = (ascending && di > dl) || (!ascending && di < dl);
                    if (do_swap)
                    {
                        dist[i] = dl;
                        dist[l] = di;
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
// Flush one warp's candidate buffer into its intermediate top-k result.
// This implements the specified three-step procedure:
//
// 0. Invariant: intermediate result is sorted ascending.
// 1. Sort the buffer (size k) using Bitonic Sort (ascending).
// 2. Merge buffer and intermediate into a bitonic sequence of size k by
//    taking, for each position i, the minimum of buffer[i] and
//    intermediate[k - i - 1].
// 3. Sort the merged result (size k) using Bitonic Sort (ascending) to
//    obtain the updated intermediate result.
//
// The candidate buffer is stored in `candDist` / `candIdx` and is
// overwritten with the merged, sorted result.
// The intermediate result is stored in `intermDist` / `intermIdx` and is
// updated in-place at the end.
// The per-warp candidate count is reset to 0, and maxDist[warp] is set to
// the distance of the k-th nearest neighbor.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void flush_candidates_for_warp(
    int warpId,
    int k,
    float *candDistBase,
    int   *candIdxBase,
    float *intermDistBase,
    int   *intermIdxBase,
    int   *candCountArr,
    float *maxDistArr)
{
    int count = candCountArr[warpId];
    if (count == 0)
        return; // Nothing to flush.

    float *candDist   = candDistBase   + warpId * k;
    int   *candIdx    = candIdxBase    + warpId * k;
    float *intermDist = intermDistBase + warpId * k;
    int   *intermIdx  = intermIdxBase  + warpId * k;

    // Pad the buffer with "infinite" distances so its length is k.
    for (int i = count; i < k; ++i)
    {
        candDist[i] = FLT_MAX;
        candIdx[i]  = -1;
    }

    // Step 1: sort the buffer.
    bitonic_sort_pairs(candDist, candIdx, k);

    // Step 2: merge buffer and intermediate into a bitonic sequence.
    // The merged result is written back into the candidate arrays.
    for (int i = 0; i < k; ++i)
    {
        float dBuf = candDist[i];
        int   iBuf = candIdx[i];

        float dInt = intermDist[k - 1 - i];
        int   iInt = intermIdx[k - 1 - i];

        if (dBuf <= dInt)
        {
            candDist[i] = dBuf;
            candIdx[i]  = iBuf;
        }
        else
        {
            candDist[i] = dInt;
            candIdx[i]  = iInt;
        }
    }

    // Step 3: sort merged result (now in candDist/candIdx) and store into intermediate.
    bitonic_sort_pairs(candDist, candIdx, k);

    for (int i = 0; i < k; ++i)
    {
        intermDist[i] = candDist[i];
        intermIdx[i]  = candIdx[i];
    }

    // Update maxDist and reset candidate count.
    candCountArr[warpId] = 0;
    maxDistArr[warpId]   = intermDist[k - 1];
}

// ---------------------------------------------------------------------------
// CUDA kernel implementing k-NN with one warp per query.
// Each block caches TILE_SIZE data points into shared memory, and each warp
// in that block processes a different query point. For each query, a
// per-warp candidate buffer of size k is maintained in shared memory.
// ---------------------------------------------------------------------------
__global__
void knn_kernel(
    const float2 * __restrict__ query,
    int                    query_count,
    const float2 * __restrict__ data,
    int                    data_count,
    PairIF                * __restrict__ result,
    int                    k)
{
    extern __shared__ unsigned char shared_raw[];

    // Shared memory layout:
    // [0]                          : float2 dataTile[TILE_SIZE]
    // [1]                          : int   candIdx[WARPS_PER_BLOCK * k]
    // [2]                          : float candDist[WARPS_PER_BLOCK * k]
    // [3]                          : int   intermIdx[WARPS_PER_BLOCK * k]
    // [4]                          : float intermDist[WARPS_PER_BLOCK * k]
    // [5]                          : int   candCount[WARPS_PER_BLOCK]
    // [6]                          : float maxDist[WARPS_PER_BLOCK]

    size_t offset = 0;

    float2 *dataTile = reinterpret_cast<float2 *>(shared_raw + offset);
    offset += TILE_SIZE * sizeof(float2);

    int *candIdxBase = reinterpret_cast<int *>(shared_raw + offset);
    offset += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);

    float *candDistBase = reinterpret_cast<float *>(shared_raw + offset);
    offset += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float);

    int *intermIdxBase = reinterpret_cast<int *>(shared_raw + offset);
    offset += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);

    float *intermDistBase = reinterpret_cast<float *>(shared_raw + offset);
    offset += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float);

    int *candCountArr = reinterpret_cast<int *>(shared_raw + offset);
    offset += WARPS_PER_BLOCK * sizeof(int);

    float *maxDistArr = reinterpret_cast<float *>(shared_raw + offset);
    // offset += WARPS_PER_BLOCK * sizeof(float); // No further use of offset.

    const int threadId = threadIdx.x;
    const int warpId   = threadId / WARP_SIZE;
    const int laneId   = threadId % WARP_SIZE;

    const int globalWarpId = blockIdx.x * WARPS_PER_BLOCK + warpId;
    const int queryId      = globalWarpId;

    const bool hasQuery = (queryId < query_count);

    // Load query point for this warp and broadcast within warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (hasQuery && laneId == 0)
    {
        float2 q = query[queryId];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // Initialize per-warp intermediate result and candidate state.
    if (hasQuery && laneId == 0)
    {
        candCountArr[warpId] = 0;
        maxDistArr[warpId]   = FLT_MAX;

        float *intermDist = intermDistBase + warpId * k;
        int   *intermIdx  = intermIdxBase  + warpId * k;

        // Intermediate result starts as "all infinities" (no neighbors yet).
        for (int i = 0; i < k; ++i)
        {
            intermDist[i] = FLT_MAX;
            intermIdx[i]  = -1;
        }
    }

    __syncthreads();

    // Sweep over data points in tiles.
    for (int dataStart = 0; dataStart < data_count; dataStart += TILE_SIZE)
    {
        const int tileSize = min(TILE_SIZE, data_count - dataStart);

        // Load tile into shared memory cooperatively by all threads in block.
        for (int i = threadId; i < tileSize; i += blockDim.x)
        {
            dataTile[i] = data[dataStart + i];
        }

        __syncthreads();

        // Process the tile: each warp handles its query.
        if (hasQuery)
        {
            float *candDistWarp = candDistBase + warpId * k;
            int   *candIdxWarp  = candIdxBase  + warpId * k;

            // Each iteration processes up to WARP_SIZE points from the tile.
            for (int jBase = 0; jBase < tileSize; jBase += WARP_SIZE)
            {
                int j = jBase + laneId;
                bool hasPoint = (j < tileSize);

                float dist       = 0.0f;
                bool isCandidate = false;

                if (hasPoint)
                {
                    float2 p = dataTile[j];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist = dx * dx + dy * dy;

                    float curMax = maxDistArr[warpId];
                    isCandidate  = (dist < curMax);
                }

                unsigned int mask = __ballot_sync(0xFFFFFFFFu, isCandidate);
                int numCandidates  = __popc(mask);

                if (numCandidates > 0)
                {
                    bool needFlush = false;

                    // Check if buffer has enough room for these candidates; if not, flush first.
                    if (laneId == 0)
                    {
                        int curCount = candCountArr[warpId];
                        if (curCount + numCandidates > k)
                        {
                            flush_candidates_for_warp(
                                warpId,
                                k,
                                candDistBase,
                                candIdxBase,
                                intermDistBase,
                                intermIdxBase,
                                candCountArr,
                                maxDistArr);

                            needFlush = true;
                        }
                    }

                    needFlush = __shfl_sync(0xFFFFFFFFu, needFlush, 0);

                    if (needFlush)
                    {
                        // After flushing, maxDist may have decreased, so re-evaluate candidates.
                        if (hasPoint)
                        {
                            float curMax2 = maxDistArr[warpId];
                            isCandidate   = (dist < curMax2);
                        }
                        else
                        {
                            isCandidate = false;
                        }

                        mask          = __ballot_sync(0xFFFFFFFFu, isCandidate);
                        numCandidates = __popc(mask);

                        if (numCandidates == 0)
                        {
                            // No candidates remain for this wave after threshold tightening.
                            continue;
                        }
                    }

                    // Insert candidates: lane 0 reserves a block of positions with atomicAdd.
                    int base = 0;
                    if (laneId == 0)
                    {
                        base = atomicAdd(&candCountArr[warpId], numCandidates);
                    }
                    base = __shfl_sync(0xFFFFFFFFu, base, 0);

                    if (isCandidate)
                    {
                        // Compute per-lane offset within this wave using prefix of the mask bits.
                        unsigned int laneMask = mask & ((1u << laneId) - 1u);
                        int offset            = __popc(laneMask);
                        int pos               = base + offset;

                        candDistWarp[pos] = dist;
                        candIdxWarp[pos]  = dataStart + j;
                    }
                }
            }
        }

        __syncthreads();
    }

    // After processing all tiles, flush remaining candidates and write results.
    if (hasQuery && laneId == 0)
    {
        if (candCountArr[warpId] > 0)
        {
            flush_candidates_for_warp(
                warpId,
                k,
                candDistBase,
                candIdxBase,
                intermDistBase,
                intermIdxBase,
                candCountArr,
                maxDistArr);
        }

        float *intermDist = intermDistBase + warpId * k;
        int   *intermIdx  = intermIdxBase  + warpId * k;

        // Write final k nearest neighbors for this query, sorted ascending by distance.
        for (int i = 0; i < k; ++i)
        {
            int outPos = queryId * k + i;
            result[outPos].first  = intermIdx[i];
            result[outPos].second = intermDist[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Host interface: launches the CUDA kernel.
// Assumes `query`, `data`, and `result` are device pointers allocated by
// cudaMalloc, and that data_count >= k, with k being a power of two between
// 32 and 1024 (inclusive).
// ---------------------------------------------------------------------------
void run_knn(
    const float2 *query,
    int           query_count,
    const float2 *data,
    int           data_count,
    PairIF       *result,
    int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    // Grid configuration: one warp per query.
    int numWarps   = (query_count + 0) / 1; // one query per warp
    int numBlocks  = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim(numBlocks, 1, 1);

    // Compute dynamic shared memory size.
    size_t shared_bytes = 0;
    shared_bytes += TILE_SIZE * sizeof(float2);                    // dataTile
    shared_bytes += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);   // candIdx
    shared_bytes += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float); // candDist
    shared_bytes += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);   // intermIdx
    shared_bytes += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float); // intermDist
    shared_bytes += WARPS_PER_BLOCK * sizeof(int);                            // candCount
    shared_bytes += WARPS_PER_BLOCK * sizeof(float);                          // maxDist

    // Allow kernel to use the requested dynamic shared memory size (for GPUs that require opt-in).
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    knn_kernel<<<gridDim, blockDim, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k);

    // Caller can synchronize or check errors as needed.
}