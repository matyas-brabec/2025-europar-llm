// Optimized CUDA k-NN (k-nearest neighbors) for 2D points.
// -----------------------------------------------------------------------------
// Design overview (see below for detailed implementation):
//   * Hardware target: modern NVIDIA data-center GPUs (e.g., A100, H100).
//   * Parallelization scheme:
//       - One warp (32 threads) processes one query point.
//       - A block contains WARPS_PER_BLOCK warps -> processes WARPS_PER_BLOCK queries.
//   * Memory usage:
//       - Data points are processed in tiles cached in shared memory.
//       - For each warp (query), a private top-k neighbor list is stored in shared
//         memory: distances and indices.
//   * Algorithm:
//       - For each query warp:
//           1. Initialize its top-k list to +inf distances.
//           2. Loop over the data points in shared-memory tiles.
//           3. Each tile is loaded cooperatively by all threads in the block.
//           4. Within the tile, each warp computes distances in parallel for 32
//              points at a time, then sequentially inserts each candidate into
//              the warp's shared top-k structure using warp-synchronous logic.
//              Only candidates better than the current worst in top-k trigger
//              an expensive update; this happens at most O(k log(N/k)) times.
//           5. After all data are processed, each warp performs a bitonic sort
//              of its k candidates (by distance) in shared memory.
//           6. Sorted neighbors (indices + squared distances) are written to the
//              output array.
//
//   * Guarantees:
//       - Exact k nearest neighbors in squared Euclidean distance.
//       - k is a power of two between 32 and 1024 inclusive (assumed).
//       - No additional device memory allocations (only shared + registers).
//       - Result for query i is stored in result[i * k + j] with j-th neighbor
//         sorted by ascending distance.
//
//   * Implementation details:
//       - Shared memory layout per block:
//           - float2  data tile:            TILE_POINTS elements
//           - float   knn distances:        WARPS_PER_BLOCK x MAX_K
//           - int     knn indices:          WARPS_PER_BLOCK x MAX_K
//           - float   current warp worst:   WARPS_PER_BLOCK
//           - int     current warp worst id:WARPS_PER_BLOCK
//       - TILE_POINTS is chosen so that total static shared memory usage
//         is below the 48 KB per-block limit on Ampere/Hopper without
//         requiring opt-in for larger shared memory.
//       - Warp-level primitives (__shfl_sync, __syncwarp) are used to
//         coordinate threads within each warp.

#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable hyper-parameters (fixed here for simplicity).
// These values are chosen to keep static shared memory under 48 KB per block.
static constexpr int WARPS_PER_BLOCK    = 4;                       // 4 warps per block => 128 threads
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int MAX_K              = 1024;                    // max supported k
static constexpr int TILE_POINTS        = 2016;                    // number of data points per shared-memory tile

// -----------------------------------------------------------------------------
// warp_knn_insert:
//   Warp-synchronous insertion of a single candidate (distance, index) into the
//   per-warp top-k structure stored in shared memory.
//
//   * knnDist, knnIdx:
//       - Pointers to the beginning of this warp's shared top-k arrays
//         knnDist[0..k-1], knnIdx[0..k-1].
//   * warpMaxDist, warpMaxPos:
//       - Shared arrays of length WARPS_PER_BLOCK storing, per warp, the current
//         largest distance in the top-k list and its index.
//   * warpId:
//       - Warp index within the block (0..WARPS_PER_BLOCK-1).
//
//   The function:
//       1. Checks if the candidate index is valid and if its distance is smaller
//          than the current worst distance in the warp's list.
//       2. If not, returns immediately (cheap path).
//       3. If yes, replaces the current worst entry with the candidate.
//       4. Recomputes the new worst entry using all threads in the warp by
//          scanning the k distances (each thread handles a strided subset) and
//          performing a warp-wide max reduction.
//
__device__ __forceinline__
void warp_knn_insert(float cand_dist,
                     int   cand_idx,
                     float *knnDist,
                     int   *knnIdx,
                     int   k,
                     float *warpMaxDist,
                     int   *warpMaxPos,
                     int   warpId)
{
    const unsigned fullMask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Ignore invalid candidates (e.g., lanes that didn't produce a point).
    if (cand_idx < 0)
        return;

    // Read current worst distance in this warp's top-k list from shared memory.
    float worstDist = warpMaxDist[warpId];

    // If the candidate is not better (i.e., has distance >= worst), skip.
    if (cand_dist >= worstDist)
        return;

    // Candidate should be inserted: replace the current worst entry.
    // Only one thread needs to perform the store; choose lane 0.
    if (lane == 0)
    {
        int worstPos = warpMaxPos[warpId];
        knnDist[worstPos] = cand_dist;
        knnIdx[worstPos]  = cand_idx;
    }

    // Ensure all warp threads see the updated top-k arrays before recomputing worst.
    __syncwarp(fullMask);

    // Recompute the new worst entry across knnDist[0..k-1].
    // Each thread scans a strided subset, then we do a warp-wide reduction.
    float localMax = -FLT_MAX;
    int   localPos = 0;

    for (int i = lane; i < k; i += WARP_SIZE)
    {
        float d = knnDist[i];
        if (d > localMax)
        {
            localMax = d;
            localPos = i;
        }
    }

    // Warp-wide reduction to find the maximum distance and its position.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        float otherMax = __shfl_down_sync(fullMask, localMax, offset);
        int   otherPos = __shfl_down_sync(fullMask, localPos, offset);
        if (otherMax > localMax)
        {
            localMax = otherMax;
            localPos = otherPos;
        }
    }

    // Lane 0 updates the shared "worst entry" info for this warp.
    if (lane == 0)
    {
        warpMaxDist[warpId] = localMax;
        warpMaxPos[warpId]  = localPos;
    }

    __syncwarp(fullMask);
}

// -----------------------------------------------------------------------------
// warp_bitonic_sort:
//   Bitonic sort of the k-element neighbor list for a single warp.
//
//   * dist, idx:
//       - Pointers to the beginning of this warp's shared arrays
//         dist[0..k-1], idx[0..k-1].
//   * k:
//       - Number of neighbors to sort (power of two between 32 and 1024).
//
//   The sort is ascending by distance. All 32 threads in the warp cooperate,
//   each processing multiple indices in a strided fashion. Synchronization is
//   done via __syncwarp between phases.
//
__device__ __forceinline__
void warp_bitonic_sort(float *dist, int *idx, int k)
{
    const unsigned fullMask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Standard bitonic sort network for power-of-two sizes.
    for (unsigned int size = 2; size <= static_cast<unsigned int>(k); size <<= 1)
    {
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (int i = lane; i < k; i += WARP_SIZE)
            {
                unsigned int ixj = i ^ stride;
                if (ixj > static_cast<unsigned int>(i) && ixj < static_cast<unsigned int>(k))
                {
                    bool ascending = ((i & size) == 0);

                    float val_i = dist[i];
                    float val_j = dist[ixj];
                    int   idx_i = idx[i];
                    int   idx_j = idx[ixj];

                    bool doSwap = ascending ? (val_i > val_j) : (val_i < val_j);
                    if (doSwap)
                    {
                        dist[i]  = val_j;
                        dist[ixj] = val_i;
                        idx[i]   = idx_j;
                        idx[ixj] = idx_i;
                    }
                }
            }
            __syncwarp(fullMask);
        }
    }
}

// -----------------------------------------------------------------------------
// k-NN kernel:
//   Each warp processes exactly one query point, computing its k nearest
//   neighbors among 'data_count' 2D points.
//
//   Parameters:
//       query       - [query_count] array of float2 (x,y).
//       query_count - number of queries.
//       data        - [data_count] array of float2 (x,y).
//       data_count  - number of data points.
//       result      - [query_count * k] array of (index, distance) pairs.
//       k           - number of neighbors (power of two, 32 <= k <= 1024).
//
//   The kernel uses shared memory tiling for 'data' and shared per-warp arrays
//   for k-NN candidates.
//
__global__
void knn_kernel(const float2 *__restrict__ query,
                int query_count,
                const float2 *__restrict__ data,
                int data_count,
                std::pair<int, float> *__restrict__ result,
                int k)
{
    // Shared memory layout:
    //   sDataTile        : TILE_POINTS float2
    //   sWarpDist        : WARPS_PER_BLOCK x MAX_K float
    //   sWarpIdx         : WARPS_PER_BLOCK x MAX_K int
    //   sWarpMaxDist     : WARPS_PER_BLOCK float
    //   sWarpMaxPos      : WARPS_PER_BLOCK int
    __shared__ float2 sDataTile[TILE_POINTS];
    __shared__ float  sWarpDist[WARPS_PER_BLOCK][MAX_K];
    __shared__ int    sWarpIdx[WARPS_PER_BLOCK][MAX_K];
    __shared__ float  sWarpMaxDist[WARPS_PER_BLOCK];
    __shared__ int    sWarpMaxPos[WARPS_PER_BLOCK];

    const int threadId = threadIdx.x;
    const int lane     = threadId & (WARP_SIZE - 1);      // lane within warp
    const int warpId   = threadId >> 5;                   // warp index within block

    const int queryIndex = blockIdx.x * WARPS_PER_BLOCK + warpId;
    const bool warpActive = (queryIndex < query_count);

    // Load the query point for this warp.
    float2 q;
    if (warpActive)
    {
        q = query[queryIndex];
    }

    // Initialize per-warp top-k arrays in shared memory.
    if (warpActive)
    {
        // Distribute initialization of the k entries among the 32 lanes.
        for (int i = lane; i < k; i += WARP_SIZE)
        {
            sWarpDist[warpId][i] = FLT_MAX;
            sWarpIdx[warpId][i]  = -1;
        }

        // Initialize current worst distance as +inf at position 0.
        if (lane == 0)
        {
            sWarpMaxDist[warpId] = FLT_MAX;
            sWarpMaxPos[warpId]  = 0;
        }
    }

    // All threads synchronize to ensure shared memory initialization is visible.
    __syncthreads();

    // Loop over data points in tiles, cached in shared memory.
    for (int base = 0; base < data_count; base += TILE_POINTS)
    {
        // Number of points in this tile.
        int tileCount = data_count - base;
        if (tileCount > TILE_POINTS)
            tileCount = TILE_POINTS;

        // Load tile into shared memory using all threads in the block.
        for (int i = threadId; i < tileCount; i += blockDim.x)
        {
            sDataTile[i] = data[base + i];
        }

        __syncthreads();

        // Each warp processes the shared tile for its query.
        if (warpActive)
        {
            const unsigned fullMask = 0xFFFFFFFFu;
            float *warpDist = &sWarpDist[warpId][0];
            int   *warpIdx  = &sWarpIdx[warpId][0];

            // Process the tile in chunks of 32 points (one per lane).
            const int numSteps = (tileCount + WARP_SIZE - 1) / WARP_SIZE;

            for (int step = 0; step < numSteps; ++step)
            {
                const int idxInTile = step * WARP_SIZE + lane;
                float dist = FLT_MAX;
                int   idx  = -1;

                // Compute squared distance for this lane's point, if it exists.
                if (idxInTile < tileCount)
                {
                    float2 p = sDataTile[idxInTile];
                    float dx = q.x - p.x;
                    float dy = q.y - p.y;
                    dist = dx * dx + dy * dy;
                    idx  = base + idxInTile;
                }

                // Sequentially insert the 0..(validLanes-1) candidates into top-k.
                int validLanes = tileCount - step * WARP_SIZE;
                if (validLanes > WARP_SIZE)
                    validLanes = WARP_SIZE;

                for (int srcLane = 0; srcLane < validLanes; ++srcLane)
                {
                    float cand_dist = __shfl_sync(fullMask, dist, srcLane);
                    int   cand_idx  = __shfl_sync(fullMask, idx,  srcLane);

                    warp_knn_insert(cand_dist,
                                    cand_idx,
                                    warpDist,
                                    warpIdx,
                                    k,
                                    sWarpMaxDist,
                                    sWarpMaxPos,
                                    warpId);
                }
            }
        }

        __syncthreads();
    }

    // After processing all data, sort each warp's k candidates and write results.
    if (warpActive)
    {
        float *warpDist = &sWarpDist[warpId][0];
        int   *warpIdx  = &sWarpIdx[warpId][0];

        // Sort by distance ascending using bitonic sort.
        warp_bitonic_sort(warpDist, warpIdx, k);

        // Write sorted neighbors to global memory.
        // Each lane writes multiple entries in a strided fashion.
        for (int j = lane; j < k; j += WARP_SIZE)
        {
            int outPos = queryIndex * k + j;
            result[outPos].first  = warpIdx[j];
            result[outPos].second = warpDist[j];
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface:
//   void run_knn(const float2 *query, int query_count,
//                const float2 *data,  int data_count,
//                std::pair<int, float> *result, int k);
//
// Description:
//   Launches the CUDA kernel to compute k nearest neighbors for each query.
//   Input arrays (query, data, result) are assumed to be device pointers
//   allocated with cudaMalloc.
//
// Notes:
//   - k must satisfy 32 <= k <= 1024 and be a power of two.
//   - query_count, data_count > 0 are assumed.
//   - No additional device allocations are performed here.
//
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    // Grid: each block handles WARPS_PER_BLOCK queries.
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    knn_kernel<<<gridDim, blockDim>>>(query,
                                      query_count,
                                      data,
                                      data_count,
                                      result,
                                      k);

    // Optional: synchronize / error checking can be added here if desired.
    // cudaDeviceSynchronize();
}