#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Simple POD equivalent of std::pair<int, float> for device use.
// The host code will reinterpret_cast between this and std::pair<int, float>.
struct KnnPair {
    int   first;
    float second;
};

// Tunable kernel configuration parameters.
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;    // 4 warps per block -> 128 threads/block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_SIZE         = 1024; // Number of data points cached per block in shared memory

// Warp-wide function that finds the current worst (maximum) distance in a Top-K buffer.
// The Top-K buffer is shared among threads of the warp and stored in shared memory.
// All threads in the warp must call this function with the same arguments.
//
// topkDistances: pointer to the shared-memory array of distances for all warps in the block
// base         : starting index of this warp's Top-K buffer in topkDistances
// k            : number of neighbors to keep
// worstDist    : (output) maximum distance in the Top-K buffer
// worstPos     : (output) position of the maximum distance within this warp's Top-K buffer (0..k-1)
__device__ __forceinline__
void warp_find_worst(const float *topkDistances,
                     int base,
                     int k,
                     float &worstDist,
                     int &worstPos)
{
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    float localMaxDist = -FLT_MAX;
    int   localMaxPos  = 0;

    // Each lane scans a strided subset of the Top-K buffer.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        float d = topkDistances[base + i];
        if (d > localMaxDist) {
            localMaxDist = d;
            localMaxPos  = i;
        }
    }

    // Parallel reduction across the warp to find the global maximum.
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float otherDist = __shfl_down_sync(mask, localMaxDist, offset);
        int   otherPos  = __shfl_down_sync(mask, localMaxPos,  offset);
        if (otherDist > localMaxDist) {
            localMaxDist = otherDist;
            localMaxPos  = otherPos;
        }
    }

    // Broadcast the result from lane 0 to all lanes so every thread has the same worstDist/worstPos.
    worstDist = __shfl_sync(mask, localMaxDist, 0);
    worstPos  = __shfl_sync(mask, localMaxPos,  0);
}

// Kernel: each warp processes exactly one query point.
// The block loads data points in tiles into shared memory; each warp then reuses the tile to
// compute distances from its query to all points in the tile.
// A per-warp Top-K buffer (in shared memory) keeps the current k nearest neighbors.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           KnnPair * __restrict__ result,
                           int k)
{
    extern __shared__ unsigned char smem[];

    // Shared-memory layout:
    // [0 .. TILE_SIZE-1]                  : float2 data tile
    // [TILE_SIZE .. TILE_SIZE + W* k - 1] : int   Top-K indices for all warps in block
    // [... next ...]                      : float Top-K distances for all warps in block
    float2 *sharedData     = reinterpret_cast<float2*>(smem);
    int    *topkIndices    = reinterpret_cast<int*>(sharedData + TILE_SIZE);
    float  *topkDistances  = reinterpret_cast<float*>(topkIndices + WARPS_PER_BLOCK * k);

    const int tid            = threadIdx.x;
    const int warpIdInBlock  = tid / WARP_SIZE;        // 0..WARPS_PER_BLOCK-1
    const int laneId         = tid & (WARP_SIZE - 1);  // 0..31
    const int globalWarpId   = blockIdx.x * WARPS_PER_BLOCK + warpIdInBlock;
    const bool active        = (globalWarpId < query_count);
    const int  queryIdx      = globalWarpId;

    const unsigned fullMask  = 0xffffffffu;

    // Load query point and broadcast its coordinates to all lanes in the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && laneId == 0) {
        float2 q = query[queryIdx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(fullMask, qx, 0);
    qy = __shfl_sync(fullMask, qy, 0);

    // Per-warp Top-K buffer base offset in shared memory.
    const int topkBase = warpIdInBlock * k;

    // Initialize Top-K buffers for this warp: distances to +infinity, indices to -1.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        topkDistances[topkBase + i] = FLT_MAX;
        topkIndices[topkBase + i]   = -1;
    }

    // Current worst (maximum) distance among the Top-K entries for this warp.
    float currentWorstDist = FLT_MAX;
    int   currentWorstPos  = 0;

    // Ensure that all Top-K buffers and the shared-memory layout are initialized before use.
    __syncthreads();

    // Process all data points in tiles of size TILE_SIZE.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        int tileSize = data_count - tileStart;
        if (tileSize > TILE_SIZE) tileSize = TILE_SIZE;

        // Load the current tile into shared memory cooperatively using all threads in the block.
        for (int idxInTile = tid; idxInTile < tileSize; idxInTile += blockDim.x) {
            sharedData[idxInTile] = data[tileStart + idxInTile];
        }

        // Synchronize to ensure the tile is fully loaded before distance computations.
        __syncthreads();

        if (active) {
            // Process the tile in groups of WARP_SIZE points at a time.
            // Each lane computes distance to one point in the group; then the warp inserts
            // all 32 candidates sequentially into its Top-K buffer.
            for (int tileOffset = 0; tileOffset < tileSize; tileOffset += WARP_SIZE) {
                const int localIndex = tileOffset + laneId;

                // Each lane computes squared Euclidean distance for its candidate point.
                // Out-of-range lanes produce a dummy candidate with index -1.
                float candDist = FLT_MAX;
                int   candIdx  = -1;
                if (localIndex < tileSize) {
                    float2 p = sharedData[localIndex];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    candDist = dx * dx + dy * dy;
                    candIdx  = tileStart + localIndex;  // global index in data[]
                }

                // Sequentially insert each lane's candidate into the Top-K buffer.
                // We iterate over srcLane = 0..31; at each step, all lanes see the same
                // candidate (distance, index) via warp shuffles. Only lane 0 performs
                // the actual write into Top-K; all lanes participate in recomputing
                // the new worst distance via warp_find_worst.
                for (int srcLane = 0; srcLane < WARP_SIZE; ++srcLane) {
                    float d   = __shfl_sync(fullMask, candDist, srcLane);
                    int   idx = __shfl_sync(fullMask, candIdx,  srcLane);

                    // Skip dummy candidates (out of range).
                    if (idx < 0) {
                        continue;
                    }

                    // If this candidate is better (smaller distance) than the current worst,
                    // insert it by replacing the current worst entry, then recompute worst.
                    if (d < currentWorstDist) {
                        if (laneId == 0) {
                            topkDistances[topkBase + currentWorstPos] = d;
                            topkIndices[topkBase + currentWorstPos]   = idx;
                        }

                        // Ensure the write is visible to all lanes before recomputing worst.
                        __syncwarp(fullMask);

                        // Recompute worstDist and worstPos across the warp's Top-K buffer.
                        warp_find_worst(topkDistances, topkBase, k,
                                        currentWorstDist, currentWorstPos);
                    }
                }
            }
        }

        // Synchronize before loading the next tile to avoid overwriting sharedData early.
        __syncthreads();
    }

    // After all tiles are processed, each active warp's Top-K buffer holds the k nearest
    // neighbors for its query, but in arbitrary order. Sort them by distance and write out.
    if (active && laneId == 0) {
        float *topDist = &topkDistances[topkBase];
        int   *topIdx  = &topkIndices[topkBase];

        // Simple in-place selection sort of size k in ascending order of distance.
        // This is done by lane 0 only; other lanes remain idle.
        for (int i = 0; i < k - 1; ++i) {
            int   minPos  = i;
            float minDist = topDist[i];
            for (int j = i + 1; j < k; ++j) {
                float d = topDist[j];
                if (d < minDist) {
                    minDist = d;
                    minPos  = j;
                }
            }
            if (minPos != i) {
                float tmpD   = topDist[i];
                topDist[i]   = topDist[minPos];
                topDist[minPos] = tmpD;

                int tmpI     = topIdx[i];
                topIdx[i]    = topIdx[minPos];
                topIdx[minPos] = tmpI;
            }
        }

        // Store sorted neighbors (index and distance) to the global result array.
        for (int j = 0; j < k; ++j) {
            int outPos = queryIdx * k + j;
            result[outPos].first  = topIdx[j];
            result[outPos].second = topDist[j];
        }
    }
}

// Host-side interface as specified.
// query       : device pointer to query_count float2 points
// data        : device pointer to data_count  float2 points
// result      : device pointer to query_count * k std::pair<int, float>
// k           : number of nearest neighbors per query (power of two, 32..1024)
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // reinterpret_cast result pointer to a simple POD type usable in device code.
    KnnPair *deviceResult = reinterpret_cast<KnnPair*>(result);

    // Kernel launch configuration.
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    int  numWarps  = query_count;
    int  numBlocks = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(numBlocks, 1, 1);

    // Compute required dynamic shared memory size:
    // - TILE_SIZE float2 for the data tile
    // - WARPS_PER_BLOCK * k ints for indices
    // - WARPS_PER_BLOCK * k floats for distances
    size_t sharedMemBytes =
        sizeof(float2) * TILE_SIZE +
        sizeof(int)    * static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) +
        sizeof(float)  * static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k);

    knn_kernel<<<gridDim, blockDim, sharedMemBytes>>>(query,
                                                      query_count,
                                                      data,
                                                      data_count,
                                                      deviceResult,
                                                      k);

    // In production code, one may want to add error checking or synchronization here.
}