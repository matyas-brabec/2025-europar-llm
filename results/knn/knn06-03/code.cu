#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Optimized k-NN (2D, squared Euclidean distance) for modern NVIDIA GPUs.
// Each query is processed by a single warp (32 threads). The kernel processes
// the data points in shared-memory tiles. For each query (warp), we maintain:
//
//   - A private intermediate top-k result in registers, distributed across
//     the 32 threads of the warp.
//   - A candidate buffer of size k (indices + distances) in shared memory.
//   - A shared integer tracking the number of candidates currently in the buffer.
//   - A max_distance variable (per warp) storing the distance of the current
//     k-th nearest neighbor (i.e., the worst in the top-k set).
//
// New data points are filtered against max_distance and, if closer, appended
// to the candidate buffer using an atomicAdd on the shared counter. Whenever
// adding a new batch of candidates would overflow the buffer, the warp merges
// the candidate buffer into its intermediate top-k result in parallel using
// a bitonic sort over 2k elements in shared memory. A final merge is performed
// after all data has been processed if the candidate buffer is non-empty.

constexpr int KNN_WARP_SIZE        = 32;
constexpr int KNN_WARPS_PER_BLOCK  = 4;     // 4 warps (128 threads) per block
constexpr int KNN_THREADS_PER_BLOCK= KNN_WARPS_PER_BLOCK * KNN_WARP_SIZE;
constexpr int KNN_DATA_TILE_SIZE   = 1024;  // Number of data points cached in shared memory per tile
constexpr int KNN_MAX_K            = 1024;  // Maximum supported k (power of two, <= 1024)

// Ensure we are compiling for 32-thread warps.
static_assert(KNN_WARP_SIZE == 32, "This implementation assumes warp size 32.");

// Merge the current candidate buffer for one warp into its intermediate top-k result.
//
// The candidate buffer is stored in shared memory as:
//   - warpDist[0 .. k-1]: candidate distances (unused slots are filled with +Inf);
//   - warpIdx [0 .. k-1]: corresponding indices.
// The current top-k result is stored in registers, distributed across the 32 threads:
//   - bestDist[t], bestIdx[t] for t in [0, perThreadK).
//
// This function constructs a 2k-element array in shared memory:
//
//   distances: warpDist[0 .. k-1]   = candidate buffer (padded with +Inf)
//              warpDist[k .. 2k-1]  = current top-k
//   indices:   warpIdx [0 .. k-1]   = candidate indices
//              warpIdx [k .. 2k-1]  = top-k indices
//
// Then it performs an in-place bitonic sort on the 2k pairs, in ascending order
// of distance, cooperatively across the warp. The smallest k elements after the
// sort are copied back into bestDist/bestIdx, and maxDist is updated to be the
// distance of the k-th neighbor (i.e., warpDist[k-1] after sorting).
//
// All 32 threads in the warp must call this function together (warp-synchronous).
__device__ __forceinline__
void merge_candidate_buffer(
    int laneId,
    int k,
    int perThreadK,
    int *candCountPtr,
    float *warpDist,
    int *warpIdx,
    float *bestDist,
    int *bestIdx,
    float &maxDist)
{
    const unsigned FULL_MASK = 0xffffffffu;

    // Read the current number of candidates from shared memory (lane 0) and broadcast.
    int candCount = 0;
    if (laneId == 0) {
        candCount = *candCountPtr;
    }
    candCount = __shfl_sync(FULL_MASK, candCount, 0);

    // If there are no candidates to merge, nothing to do.
    if (candCount == 0) {
        return;
    }

    // Fill any unused candidate slots [candCount, k) with +Inf so they sink to the end.
    if (candCount < k) {
        for (int i = laneId + candCount; i < k; i += KNN_WARP_SIZE) {
            warpDist[i] = CUDART_INF_F;
            warpIdx[i]  = -1;
        }
    }

    // Copy current top-k entries from registers into the second half of the array in shared memory.
    // Mapping: globalSlot = t * warpSize + laneId, where t in [0, perThreadK).
    for (int t = 0; t < perThreadK; ++t) {
        int globalSlot = t * KNN_WARP_SIZE + laneId;
        if (globalSlot < k) {
            warpDist[k + globalSlot] = bestDist[t];
            warpIdx [k + globalSlot] = bestIdx[t];
        }
    }

    __syncwarp();

    // Perform a bitonic sort over 2k elements in shared memory, sorting by distance ascending.
    const int total = 2 * k;

    // Standard bitonic sort network:
    // for (size = 2; size <= total; size <<= 1)
    //   for (stride = size >> 1; stride > 0; stride >>= 1)
    //     for each i: compare-and-swap with i^stride depending on ascending/descending.
    for (int size = 2; size <= total; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int idx = laneId; idx < total; idx += KNN_WARP_SIZE) {
                int partner = idx ^ stride;
                if (partner > idx && partner < total) {
                    // Ascending in the lower half, descending in the upper half.
                    bool ascending = ((idx & size) == 0);

                    float dist_i = warpDist[idx];
                    float dist_p = warpDist[partner];
                    int   idx_i  = warpIdx[idx];
                    int   idx_p  = warpIdx[partner];

                    bool swapNeeded = ascending ? (dist_i > dist_p) : (dist_i < dist_p);
                    if (swapNeeded) {
                        warpDist[idx]    = dist_p;
                        warpDist[partner]= dist_i;
                        warpIdx[idx]     = idx_p;
                        warpIdx[partner] = idx_i;
                    }
                }
            }
            __syncwarp();
        }
    }

    // Copy the smallest k elements back into the per-thread top-k registers.
    for (int t = 0; t < perThreadK; ++t) {
        int globalSlot = t * KNN_WARP_SIZE + laneId;
        if (globalSlot < k) {
            bestDist[t] = warpDist[globalSlot];
            bestIdx[t]  = warpIdx[globalSlot];
        }
    }

    __syncwarp();

    // Update the maxDist to be the distance of the k-th neighbor (largest in the top-k set).
    float kthDist = 0.0f;
    if (laneId == 0) {
        kthDist = warpDist[k - 1];
    }
    kthDist = __shfl_sync(FULL_MASK, kthDist, 0);
    maxDist = kthDist;
}

// CUDA kernel: each warp processes one query point and computes its k nearest neighbors.
//
// query   : array of query points (float2), size query_count
// data    : array of data points (float2), size data_count
// result  : output array of std::pair<int,float>, size query_count * k
//
// result[q * k + j].first  = index of the j-th nearest data point for query q
// result[q * k + j].second = squared distance to that neighbor
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k)
{
    // k must be a power of two in [32, 1024] and divisible by warp size (32).
    if (k <= 0 || k > KNN_MAX_K || (k & (k - 1)) != 0 || (k % KNN_WARP_SIZE) != 0) {
        return; // All threads return before any __syncthreads, so this is safe.
    }

    extern __shared__ unsigned char shared_mem[];

    // Shared memory layout:
    //   - First KNN_DATA_TILE_SIZE float2's: data tile for the entire block.
    //   - Then, for each warp: candidate buffer + scratch area + candidate counter.
    //
    // Per-warp shared memory layout (in bytes), computed from k at runtime:
    //   int candidateCount;
    //   float dist[2 * k];
    //   int   idx [2 * k];
    //
    // We round perWarpBytes up to 8-byte alignment to keep things aligned.
    const size_t dataTileBytes = static_cast<size_t>(KNN_DATA_TILE_SIZE) * sizeof(float2);
    float2 *smemData = reinterpret_cast<float2*>(shared_mem);
    unsigned char *warpSharedBase = shared_mem + dataTileBytes;

    const int warpIdInBlock = threadIdx.x / KNN_WARP_SIZE;
    const int laneId        = threadIdx.x & (KNN_WARP_SIZE - 1);
    const int globalWarpId  = blockIdx.x * KNN_WARPS_PER_BLOCK + warpIdInBlock;
    const bool warpActive   = (globalWarpId < query_count);
    const unsigned FULL_MASK = 0xffffffffu;

    // Compute per-warp shared-memory size using the runtime k.
    size_t perWarpBytes = sizeof(int)
                        + static_cast<size_t>(2 * k) * sizeof(float)
                        + static_cast<size_t>(2 * k) * sizeof(int);
    // Align to 8 bytes for safety.
    perWarpBytes = (perWarpBytes + 7) & ~static_cast<size_t>(7);

    unsigned char *warpBase = warpSharedBase + warpIdInBlock * perWarpBytes;
    int   *candCountPtr = reinterpret_cast<int*>(warpBase);
    float *warpDist     = reinterpret_cast<float*>(candCountPtr + 1);
    int   *warpIdx      = reinterpret_cast<int*>(warpDist + 2 * k);

    // Each warp (query) has per-thread storage for its share of the top-k neighbors.
    const int perThreadK = k / KNN_WARP_SIZE; // guaranteed integer due to k % 32 == 0
    float bestDist[KNN_MAX_K / KNN_WARP_SIZE];
    int   bestIdx [KNN_MAX_K / KNN_WARP_SIZE];

    // Initialize per-warp state.
    float qx = 0.0f;
    float qy = 0.0f;
    float maxDist = CUDART_INF_F; // Distance of the current k-th nearest neighbor.

    if (warpActive) {
        // Load the query point for this warp (lane 0) and broadcast to all lanes.
        if (laneId == 0) {
            float2 q = query[globalWarpId];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);

        // Initialize intermediate top-k result (all distances = +Inf, indices = -1).
        for (int t = 0; t < perThreadK; ++t) {
            bestDist[t] = CUDART_INF_F;
            bestIdx[t]  = -1;
        }

        // Initialize candidate buffer count in shared memory.
        if (laneId == 0) {
            *candCountPtr = 0;
        }
    }

    // Ensure all warps see initialized shared memory before processing tiles.
    __syncthreads();

    // Loop over the data points in tiles and process them from shared memory.
    for (int tileBase = 0; tileBase < data_count; tileBase += KNN_DATA_TILE_SIZE) {
        const int tileSize = min(KNN_DATA_TILE_SIZE, data_count - tileBase);

        // Load the current tile of data points into shared memory cooperatively by the block.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            smemData[i] = data[tileBase + i];
        }

        // Wait for all threads to finish loading the tile.
        __syncthreads();

        // Each active warp processes the tile for its query.
        if (warpActive) {
            for (int i = laneId; i < tileSize; i += KNN_WARP_SIZE) {
                const int globalDataIndex = tileBase + i;
                float2 d = smemData[i];

                // Compute squared Euclidean distance between query and data point.
                float dx = d.x - qx;
                float dy = d.y - qy;
                float dist = dx * dx + dy * dy;

                // Filter by current maxDist (distance of k-th neighbor in intermediate result).
                bool isCandidate = (dist < maxDist);

                unsigned mask = __ballot_sync(FULL_MASK, isCandidate);
                int numCand = __popc(mask);

                if (numCand > 0) {
                    // Read current candidate count (lane 0) and broadcast.
                    int currCount = 0;
                    if (laneId == 0) {
                        currCount = *candCountPtr;
                    }
                    currCount = __shfl_sync(FULL_MASK, currCount, 0);

                    // Check whether adding numCand candidates would overflow the buffer.
                    bool needMerge = (currCount + numCand > k);

                    // If the buffer would overflow, merge existing candidates with the
                    // intermediate top-k result before inserting new ones.
                    if (__any_sync(FULL_MASK, needMerge)) {
                        merge_candidate_buffer(
                            laneId,
                            k,
                            perThreadK,
                            candCountPtr,
                            warpDist,
                            warpIdx,
                            bestDist,
                            bestIdx,
                            maxDist);

                        if (laneId == 0) {
                            *candCountPtr = 0;
                        }
                        __syncwarp();
                    }

                    // Now there is enough space in the candidate buffer for all numCand entries.
                    int baseIndex = 0;
                    if (laneId == 0) {
                        // Use atomicAdd on shared memory to reserve numCand slots in the buffer.
                        // This single atomicAdd per warp batch both updates the number of stored
                        // candidates and provides each new candidate with a base position.
                        baseIndex = atomicAdd(candCountPtr, numCand);
                    }
                    baseIndex = __shfl_sync(FULL_MASK, baseIndex, 0);

                    // Compute this lane's rank among the candidate-producing lanes in the warp.
                    int laneRank = __popc(mask & ((1u << laneId) - 1));

                    // Insert this candidate into the shared-memory buffer at a unique position.
                    if (isCandidate) {
                        int pos = baseIndex + laneRank;
                        // pos is guaranteed to be < k due to the pre-merge capacity check.
                        warpDist[pos] = dist;
                        warpIdx[pos]  = globalDataIndex;
                    }
                }
            }
        }

        // Synchronize before loading the next tile.
        __syncthreads();
    }

    // After processing all tiles, merge any remaining candidates for each active warp.
    if (warpActive) {
        int leftover = 0;
        if (laneId == 0) {
            leftover = *candCountPtr;
        }
        leftover = __shfl_sync(FULL_MASK, leftover, 0);

        if (leftover > 0) {
            merge_candidate_buffer(
                laneId,
                k,
                perThreadK,
                candCountPtr,
                warpDist,
                warpIdx,
                bestDist,
                bestIdx,
                maxDist);

            if (laneId == 0) {
                *candCountPtr = 0;
            }
            __syncwarp();
        }

        // Write the final top-k result from registers to global memory.
        const int outBase = globalWarpId * k;

        for (int t = 0; t < perThreadK; ++t) {
            int globalSlot = t * KNN_WARP_SIZE + laneId;
            if (globalSlot < k) {
                int   idx  = bestIdx[t];
                float dist = bestDist[t];

                std::pair<int, float> &dst = result[outBase + globalSlot];
                dst.first  = idx;
                dst.second = dist;
            }
        }
    }
}

// Host-side interface as requested.
//
// query       : device pointer to float2, size query_count
// data        : device pointer to float2, size data_count
// result      : device pointer to std::pair<int,float>, size query_count * k
// query_count : number of queries
// data_count  : number of data points
// k           : number of neighbors (power of two between 32 and 1024 inclusive)
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    // Compute grid configuration: one warp per query.
    const int threadsPerBlock = KNN_THREADS_PER_BLOCK;
    const int warpsNeeded     = query_count;
    const int blocks          = (warpsNeeded + KNN_WARPS_PER_BLOCK - 1) / KNN_WARPS_PER_BLOCK;

    // Shared memory size per block:
    //   - Data tile: KNN_DATA_TILE_SIZE float2's.
    //   - Per warp: candidate buffer + scratch + candidate counter.
    size_t dataTileBytes = static_cast<size_t>(KNN_DATA_TILE_SIZE) * sizeof(float2);

    size_t perWarpBytes = sizeof(int)
                        + static_cast<size_t>(2 * k) * sizeof(float)
                        + static_cast<size_t>(2 * k) * sizeof(int);
    perWarpBytes = (perWarpBytes + 7) & ~static_cast<size_t>(7); // 8-byte alignment

    size_t sharedBytes = dataTileBytes + static_cast<size_t>(KNN_WARPS_PER_BLOCK) * perWarpBytes;

    knn_kernel<<<blocks, threadsPerBlock, sharedBytes>>>(query,
                                                         query_count,
                                                         data,
                                                         data_count,
                                                         result,
                                                         k);
}