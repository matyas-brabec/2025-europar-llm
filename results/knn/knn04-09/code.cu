#include <cuda_runtime.h>
#include <utility>
#include <cfloat>
#include <cassert>

/*
 * Optimized k-NN for 2D points using CUDA.
 *
 * - Each query is processed by exactly one warp (32 threads).
 * - The dataset is traversed in tiles cached in shared memory.
 * - For each query, we maintain a private top-k list (indices + distances)
 *   distributed across the 32 threads of its warp.
 * - The top-k list is updated cooperatively by the warp using warp
 *   intrinsics (__ballot_sync, __shfl_sync).
 * - Candidate distances are generated in batches of 32 per warp and
 *   integrated into the top-k list, potentially inserting multiple
 *   candidates per batch.
 * - After processing all data points, each warp writes its top-k list to
 *   global memory and sorts the k neighbors for its query.
 */

// Basic constants for this implementation.
constexpr int WARP_SIZE          = 32;
constexpr int THREADS_PER_BLOCK  = 256;                     // 8 warps per block
constexpr int WARPS_PER_BLOCK    = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int MAX_K              = 1024;                    // As per problem statement
constexpr int MAX_PER_THREAD     = MAX_K / WARP_SIZE;       // 1024 / 32 = 32
constexpr int TILE_SIZE          = 4096;                    // Number of data points per shared-memory tile

// Simple POD type to use in device code instead of std::pair<int,float>.
struct KnnPair {
    int   idx;
    float dist;
};

/*
 * Recompute the current worst (largest) distance in the warp's distributed
 * top-k list.
 *
 * Layout of the distributed list:
 *   - Each thread has an array distList[0 .. perThread-1].
 *   - The global logical index of distList[j] in the warp's top-k array is:
 *         linearIdx = j * WARP_SIZE + laneId
 *     where laneId is 0..31.
 *   - Thus the global indices 0..k-1 are covered exactly once.
 *
 * This function:
 *   - Computes, in parallel, the maximum distance over all k entries.
 *   - Returns that maximum in worstDist and its global index in worstIdx.
 *   - All threads in the warp receive the same worstDist/worstIdx via
 *     warp-wide broadcast.
 */
__device__ __forceinline__
void warp_recompute_worst(int perThread,
                          float (&distList)[MAX_PER_THREAD],
                          float &worstDist,
                          int &worstIdx)
{
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Compute per-thread local maximum over its perThread entries.
    float localMax = distList[0];
    int   localPos = 0;

#pragma unroll
    for (int i = 1; i < MAX_PER_THREAD; ++i) {
        if (i < perThread) {
            float v = distList[i];
            if (v > localMax) {
                localMax = v;
                localPos = i;
            }
        }
    }

    // Map local position to global linear index in [0, k-1].
    int linearIdx = localPos * WARP_SIZE + lane;

    // Warp-wide argmax reduction on (localMax, linearIdx).
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float otherMax = __shfl_down_sync(mask, localMax, offset);
        int   otherIdx = __shfl_down_sync(mask, linearIdx, offset);
        if (otherMax > localMax) {
            localMax = otherMax;
            linearIdx = otherIdx;
        }
    }

    // Lane 0 stores the final result, then broadcast to the warp.
    if (lane == 0) {
        worstDist = localMax;
        worstIdx  = linearIdx;
    }
    worstDist = __shfl_sync(mask, worstDist, 0);
    worstIdx  = __shfl_sync(mask, worstIdx, 0);
}

/*
 * Main k-NN kernel.
 *
 * Parameters:
 *   query       - array of query points (float2), length query_count
 *   query_count - number of queries
 *   data        - array of data points (float2), length data_count
 *   data_count  - number of data points
 *   k           - number of neighbors to find (power of 2 in [32, 1024])
 *   result      - output array of length query_count * k, where for query i
 *                 the entries [i*k .. i*k + k-1] contain (index, distance)
 *                 pairs for its k nearest neighbors.
 *
 * Each block:
 *   - Loads tiles of the data array into shared memory cooperatively.
 *   - Contains WARPS_PER_BLOCK warps, each assigned to one query.
 *
 * Each warp:
 *   - Maintains a private top-k list distributed across threads.
 *   - For each tile, computes squared distances to its query using the
 *     cached data in shared memory.
 *   - Updates the top-k list using warp-synchronous operations, handling
 *     multiple candidate insertions per 32-candidate batch.
 */
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           int k,
                           KnnPair * __restrict__ result)
{
    extern __shared__ float2 sh_data[]; // Shared tile of data points

    const unsigned fullMask = 0xffffffffu;
    const int lane          = threadIdx.x & (WARP_SIZE - 1);
    const int warpInBlock   = threadIdx.x >> 5; // threadIdx.x / WARP_SIZE
    const int warpGlobal    = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;

    const bool isActiveWarp = (warpGlobal < query_count);

    // Per-thread number of entries in the distributed top-k list.
    // Assumes k is a power of two and >= 32, hence divisible by 32.
    const int perThread = k / WARP_SIZE;

    // Per-thread portion of the warp's top-k list.
    float bestDist[MAX_PER_THREAD];
    int   bestIdx [MAX_PER_THREAD];

    // Initialize the per-thread top-k list with "infinite" distances.
#pragma unroll
    for (int i = 0; i < MAX_PER_THREAD; ++i) {
        if (i < perThread) {
            bestDist[i] = FLT_MAX;
            bestIdx[i]  = -1;
        }
    }

    // Current worst distance and its global index in the distributed top-k list.
    float worstDist = FLT_MAX;
    int   worstIdx  = 0;

    // Load the warp's query point into registers and broadcast across the warp.
    float2 q;
    if (isActiveWarp) {
        if (lane == 0) {
            q = query[warpGlobal];
        }
        q.x = __shfl_sync(fullMask, q.x, 0);
        q.y = __shfl_sync(fullMask, q.y, 0);
    }

    // Process the data in tiles cached in shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        int tileSize = data_count - tileStart;
        if (tileSize > TILE_SIZE) tileSize = TILE_SIZE;

        // Cooperative loading of the tile into shared memory by the whole block.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            sh_data[i] = data[tileStart + i];
        }
        __syncthreads();

        // Only warps that correspond to a valid query perform k-NN work.
        if (isActiveWarp) {
            // Traverse the tile in batches of 32 points per warp.
            for (int base = 0; base < tileSize; base += WARP_SIZE) {
                const int idxInTile = base + lane;
                bool valid = (idxInTile < tileSize);

                float candDist = 0.0f;
                int   candIdx  = -1;

                // Compute squared Euclidean distance for this thread's candidate.
                if (valid) {
                    const float2 p = sh_data[idxInTile];
                    const float dx = p.x - q.x;
                    const float dy = p.y - q.y;
                    candDist = dx * dx + dy * dy;
                    candIdx  = tileStart + idxInTile;
                }

                /*
                 * Update the warp's top-k list with the candidates from this batch.
                 *
                 * - Each thread holds one candidate (distance + index) in registers.
                 * - We repeatedly:
                 *     * Find which threads currently have candidates smaller than the
                 *       current worst distance in the top-k list.
                 *     * Select one such thread (the lowest lane in the mask).
                 *     * Broadcast its candidate to the warp.
                 *     * Replace the current worst entry in the top-k list with that
                 *       candidate.
                 *     * Recompute the worst entry in the updated top-k list.
                 *     * Mark the selected thread's candidate as consumed.
                 * - This loop may insert multiple candidates from the batch if
                 *   several are better than the evolving worst distance.
                 */
                while (true) {
                    const float curWorst = worstDist;

                    // Mask of threads whose candidate is valid and beats the current worst.
                    const unsigned betterMask =
                        __ballot_sync(fullMask, valid && (candDist < curWorst));

                    if (betterMask == 0u) {
                        // No more candidates in this batch can improve the top-k list.
                        break;
                    }

                    // Select one candidate lane to insert (lowest set bit in mask).
                    const int srcLane = __ffs(betterMask) - 1;

                    // Broadcast the chosen candidate to all threads in the warp.
                    const float chosenDist = __shfl_sync(fullMask, candDist, srcLane);
                    const int   chosenIdx  = __shfl_sync(fullMask, candIdx,  srcLane);

                    // Compute which thread and local slot own the current worst entry.
                    const int ownerLane   = worstIdx % WARP_SIZE;
                    const int ownerOffset = worstIdx / WARP_SIZE;

                    // The owning thread updates its local segment of the top-k list.
                    if (lane == ownerLane) {
                        bestDist[ownerOffset] = chosenDist;
                        bestIdx [ownerOffset] = chosenIdx;
                    }

                    // Recompute the worst entry in the updated top-k list cooperatively.
                    warp_recompute_worst(perThread, bestDist, worstDist, worstIdx);

                    // Consume the candidate so it won't be considered again.
                    if (lane == srcLane) {
                        valid = false;
                    }
                }
            } // end for base
        }

        // Ensure all threads are done using this tile before loading the next one.
        __syncthreads();
    } // end for tileStart

    // Only warps corresponding to valid queries write results.
    if (isActiveWarp) {
        const int baseOut = warpGlobal * k;

        /*
         * Write the warp's distributed top-k list to global memory in a
         * contiguous fashion. Given our logical layout:
         *   global index = localIdx * WARP_SIZE + lane
         * this results in fully coalesced writes.
         */
#pragma unroll
        for (int j = 0; j < MAX_PER_THREAD; ++j) {
            if (j < perThread) {
                const int linear = j * WARP_SIZE + lane;
                if (linear < k) {
                    KnnPair p;
                    p.idx  = bestIdx[j];
                    p.dist = bestDist[j];
                    result[baseOut + linear] = p;
                }
            }
        }

        /*
         * Final sort of the k neighbors for this query by ascending distance.
         *
         * We perform a simple insertion sort in-place in global memory.
         * This is executed by lane 0 only; k <= 1024, so O(k^2) comparisons
         * are acceptable overhead relative to the O(data_count) distance
         * computations per query.
         */
        if (lane == 0) {
            KnnPair *out = result + baseOut;
            for (int i = 1; i < k; ++i) {
                KnnPair key = out[i];
                float keyDist = key.dist;
                int j = i - 1;
                while (j >= 0 && out[j].dist > keyDist) {
                    out[j + 1] = out[j];
                    --j;
                }
                out[j + 1] = key;
            }
        }
    }
}

/*
 * Host interface:
 *
 *   void run_knn(const float2 *query, int query_count,
 *                const float2 *data, int data_count,
 *                std::pair<int, float> *result, int k);
 *
 * All pointers (query, data, result) refer to device memory allocated with
 * cudaMalloc. This function configures and launches the kernel.
 */
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Basic validations according to the problem constraints.
    assert(k >= 32 && k <= MAX_K);
    assert((k & (k - 1)) == 0);    // k is power of two
    assert(data_count >= k);       // guaranteed by the problem statement

    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int warpsPerBlock   = WARPS_PER_BLOCK;

    // One warp per query.
    const int totalWarps = query_count;
    const int numBlocks  = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory for one tile of data points.
    const size_t sharedMemBytes = TILE_SIZE * sizeof(float2);

    // Reinterpret the std::pair<int,float>* as KnnPair* for device access.
    KnnPair *deviceResult = reinterpret_cast<KnnPair*>(result);

    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        query,
        query_count,
        data,
        data_count,
        k,
        deviceResult
    );

    // The caller can choose whether to synchronize or check errors.
    // cudaDeviceSynchronize(); // Optional: uncomment if needed.
}