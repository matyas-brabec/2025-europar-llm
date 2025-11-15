// CUDA k-NN (k-nearest neighbors) for 2D points, using one warp (32 threads) per query.
//
// The kernel processes the data points in batches. Each batch of data points is loaded into
// shared memory by the whole block. Each warp then computes distances from its own query
// point to the cached data points. Distances below the current max_distance are added to
// a per-warp candidate buffer using atomicAdd. When the candidate buffer is sufficiently
// full, it is merged with the warp's intermediate k-NN result using a warp-cooperative
// selection algorithm. After all batches are processed, any remaining candidates are
// merged, and the final k nearest neighbors (indices and squared distances) are written
// to the output array.
//
// Assumptions:
// - Target GPU: modern NVIDIA data-center GPU (e.g., A100, H100).
// - k is a power of two in [32, 1024].
// - data_count >= k.
// - query, data, and result are pointers to device memory allocated via cudaMalloc.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Tuneable constants for this implementation.
static constexpr int WARP_SIZE        = 32;    // Hardware warp size.
static constexpr int WARPS_PER_BLOCK  = 4;     // Warps (queries) per block.
static constexpr int K_MAX            = 1024;  // Maximum supported k.
static constexpr int DATA_BATCH       = 1024;  // Number of data points cached per batch in shared memory.

// Warp-cooperative merge of candidates into the current top-k result for one warp.
//
// Parameters:
//   warpId       - warp index within the block (0..WARPS_PER_BLOCK-1)
//   k            - actual k for this run (<= K_MAX)
//   candCount    - number of candidates currently stored for this warp
//   s_res_idx    - [WARPS_PER_BLOCK * K_MAX] shared array: result indices
//   s_res_dist   - [WARPS_PER_BLOCK * K_MAX] shared array: result distances
//   s_cand_idx   - [WARPS_PER_BLOCK * K_MAX] shared array: candidate indices
//   s_cand_dist  - [WARPS_PER_BLOCK * K_MAX] shared array: candidate distances
//   s_tmp_idx    - [WARPS_PER_BLOCK * K_MAX] shared array: temporary indices for merged result
//   s_tmp_dist   - [WARPS_PER_BLOCK * K_MAX] shared array: temporary distances for merged result
//   max_distance - (in/out) current distance of the k-th nearest neighbor
//
// Algorithm:
//   - Consider the union of the existing result set (k entries) and the candidate set
//     (candCount entries), i.e., up to k + candCount <= 2k items.
//   - Repeatedly (k times) select the smallest remaining distance from this union using
//     warp-level parallel reduction; store it into s_tmp_*.
//   - The selected item is marked as used by setting its distance in the union to +INF.
//   - After k selections, tmp arrays hold the k smallest distances in sorted order.
//   - Copy tmp back to result arrays and recompute max_distance as the maximum of the
//     k distances (i.e., the k-th nearest neighbor).
__device__ __forceinline__
void merge_knn_warp(
    int   warpId,
    int   k,
    int   candCount,
    int * s_res_idx,
    float* s_res_dist,
    int * s_cand_idx,
    float* s_cand_dist,
    int * s_tmp_idx,
    float* s_tmp_dist,
    float& max_distance)
{
    const unsigned fullMask = 0xFFFFFFFFu;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);

    const int warpBase = warpId * K_MAX;  // Base offset for this warp in per-warp arrays.
    const int tmpBase  = warpId * K_MAX;

    const int unionCount = k + candCount; // Total elements considered in union (res + cand).

    // Select k smallest distances from the union.
    for (int outPos = 0; outPos < k; ++outPos) {

        // Each lane scans a strided subset of the union for its local minimum.
        float bestDist = CUDART_INF_F;
        int   bestPos  = -1;

        for (int i = laneId; i < unionCount; i += WARP_SIZE) {
            float d = (i < k)
                      ? s_res_dist[warpBase + i]
                      : s_cand_dist[warpBase + (i - k)];
            if (d < bestDist) {
                bestDist = d;
                bestPos  = i;
            }
        }

        // Warp-level reduction to find global minimum (distance, position).
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            float otherDist = __shfl_down_sync(fullMask, bestDist, offset);
            int   otherPos  = __shfl_down_sync(fullMask, bestPos,  offset);
            if (otherDist < bestDist || (otherDist == bestDist && otherPos < bestPos)) {
                bestDist = otherDist;
                bestPos  = otherPos;
            }
        }

        // Broadcast winning (distance, position) from lane 0.
        bestDist = __shfl_sync(fullMask, bestDist, 0);
        bestPos  = __shfl_sync(fullMask, bestPos,  0);

        // Lane 0 records the selected neighbor into the temporary result and
        // marks the source position in the union as used by setting its distance to +INF.
        if (laneId == 0) {
            int bestIdx;
            if (bestPos < k) {
                bestIdx = s_res_idx[warpBase + bestPos];
                s_res_dist[warpBase + bestPos] = CUDART_INF_F;
            } else {
                int cpos = bestPos - k;
                bestIdx = s_cand_idx[warpBase + cpos];
                s_cand_dist[warpBase + cpos] = CUDART_INF_F;
            }
            s_tmp_idx [tmpBase  + outPos] = bestIdx;
            s_tmp_dist[tmpBase  + outPos] = bestDist;
        }

        __syncwarp(fullMask);
    }

    // Copy tmp result back into the result arrays for this warp.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        s_res_idx [warpBase + i] = s_tmp_idx [tmpBase + i];
        s_res_dist[warpBase + i] = s_tmp_dist[tmpBase + i];
    }
    __syncwarp(fullMask);

    // Recompute max_distance = maximum distance in the updated k-NN list.
    float localMax = -CUDART_INF_F;
    for (int i = laneId; i < k; i += WARP_SIZE) {
        float d = s_res_dist[warpBase + i];
        if (d > localMax) localMax = d;
    }

    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(fullMask, localMax, offset);
        if (other > localMax) localMax = other;
    }

    max_distance = __shfl_sync(fullMask, localMax, 0);
}

// Kernel implementing k-NN for 2D points using one warp per query.
__global__
void knn_kernel(
    const float2* __restrict__ query,
    int                       query_count,
    const float2* __restrict__ data,
    int                       data_count,
    std::pair<int, float>*    __restrict__ result,
    int                       k)
{
    const int tid    = threadIdx.x;
    const int laneId = tid & (WARP_SIZE - 1);
    const int warpId = tid / WARP_SIZE;

    const int warpsPerBlock = WARPS_PER_BLOCK;
    const int warpGlobal    = blockIdx.x * warpsPerBlock + warpId;
    const bool isActiveWarp = (warpGlobal < query_count);
    const unsigned fullMask = 0xFFFFFFFFu;

    // Merge threshold for candidate buffer:
    // To avoid overflowing the buffer of size k when up to WARP_SIZE elements
    // can be added in one step, we start merging when candCount >= k - WARP_SIZE + 1.
    const int merge_threshold = (k <= WARP_SIZE) ? 1 : (k - WARP_SIZE + 1);

    // Shared memory layout:
    // [0]   float2  s_data[DATA_BATCH]
    // [1]   int     s_res_idx [WARPS_PER_BLOCK * K_MAX]
    // [2]   float   s_res_dist[WARPS_PER_BLOCK * K_MAX]
    // [3]   int     s_cand_idx[WARPS_PER_BLOCK * K_MAX]
    // [4]   float   s_cand_dist[WARPS_PER_BLOCK * K_MAX]
    // [5]   int     s_tmp_idx [WARPS_PER_BLOCK * K_MAX]
    // [6]   float   s_tmp_dist[WARPS_PER_BLOCK * K_MAX]
    // [7]   int     s_cand_count[WARPS_PER_BLOCK]
    extern __shared__ unsigned char shared_mem[];
    unsigned char* ptr = shared_mem;

    float2* s_data = reinterpret_cast<float2*>(ptr);
    ptr += DATA_BATCH * sizeof(float2);

    int* s_res_idx = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(int);

    float* s_res_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(float);

    int* s_cand_idx = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(int);

    float* s_cand_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(float);

    int* s_tmp_idx = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(int);

    float* s_tmp_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K_MAX * sizeof(float);

    int* s_cand_count = reinterpret_cast<int*>(ptr);

    const int warpBase = warpId * K_MAX;

    // Initialize per-warp intermediate result (top-k list) and candidate count.
    if (isActiveWarp) {
        for (int i = laneId; i < k; i += WARP_SIZE) {
            s_res_idx [warpBase + i] = -1;
            s_res_dist[warpBase + i] = CUDART_INF_F;
        }
    }

    if (laneId == 0) {
        s_cand_count[warpId] = 0;
    }

    __syncwarp(fullMask);

    // Load query point for this warp into registers and broadcast via shuffles.
    float2 q{0.0f, 0.0f};
    if (isActiveWarp) {
        if (laneId == 0) {
            q = query[warpGlobal];
        }
        q.x = __shfl_sync(fullMask, q.x, 0);
        q.y = __shfl_sync(fullMask, q.y, 0);
    }

    // max_distance is the distance of the current k-th nearest neighbor.
    // Start with +INF so that all points are initially accepted as candidates
    // until k finite distances have been collected.
    float max_distance = CUDART_INF_F;

    // Process data points in batches, caching each batch into shared memory.
    for (int dataBase = 0; dataBase < data_count; dataBase += DATA_BATCH) {

        const int remaining = data_count - dataBase;
        const int batchSize = (remaining < DATA_BATCH) ? remaining : DATA_BATCH;

        // Load this batch of data points into shared memory (all threads cooperate).
        for (int i = tid; i < batchSize; i += blockDim.x) {
            s_data[i] = data[dataBase + i];
        }

        __syncthreads();

        if (isActiveWarp) {
            // Process the batch in groups of WARP_SIZE points so that each lane
            // handles at most one point per group.
            for (int jb = 0; jb < batchSize; jb += WARP_SIZE) {
                const int j = jb + laneId;
                const bool valid = (j < batchSize);

                float dist = 0.0f;
                int   globalDataIdx = dataBase + j;
                bool  isCandidate = false;

                if (valid) {
                    float2 p = s_data[j];
                    const float dx = p.x - q.x;
                    const float dy = p.y - q.y;
                    dist = dx * dx + dy * dy;

                    if (dist < max_distance) {
                        isCandidate = true;
                    }
                }

                // Insert candidate into the per-warp candidate buffer using atomicAdd.
                if (isCandidate) {
                    const int pos = atomicAdd(&s_cand_count[warpId], 1);
                    if (pos < k) {
                        s_cand_idx [warpBase + pos] = globalDataIdx;
                        s_cand_dist[warpBase + pos] = dist;
                    }
                    // If pos >= k, the candidate is ignored; the merge_threshold logic
                    // plus k >= WARP_SIZE ensures this should be extremely rare.
                }

                __syncwarp(fullMask);

                // After processing this group of up to WARP_SIZE points, check whether
                // the candidate buffer for this warp should be merged.
                int  candCount = 0;
                bool do_merge  = false;

                if (laneId == 0) {
                    candCount = s_cand_count[warpId];
                    if (candCount >= merge_threshold) {
                        do_merge = true;
                    }
                }

                do_merge  = __shfl_sync(fullMask, do_merge,  0);
                candCount = __shfl_sync(fullMask, candCount, 0);

                if (do_merge && candCount > 0) {
                    merge_knn_warp(
                        warpId,
                        k,
                        candCount,
                        s_res_idx,
                        s_res_dist,
                        s_cand_idx,
                        s_cand_dist,
                        s_tmp_idx,
                        s_tmp_dist,
                        max_distance);

                    if (laneId == 0) {
                        s_cand_count[warpId] = 0;
                    }
                    __syncwarp(fullMask);
                }
            }
        }

        __syncthreads();
    }

    // After the last batch, merge any remaining candidates for this warp.
    if (isActiveWarp) {
        int finalCandCount = 0;
        if (laneId == 0) {
            finalCandCount = s_cand_count[warpId];
        }
        finalCandCount = __shfl_sync(fullMask, finalCandCount, 0);

        if (finalCandCount > 0) {
            merge_knn_warp(
                warpId,
                k,
                finalCandCount,
                s_res_idx,
                s_res_dist,
                s_cand_idx,
                s_cand_dist,
                s_tmp_idx,
                s_tmp_dist,
                max_distance);

            if (laneId == 0) {
                s_cand_count[warpId] = 0;
            }
            __syncwarp(fullMask);
        }

        // Write the final k-NN results for this query to global memory.
        // The result list is sorted in ascending distance.
        for (int i = laneId; i < k; i += WARP_SIZE) {
            std::pair<int, float> out;
            out.first  = s_res_idx [warpBase + i];
            out.second = s_res_dist[warpBase + i];
            result[warpGlobal * k + i] = out;
        }
    }
}

// Host interface: launch the k-NN kernel.
void run_knn(
    const float2* query,
    int           query_count,
    const float2* data,
    int           data_count,
    std::pair<int, float>* result,
    int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // k is guaranteed (by problem statement) to be a power of two in [32, 1024].
    if (k > K_MAX) {
        // For safety; should not happen under given constraints.
        k = K_MAX;
    }

    const int threadsPerBlock = WARPS_PER_BLOCK * WARP_SIZE;
    const int numBlocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Compute required dynamic shared memory size.
    size_t sharedBytes = 0;
    sharedBytes += sizeof(float2) * DATA_BATCH;                  // s_data
    sharedBytes += sizeof(int)   * WARPS_PER_BLOCK * K_MAX;      // s_res_idx
    sharedBytes += sizeof(float) * WARPS_PER_BLOCK * K_MAX;      // s_res_dist
    sharedBytes += sizeof(int)   * WARPS_PER_BLOCK * K_MAX;      // s_cand_idx
    sharedBytes += sizeof(float) * WARPS_PER_BLOCK * K_MAX;      // s_cand_dist
    sharedBytes += sizeof(int)   * WARPS_PER_BLOCK * K_MAX;      // s_tmp_idx
    sharedBytes += sizeof(float) * WARPS_PER_BLOCK * K_MAX;      // s_tmp_dist
    sharedBytes += sizeof(int)   * WARPS_PER_BLOCK;              // s_cand_count

    // Optionally request enough dynamic shared memory (mainly relevant on
    // architectures that require opt-in for >48 KB).
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sharedBytes));

    // Launch kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k);

    // Synchronize to ensure completion before returning (optional; can be removed
    // if the caller is responsible for synchronization).
    cudaDeviceSynchronize();
}