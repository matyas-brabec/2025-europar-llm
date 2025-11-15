#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;    // 4 warps per block => 128 threads
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_K             = 1024; // maximum supported k
constexpr int BATCH_SIZE        = 2048; // number of data points cached per block in shared memory

// Simple POD pair type with the same layout as std::pair<int, float>
struct PairIntFloat {
    int   first;
    float second;
};

static_assert(sizeof(PairIntFloat) == sizeof(std::pair<int, float>),
              "PairIntFloat must match std::pair<int,float> layout");

// Bitonic sort of an array of length n (power of two, 32 <= n <= MAX_K).
// The array is stored in shared memory and cooperatively sorted by a single warp.
static __device__ __forceinline__
void bitonic_sort_shared(float *dist, int *idx, int n)
{
    const unsigned mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ stride;
                if (l > i && l < n) {
                    bool ascending = ((i & size) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    int   ii = idx[i];
                    int   il = idx[l];
                    bool doSwap = (ascending && di > dl) || (!ascending && di < dl);
                    if (doSwap) {
                        dist[i] = dl;
                        dist[l] = di;
                        idx[i]  = il;
                        idx[l]  = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Merge a full (or partially filled) candidate buffer with the current intermediate
// k-NN result for a query. The intermediate result is kept sorted in ascending order.
// The candidate buffer is sorted, merged, and the merged result is sorted again using
// bitonic sort. The candidate buffer is then reset to empty.
// Returns the updated maxDist (distance of the k-th nearest neighbor).
static __device__ __forceinline__
float merge_buffer_with_intermediate(float *interDist, int *interIdx,
                                     float *bufDist,   int *bufIdx,
                                     int k,
                                     int *bufCount,
                                     float currentMaxDist)
{
    const unsigned mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // 1. Sort the candidate buffer (size k) in ascending order
    bitonic_sort_shared(bufDist, bufIdx, k);
    __syncwarp(mask);

    // 2. Merge buffer and intermediate result into a bitonic sequence
    // merged[i] = min(inter[i], buf[k-1-i])
    for (int i = lane; i < k; i += WARP_SIZE) {
        int j = k - 1 - i;
        float di = interDist[i];
        float db = bufDist[j];
        if (db < di) {
            interDist[i] = db;
            interIdx[i]  = bufIdx[j];
        }
    }
    __syncwarp(mask);

    // 3. Sort the merged sequence (stored in interDist/interIdx) in ascending order
    bitonic_sort_shared(interDist, interIdx, k);
    __syncwarp(mask);

    // Update maxDist to distance of k-th nearest neighbor
    float newMaxDist = currentMaxDist;
    if (lane == 0) {
        newMaxDist = interDist[k - 1];
        *bufCount  = 0;
    }
    newMaxDist = __shfl_sync(mask, newMaxDist, 0);

    // Reset candidate buffer contents (optional but keeps invariants simple)
    const float INF = FLT_MAX;
    for (int i = lane; i < k; i += WARP_SIZE) {
        bufDist[i] = INF;
        bufIdx[i]  = -1;
    }
    __syncwarp(mask);

    return newMaxDist;
}

// Each warp processes one query point and finds its k nearest neighbors among all data points.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           int k,
                           PairIntFloat * __restrict__ result)
{
    extern __shared__ unsigned char shared_raw[];
    unsigned char *ptr = shared_raw;

    // Shared memory layout:
    // [0] float2 sData[BATCH_SIZE];
    // [1] float interDist[WARPS_PER_BLOCK][MAX_K];
    // [2] int   interIdx [WARPS_PER_BLOCK][MAX_K];
    // [3] float bufDist  [WARPS_PER_BLOCK][MAX_K];
    // [4] int   bufIdx   [WARPS_PER_BLOCK][MAX_K];
    // [5] int   bufCount [WARPS_PER_BLOCK];

    float2 *sData = reinterpret_cast<float2*>(ptr);
    ptr += BATCH_SIZE * sizeof(float2);

    float *interDistAll = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(float);

    int *interIdxAll = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(int);

    float *bufDistAll = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(float);

    int *bufIdxAll = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * MAX_K * sizeof(int);

    int *bufCountAll = reinterpret_cast<int*>(ptr);
    // ptr += WARPS_PER_BLOCK * sizeof(int); // no further use

    const int tid    = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int lane   = tid & (WARP_SIZE - 1);
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int globalWarpId  = blockIdx.x * warpsPerBlock + warpId;
    const bool warpActive   = (globalWarpId < query_count);

    // Per-warp views into shared memory
    float *interDist = interDistAll + warpId * MAX_K;
    int   *interIdx  = interIdxAll  + warpId * MAX_K;
    float *bufDist   = bufDistAll   + warpId * MAX_K;
    int   *bufIdx    = bufIdxAll    + warpId * MAX_K;
    int   *bufCount  = bufCountAll  + warpId;

    const unsigned fullMask = 0xFFFFFFFFu;

    // Initialize per-warp intermediate result and candidate buffer
    if (warpActive) {
        const float INF = FLT_MAX;
        for (int i = lane; i < k; i += WARP_SIZE) {
            interDist[i] = INF;
            interIdx[i]  = -1;
            bufDist[i]   = INF;
            bufIdx[i]    = -1;
        }
        if (lane == 0) {
            *bufCount = 0;
        }
    }
    __syncthreads();

    float maxDist = FLT_MAX;

    // Load query point for this warp and broadcast within the warp
    float qx = 0.0f, qy = 0.0f;
    if (warpActive) {
        float2 q;
        if (lane == 0) {
            q = query[globalWarpId];
        }
        qx = __shfl_sync(fullMask, q.x, 0);
        qy = __shfl_sync(fullMask, q.y, 0);
    }

    // Process the data points in batches cached in shared memory
    for (int batchStart = 0; batchStart < data_count; batchStart += BATCH_SIZE) {
        int remainingGlobal = data_count - batchStart;
        int batchSize = remainingGlobal < BATCH_SIZE ? remainingGlobal : BATCH_SIZE;

        // Cooperative load of this batch into shared memory
        for (int i = tid; i < batchSize; i += blockDim.x) {
            sData[i] = data[batchStart + i];
        }
        __syncthreads();

        if (warpActive) {
            int localPos = 0;
            while (localPos < batchSize) {
                // If the candidate buffer is full, merge it with the intermediate result
                int bufCountVal = 0;
                if (lane == 0) {
                    bufCountVal = *bufCount;
                }
                bufCountVal = __shfl_sync(fullMask, bufCountVal, 0);

                if (bufCountVal >= k) {
                    maxDist = merge_buffer_with_intermediate(interDist, interIdx,
                                                             bufDist, bufIdx,
                                                             k, bufCount,
                                                             maxDist);
                    continue;
                }

                int remainingPoints = batchSize - localPos;
                if (remainingPoints <= 0) {
                    break;
                }

                int capacity    = k - bufCountVal;
                int activeLanes = capacity < WARP_SIZE ? capacity : WARP_SIZE;
                if (activeLanes > remainingPoints) {
                    activeLanes = remainingPoints;
                }
                if (activeLanes <= 0) {
                    // No capacity or no points; buffer-full case is handled above
                    break;
                }

                bool laneActive = (lane < activeLanes);
                int idxLocal    = localPos + lane;
                float d         = 0.0f;
                int dataIndex   = -1;
                bool isCandidate = false;

                if (laneActive) {
                    float2 p = sData[idxLocal];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    d = dx * dx + dy * dy;
                    dataIndex = batchStart + idxLocal;
                    if (d < maxDist) {
                        isCandidate = true;
                    }
                }

                unsigned int candidateMask = __ballot_sync(fullMask, laneActive && isCandidate);
                int numCandidates = __popc(candidateMask);

                if (numCandidates > 0) {
                    int basePos = 0;
                    if (lane == 0) {
                        basePos = atomicAdd(bufCount, numCandidates);
                    }
                    basePos = __shfl_sync(fullMask, basePos, 0);

                    if (laneActive && isCandidate) {
                        unsigned int maskBefore = candidateMask & ((1u << lane) - 1);
                        int offset = __popc(maskBefore);
                        int pos = basePos + offset;
                        // pos is guaranteed to be < k (numCandidates <= capacity)
                        bufDist[pos] = d;
                        bufIdx[pos]  = dataIndex;
                    }
                }

                localPos += activeLanes;
            } // while localPos
        }

        __syncthreads();
    } // for batchStart

    // After processing all batches, merge any remaining candidates
    if (warpActive) {
        int bufCountVal = 0;
        if (lane == 0) {
            bufCountVal = *bufCount;
        }
        bufCountVal = __shfl_sync(fullMask, bufCountVal, 0);

        if (bufCountVal > 0) {
            maxDist = merge_buffer_with_intermediate(interDist, interIdx,
                                                     bufDist, bufIdx,
                                                     k, bufCount,
                                                     maxDist);
        }

        // Write the final k nearest neighbors for this query
        for (int i = lane; i < k; i += WARP_SIZE) {
            int outIdx = globalWarpId * k + i;
            result[outIdx].first  = interIdx[i];
            result[outIdx].second = interDist[i];
        }
    }
}

// Host interface as specified.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Basic validation; assumes inputs are otherwise valid as per problem statement.
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }
    if (k > MAX_K) {
        // Clamp to MAX_K for safety; in production code you might handle this differently.
        k = MAX_K;
    }

    dim3 blockDim(THREADS_PER_BLOCK);
    int warpsPerBlock = WARPS_PER_BLOCK;
    int totalWarps    = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    dim3 gridDim(totalWarps);

    // Compute required dynamic shared memory size
    size_t sharedMemBytes = 0;
    sharedMemBytes += BATCH_SIZE * sizeof(float2);              // sData
    sharedMemBytes += WARPS_PER_BLOCK * MAX_K * sizeof(float);  // interDist
    sharedMemBytes += WARPS_PER_BLOCK * MAX_K * sizeof(int);    // interIdx
    sharedMemBytes += WARPS_PER_BLOCK * MAX_K * sizeof(float);  // bufDist
    sharedMemBytes += WARPS_PER_BLOCK * MAX_K * sizeof(int);    // bufIdx
    sharedMemBytes += WARPS_PER_BLOCK * sizeof(int);            // bufCount

    // Ensure enough dynamic shared memory is available (for A100/H100 this is supported)
    cudaFuncSetAttribute(knn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(sharedMemBytes));

    PairIntFloat *resultRaw = reinterpret_cast<PairIntFloat*>(result);

    knn_kernel<<<gridDim, blockDim, sharedMemBytes>>>(query, query_count,
                                                      data,  data_count,
                                                      k, resultRaw);
    cudaDeviceSynchronize();
}