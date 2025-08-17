#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA kernel for k-NN (2D, squared Euclidean) with one warp (32 threads) per query.
// - Data is processed in tiles loaded into shared memory by the entire block.
// - Each warp maintains a private intermediate top-k (distributed across lanes, in registers).
// - Each warp has a shared-memory candidate buffer of size k (indices and distances) with an atomic counter.
// - Distances below the current max_distance (k-th best) are added to the candidate buffer.
// - When the buffer fills up (detected via atomicAdd overflow), the warp merges candidates into the intermediate result.
// - After the final tile, any remaining candidates in the buffer are merged.
// - Final results for each query are written in sorted ascending order of distance.
// Assumptions:
// - k is a power of two between 32 and 1024 inclusive.
// - data_count >= k.
// - All pointers are device pointers obtained via cudaMalloc.
// - query_count and data_count are large, typical for GPU workloads.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable kernel parameters chosen for modern NVIDIA datacenter GPUs (A100/H100):
// - 8 warps per block (256 threads).
// - Shared tile of 4096 float2 points (32KB).
// - Per-warp candidate buffer sized to MAX_K (1024).
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS = 4096;
static constexpr int MAX_K = 1024;
static constexpr int MAX_PER_LANE = MAX_K / WARP_SIZE;

struct PairDevice {
    int   first;
    float second;
};

__device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

__device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x / WARP_SIZE;
}

// Shared memory:
// - dataTile: cached data points for current tile
// - cand_idx/cand_dist: per-warp candidate buffers of size MAX_K (we will only use first k slots).
// - cand_count: per-warp candidate counters.
__shared__ float2 s_dataTile[TILE_POINTS];
__shared__ int    s_cand_idx[WARPS_PER_BLOCK][MAX_K];
__shared__ float  s_cand_dist[WARPS_PER_BLOCK][MAX_K];
__shared__ int    s_cand_count[WARPS_PER_BLOCK];

// Find the current worst (maximum distance) entry across the warp's private top-k.
// Inputs:
// - best_d: per-lane array of distances (length chunk_len).
// - chunk_len: number of valid entries per lane (k / 32).
// Outputs (via references):
// - worstVal: the maximum distance among all top-k entries.
// - worstLane: the lane ID (0..31) that holds the worst entry.
// - worstPos: the index within worstLane's local array (0..chunk_len-1).
__device__ __forceinline__ void warpFindWorst(const float best_d[MAX_PER_LANE],
                                              int chunk_len,
                                              float &worstVal,
                                              int &worstLane,
                                              int &worstPos) {
    const unsigned mask = __activemask();
    const int lane = lane_id();

    // Find local maximum and its local position
    float localMax = -CUDART_INF_F;
    int localPos = 0;
    #pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
        if (i < chunk_len) {
            float v = best_d[i];
            if (v > localMax) {
                localMax = v;
                localPos = i;
            }
        }
    }

    // Warp-wide reduction to find global maximum and its lane+pos
    float v = localMax;
    int l = lane;
    int p = localPos;

    // Reduce in log2(32)=5 steps using shfl_down
    // Note: If tie, keep the earlier lane by leaving condition as (ov > v).
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, v, offset);
        int   ol = __shfl_down_sync(mask, l, offset);
        int   op = __shfl_down_sync(mask, p, offset);
        if (ov > v) {
            v = ov; l = ol; p = op;
        }
    }

    // Broadcast results from lane 0
    worstVal  = __shfl_sync(mask, v, 0);
    worstLane = __shfl_sync(mask, l, 0);
    worstPos  = __shfl_sync(mask, p, 0);
}

// Recompute and broadcast the current worst (k-th) distance across the warp's private top-k.
// Returns the new worst distance value in maxDist (in all lanes).
__device__ __forceinline__ void warpRecomputeMaxDist(const float best_d[MAX_PER_LANE],
                                                     int chunk_len,
                                                     float &maxDist) {
    const unsigned mask = __activemask();

    float localMax = -CUDART_INF_F;
    #pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
        if (i < chunk_len) {
            float v = best_d[i];
            if (v > localMax) localMax = v;
        }
    }
    // Warp max reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, localMax, offset);
        if (ov > localMax) localMax = ov;
    }
    maxDist = __shfl_sync(mask, localMax, 0);
}

// Attempt to insert a candidate (idx, dist) into the warp's private top-k if it improves the result.
// This function uses the warp to cooperatively locate the current worst and update it.
// Updates maxDist whenever the k-th neighbor changes.
__device__ __forceinline__ void tryInsertCandidate(int idx, float dist,
                                                   float best_d[MAX_PER_LANE],
                                                   int   best_i[MAX_PER_LANE],
                                                   int chunk_len,
                                                   float &maxDist) {
    float worstVal;
    int worstLane, worstPos;
    warpFindWorst(best_d, chunk_len, worstVal, worstLane, worstPos);

    const int lane = lane_id();
    if (dist < worstVal) {
        // Replace the current worst element with the new candidate
        if (lane == worstLane) {
            best_d[worstPos] = dist;
            best_i[worstPos] = idx;
        }
        // Ensure the replacement is visible within the warp before recomputing maxDist
        __syncwarp();

        // Recompute the k-th distance after the update
        warpRecomputeMaxDist(best_d, chunk_len, maxDist);
    }
    // If not better than worstVal, nothing to do
}

// Merge the candidate buffer (per warp) into the private top-k using the warp.
// Only the first min(count, k) entries are valid in the buffer.
// After merge, the candidate count is reset to 0.
// maxDist is updated to reflect the k-th neighbor after merge.
__device__ __forceinline__ void flush_and_merge_candidates(int warpSlot,
                                                           int k,
                                                           int chunk_len,
                                                           float best_d[MAX_PER_LANE],
                                                           int   best_i[MAX_PER_LANE],
                                                           float &maxDist) {
    const unsigned mask = __activemask();
    // Synchronize warp before reading shared memory buffer
    __syncwarp(mask);

    int count = s_cand_count[warpSlot];
    int valid = (count < k) ? count : k;

    // Process candidates one by one; all lanes cooperate on each insertion.
    for (int j = 0; j < valid; ++j) {
        int c_idx = 0;
        float c_dist = 0.0f;
        if (lane_id() == 0) {
            c_idx = s_cand_idx[warpSlot][j];
            c_dist = s_cand_dist[warpSlot][j];
        }
        c_idx  = __shfl_sync(mask, c_idx, 0);
        c_dist = __shfl_sync(mask, c_dist, 0);

        // Only attempt insert if strictly better than current maxDist
        if (c_dist < maxDist) {
            tryInsertCandidate(c_idx, c_dist, best_d, best_i, chunk_len, maxDist);
        }
        __syncwarp(mask);
    }

    // Reset candidate count to zero
    if (lane_id() == 0) {
        s_cand_count[warpSlot] = 0;
    }
    __syncwarp(mask);
}

// Per-lane insertion sort (ascending) on the local chunk to prepare for k-way merge.
// length = chunk_len <= MAX_PER_LANE.
__device__ __forceinline__ void lane_insertion_sort(float d[MAX_PER_LANE], int idx[MAX_PER_LANE], int length) {
    #pragma unroll
    for (int i = 1; i < MAX_PER_LANE; ++i) {
        if (i >= length) break;
        float keyd = d[i];
        int keyi = idx[i];
        int j = i - 1;
        for (; j >= 0; --j) {
            float v = d[j];
            if (v <= keyd) break;
            d[j + 1] = v;
            idx[j + 1] = idx[j];
        }
        d[j + 1] = keyd;
        idx[j + 1] = keyi;
    }
}

// Warp-wise argmin: find lane with minimal 'val'. Returns (minVal, minLane).
__device__ __forceinline__ void warp_argmin(float val, int &minLane, float &minVal) {
    const unsigned mask = __activemask();
    int lane = lane_id();
    float v = val;
    int l = lane;
    // Reduce to find minimum value with its lane
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, v, offset);
        int   ol = __shfl_down_sync(mask, l, offset);
        if (ov < v) { v = ov; l = ol; }
    }
    minVal = __shfl_sync(mask, v, 0);
    minLane = __shfl_sync(mask, l, 0);
}

__global__ void knn_kernel(const float2 *__restrict__ query,
                           int query_count,
                           const float2 *__restrict__ data,
                           int data_count,
                           PairDevice *__restrict__ result,
                           int k) {
    const int warpInBlock = warp_id_in_block();
    const int lane = lane_id();
    const int warpGlobal = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    if (warpGlobal >= query_count) return;

    // Per-warp candidate counter reset
    if (lane == 0) s_cand_count[warpInBlock] = 0;
    __syncwarp();

    // Load the warp's query point and broadcast to all lanes
    float2 q;
    if (lane == 0) {
        q = query[warpGlobal];
    }
    q.x = __shfl_sync(0xffffffffu, q.x, 0);
    q.y = __shfl_sync(0xffffffffu, q.y, 0);

    // Per-warp parameters
    const int chunk_len = k / WARP_SIZE; // since k is power of two and >= 32
    float best_d[MAX_PER_LANE];
    int   best_i[MAX_PER_LANE];

    // Initialize intermediate top-k to +inf distances and invalid indices
    #pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
        if (i < chunk_len) {
            best_d[i] = CUDART_INF_F;
            best_i[i] = -1;
        }
    }
    float maxDist = CUDART_INF_F; // k-th neighbor distance (threshold)

    // Process data in tiles cached in shared memory
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tileCount = data_count - base;
        if (tileCount > TILE_POINTS) tileCount = TILE_POINTS;

        // Load tile into shared memory cooperatively by entire block
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
            s_dataTile[t] = data[base + t];
        }
        __syncthreads();

        // Each warp processes the tile for its query
        for (int i = lane; i < tileCount; i += WARP_SIZE) {
            float2 p = s_dataTile[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = fmaf(dy, dy, dx * dx); // squared L2

            // Quick filter by current threshold
            if (dist < maxDist) {
                // Try to push candidate; if buffer overflows, flush/merge then retry.
                bool done = false;
                while (!done) {
                    int pos = atomicAdd(&s_cand_count[warpInBlock], 1);
                    bool overflow = (pos >= k);
                    if (!overflow) {
                        s_cand_dist[warpInBlock][pos] = dist;
                        s_cand_idx[warpInBlock][pos]  = base + i;
                        done = true;
                    }
                    // If any lane overflowed, flush and merge now (all lanes participate)
                    if (__any_sync(0xffffffffu, overflow)) {
                        flush_and_merge_candidates(warpInBlock, k, chunk_len, best_d, best_i, maxDist);
                        // After flush, if this lane had overflow, it will retry insertion
                    }
                }
            }
        }

        // After finishing the tile, flush remaining candidates (if any) to quickly tighten maxDist
        int needFlush = 0;
        if (lane == 0) needFlush = (s_cand_count[warpInBlock] > 0);
        needFlush = __shfl_sync(0xffffffffu, needFlush, 0);
        if (needFlush) {
            flush_and_merge_candidates(warpInBlock, k, chunk_len, best_d, best_i, maxDist);
        }

        __syncthreads();
    }

    // Ensure no leftover (safety, though we flushed at tile boundaries)
    int leftover = 0;
    if (lane == 0) leftover = (s_cand_count[warpInBlock] > 0);
    leftover = __shfl_sync(0xffffffffu, leftover, 0);
    if (leftover) {
        flush_and_merge_candidates(warpInBlock, k, chunk_len, best_d, best_i, maxDist);
    }

    // Sort each lane's local portion of the top-k to prepare for k-way merge
    lane_insertion_sort(best_d, best_i, chunk_len);
    __syncwarp();

    // Perform 32-way merge to output final sorted results
    const int outBase = warpGlobal * k;
    int head = 0; // per-lane head within its sorted local array
    for (int outPos = 0; outPos < k; ++outPos) {
        float myVal = (head < chunk_len) ? best_d[head] : CUDART_INF_F;
        int winnerLane;
        float winnerVal;
        warp_argmin(myVal, winnerLane, winnerVal);

        // Broadcast winner's index (data point id) and write to output in order
        int winnerIdx = __shfl_sync(0xffffffffu, (lane == winnerLane && head < chunk_len) ? best_i[head] : 0, winnerLane);

        // Increment head of winner lane
        if (lane == winnerLane) {
            head++;
        }
        // Write output (single lane to avoid write conflicts; lane 0 does the store)
        if (lane == 0) {
            result[outBase + outPos].first  = winnerIdx;
            result[outBase + outPos].second = winnerVal;
        }
        __syncwarp();
    }
}

// Host wrapper to launch the kernel.
// query, data, result must be device pointers allocated via cudaMalloc.
// k must be a power of two between 32 and 1024 inclusive.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Reinterpret std::pair<int,float> as a plain device pair type without relying on std on device.
    PairDevice *result_dev = reinterpret_cast<PairDevice*>(result);

    // Launch kernel
    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result_dev, k);

    // No device memory allocation inside kernel. Synchronization/stream handling is left to the caller if needed.
}