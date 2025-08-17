#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <utility>

// This implementation assigns one warp (32 threads) per query.
// The block cooperatively loads batches ("tiles") of data points into shared memory.
// Each warp computes squared L2 distances from its query to the points in the tile.
// For each group of 32 candidates (one per lane), the warp performs a batched insertion
// into its intermediate top-k structure, replacing the current global worsts as needed.
// Each query's intermediate top-k is held privately by its warp, distributed across threads:
// each lane stores k/32 candidates in registers. This avoids extra global or shared memory.
// After processing all data, each warp merges its distributed per-lane lists to produce the
// final k nearest neighbors, written in ascending distance order.
//
// Notes on parameters and assumptions:
// - k is a power of two between 32 and 1024 (inclusive). Thus k % 32 == 0.
// - data_count >= k.
// - Inputs and outputs are allocated in device memory by cudaMalloc.
// - result is std::pair<int, float>*, but the kernel writes via a POD with the same layout.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct PairIF {
    int   first;
    float second;
};

static inline __device__ unsigned lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

// Pair for warp-level reductions carrying value and lane id.
struct PairValLane {
    float val;
    int   lane;
};

// Warp-wide min reduction on (val, lane). Tie-breaks by smaller lane id.
static inline __device__ PairValLane warp_min_pair(PairValLane a, unsigned mask) {
    // Tree reduction: offsets 16,8,4,2,1 for 32-lane warp
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float bval = __shfl_down_sync(mask, a.val, offset);
        int   blane = __shfl_down_sync(mask, a.lane, offset);
        // Choose minimum value; on tie, smaller lane id.
        if (bval < a.val || (bval == a.val && blane < a.lane)) {
            a.val  = bval;
            a.lane = blane;
        }
    }
    return a;
}

// Warp-wide max reduction on (val, lane). Tie-breaks by smaller lane id (arbitrary but deterministic).
static inline __device__ PairValLane warp_max_pair(PairValLane a, unsigned mask) {
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float bval = __shfl_down_sync(mask, a.val, offset);
        int   blane = __shfl_down_sync(mask, a.lane, offset);
        if (bval > a.val || (bval == a.val && blane < a.lane)) {
            a.val  = bval;
            a.lane = blane;
        }
    }
    return a;
}

// Recompute the local worst (maximum distance) and its position for a thread's local list.
static inline __device__ void recompute_local_worst(const float* __restrict__ dists, int S, float &worst, int &pos) {
    float w = dists[0];
    int p = 0;
    #pragma unroll
    for (int i = 1; i < 32; ++i) {
        if (i >= S) break;
        float v = dists[i];
        if (v > w) {
            w = v;
            p = i;
        }
    }
    worst = w;
    pos = p;
}

// Batched insertion of up to 32 candidates (one per lane) into the warp's distributed top-k.
// Each iteration:
//  - Find the best (minimum distance) pending candidate across 32 lanes.
//  - Find the current global worst (maximum distance) across the warp's local lists.
//  - If the best candidate improves over the global worst, replace the owner lane's worst slot.
//  - Mark the inserted candidate as consumed and repeat, up to 32 times or until no improvement.
static inline __device__ void batch_insert_32(float candDist, int candIdx,
                                              float* __restrict__ localDists,
                                              int*   __restrict__ localIdx,
                                              int S,
                                              float &localWorst, int &localWorstPos,
                                              unsigned mask) {
    unsigned lid = lane_id();
    float myCandDist = candDist;
    int   myCandIdx  = candIdx;

    #pragma unroll
    for (int it = 0; it < WARP_SIZE; ++it) {
        PairValLane bestCand { myCandDist, (int)lid };
        bestCand = warp_min_pair(bestCand, mask);

        PairValLane worstNow { localWorst, (int)lid };
        worstNow = warp_max_pair(worstNow, mask);

        float minCand = bestCand.val;
        float maxWorst = worstNow.val;

        // If even the best pending candidate cannot improve over the current global worst,
        // then none of the current 32 candidates can.
        if (!(minCand < maxWorst)) {
            break;
        }

        // Broadcast selected candidate (distance and index) and target position
        float insDist = __shfl_sync(mask, myCandDist, bestCand.lane);
        int   insIdx  = __shfl_sync(mask, myCandIdx,  bestCand.lane);
        int   tgtPos  = __shfl_sync(mask, localWorstPos, worstNow.lane);

        // The owner of the current global worst replaces its worst slot.
        if ((int)lid == worstNow.lane) {
            localDists[tgtPos] = insDist;
            localIdx[tgtPos]   = insIdx;
            // Recompute local worst for next iterations.
            recompute_local_worst(localDists, S, localWorst, localWorstPos);
        }
        // Mark candidate as consumed at its producing lane.
        if ((int)lid == bestCand.lane) {
            myCandDist = CUDART_INF_F;
            myCandIdx  = -1;
        }
    }
}

__global__ void knn_kernel_warp_per_query(const float2* __restrict__ query,
                                          int query_count,
                                          const float2* __restrict__ data,
                                          int data_count,
                                          PairIF* __restrict__ result,
                                          int k,
                                          int tile_size) {
    extern __shared__ float2 shPoints[]; // tile of data points: tile_size float2's

    // Thread and warp identifiers
    const int lid  = lane_id();
    const int warp_in_block = threadIdx.x >> 5;  // / 32
    const int warps_per_block = blockDim.x >> 5;
    const int q = blockIdx.x * warps_per_block + warp_in_block; // one query per warp
    const bool valid_warp = (q < query_count);
    const unsigned full_mask = 0xFFFFFFFFu;

    // Load the query point once per warp and broadcast
    float2 qpt = make_float2(0.f, 0.f);
    if (valid_warp && lid == 0) {
        qpt = query[q];
    }
    qpt.x = __shfl_sync(full_mask, qpt.x, 0);
    qpt.y = __shfl_sync(full_mask, qpt.y, 0);

    // Each lane will hold S = k/32 candidates (indices and distances) in registers.
    // k is a power of two in [32, 1024], so S in [1, 32].
    const int S = k >> 5;
    // Reserve the maximum needed (32) and use only the first S entries.
    float best_dists[32];
    int   best_idx[32];

    // Initialize per-lane list with +inf distances.
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i < S) {
            best_dists[i] = CUDART_INF_F;
            best_idx[i]   = -1;
        }
    }
    // Track the local worst (max distance) and its position.
    float localWorst;
    int   localWorstPos;
    recompute_local_worst(best_dists, S, localWorst, localWorstPos);

    // Iterate over data in tiles cooperatively loaded into shared memory.
    for (int base = 0; base < data_count; base += tile_size) {
        const int tile_count = min(tile_size, data_count - base);

        // Cooperative load into shared memory
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            shPoints[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes the tile against its query.
        if (valid_warp) {
            // Process the tile in groups of 32 candidates: one per lane.
            for (int t = 0; t < tile_count; t += WARP_SIZE) {
                const int it = t + lid;
                float candDist = CUDART_INF_F;
                int   candIdx  = -1;

                if (it < tile_count) {
                    float2 p = shPoints[it];
                    float dx = qpt.x - p.x;
                    float dy = qpt.y - p.y;
                    // Squared Euclidean distance
                    float d = dx * dx + dy * dy;
                    candDist = d;
                    candIdx  = base + it;
                }

                // Batched insertion of up to 32 candidates.
                batch_insert_32(candDist, candIdx,
                                best_dists, best_idx, S,
                                localWorst, localWorstPos,
                                full_mask);
            }
        }

        __syncthreads(); // Ensure tile is no longer used before loading the next one
    }

    // Finalize: for each warp (query), sort and write the k nearest neighbors in ascending order.
    if (valid_warp) {
        // Sort each lane's local list ascending (S <= 32) via insertion sort.
        #pragma unroll
        for (int i = 1; i < 32; ++i) {
            if (i >= S) break;
            float keyd = best_dists[i];
            int   keyi = best_idx[i];
            int j = i - 1;
            while (j >= 0 && best_dists[j] > keyd) {
                best_dists[j + 1] = best_dists[j];
                best_idx[j + 1]   = best_idx[j];
                --j;
            }
            best_dists[j + 1] = keyd;
            best_idx[j + 1]   = keyi;
        }

        // Multiway merge across 32 sorted sublists (one per lane) to produce k sorted results.
        int ptr = 0; // pointer into this lane's local sorted list
        float headDist = (ptr < S) ? best_dists[ptr] : CUDART_INF_F;
        int   headIdx  = (ptr < S) ? best_idx[ptr]   : -1;

        for (int out = 0; out < k; ++out) {
            // Find the minimum head across lanes
            PairValLane cand { headDist, (int)lid };
            PairValLane mn = warp_min_pair(cand, full_mask);

            // Broadcast the selected head (distance and index) from the winning lane
            float selDist = __shfl_sync(full_mask, headDist, mn.lane);
            int   selIdx  = __shfl_sync(full_mask, headIdx,  mn.lane);

            // Write output for this query at position 'out'. Use a single lane to perform the store.
            if (lid == 0) {
                const int out_offset = q * k + out;
                result[out_offset].first  = selIdx;
                result[out_offset].second = selDist;
            }

            // Advance the pointer in the winning lane
            if ((int)lid == mn.lane) {
                ++ptr;
                headDist = (ptr < S) ? best_dists[ptr] : CUDART_INF_F;
                headIdx  = (ptr < S) ? best_idx[ptr]   : -1;
            }
        }
    }
}

// Host-side function: determines launch configuration and shared memory tile size,
// then launches the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Configuration: 8 warps per block (256 threads) is a good balance for A100/H100.
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Decide shared memory tile size (number of float2 points) based on device capability.
    // Target a large tile for good cache reuse; adjust to hardware limits.
    int device = 0;
    cudaGetDevice(&device);

    int max_smem_default = 0;
    int max_smem_optin   = 0;
    cudaDeviceGetAttribute(&max_smem_default, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&max_smem_optin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Target tile: 8192 points -> 64KB shared memory (8192 * 8 bytes).
    int tile_target = 8192;
    // Cap by data_count and at least 32 (one warp-wide batch).
    tile_target = (data_count < tile_target) ? data_count : tile_target;
    if (tile_target < WARP_SIZE) tile_target = WARP_SIZE;

    // Maximum dynamic shared memory we can request for this kernel.
    int max_smem_available = (max_smem_optin > 0) ? max_smem_optin : max_smem_default;
    size_t bytes_wanted = static_cast<size_t>(tile_target) * sizeof(float2);
    size_t bytes_cap    = static_cast<size_t>(max_smem_available);

    // If we want more than default, opt-in to larger dynamic shared memory.
    if ((int)bytes_wanted > max_smem_default && max_smem_optin > 0) {
        // Request the exact amount we plan to use; driver will clamp to maximum if needed.
        int bytes_request = (int)min(bytes_wanted, bytes_cap);
        cudaFuncSetAttribute(knn_kernel_warp_per_query,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             bytes_request);
    }

    // Final tile size limited by what we can get.
    size_t bytes_final_cap = (size_t)((max_smem_optin > 0) ? max_smem_optin : max_smem_default);
    size_t bytes_final = bytes_wanted <= bytes_final_cap ? bytes_wanted : bytes_final_cap;
    int tile_size = static_cast<int>(bytes_final / sizeof(float2));
    // Ensure tile_size is a positive multiple of 32 for coalesced warp processing.
    tile_size = (tile_size / WARP_SIZE) * WARP_SIZE;
    if (tile_size < WARP_SIZE) tile_size = WARP_SIZE;
    if (tile_size > data_count) tile_size = (data_count / WARP_SIZE) * WARP_SIZE + ((data_count % WARP_SIZE) ? WARP_SIZE : 0);
    if (tile_size > data_count) tile_size = data_count; // clamp if overshot on rounding

    // Grid dimensions: one warp per query.
    int warp_count = (query_count + warps_per_block - 1) / warps_per_block;
    dim3 grid(warp_count);
    dim3 block(threads_per_block);

    // Reinterpret result as POD to write from device easily.
    PairIF* result_if = reinterpret_cast<PairIF*>(result);

    // Launch kernel with dynamic shared memory equal to tile_size * sizeof(float2).
    size_t smem_bytes = static_cast<size_t>(tile_size) * sizeof(float2);
    knn_kernel_warp_per_query<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result_if, k, tile_size);
}