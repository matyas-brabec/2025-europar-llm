#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <limits>

// This implementation assigns exactly one warp (32 threads) to each query point.
// Each warp maintains a per-query global top-k (sorted ascending by distance)
// entirely in shared memory. The input data points are processed in large tiles
// that are cached in shared memory. For each tile, the warp iterates through the
// points in chunks of 32: every thread computes the distance between the query
// and one data point, the warp cooperatively sorts these 32 distances via a
// warp-level bitonic sort, and then merges the sorted 32-candidate list into
// the warp's top-k list using a parallel "merge path" algorithm. The merge
// produces the updated top-k in shared memory. This approach guarantees exact
// k-NN results and uses no device-side dynamic allocations beyond shared memory.
//
// Key characteristics and optimizations:
// - Warp-per-query to fully utilize warp-level collectives (shuffle, sync).
// - Data processed in shared-memory tiles to reduce global memory traffic.
// - Warp-level bitonic sort turns 32 per-lane candidates into a sorted sequence.
// - Parallel merge-path merges the 32-sorted candidates into a per-warp
//   top-k array in O(k) work per 32 candidates, split across threads.
// - The per-warp top-k arrays are kept sorted (ascending distances), ensuring
//   the final writeback is in the required order.
//
// Assumptions:
// - k is a power of two in [32, 1024]; query_count and data_count are valid.
// - data_count >= k.
// - Pointers query, data, result refer to device memory allocated with cudaMalloc.
// - Target is a modern datacenter GPU (e.g., A100/H100), with large shared memory.
//
// Tunables chosen for a good trade-off on A100/H100:
// - WARPS_PER_BLOCK = 4 (128 threads per block).
// - DATA_TILE_POINTS = 8192 (64 KB for tile cache).
// These choices keep shared memory usage under ~132 KB per block for the worst
// case k=1024, allowing at least one resident block per SM and good throughput.

#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 4
#endif

#ifndef DATA_TILE_POINTS
#define DATA_TILE_POINTS 8192
#endif

// Full warp mask for sync/shuffle
#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFFu
#endif

// Lane ID within the warp
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

// Warp ID within the block
__device__ __forceinline__ int warp_id() {
    return threadIdx.x >> 5;
}

// Warp-level bitonic sort of 32 (distance, index) pairs in ascending order by distance.
// After the call, lane i holds the i-th smallest pair among the 32 inputs.
__device__ __forceinline__ void warp_bitonic_sort_32_asc(float &key, int &val) {
    const unsigned mask = FULL_MASK;
    int lane = lane_id();
    // Standard XOR-based bitonic sort network for 32 elements (warp size).
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            float other_key = __shfl_xor_sync(mask, key, j);
            int   other_val = __shfl_xor_sync(mask, val, j);
            bool up = ((lane & k) == 0);
            // Swap to enforce bitonic sorting in 'up' direction.
            if ((key > other_key) == up) {
                float tmpk = key; key = other_key; other_key = tmpk;
                int tmpv = val;  val = other_val;  other_val = tmpv;
            }
        }
    }
}

// Merge-path partition search: find how many elements to take from A for a given 'diag'
// (number of elements to output in total from A and B).
// A and B must be sorted ascending by distance.
// Returns ai such that ai in [max(0, diag - Blen), min(diag, Alen)] and
// the merge prefix (of length 'diag') consists of ai elements from A and (diag - ai) from B.
__device__ __forceinline__ int merge_path_search(const float* __restrict__ A, int Alen,
                                                 const float* __restrict__ B, int Blen,
                                                 int diag) {
    int a_low = max(0, diag - Blen);
    int a_high = min(diag, Alen);
    // Binary search to find the split point
    while (a_low < a_high) {
        int a_mid = (a_low + a_high) >> 1;
        int b_mid = diag - a_mid;

        float a_mid_val = (a_mid < Alen) ? A[a_mid] : CUDART_INF_F;
        float b_prev_val = (b_mid > 0) ? B[b_mid - 1] : -CUDART_INF_F;

        if (a_mid_val < b_prev_val) {
            a_low = a_mid + 1;
        } else {
            a_high = a_mid;
        }
    }
    return a_low;
}

// Merge B (length M) into top-k array A (length K), both sorted ascending,
// producing the smallest K elements of the union into C (length K).
// Work is split among the 32 threads in the warp using merge-path partitioning.
__device__ __forceinline__ void warp_merge_topk(const float* __restrict__ A_dist,
                                                const int*   __restrict__ A_idx,
                                                const float* __restrict__ B_dist,
                                                const int*   __restrict__ B_idx,
                                                int K, int M,
                                                float* __restrict__ C_dist,
                                                int*   __restrict__ C_idx) {
    int lane = lane_id();

    // We only need the first K elements of the merge of A (K) and B (M)
    const int TOT = K;
    const int seg = (TOT + 31) >> 5; // ceil(TOT / 32)

    int d0 = min(TOT, lane * seg);
    int d1 = min(TOT, d0 + seg);

    int a0 = merge_path_search(A_dist, K, B_dist, M, d0);
    int b0 = d0 - a0;

    int a1 = merge_path_search(A_dist, K, B_dist, M, d1);
    int b1 = d1 - a1;

    int ai = a0;
    int bi = b0;
    int out = d0;

    while (out < d1) {
        float av = (ai < K) ? A_dist[ai] : CUDART_INF_F;
        float bv = (bi < M) ? B_dist[bi] : CUDART_INF_F;
        bool takeA = (av <= bv);
        if (takeA) {
            C_dist[out] = av;
            C_idx[out]  = A_idx[ai];
            ++ai;
        } else {
            C_dist[out] = bv;
            C_idx[out]  = B_idx[bi];
            ++bi;
        }
        ++out;
    }
}

// Main kernel: one warp per query.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k) {
    // Shared memory layout:
    // [0]  float2 s_data[DATA_TILE_POINTS];
    // [1]  float  s_topDist[WARPS_PER_BLOCK * k];
    // [2]  int    s_topIdx [WARPS_PER_BLOCK * k];
    // [3]  float  s_outDist[WARPS_PER_BLOCK * k];
    // [4]  int    s_outIdx [WARPS_PER_BLOCK * k];
    // [5]  float  s_candDist[WARPS_PER_BLOCK * 32];
    // [6]  int    s_candIdx [WARPS_PER_BLOCK * 32];

    extern __shared__ unsigned char smem[];
    unsigned char* sm_ptr = smem;

    float2* s_data = reinterpret_cast<float2*>(sm_ptr);
    sm_ptr += DATA_TILE_POINTS * sizeof(float2);

    float* s_topDist_all = reinterpret_cast<float*>(sm_ptr);
    sm_ptr += (size_t)KNN_WARPS_PER_BLOCK * (size_t)k * sizeof(float);

    int*   s_topIdx_all  = reinterpret_cast<int*>(sm_ptr);
    sm_ptr += (size_t)KNN_WARPS_PER_BLOCK * (size_t)k * sizeof(int);

    float* s_outDist_all = reinterpret_cast<float*>(sm_ptr);
    sm_ptr += (size_t)KNN_WARPS_PER_BLOCK * (size_t)k * sizeof(float);

    int*   s_outIdx_all  = reinterpret_cast<int*>(sm_ptr);
    sm_ptr += (size_t)KNN_WARPS_PER_BLOCK * (size_t)k * sizeof(int);

    float* s_candDist_all = reinterpret_cast<float*>(sm_ptr);
    sm_ptr += (size_t)KNN_WARPS_PER_BLOCK * 32 * sizeof(float);

    int*   s_candIdx_all  = reinterpret_cast<int*>(sm_ptr);
    // sm_ptr final not used further

    const int warp = warp_id();
    const int lane = lane_id();
    const int warps_per_block = KNN_WARPS_PER_BLOCK;

    const int query_idx = blockIdx.x * warps_per_block + warp;

    // Pointers to this warp's private regions in shared memory
    float* s_topDist  = s_topDist_all  + (size_t)warp * (size_t)k;
    int*   s_topIdx   = s_topIdx_all   + (size_t)warp * (size_t)k;
    float* s_outDist  = s_outDist_all  + (size_t)warp * (size_t)k;
    int*   s_outIdx   = s_outIdx_all   + (size_t)warp * (size_t)k;
    float* s_candDist = s_candDist_all + (size_t)warp * 32;
    int*   s_candIdx  = s_candIdx_all  + (size_t)warp * 32;

    // Load query point into registers and broadcast within the warp
    float qx = 0.0f, qy = 0.0f;
    if (lane == 0 && query_idx < query_count) {
        float2 q = query[query_idx];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Initialize per-warp top-k buffer (ascending, initialized to +inf, idx -1)
    if (query_idx < query_count) {
        for (int i = lane; i < k; i += 32) {
            s_topDist[i] = CUDART_INF_F;
            s_topIdx[i]  = -1;
        }
    }
    __syncwarp();

    // Process data in shared-memory tiles
    for (int data_base = 0; data_base < data_count; data_base += DATA_TILE_POINTS) {
        int tile_count = data_count - data_base;
        if (tile_count > DATA_TILE_POINTS) tile_count = DATA_TILE_POINTS;

        // Cooperative load of the tile into shared memory by the whole block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_data[i] = data[data_base + i];
        }
        __syncthreads();

        if (query_idx < query_count) {
            // Iterate through the tile in chunks of 32 points
            const int steps = (tile_count + 31) >> 5;
            for (int step = 0; step < steps; ++step) {
                // Each lane picks one point from the tile (or INF if out-of-range)
                int idx_in_tile = (step << 5) + lane;
                float cand_dist;
                int cand_idx = -1;
                if (idx_in_tile < tile_count) {
                    float2 p = s_data[idx_in_tile];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // Squared Euclidean distance (no sqrt)
                    cand_dist = fmaf(dx, dx, dy * dy);
                    cand_idx = data_base + idx_in_tile;
                } else {
                    cand_dist = CUDART_INF_F;
                    cand_idx = -1;
                }

                // Sort the 32 candidates in ascending order across the warp
                warp_bitonic_sort_32_asc(cand_dist, cand_idx);

                // Store sorted candidates to shared memory (per-warp region)
                s_candDist[lane] = cand_dist;
                s_candIdx [lane] = cand_idx;
                __syncwarp();

                // Number of valid candidates in this chunk (last chunk may be partial)
                int m = tile_count - (step << 5);
                if (m > 32) m = 32;
                if (m < 0)  m = 0;

                // Merge the 32 sorted candidates into the warp's top-k
                warp_merge_topk(s_topDist, s_topIdx, s_candDist, s_candIdx, k, m, s_outDist, s_outIdx);
                __syncwarp();

                // Copy merged result back to the per-warp top-k buffer
                for (int i = lane; i < k; i += 32) {
                    s_topDist[i] = s_outDist[i];
                    s_topIdx[i]  = s_outIdx[i];
                }
                __syncwarp();
            }
        }

        __syncthreads(); // ensure tile is no longer in use before overwriting
    }

    // Write final top-k results for this query back to global memory in ascending order
    if (query_idx < query_count) {
        const int base = query_idx * k;
        for (int i = lane; i < k; i += 32) {
            // std::pair<int,float>: write members directly
            result[base + i].first  = s_topIdx[i];
            result[base + i].second = s_topDist[i];
        }
    }
}

// Host-side entry point
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Grid/block configuration
    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    const int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size calculation.
    // Components:
    // - DATA_TILE_POINTS * sizeof(float2)
    // - For each warp:
    //     * topDist[k] + topIdx[k] + outDist[k] + outIdx[k]
    //     * candDist[32] + candIdx[32]
    size_t smem_bytes = 0;
    smem_bytes += (size_t)DATA_TILE_POINTS * sizeof(float2);
    smem_bytes += (size_t)warps_per_block * ( (size_t)k * sizeof(float)   // topDist
                                            + (size_t)k * sizeof(int)     // topIdx
                                            + (size_t)k * sizeof(float)   // outDist
                                            + (size_t)k * sizeof(int)     // outIdx
                                            + 32 * sizeof(float)          // candDist
                                            + 32 * sizeof(int) );         // candIdx

    // Ensure kernel is allowed to use the necessary dynamic shared memory on capable GPUs
    int device = 0;
    cudaGetDevice(&device);
    int maxDynSmem = 0;
    /// @FIXED
    /// cudaDeviceGetAttribute(&maxDynSmem, cudaDevAttrMaxDynamicSharedMemorySize, device);
    cudaDeviceGetAttribute(&maxDynSmem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if ((int)smem_bytes <= maxDynSmem) {
        cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    // Launch kernel
    dim3 block(threads_per_block, 1, 1);
    dim3 grid(grid_blocks, 1, 1);
    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}
