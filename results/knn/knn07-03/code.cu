#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) per query and processes the data in shared-memory tiles.
// For each warp/query, we keep two per-warp shared-memory arrays of length k:
//   - intermediate result (sorted ascending): distances and indices
//   - candidate buffer (unsorted, up to k elements) with a shared atomic counter
// We also keep per-warp max_distance (distance of current k-th nearest neighbor).
// The algorithm iterates over the data in tiles cached in shared memory. Each lane computes distances to its
// assigned points, filters them by max_distance, and tries to insert them into the warp's candidate buffer using
// atomicAdd to get an insertion position. When the buffer is full (or at the end of the data), the warp:
//   1) Sorts the candidate buffer using Bitonic Sort (ascending).
//   2) Forms a bitonic-merged result of size k by taking pairwise minima of the candidate buffer (ascending)
//      and the intermediate result reversed (descending), i.e., out[i] = min(cand[i], inter[k-1-i]).
//   3) Sorts that merged result (a bitonic sequence) again using Bitonic Sort to produce the updated intermediate.
// After processing all tiles and performing a final flush, results are written back in row-major order.
//
// Notes on design:
// - k is guaranteed to be a power of two between 32 and 1024, inclusive.
// - We keep per-warp "private" buffers in shared memory (distinct regions per warp).
// - We use warp-synchronous programming: __syncwarp() within warp tasks and __syncthreads() for block-wide tasks.
// - We use a fixed number of warps per block (WARPS_PER_BLOCK = 4) and a fixed tile size (TILE_POINTS = 4096),
//   which fits comfortably in the shared memory budgets of A100/H100 even for k=1024.
// - The bitonic sort is parallelized across 32 threads of a warp by mapping compare-and-swap work over indices.
//
// Result writing:
// - The output array is std::pair<int, float>, allocated with cudaMalloc. On device we reinterpret it as a simple
//   POD struct with identical layout, to safely write (first=index, second=distance).

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of warps per block. 4 warps per block -> 128 threads per block.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

// Number of data points loaded per tile into shared memory. 4096 points = 32KB of shared memory for float2.
#ifndef TILE_POINTS
#define TILE_POINTS 4096
#endif

// POD struct to mirror std::pair<int,float> layout on device.
struct PairIF {
    int   first;
    float second;
};

// Utility to compute the minimum of two (distance, index) pairs by distance.
__device__ __forceinline__ void min_pair(float da, int ia, float db, int ib, float &do_, int &io_) {
    if (da < db) {
        do_ = da; io_ = ia;
    } else {
        do_ = db; io_ = ib;
    }
}

// Bitonic sort (ascending) on shared arrays of length n (power of two), executed by one warp.
// All 32 threads in the warp cooperate. Uses shared memory arrays dist[] and idx[].
__device__ __forceinline__ void bitonic_sort_warp(float *dist, int *idx, int n) {
    // Warp-wide sort with compare-and-swap pattern; the "n" is power-of-two.
    // Two nested loops follow the standard bitonic sort network:
    // - size: current size of subsequences to be merged (2,4,8,...,n)
    // - stride: distance between compared elements (size/2, size/4, ..., 1)
    // Threads distribute the i indices (0..n-1) in steps of warpSize.
    const unsigned mask = 0xFFFFFFFFu;
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each lane processes indices i in [lane, n) with step WARP_SIZE
            int lane = threadIdx.x & (WARP_SIZE - 1);
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ stride;
                if (l > i) {
                    bool up = ((i & size) == 0); // ascending in this subsequence
                    float di = dist[i];
                    float dl = dist[l];
                    int   ii = idx[i];
                    int   il = idx[l];
                    bool cond = (up ? (di > dl) : (di < dl));
                    if (cond) {
                        // swap
                        dist[i] = dl; dist[l] = di;
                        idx[i]  = il; idx[l]  = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Kernel implementing k-NN for 2D points as described above.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           PairIF * __restrict__ result,
                           int k)
{
    // Block configuration and warp/lane indices.
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int lane_id          = threadIdx.x & (WARP_SIZE - 1);
    const int warps_per_block  = blockDim.x / WARP_SIZE;
    const int warp_global_id   = blockIdx.x * warps_per_block + warp_id_in_block;
    const bool active_warp     = (warp_global_id < query_count);

    // Shared memory layout:
    // [0]             float2 tile_points[TILE_POINTS]
    // [tile_bytes]    float inter_d[warps_per_block][k]
    // [...          ] int   inter_i[warps_per_block][k]
    // [...          ] float cand_d [warps_per_block][k]
    // [...          ] int   cand_i [warps_per_block][k]
    // [...          ] int   cand_count[warps_per_block]
    // [...          ] float max_d    [warps_per_block]
    extern __shared__ unsigned char smem[];
    size_t offset = 0;

    // Tile of data points cached by the entire block.
    float2 *tile_points = reinterpret_cast<float2*>(smem + offset);
    const size_t tile_bytes = sizeof(float2) * TILE_POINTS;
    offset += tile_bytes;

    // Align the offset to 8 bytes for good measure
    offset = (offset + 7) & ~size_t(7);

    // Per-warp arrays: inter distances
    float *s_inter_d = reinterpret_cast<float*>(smem + offset);
    offset += sizeof(float) * size_t(warps_per_block) * size_t(k);

    // Per-warp arrays: inter indices
    int *s_inter_i = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * size_t(warps_per_block) * size_t(k);

    // Per-warp arrays: candidate distances
    float *s_cand_d = reinterpret_cast<float*>(smem + offset);
    offset += sizeof(float) * size_t(warps_per_block) * size_t(k);

    // Per-warp arrays: candidate indices
    int *s_cand_i = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * size_t(warps_per_block) * size_t(k);

    // Per-warp atomic counters: candidate counts
    int *s_cand_count = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * size_t(warps_per_block);

    // Per-warp max distances
    float *s_max_d = reinterpret_cast<float*>(smem + offset);
    offset += sizeof(float) * size_t(warps_per_block);

    // Warp-local base offsets for its shared arrays
    const int warp_base = warp_id_in_block * k;

    // Initialize per-warp state: intermediate result distances to +inf, indices to -1, cand_count=0, max_d=+inf.
    if (active_warp) {
        for (int pos = lane_id; pos < k; pos += WARP_SIZE) {
            s_inter_d[warp_base + pos] = CUDART_INF_F;
            s_inter_i[warp_base + pos] = -1;
        }
    }
    if (lane_id == 0) {
        s_cand_count[warp_id_in_block] = 0;
        s_max_d[warp_id_in_block] = CUDART_INF_F;
    }
    __syncthreads();

    // Helper lambda to flush the candidate buffer into the intermediate result for this warp.
    auto flush_candidates = [&]() {
        __syncwarp();

        // Fill any unused candidate slots with +inf so we can sort exactly k elements.
        if (active_warp) {
            int cnt = s_cand_count[warp_id_in_block];
            // Cap cnt to at most k; entries >= k are ignored.
            if (cnt > k) cnt = k;
            // Initialize distances/indices for slots [cnt, k) to +inf/-1
            for (int pos = lane_id + cnt; pos < k; pos += WARP_SIZE) {
                s_cand_d[warp_base + pos] = CUDART_INF_F;
                s_cand_i[warp_base + pos] = -1;
            }
        }
        __syncwarp();

        if (active_warp) {
            // 1) Sort the candidate buffer (ascending).
            bitonic_sort_warp(&s_cand_d[warp_base], &s_cand_i[warp_base], k);

            // 2) Merge with the current intermediate result by taking pairwise minima of:
            //    - s_cand_d[i] (ascending)
            //    - s_inter_d[k-1-i] (descending)
            //    Store the minima back into the candidate buffer.
            for (int i = lane_id; i < k; i += WARP_SIZE) {
                int j = k - 1 - i;
                float out_d; int out_i;
                min_pair(s_cand_d[warp_base + i], s_cand_i[warp_base + i],
                         s_inter_d[warp_base + j], s_inter_i[warp_base + j],
                         out_d, out_i);
                s_cand_d[warp_base + i] = out_d;
                s_cand_i[warp_base + i] = out_i;
            }
            __syncwarp();

            // 3) Sort the merged result (bitonic sequence) ascending to get updated intermediate.
            bitonic_sort_warp(&s_cand_d[warp_base], &s_cand_i[warp_base], k);

            // Copy merged result into the intermediate arrays and update max_d (k-th neighbor distance).
            for (int pos = lane_id; pos < k; pos += WARP_SIZE) {
                s_inter_d[warp_base + pos] = s_cand_d[warp_base + pos];
                s_inter_i[warp_base + pos] = s_cand_i[warp_base + pos];
            }
            __syncwarp();
            if (lane_id == 0) {
                s_cand_count[warp_id_in_block] = 0;
                s_max_d[warp_id_in_block] = s_inter_d[warp_base + (k - 1)];
            }
        }
        __syncwarp();
    };

    // Main loop: process data points in shared-memory tiles.
    const unsigned full_mask = 0xFFFFFFFFu;
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_count = data_count - tile_start;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Load the tile into shared memory cooperatively by the entire block.
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            tile_points[t] = data[tile_start + t];
        }
        __syncthreads();

        // Broadcast the query point to all lanes in the warp.
        float qx = 0.0f, qy = 0.0f;
        if (active_warp) {
            if (lane_id == 0) {
                float2 q = query[warp_global_id];
                qx = q.x;
                qy = q.y;
            }
            qx = __shfl_sync(full_mask, qx, 0);
            qy = __shfl_sync(full_mask, qy, 0);
        }

        // Process the tile: each lane works on indices t = it*WARP_SIZE + lane_id (warp-synchronous loop).
        int iters = (tile_count + WARP_SIZE - 1) / WARP_SIZE;
        for (int it = 0; it < iters; ++it) {
            bool request_flush = false;
            if (active_warp) {
                int t = it * WARP_SIZE + lane_id;
                if (t < tile_count) {
                    float2 p = tile_points[t];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    float dist = dx * dx + dy * dy;

                    float maxd = s_max_d[warp_id_in_block];
                    if (dist < maxd) {
                        int pos = atomicAdd(&s_cand_count[warp_id_in_block], 1);
                        if (pos < k) {
                            s_cand_d[warp_base + pos] = dist;
                            s_cand_i[warp_base + pos] = tile_start + t;
                            // If we just filled the k-th slot (pos == k-1), request a flush.
                            if (pos + 1 == k) request_flush = true;
                        } else {
                            // Buffer overflow: request flush; this candidate is dropped.
                            request_flush = true;
                        }
                    }
                }
            }
            // If any lane in the warp requested a flush, perform it now.
            unsigned mask = __ballot_sync(full_mask, request_flush);
            if (mask) {
                flush_candidates();
            }
        }
        __syncthreads();
    }

    // Final flush after all tiles: merge any remaining candidates.
    if (active_warp) {
        // All lanes must see the same condition/value for synchronization.
        int cnt = s_cand_count[warp_id_in_block];
        unsigned mask = __ballot_sync(full_mask, cnt > 0);
        if (mask && (lane_id == 0 || true)) {
            // All lanes enter the flush function to keep warp-synchronous semantics.
            flush_candidates();
        }
    }
    __syncthreads();

    // Write the final top-k results for this query in ascending order.
    if (active_warp) {
        // Store to global memory in row-major: result[qid * k + j] = {index, distance}
        int out_base = warp_global_id * k;
        for (int pos = lane_id; pos < k; pos += WARP_SIZE) {
            PairIF out;
            out.first  = s_inter_i[warp_base + pos];
            out.second = s_inter_d[warp_base + pos];
            result[out_base + pos] = out;
        }
    }
}

// Host API: run_knn
// This function launches the kernel with an appropriate configuration for the target GPU.
// - query: pointer to query_count float2 points
// - data: pointer to data_count float2 points
// - result: pointer to result array of length query_count * k (std::pair<int,float>)
// - k: number of neighbors (power of two, 32..1024)
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Kernel launch configuration.
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Compute dynamic shared memory size:
    // - TILE_POINTS * sizeof(float2) for the data tile
    // - per-warp: 2 arrays (inter) of k*(float + int) + 2 arrays (cand) of k*(float + int)
    // - per-warp: 1 int (cand_count) + 1 float (max_d)
    size_t tile_bytes = sizeof(float2) * size_t(TILE_POINTS);
    size_t per_warp_bytes = (size_t)k * (sizeof(float) + sizeof(int)) * 2  // inter
                          + (size_t)k * (sizeof(float) + sizeof(int)) * 2  // cand
                          + sizeof(int) + sizeof(float);                   // counters
    size_t shared_bytes = tile_bytes + per_warp_bytes * warps_per_block;

    // Launch kernel. Reinterpret result pointer as POD PairIF.
    PairIF *result_if = reinterpret_cast<PairIF*>(result);
    knn_kernel<<<blocks, threads_per_block, shared_bytes>>>(query, query_count, data, data_count, result_if, k);
}