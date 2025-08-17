#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) to each query. The block cooperatively loads
// tiles of the data points into shared memory, and each warp computes distances from its assigned
// query to the cached points. Each warp keeps an intermediate top-k (k is a power of two and
// multiple of 32; each lane owns k/32 slots). Updates to the intermediate result from 32 candidates
// are performed cooperatively using warp shuffles. At the end, each warp writes its k results into
// shared memory and performs an in-warp bitonic sort (ascending by distance) to produce the final
// k-NN list per query, which is then written to the output array.
//
// Design highlights:
// - One warp per query for predictable control flow and warp-level communication.
// - Block-wide shared memory tiling of the data for cache reuse, with a tile size chosen to balance
//   shared memory usage between the tile and the final per-warp sorting scratch buffer.
// - Incremental top-k maintenance using a distributed replacement of the current global maximum
//   among the k kept candidates, processing up to 32 candidates per iteration.
// - Final ordering guaranteed by a single in-warp bitonic sort over the k intermediate results.
// - No additional global memory allocations; only shared memory is used.

#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 8
#endif

#ifndef KNN_TILE_SIZE
#define KNN_TILE_SIZE 4096
#endif

// Pair struct used in shared memory scratch for sorting (distance, index).
struct PairDF {
    float dist;
    int   idx;
};

// Warp-level argmax reduction for (value, lane, pos) triple. Returns the triple broadcast to all lanes.
__device__ __forceinline__ void warp_argmax(float local_val, int local_pos, int lane_id,
                                            float &out_val, int &out_lane, int &out_pos, unsigned mask) {
    float val = local_val;
    int pos   = local_pos;
    int ln    = lane_id;

    // Tree reduction by comparing values, breaking ties by lane then position to get deterministic behavior.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float oth_val = __shfl_down_sync(mask, val, offset);
        int   oth_ln  = __shfl_down_sync(mask, ln,  offset);
        int   oth_pos = __shfl_down_sync(mask, pos, offset);

        bool take_other = (oth_val > val) || (oth_val == val && (oth_ln > ln || (oth_ln == ln && oth_pos > pos)));
        if (take_other) {
            val = oth_val;
            ln  = oth_ln;
            pos = oth_pos;
        }
    }
    // Broadcast the result from lane 0 to all lanes in the warp.
    out_val  = __shfl_sync(mask, val, 0);
    out_lane = __shfl_sync(mask, ln,  0);
    out_pos  = __shfl_sync(mask, pos, 0);
}

__global__ void knn_kernel(const float2 * __restrict__ query, int query_count,
                           const float2 * __restrict__ data,  int data_count,
                           std::pair<int, float> * __restrict__ result, int k)
{
    // Identify warp and lane.
    const int lane_id        = threadIdx.x & 31;
    const int warp_id_in_blk = threadIdx.x >> 5;
    const int warps_per_blk  = blockDim.x >> 5;
    const int warp_global_id = blockIdx.x * warps_per_blk + warp_id_in_blk;

    const bool warp_active = (warp_global_id < query_count);
    const unsigned full_mask = 0xFFFFFFFFu;

    // Shared memory layout:
    // [ 0 .. KNN_TILE_SIZE * sizeof(float2) ) => tile of data points (float2)
    // [ KNN_TILE_SIZE * sizeof(float2) .. )   => per-warp scratch buffers for final sort (k PairDF per warp)
    extern __shared__ unsigned char smem_raw[];
    float2 *tile = reinterpret_cast<float2 *>(smem_raw);
    PairDF *scratch_base = reinterpret_cast<PairDF *>(smem_raw + sizeof(float2) * KNN_TILE_SIZE);
    PairDF *warp_scratch = scratch_base + warp_id_in_blk * k; // warp-private scratch of length k

    // Each warp is responsible for one query index.
    int q_idx = warp_global_id;

    // Broadcast the query point to all lanes of the warp.
    float qx = 0.f, qy = 0.f;
    if (warp_active) {
        float2 q;
        if (lane_id == 0) {
            q = query[q_idx];
        }
        unsigned mask = __activemask();
        qx = __shfl_sync(mask, q.x, 0);
        qy = __shfl_sync(mask, q.y, 0);
    }

    // Per-lane local storage for the warp's distributed top-k (each lane holds s = k/32 items).
    const int s = k >> 5; // since k is a power of two between 32 and 1024, s in [1, 32]
    float best_dist[32]; // max size; only first s used
    int   best_idx[32];
    int   filled_lane = 0;        // number of filled slots in this lane (for prefill phase)
    bool  has_threshold = false;  // whether the initial K candidates have been collected

    // Local cached "max" per lane among its s entries (valid only when has_threshold true).
    float local_max_val = -CUDART_INF_F;
    int   local_max_pos = 0;

    // Current global threshold triple (value, lane, pos) for the worst (maximum distance) among the K kept.
    float thr_val = CUDART_INF_F;
    int   thr_lane = 0, thr_pos = 0;

    // Process data in tiles loaded into shared memory by the whole thread block.
    for (int tile_start = 0; tile_start < data_count; tile_start += KNN_TILE_SIZE) {
        int tile_count = data_count - tile_start;
        if (tile_count > KNN_TILE_SIZE) tile_count = KNN_TILE_SIZE;

        // All threads in the block cooperatively load the tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes the tile against its query.
        if (warp_active) {
            // Iterate over the tile in groups of 32 so that each lane handles one candidate at a time.
            for (int j_base = 0; j_base < tile_count; j_base += 32) {
                int j = j_base + lane_id;
                bool valid = (j < tile_count);

                float2 p = make_float2(0.f, 0.f);
                if (valid) p = tile[j];

                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dy, dy, dx * dx);  // squared L2 distance
                int   didx = tile_start + j;

                // Prefill phase: keep inserting until each lane has filled 's' items.
                if (!has_threshold) {
                    if (valid && (filled_lane < s)) {
                        best_dist[filled_lane] = dist;
                        best_idx[filled_lane]  = didx;
                        ++filled_lane;
                    }
                    // After attempting to fill this round, check if the whole warp has completed prefill.
                    unsigned mask = __activemask();
                    int lane_full = (filled_lane >= s) ? 1 : 0;
                    unsigned full_mask_ballot = __ballot_sync(mask, lane_full);
                    if (full_mask_ballot == full_mask) {
                        // Compute initial local maxima per lane.
                        float mval = best_dist[0];
                        int mpos = 0;
                        #pragma unroll
                        for (int t = 1; t < s; ++t) {
                            float v = best_dist[t];
                            if (v > mval) { mval = v; mpos = t; }
                        }
                        local_max_val = mval;
                        local_max_pos = mpos;
                        // Compute global maximum across the warp to set the initial threshold.
                        warp_argmax(local_max_val, local_max_pos, lane_id, thr_val, thr_lane, thr_pos, mask);
                        has_threshold = true;
                    }
                    // Continue to next 32-candidate batch.
                    continue;
                }

                // After prefill: filter candidates against the current threshold and replace the worst if needed.
                // Build a mask of candidates that are valid and would improve the current top-k.
                unsigned mask = __activemask();
                int accept = (valid && (dist < thr_val)) ? 1 : 0;
                unsigned accept_mask = __ballot_sync(mask, accept);

                // Process the accepted candidates one by one (at most 32 per batch).
                while (accept_mask) {
                    // Lane id of the next candidate to insert (lowest set bit).
                    int src_lane = __ffs(accept_mask) - 1;

                    // Broadcast the candidate from src_lane to all lanes.
                    float cand_dist = __shfl_sync(mask, dist, src_lane);
                    int   cand_idx  = __shfl_sync(mask, didx, src_lane);

                    // Replace the current global worst with this candidate.
                    if (lane_id == thr_lane) {
                        best_dist[thr_pos] = cand_dist;
                        best_idx[thr_pos]  = cand_idx;
                        // Recompute this lane's local maximum.
                        float mval = best_dist[0];
                        int mpos = 0;
                        #pragma unroll
                        for (int t = 1; t < s; ++t) {
                            float v = best_dist[t];
                            if (v > mval) { mval = v; mpos = t; }
                        }
                        local_max_val = mval;
                        local_max_pos = mpos;
                    }

                    // Recompute the global threshold after the replacement.
                    warp_argmax(local_max_val, local_max_pos, lane_id, thr_val, thr_lane, thr_pos, mask);

                    // Clear this candidate from the accept mask and continue.
                    accept_mask &= ~(1u << src_lane);
                }
            } // end for j_base
        } // end if warp_active

        __syncthreads(); // ensure tile is not accessed before being overwritten by the next tile
    } // end for tile_start

    // Final sort and writeback of results.
    if (warp_active) {
        // Write the warp's k candidates into its reserved shared memory buffer.
        for (int t = 0; t < s; ++t) {
            int g = lane_id * s + t;
            warp_scratch[g].dist = best_dist[t];
            warp_scratch[g].idx  = best_idx[t];
        }
        __syncwarp();

        // In-warp bitonic sort over k elements in shared memory (ascending by distance).
        // Each thread processes a strided subset of indices to perform compare-exchange on disjoint pairs.
        for (int size = 2; size <= k; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = lane_id; i < k; i += 32) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool ascending = ((i & size) == 0);
                        PairDF a = warp_scratch[i];
                        PairDF b = warp_scratch[j];
                        bool do_swap = ascending ? (a.dist > b.dist) : (a.dist < b.dist);
                        if (do_swap) {
                            warp_scratch[i] = b;
                            warp_scratch[j] = a;
                        }
                    }
                }
                __syncwarp();
            }
        }

        // Write the sorted neighbors to the output array for this query.
        // Output layout: for query q_idx, result[q_idx * k + j] = (index, distance) of the j-th nearest neighbor.
        for (int j = lane_id; j < k; j += 32) {
            int out_idx = q_idx * k + j;
            PairDF p = warp_scratch[j];
            // std::pair<int,float> is assumed trivially assignable in device code.
            result[out_idx] = std::pair<int, float>(p.idx, p.dist);
        }
    }
}

// Host entry point. Launches the kernel with chosen hyperparameters and dynamic shared memory size.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose execution configuration.
    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    const int total_warps = (query_count + 0) / 1; // one warp per query
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size:
    // - Tile of float2: KNN_TILE_SIZE * sizeof(float2)
    // - Per-warp scratch for final sort: warps_per_block * k * sizeof(PairDF)
    size_t shmem_bytes = sizeof(float2) * KNN_TILE_SIZE + sizeof(PairDF) * (size_t)warps_per_block * (size_t)k;

    // Opt-in to larger dynamic shared memory if needed (on modern GPUs this is supported).
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

    // Launch the kernel.
    dim3 grid(blocks);
    dim3 block(threads_per_block);
    knn_kernel<<<grid, block, shmem_bytes>>>(query, query_count, data, data_count, result, k);
}