#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ============================================================================
// Optimized k-NN CUDA implementation for 2D points (squared Euclidean distance)
// Each query is processed by a single warp (32 threads).
// Top-k neighbors per query are maintained cooperatively by the warp.
// The code is specialized for k in {32, 64, 128, 256, 512, 1024} via templates.
// ============================================================================

namespace {

/* Common constants */
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;                        // 8 warps -> 256 threads per block
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_SIZE = 4096;                           // Number of data points cached in shared memory per block

/* Full warp mask for sync/shuffle/ballot */
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

/**
 * Warp-wide reduction: compute maximum of 'val' across the warp.
 * Returns the maximum in lane 0; caller may broadcast if needed.
 */
__device__ __forceinline__
float warp_reduce_max(float val) {
    // Tree reduction with shfl_down
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * Warp-wide argmax: find lane index of maximum value.
 * Inputs:
 *   val      - local value in each lane
 * Outputs (by reference, same in all lanes after return):
 *   max_val  - maximum value
 *   max_lane - lane index (0..31) of one occurrence of maximum
 */
__device__ __forceinline__
void warp_argmax(float val, float &max_val, int &max_lane) {
    max_val  = val;
    max_lane = threadIdx.x & (WARP_SIZE - 1);

    // Parallel reduction to find maximum value and its lane index
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val  = __shfl_down_sync(FULL_MASK, max_val,  offset);
        int   other_lane = __shfl_down_sync(FULL_MASK, max_lane, offset);
        if (other_val > max_val) {
            max_val  = other_val;
            max_lane = other_lane;
        }
    }

    // Broadcast final result from lane 0 to all lanes
    max_val  = __shfl_sync(FULL_MASK, max_val,  0);
    max_lane = __shfl_sync(FULL_MASK, max_lane, 0);
}

/**
 * Warp-wide argmin with associated index.
 * Each lane provides (val, idx). The minimum (val, idx, lane) is selected.
 * Results are returned in all lanes via references.
 */
__device__ __forceinline__
void warp_argmin_with_index(float val, int idx, float &min_val, int &min_lane, int &min_idx) {
    int lane = threadIdx.x & (WARP_SIZE - 1);

    min_val  = val;
    min_lane = lane;
    min_idx  = idx;

    // Parallel reduction to find minimum value and its lane/index
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val  = __shfl_down_sync(FULL_MASK, min_val,  offset);
        int   other_lane = __shfl_down_sync(FULL_MASK, min_lane, offset);
        int   other_idx  = __shfl_down_sync(FULL_MASK, min_idx,  offset);
        if (other_val < min_val) {
            min_val  = other_val;
            min_lane = other_lane;
            min_idx  = other_idx;
        }
    }

    // Broadcast from lane 0
    min_val  = __shfl_sync(FULL_MASK, min_val,  0);
    min_lane = __shfl_sync(FULL_MASK, min_lane, 0);
    min_idx  = __shfl_sync(FULL_MASK, min_idx,  0);
}

/**
 * Insert a candidate (dist, idx) into a per-lane top-K list kept sorted in ascending order.
 * The list has fixed length K_PER_LANE. The worst (largest distance) entry is at index K_PER_LANE-1.
 * Precondition: dist < topk_dist[K_PER_LANE - 1] (so the candidate must enter the list).
 *
 * This is an insertion-by-bubbling algorithm:
 *   - We iterate from best to worst, and whenever the new distance is smaller than the current
 *     entry, we swap it in and bubble the displaced entry downwards.
 *   - After K_PER_LANE iterations, the worst entry is evicted.
 */
template <int K_PER_LANE>
__device__ __forceinline__
void insert_candidate_into_lane(float dist, int idx,
                                float (&topk_dist)[K_PER_LANE],
                                int   (&topk_idx)[K_PER_LANE]) {
    float cur_dist = dist;
    int   cur_idx  = idx;

    #pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        if (cur_dist < topk_dist[i]) {
            // Swap current candidate with entry at position i
            float tmp_dist = topk_dist[i];
            int   tmp_idx  = topk_idx[i];

            topk_dist[i] = cur_dist;
            topk_idx[i]  = cur_idx;

            cur_dist = tmp_dist;
            cur_idx  = tmp_idx;
        }
    }
    // After this, the largest value among the (K_PER_LANE + 1) values
    // has been bumped off the end of the array.
}

/**
 * Kernel template for a fixed K (number of neighbors).
 * K is one of {32, 64, 128, 256, 512, 1024}.
 *
 * Each query is processed by a single warp. All warps in the block cooperatively
 * load batches of data points into shared memory. Each warp maintains its own
 * distributed top-K structure across lanes.
 */
template <int K>
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result)
{
    static_assert(K % WARP_SIZE == 0, "K must be a multiple of warp size");
    constexpr int K_PER_LANE = K / WARP_SIZE;  // Each lane holds this many entries.

    // Shared memory tile caching data points for this block.
    __shared__ float2 shared_data[TILE_SIZE];

    const int lane_id  = threadIdx.x & (WARP_SIZE - 1);   // 0..31
    const int warp_id  = threadIdx.x >> 5;                // 0..WARPS_PER_BLOCK-1
    const int warp_per_block = blockDim.x / WARP_SIZE;

    const int global_warp_id = blockIdx.x * warp_per_block + warp_id;
    const bool valid_query   = (global_warp_id < query_count);

    // Per-lane top-K arrays (ascending order: best at index 0, worst at index K_PER_LANE-1)
    float topk_dist[K_PER_LANE];
    int   topk_idx [K_PER_LANE];

    // Initialize top-K with "infinite" distances.
    #pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        topk_dist[i] = FLT_MAX;
        topk_idx[i]  = -1;
    }

    // Load query point into registers (lane 0) and broadcast within warp.
    float qx = 0.0f, qy = 0.0f;
    if (valid_query && lane_id == 0) {
        float2 q = query[global_warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Loop over data in tiles cached into shared memory.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_SIZE) {
        const int tile_size = (tile_base + TILE_SIZE <= data_count)
                            ? TILE_SIZE
                            : (data_count - tile_base);

        // Cooperative loading of a tile of data points into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_data[i] = data[tile_base + i];
        }

        // Synchronize to ensure the tile is fully loaded.
        __syncthreads();

        // Process the tile in batches of WARP_SIZE points.
        for (int base = 0; base < tile_size; base += WARP_SIZE) {

            // Snapshot of current global worst distance for this warp's top-K.
            // Each lane contributes its local worst (largest) distance.
            float local_worst = topk_dist[K_PER_LANE - 1];
            float global_worst_snapshot = warp_reduce_max(local_worst);
            global_worst_snapshot = __shfl_sync(FULL_MASK, global_worst_snapshot, 0);

            // Compute candidate distance for this lane in the current batch.
            int   data_idx_in_tile = base + lane_id;
            float cand_dist = FLT_MAX;
            int   cand_idx  = -1;

            if (data_idx_in_tile < tile_size) {
                float2 p = shared_data[data_idx_in_tile];
                float dx = p.x - qx;
                float dy = p.y - qy;
                // Squared Euclidean distance; fmaf may map to FMA hardware.
                cand_dist = fmaf(dy, dy, dx * dx);
                cand_idx  = tile_base + data_idx_in_tile;
            }

            // Determine which lanes have potentially interesting candidates
            // (i.e., better than the current snapshot worst).
            int is_better = (cand_idx >= 0) && (cand_dist < global_worst_snapshot);
            unsigned better_mask = __ballot_sync(FULL_MASK, is_better);

            // Process all promising candidates from this batch.
            // We iterate over the set bits in better_mask. For each such lane,
            // we cooperatively insert its candidate into the warp's global top-K.
            unsigned active = better_mask;
            while (active) {
                // Extract index of the least significant set bit (lane with a candidate).
                int src_lane = __ffs(active) - 1;  // 0-based lane index

                // Broadcast that lane's candidate to all lanes.
                float cd = __shfl_sync(FULL_MASK, cand_dist, src_lane);
                int   ci = __shfl_sync(FULL_MASK, cand_idx,  src_lane);

                // Find current global worst among all per-lane worst entries.
                float lane_worst = topk_dist[K_PER_LANE - 1];
                float max_val;
                int   max_lane;
                warp_argmax(lane_worst, max_val, max_lane);

                // If this candidate is better than the current global worst, insert it.
                if (cd < max_val) {
                    if ((threadIdx.x & (WARP_SIZE - 1)) == max_lane) {
                        insert_candidate_into_lane<K_PER_LANE>(cd, ci, topk_dist, topk_idx);
                    }
                }

                // Clear this bit and move to the next candidate.
                active &= active - 1;
            }
        }

        // Synchronize before loading next tile into shared memory.
        __syncthreads();
    }

    // Final stage: for valid queries, merge the per-lane sorted lists into a single
    // globally sorted list of K neighbors using a warp-level multiway merge.
    if (!valid_query) {
        return;
    }

    const int base_out_index = global_warp_id * K;

    // Each lane maintains a pointer into its local sorted list.
    int ptr = 0;

    // Perform K-way merge: each iteration selects the smallest among lane heads.
    for (int out_pos = 0; out_pos < K; ++out_pos) {
        float candidate_val = (ptr < K_PER_LANE) ? topk_dist[ptr] : FLT_MAX;
        int   candidate_idx = (ptr < K_PER_LANE) ? topk_idx[ptr]  : -1;

        float min_val;
        int   min_lane, min_idx;
        warp_argmin_with_index(candidate_val, candidate_idx,
                               min_val, min_lane, min_idx);

        // Lane 0 writes the selected neighbor to global result array.
        if (lane_id == 0) {
            result[base_out_index + out_pos].first  = min_idx;
            result[base_out_index + out_pos].second = min_val;
        }

        // The lane that contributed the minimum advances its pointer.
        if (lane_id == min_lane) {
            ++ptr;
        }
    }
}

} // anonymous namespace

// ============================================================================
// Host interface
// ============================================================================

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

    // Compute launch configuration.
    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    int  num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(num_blocks, 1, 1);

    // Dispatch to the appropriate kernel specialization based on k.
    switch (k) {
        case 32:
            knn_kernel<32><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel<64><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel<128><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel<256><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel<512><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel<1024><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        default:
            // k is guaranteed (by problem statement) to be a power of 2 between 32 and 1024.
            // If this code path is reached, inputs are invalid; do nothing.
            return;
    }

    // Optionally, synchronize or check for errors here if desired by the caller.
    // cudaDeviceSynchronize();
    // cudaGetLastError();
}