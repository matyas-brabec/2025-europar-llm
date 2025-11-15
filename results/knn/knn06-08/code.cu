#include <cuda_runtime.h>
#include <utility>  // for std::pair in the run_knn signature

// This implementation computes k-NN (squared Euclidean distances in 2D) on the GPU.
// - One warp (32 threads) processes one query point.
// - Data points are processed in shared-memory tiles.
// - Each warp keeps a "private" intermediate top-k in registers (distributed across lanes).
// - Each warp also has a candidate buffer of size k in shared memory, plus an atomic counter.
// - New candidates whose distance is less than the current max_distance are staged into
//   the shared candidate buffer using atomicAdd. When the buffer is full, the warp merges
//   the candidates into the intermediate top-k using warp-level cooperation.
// - After processing all tiles, any remaining candidates are merged, the final top-k is
//   sorted in ascending order of distance with a warp-level bitonic sort, and written out.
//
// Assumptions:
// - k is a power of two in [32, 1024].
// - data_count >= k.
// - query_count and data_count are large enough to warrant GPU processing.
// - Memory for query, data, and result has already been cudaMalloc'ed by the caller.
// - result is an array of std::pair<int, float> stored in row-major layout
//   (result[query_idx * k + j]).

// ============================================================================
// Global configuration constants
// ============================================================================

static constexpr int WARP_SIZE          = 32;
static constexpr int MAX_K              = 1024;
static constexpr int MAX_K_PER_LANE     = MAX_K / WARP_SIZE; // 32
static constexpr int WARPS_PER_BLOCK    = 4;                 // 4 warps -> 128 threads
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int DATA_TILE_SIZE     = 1024;              // Number of data points per shared-memory tile

// Result pair type used inside the kernel. This is intended to mirror the layout of
// std::pair<int, float> (two 4-byte fields, total 8 bytes, no extra padding).
struct ResultPair
{
    int   first;
    float second;
};

// ============================================================================
// Warp-level helper functions
// ============================================================================

/**
 * Find the maximum distance in the per-warp top-k buffers and its position.
 *
 * The top-k buffers are distributed across threads in a warp:
 *   - Each lane stores 'segments' elements in its local dist array.
 *   - segments = k / WARP_SIZE (since k is a multiple of 32).
 * This function:
 *   - Scans local segments in each lane to find the local maximum and slot.
 *   - Performs a warp-wide reduction via shuffles to find the global maximum
 *     value, the lane that owns it, and the per-lane slot index.
 *
 * @param dist_local  Per-thread local distance buffer (size MAX_K_PER_LANE).
 * @param segments    Number of valid entries per thread (k / WARP_SIZE).
 * @param lane_id     Lane index within warp [0..31].
 * @param max_lane    (out) Lane index in which the global maximum resides.
 * @param max_slot    (out) Slot index within that lane's local buffer.
 * @return            The maximum distance value.
 */
__device__ __forceinline__
float warp_find_max_in_topk(const float dist_local[MAX_K_PER_LANE],
                            int segments,
                            int lane_id,
                            int &max_lane,
                            int &max_slot)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Find local maximum among this lane's segments.
    float local_max  = -CUDART_INF_F;
    int   local_slot = -1;

    #pragma unroll
    for (int s = 0; s < MAX_K_PER_LANE; ++s)
    {
        if (s < segments)
        {
            float val = dist_local[s];
            if (val > local_max)
            {
                local_max  = val;
                local_slot = s;
            }
        }
    }

    // Initialize reduction values.
    float max_val   = local_max;
    int   max_lane_ = lane_id;
    int   max_slot_ = local_slot;

    // Warp-wide reduction to find the global maximum.
    // We propagate both the value and position (lane, slot).
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        float other_val   = __shfl_down_sync(FULL_MASK, max_val,   offset);
        int   other_lane  = __shfl_down_sync(FULL_MASK, max_lane_, offset);
        int   other_slot  = __shfl_down_sync(FULL_MASK, max_slot_, offset);

        // Ignore lanes whose local_slot might be -1 (no valid entries);
        // for our usage, this normally shouldn't happen if k >= 32.
        if (other_slot >= 0 && other_val > max_val)
        {
            max_val   = other_val;
            max_lane_ = other_lane;
            max_slot_ = other_slot;
        }
    }

    // Broadcast final maximum and its position to all lanes.
    max_val   = __shfl_sync(FULL_MASK, max_val,   0);
    max_lane_ = __shfl_sync(FULL_MASK, max_lane_, 0);
    max_slot_ = __shfl_sync(FULL_MASK, max_slot_, 0);

    max_lane = max_lane_;
    max_slot = max_slot_;
    return max_val;
}

/**
 * Insert a single candidate (cand_idx, cand_dist) into the intermediate
 * per-warp top-k structure if it improves the current top-k set.
 *
 * The top-k set is represented as:
 *   - topk_idx_local[s], topk_dist_local[s] in each lane (0 <= s < segments),
 *   - distributed across lanes so that logical position p (0..k-1) resides at
 *       lane = p % WARP_SIZE,
 *       slot = p / WARP_SIZE.
 *
 * Algorithm:
 *   - Find the current maximum distance (and its position) in the top-k set.
 *   - If cand_dist >= current maximum, ignore candidate.
 *   - Otherwise, replace the current maximum with the candidate.
 *   - Recompute the maximum distance in the new top-k and update max_distance.
 *
 * All threads in the warp participate; communication is via warp shuffles and
 * __syncwarp barriers.
 */
__device__ __forceinline__
void warp_insert_into_topk(int   cand_idx,
                           float cand_dist,
                           int   k,
                           int   segments,
                           int   lane_id,
                           int   (&topk_idx_local)[MAX_K_PER_LANE],
                           float (&topk_dist_local)[MAX_K_PER_LANE],
                           float &max_distance)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Find current maximum distance and its position.
    int   max_lane = 0;
    int   max_slot = 0;
    float current_max = warp_find_max_in_topk(topk_dist_local, segments, lane_id,
                                              max_lane, max_slot);
    max_distance = current_max;  // keep max_distance consistent across lanes

    // If candidate is not better than the current worst, ignore it.
    if (cand_dist >= current_max)
    {
        return;
    }

    // Replace the worst (maximum) entry with the new candidate.
    if (lane_id == max_lane)
    {
        topk_dist_local[max_slot] = cand_dist;
        topk_idx_local[max_slot]  = cand_idx;
    }

    __syncwarp(FULL_MASK);

    // Recompute maximum distance after replacement.
    current_max = warp_find_max_in_topk(topk_dist_local, segments, lane_id,
                                        max_lane, max_slot);
    max_distance = current_max;
}

/**
 * Merge all staged candidates from the per-warp shared candidate buffer into
 * the intermediate top-k structure.
 *
 * - The candidate buffer for warp_local is stored in contiguous regions of
 *   sh_candidate_idx and sh_candidate_dist of length MAX_K.
 * - The actual number of staged candidates is sh_candidate_count[warp_local]
 *   (<= k, but we allocate MAX_K for simplicity).
 * - Each candidate is processed sequentially, but warp-wide cooperation is
 *   used for inserting into the top-k (warp_insert_into_topk).
 *
 * After merging, candidate_count is reset to 0.
 */
__device__ __forceinline__
void warp_merge_candidates(int   warp_local,
                           int   lane_id,
                           int  *sh_candidate_count,
                           int  *sh_candidate_idx,
                           float *sh_candidate_dist,
                           int   k,
                           int   segments,
                           int   (&topk_idx_local)[MAX_K_PER_LANE],
                           float (&topk_dist_local)[MAX_K_PER_LANE],
                           float &max_distance)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Read the current number of candidates for this warp.
    int count = 0;
    if (lane_id == 0)
    {
        count = sh_candidate_count[warp_local];
    }
    count = __shfl_sync(FULL_MASK, count, 0);

    if (count <= 0)
    {
        return;
    }

    const int warp_offset = warp_local * MAX_K;

    // Process each candidate sequentially, using warp-wide insertion.
    for (int ci = 0; ci < count; ++ci)
    {
        float cand_dist = 0.0f;
        int   cand_idx  = -1;

        if (lane_id == 0)
        {
            cand_dist = sh_candidate_dist[warp_offset + ci];
            cand_idx  = sh_candidate_idx [warp_offset + ci];
        }

        cand_dist = __shfl_sync(FULL_MASK, cand_dist, 0);
        cand_idx  = __shfl_sync(FULL_MASK, cand_idx,  0);

        warp_insert_into_topk(cand_idx, cand_dist, k, segments, lane_id,
                              topk_idx_local, topk_dist_local, max_distance);

        __syncwarp(FULL_MASK);
    }

    // Reset candidate count for this warp.
    if (lane_id == 0)
    {
        sh_candidate_count[warp_local] = 0;
    }
    __syncwarp(FULL_MASK);
}

/**
 * Add a single candidate (cand_idx, cand_dist) to the per-warp shared candidate
 * buffer, with overflow handling and merging when the buffer is full.
 *
 * Candidate buffer layout:
 *   - sh_candidate_idx[warp_local * MAX_K ... warp_local * MAX_K + k - 1]
 *   - sh_candidate_dist[...] same indexing
 *   - sh_candidate_count[warp_local] is the atomic count of currently stored
 *     candidates (0 <= count <= k).
 *
 * Steps:
 *   - While true:
 *       * Reserve a slot with atomicAdd on sh_candidate_count (performed by lane 0).
 *       * If the reserved slot index < k, store the candidate and exit.
 *       * Otherwise, rollback (atomicSub) and perform a merge of all current
 *         candidates with the intermediate top-k (warp_merge_candidates),
 *         then re-check whether this candidate is still valid (dist < max_distance).
 *         If it no longer qualifies, exit; otherwise, retry.
 *
 * This ensures no candidate that might influence the final top-k is lost.
 * The use of atomicAdd satisfies the requirement to use atomics for candidate
 * buffer insertion.
 */
__device__ __forceinline__
void warp_add_candidate(int   warp_local,
                        int   lane_id,
                        int   cand_idx,
                        float cand_dist,
                        int  *sh_candidate_count,
                        int  *sh_candidate_idx,
                        float *sh_candidate_dist,
                        int   k,
                        int   segments,
                        int   (&topk_idx_local)[MAX_K_PER_LANE],
                        float (&topk_dist_local)[MAX_K_PER_LANE],
                        float &max_distance)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int warp_offset = warp_local * MAX_K;

    while (true)
    {
        int pos = 0;

        // Lane 0 reserves a slot using atomicAdd.
        if (lane_id == 0)
        {
            pos = atomicAdd(&sh_candidate_count[warp_local], 1);
        }
        pos = __shfl_sync(FULL_MASK, pos, 0);

        if (pos < k)
        {
            // There was space in the candidate buffer.
            if (lane_id == 0)
            {
                sh_candidate_idx [warp_offset + pos] = cand_idx;
                sh_candidate_dist[warp_offset + pos] = cand_dist;
            }
            __syncwarp(FULL_MASK);
            break;
        }
        else
        {
            // Buffer was full or over-reserved. Roll back the increment and merge.
            if (lane_id == 0)
            {
                atomicSub(&sh_candidate_count[warp_local], 1);
            }
            __syncwarp(FULL_MASK);

            // Merge existing candidates into top-k; this will reset candidate_count to 0.
            warp_merge_candidates(warp_local, lane_id,
                                  sh_candidate_count,
                                  sh_candidate_idx,
                                  sh_candidate_dist,
                                  k, segments,
                                  topk_idx_local,
                                  topk_dist_local,
                                  max_distance);

            // After merging, the current max_distance may have decreased.
            // Re-check whether this candidate is still better than max_distance.
            if (cand_dist >= max_distance)
            {
                break;  // Candidate no longer qualifies.
            }
            // Otherwise, retry insertion (loop continues).
        }
    }
}

// ============================================================================
// Warp-level bitonic sort for final top-k ordering
// ============================================================================

/**
 * Perform an in-place bitonic sort (ascending order by distance) on k elements
 * in shared memory for a single warp.
 *
 * - k is a power of two (32 <= k <= 1024).
 * - Distances and indices are stored in:
 *     sh_dist[warp_offset + 0 .. warp_offset + k-1]
 *     sh_idx [warp_offset + 0 .. warp_offset + k-1]
 * - 32 threads cooperate to sort k elements. Each thread handles multiple
 *   indices: j = lane_id, lane_id + WARP_SIZE, lane_id + 2*WARP_SIZE, ...
 *
 * This uses the standard bitonic sorting network:
 *   for (size = 2; size <= k; size <<= 1)
 *     for (stride = size >> 1; stride > 0; stride >>= 1)
 *       for all i in [0, k):
 *         j = i ^ stride
 *         if (j > i) compare-and-swap(i, j) with direction determined by 'size'
 */
__device__ __forceinline__
void warp_bitonic_sort_topk(int   warp_local,
                            int   lane_id,
                            int   k,
                            int  *sh_idx,
                            float *sh_dist)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int warp_offset = warp_local * MAX_K;

    // Bitonic sort network
    for (int size = 2; size <= k; size <<= 1)
    {
        // 'size' is the size of the current bitonic sequence.
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            // Each lane processes multiple indices i in [0, k).
            for (int i = lane_id; i < k; i += WARP_SIZE)
            {
                int ixj = i ^ stride;
                if (ixj > i)
                {
                    bool up = ((i & size) == 0);  // true: ascending in this region

                    float di = sh_dist[warp_offset + i];
                    float dj = sh_dist[warp_offset + ixj];
                    int   ii = sh_idx [warp_offset + i];
                    int   ij = sh_idx [warp_offset + ixj];

                    bool cmp = (di > dj);
                    if ((cmp && up) || (!cmp && !up))
                    {
                        // Swap
                        sh_dist[warp_offset + i]   = dj;
                        sh_dist[warp_offset + ixj] = di;
                        sh_idx [warp_offset + i]   = ij;
                        sh_idx [warp_offset + ixj] = ii;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// ============================================================================
// Main CUDA kernel
// ============================================================================

/**
 * Kernel to compute k-nearest neighbors for 2D points.
 *
 * Each warp processes a single query point:
 *   - Loads the query point into registers (broadcast from lane 0).
 *   - Iterates over data points in tiles cached in shared memory.
 *   - For each data point, computes squared Euclidean distance, compares with
 *     current max_distance, and if better, stages as a candidate in shared memory.
 *   - When the candidate buffer is full, merges candidates into the intermediate
 *     top-k buffer.
 *   - After all tiles are processed, merges any remaining candidates, sorts the
 *     final top-k by distance, and writes them to global memory.
 */
__global__ void knn_kernel(const float2 *__restrict__ query,
                           int                     query_count,
                           const float2 *__restrict__ data,
                           int                     data_count,
                           int                     k,
                           ResultPair *__restrict__ results)
{
    // Shared memory:
    // - sh_data:        shared tile of data points
    // - sh_candidate_*: per-warp candidate buffers and counts
    __shared__ float2 sh_data[DATA_TILE_SIZE];

    __shared__ int   sh_candidate_count[WARPS_PER_BLOCK];
    __shared__ int   sh_candidate_idx[WARPS_PER_BLOCK * MAX_K];
    __shared__ float sh_candidate_dist[WARPS_PER_BLOCK * MAX_K];

    const int tid             = threadIdx.x;
    const int warp_in_block   = tid / WARP_SIZE;           // [0 .. WARPS_PER_BLOCK-1]
    const int lane_id         = tid & (WARP_SIZE - 1);     // [0 .. 31]
    const int global_warp_id  = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    const bool warp_active    = (global_warp_id < query_count);

    // Only active warps need to know segments; inactive warps can leave it zero.
    const int segments = warp_active ? (k / WARP_SIZE) : 0;

    // Initialize per-warp candidate count once.
    if (warp_in_block < WARPS_PER_BLOCK && lane_id == 0)
    {
        sh_candidate_count[warp_in_block] = 0;
    }

    // Per-thread local top-k buffers (distributed representation).
    int   topk_idx_local[MAX_K_PER_LANE];
    float topk_dist_local[MAX_K_PER_LANE];

    // Initialize intermediate top-k with "infinite" distances.
    #pragma unroll
    for (int s = 0; s < MAX_K_PER_LANE; ++s)
    {
        if (warp_active && s < segments)
        {
            topk_idx_local[s]  = -1;
            topk_dist_local[s] = CUDART_INF_F;
        }
    }

    // Per-warp current maximum distance among its k nearest neighbors.
    // This is kept consistent across all lanes via warp cooperation.
    float max_distance = CUDART_INF_F;

    // Load query point into registers for this warp (only if active).
    float qx = 0.0f, qy = 0.0f;
    if (warp_active && lane_id == 0)
    {
        float2 q = query[global_warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    __syncwarp(0xFFFFFFFFu);

    // Iterate over data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE_SIZE)
    {
        const int tile_size = min(DATA_TILE_SIZE, data_count - tile_start);

        // Load tile into shared memory (all threads in block cooperate).
        for (int i = tid; i < tile_size; i += blockDim.x)
        {
            sh_data[i] = data[tile_start + i];
        }

        __syncthreads();

        // Each warp processes the tile for its query.
        if (warp_active)
        {
            for (int i = lane_id; i < tile_size; i += WARP_SIZE)
            {
                float2 dp = sh_data[i];
                float dx = qx - dp.x;
                float dy = qy - dp.y;

                // Squared Euclidean distance (dx*dx + dy*dy).
                float dist = fmaf(dx, dx, dy * dy);

                // Check if this point is potentially among the k nearest neighbors.
                bool is_candidate = (dist < max_distance);

                // Warp-wide ballot of candidate lanes.
                unsigned int mask = __ballot_sync(0xFFFFFFFFu, is_candidate);

                // Process each candidate lane sequentially within the warp, but
                // using warp-wide cooperation for insertion and merging.
                while (mask)
                {
                    int candidate_lane = __ffs(mask) - 1;  // lane index [0..31] with a pending candidate

                    // Broadcast candidate distance and index (global index of data point).
                    float cand_dist = __shfl_sync(0xFFFFFFFFu, dist, candidate_lane);
                    int   local_i   = __shfl_sync(0xFFFFFFFFu, i,    candidate_lane);
                    int   cand_idx  = tile_start + local_i;

                    // Insert candidate into shared candidate buffer with overflow
                    // handling and merging as needed.
                    warp_add_candidate(warp_in_block,
                                       lane_id,
                                       cand_idx,
                                       cand_dist,
                                       sh_candidate_count,
                                       sh_candidate_idx,
                                       sh_candidate_dist,
                                       k,
                                       segments,
                                       topk_idx_local,
                                       topk_dist_local,
                                       max_distance);

                    // Clear this bit and move to the next candidate lane.
                    mask &= (mask - 1);
                    __syncwarp(0xFFFFFFFFu);
                }
            }
        }

        __syncthreads();
    }

    // After all tiles, merge remaining candidates for active warps.
    if (warp_active)
    {
        int remaining_count = 0;
        if (lane_id == 0)
        {
            remaining_count = sh_candidate_count[warp_in_block];
        }
        remaining_count = __shfl_sync(0xFFFFFFFFu, remaining_count, 0);

        if (remaining_count > 0)
        {
            warp_merge_candidates(warp_in_block,
                                  lane_id,
                                  sh_candidate_count,
                                  sh_candidate_idx,
                                  sh_candidate_dist,
                                  k,
                                  segments,
                                  topk_idx_local,
                                  topk_dist_local,
                                  max_distance);
        }
    }

    // Final step: sort the top-k neighbors by distance (ascending) and write
    // them out to global memory.
    if (warp_active)
    {
        const int warp_offset = warp_in_block * MAX_K;

        // Copy intermediate top-k from registers to shared memory arrays to
        // perform bitonic sort.
        #pragma unroll
        for (int s = 0; s < MAX_K_PER_LANE; ++s)
        {
            if (s < segments)
            {
                int global_pos = s * WARP_SIZE + lane_id;  // position in [0..k-1]
                sh_candidate_idx [warp_offset + global_pos] = topk_idx_local[s];
                sh_candidate_dist[warp_offset + global_pos] = topk_dist_local[s];
            }
        }

        __syncwarp(0xFFFFFFFFu);

        // Sort k elements in ascending order of distance.
        warp_bitonic_sort_topk(warp_in_block,
                               lane_id,
                               k,
                               sh_candidate_idx,
                               sh_candidate_dist);

        __syncwarp(0xFFFFFFFFu);

        // Write sorted results to global memory.
        ResultPair *query_result = results + static_cast<size_t>(global_warp_id) * k;

        for (int pos = lane_id; pos < k; pos += WARP_SIZE)
        {
            ResultPair out;
            out.first  = sh_candidate_idx [warp_offset + pos];
            out.second = sh_candidate_dist[warp_offset + pos];
            query_result[pos] = out;
        }
    }
}

// ============================================================================
// Host interface
// ============================================================================

/**
 * Host function to launch the k-NN kernel.
 *
 * @param query        Device pointer to query points (float2[query_count]).
 * @param query_count  Number of query points.
 * @param data         Device pointer to data points (float2[data_count]).
 * @param data_count   Number of data points.
 * @param result       Device pointer to result array (std::pair<int,float>[query_count * k]).
 * @param k            Number of nearest neighbors to find (power of two, 32..1024).
 */
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
    {
        return;
    }

    // Basic validation per problem statement (not strictly necessary, but helps avoid misuse).
    if (k < 32 || k > MAX_K || (k & (k - 1)) != 0)
    {
        // k must be a power of two in [32, 1024].
        return;
    }

    // Number of warps needed equals number of queries; each warp handles one query.
    const int num_warps  = query_count;
    const int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Reinterpret std::pair<int,float>* as our internal ResultPair*.
    // Both are expected to be two 4-byte fields laid out contiguously.
    ResultPair *result_internal = reinterpret_cast<ResultPair*>(result);

    // Launch kernel.
    dim3 grid(num_blocks);
    dim3 block(THREADS_PER_BLOCK);

    knn_kernel<<<grid, block>>>(query,
                                query_count,
                                data,
                                data_count,
                                k,
                                result_internal);

    // Optional synchronization; in a larger application, error handling and
    // stream-based execution might be preferred instead.
    cudaDeviceSynchronize();
}