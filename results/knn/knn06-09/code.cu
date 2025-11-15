#include <cuda_runtime.h>
#include <utility>

// Simple POD type matching the layout of std::pair<int, float>.
// Used on the device side and reinterpreted from/to std::pair<int,float>* on the host side.
struct KNNResult
{
    int   index;
    float dist;
};

constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 8;    // 8 warps -> 256 threads per block
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_POINTS      = 2048; // number of data points cached in shared memory per batch

// Warp-level merge of existing k-NN results with a candidate buffer.
// All threads in the warp must call this function with the same arguments (except lane_id).
// The function uses a parallel selection algorithm to find the k smallest distances
// from the union of the current k-NN list (size knn_size) and the candidate list (size cand_count).
// Results are written back into knn[0..knn_size-1] (sorted by distance ascending).
__device__ __forceinline__
void warp_merge_knn(
    KNNResult* knn,          // [in/out] current best neighbors (size >= k)
    int&       knn_size,     // [in/out] number of valid entries currently in knn (<= k)
    KNNResult* candidates,   // [in]     candidate buffer (size >= cand_count)
    int        cand_count,   // [in]     number of valid candidates
    int        k,            // [in]     desired number of neighbors
    float&     max_distance, // [in/out] distance of the k-th neighbor (or INF if knn_size < k)
    int        lane_id,      // [in]     lane id within warp
    unsigned   mask          // [in]     active mask for this warp
)
{
    const float INF = CUDART_INF_F;

    int n_total = knn_size + cand_count;
    if (n_total <= 0)
    {
        return;
    }

    // We will produce up to k best neighbors.
    int new_size = (n_total < k) ? n_total : k;

    // Parallel selection for new_size smallest elements.
    for (int t = 0; t < new_size; ++t)
    {
        float best_dist    = INF;
        int   best_segment = 0;   // 0 = candidates[], 1 = knn[]
        int   best_idx     = -1;

        // Each lane scans a strided subset of the combined array.
        // Combined indices: 0..knn_size-1 map to knn[], knn_size..knn_size+cand_count-1 map to candidates[].
        for (int idx = lane_id; idx < n_total; idx += WARP_SIZE)
        {
            float d;
            int   seg;
            int   local_idx;

            if (idx < knn_size)
            {
                d         = knn[idx].dist;
                seg       = 1;
                local_idx = idx;
            }
            else
            {
                int ci = idx - knn_size;
                if (ci >= cand_count)
                {
                    continue;
                }
                d         = candidates[ci].dist;
                seg       = 0;
                local_idx = ci;
            }

            // Entries that have already been selected in previous iterations are
            // marked with distance = INF and are automatically ignored here.
            if (d < best_dist)
            {
                best_dist    = d;
                best_segment = seg;
                best_idx     = local_idx;
            }
        }

        // Warp-wide reduction to find global minimum distance.
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            float other_dist    = __shfl_down_sync(mask, best_dist,    offset);
            int   other_idx     = __shfl_down_sync(mask, best_idx,     offset);
            int   other_segment = __shfl_down_sync(mask, best_segment, offset);

            if (other_dist < best_dist)
            {
                best_dist    = other_dist;
                best_idx     = other_idx;
                best_segment = other_segment;
            }
        }

        int   global_best_idx     = __shfl_sync(mask, best_idx,     0);
        int   global_best_segment = __shfl_sync(mask, best_segment, 0);
        float global_best_dist    = __shfl_sync(mask, best_dist,    0);

        if (lane_id == 0)
        {
            KNNResult selected;
            if (global_best_segment == 1)
            {
                // Selected from existing knn[]
                selected           = knn[global_best_idx];
                knn[global_best_idx].dist = INF; // mark as used
            }
            else
            {
                // Selected from candidates[]
                selected                = candidates[global_best_idx];
                candidates[global_best_idx].dist = INF; // mark as used
            }

            // Store the selected element in the output position t.
            knn[t] = selected;

            // If this is the last selected element, update max_distance:
            //   - if we already have k neighbors, it's the largest distance among them
            //   - otherwise, max_distance remains INF to allow further candidates
            if (t == new_size - 1)
            {
                if (new_size == k)
                    max_distance = selected.dist;
                else
                    max_distance = INF;
            }
        }

        __syncwarp(mask);
    }

    // Update knn_size and broadcast updated values to all lanes
    if (lane_id == 0)
    {
        knn_size = new_size;
    }
    knn_size     = __shfl_sync(mask, knn_size,     0);
    max_distance = __shfl_sync(mask, max_distance, 0);
}

// Kernel: each warp processes one query point and finds its k nearest neighbors among `data`.
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int                        query_count,
    const float2* __restrict__ data,
    int                        data_count,
    KNNResult* __restrict__    results,
    int                        k)
{
    // Shared memory layout (dynamic):
    // [0                                      ... TILE_POINTS-1]         : float2 cached data points
    // [TILE_POINTS                             ... + WARPS_PER_BLOCK*k]  : KNNResult intermediate knn results
    // [TILE_POINTS + WARPS_PER_BLOCK*k         ... + 2*WARPS_PER_BLOCK*k]: KNNResult candidate buffers

    extern __shared__ unsigned char shared_mem[];

    float2*    shared_points   = reinterpret_cast<float2*>(shared_mem);
    size_t     offset_knn      = TILE_POINTS * sizeof(float2);
    KNNResult* shared_knn_all  = reinterpret_cast<KNNResult*>(shared_mem + offset_knn);
    KNNResult* shared_cand_all = shared_knn_all + WARPS_PER_BLOCK * k;

    // Per-warp candidate counters in shared memory.
    __shared__ int s_candidate_count[WARPS_PER_BLOCK];

    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int thread_id         = threadIdx.x;
    int lane_id           = thread_id & (WARP_SIZE - 1);    // 0..31
    int warp_id_in_block  = thread_id >> 5;                 // 0..WARPS_PER_BLOCK-1

    int global_warp_id    = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int query_idx         = global_warp_id;

    if (query_idx >= query_count)
    {
        return; // entire warp exits, since query_idx is uniform within the warp
    }

    // Pointers to this warp's private sections in shared memory.
    KNNResult* warp_knn  = shared_knn_all  + warp_id_in_block * k;
    KNNResult* warp_cand = shared_cand_all + warp_id_in_block * k;

    // Initialize candidate count for this warp.
    if (lane_id == 0)
    {
        s_candidate_count[warp_id_in_block] = 0;
    }
    __syncwarp(FULL_MASK);

    // Load the query point for this warp (lane 0) and broadcast to all lanes.
    float2 q;
    if (lane_id == 0)
    {
        q = query[query_idx];
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Intermediate k-NN state.
    float max_distance = CUDART_INF_F; // distance of k-th neighbor; INF while knn_size < k
    int   knn_size     = 0;           // number of valid entries in warp_knn

    // Process data in tiles cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS)
    {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_POINTS)
            tile_size = TILE_POINTS;

        // Load this tile of data points into shared memory cooperatively by the whole block.
        for (int idx = thread_id; idx < tile_size; idx += blockDim.x)
        {
            shared_points[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // Each warp processes all points in the tile, one per lane per iteration.
        int num_iters = (tile_size + WARP_SIZE - 1) / WARP_SIZE;
        for (int it = 0; it < num_iters; ++it)
        {
            int i        = it * WARP_SIZE + lane_id; // index within tile
            bool in_range = (i < tile_size);

            float dist     = 0.0f;
            int   data_idx = -1;

            if (in_range)
            {
                float2 p = shared_points[i];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                dist     = dx * dx + dy * dy;
                data_idx = tile_start + i;
            }

            // Determine whether this point is a candidate (closer than current max_distance).
            bool pending = in_range && (dist < max_distance);

            // Insert candidates into the per-warp shared candidate buffer, flushing (merging)
            // whenever the buffer would become full.
            while (__any_sync(FULL_MASK, pending))
            {
                unsigned pending_mask = __ballot_sync(FULL_MASK, pending);
                int      n_pending    = __popc(pending_mask);
                if (n_pending == 0)
                    break;

                int cur_cand_count = 0;
                if (lane_id == 0)
                {
                    cur_cand_count = s_candidate_count[warp_id_in_block];
                }
                cur_cand_count = __shfl_sync(FULL_MASK, cur_cand_count, 0);

                // If adding all pending candidates would overflow the buffer,
                // first merge the existing buffer with the current k-NN result.
                if (cur_cand_count + n_pending > k)
                {
                    if (cur_cand_count > 0)
                    {
                        warp_merge_knn(
                            warp_knn,
                            knn_size,
                            warp_cand,
                            cur_cand_count,
                            k,
                            max_distance,
                            lane_id,
                            FULL_MASK);

                        if (lane_id == 0)
                        {
                            s_candidate_count[warp_id_in_block] = 0;
                        }
                        __syncwarp(FULL_MASK);
                    }

                    // Re-evaluate pending status with updated max_distance.
                    pending = in_range && (dist < max_distance);
                }
                else
                {
                    // There is enough space in the candidate buffer for all pending candidates.
                    int base = 0;
                    if (lane_id == 0)
                    {
                        // Atomic add to get a block of slots for this batch of candidates.
                        base = atomicAdd(&s_candidate_count[warp_id_in_block], n_pending);
                    }
                    base = __shfl_sync(FULL_MASK, base, 0);

                    if (pending)
                    {
                        // Compute per-lane offset within this batch using a prefix sum of the pending mask.
                        unsigned mask_lt = pending_mask & ((1u << lane_id) - 1u);
                        int      offset  = __popc(mask_lt);
                        int      pos     = base + offset;

                        // Insert candidate into shared buffer.
                        warp_cand[pos].index = data_idx;
                        warp_cand[pos].dist  = dist;

                        pending = false;
                    }

                    // All pending candidates have been inserted in this iteration.
                    break;
                }
            }
        }

        __syncthreads();
    }

    // After processing all tiles, merge any remaining candidates in the buffer.
    int final_cand_count = 0;
    if (lane_id == 0)
    {
        final_cand_count = s_candidate_count[warp_id_in_block];
    }
    final_cand_count = __shfl_sync(FULL_MASK, final_cand_count, 0);

    if (final_cand_count > 0)
    {
        warp_merge_knn(
            warp_knn,
            knn_size,
            warp_cand,
            final_cand_count,
            k,
            max_distance,
            lane_id,
            FULL_MASK);

        if (lane_id == 0)
        {
            s_candidate_count[warp_id_in_block] = 0;
        }
        __syncwarp(FULL_MASK);
    }

    // Ensure we have exactly k outputs; data_count is guaranteed >= k by problem statement.
    if (knn_size < k)
    {
        if (lane_id == 0)
        {
            // Fill remaining slots with dummy values (should not be needed in practice).
            for (int i = knn_size; i < k; ++i)
            {
                warp_knn[i].index = -1;
                warp_knn[i].dist  = CUDART_INF_F;
            }
            knn_size = k;
        }
        knn_size = __shfl_sync(FULL_MASK, knn_size, 0);
    }

    // Write the k nearest neighbors for this query to global memory.
    int out_base = query_idx * k;
    for (int j = lane_id; j < k; j += WARP_SIZE)
    {
        results[out_base + j] = warp_knn[j];
    }
}

// Host-side interface as specified in the problem statement.
// The input and output pointers are assumed to be device pointers allocated with cudaMalloc.
void run_knn(
    const float2*               query,
    int                         query_count,
    const float2*               data,
    int                         data_count,
    std::pair<int, float>*      result,
    int                         k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
    {
        return;
    }

    // Launch configuration: one warp per query.
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Dynamic shared memory size: tile of data points + per-warp knn and candidate buffers.
    size_t shared_bytes =
        static_cast<size_t>(TILE_POINTS) * sizeof(float2) +
        static_cast<size_t>(2) * WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(KNNResult);

    // Opt-in to larger dynamic shared memory if necessary (for modern GPUs like A100/H100).
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    // Reinterpret the result pointer to the POD type used on device.
    KNNResult* d_results = reinterpret_cast<KNNResult*>(result);

    // Launch the kernel.
    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        d_results,
        k);

    // Synchronization/error handling is left to the caller if desired.
}