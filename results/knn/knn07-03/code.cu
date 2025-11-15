// CUDA k-NN for 2D points using one warp (32 threads) per query.
// Implements the algorithm described in the prompt, including:
//  - Per-query intermediate result (k best neighbors so far)
//  - Per-query candidate buffer in shared memory (size k)
//  - AtomicAdd-based insertion into candidate buffer
//  - Batch processing of data points cached in shared memory
//  - Bitonic sort for buffer and merge stage
//  - Warp-synchronous programming (no extra device allocations)

#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Tuneable parameters for modern data-center GPUs (A100/H100)
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 4;        // 4 warps = 128 threads per block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_SIZE        = 1024;     // data points loaded per block per batch

// Warp-level bitonic sort for an array of length n (power of two, n <= 1024).
// The array is split across the warp: each thread processes indices i = lane, i += WARP_SIZE,
// performing compare-and-swap operations according to the classic serial bitonic sort,
// but parallelized over the warp.
//
// 'dist' and 'idx' hold distances and corresponding indices, respectively, and are sorted
// in ascending order of 'dist'.
// 'mask' is the active lane mask for __syncwarp/__shfl_sync.
// 'lane' is the lane index within the warp (0..31).
__device__ __forceinline__
void warp_bitonic_sort(float *dist, int *idx, int n, unsigned mask, int lane)
{
    // n is guaranteed to be a power of two and >= 2 (k is between 32 and 1024).
    for (int size = 2; size <= n; size <<= 1)
    {
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            // Ensure all threads see results from previous stage before continuing.
            __syncwarp(mask);

            // Parallelize the outermost "i" loop of the serial bitonic sort over the warp.
            for (int i = lane; i < n; i += WARP_SIZE)
            {
                int l = i ^ stride;
                if (l > i)
                {
                    bool ascending = ((i & size) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    int   idi = idx[i];
                    int   idl = idx[l];

                    bool should_swap = (ascending && di > dl) || (!ascending && di < dl);
                    if (should_swap)
                    {
                        dist[i] = dl;
                        dist[l] = di;
                        idx[i]  = idl;
                        idx[l]  = idi;
                    }
                }
            }
        }
    }
    __syncwarp(mask);
}

// Flush/merge the candidate buffer into the intermediate result for a single query
// handled by a single warp.
//
// - best_dist/best_idx: current intermediate result, sorted ascending by distance (size k)
// - cand_dist/cand_idx: candidate buffer, containing 'cand_count' valid elements (size k capacity)
// - k:                   number of neighbors
// - lane:                lane index of the calling thread (0..31)
// - mask:                active lane mask for warp primitives
// - cand_count_ptr:      pointer to the candidate count in shared memory (per warp)
// - max_dist:            reference to per-warp variable storing distance of k-th neighbor
//
// Steps (exactly as specified in the prompt):
//   0) Invariant: best_dist[] is sorted ascending.
//   1) Sort candidate buffer with Bitonic sort (ascending).
//   2) Merge candidate buffer and best array: for each i, best[i] = min(best[i], cand[k-1-i]).
//      This in-place update produces a bitonic sequence in best_dist[] containing the first k
//      elements of the union of candidate buffer and intermediate result.
//   3) Sort the merged best_dist[] using Bitonic sort (ascending), producing the updated
//      intermediate result and updating max_dist = best_dist[k-1].
//
// This function is warp-synchronous and must be called by all threads of the warp together.
__device__ __forceinline__
void warp_flush_candidates(float *best_dist, int *best_idx,
                           float *cand_dist, int *cand_idx,
                           int k, int lane, unsigned mask,
                           int *cand_count_ptr,
                           float &max_dist)
{
    int cand_count = *cand_count_ptr;
    if (cand_count <= 0)
        return;

    const float INF = FLT_MAX;

    // If the candidate buffer is not completely full, pad the unused entries with INF.
    // This lets us always sort exactly k elements, as required, while ensuring that
    // the dummy entries do not affect the final result.
    for (int i = lane + cand_count; i < k; i += WARP_SIZE)
    {
        cand_dist[i] = INF;
        cand_idx[i]  = -1;
    }
    __syncwarp(mask);

    // Step 1: sort candidate buffer ascending.
    warp_bitonic_sort(cand_dist, cand_idx, k, mask, lane);

    // Step 2: merge candidate buffer and intermediate result into a bitonic sequence.
    // best[i] becomes min(best[i], cand[k-1-i]) for all i.
    for (int i = lane; i < k; i += WARP_SIZE)
    {
        float bd = best_dist[i];
        float cd = cand_dist[k - 1 - i];
        int   bi = best_idx[i];
        int   ci = cand_idx[k - 1 - i];

        if (cd < bd)
        {
            best_dist[i] = cd;
            best_idx[i]  = ci;
        }
    }
    __syncwarp(mask);

    // Step 3: sort the merged bitonic sequence in best_dist[] using Bitonic sort.
    warp_bitonic_sort(best_dist, best_idx, k, mask, lane);

    // Update candidate count and max_dist (distance of k-th nearest neighbor).
    if (lane == 0)
    {
        *cand_count_ptr = 0;
        max_dist = best_dist[k - 1];
    }
    max_dist = __shfl_sync(mask, max_dist, 0);
    __syncwarp(mask);
}

// Kernel: each warp processes one query point.
// The block cooperatively loads data points into shared memory in batches of TILE_SIZE,
// and each warp reuses these cached data points to compute distances and maintain
// its own k-NN result via a shared candidate buffer and bitonic sort/merge stages.
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result,
                int k)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int tid      = threadIdx.x;
    int warp_id  = tid / WARP_SIZE;
    int lane     = tid % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    bool active = (global_warp_id < query_count);

    extern __shared__ unsigned char shared_mem[];
    size_t offset = 0;

    // Shared memory layout:
    // [ 0 .. TILE_SIZE-1 ]:                 float2 s_data[TILE_SIZE]
    // [ next WARPS_PER_BLOCK * k floats ]:   best_dist per warp
    // [ next WARPS_PER_BLOCK * k ints   ]:   best_idx per warp
    // [ next WARPS_PER_BLOCK * k floats ]:   cand_dist per warp
    // [ next WARPS_PER_BLOCK * k ints   ]:   cand_idx per warp
    // [ next WARPS_PER_BLOCK ints       ]:   cand_count per warp

    float2 *s_data = reinterpret_cast<float2 *>(shared_mem + offset);
    offset += TILE_SIZE * sizeof(float2);

    float *s_best_dist = reinterpret_cast<float *>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(float);

    int *s_best_idx = reinterpret_cast<int *>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(int);

    float *s_cand_dist = reinterpret_cast<float *>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(float);

    int *s_cand_idx = reinterpret_cast<int *>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(int);

    int *s_cand_count = reinterpret_cast<int *>(shared_mem + offset);

    // Per-warp slices in shared memory
    float *best_dist     = s_best_dist + warp_id * k;
    int   *best_idx      = s_best_idx  + warp_id * k;
    float *cand_dist     = s_cand_dist + warp_id * k;
    int   *cand_idx      = s_cand_idx  + warp_id * k;
    int   *cand_count_ptr = s_cand_count + warp_id;

    float max_dist = FLT_MAX;  // distance of current k-th nearest neighbor for this query

    // Initialize intermediate result and candidate buffer state for active warps.
    if (active)
    {
        if (lane == 0)
        {
            *cand_count_ptr = 0;
        }
        // Initialize intermediate result with "infinite" distances and invalid indices.
        for (int i = lane; i < k; i += WARP_SIZE)
        {
            best_dist[i] = max_dist;
            best_idx[i]  = -1;
        }
    }

    // Ensure all threads see initialized shared state before starting batch processing.
    __syncthreads();

    // Load query point for this warp and broadcast to all its lanes.
    float2 q;
    if (lane == 0)
    {
        if (active)
            q = query[global_warp_id];
        else
        {
            // Dummy value for inactive warps (they will not compute or write results).
            q.x = 0.0f;
            q.y = 0.0f;
        }
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Process data points in batches, cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE)
    {
        int remaining   = data_count - tile_start;
        int tile_size   = (remaining < TILE_SIZE) ? remaining : TILE_SIZE;

        // Block-wide load of the current batch of data points into shared memory.
        for (int i = tid; i < tile_size; i += blockDim.x)
        {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active)
        {
            // Each warp processes all points in the current tile.
            for (int i_local = lane; i_local < tile_size; i_local += WARP_SIZE)
            {
                int    data_idx_global = tile_start + i_local;
                float2 p               = s_data[i_local];

                float dx   = q.x - p.x;
                float dy   = q.y - p.y;
                float dist = dx * dx + dy * dy;

                // Rough candidate filtering based on current max_dist.
                bool is_candidate = (dist < max_dist);

                // Identify which lanes have candidates for this data point.
                unsigned candidate_mask = __ballot_sync(FULL_MASK, is_candidate);

                // Process candidate lanes one by one in a warp-synchronous manner.
                while (candidate_mask)
                {
                    int cand_lane = __ffs(candidate_mask) - 1;

                    // Broadcast candidate distance and index from the candidate lane.
                    float cand_dist_val = __shfl_sync(FULL_MASK, dist, cand_lane);
                    int   cand_idx_val  = __shfl_sync(FULL_MASK, data_idx_global, cand_lane);

                    bool flush_needed = false;

                    // Lane 0 performs the actual insertion into the shared candidate buffer.
                    if (lane == 0)
                    {
                        // Re-check against current max_dist in case it changed since is_candidate was computed.
                        if (cand_dist_val < max_dist)
                        {
                            int pos = atomicAdd(cand_count_ptr, 1);
                            if (pos < k)
                            {
                                cand_dist[pos] = cand_dist_val;
                                cand_idx[pos]  = cand_idx_val;

                                // Buffer just became full: we need to flush/merge it.
                                if (pos == k - 1)
                                {
                                    flush_needed = true;
                                }
                            }
                            else
                            {
                                // This should be rare; cap the count and force a flush.
                                *cand_count_ptr = k;
                                flush_needed = true;
                            }
                        }
                    }

                    flush_needed = __shfl_sync(FULL_MASK, flush_needed, 0);

                    if (flush_needed)
                    {
                        // Merge candidate buffer into the intermediate result.
                        warp_flush_candidates(best_dist, best_idx,
                                              cand_dist, cand_idx,
                                              k, lane, FULL_MASK,
                                              cand_count_ptr,
                                              max_dist);
                    }

                    // Remove this lane from the candidate mask.
                    candidate_mask &= candidate_mask - 1;
                }
            }
        }

        // Synchronize before loading the next tile.
        __syncthreads();
    }

    // After all batches, flush any remaining candidates.
    if (active)
    {
        if (*cand_count_ptr > 0)
        {
            warp_flush_candidates(best_dist, best_idx,
                                  cand_dist, cand_idx,
                                  k, lane, FULL_MASK,
                                  cand_count_ptr,
                                  max_dist);
        }

        // Write out the final k nearest neighbors for this query.
        for (int i = lane; i < k; i += WARP_SIZE)
        {
            int out_idx = global_warp_id * k + i;
            result[out_idx].first  = best_idx[i];
            result[out_idx].second = best_dist[i];
        }
    }
}

// Host interface as requested.
//
// Assumptions:
//  - 'query', 'data', and 'result' are device pointers obtained via cudaMalloc.
//  - data_count >= k, and k is a power of two between 32 and 1024 inclusive.
//  - query_count and data_count are large enough to benefit from GPU execution.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    dim3 block_dim(THREADS_PER_BLOCK);
    int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(num_blocks);

    // Compute required dynamic shared memory size.
    size_t shared_bytes = 0;
    shared_bytes += static_cast<size_t>(TILE_SIZE) * sizeof(float2);      // s_data
    shared_bytes += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float); // s_best_dist
    shared_bytes += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);   // s_best_idx
    shared_bytes += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float); // s_cand_dist
    shared_bytes += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);   // s_cand_idx
    shared_bytes += static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);       // s_cand_count

    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(query, query_count,
                                                      data, data_count,
                                                      result, k);
    cudaDeviceSynchronize();
}