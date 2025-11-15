#include <cuda_runtime.h>
#include <float.h>
#include <utility>

/*
 * Optimized CUDA implementation of k-nearest neighbors (k-NN) for 2D points.
 *
 * Design overview:
 * ----------------
 * - Each query is processed by a single warp (32 threads).
 * - A thread block contains WARPS_PER_BLOCK warps, so each block handles
 *   WARPS_PER_BLOCK queries.
 * - The dataset is processed in batches; each batch of points is cached into
 *   shared memory by the whole block.
 * - For each query/warp:
 *     * We maintain an intermediate top-k result in shared memory
 *       (indices + distances).
 *     * We maintain a candidate buffer of up to k elements in shared memory
 *       (indices + distances) and a shared candidate counter.
 *     * We maintain a per-warp max_distance which is the current k-th smallest
 *       distance (i.e., the distance of the worst neighbor in the intermediate
 *       top-k result). Distances >= max_distance are ignored.
 * - New candidate distances are filtered by max_distance. When accepted, they
 *   are added to the candidate buffer. The number of stored candidates is
 *   updated via atomicAdd on the shared candidate counter as required.
 * - When the candidate buffer would overflow (cand_count + new_candidates > k),
 *   the warp merges the candidate buffer into the intermediate top-k result:
 *     * We interpret the per-warp shared arrays as 2*k entries:
 *         - [0 .. k-1]   : current top-k result (best)
 *         - [k .. 2*k-1] : candidate buffer
 *     * Unused slots are filled with sentinel values (distance = FLT_MAX,
 *       index = -1).
 *     * We perform a warp-synchronous bitonic sort on the 2*k entries
 *       (using all 32 threads), and keep the first k entries as the new
 *       intermediate top-k.
 *     * We update max_distance to the k-th distance when we have at least k
 *       neighbors; otherwise it stays FLT_MAX.
 * - After all data batches are processed, each warp does a final merge if
 *   the candidate buffer is non-empty, and writes its top-k result to the
 *   output array.
 *
 * Constraints and assumptions:
 * ----------------------------
 * - k is a power of two, between 32 and 1024 inclusive.
 * - data_count >= k.
 * - query, data, and result point to device memory allocated with cudaMalloc.
 * - No additional device memory allocations are performed.
 * - Code targets modern NVIDIA data center GPUs (A100/H100) with sufficient
 *   shared memory per block.
 * - Distances are squared Euclidean distances in 2D: (dx^2 + dy^2).
 */

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of warps per block: 8 warps * 32 threads = 256 threads per block.
static constexpr int WARPS_PER_BLOCK    = 8;
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;

// Batch size for caching data points in shared memory.
// Chosen to stay within shared-memory limits for the worst case (k=1024).
static constexpr int BATCH_SIZE = 2048;

/*
 * Warp-synchronous bitonic sort on an array of N elements (N is a power of two).
 * - dist[i]  : distances
 * - idx[i]   : corresponding indices
 * - N        : number of elements, up to 2*k (k <= 1024)
 *
 * The sort is in ascending order by dist.
 * All 32 threads of the warp participate. Each thread processes indices
 * i = lane, lane + WARP_SIZE, lane + 2*WARP_SIZE, ...
 */
__device__ __forceinline__
void warp_bitonic_sort(float *dist, int *idx, int N)
{
    const unsigned mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Standard bitonic sort network
    for (int size = 2; size <= N; size <<= 1)
    {
        // For each size, perform log2(size) merge steps
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (int i = lane; i < N; i += WARP_SIZE)
            {
                int partner = i ^ stride;
                if (partner > i && partner < N)
                {
                    bool ascending = ((i & size) == 0);

                    float di = dist[i];
                    float dj = dist[partner];
                    int   ii = idx[i];
                    int   ij = idx[partner];

                    bool swap_needed = (di > dj);
                    if (swap_needed == ascending)
                    {
                        dist[i]      = dj;
                        dist[partner] = di;
                        idx[i]       = ij;
                        idx[partner] = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

/*
 * Merge the candidate buffer for a given warp into its intermediate top-k result.
 *
 * Shared-memory layout per warp (for warp_id):
 * - indices[warp_id * (2*k) + 0 .. k-1]   : current best (top-k)
 * - indices[warp_id * (2*k) + k .. 2*k-1] : candidates
 * Same layout for distances[].
 *
 * Inputs (per warp):
 * - sh_best_counts[warp_id] : number of valid entries in current best
 * - sh_cand_counts[warp_id] : number of valid entries in candidates
 * - sh_max_dists[warp_id]   : current max_distance (k-th neighbor), updated here
 *
 * Steps:
 * 1. Fill unused best slots [best_count .. k-1] with sentinel (FLT_MAX, -1).
 * 2. Fill unused candidate slots [cand_count .. k-1] in the candidate segment
 *    with sentinel.
 * 3. Run warp_bitonic_sort on the 2*k combined entries.
 * 4. The first k entries after sort are the new best set; update best_count,
 *    reset cand_count to 0, and update max_distance accordingly.
 */
__device__ __forceinline__
void flush_candidates_for_warp(
    int   warp_id,
    int   top_k,
    int  *sh_indices,
    float *sh_dists,
    int  *sh_best_counts,
    int  *sh_cand_counts,
    float *sh_max_dists)
{
    const unsigned mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    int cand_count = sh_cand_counts[warp_id];
    if (cand_count == 0)
    {
        __syncwarp(mask);
        return;
    }

    int best_count = sh_best_counts[warp_id];

    int   *warp_indices = sh_indices + warp_id * (2 * top_k);
    float *warp_dists   = sh_dists   + warp_id * (2 * top_k);

    // Fill unused best slots [best_count, top_k) with sentinel.
    for (int i = lane + best_count; i < top_k; i += WARP_SIZE)
    {
        warp_dists[i]   = FLT_MAX;
        warp_indices[i] = -1;
    }

    // Fill unused candidate slots [cand_count, top_k) in candidate segment
    // (positions [top_k + cand_count, top_k + top_k)).
    for (int i = lane + cand_count; i < top_k; i += WARP_SIZE)
    {
        int pos = top_k + i;
        warp_dists[pos]   = FLT_MAX;
        warp_indices[pos] = -1;
    }

    __syncwarp(mask);

    // Sort the combined 2*top_k elements in ascending order of distance.
    int N = 2 * top_k;
    warp_bitonic_sort(warp_dists, warp_indices, N);

    __syncwarp(mask);

    // Update best_count, candidate_count, and max_distance.
    if (lane == 0)
    {
        int new_best_count = best_count + cand_count;
        if (new_best_count > top_k)
            new_best_count = top_k;

        sh_best_counts[warp_id] = new_best_count;
        sh_cand_counts[warp_id] = 0;

        // If we don't yet have k elements, keep max_distance = FLT_MAX so
        // that all upcoming distances are accepted as candidates.
        if (new_best_count < top_k)
            sh_max_dists[warp_id] = FLT_MAX;
        else
            sh_max_dists[warp_id] = warp_dists[top_k - 1];
    }

    __syncwarp(mask);
}

/*
 * Main CUDA kernel implementing k-NN for 2D points using a warp-per-query scheme.
 *
 * Parameters:
 * - query[query_count] : array of 2D query points (float2)
 * - data[data_count]   : array of 2D data points (float2)
 * - result             : output array of size query_count * k, where
 *       result[i * k + j].first  = index of j-th nearest neighbor for query i
 *       result[i * k + j].second = squared distance to that neighbor
 * - top_k              : number of neighbors (k), power of two in [32, 1024]
 */
__global__ void knn_kernel(
    const float2 *__restrict__ query,
    int                     query_count,
    const float2 *__restrict__ data,
    int                     data_count,
    std::pair<int, float> *__restrict__ result,
    int                     top_k)
{
    // Dynamic shared memory layout:
    // [0 .. BATCH_SIZE-1] : float2 sh_data[ BATCH_SIZE ]
    // Then per-warp arrays:
    // int   sh_indices[ WARPS_PER_BLOCK * 2 * top_k ];
    // float sh_dists  [ WARPS_PER_BLOCK * 2 * top_k ];
    // int   sh_best_counts[ WARPS_PER_BLOCK ];
    // int   sh_cand_counts[ WARPS_PER_BLOCK ];
    // float sh_max_dists  [ WARPS_PER_BLOCK ];

    extern __shared__ unsigned char smem[];

    float2 *sh_data = reinterpret_cast<float2*>(smem);
    size_t offset   = sizeof(float2) * BATCH_SIZE;

    int *sh_indices = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * WARPS_PER_BLOCK * 2 * top_k;

    float *sh_dists = reinterpret_cast<float*>(smem + offset);
    offset += sizeof(float) * WARPS_PER_BLOCK * 2 * top_k;

    int *sh_best_counts = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * WARPS_PER_BLOCK;

    int *sh_cand_counts = reinterpret_cast<int*>(smem + offset);
    offset += sizeof(int) * WARPS_PER_BLOCK;

    float *sh_max_dists = reinterpret_cast<float*>(smem + offset);
    // offset += sizeof(float) * WARPS_PER_BLOCK; // not needed further

    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id   = threadIdx.x >> 5;  // threadIdx.x / WARP_SIZE
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const bool valid_warp = (warp_global < query_count);

    // Load the query point for this warp (only lane 0 actually loads).
    float2 q;
    if (lane == 0 && valid_warp)
    {
        q = query[warp_global];
    }

    // Broadcast query coordinates to all lanes in the warp.
    q.x = __shfl_sync(0xFFFFFFFFu, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFFu, q.y, 0);

    // Initialize per-warp bookkeeping (for all warps; unused warps won't process data).
    if (lane == 0)
    {
        sh_best_counts[warp_id] = 0;
        sh_cand_counts[warp_id] = 0;
        sh_max_dists[warp_id]   = FLT_MAX;
    }
    __syncwarp();

    // Process the data points in batches, caching each batch in shared memory.
    for (int base = 0; base < data_count; base += BATCH_SIZE)
    {
        int batch_size = data_count - base;
        if (batch_size > BATCH_SIZE)
            batch_size = BATCH_SIZE;

        // Load current batch of data points into shared memory using the whole block.
        for (int idx = threadIdx.x; idx < batch_size; idx += blockDim.x)
        {
            sh_data[idx] = data[base + idx];
        }

        __syncthreads();  // Ensure sh_data is fully populated.

        if (valid_warp)
        {
            // Each warp computes distances from its query to all points in current batch.
            for (int idx = lane; idx < batch_size; idx += WARP_SIZE)
            {
                float2 p = sh_data[idx];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                float dist = dx * dx + dy * dy;

                // Filter by current max_distance.
                float max_dist = sh_max_dists[warp_id];
                bool is_candidate = (dist < max_dist);

                // Determine which lanes have candidates in this iteration.
                unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, is_candidate);
                if (active_mask == 0)
                    continue;

                int num_candidates = __popc(active_mask);

                // Lane 0 checks if the candidate buffer would overflow.
                int cand_count = 0;
                if (lane == 0)
                {
                    cand_count = sh_cand_counts[warp_id];
                }
                cand_count = __shfl_sync(0xFFFFFFFFu, cand_count, 0);

                int need_flush = (cand_count + num_candidates > top_k) ? 1 : 0;
                need_flush = __shfl_sync(0xFFFFFFFFu, need_flush, 0);

                if (need_flush)
                {
                    // Merge existing candidates into the intermediate result.
                    flush_candidates_for_warp(
                        warp_id, top_k,
                        sh_indices, sh_dists,
                        sh_best_counts, sh_cand_counts,
                        sh_max_dists);

                    // Re-read candidate count after flush (should be 0).
                    if (lane == 0)
                    {
                        cand_count = sh_cand_counts[warp_id];
                    }
                    cand_count = __shfl_sync(0xFFFFFFFFu, cand_count, 0);
                }

                // Reserve positions in candidate buffer using atomicAdd on the shared counter.
                int base_index = 0;
                if (lane == 0)
                {
                    base_index = atomicAdd(&sh_cand_counts[warp_id], num_candidates);
                }
                base_index = __shfl_sync(0xFFFFFFFFu, base_index, 0);

                // Compute per-lane offset within this group's candidates.
                int offset_in_group = __popc(active_mask & ((1u << lane) - 1));

                if (is_candidate)
                {
                    int pos = base_index + offset_in_group;

                    int   *cand_indices = sh_indices + warp_id * (2 * top_k) + top_k;
                    float *cand_dists   = sh_dists   + warp_id * (2 * top_k) + top_k;

                    cand_indices[pos] = base + idx;  // global index of the data point
                    cand_dists[pos]   = dist;
                }
            }
        }

        __syncthreads();  // All warps finish with this batch before loading the next.
    }

    if (valid_warp)
    {
        // Final merge of any remaining candidates.
        flush_candidates_for_warp(
            warp_id, top_k,
            sh_indices, sh_dists,
            sh_best_counts, sh_cand_counts,
            sh_max_dists);

        int   *warp_indices = sh_indices + warp_id * (2 * top_k);
        float *warp_dists   = sh_dists   + warp_id * (2 * top_k);

        int best_count = sh_best_counts[warp_id];
        int write_k    = (best_count < top_k) ? best_count : top_k;

        int out_base = warp_global * top_k;

        // Write the k nearest neighbors for this query.
        // The array warp_dists[0..top_k-1] is sorted ascending, so index 0 is the nearest.
        for (int i = lane; i < write_k; i += WARP_SIZE)
        {
            int   idx  = warp_indices[i];
            float dist = warp_dists[i];
            result[out_base + i].first  = idx;
            result[out_base + i].second = dist;
        }

        // In the unlikely case that best_count < top_k, fill remaining slots with dummy values.
        if (write_k < top_k)
        {
            for (int i = lane + write_k; i < top_k; i += WARP_SIZE)
            {
                result[out_base + i].first  = -1;
                result[out_base + i].second = FLT_MAX;
            }
        }
    }
}

/*
 * Host interface:
 *
 * Launches the knn_kernel with appropriate grid/block dimensions and dynamic
 * shared memory size. All pointers (query, data, result) are assumed to be
 * device pointers allocated with cudaMalloc.
 */
void run_knn(
    const float2 *query, int query_count,
    const float2 *data,  int data_count,
    std::pair<int, float> *result,
    int k)
{
    // Compute grid and block dimensions.
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Compute required dynamic shared memory size.
    //   BATCH_SIZE * sizeof(float2) for sh_data
    // + WARPS_PER_BLOCK * 2*k * (sizeof(int) + sizeof(float)) for per-warp indices/distances
    // + WARPS_PER_BLOCK * (sizeof(int) + sizeof(int) + sizeof(float)) for counts and max distances
    size_t shared_mem_bytes =
        sizeof(float2) * BATCH_SIZE +
        WARPS_PER_BLOCK * (2 * k * (sizeof(int) + sizeof(float)) +
                           sizeof(int) + sizeof(int) + sizeof(float));

    knn_kernel<<<grid, block, shared_mem_bytes>>>(
        query, query_count,
        data,  data_count,
        result, k);

    // Synchronize to ensure kernel completion and to expose any launch errors.
    cudaDeviceSynchronize();
}