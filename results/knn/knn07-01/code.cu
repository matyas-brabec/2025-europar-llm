// kNN CUDA implementation for 2D points using a warp-per-query approach.
// Each warp processes one query and maintains its own intermediate top-k
// result and candidate buffer in shared memory.
//
// The algorithm follows the specification:
//  - Process data in batches cached in shared memory (per block).
//  - For each query (one warp):
//      * Maintain a sorted intermediate result of k nearest neighbors.
//      * Maintain a candidate buffer of size k in shared memory.
//      * Use atomicAdd on a shared counter to assign positions in the
//        candidate buffer for each new candidate.
//      * When the candidate buffer is full, sort it with Bitonic Sort,
//        merge it with the intermediate result using a bitonic merge
//        step, then sort the merged result again with Bitonic Sort to
//        update the intermediate top-k.
//  - After the last batch, if the candidate buffer is non-empty, perform
//    a final merge.
//
// Implementation notes:
//  - Exactly one warp (32 threads) handles each query.
//  - Bitonic Sort is implemented serially (lane 0 in the warp) over the
//    k elements. This is acceptable because the expensive work is the
//    distance computation over millions of data points.
//  - Threads in a warp cooperate via __shfl_sync to pass candidate
//    distances/indices to lane 0 for insertion into the candidate
//    buffer.
//  - Shared memory includes:
//      * A data tile for caching a batch of data points per block.
//      * Per-warp regions for:
//          - Candidate distances and indices (k each).
//          - Intermediate result distances and indices (k each).
//          - Temporary merged distances and indices (k each).
//          - A shared int candidate_count per warp.
//  - No additional device allocations are used (only shared memory).
//
// This code assumes compilation with a recent CUDA toolkit and a
// data-center GPU such as A100/H100.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Constants for warp size and warps per block.
// One warp (32 threads) processes one query.
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 6;   // 6 warps => 192 threads per block

// Simple helper for swapping a pair (distance, index).
__device__ __forceinline__ void swap_pair(float &a_dist, int &a_idx,
                                          float &b_dist, int &b_idx) {
    float tmpd = a_dist;
    a_dist = b_dist;
    b_dist = tmpd;
    int tmpi = a_idx;
    a_idx = b_idx;
    b_idx = tmpi;
}

// Serial Bitonic Sort for an array of length k (k is a power of two).
// Sorts in ascending order by distance, carrying indices along.
// This is intended to be called from lane 0 only; other threads in the
// warp simply wait for completion (via an outer __syncwarp()).
__device__ void bitonic_sort_serial(float *dist, int *idx, int k) {
    // Reference pseudocode (adapted) from the problem statement:
    // for (k = 2; k <= n; k *= 2)
    //   for (j = k/2; j > 0; j /= 2)
    //     for (i = 0; i < n; i++)
    //       l = i ^ j;
    //       if (l > i)
    //         if ( ((i & k) == 0 && arr[i] > arr[l]) ||
    //              ((i & k) != 0 && arr[i] < arr[l]) )
    //           swap(arr[i], arr[l]);

    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = 0; i < k; ++i) {
                int l = i ^ stride;
                if (l > i) {
                    bool ascending = ((i & size) == 0);
                    bool comp = (dist[i] > dist[l]);
                    if (!ascending) comp = !comp;
                    if (comp) {
                        swap_pair(dist[i], idx[i], dist[l], idx[l]);
                    }
                }
            }
        }
    }
}

// Merge the candidate buffer with the intermediate top-k result for a
// given query, following the specified algorithm:
//
// 0. The intermediate result is sorted in ascending order.
// 1. Sort the candidate buffer with Bitonic Sort (ascending).
// 2. Merge candidate buffer and intermediate result into a bitonic
//    sequence of length k: merged[i] = min(buffer[i], intermediate[k-1-i]).
// 3. Sort the merged result with Bitonic Sort to obtain the new
//    intermediate result.
//
// Precondition:
//  - The candidate buffer is of length k; if there are fewer than k
//    valid candidates, the remaining entries must already be filled with
//    very large distances (e.g., FLT_MAX) so that they are safely
//    discarded.
//  - intermediate_dist / intermediate_idx contain a sorted (ascending)
//    top-k result for this query.
//
// Postcondition:
//  - intermediate_dist / intermediate_idx contain the updated sorted
//    top-k result.
//  - max_distance is updated to the k-th (largest) distance in the
//    intermediate result.
__device__ void merge_candidate_buffer(
    float *candidate_dist,
    int   *candidate_idx,
    float *intermediate_dist,
    int   *intermediate_idx,
    float *merged_dist,
    int   *merged_idx,
    int    k,
    int    lane_id,
    float &max_distance)
{
    // All threads in the warp call this function, but only lane 0
    // performs the actual work. Other threads participate only in the
    // final __syncwarp().
    if (lane_id == 0) {
        // 1. Sort the candidate buffer (ascending) using Bitonic Sort.
        bitonic_sort_serial(candidate_dist, candidate_idx, k);

        // 2. Merge candidate buffer and intermediate result into a
        //    bitonic sequence of length k.
        //    merged[i] = min(candidate[i], intermediate[k-1-i])
        for (int i = 0; i < k; ++i) {
            float d1 = candidate_dist[i];
            float d2 = intermediate_dist[k - 1 - i];
            if (d1 < d2) {
                merged_dist[i] = d1;
                merged_idx[i]  = candidate_idx[i];
            } else {
                merged_dist[i] = d2;
                merged_idx[i]  = intermediate_idx[k - 1 - i];
            }
        }

        // 3. Sort the merged result (ascending) to obtain the updated
        //    intermediate top-k.
        bitonic_sort_serial(merged_dist, merged_idx, k);

        // Copy merged result back into the intermediate arrays.
        for (int i = 0; i < k; ++i) {
            intermediate_dist[i] = merged_dist[i];
            intermediate_idx[i]  = merged_idx[i];
        }

        // Update max_distance to the k-th nearest neighbor distance.
        max_distance = intermediate_dist[k - 1];
    }

    // Ensure that all threads in the warp see the updated intermediate
    // result before proceeding.
    __syncwarp();
}

// CUDA kernel implementing k-NN as specified.
// Each warp handles one query point.
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int                         query_count,
    const float2 * __restrict__ data,
    int                         data_count,
    std::pair<int, float> * __restrict__ result,
    int                         k)
{
    // Shared memory layout:
    // [0 .. TILE_SIZE-1]                    : float2 data tile
    // [TILE_SIZE .. end] per-warp segments:
    //   int candidate_count;
    //   float candidate_dist[k];
    //   int candidate_idx[k];
    //   float intermediate_dist[k];
    //   int intermediate_idx[k];
    //   float merged_dist[k];
    //   int merged_idx[k];

    extern __shared__ unsigned char shared_mem[];

    const int thread_id          = threadIdx.x;
    const int warp_id_in_block   = thread_id / WARP_SIZE;
    const int lane_id            = thread_id & (WARP_SIZE - 1);
    const int global_warp_id     = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    const bool warp_active       = (global_warp_id < query_count);

    const int TILE_SIZE          = blockDim.x; // One data element loaded per thread.

    // Shared memory pointer to the data tile.
    float2 *sh_data = reinterpret_cast<float2*>(shared_mem);

    // Compute per-warp shared-memory base address.
    const size_t per_warp_bytes = sizeof(int) +
                                  3 * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    unsigned char *warp_base = shared_mem +
                               static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
                               warp_id_in_block * per_warp_bytes;

    // Lay out per-warp shared structures.
    int   *candidate_count_ptr = reinterpret_cast<int*>(warp_base);
    float *candidate_dist      = reinterpret_cast<float*>(candidate_count_ptr + 1);
    int   *candidate_idx       = reinterpret_cast<int*>(candidate_dist + k);
    float *intermediate_dist   = reinterpret_cast<float*>(candidate_idx + k);
    int   *intermediate_idx    = reinterpret_cast<int*>(intermediate_dist + k);
    float *merged_dist         = reinterpret_cast<float*>(intermediate_idx + k);
    int   *merged_idx          = reinterpret_cast<int*>(merged_dist + k);

    // Initialize per-warp state.
    // All warps (active or not) initialize their region to avoid any
    // undefined behavior; inactive warps will simply not write results.
    const float INF = FLT_MAX;

    float max_distance = INF;  // Per-query max distance (k-th neighbor).

    if (lane_id == 0) {
        // Initialize candidate buffer count.
        *candidate_count_ptr = 0;

        // Initialize intermediate result with "infinite" distances and
        // invalid indices.
        for (int i = 0; i < k; ++i) {
            intermediate_dist[i] = INF;
            intermediate_idx[i]  = -1;
        }
    }
    __syncwarp();

    // Load query coordinates and broadcast within the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active && lane_id == 0) {
        float2 q = query[global_warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // Process data in tiles cached in shared memory.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_SIZE) {
        // Load a tile of data points into shared memory.
        int global_data_idx = tile_base + thread_id;
        if (global_data_idx < data_count) {
            sh_data[thread_id] = data[global_data_idx];
        }
        __syncthreads();

        int tile_count = data_count - tile_base;
        if (tile_count > TILE_SIZE) tile_count = TILE_SIZE;

        // Each warp processes all points in the tile.
        // We iterate over the tile in chunks of WARP_SIZE. In each chunk,
        // every lane computes the distance to one data point (where
        // available), then these distances are sequentially offered to
        // lane 0 via shuffles for candidate insertion.
        for (int local_base = 0; local_base < tile_count; local_base += WARP_SIZE) {
            int   local_idx = local_base + lane_id;
            float dist      = INF;
            int   idx       = -1;

            if (local_idx < tile_count && warp_active) {
                float2 p = sh_data[local_idx];
                float dx = p.x - qx;
                float dy = p.y - qy;
                dist = dx * dx + dy * dy;
                idx  = tile_base + local_idx;
            }

            // Sequentially pass each lane's (dist, idx) to lane 0.
            for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
                float cand_dist_val = __shfl_sync(0xFFFFFFFFu, dist, src_lane);
                int   cand_idx_val  = __shfl_sync(0xFFFFFFFFu, idx,  src_lane);

                bool need_merge = false;

                if (warp_active && lane_id == 0 && cand_idx_val >= 0 &&
                    cand_dist_val < max_distance)
                {
                    // Use atomicAdd to update the candidate count and
                    // determine the write position in the buffer, as
                    // required by the specification.
                    int pos = atomicAdd(candidate_count_ptr, 1);

                    // Store candidate in the buffer at the assigned position.
                    candidate_dist[pos] = cand_dist_val;
                    candidate_idx[pos]  = cand_idx_val;

                    int new_count = pos + 1;

                    // If the buffer is full, we need to merge it with the
                    // intermediate result.
                    if (new_count == k) {
                        need_merge = true;
                    }
                }

                // Determine if any lane requested a merge (only lane 0
                // can set need_merge, but we use ballot for clarity).
                unsigned int merge_mask = __ballot_sync(0xFFFFFFFFu, need_merge);

                if (merge_mask) {
                    // All threads in the warp participate in the merge
                    // function; only lane 0 performs actual work.
                    merge_candidate_buffer(
                        candidate_dist,
                        candidate_idx,
                        intermediate_dist,
                        intermediate_idx,
                        merged_dist,
                        merged_idx,
                        k,
                        lane_id,
                        max_distance);

                    if (lane_id == 0) {
                        // Reset candidate buffer count after merge.
                        *candidate_count_ptr = 0;
                    }
                    __syncwarp();
                }
            }
        }

        // Ensure all warps are done with the current tile before loading
        // the next one.
        __syncthreads();
    }

    // After processing all data tiles, perform a final merge if the
    // candidate buffer is not empty.
    int cand_count = 0;
    if (lane_id == 0) {
        cand_count = *candidate_count_ptr;
    }
    cand_count = __shfl_sync(0xFFFFFFFFu, cand_count, 0);

    if (warp_active && cand_count > 0) {
        // Fill remaining entries of the candidate buffer with INF so
        // that the Bitonic Sort operates on a full array of length k
        // (as required), while effectively ignoring the extra entries.
        if (lane_id == 0) {
            for (int i = cand_count; i < k; ++i) {
                candidate_dist[i] = INF;
                candidate_idx[i]  = -1;
            }
        }
        __syncwarp();

        merge_candidate_buffer(
            candidate_dist,
            candidate_idx,
            intermediate_dist,
            intermediate_idx,
            merged_dist,
            merged_idx,
            k,
            lane_id,
            max_distance);

        if (lane_id == 0) {
            *candidate_count_ptr = 0;
        }
        __syncwarp();
    }

    // Write final k nearest neighbors for this query to global memory.
    if (warp_active) {
        int query_idx = global_warp_id;
        int out_base  = query_idx * k;

        for (int i = lane_id; i < k; i += WARP_SIZE) {
            result[out_base + i].first  = intermediate_idx[i];
            result[out_base + i].second = intermediate_dist[i];
        }
    }
}

// Host interface function as specified.
// Assumes query, data, and result are device pointers allocated by
// cudaMalloc.
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    // Determine kernel launch configuration.
    const int threads_per_block = WARP_SIZE * WARPS_PER_BLOCK; // 192 threads
    const int blocks =
        (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Compute required dynamic shared memory size.
    const int tile_size = threads_per_block; // One data element per thread.

    const size_t per_warp_bytes =
        sizeof(int) +
        3 * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    const size_t shared_mem_bytes =
        static_cast<size_t>(tile_size) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_bytes;

    // Launch kernel.
    knn_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k);

    // Error checking and synchronization are intentionally omitted here
    // to keep the interface minimal; they can be added by the caller if
    // desired.
}