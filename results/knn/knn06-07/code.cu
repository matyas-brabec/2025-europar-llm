// CUDA k-NN (k-nearest neighbors) for 2D points.
// Each query is processed by a single warp (32 threads).
// Data points are processed in tiles that are cached in shared memory.
// For every query, we maintain:
//   - An intermediate result of k best neighbors (indices + distances) in shared memory.
//   - A candidate buffer of up to k elements (indices + distances) in shared memory.
//   - A shared counter (with atomicAdd) indicating the number of candidates.
//
// When the candidate buffer is (about to be) full, we merge it with the
// intermediate result using all threads of the warp. The merge is implemented
// as a warp-parallel selection of the k smallest distances from the union of
// current bests and candidates.
//
// Distances are squared Euclidean distances (no square root).
//
// This code targets modern data center GPUs (e.g., A100/H100) and uses
// standard CUDA C++ with dynamic shared memory.

#include <cuda_runtime.h>
#include <utility>

// Number of threads in a warp. We assume NVIDIA hardware where warp size is 32.
constexpr int WARP_SIZE = 32;

// Number of data points (float2) loaded into shared memory per tile.
// 1024 is a good compromise between shared memory usage and global memory reuse.
// Shared memory per block for k = 1024:
//   tile:           1024 * sizeof(float2)       =  8 192 B
//   cand_count:     1 * sizeof(int)            =      4 B
//   best_idx:       k * sizeof(int)            =  4 096 B
//   best_dist:      k * sizeof(float)          =  4 096 B
//   cand_idx:       k * sizeof(int)            =  4 096 B
//   cand_dist:      k * sizeof(float)          =  4 096 B
//   temp_idx:       2k * sizeof(int)           =  8 192 B
//   temp_dist:      2k * sizeof(float)         =  8 192 B
// Total (k=1024):                                40 964 B < 48 KB limit
constexpr int TILE_SIZE = 1024;

// Utility: return positive infinity for float, usable in device code.
__device__ __forceinline__ float device_inf()
{
    return CUDART_INF_F;
}

// Merge candidate buffer into the current best-k set for a single query.
//
// Parameters:
//   k           - number of neighbors.
//   cand_count  - number of candidates currently stored in cand_idx/cand_dist.
//   best_idx    - shared memory array of length k holding current neighbor indices.
//   best_dist   - shared memory array of length k holding current neighbor distances.
//   cand_idx    - shared memory candidate index buffer (length >= cand_count).
//   cand_dist   - shared memory candidate distance buffer (length >= cand_count).
//   temp_idx    - shared memory temp array (length >= k + cand_count).
//   temp_dist   - shared memory temp array (length >= k + cand_count).
//   current_max - current max_distance (distance of k-th neighbor) for this query.
//
// Returns:
//   Updated max_distance (distance of the k-th neighbor after merge).
//
// Implementation details:
//   - Single warp (32 threads) cooperates.
//   - We first copy the k current best entries and cand_count candidates into
//     a temporary pool temp_* of size pool_size = k + cand_count.
//   - Then we perform k iterations of a warp-parallel selection to find the
//     k smallest distances in this pool:
//       * Each thread scans a subset of the pool and keeps a local minimum.
//       * A warp-wide reduction via __shfl_down_sync finds the global minimum.
//       * The winner is written into best_*[out] and its position in temp_* is
//         marked with INF so it is not selected again.
//   - The result best_* is sorted by distance in ascending order.
//   - We compute the new max_distance as best_dist[k-1] and broadcast it to
//     all threads in the warp via __shfl_sync.
__device__ float merge_candidate_buffer(
    int k,
    int cand_count,
    int *best_idx,
    float *best_dist,
    int *cand_idx,
    float *cand_dist,
    int *temp_idx,
    float *temp_dist,
    float current_max)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned full_mask = 0xffffffffu;

    if (cand_count <= 0)
        return current_max;

    // Combine current best-k and candidate buffer into temp arrays.
    // best_* -> temp_*[0 .. k-1]
    for (int i = lane; i < k; i += WARP_SIZE) {
        temp_idx[i]  = best_idx[i];
        temp_dist[i] = best_dist[i];
    }

    // candidates -> temp_*[k .. k + cand_count - 1]
    for (int i = lane; i < cand_count; i += WARP_SIZE) {
        temp_idx[k + i]  = cand_idx[i];
        temp_dist[k + i] = cand_dist[i];
    }

    __syncwarp(full_mask);

    const int pool_size = k + cand_count;

    // Perform k-way selection to extract k smallest distances from pool.
    // After this loop, best_dist[0..k-1] and best_idx[0..k-1] contain the
    // k nearest neighbors in ascending distance order.
    for (int out = 0; out < k; ++out) {
        float local_min_dist = device_inf();
        int   local_min_idx  = -1;
        int   local_min_pos  = -1;

        // Each lane scans a strided subset of [0, pool_size).
        for (int i = lane; i < pool_size; i += WARP_SIZE) {
            float d = temp_dist[i];
            if (d < local_min_dist) {
                local_min_dist = d;
                local_min_idx  = temp_idx[i];
                local_min_pos  = i;
            }
        }

        // Warp-wide reduction to find the global minimum.
        float best_d = local_min_dist;
        int   best_i = local_min_idx;
        int   best_p = local_min_pos;

        // All 32 lanes are active; use full mask.
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_d = __shfl_down_sync(full_mask, best_d, offset);
            int   other_i = __shfl_down_sync(full_mask, best_i, offset);
            int   other_p = __shfl_down_sync(full_mask, best_p, offset);
            if (other_d < best_d) {
                best_d = other_d;
                best_i = other_i;
                best_p = other_p;
            }
        }

        // Broadcast winner from lane 0.
        float winner_dist = __shfl_sync(full_mask, best_d, 0);
        int   winner_idx  = __shfl_sync(full_mask, best_i, 0);
        int   winner_pos  = __shfl_sync(full_mask, best_p, 0);

        if (lane == 0) {
            best_dist[out] = winner_dist;
            best_idx[out]  = winner_idx;
            // Mark this position as used so it will not be selected again.
            if (winner_pos >= 0) {
                temp_dist[winner_pos] = device_inf();
            }
        }

        __syncwarp(full_mask);
    }

    // Compute new max_distance (distance of the k-th neighbor).
    float new_max = current_max;
    if (lane == 0) {
        new_max = best_dist[k - 1];
    }
    new_max = __shfl_sync(full_mask, new_max, 0);
    return new_max;
}

// CUDA kernel that computes k-NN for one query per block.
// Each block has exactly one warp (32 threads).
//
// Parameters:
//   query       - [query_count] array of float2 queries in device memory.
//   query_count - number of queries.
//   data        - [data_count] array of float2 data points in device memory.
//   data_count  - number of data points (>= k).
//   result      - [query_count * k] array of std::pair<int,float> in device memory.
//                 For query i, result[i * k + j] holds the j-th nearest neighbor.
//   k           - number of neighbors (power of two, 32 <= k <= 1024).
//
// Shared memory layout per block (single warp):
//   float2  tile[TILE_SIZE];          // cached data points
//   int     cand_count;               // candidate buffer size (atomicAdd)
//   int     best_idx[k];              // indices of current best neighbors
//   float   best_dist[k];             // distances of current best neighbors
//   int     cand_idx[k];              // indices in candidate buffer
//   float   cand_dist[k];             // distances in candidate buffer
//   int     temp_idx[2*k];            // temp indices for merge
//   float   temp_dist[2*k];           // temp distances for merge
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k)
{
    const int query_idx = blockIdx.x;  // One query per block.
    if (query_idx >= query_count)
        return;

    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const unsigned full_mask = 0xffffffffu;

    // Dynamic shared memory layout.
    extern __shared__ unsigned char shared_buf[];
    unsigned char *ptr = shared_buf;

    // Tile of data points.
    float2 *tile = reinterpret_cast<float2*>(ptr);
    ptr += TILE_SIZE * sizeof(float2);

    // Candidate count (single int).
    int *cand_count_ptr = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int);

    // Current best neighbors.
    int   *best_idx  = reinterpret_cast<int*>(ptr);
    ptr += k * sizeof(int);
    float *best_dist = reinterpret_cast<float*>(ptr);
    ptr += k * sizeof(float);

    // Candidate buffer.
    int   *cand_idx  = reinterpret_cast<int*>(ptr);
    ptr += k * sizeof(int);
    float *cand_dist = reinterpret_cast<float*>(ptr);
    ptr += k * sizeof(float);

    // Temporary arrays for merge.
    int   *temp_idx  = reinterpret_cast<int*>(ptr);
    ptr += (2 * k) * sizeof(int);
    float *temp_dist = reinterpret_cast<float*>(ptr);
    ptr += (2 * k) * sizeof(float);
    (void)ptr; // suppress unused-variable warning

    // Initialize candidate count and best-k buffers.
    if (threadIdx.x == 0) {
        *cand_count_ptr = 0;
    }

    // Initialize best distances to +inf and indices to -1.
    for (int i = lane; i < k; i += WARP_SIZE) {
        best_dist[i] = device_inf();
        best_idx[i]  = -1;
    }

    __syncthreads();

    // Load the query point once and broadcast to all lanes via shuffles.
    float2 q;
    if (lane == 0) {
        q = query[query_idx];
    }
    q.x = __shfl_sync(full_mask, q.x, 0);
    q.y = __shfl_sync(full_mask, q.y, 0);

    // max_distance: distance of the k-th neighbor for this query.
    // Start with +inf; it will be updated after merges.
    float max_distance = device_inf();

    // Process data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) {
            tile_size = TILE_SIZE;
        }

        // Load the current tile into shared memory using all threads in the block.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }

        // Ensure tile is fully loaded before computing distances.
        __syncthreads();

        // Each lane processes a strided subset of tile entries.
        for (int idx = lane; idx < tile_size; idx += WARP_SIZE) {
            float2 p = tile[idx];

            // Squared Euclidean distance in 2D: (qx - px)^2 + (qy - py)^2.
            float dx   = p.x - q.x;
            float dy   = p.y - q.y;
            float dist = dx * dx + dy * dy;

            // Filter by current max_distance.
            bool is_candidate = (dist < max_distance);

            // Determine how many lanes have a candidate for this point.
            unsigned mask = __ballot_sync(full_mask, is_candidate);
            int n_new = __popc(mask);
            if (n_new == 0) {
                continue; // No candidates from this warp iteration.
            }

            // Check if we need to merge before inserting n_new candidates.
            int cand_count = *cand_count_ptr;
            int need_flush = 0;
            if (lane == 0) {
                // If adding n_new would exceed capacity k, flush current candidates.
                need_flush = (cand_count + n_new > k) ? 1 : 0;
            }
            need_flush = __shfl_sync(full_mask, need_flush, 0);

            if (need_flush) {
                int current_count = 0;
                if (lane == 0) {
                    current_count = *cand_count_ptr;
                }
                current_count = __shfl_sync(full_mask, current_count, 0);

                max_distance = merge_candidate_buffer(
                    k,
                    current_count,
                    best_idx,
                    best_dist,
                    cand_idx,
                    cand_dist,
                    temp_idx,
                    temp_dist,
                    max_distance);

                if (lane == 0) {
                    *cand_count_ptr = 0;
                }
                __syncwarp(full_mask);

                // After merge, max_distance may have changed; recompute candidacy.
                is_candidate = (dist < max_distance);
                mask = __ballot_sync(full_mask, is_candidate);
                n_new = __popc(mask);
                if (n_new == 0) {
                    continue;
                }
            }

            // Reserve space in the candidate buffer using a single atomicAdd.
            int base = 0;
            if (lane == 0) {
                base = atomicAdd(cand_count_ptr, n_new);
            }
            base = __shfl_sync(full_mask, base, 0);

            // Each candidate lane writes its candidate into the reserved range.
            if (is_candidate) {
                // Offset of this lane among set bits in 'mask'.
                unsigned lane_mask = mask & ((1u << lane) - 1u);
                int offset = __popc(lane_mask);
                int pos = base + offset; // position in candidate buffer [0, k)

                cand_idx[pos]  = tile_start + idx; // global data index
                cand_dist[pos] = dist;
            }
        }

        // Synchronize before loading the next tile.
        __syncthreads();
    }

    // After processing all tiles, merge remaining candidates (if any).
    int final_cand_count = 0;
    if (threadIdx.x == 0) {
        final_cand_count = *cand_count_ptr;
    }
    final_cand_count = __shfl_sync(full_mask, final_cand_count, 0);

    if (final_cand_count > 0) {
        max_distance = merge_candidate_buffer(
            k,
            final_cand_count,
            best_idx,
            best_dist,
            cand_idx,
            cand_dist,
            temp_idx,
            temp_dist,
            max_distance);

        if (lane == 0) {
            *cand_count_ptr = 0;
        }
        __syncwarp(full_mask);
    }

    // Write final results for this query.
    // best_dist is sorted in ascending order, so result is ordered by distance.
    for (int i = lane; i < k; i += WARP_SIZE) {
        int out_index = query_idx * k + i;
        result[out_index].first  = best_idx[i];
        result[out_index].second = best_dist[i];
    }
}

// Host-side interface.
//
// Parameters:
//   query       - device pointer to query_count float2 points.
//   query_count - number of queries.
//   data        - device pointer to data_count float2 points.
//   data_count  - number of data points (>= k).
//   result      - device pointer to query_count * k std::pair<int,float>.
//   k           - number of neighbors to find (power of two, 32 <= k <= 1024).
//
// The function launches the GPU kernel with one warp (32 threads) per block
// and one block per query. It uses dynamic shared memory sized according to k.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Basic sanity assumptions as per the problem statement:
    //   - data_count >= k
    //   - 32 <= k <= 1024 and k is a power of two
    //   - pointers are device pointers allocated with cudaMalloc

    const int threads_per_block = WARP_SIZE;   // one warp per block
    const int blocks            = query_count; // one block per query

    // Compute dynamic shared memory size required by knn_kernel for this k.
    size_t shared_bytes = 0;
    shared_bytes += TILE_SIZE * sizeof(float2);     // tile
    shared_bytes += sizeof(int);                    // cand_count
    shared_bytes += k * sizeof(int);                // best_idx
    shared_bytes += k * sizeof(float);              // best_dist
    shared_bytes += k * sizeof(int);                // cand_idx
    shared_bytes += k * sizeof(float);              // cand_dist
    shared_bytes += 2 * k * sizeof(int);            // temp_idx
    shared_bytes += 2 * k * sizeof(float);          // temp_dist

    knn_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    // Error checking is intentionally omitted for brevity; in production code,
    // you may want to call cudaGetLastError()/cudaDeviceSynchronize() here.
}