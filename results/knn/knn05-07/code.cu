// CUDA implementation of k-nearest neighbors (k-NN) for 2D points.
//
// Each query is processed by a single warp (32 threads).
// The block loads chunks ("tiles") of the data set into shared memory.
// For each query, the warp maintains two shared-memory arrays of size 2*k:
//   - First half [0 .. k-1] is the intermediate result (current k nearest neighbors).
//   - Second half [k .. 2*k-1] is a buffer of up to k new candidate neighbors.
// Candidates are only added if they are closer than the current k-th neighbor in the
// intermediate result, providing an effective pruning rule.
//
// Whenever the candidate buffer is full (contains k elements), the warp merges
// intermediate and candidate neighbors by running a bitonic sort over the 2*k
// elements and keeping the first k. The merge is done cooperatively using all
// 32 threads of the warp and shared memory. After processing all data points,
// any remaining candidates are padded to size k and a final merge is performed.
//
// This implementation assumes k is a power of two between 32 and 1024 inclusive,
// and data_count >= k.
//
// The interface matches:
//   void run_knn(const float2 *query, int query_count,
//                const float2 *data,  int data_count,
//                std::pair<int, float> *result, int k);

#include <cuda_runtime.h>
#include <utility>
#include <limits>

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;    // 4 warps (128 threads) per block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// Number of data points cached in shared memory per batch.
// 1024 gives good cache reuse while keeping shared memory usage moderate.
constexpr int TILE_POINTS = 1024;

// Compute squared Euclidean distance between two 2D points.
// Squared distance is sufficient for k-NN ordering.
__device__ __forceinline__ float squared_distance_2d(const float2 a, const float2 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Warp-level bitonic sort over 2*k neighbors stored in shared memory.
//
// Input:
//   dists[0 .. 2*k-1]   - distances
//   indices[0 .. 2*k-1] - corresponding data indices
//
// The function sorts (dists, indices) in ascending order of distance.
// After completion, the first k elements are the k nearest neighbors among
// the initial 2*k, and the remainder are the worst elements.
//
// All 32 threads in the warp cooperate: each thread handles indices
// i = lane, lane + 32, lane + 64, ... . Synchronization is done with
// __syncwarp since all operations are warp-synchronous.
__device__ __forceinline__ void warp_sort_2k_neighbors(float *dists,
                                                       int   *indices,
                                                       int    k)
{
    const int total    = 2 * k;                 // total elements to sort
    const int lane     = threadIdx.x & (WARP_SIZE - 1);
    const unsigned mask = 0xFFFFFFFFu;

    // Classic bitonic sort network.
    // "size" is the size of the subsequences being merged.
    for (int size = 2; size <= total; size <<= 1) {
        // "stride" is the distance between elements being compared/swapped.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes multiple indices separated by warp size.
            for (int i = lane; i < total; i += WARP_SIZE) {
                int j = i ^ stride;
                // Process each pair only once (i < j equivalent; here j > i).
                if (j > i) {
                    // Determine whether this pair is in ascending or descending
                    // part of the bitonic sequence.
                    bool ascending = ((i & size) == 0);

                    float di = dists[i];
                    float dj = dists[j];
                    int   ii = indices[i];
                    int   ij = indices[j];

                    // If order is wrong (depending on ascending/descending),
                    // swap the elements.
                    if ((di > dj) == ascending) {
                        dists[i]   = dj;
                        dists[j]   = di;
                        indices[i] = ij;
                        indices[j] = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Kernel computing k-NN for 2D points.
// Each warp handles a single query point.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int                 query_count,
                           const float2 * __restrict__ data,
                           int                 data_count,
                           std::pair<int, float> * __restrict__ result,
                           int                 k)
{
    extern __shared__ unsigned char smem[];

    // Shared memory layout:
    //   - First TILE_POINTS float2's: cached data points for the current tile.
    //   - Then, for each warp in the block:
    //       * 2*k floats  : distances (first k = intermediate, next k = candidates)
    //       * 2*k ints    : indices  (first k = intermediate, next k = candidates)
    float2 *tile = reinterpret_cast<float2 *>(smem);
    const size_t tile_bytes = static_cast<size_t>(TILE_POINTS) * sizeof(float2);

    int warp_in_block = threadIdx.x / WARP_SIZE;           // warp index within block
    int lane          = threadIdx.x & (WARP_SIZE - 1);     // lane index within warp
    int global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    bool has_query    = (global_warp < query_count);
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Pointer to this warp's private neighbor buffers in shared memory.
    const size_t per_warp_bytes =
        static_cast<size_t>(2 * k) * sizeof(float) +
        static_cast<size_t>(2 * k) * sizeof(int);

    unsigned char *warp_base = smem + tile_bytes + warp_in_block * per_warp_bytes;
    float *warp_dists   = reinterpret_cast<float *>(warp_base);
    int   *warp_indices = reinterpret_cast<int   *>(warp_dists + 2 * k);

    const float INF = CUDART_INF_F;

    // Initialize both intermediate (first k) and candidate (second k) buffers
    // to "empty" values for warps that own a query.
    if (has_query) {
        for (int i = lane; i < 2 * k; i += WARP_SIZE) {
            warp_dists[i]   = INF;
            warp_indices[i] = -1;
        }
    }

    // Ensure all shared initialization completes before first use of the tile.
    __syncthreads();

    // Load query point for this warp if it is valid.
    float2 query_point;
    if (has_query) {
        query_point = query[global_warp];
    }

    // Per-query (per-warp) candidate count and worst distance among the
    // current k nearest neighbors in the intermediate result.
    int   candidate_count = 0;
    float worst_dist      = INF;  // current k-th best distance in intermediate

    // Process the dataset in tiles cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_POINTS) tile_size = TILE_POINTS;

        // Load tile of data points into shared memory cooperatively.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            tile[idx] = data[tile_start + idx];
        }
        __syncthreads();  // ensure tile is fully populated

        if (has_query) {
            int tile_offset = 0;

            // Process points in this tile, possibly in several chunks to ensure
            // that the candidate buffer never overflows.
            while (tile_offset < tile_size) {
                // If buffer is full from previous work, merge first.
                if (candidate_count == k) {
                    warp_sort_2k_neighbors(warp_dists, warp_indices, k);
                    candidate_count = 0;
                    worst_dist      = warp_dists[k - 1];  // new pruning threshold
                }

                int remaining_in_tile = tile_size - tile_offset;
                int buffer_capacity   = k - candidate_count;

                if (buffer_capacity <= 0) {
                    // Defensive: should not be needed due to the check above.
                    warp_sort_2k_neighbors(warp_dists, warp_indices, k);
                    candidate_count = 0;
                    worst_dist      = warp_dists[k - 1];
                    buffer_capacity = k;
                }

                // To avoid overflowing the candidate buffer even in the worst case
                // where every processed point is accepted, we restrict this chunk
                // to at most buffer_capacity points.
                int chunk_size = remaining_in_tile;
                if (chunk_size > buffer_capacity) chunk_size = buffer_capacity;

                int iterations = (chunk_size + WARP_SIZE - 1) / WARP_SIZE;

                for (int it = 0; it < iterations; ++it) {
                    int idx_in_chunk = it * WARP_SIZE + lane;
                    bool valid       = (idx_in_chunk < chunk_size);

                    float dist       = 0.0f;
                    int   data_index = -1;
                    bool  accept     = false;

                    if (valid) {
                        int tile_index = tile_offset + idx_in_chunk;
                        float2 d       = tile[tile_index];

                        dist       = squared_distance_2d(query_point, d);
                        data_index = tile_start + tile_index;

                        // Apply pruning based on the current k-th neighbor in the
                        // intermediate result. This is safe: any point with distance
                        // >= current worst_dist can never be among the final k nearest
                        // neighbors because worst_dist monotonically decreases.
                        accept = (dist < worst_dist);
                    }

                    // Warp-synchronous compaction of accepted candidates into
                    // the candidate half of the warp's shared-memory buffer.
                    unsigned accept_mask = __ballot_sync(FULL_MASK, accept);
                    int num_accept       = __popc(accept_mask);

                    if (num_accept) {
                        int base_pos = 0;
                        if (lane == 0) {
                            base_pos        = candidate_count;
                            candidate_count = candidate_count + num_accept;
                        }
                        base_pos        = __shfl_sync(FULL_MASK, base_pos, 0);
                        candidate_count = __shfl_sync(FULL_MASK, candidate_count, 0);

                        if (accept) {
                            // Offset of this lane's element among accepted in this loop iteration.
                            unsigned lane_mask = accept_mask & ((1u << lane) - 1u);
                            int offset         = __popc(lane_mask);
                            int pos            = base_pos + offset;  // 0 <= pos < candidate_count <= k

                            // Store into candidate buffer at index k + pos.
                            warp_dists[k + pos]   = dist;
                            warp_indices[k + pos] = data_index;
                        }
                    }
                }

                tile_offset += chunk_size;

                // When the candidate buffer is full, merge it with the intermediate
                // result using a bitonic sort over all 2*k elements.
                if (candidate_count == k) {
                    warp_sort_2k_neighbors(warp_dists, warp_indices, k);
                    candidate_count = 0;
                    worst_dist      = warp_dists[k - 1];
                }
            }
        }

        __syncthreads();  // ensure all warps are done using this tile
    }

    // After processing all tiles, if the candidate buffer is non-empty, pad it
    // with +inf to size k and perform one last merge.
    if (has_query && candidate_count > 0) {
        // Fill unused candidate slots with INF and invalid index.
        for (int i = lane + candidate_count; i < k; i += WARP_SIZE) {
            warp_dists[k + i]   = INF;
            warp_indices[k + i] = -1;
        }
        __syncwarp(FULL_MASK);

        // Now the candidate half contains exactly k elements (real + padded).
        warp_sort_2k_neighbors(warp_dists, warp_indices, k);
        candidate_count = 0;
        worst_dist      = warp_dists[k - 1];
    }

    // Store the final k nearest neighbors for this query.
    if (has_query) {
        for (int i = lane; i < k; i += WARP_SIZE) {
            int out_idx = global_warp * k + i;
            result[out_idx].first  = warp_indices[i];
            result[out_idx].second = warp_dists[i];
        }
    }
}

// Host-side wrapper that configures and launches the knn_kernel.
// Assumes all pointers (query, data, result) are already allocated on the device
// via cudaMalloc, and that data_count >= k.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Each warp handles one query.
    int num_warps  = query_count;
    int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Compute required dynamic shared memory size:
    //   - TILE_POINTS float2's for the current tile
    //   - WARPS_PER_BLOCK * (2*k floats + 2*k ints) for per-warp neighbor buffers
    size_t per_warp_bytes =
        static_cast<size_t>(2 * k) * sizeof(float) +
        static_cast<size_t>(2 * k) * sizeof(int);
    size_t shared_bytes =
        static_cast<size_t>(TILE_POINTS) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_bytes;

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(num_blocks);

    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(query,
                                                      query_count,
                                                      data,
                                                      data_count,
                                                      result,
                                                      k);

    // The caller can add error checking / synchronization as needed.
}