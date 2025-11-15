// Optimized CUDA implementation of k-NN for 2D points.
// Each query point is processed by a single warp (32 threads).
// Data points are processed in shared-memory tiles. Within each tile,
// all warps in the block reuse the cached data to compute distances.
// For each query, the k nearest neighbors are maintained in a warp-private
// structure distributed across the 32 lanes, and updated cooperatively
// using warp shuffles.
//
// This implementation assumes:
//   * Target GPUs: modern datacenter GPUs (e.g., A100/H100).
//   * CUDA Toolkit: recent version (supporting __shfl_sync, etc.).
//   * k is a power of two between 32 and 1024 (inclusive).
//   * data_count >= k.
//   * All pointers are device pointers allocated via cudaMalloc.

#include <cuda_runtime.h>
#include <utility>

// Use a simple struct with the same layout as std::pair<int,float>.
// This avoids having to rely on any library device annotations.
struct PairIF {
    int   first;
    float second;
};

static constexpr int WARP_SIZE = 32;

// Warp-cooperative insertion of a candidate (distance, index) into a
// distributed sorted array of size K. The array is stored across lanes:
//
//   - K is divisible by WARP_SIZE.
//   - Each lane owns K_PER_LANE = K / WARP_SIZE consecutive elements
//     in the global sorted array.
//   - best_dist[i] and best_idx[i] (per lane) store that lane's segment.
//   - The global array is sorted in ascending order of distance.
//
// This routine assumes caller has already checked that cand_dist is
// strictly better (smaller) than the current worst element, so the
// candidate must be inserted and the global worst dropped.
//
// All threads in the warp must call this function with the same
// cand_dist and cand_idx.
template<int K>
__device__ __forceinline__
void warp_insert_candidate(float cand_dist,
                           int   cand_idx,
                           float (&best_dist)[K / WARP_SIZE],
                           int   (&best_idx)[K / WARP_SIZE])
{
    constexpr int K_PER_LANE = K / WARP_SIZE;
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: Compute, in parallel, how many elements in the global
    // sorted array are strictly smaller than cand_dist.
    //
    // Each lane counts in its own segment, then we perform a warp
    // reduction to get the global count.
    int local_count = 0;
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        local_count += (best_dist[i] < cand_dist) ? 1 : 0;
    }

    int warp_sum = local_count;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(FULL_MASK, warp_sum, offset);
        warp_sum += other;
    }
    // Now lane 0 holds the sum; broadcast to all lanes.
    int total_less = __shfl_sync(FULL_MASK, warp_sum, 0);

    // Insertion position in the global array.
    const int pos       = total_less;                // 0 .. K-1
    const int pos_lane  = pos / K_PER_LANE;          // lane owning position
    const int pos_off   = pos - pos_lane * K_PER_LANE;

    // Step 2: Build a new distributed array with the candidate inserted.
    // We compute new values from the old array without overwriting the
    // old values before all lanes are done (to avoid read-after-write
    // hazards across lanes).
    float new_dist[K_PER_LANE];
    int   new_idx[K_PER_LANE];

    if (lane < pos_lane) {
        // Lanes before the insertion lane: no elements change.
#pragma unroll
        for (int i = 0; i < K_PER_LANE; ++i) {
            new_dist[i] = best_dist[i];
            new_idx[i]  = best_idx[i];
        }
    } else if (lane == pos_lane) {
        // Lane containing the insertion position.
#pragma unroll
        for (int i = 0; i < K_PER_LANE; ++i) {
            if (i < pos_off) {
                // Before insertion point: unchanged.
                new_dist[i] = best_dist[i];
                new_idx[i]  = best_idx[i];
            } else if (i == pos_off) {
                // Insert candidate.
                new_dist[i] = cand_dist;
                new_idx[i]  = cand_idx;
            } else {
                // After insertion point: shift old elements right by 1.
                new_dist[i] = best_dist[i - 1];
                new_idx[i]  = best_idx[i - 1];
            }
        }
    } else {
        // Lanes after the insertion lane: entire segment shifts right by 1.
        // new[0] takes the last element of the previous lane; the rest
        // shift within the same lane.
        float prev_last_dist = __shfl_sync(FULL_MASK, best_dist[K_PER_LANE - 1], lane - 1);
        int   prev_last_idx  = __shfl_sync(FULL_MASK, best_idx[K_PER_LANE - 1], lane - 1);

        new_dist[0] = prev_last_dist;
        new_idx[0]  = prev_last_idx;

#pragma unroll
        for (int i = 1; i < K_PER_LANE; ++i) {
            new_dist[i] = best_dist[i - 1];
            new_idx[i]  = best_idx[i - 1];
        }
    }

    // Step 3: Commit the new values back to the distributed array.
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        best_dist[i] = new_dist[i];
        best_idx[i]  = new_idx[i];
    }
}

// Main kernel:
//   - Each warp processes one query point.
//   - The block cooperatively loads tiles of data points into shared memory.
//   - Each active warp scans the data tiles, computing distances and
//     updating its warp-private top-k list using warp_insert_candidate.
template<int K, int TILE_SIZE>
__global__ void knn_kernel(const float2 *__restrict__ query,
                           int                   query_count,
                           const float2 *__restrict__ data,
                           int                   data_count,
                           PairIF *__restrict__  result)
{
    extern __shared__ float2 smem_data[];  // shared-memory tile of data points

    constexpr int K_PER_LANE = K / WARP_SIZE;
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    const int thread_id      = threadIdx.x;
    const int warp_in_block  = thread_id / WARP_SIZE;
    const int lane           = thread_id & (WARP_SIZE - 1);
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int query_idx   = global_warp;

    // Determine whether this warp is assigned a valid query.
    const bool warp_active = (query_idx < query_count);

    // Load query point for this warp (lane 0), then broadcast via shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active && lane == 0) {
        float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Warp-private top-k storage, distributed across lanes:
    // each lane owns K_PER_LANE entries.
    float best_dist[K_PER_LANE];
    int   best_idx[K_PER_LANE];

    // Initialize with +infinity and invalid indices.
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        best_dist[i] = CUDART_INF_F;
        best_idx[i]  = -1;
    }

    // Process data in tiles loaded into shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Load this tile into shared memory. All threads in the block
        // participate in loading to maximize bandwidth.
        for (int idx = thread_id; idx < tile_size; idx += blockDim.x) {
            smem_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // Each active warp processes the tile, computing distances
        // and updating its top-k list.
        if (warp_active) {
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                const int idx_in_tile = base + lane;

                float cand_dist = CUDART_INF_F;
                int   cand_idx  = -1;

                // Each lane computes distance to one candidate point (if valid).
                if (idx_in_tile < tile_size) {
                    float2 p = smem_data[idx_in_tile];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    cand_dist = dx * dx + dy * dy;
                    cand_idx  = tile_start + idx_in_tile;
                }

                // Build a mask of lanes whose candidate is valid.
                unsigned valid_mask = __ballot_sync(FULL_MASK, idx_in_tile < tile_size);

                // Process candidates one by one. For each candidate, all lanes
                // cooperate to insert it into the distributed top-k structure.
                while (valid_mask) {
                    int src_lane = __ffs(valid_mask) - 1;  // lane id of next candidate

                    float d = __shfl_sync(FULL_MASK, cand_dist, src_lane);
                    int   i = __shfl_sync(FULL_MASK, cand_idx,  src_lane);

                    // Check against current worst distance; skip if not better.
                    float current_worst = __shfl_sync(
                        FULL_MASK,
                        best_dist[K_PER_LANE - 1],
                        WARP_SIZE - 1
                    );

                    if (d < current_worst) {
                        warp_insert_candidate<K>(d, i, best_dist, best_idx);
                    }

                    // Clear the processed lane from the mask.
                    valid_mask &= valid_mask - 1;
                }
            }
        }

        __syncthreads();
    }

    // Write out the final top-k neighbors for this query.
    // The global array is stored in ascending order, distributed across lanes.
    if (warp_active) {
        const int base_out = query_idx * K;
        const int g_start  = (threadIdx.x & (WARP_SIZE - 1)) * K_PER_LANE;

#pragma unroll
        for (int i = 0; i < K_PER_LANE; ++i) {
            const int g_idx = g_start + i;  // global offset within k
            result[base_out + g_idx].first  = best_idx[i];
            result[base_out + g_idx].second = best_dist[i];
        }
    }
}

// Host entry point: dispatches to the appropriate kernel specialization
// based on k. k is assumed to be a power of two in [32, 1024].
void run_knn(const float2 *query,
             int            query_count,
             const float2 *data,
             int            data_count,
             std::pair<int, float> *result,
             int            k)
{
    // Tunable parameters:
    //   - WARPS_PER_BLOCK determines how many queries each block processes.
    //   - TILE_SIZE controls how many data points are cached in shared memory
    //     at once. 4096 points => 32 KiB of shared memory per block.
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TILE_SIZE = 4096;

    const int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    size_t shared_mem_bytes = TILE_SIZE * sizeof(float2);

    PairIF *result_raw = reinterpret_cast<PairIF *>(result);

    switch (k) {
        case 32:
            knn_kernel<32, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        case 64:
            knn_kernel<64, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        case 128:
            knn_kernel<128, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        case 256:
            knn_kernel<256, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        case 512:
            knn_kernel<512, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        case 1024:
            knn_kernel<1024, TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
                query, query_count, data, data_count, result_raw);
            break;
        default:
            // k is guaranteed to be a power of two between 32 and 1024.
            // No fallback is implemented for invalid k.
            break;
    }

    // The kernel is launched asynchronously; callers can synchronize
    // or check for errors as needed.
}