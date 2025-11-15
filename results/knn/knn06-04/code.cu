#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// Internal representation of a k-NN entry (index, distance).
// Layout matches std::pair<int,float> for practical purposes.
struct PairIntFloat {
    int   first;   // data point index
    float second;  // squared Euclidean distance
};

// Warp-level helper to merge the candidate buffer into the current k-NN set.
// - knn:        current k-NN array [0..k-1] in shared memory
// - candidates: candidate buffer [0..cand_count-1] in shared memory
// - k:          number of neighbors
// - cand_count: number of valid entries in candidates
// - max_dist:   (in/out) distance of the current k-th nearest neighbor
// - lane_id:    lane index within the warp (0..31)
// - mask:       active-lane mask for this warp
//
// After the merge:
// - knn[0..k-1] contain the k nearest neighbors among previous knn and candidates,
//   sorted by ascending distance (second).
// - max_dist is updated to knn[k-1].second.
// - Candidates are marked as consumed (their distances set to FLT_MAX); caller should
//   reset cand_count to 0 afterwards.
static __device__ __forceinline__ void merge_candidates(
    PairIntFloat* knn,
    PairIntFloat* candidates,
    int            k,
    int            cand_count,
    float&         max_dist,
    int            lane_id,
    unsigned       mask)
{
    // Repeatedly select the smallest element among:
    //   - knn[i] for i in [out..k-1]
    //   - candidates[j] for j in [0..cand_count-1]
    // and place it at knn[out].
    for (int out = 0; out < k; ++out) {
        float best_dist  = FLT_MAX;
        int   best_index = -1;
        int   best_from  = 0;   // 0 = from knn, 1 = from candidates
        int   best_pos   = -1;  // index within the chosen array

        // Search current knn segment [out..k-1]
        for (int i = out + lane_id; i < k; i += warpSize) {
            float d = knn[i].second;
            int   idx = knn[i].first;
            if (d < best_dist) {
                best_dist  = d;
                best_index = idx;
                best_from  = 0;
                best_pos   = i;
            }
        }

        // Search candidates [0..cand_count-1]
        for (int j = lane_id; j < cand_count; j += warpSize) {
            float d = candidates[j].second;
            int   idx = candidates[j].first;
            if (d < best_dist) {
                best_dist  = d;
                best_index = idx;
                best_from  = 1;
                best_pos   = j;
            }
        }

        // Warp-wide reduction to find the global best (smallest distance).
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_dist  = __shfl_down_sync(mask, best_dist,  offset);
            int   other_index = __shfl_down_sync(mask, best_index, offset);
            int   other_from  = __shfl_down_sync(mask, best_from,  offset);
            int   other_pos   = __shfl_down_sync(mask, best_pos,   offset);

            if (other_dist < best_dist) {
                best_dist  = other_dist;
                best_index = other_index;
                best_from  = other_from;
                best_pos   = other_pos;
            }
        }

        // Lane 0 commits the selected element and marks its source as consumed.
        if (lane_id == 0) {
            knn[out].first  = best_index;
            knn[out].second = best_dist;

            if (best_from == 0) {
                // Consumed from knn; mark its original slot as inactive.
                knn[best_pos].second = FLT_MAX;
            } else {
                // Consumed from candidates.
                candidates[best_pos].second = FLT_MAX;
            }
        }

        // Ensure all threads see updated FLT_MAX markers before next iteration.
        __syncwarp(mask);
    }

    // Update max_dist to distance of the k-th (last) nearest neighbor.
    if (lane_id == 0) {
        max_dist = knn[k - 1].second;
    }
    max_dist = __shfl_sync(mask, max_dist, 0);
}

// Kernel implementing k-NN search for 2D points with one warp per query.
// Template parameters:
// - TILE_SIZE:        number of data points cached per block in shared memory.
// - WARPS_PER_BLOCK:  number of warps (queries) processed per block.
//
// Each warp maintains:
// - a private k-element intermediate result (knn) in shared memory,
// - a k-element candidate buffer in shared memory,
// - a shared candidate counter and max_distance.
//
// The block cooperatively loads tiles of data points into shared memory.
// Each active warp processes the tile for its own query, filters by max_distance,
// appends candidates using atomicAdd, and merges when the buffer becomes full.
template <int TILE_SIZE, int WARPS_PER_BLOCK>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int                       query_count,
    const float2* __restrict__ data,
    int                       data_count,
    PairIntFloat* __restrict__ result,
    int                       k)
{
    extern __shared__ unsigned char smem[];

    // Layout of dynamic shared memory:
    // [0 .. TILE_SIZE-1] float2      : cached data points
    // next: WARPS_PER_BLOCK * 2*k    : PairIntFloat (k knn + k candidates per warp)
    // next: WARPS_PER_BLOCK          : int candidate_counts
    // next: WARPS_PER_BLOCK          : float max_dists
    float2* s_points = reinterpret_cast<float2*>(smem);

    PairIntFloat* s_pairs = reinterpret_cast<PairIntFloat*>(s_points + TILE_SIZE);
    int*          s_candidate_counts = reinterpret_cast<int*>(s_pairs + WARPS_PER_BLOCK * 2 * k);
    float*        s_max_dists        = reinterpret_cast<float*>(s_candidate_counts + WARPS_PER_BLOCK);

    const int thread_id       = threadIdx.x;
    const int lane_id         = thread_id & (warpSize - 1);      // lane index in warp
    const int warp_id_in_block = thread_id >> 5;                 // warp index in block
    const int warps_per_block = WARPS_PER_BLOCK;
    const int warp_global_id  = blockIdx.x * warps_per_block + warp_id_in_block;

    const bool warp_active = (warp_global_id < query_count);

    // Active-lane mask for this warp.
    const unsigned full_mask = __activemask();

    // Per-warp pointers into shared memory for knn and candidate buffers.
    PairIntFloat* warp_base      = s_pairs + (warp_id_in_block * 2 * k);
    PairIntFloat* warp_knn       = warp_base;           // [0 .. k-1]
    PairIntFloat* warp_candidates = warp_base + k;      // [0 .. k-1], only [0..cand_count-1] used

    int&   warp_candidate_count = s_candidate_counts[warp_id_in_block];
    float& warp_max_dist        = s_max_dists[warp_id_in_block];

    // Initialize per-warp state (only for active warps).
    if (warp_active) {
        if (lane_id == 0) {
            warp_candidate_count = 0;
            warp_max_dist        = FLT_MAX;
        }
        // Initialize intermediate k-NN result with "infinite" distances.
        for (int i = lane_id; i < k; i += warpSize) {
            warp_knn[i].first  = -1;
            warp_knn[i].second = FLT_MAX;
        }
    }

    // All threads must reach this barrier because shared memory is reused across warps.
    __syncthreads();

    float2 query_point;
    if (warp_active) {
        // Load query point once (lane 0) and broadcast within warp.
        if (lane_id == 0) {
            query_point = query[warp_global_id];
        }
        query_point.x = __shfl_sync(full_mask, query_point.x, 0);
        query_point.y = __shfl_sync(full_mask, query_point.y, 0);
    }

    // Process data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (tile_start + tile_size > data_count) {
            tile_size = data_count - tile_start;
        }

        // Block-wide cooperative load of the current tile into shared memory.
        for (int i = thread_id; i < tile_size; i += blockDim.x) {
            s_points[i] = data[tile_start + i];
        }

        __syncthreads();

        if (warp_active) {
            // Local copy of max distance used for filtering within this tile.
            float max_d = warp_max_dist;

            // Strided over tile points by warp lanes.
            for (int idx_in_tile = lane_id; idx_in_tile < tile_size; idx_in_tile += warpSize) {
                float2 p  = s_points[idx_in_tile];
                float dx  = p.x - query_point.x;
                float dy  = p.y - query_point.y;
                float dist = dx * dx + dy * dy;

                // Filter by current max distance.
                bool is_candidate = (dist < max_d);

                // Determine which lanes have a candidate in this iteration.
                unsigned candidate_mask = __ballot_sync(full_mask, is_candidate);
                int local_count = __popc(candidate_mask);

                if (local_count > 0) {
                    // Check if candidate buffer has enough free slots; if not, merge.
                    int need_merge = 0;
                    int base_count = 0;

                    if (lane_id == 0) {
                        base_count = warp_candidate_count;
                        if (base_count + local_count > k) {
                            need_merge = 1;
                        }
                    }

                    base_count = __shfl_sync(full_mask, base_count, 0);
                    need_merge = __shfl_sync(full_mask, need_merge, 0);

                    if (need_merge) {
                        // Merge existing candidates into knn before inserting new ones.
                        int cand_count = 0;
                        if (lane_id == 0) {
                            cand_count = warp_candidate_count;
                        }
                        cand_count = __shfl_sync(full_mask, cand_count, 0);

                        if (cand_count > 0) {
                            merge_candidates(
                                warp_knn,
                                warp_candidates,
                                k,
                                cand_count,
                                warp_max_dist,
                                lane_id,
                                full_mask);
                            // Update local max distance after merge.
                            max_d = warp_max_dist;
                        }

                        if (lane_id == 0) {
                            warp_candidate_count = 0;
                            base_count = 0;
                        }
                        base_count = __shfl_sync(full_mask, base_count, 0);
                    }

                    // Reserve slots in candidate buffer using a single atomicAdd per warp.
                    int offset = 0;
                    if (lane_id == 0) {
                        offset = atomicAdd(&warp_candidate_count, local_count);
                    }
                    offset = __shfl_sync(full_mask, offset, 0);

                    // Each lane with a candidate writes its element to the reserved range.
                    if (is_candidate) {
                        unsigned lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
                        int pos_in_local = __popc(candidate_mask & lane_mask);
                        int pos          = offset + pos_in_local;

                        // pos is guaranteed < k due to the capacity check above.
                        warp_candidates[pos].first  = tile_start + idx_in_tile;
                        warp_candidates[pos].second = dist;
                    }
                }
            } // end loop over tile points
        }

        __syncthreads();
    } // end loop over tiles

    // Final merge for remaining candidates after processing all tiles.
    if (warp_active) {
        int cand_count = 0;
        if (lane_id == 0) {
            cand_count = warp_candidate_count;
        }
        cand_count = __shfl_sync(full_mask, cand_count, 0);

        if (cand_count > 0) {
            merge_candidates(
                warp_knn,
                warp_candidates,
                k,
                cand_count,
                warp_max_dist,
                lane_id,
                full_mask);

            if (lane_id == 0) {
                warp_candidate_count = 0;
            }
        }

        // Write final k-NN results back to global memory.
        // Each lane handles a strided subset.
        for (int i = lane_id; i < k; i += warpSize) {
            PairIntFloat p = warp_knn[i];
            result[warp_global_id * k + i].first  = p.first;
            result[warp_global_id * k + i].second = p.second;
        }
    }
}

// Host-side C++ interface as specified.
// query, data, and result are device pointers allocated with cudaMalloc.
// - query:  query_count  float2 points
// - data:   data_count   float2 points
// - result: query_count * k std::pair<int,float> entries
/// @FIXED
/// extern "C" void run_knn(
void run_knn(
    const float2*                 query,
    int                           query_count,
    const float2*                 data,
    int                           data_count,
    std::pair<int, float>*        result,
    int                           k)
{
    // Configuration parameters:
    // - TILE_SIZE: number of data points cached per block
    // - WARPS_PER_BLOCK: number of queries processed per block
    // For k up to 1024 and TILE_SIZE = 1024, WARPS_PER_BLOCK = 2 keeps shared memory
    // usage per block within ~41 KB, which is safe on modern data center GPUs.
    constexpr int TILE_SIZE        = 1024;
    constexpr int WARPS_PER_BLOCK  = 2;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    // One warp per query.
    int total_warps  = query_count;
    int num_blocks   = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Shared memory size calculation:
    // TILE_SIZE * sizeof(float2) for cached data points +
    // WARPS_PER_BLOCK * 2 * k * sizeof(PairIntFloat) for knn and candidates +
    // WARPS_PER_BLOCK * sizeof(int) for candidate counts +
    // WARPS_PER_BLOCK * sizeof(float) for max distances.
    size_t shared_bytes =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * 2 * static_cast<size_t>(k) * sizeof(PairIntFloat) +
        static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int) +
        static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(float);

    // Launch kernel. Cast result pointer to internal PairIntFloat layout.
    PairIntFloat* result_internal = reinterpret_cast<PairIntFloat*>(result);

    knn_kernel<TILE_SIZE, WARPS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK, shared_bytes>>>(
            query, query_count,
            data,  data_count,
            result_internal,
            k);

    // Synchronize to ensure completion before the caller uses the results.
    cudaDeviceSynchronize();
}
