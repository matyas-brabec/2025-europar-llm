#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// This implementation assumes compilation with NVCC and execution on a modern
// NVIDIA data-center GPU (A100/H100 class).  It uses one warp (32 threads) per
// query and processes the data points in shared-memory tiles.
//
// The algorithm maintains, per query/warp:
//   - An intermediate result of k nearest neighbors stored in registers,
//     distributed so that each thread keeps k/32 consecutive neighbors.
//   - A candidate buffer of size k in shared memory (indices + distances).
//   - A candidate counter and a max_distance (distance of k-th neighbor).
//
// Whenever the candidate buffer would overflow, it is merged into the
// intermediate result using the specified bitonic-sort-based procedure.

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;     // 8 warps per block => 256 threads
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int DATA_TILE_SIZE    = 1024;  // Number of data points cached per block

// Maximum supported k (power of two between 32 and 1024 inclusive)
constexpr int MAX_K             = 1024;
constexpr int MAX_PER_THREAD    = MAX_K / WARP_SIZE; // 32

// Layout-compatible helper for std::pair<int,float> on device.
struct Pair {
    int   first;
    float second;
};

// -------------------------------------------------------------------------------------------------
// Bitonic sort on k = WARP_SIZE * per_thread elements distributed across a warp.
//
// Each thread owns 'per_thread' consecutive elements:
//
//   global index i in [0, k):
//     lane_id = i / per_thread
//     local   = i % per_thread
//
// Thus each thread stores:
//     reg_dists[0 .. per_thread-1]
//     reg_indices[0 .. per_thread-1]
//
// The algorithm is a standard bitonic sorting network adapted to this layout:
//
//   - For j < per_thread, comparisons are entirely within a thread, and we
//     perform compare-and-swap between reg_dists[local] and reg_dists[local ^ j].
//   - For j >= per_thread, comparisons are between threads.  Because per_thread
//     is a power of two and k is a power of two, these comparisons always pair
//     elements with the same local index 't' but different lanes.  We use
//     __shfl_xor_sync to exchange values between lanes.
//
// Direction (ascending/descending) at each comparator is determined by the
// standard condition (i & stage_k) == 0, where 'i' is the global index.
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__
void bitonic_sort_warp(float *reg_dists,
                       int   *reg_indices,
                       int    per_thread,
                       int    k,
                       int    lane_id)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Outer loop over the size of subsequences being merged (stage_k).
    for (int stage_k = 2; stage_k <= k; stage_k <<= 1) {
        // Inner loop over the distance to comparison partner (j).
        for (int j = stage_k >> 1; j > 0; j >>= 1) {

            if (j < per_thread) {
                // Comparisons within a thread: partner index differs only in the
                // low bits representing the local index ('t').
                for (int t = 0; t < per_thread; ++t) {
                    int global_i  = lane_id * per_thread + t;
                    int partner_t = t ^ j;  // Partner within this thread

                    if (partner_t > t) {
                        float self      = reg_dists[t];
                        float other     = reg_dists[partner_t];
                        int   self_idx  = reg_indices[t];
                        int   other_idx = reg_indices[partner_t];

                        bool up   = ((global_i & stage_k) == 0);
                        bool swap = ((self > other) == up);

                        if (swap) {
                            reg_dists[t]         = other;
                            reg_dists[partner_t] = self;
                            reg_indices[t]       = other_idx;
                            reg_indices[partner_t] = self_idx;
                        }
                    }
                }
            } else {
                // Comparisons between threads: partner has the same local index
                // 't' in lane (lane_id ^ lane_mask), where lane_mask = j / per_thread.
                int lane_mask = j / per_thread;

                for (int t = 0; t < per_thread; ++t) {
                    int global_i = lane_id * per_thread + t;

                    float self      = reg_dists[t];
                    int   self_idx  = reg_indices[t];

                    float other     = __shfl_xor_sync(FULL_MASK, self,     lane_mask);
                    int   other_idx = __shfl_xor_sync(FULL_MASK, self_idx, lane_mask);

                    bool up   = ((global_i & stage_k) == 0);
                    bool swap = ((self > other) == up);

                    float selected_val = swap ? other     : self;
                    int   selected_idx = swap ? other_idx : self_idx;

                    reg_dists[t]   = selected_val;
                    reg_indices[t] = selected_idx;
                }
            }

            // Ensure all compare-and-swap operations for this 'j' are visible
            // before proceeding to the next step in the sorting network.
            __syncwarp();
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Flush the candidate buffer for a warp and merge it into the intermediate result.
//
// Inputs / state:
//   - reg_dists / reg_indices: current intermediate result in registers,
//       sorted in ascending order (k elements distributed across the warp).
//   - warp_cand_dists / warp_cand_indices: candidate buffer in shared memory,
//       holding 'candidate_count' valid entries in [0, candidate_count).
//   - k, per_thread, lane_id
//   - max_distance: distance of the current k-th nearest neighbor (broadcast
//       across the warp).
//
// Procedure (per the problem statement):
//   0. Intermediate result (reg_*) is sorted ascending.
//   1. Fill the unused part of the buffer with +inf and swap reg_* with buffer
//      so that buffer elements move into registers and the previous intermediate
//      result is stored in shared memory.
//   2. Sort the buffer (now in registers) using bitonic sort.
//   3. Merge buffer and intermediate result into the registers by taking, for
//      each i, the minimum of buffer[i] and intermediate[k - 1 - i].  This
//      produces a bitonic sequence containing the smallest k elements.
//   4. Sort the merged bitonic sequence using bitonic sort to restore full
//      ascending order.
//   5. Update max_distance to the distance of the k-th neighbor.
//   6. Reset candidate_count to zero.
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__
void warp_flush_candidates(float *reg_dists,
                           int   *reg_indices,
                           float *warp_cand_dists,
                           int   *warp_cand_indices,
                           int    k,
                           int    per_thread,
                           int    lane_id,
                           int   &candidate_count,
                           float &max_distance)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    if (candidate_count <= 0) {
        return;
    }

    // 1. Fill remaining candidate slots with +inf, so buffer always has k elements.
    for (int idx = lane_id; idx < k; idx += WARP_SIZE) {
        if (idx >= candidate_count) {
            warp_cand_dists[idx]   = CUDART_INF_F;
            warp_cand_indices[idx] = -1;
        }
    }
    __syncwarp();

    // 2. Swap intermediate result (in registers) with buffer (in shared memory).
    //    After this, reg_* hold the buffer, and shared memory holds the previous
    //    intermediate result.
    for (int t = 0; t < per_thread; ++t) {
        int global_i = lane_id * per_thread + t;

        float buf_d = warp_cand_dists[global_i];
        int   buf_i = warp_cand_indices[global_i];

        warp_cand_dists[global_i]   = reg_dists[t];
        warp_cand_indices[global_i] = reg_indices[t];

        reg_dists[t]   = buf_d;
        reg_indices[t] = buf_i;
    }
    __syncwarp();

    // 3. Sort the buffer (now in registers) in ascending order.
    bitonic_sort_warp(reg_dists, reg_indices, per_thread, k, lane_id);
    __syncwarp();

    // 4. Merge the sorted buffer and the previous intermediate result into a
    //    bitonic sequence by taking the minimum between:
    //        buffer[i]        (now in reg_*)
    //        intermediate[k-1-i] (now in warp_cand_*)
    for (int t = 0; t < per_thread; ++t) {
        int global_i  = lane_id * per_thread + t;
        int other_pos = k - 1 - global_i;

        float from_buf      = reg_dists[t];
        int   from_buf_idx  = reg_indices[t];

        float from_inter    = warp_cand_dists[other_pos];
        int   from_inter_idx= warp_cand_indices[other_pos];

        if (from_inter < from_buf) {
            from_buf     = from_inter;
            from_buf_idx = from_inter_idx;
        }

        reg_dists[t]   = from_buf;
        reg_indices[t] = from_buf_idx;
    }
    __syncwarp();

    // 5. Sort the merged bitonic sequence in ascending order to obtain the
    //    updated intermediate result.
    bitonic_sort_warp(reg_dists, reg_indices, per_thread, k, lane_id);
    __syncwarp();

    // 6. Update max_distance to the distance of the k-th nearest neighbor,
    //    i.e., the last element (global index k-1), which resides in lane
    //    WARP_SIZE-1 at local index per_thread-1.
    float kth = 0.0f;
    if (lane_id == WARP_SIZE - 1) {
        kth = reg_dists[per_thread - 1];
    }
    kth = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
    max_distance = kth;

    // 7. Reset candidate count for this warp.
    candidate_count = 0;
    __syncwarp();
}

// -------------------------------------------------------------------------------------------------
// CUDA kernel: one warp per query.
//
// Each block processes WARPS_PER_BLOCK queries (warps), and all threads in the
// block cooperate to load shared-memory tiles of DATA_TILE_SIZE points from the
// 'data' array.  Each warp then uses the cached points to compute distances to
// its own query and updates its per-warp candidate buffer and intermediate
// result as described above.
// -------------------------------------------------------------------------------------------------
__global__
void knn_kernel(const float2 * __restrict__ query,
                int                    query_count,
                const float2 * __restrict__ data,
                int                    data_count,
                Pair          * __restrict__ result,
                int                    k)
{
    extern __shared__ unsigned char smem[];

    // Shared memory layout:
    //   [0, DATA_TILE_SIZE)                     : float2 tile of data points
    //   [DATA_TILE_SIZE, ... )                  : candidate distances for all warps
    //   [.. after distances ..)                 : candidate indices for all warps
    //   [.. after indices ..)                   : candidate counts (one per warp)
    float2 *shared_data = reinterpret_cast<float2*>(smem);
    float  *shared_cand_dists   = reinterpret_cast<float*>(shared_data + DATA_TILE_SIZE);
    int    *shared_cand_indices = reinterpret_cast<int*>(shared_cand_dists + WARPS_PER_BLOCK * k);
    int    *shared_cand_counts  = reinterpret_cast<int*>(shared_cand_indices + WARPS_PER_BLOCK * k);

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);

    // Global warp index -> query index
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (warp_global >= query_count) {
        return;
    }

    // Pointer to this warp's candidate buffer and count
    float *warp_cand_dists   = shared_cand_dists   + warp_id * k;
    int   *warp_cand_indices = shared_cand_indices + warp_id * k;
    int   *warp_cand_count   = shared_cand_counts  + warp_id;

    // Each thread stores k / WARP_SIZE consecutive neighbors in registers.
    const int per_thread = k / WARP_SIZE;

    // Per-thread registers: intermediate result (k nearest neighbors)
    float reg_dists[MAX_PER_THREAD];
    int   reg_indices[MAX_PER_THREAD];

    // Initialize intermediate result to +inf distance and invalid indices.
    for (int i = 0; i < per_thread; ++i) {
        reg_dists[i]   = CUDART_INF_F;
        reg_indices[i] = -1;
    }

    // Initialize candidate count for this warp.
    if (lane_id == 0) {
        *warp_cand_count = 0;
    }
    __syncwarp();

    // Load this warp's query point and broadcast to all lanes.
    float2 q;
    if (lane_id == 0) {
        q = query[warp_global];
    }
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // max_distance is the distance of the current k-th nearest neighbor.
    // Initialize to +inf so that all points are initially candidates.
    float max_distance = CUDART_INF_F;

    // Process data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > DATA_TILE_SIZE) {
            tile_size = DATA_TILE_SIZE;
        }

        // Load tile into shared memory cooperatively across the whole block.
        for (int idx = tid; idx < tile_size; idx += blockDim.x) {
            shared_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // Each warp processes the shared tile for its own query.
        for (int j = lane_id; j < tile_size; j += WARP_SIZE) {
            float2 p = shared_data[j];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;
            int   data_index = tile_start + j;

            bool done = false;
            while (!done) {
                // Determine whether this point is a candidate given current max_distance.
                float current_max = max_distance;
                int is_candidate  = (dist < current_max) ? 1 : 0;

                // Count how many lanes found a candidate in this iteration.
                unsigned mask     = __ballot_sync(FULL_MASK, is_candidate);
                int warp_add      = __popc(mask);

                // Load current candidate count for this warp (broadcast from lane 0).
                int cand_count = 0;
                if (lane_id == 0) {
                    cand_count = *warp_cand_count;
                }
                cand_count = __shfl_sync(FULL_MASK, cand_count, 0);

                // Decide whether we need to flush the candidate buffer.
                int need_flush = (cand_count + warp_add > k) ? 1 : 0;
                need_flush     = __shfl_sync(FULL_MASK, need_flush, 0);

                if (need_flush) {
                    // Flush the candidate buffer and recompute candidacy for this point.
                    if (cand_count > 0) {
                        warp_flush_candidates(reg_dists,
                                              reg_indices,
                                              warp_cand_dists,
                                              warp_cand_indices,
                                              k,
                                              per_thread,
                                              lane_id,
                                              cand_count,
                                              max_distance);
                        // Store updated candidate count (reset to zero) back to shared memory.
                        if (lane_id == 0) {
                            *warp_cand_count = cand_count;
                        }
                        __syncwarp();
                    } else {
                        // Defensive break; should not happen as need_flush implies cand_count>0
                        done = true;
                    }
                    // Loop again with updated max_distance.
                } else {
                    // We can safely insert all candidates from this iteration into the buffer.
                    if (warp_add > 0) {
                        int base = 0;
                        if (lane_id == 0) {
                            base = cand_count;
                            *warp_cand_count = cand_count + warp_add;
                        }
                        base = __shfl_sync(FULL_MASK, base, 0);

                        if (is_candidate) {
                            unsigned lane_mask = (1u << lane_id) - 1u;
                            int offset = __popc(mask & lane_mask);
                            int pos    = base + offset;

                            warp_cand_dists[pos]   = dist;
                            warp_cand_indices[pos] = data_index;
                        }
                    }
                    done = true;
                }
            } // while (!done)
        } // for j

        // Ensure all warps are done with this tile before loading the next one.
        __syncthreads();
    } // for tile_start

    // After processing all tiles, flush any remaining candidates for this warp.
    int final_cand_count = 0;
    if (lane_id == 0) {
        final_cand_count = *warp_cand_count;
    }
    final_cand_count = __shfl_sync(FULL_MASK, final_cand_count, 0);

    if (final_cand_count > 0) {
        warp_flush_candidates(reg_dists,
                              reg_indices,
                              warp_cand_dists,
                              warp_cand_indices,
                              k,
                              per_thread,
                              lane_id,
                              final_cand_count,
                              max_distance);
        if (lane_id == 0) {
            *warp_cand_count = final_cand_count; // now zero
        }
        __syncwarp();
    }

    // Write final k nearest neighbors for this query to global memory.
    // Each thread writes its per_thread consecutive neighbors.
    Pair *warp_result = result + warp_global * k;

    for (int t = 0; t < per_thread; ++t) {
        int global_pos = lane_id * per_thread + t;
        if (global_pos < k) {
            warp_result[global_pos].first  = reg_indices[t];
            warp_result[global_pos].second = reg_dists[t];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Host interface
// -------------------------------------------------------------------------------------------------
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    // Grid configuration: one warp per query, WARPS_PER_BLOCK warps per block.
    int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Dynamic shared memory size:
    //   DATA_TILE_SIZE * sizeof(float2)           : shared data tile
    //   WARPS_PER_BLOCK * k * sizeof(float)       : candidate distances
    //   WARPS_PER_BLOCK * k * sizeof(int)         : candidate indices
    //   WARPS_PER_BLOCK * sizeof(int)             : candidate counts
    size_t shared_bytes =
        static_cast<size_t>(DATA_TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float) +
        static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int) +
        static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);

    // Reinterpret std::pair<int,float>* as Pair*; std::pair is layout-compatible
    // for the purpose of plain data storage.
    Pair *device_result = reinterpret_cast<Pair*>(result);

    knn_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        query, query_count, data, data_count, device_result, k);

    // Optionally synchronize here to ensure completion before returning.
    cudaDeviceSynchronize();
}