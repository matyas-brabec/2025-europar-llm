#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

/*
 * Optimized CUDA implementation of k-NN for 2D points.
 *
 * Design overview (high level, see detailed comments in code body):
 *
 * - Each query is processed by a single warp (32 threads).
 * - For each query, we maintain the current k nearest neighbors (NNs)
 *   in registers, distributed across the warp:
 *        k is a power of two in [32, 1024].
 *        Each thread stores k/32 consecutive elements
 *        (indices and distances) in its registers.
 * - For each query/warp we also maintain a candidate buffer of size k
 *   in shared memory (indices + distances), plus a candidate counter.
 * - We process the data points in tiles. Each tile is cooperatively
 *   loaded into shared memory by the entire block. Then each warp
 *   computes distances from its query point to all points in the tile.
 * - For each batch of 32 distances (one per thread in the warp),
 *   we use a warp ballot to determine which of them are closer
 *   than the current max_distance (the distance of the k-th NN).
 *   Those that pass the filter are appended to the candidate buffer
 *   using warp-aggregated indexing.
 * - Whenever the candidate buffer reaches capacity k, or when it is
 *   non-empty after all data points have been processed, we merge it
 *   with the current register-resident NN list:
 *
 *   Merge procedure:
 *     0. Invariant: the register-resident NN list is sorted ascending.
 *     1. Swap the contents of the candidate buffer (shared) and the
 *        register-resident NN list so that candidates move into
 *        registers and the previous NN list moves into shared memory.
 *     2. Sort the candidate list in registers using a bitonic sort
 *        over k elements distributed across the warp.
 *     3. Merge the sorted candidate list (in registers) and the
 *        previous NN list (in shared memory) by forming a bitonic
 *        sequence: for each position i in [0, k-1], we take the
 *        minimum of candidate[i] and prev[k-1-i]. This yields a
 *        bitonic sequence containing the best k elements of the
 *        2k-element union.
 *     4. Bitonic-sort the bitonic sequence (in registers) again to
 *        restore a sorted list of length k in registers, and update
 *        max_distance to the last element.
 *
 * - Bitonic sort implementation:
 *     * k is a power of two (<= 1024).
 *     * Warp size is 32, so each thread stores k/32 elements.
 *     * Because k/32 is also a power of two, the global indices
 *       of elements (0..k-1) can be mapped to (lane, local_index)
 *       with:
 *           global = lane * elems_per_thread + local_index
 *       where elems_per_thread = k/32.
 *     * In the bitonic network, for each stage parameter j:
 *           partner = global ^ j
 *       If j < elems_per_thread, the partner lies in the same thread
 *       but at a different local_index.
 *       If j >= elems_per_thread, j is a multiple of elems_per_thread
 *       and the partner lies in a different thread but at the same
 *       local_index. In that case, we exchange values via warp
 *       shuffle (__shfl_xor_sync).
 *     * Intra-thread exchanges are implemented with simple register
 *       swaps. Inter-thread exchanges use shuffles, and both threads
 *       update their own registers in a symmetric fashion (standard
 *       warp-synchronous bitonic sort pattern).
 *
 * - The final sorted k NNs for each query are written back to the
 *   result array, which is laid out as:
 *       result[query_index * k + j] = { index_of_data, distance }.
 */

static constexpr int WARP_SIZE          = 32;
static constexpr int MAX_K              = 1024;
static constexpr int MAX_K_PER_THREAD   = MAX_K / WARP_SIZE;  // 32
static constexpr int WARPS_PER_BLOCK    = 4;                  // 4 warps -> 128 threads
static constexpr int BLOCK_SIZE         = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_SIZE          = 512;                // data points per tile

// Device-side bitonic sort over k elements distributed across a warp.
// Each thread holds elems_per_thread = k / WARP_SIZE consecutive elements
// in dist[] / idx[], which are stored in registers.
//
// - dist[0..elems_per_thread-1], idx[0..elems_per_thread-1] contain
//   the data for this thread.
// - n == k is the total number of elements in the warp.
// - elems_per_thread must be a power of two, and n a power of two.
//
// Cross-thread exchanges always swap elements with the same local index
// in different threads, allowing us to use warp shuffles on scalars.
__device__ __forceinline__
void bitonic_sort_warp(float dist[MAX_K_PER_THREAD],
                       int   idx [MAX_K_PER_THREAD],
                       int   n,
                       int   elems_per_thread)
{
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const unsigned mask = 0xffffffffu;

    // Outer loop over the bitonic network parameter 'size' (k in pseudocode).
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop over 'j' (distance between partners).
        for (int j = size >> 1; j > 0; j >>= 1) {

            if (j < elems_per_thread) {
                // Intra-thread compare-exchange:
                // All pairs (i, i^j) lie within the same thread.
                #pragma unroll
                for (int r = 0; r < MAX_K_PER_THREAD; ++r) {
                    if (r >= elems_per_thread) break;

                    int partner_r = r ^ j;
                    // Only process each pair once; choose the one with smaller local index.
                    if (partner_r > r && partner_r < elems_per_thread) {
                        int global_i = lane * elems_per_thread + r;
                        bool up = ((global_i & size) == 0);

                        float a = dist[r];
                        float b = dist[partner_r];
                        bool do_swap = ((a > b) == up);

                        if (do_swap) {
                            dist[r]          = b;
                            dist[partner_r]  = a;
                            int ia           = idx[r];
                            int ib           = idx[partner_r];
                            idx[r]          = ib;
                            idx[partner_r]  = ia;
                        }
                    }
                }
            } else {
                // Inter-thread compare-exchange:
                // j >= elems_per_thread => j is a multiple of elems_per_thread,
                // so the partner lies in a different thread but at the same
                // local index r.
                int delta_lane = j / elems_per_thread;

                #pragma unroll
                for (int r = 0; r < MAX_K_PER_THREAD; ++r) {
                    if (r >= elems_per_thread) break;

                    int global_i = lane * elems_per_thread + r;
                    bool up = ((global_i & size) == 0);

                    float my_d   = dist[r];
                    int   my_idx = idx[r];

                    float other_d = __shfl_xor_sync(mask, my_d,   delta_lane);
                    int   other_i = __shfl_xor_sync(mask, my_idx, delta_lane);

                    bool do_swap = ((my_d > other_d) == up);
                    if (do_swap) {
                        dist[r] = other_d;
                        idx[r]  = other_i;
                    }
                }
            }

            // Ensure all threads in the warp complete this j-stage
            // before moving to the next j or size.
            __syncwarp(mask);
        }
    }
}

// Merge the per-warp candidate buffer (in shared memory) into the current
// intermediate result (in registers).
//
// Parameters:
//   best_dist, best_idx    : per-thread register arrays holding the current
//                            (sorted) k nearest neighbors for this query.
//   elems_per_thread       : k / WARP_SIZE.
//   warp_id                : warp index within the block (0..WARPS_PER_BLOCK-1).
//   k                      : number of nearest neighbors (power of two).
//   valid_candidates       : number of valid entries currently stored in
//                            the candidate buffer (0 < valid_candidates <= k).
//   cand_dist_array, cand_idx_array : shared-memory 2D arrays holding the
//                            candidate buffers for all warps in the block.
//   max_distance           : (reference) distance of the current k-th neighbor;
//                            updated at the end of the merge.
//
__device__ __forceinline__
void merge_buffer_with_result(float best_dist[MAX_K_PER_THREAD],
                              int   best_idx [MAX_K_PER_THREAD],
                              int   elems_per_thread,
                              int   warp_id,
                              int   k,
                              int   valid_candidates,
                              float cand_dist_array[WARPS_PER_BLOCK][MAX_K],
                              int   cand_idx_array [WARPS_PER_BLOCK][MAX_K],
                              float &max_distance)
{
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const unsigned mask = 0xffffffffu;

    float *buf_dist = cand_dist_array[warp_id];
    int   *buf_idx  = cand_idx_array [warp_id];

    // If the buffer is not full, pad remaining entries with +INF and dummy
    // indices so that bitonic sort / merge can operate on exactly k elements.
    if (valid_candidates < k) {
        for (int i = valid_candidates + lane; i < k; i += WARP_SIZE) {
            buf_dist[i] = FLT_MAX;
            buf_idx[i]  = -1;
        }
    }
    __syncwarp(mask);

    // Step 1: swap content of buffer (shared) and intermediate result (registers).
    // After this, registers contain the k candidates, and shared memory holds
    // the previous NN list.
    #pragma unroll
    for (int r = 0; r < MAX_K_PER_THREAD; ++r) {
        if (r >= elems_per_thread) break;

        int   g        = lane * elems_per_thread + r;
        float tmp_d    = best_dist[r];
        int   tmp_i    = best_idx [r];
        float cand_d   = buf_dist[g];
        int   cand_i   = buf_idx [g];
        best_dist[r]   = cand_d;
        best_idx [r]   = cand_i;
        buf_dist[g]    = tmp_d;
        buf_idx [g]    = tmp_i;
    }
    __syncwarp(mask);

    // Step 2: sort the buffer (now in registers) in ascending order.
    bitonic_sort_warp(best_dist, best_idx, k, elems_per_thread);

    // Step 3: merge buffer (sorted in registers) with previous intermediate
    // result (sorted in shared memory) into a bitonic sequence of length k.
    // For each position g in [0, k-1], we take the minimum of:
    //   buffer[g] and prev[k - 1 - g].
    #pragma unroll
    for (int r = 0; r < MAX_K_PER_THREAD; ++r) {
        if (r >= elems_per_thread) break;

        int   g        = lane * elems_per_thread + r;
        int   partner  = k - 1 - g;
        float prev_d   = buf_dist[partner];
        int   prev_i   = buf_idx [partner];
        float cand_d   = best_dist[r];
        int   cand_i   = best_idx [r];

        if (prev_d < cand_d) {
            best_dist[r] = prev_d;
            best_idx [r] = prev_i;
        }
    }
    __syncwarp(mask);

    // Step 4: sort the resulting bitonic sequence in ascending order
    // to produce the updated intermediate result.
    bitonic_sort_warp(best_dist, best_idx, k, elems_per_thread);

    // Update max_distance (the distance of the k-th nearest neighbor).
    // The k-th element is the last one in the sorted sequence, which is
    // stored in lane WARP_SIZE-1 at local index elems_per_thread - 1.
    float kth = 0.0f;
    if (lane == WARP_SIZE - 1) {
        kth = best_dist[elems_per_thread - 1];
    }
    kth = __shfl_sync(mask, kth, WARP_SIZE - 1);
    max_distance = kth;
}

// CUDA kernel implementing k-NN for multiple queries.
// Each warp handles one query.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int                     query_count,
                           const float2 * __restrict__ data,
                           int                     data_count,
                           std::pair<int, float> * __restrict__ result,
                           int                     k)
{
    // Shared memory:
    // - s_data: tile of data points.
    // - s_cand_dist / s_cand_idx: candidate buffers for each warp in block.
    __shared__ float2 s_data[TILE_SIZE];
    __shared__ float  s_cand_dist[WARPS_PER_BLOCK][MAX_K];
    __shared__ int    s_cand_idx [WARPS_PER_BLOCK][MAX_K];

    const int thread_id       = threadIdx.x;
    const int warp_id_in_block= thread_id / WARP_SIZE;
    const int lane            = thread_id & (WARP_SIZE - 1);
    const int warp_global_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (warp_global_id >= query_count)
        return;  // Entire warp is inactive for this block.

    const unsigned full_mask = 0xffffffffu;

    // Each warp owns one row of the candidate buffers.
    float *cand_dist = s_cand_dist[warp_id_in_block];
    int   *cand_idx  = s_cand_idx [warp_id_in_block];

    // Number of elements stored per thread in the intermediate result.
    const int elems_per_thread = k / WARP_SIZE;

    // Register-resident intermediate result: k-best neighbors per query.
    float best_dist[MAX_K_PER_THREAD];
    int   best_idx [MAX_K_PER_THREAD];

    // Initialize intermediate result with +INF distances and dummy indices.
    #pragma unroll
    for (int i = 0; i < MAX_K_PER_THREAD; ++i) {
        if (i < elems_per_thread) {
            best_dist[i] = FLT_MAX;
            best_idx [i] = -1;
        }
    }

    // Load query point for this warp and broadcast to all lanes.
    float2 q;
    if (lane == 0) {
        q = query[warp_global_id];
    }
    q.x = __shfl_sync(full_mask, q.x, 0);
    q.y = __shfl_sync(full_mask, q.y, 0);

    // Current max distance (distance of the k-th nearest neighbor).
    float max_distance = FLT_MAX;

    // Candidate buffer size for this warp (kept in registers, synchronized
    // among lanes via shuffles).
    int cand_count = 0;

    // Process data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Load tile into shared memory cooperatively.
        for (int i = thread_id; i < tile_size; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes all points in the tile.
        for (int idx_in_tile = lane; idx_in_tile < tile_size; idx_in_tile += WARP_SIZE) {

            // Preemptive merge to avoid candidate buffer overflow.
            // In one iteration we can add at most WARP_SIZE candidates.
            // If k > WARP_SIZE and cand_count > k - WARP_SIZE, we merge now.
            if (k > WARP_SIZE && cand_count > k - WARP_SIZE) {
                if (cand_count > 0) {
                    merge_buffer_with_result(best_dist, best_idx,
                                             elems_per_thread,
                                             warp_id_in_block,
                                             k,
                                             cand_count,
                                             s_cand_dist,
                                             s_cand_idx,
                                             max_distance);
                }
                cand_count = 0;
            }

            // Load data point from shared memory and compute squared L2 distance.
            float2 p = s_data[idx_in_tile];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;

            float current_max = max_distance;
            bool is_candidate = (dist < current_max);

            // Warp-wide ballot to identify candidates.
            unsigned cand_mask = __ballot_sync(full_mask, is_candidate);
            int votes = __popc(cand_mask);

            if (votes > 0) {
                // Reserve [warp_offset, warp_offset + votes) slots in the
                // candidate buffer for this warp.
                int warp_offset = 0;
                if (lane == 0) {
                    warp_offset = cand_count;
                    cand_count += votes;
                }
                warp_offset = __shfl_sync(full_mask, warp_offset, 0);
                cand_count  = __shfl_sync(full_mask, cand_count, 0);

                // Each candidate thread computes its offset within the
                // new block of 'votes' elements.
                if (is_candidate) {
                    unsigned mask_before = cand_mask & ((1u << lane) - 1);
                    int local_pos        = __popc(mask_before);
                    int pos              = warp_offset + local_pos;
                    // By construction, cand_count <= k, so pos < k.
                    cand_dist[pos] = dist;
                    cand_idx [pos] = tile_start + idx_in_tile;
                }
            }

            // If the candidate buffer is exactly full, merge it with
            // the intermediate result.
            if (cand_count == k) {
                merge_buffer_with_result(best_dist, best_idx,
                                         elems_per_thread,
                                         warp_id_in_block,
                                         k,
                                         cand_count,
                                         s_cand_dist,
                                         s_cand_idx,
                                         max_distance);
                cand_count = 0;
            }
        }

        // Ensure all warps are done with this tile before reusing s_data.
        __syncthreads();
    }

    // After processing all tiles, if there are remaining candidates in the
    // buffer, merge them as well.
    if (cand_count > 0) {
        merge_buffer_with_result(best_dist, best_idx,
                                 elems_per_thread,
                                 warp_id_in_block,
                                 k,
                                 cand_count,
                                 s_cand_dist,
                                 s_cand_idx,
                                 max_distance);
        cand_count = 0;
    }

    // Write final k nearest neighbors for this query to global memory.
    // The intermediate result is sorted ascending by distance and
    // distributed across the warp as k/32 consecutive elements per thread.
    int base_out = warp_global_id * k;
    #pragma unroll
    for (int i = 0; i < MAX_K_PER_THREAD; ++i) {
        if (i >= elems_per_thread) break;

        int global_pos = lane * elems_per_thread + i;
        std::pair<int, float> &out = result[base_out + global_pos];
        out.first  = best_idx [i];
        out.second = best_dist[i];
    }
}

// Host interface: runs the k-NN CUDA kernel.
//
// query       : pointer to device array of float2 (queries), length query_count.
// data        : pointer to device array of float2 (data points), length data_count.
// result      : pointer to device array of std::pair<int,float>, length query_count * k.
// k           : number of nearest neighbors to find (power of two, in [32, 1024]).
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // One warp per query.
    dim3 block(BLOCK_SIZE);
    int num_warps = (query_count + 0); // each warp handles exactly one query
    int grid_x    = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_x);

    knn_kernel<<<grid, block>>>(query, query_count,
                                data,  data_count,
                                result, k);
}