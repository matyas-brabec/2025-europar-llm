#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include <math_constants.h>

// Optimized k-NN (k-nearest neighbors) kernel for 2D points using one warp (32 threads) per query.
// Each warp maintains the intermediate top-k result in registers, distributed so that
// each thread holds k/32 consecutive entries. A per-warp candidate buffer of size k
// (indices and distances) is maintained in shared memory. The input data is processed in
// shared-memory tiles to reduce global memory traffic. When the candidate buffer is full,
// it is merged with the intermediate result using a bitonic sort-based procedure.
//
// Important implementation details:
// - k is a power of two between 32 and 1024 (inclusive). Therefore, items_per_thread = k / 32 is also a power of two in [1, 32].
// - Warp-level bitonic sort on a distributed array of length k is implemented using warp shuffles.
//   For pairs where both elements are in the same lane, a local compare-exchange updates both.
// - The "max_distance" threshold is the current k-th (largest) distance in the intermediate result,
//   stored in the last element. It is updated after each merge. Candidates farther than this can be filtered early.
// - A per-warp candidate buffer in shared memory accumulates candidates detected via warp ballots.
//   When full, it is merged with the intermediate result following the specified 4-step procedure.
//
// Thread block organization and shared memory layout:
// - blockDim.x is a multiple of 32 (warp size). Each warp handles a different query.
// - Shared memory consists of:
//     * A common tile cache of float2 points for the entire block.
//     * Per-warp candidate buffers: indices[k] and distances[k].
//     * Per-warp candidate counts.
// - The tile size is chosen to fit within the default 48KB dynamic shared memory limit together with buffers
//   for up to 4 warps and k up to 1024. This avoids requiring a special shared memory attribute.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Maximum k and derived per-thread storage.
// The per-thread register storage arrays are statically sized to cover the maximum k.
constexpr int MAX_K = 1024;
constexpr int MAX_ITEMS_PER_THREAD = MAX_K / WARP_SIZE;

// Shared-memory tile size (number of points loaded per batch).
// 1024 points = 8KB (float2). With 4 warps and k=1024, candidate buffers need ~32KB,
// plus counts. Total ~40KB < 48KB dynamic shared memory default. Safe for A100/H100.
constexpr int TILE_POINTS = 1024;

// Utility: lane id within the warp.
static __device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

// Utility: warp id within the block.
static __device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

// Warp-synchronous barrier.
static __device__ __forceinline__ void warp_sync() {
#if __CUDACC_VER_MAJOR__ >= 9
    __syncwarp();
#endif
}

// Bitonic sort (ascending) on a warp-distributed array of length n = k.
// Each lane holds items_per_thread consecutive elements in its registers.
// The global index mapping is: global_i = lane * items_per_thread + j (j in [0, items_per_thread-1]).
// For communication across lanes, warp shuffles are used. When both elements of a pair
// in the bitonic network lie in the same lane, a local compare-exchange updates both registers.
static __device__ __forceinline__
void warp_bitonic_sort_distributed(float dist_reg[MAX_ITEMS_PER_THREAD],
                                   int   idx_reg [MAX_ITEMS_PER_THREAD],
                                   int items_per_thread, int n, int items_per_thread_log2)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();
    const int local_mask = (1 << items_per_thread_log2) - 1;

    // Outer loop: size of subsequences being merged (k in pseudocode), doubling each iteration
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop: distance between compared elements (j in pseudocode), halving each iteration
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Process each local register index j
            for (int j = 0; j < items_per_thread; ++j) {
                // Compute global indices and partner location
                const int i_global = (lane << items_per_thread_log2) + j;
                const int l_global = i_global ^ stride;

                const int lane_l = l_global >> items_per_thread_log2;
                const int j_l    = l_global & local_mask;

                const bool asc   = ((i_global & size) == 0);

                if (lane_l == lane) {
                    // Both elements are in the same lane: update both if l_global > i_global to avoid double-processing.
                    if (l_global > i_global) {
                        float a_d = dist_reg[j];
                        int   a_i = idx_reg[j];
                        float b_d = dist_reg[j_l];
                        int   b_i = idx_reg[j_l];

                        // Compute min/max with a stable-ish tie-break by index (optional)
                        bool a_lt_b = (a_d < b_d) || ((a_d == b_d) && (a_i <= b_i));
                        float minv = a_lt_b ? a_d : b_d;
                        int   mini = a_lt_b ? a_i : b_i;
                        float maxv = a_lt_b ? b_d : a_d;
                        int   maxi = a_lt_b ? b_i : a_i;

                        const bool lower = ((i_global & stride) == 0); // i_global is the lower index in the pair
                        if (asc) {
                            // ascending: lower keeps min, upper keeps max
                            dist_reg[j]   = minv; idx_reg[j]   = mini;
                            dist_reg[j_l] = maxv; idx_reg[j_l] = maxi;
                        } else {
                            // descending: lower keeps max, upper keeps min
                            dist_reg[j]   = maxv; idx_reg[j]   = maxi;
                            dist_reg[j_l] = minv; idx_reg[j_l] = mini;
                        }
                    }
                } else {
                    // Cross-lane pair: symmetric update. Each lane updates its own element.
                    float other_d = __shfl_sync(full_mask, dist_reg[j_l], lane_l);
                    int   other_i = __shfl_sync(full_mask, idx_reg [j_l], lane_l);

                    float my_d = dist_reg[j];
                    int   my_i = idx_reg [j];

                    // Compute min/max with a stable-ish tie-break by index
                    bool my_lt_other = (my_d < other_d) || ((my_d == other_d) && (my_i <= other_i));
                    float minv = my_lt_other ? my_d : other_d;
                    int   mini = my_lt_other ? my_i : other_i;
                    float maxv = my_lt_other ? other_d : my_d;
                    int   maxi = my_lt_other ? other_i : my_i;

                    const bool lower = ((i_global & stride) == 0);
                    // If ascending: lower keeps min, upper keeps max.
                    // If descending: lower keeps max, upper keeps min.
                    float out_d = asc ? (lower ? minv : maxv) : (lower ? maxv : minv);
                    int   out_i = asc ? (lower ? mini : maxi) : (lower ? maxi : mini);

                    dist_reg[j] = out_d;
                    idx_reg [j] = out_i;
                }
            }
            // Synchronize to ensure updated register values are visible for the next compare-exchange stage
            warp_sync();
        }
    }
}

// Merge the per-warp candidate buffer (shared memory) with the intermediate result (registers),
// following the specified 4 steps:
// 0) Intermediate result (registers) is sorted ascending (invariant).
// 1) Swap contents: registers <-> buffer, so that the buffer is now in registers.
// 2) Sort the (buffer) now in registers ascending using bitonic sort.
// 3) Merge registers and the (old intermediate) in shared memory into a bitonic sequence by:
//      reg[i] = min( reg[i], shared[k - 1 - i] )
// 4) Sort the merged result (registers) ascending.
//
// The candidate buffer may be partially filled; pad the remainder with +inf so we always operate on k elements.
static __device__ __forceinline__
void warp_merge_full(int warp_local_id,
                     int k,
                     int items_per_thread,
                     int items_per_thread_log2,
                     float* s_cand_dist_all,
                     int*   s_cand_idx_all,
                     int*   s_cand_counts,
                     float dist_reg[MAX_ITEMS_PER_THREAD],
                     int   idx_reg [MAX_ITEMS_PER_THREAD],
                     float &max_distance)
{
    const int lane = lane_id();
    const unsigned full_mask = 0xFFFFFFFFu;

    float* s_cand_dist = s_cand_dist_all + warp_local_id * k;
    int*   s_cand_idx  = s_cand_idx_all  + warp_local_id * k;

    // Pad the candidate buffer to size k with +inf/-1 so sort/merge operates on exactly k elements.
    int c = s_cand_counts[warp_local_id];
    for (int j = 0; j < items_per_thread; ++j) {
        int i_global = (lane << items_per_thread_log2) + j;
        if (i_global >= c && i_global < k) {
            s_cand_dist[i_global] = CUDART_INF_F;
            s_cand_idx [i_global] = -1;
        }
    }
    warp_sync();

    // Step 1: Swap content (buffer <-> registers), placing buffer into registers.
    for (int j = 0; j < items_per_thread; ++j) {
        int i_global = (lane << items_per_thread_log2) + j;
        float tmp_d = s_cand_dist[i_global];
        int   tmp_i = s_cand_idx [i_global];
        s_cand_dist[i_global] = dist_reg[j];
        s_cand_idx [i_global] = idx_reg [j];
        dist_reg[j] = tmp_d;
        idx_reg [j] = tmp_i;
    }
    warp_sync();

    // Step 2: Sort the buffer (now in registers) ascending using bitonic sort.
    warp_bitonic_sort_distributed(dist_reg, idx_reg, items_per_thread, k, items_per_thread_log2);
    warp_sync();

    // Step 3: Merge buffer (registers) with intermediate (shared) into registers via min(A[i], B[k-1-i]).
    for (int j = 0; j < items_per_thread; ++j) {
        int i_global = (lane << items_per_thread_log2) + j;
        int comp_idx = (k - 1) - i_global;
        float other_d = s_cand_dist[comp_idx];
        int   other_i = s_cand_idx [comp_idx];

        // Keep the smaller of the two; for ties, pick the one with smaller index (optional).
        if ((other_d < dist_reg[j]) || ((other_d == dist_reg[j]) && (other_i < idx_reg[j]))) {
            dist_reg[j] = other_d;
            idx_reg [j] = other_i;
        }
    }
    warp_sync();

    // Step 4: Sort the merged (bitonic) sequence in registers ascending.
    warp_bitonic_sort_distributed(dist_reg, idx_reg, items_per_thread, k, items_per_thread_log2);
    warp_sync();

    // Update max_distance to the k-th neighbor distance (last element).
    float new_max = __shfl_sync(full_mask, dist_reg[items_per_thread - 1], WARP_SIZE - 1);
    max_distance = new_max;

    // Reset candidate count for this warp.
    if (lane == 0) s_cand_counts[warp_local_id] = 0;
    warp_sync();
}

// Kernel: one warp processes one query. The per-warp intermediate result is kept
// in registers, and a per-warp candidate buffer is kept in shared memory.
// The data points are processed in shared-memory tiles for cache efficiency.
__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             int k,
                             std::pair<int, float>* __restrict__ result)
{
    const int lane    = lane_id();
    const int warp_lb = warp_id_in_block();
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_global = blockIdx.x * warps_per_block + warp_lb;
    if (warp_global >= query_count) return;

    // Compute per-thread register storage size. k is guaranteed divisible by 32.
    const int items_per_thread = k / WARP_SIZE;
    // log2(items_per_thread) used to map global index to lane/local registers efficiently.
    int items_per_thread_log2 = 0;
    if (items_per_thread > 1) {
        // Since items_per_thread is power of two, __ffs(items_per_thread) - 1 = log2(items_per_thread)
        items_per_thread_log2 = __ffs(items_per_thread) - 1;
    }

    // Shared memory layout:
    // [0, TILE_POINTS) float2 tile buffer for data points (common to block)
    // then warps_per_block * k int indices
    // then warps_per_block * k float distances
    // then warps_per_block int candidate counts
    extern __shared__ unsigned char smem[];
    float2* s_tile = reinterpret_cast<float2*>(smem);
    size_t offset_bytes = size_t(TILE_POINTS) * sizeof(float2);

    int* s_cand_idx_all = reinterpret_cast<int*>(smem + offset_bytes);
    offset_bytes += size_t(warps_per_block) * size_t(k) * sizeof(int);

    float* s_cand_dist_all = reinterpret_cast<float*>(smem + offset_bytes);
    offset_bytes += size_t(warps_per_block) * size_t(k) * sizeof(float);

    int* s_cand_counts = reinterpret_cast<int*>(smem + offset_bytes);

    // Pointers to this warp's candidate buffer
    int*   s_cand_idx  = s_cand_idx_all  + warp_lb * k;
    float* s_cand_dist = s_cand_dist_all + warp_lb * k;

    // Initialize per-warp candidate count
    if (lane == 0) s_cand_counts[warp_lb] = 0;
    warp_sync();

    // Load the query point (all threads in the warp will use it). Use lane 0 to load and broadcast.
    float2 q;
    if (lane == 0) {
        q = query[warp_global];
    }
    q.x = __shfl_sync(0xFFFFFFFFu, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFFu, q.y, 0);

    // Initialize the intermediate top-k in registers to +inf distances and -1 indices.
    float reg_dist[MAX_ITEMS_PER_THREAD];
    int   reg_idx [MAX_ITEMS_PER_THREAD];
    for (int j = 0; j < items_per_thread; ++j) {
        reg_dist[j] = CUDART_INF_F;
        reg_idx [j] = -1;
    }
    // The k-th neighbor threshold; initially +inf (no filtering).
    float max_distance = CUDART_INF_F;

    // Process data in tiles. The whole block cooperatively loads each tile into shared memory.
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        const int tile_count = min(TILE_POINTS, data_count - base);

        // Cooperative load into shared memory tile.
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            s_tile[t] = data[base + t];
        }
        __syncthreads(); // Ensure tile is loaded before any warp uses it.

        // Each warp processes all points in the shared tile for its own query.
        for (int t = lane; t < tile_count; t += WARP_SIZE) {
            float2 p = s_tile[t];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float d  = dx * dx + dy * dy;
            int   gidx = base + t;

            // Filter by current threshold and compact using a warp ballot.
            unsigned mask = __ballot_sync(0xFFFFFFFFu, d < max_distance);
            int cnt = __popc(mask);

            if (cnt > 0) {
                // Check for buffer overflow: if adding cnt exceeds k, merge buffer with current result first.
                // Lane 0 decides and broadcasts to the warp.
                int need_flush_int = 0;
                if (lane == 0) {
                    int curr = s_cand_counts[warp_lb];
                    need_flush_int = (curr + cnt > k) ? 1 : 0;
                }
                need_flush_int = __shfl_sync(0xFFFFFFFFu, need_flush_int, 0);

                if (need_flush_int) {
                    // Merge current (possibly partial) buffer: pad to k, swap, sort, merge-min, sort.
                    warp_merge_full(warp_lb, k, items_per_thread, items_per_thread_log2,
                                    s_cand_dist_all, s_cand_idx_all, s_cand_counts,
                                    reg_dist, reg_idx, max_distance);
                    // Re-evaluate the candidate predicate with the updated threshold.
                    mask = __ballot_sync(0xFFFFFFFFu, d < max_distance);
                    cnt = __popc(mask);
                }

                if (cnt > 0) {
                    // Reserve slots and write the new candidates into the per-warp buffer.
                    int base_pos = 0;
                    if (lane == 0) {
                        base_pos = s_cand_counts[warp_lb];
                        s_cand_counts[warp_lb] = base_pos + cnt;
                    }
                    base_pos = __shfl_sync(0xFFFFFFFFu, base_pos, 0);

                    // Compute rank of this lane among the true bits for coalesced writes.
/// @FIXED
/// #if __CUDACC_VER_MAJOR__ >= 9
#if __CUDACC_VER_MAJOR__ >= 9 && __CUDACC_VER_MAJOR__ < 11
                    unsigned lane_mask_lt = __lanemask_lt();
/// @FIXED
#elif __CUDACC_VER_MAJOR__ >= 11
/// @FIXED
                    unsigned lane_mask_lt;
/// @FIXED
                    asm("mov.u32 %0, %lanemask_lt;" : "=r"(lane_mask_lt));
#else
                    unsigned lane_mask_lt = (1u << lane) - 1u; // Fallback; lane=31 yields undefined shift, but this path is for older toolkits.
#endif
                    int rank = __popc(mask & lane_mask_lt);

                    if (d < max_distance) {
                        int pos = base_pos + rank;
                        s_cand_dist[pos] = d;
                        s_cand_idx [pos] = gidx;
                    }

                    // If buffer is now full, merge it immediately.
                    int do_flush_int = 0;
                    if (lane == 0) {
                        do_flush_int = (s_cand_counts[warp_lb] == k) ? 1 : 0;
                    }
                    do_flush_int = __shfl_sync(0xFFFFFFFFu, do_flush_int, 0);
                    if (do_flush_int) {
                        warp_merge_full(warp_lb, k, items_per_thread, items_per_thread_log2,
                                        s_cand_dist_all, s_cand_idx_all, s_cand_counts,
                                        reg_dist, reg_idx, max_distance);
                    }
                }
            }
        }
        __syncthreads(); // Before loading the next tile
    }

    // After processing all tiles, if the candidate buffer is not empty, merge it once more.
    int leftover = 0;
    if (lane == 0) leftover = s_cand_counts[warp_lb];
    leftover = __shfl_sync(0xFFFFFFFFu, leftover, 0);
    if (leftover > 0) {
        warp_merge_full(warp_lb, k, items_per_thread, items_per_thread_log2,
                        s_cand_dist_all, s_cand_idx_all, s_cand_counts,
                        reg_dist, reg_idx, max_distance);
    }

    // Write out the final sorted top-k for this query.
    // Each thread writes its k/32 consecutive entries.
    const int out_base = warp_global * k;
    for (int j = 0; j < items_per_thread; ++j) {
        int out_idx = out_base + (lane * items_per_thread + j);
        result[out_idx].first  = reg_idx[j];
        result[out_idx].second = reg_dist[j];
    }
}

// Public host interface. Assumes query, data, and result are device pointers allocated with cudaMalloc.
// k is a power of two in [32, 1024]. data_count >= k.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block and grid configuration.
    // We use 4 warps (128 threads) per block to keep shared memory within 48KB for k up to 1024.
    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_warps = (query_count + 0) / 1; // one warp per query
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Compute dynamic shared memory size: tile + per-warp buffers + per-warp counts.
    size_t smem_bytes = 0;
    smem_bytes += size_t(TILE_POINTS) * sizeof(float2);                           // tile
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(int);              // per-warp candidate idx
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(float);            // per-warp candidate dist
    smem_bytes += size_t(warps_per_block) * sizeof(int);                          // per-warp candidate counts

    // Launch kernel.
    knn2d_kernel<<<blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, k, result);
}
