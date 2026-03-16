#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace knn_detail {

// Problem-specific constants tuned for modern data-center GPUs.
// - 16 warps/block maximizes reuse of each shared-memory data tile across queries.
// - 4096 float2 points = 32 KiB of shared memory, which is a good balance between
//   occupancy and synchronization overhead on A100/H100-class GPUs.
constexpr unsigned FULL_MASK               = 0xFFFFFFFFu;
constexpr int      WARP_SIZE               = 32;
constexpr int      SELECT_WARPS_PER_BLOCK  = 16;
constexpr int      SELECT_THREADS_PER_BLOCK = SELECT_WARPS_PER_BLOCK * WARP_SIZE;
constexpr int      DATA_TILE_POINTS        = 4096;

constexpr int      SORT_WARPS_PER_BLOCK    = 4;
constexpr int      SORT_THREADS_PER_BLOCK  = SORT_WARPS_PER_BLOCK * WARP_SIZE;

static_assert(DATA_TILE_POINTS % WARP_SIZE == 0, "Tile size must be a multiple of warp size.");
static_assert(DATA_TILE_POINTS >= 1024, "The first tile must be large enough to seed the largest supported K.");

// Device-side POD mirror of std::pair<int,float>.
// We intentionally avoid constructing std::pair in device code because device annotations on
// standard-library constructors/operators vary across libstdc++/libc++ configurations.
// The output buffer is allocated by cudaMalloc, so treating it as raw storage with an identical POD
// layout is safe as long as size/alignment match.
struct PairIF {
    int   first;
    float second;
};

static_assert(std::is_trivially_copyable<PairIF>::value, "PairIF must be trivially copyable.");
static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "Unexpected std::pair<int,float> size.");
static_assert(alignof(PairIF) == alignof(std::pair<int, float>), "Unexpected std::pair<int,float> alignment.");

// Total ordering used only for the final output ordering.
// Tie resolution is arbitrary per the problem statement; using the index simply makes the order deterministic.
__device__ __forceinline__ bool pair_less(float a_dist, int a_idx, float b_dist, int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ void swap_pair(float &a_dist, int &a_idx, float &b_dist, int &b_idx) {
    const float td = a_dist;
    const int   ti = a_idx;
    a_dist = b_dist;
    a_idx  = b_idx;
    b_dist = td;
    b_idx  = ti;
}

// Warp-wide argmax over each lane's current local worst distance.
// This identifies the globally worst stored entry among the distributed top-K reservoir.
// Replacing exactly that entry whenever a better candidate appears preserves the exact K smallest elements.
__device__ __forceinline__ void warp_argmax(float local_value, int lane,
                                            float &max_value, int &max_lane) {
    max_value = local_value;
    max_lane  = lane;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(FULL_MASK, max_value, offset);
        const int   other_lane  = __shfl_down_sync(FULL_MASK, max_lane,  offset);

        if ((other_value > max_value) || ((other_value == max_value) && (other_lane < max_lane))) {
            max_value = other_value;
            max_lane  = other_lane;
        }
    }

    max_value = __shfl_sync(FULL_MASK, max_value, 0);
    max_lane  = __shfl_sync(FULL_MASK, max_lane,  0);
}

// Predicated slot update used instead of true dynamic indexing.
// This is important for performance: dynamic indexing into fixed-size arrays often forces the compiler
// to spill them to local memory, while an unrolled predicated update lets it keep them scalarized in registers.
template <int ITEMS>
__device__ __forceinline__ void replace_slot(float (&dist)[ITEMS], int (&idx)[ITEMS],
                                             int slot, float new_dist, int new_idx) {
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (slot == i) {
            dist[i] = new_dist;
            idx[i]  = new_idx;
        }
    }
}

// Scan one lane's private register-resident portion of the distributed reservoir to find its local worst item.
template <int ITEMS>
__device__ __forceinline__ void recompute_local_worst(const float (&dist)[ITEMS],
                                                      float &worst_dist, int &worst_slot) {
    worst_dist = dist[0];
    worst_slot = 0;

    #pragma unroll
    for (int i = 1; i < ITEMS; ++i) {
        if (dist[i] > worst_dist) {
            worst_dist = dist[i];
            worst_slot = i;
        }
    }
}

// Phase 1: exact top-K selection, but intentionally kept unordered.
// Each warp owns one query. A whole thread block loads a tile of data points into shared memory, and all warps
// reuse that tile. The per-query top-K is distributed across warp lanes: lane l owns K/32 entries.
//
// The hot path keeps the reservoir unordered and only tracks the current global worst element.
// This keeps update work low and, more importantly, avoids any need to keep the reservoir sorted during the
// full scan over millions of points. Final ordering is delegated to a tiny second kernel.
template <int K>
__global__ __launch_bounds__(SELECT_THREADS_PER_BLOCK, 1)
void knn_select_kernel(const float2 * __restrict__ query,
                       int query_count,
                       const float2 * __restrict__ data,
                       int data_count,
                       PairIF * __restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= WARP_SIZE && K <= 1024, "K out of supported range.");
    static_assert(DATA_TILE_POINTS >= K, "The first tile must be able to seed the initial reservoir.");

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    __shared__ float2 s_data[DATA_TILE_POINTS];

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;
    const int query_id = static_cast<int>(blockIdx.x) * SELECT_WARPS_PER_BLOCK + warp_id;
    const bool valid_query = (query_id < query_count);

    // Load one query point per warp and broadcast it with shuffles so we do not issue 32 identical global loads.
    float qx = 0.0f;
    float qy = 0.0f;
    if (valid_query && lane == 0) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Register-resident distributed reservoir.
    float best_dist[ITEMS_PER_LANE];
    int   best_idx [ITEMS_PER_LANE];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        best_dist[i] = CUDART_INF_F;
        best_idx[i]  = -1;
    }

    float local_worst_dist = CUDART_INF_F;
    int   local_worst_slot = 0;
    float warp_worst_dist  = CUDART_INF_F;
    int   warp_worst_lane  = 0;

    // Because K <= 1024 and the tile is 4096 points, the first K points always fit in the first tile.
    // We exploit that by seeding the reservoir directly with the first K distances (one slot per lane per step)
    // before starting the usual replace-the-current-worst logic.
    int filled_slots = 0;

    for (int batch_base = 0; batch_base < data_count; batch_base += DATA_TILE_POINTS) {
        int batch_count = data_count - batch_base;
        if (batch_count > DATA_TILE_POINTS) batch_count = DATA_TILE_POINTS;

        // Cooperative block-wide load of a contiguous data tile into shared memory.
        for (int i = tid; i < batch_count; i += SELECT_THREADS_PER_BLOCK) {
            s_data[i] = data[batch_base + i];
        }
        __syncthreads();

        if (valid_query) {
            // Warp processes the tile in lock-step chunks of 32 points so every lane stays active for warp collectives.
            for (int t_base = 0; t_base < batch_count; t_base += WARP_SIZE) {

                // Fast-path warm-up: the first ITEMS_PER_LANE steps simply populate the reservoir.
                // For these steps, all lanes are guaranteed to be in-bounds by the DATA_TILE_POINTS >= K assertion.
                if (filled_slots < ITEMS_PER_LANE) {
                    const int    t        = t_base + lane;
                    const float2 p        = s_data[t];
                    const float  dx       = qx - p.x;
                    const float  dy       = qy - p.y;
                    const float  cand_dist = fmaf(dx, dx, dy * dy);
                    const int    cand_idx  = batch_base + t;

                    replace_slot(best_dist, best_idx, filled_slots, cand_dist, cand_idx);
                    ++filled_slots;

                    if (filled_slots == ITEMS_PER_LANE) {
                        recompute_local_worst(best_dist, local_worst_dist, local_worst_slot);
                        __syncwarp(FULL_MASK);
                        warp_argmax(local_worst_dist, lane, warp_worst_dist, warp_worst_lane);
                    }
                    continue;
                }

                const int t = t_base + lane;

                float cand_dist = CUDART_INF_F;
                int   cand_idx  = -1;

                if (t < batch_count) {
                    const float2 p  = s_data[t];
                    const float  dx = qx - p.x;
                    const float  dy = qy - p.y;
                    cand_dist = fmaf(dx, dx, dy * dy);
                    cand_idx  = batch_base + t;
                }

                // Only candidates better than the current K-th neighbor can matter.
                unsigned active = __ballot_sync(FULL_MASK, cand_dist < warp_worst_dist);

                // Process qualifying candidates in the natural lane/data order for this 32-point step.
                while (active) {
                    const int   sel_lane = __ffs(static_cast<int>(active)) - 1;
                    const float sel_dist = __shfl_sync(FULL_MASK, cand_dist, sel_lane);
                    const int   sel_idx  = __shfl_sync(FULL_MASK, cand_idx,  sel_lane);

                    // Only the lane that owns the current global worst slot performs the replacement.
                    if (lane == warp_worst_lane) {
                        replace_slot(best_dist, best_idx, local_worst_slot, sel_dist, sel_idx);
                        recompute_local_worst(best_dist, local_worst_dist, local_worst_slot);
                    }

                    if (lane == sel_lane) {
                        cand_dist = CUDART_INF_F; // consume this candidate
                    }

                    __syncwarp(FULL_MASK);
                    warp_argmax(local_worst_dist, lane, warp_worst_dist, warp_worst_lane);
                    active = __ballot_sync(FULL_MASK, cand_dist < warp_worst_dist);
                }
            }
        }

        // Required because the next tile load overwrites the same shared-memory buffer.
        __syncthreads();
    }

    // Store the exact but still unordered top-K into the output buffer.
    // Layout is interleaved by slot: rank = slot * 32 + lane.
    // A second kernel will sort each query's K pairs in-place without any extra device allocation.
    if (valid_query) {
        const std::size_t out_base = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
            const int rank = (slot << 5) + lane;
            result[out_base + static_cast<std::size_t>(rank)] = PairIF{best_idx[slot], best_dist[slot]};
        }
    }
}

// Phase 2: final in-place sort of each query's K output pairs.
// This kernel is intentionally separated from the long scan kernel above:
// - The selection phase stays register-resident and avoids dynamic-index spills.
// - Sorting K <= 1024 values per query is tiny compared to scanning millions of data points.
// - We reuse the already-allocated result array; no extra device memory is needed.
//
// The same interleaved ownership is used here: rank = slot * 32 + lane.
// Bitonic-sort stages are split as follows:
// - stride < 32  : partner is in another lane but the same slot
// - stride >= 32 : partner is in the same lane but another slot
//
// The split is important because in-lane partner stages must be handled pair-once to avoid
// overwriting one partner before the other partner has read the old value.
template <int K>
__global__ __launch_bounds__(SORT_THREADS_PER_BLOCK)
void knn_sort_kernel(PairIF * __restrict__ result, int query_count) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= WARP_SIZE && K <= 1024, "K out of supported range.");

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    __shared__ float s_dist[SORT_WARPS_PER_BLOCK][K];
    __shared__ int   s_idx [SORT_WARPS_PER_BLOCK][K];

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;
    const int query_id = static_cast<int>(blockIdx.x) * SORT_WARPS_PER_BLOCK + warp_id;

    if (query_id >= query_count) {
        return;
    }

    float *warp_dist = s_dist[warp_id];
    int   *warp_idx  = s_idx [warp_id];

    const std::size_t base = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

    // Load this warp's query result slice from global memory into a warp-private shared-memory slice.
    #pragma unroll
    for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
        const int rank = (slot << 5) + lane;
        const PairIF p = result[base + static_cast<std::size_t>(rank)];
        warp_dist[rank] = p.second;
        warp_idx [rank] = p.first;
    }
    __syncwarp(FULL_MASK);

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride >= WARP_SIZE) {
                // Same-lane partner stages: process each local pair exactly once.
                const int slot_xor = stride >> 5;

                #pragma unroll
                for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
                    const int partner_slot = slot ^ slot_xor;

                    if (partner_slot > slot) {
                        const int rank_a = (slot         << 5) + lane;
                        const int rank_b = (partner_slot << 5) + lane;

                        float a_dist = warp_dist[rank_a];
                        int   a_idx  = warp_idx [rank_a];
                        float b_dist = warp_dist[rank_b];
                        int   b_idx  = warp_idx [rank_b];

                        const bool ascending = ((rank_a & size) == 0);

                        if (ascending) {
                            if (pair_less(b_dist, b_idx, a_dist, a_idx)) {
                                swap_pair(a_dist, a_idx, b_dist, b_idx);
                            }
                        } else {
                            if (pair_less(a_dist, a_idx, b_dist, b_idx)) {
                                swap_pair(a_dist, a_idx, b_dist, b_idx);
                            }
                        }

                        warp_dist[rank_a] = a_dist;
                        warp_idx [rank_a] = a_idx;
                        warp_dist[rank_b] = b_dist;
                        warp_idx [rank_b] = b_idx;
                    }
                }
            } else {
                // Cross-lane partner stages: partner remains in the same slot, so every rank can update independently.
                #pragma unroll
                for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
                    const int rank    = (slot << 5) + lane;
                    const int partner = rank ^ stride;

                    const float self_dist  = warp_dist[rank];
                    const int   self_idx   = warp_idx [rank];
                    const float other_dist = warp_dist[partner];
                    const int   other_idx  = warp_idx [partner];

                    const bool ascending = ((rank & size) == 0);
                    const bool keep_min  = ((rank < partner) == ascending);

                    float out_dist = self_dist;
                    int   out_idx  = self_idx;

                    if (keep_min) {
                        if (pair_less(other_dist, other_idx, self_dist, self_idx)) {
                            out_dist = other_dist;
                            out_idx  = other_idx;
                        }
                    } else {
                        if (pair_less(self_dist, self_idx, other_dist, other_idx)) {
                            out_dist = other_dist;
                            out_idx  = other_idx;
                        }
                    }

                    warp_dist[rank] = out_dist;
                    warp_idx [rank] = out_idx;
                }
            }

            __syncwarp(FULL_MASK);
        }
    }

    // Write the final sorted neighbors back to global memory.
    #pragma unroll
    for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
        const int rank = (slot << 5) + lane;
        result[base + static_cast<std::size_t>(rank)] = PairIF{warp_idx[rank], warp_dist[rank]};
    }
}

template <int K>
inline void launch_knn_impl(const float2 *query,
                            int query_count,
                            const float2 *data,
                            int data_count,
                            PairIF *result) {
    const dim3 select_block(SELECT_THREADS_PER_BLOCK);
    const dim3 select_grid((query_count + SELECT_WARPS_PER_BLOCK - 1) / SELECT_WARPS_PER_BLOCK);
    knn_select_kernel<K><<<select_grid, select_block>>>(query, query_count, data, data_count, result);

    const dim3 sort_block(SORT_THREADS_PER_BLOCK);
    const dim3 sort_grid((query_count + SORT_WARPS_PER_BLOCK - 1) / SORT_WARPS_PER_BLOCK);
    knn_sort_kernel<K><<<sort_grid, sort_block>>>(result, query_count);
}

} // namespace knn_detail

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    // Intentionally asynchronous: no device-wide synchronization or implicit error handling is inserted here.
    // The caller can synchronize/check launch errors according to its own stream-management policy.
    if (query_count <= 0) {
        return;
    }

    auto *pod_result = reinterpret_cast<knn_detail::PairIF *>(result);

    switch (k) {
        case 32:   knn_detail::launch_knn_impl<32>  (query, query_count, data, data_count, pod_result); break;
        case 64:   knn_detail::launch_knn_impl<64>  (query, query_count, data, data_count, pod_result); break;
        case 128:  knn_detail::launch_knn_impl<128> (query, query_count, data, data_count, pod_result); break;
        case 256:  knn_detail::launch_knn_impl<256> (query, query_count, data, data_count, pod_result); break;
        case 512:  knn_detail::launch_knn_impl<512> (query, query_count, data, data_count, pod_result); break;
        case 1024: knn_detail::launch_knn_impl<1024>(query, query_count, data, data_count, pod_result); break;
        default:
            // Per the problem statement, K is always a supported power of two in [32, 1024].
            break;
    }
}