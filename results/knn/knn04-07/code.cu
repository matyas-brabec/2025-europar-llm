#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace {

// The hardware warp size is fixed at 32 on NVIDIA GPUs.
constexpr int WARP_SIZE_CONST   = 32;

// 256 threads = 8 warps/query-workers per block.
// This is a good A100/H100 trade-off:
//   * enough warps per block to reuse each shared-memory data batch 8x,
//   * still modest shared-memory usage,
//   * manageable register pressure even for k = 1024.
constexpr int BLOCK_THREADS     = 256;
constexpr int WARPS_PER_BLOCK   = BLOCK_THREADS / WARP_SIZE_CONST;
constexpr unsigned FULL_MASK    = 0xffffffffu;

// Invalid/sentinel entry.
// Distance is +inf; index is negative so it is distinguishable from a real point
// even if a real squared distance overflows to +inf.
constexpr float INVALID_DIST    = CUDART_INF_F;
constexpr int   INVALID_INDEX   = -1;

static_assert(BLOCK_THREADS % WARP_SIZE_CONST == 0, "BLOCK_THREADS must be a multiple of 32");

__device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE_CONST - 1);
}

__device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

// "Better" means "should appear earlier in the sorted top-k list".
// Ordering is primarily by distance. Ties are intentionally arbitrary except for one
// special case: a real candidate must beat an unused sentinel slot when both hold +inf.
// This lets genuine +inf distances fill the queue correctly if the arithmetic overflows.
__device__ __forceinline__ bool better(const float a_dist, const int a_idx,
                                       const float b_dist, const int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx >= 0) && (b_idx < 0));
}

// Keep either the minimum or the maximum of (self, other), according to the ordering above.
__device__ __forceinline__ void compare_exchange_keep(float &self_dist, int &self_idx,
                                                      const float other_dist, const int other_idx,
                                                      const bool keep_min) {
    const bool take_other = keep_min ? better(other_dist, other_idx, self_dist, self_idx)
                                     : better(self_dist, self_idx, other_dist, other_idx);
    if (take_other) {
        self_dist = other_dist;
        self_idx  = other_idx;
    }
}

// Full 32-way bitonic sort in ascending order.
//
// Each lane owns one (distance, index) pair. Communication is via warp shuffles;
// those collectives are warp-synchronous, so no extra __syncwarp() is required here.
__device__ __forceinline__ void warp_bitonic_sort_32(float &dist, int &idx) {
    const int lane = lane_id();

#pragma unroll
    for (int k = 2; k <= WARP_SIZE_CONST; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const float other_dist = __shfl_xor_sync(FULL_MASK, dist, j);
            const int   other_idx  = __shfl_xor_sync(FULL_MASK, idx,  j);

            const bool lower_lane = ((lane & j) == 0);
            const bool dir_asc    = ((lane & k) == 0);
            const bool keep_min   = (lower_lane == dir_asc);

            compare_exchange_keep(dist, idx, other_dist, other_idx, keep_min);
        }
    }
}

// Bitonic merge for a 32-element bitonic sequence to ascending order.
__device__ __forceinline__ void warp_bitonic_merge_32(float &dist, int &idx) {
    const int lane = lane_id();

#pragma unroll
    for (int j = WARP_SIZE_CONST >> 1; j > 0; j >>= 1) {
        const float other_dist = __shfl_xor_sync(FULL_MASK, dist, j);
        const int   other_idx  = __shfl_xor_sync(FULL_MASK, idx,  j);

        const bool lower_lane = ((lane & j) == 0);
        compare_exchange_keep(dist, idx, other_dist, other_idx, lower_lane);
    }
}

// Merge two already-sorted 32-element ascending sequences A and B.
// Input : lane i holds A[i] and B[i].
// Output: A becomes the smallest 32 merged elements (ascending),
//         B becomes the largest  32 merged elements (ascending).
//
// Implementation detail:
//   1) Reverse B to make A || reverse(B) a 64-element bitonic sequence.
//   2) First compare-exchange between the lower and upper halves.
//   3) Each half is now a 32-element bitonic sequence; merge each half independently.
__device__ __forceinline__ void merge_sorted_32x32(float &a_dist, int &a_idx,
                                                   float &b_dist, int &b_idx) {
    const int lane = lane_id();

    const float b_rev_dist = __shfl_sync(FULL_MASK, b_dist, (WARP_SIZE_CONST - 1) - lane);
    const int   b_rev_idx  = __shfl_sync(FULL_MASK, b_idx,  (WARP_SIZE_CONST - 1) - lane);

    float lo_dist, hi_dist;
    int   lo_idx,  hi_idx;

    if (better(a_dist, a_idx, b_rev_dist, b_rev_idx)) {
        lo_dist = a_dist;     lo_idx = a_idx;
        hi_dist = b_rev_dist; hi_idx = b_rev_idx;
    } else {
        lo_dist = b_rev_dist; lo_idx = b_rev_idx;
        hi_dist = a_dist;     hi_idx = a_idx;
    }

    warp_bitonic_merge_32(lo_dist, lo_idx);
    warp_bitonic_merge_32(hi_dist, hi_idx);

    a_dist = lo_dist; a_idx = lo_idx;
    b_dist = hi_dist; b_idx = hi_idx;
}

__device__ __forceinline__ float squared_l2_2d(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

// One warp handles exactly one query.
// K is specialized at compile time so that:
//   * the private top-k state can stay in registers,
//   * merge loops can be fully unrolled,
//   * the compiler can aggressively optimize the hot path.
template <int K>
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> * __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024]");
    static_assert(K % WARP_SIZE_CONST == 0, "K must be a multiple of warp size");

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE_CONST;
    constexpr int LAST_CHUNK     = ITEMS_PER_LANE - 1;

    // Shared-memory cache for one data batch.
    // Every thread loads one float2, then all 8 warps reuse the batch.
    __shared__ float2 sh_data[BLOCK_THREADS];

    const int lane   = lane_id();
    const int warp   = warp_id_in_block();
    const int q_base = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK;
    const int qid    = q_base + warp;

    // Uniform early exit for a completely out-of-range block.
    if (q_base >= query_count) {
        return;
    }

    const bool active = (qid < query_count);

    // Load the query point once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Warp-private top-k state.
    // Layout is "striped" by 32-element chunks:
    //   top_dist[t] / top_idx[t] in lane i stores global rank (t * 32 + i).
    // Each chunk is always sorted across lanes, and the chunks are globally ordered.
    float top_dist[ITEMS_PER_LANE];
    int   top_idx[ITEMS_PER_LANE];

#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        top_dist[t] = INVALID_DIST;
        top_idx[t]  = INVALID_INDEX;
    }

    // Iterate over the data set in shared-memory batches.
    for (int batch_base = 0; batch_base < data_count; batch_base += BLOCK_THREADS) {
        int points_in_batch = data_count - batch_base;
        if (points_in_batch > BLOCK_THREADS) points_in_batch = BLOCK_THREADS;

        // Cooperative block-wide load of the next batch into shared memory.
        if (threadIdx.x < points_in_batch) {
            sh_data[threadIdx.x] = data[batch_base + threadIdx.x];
        }

        // Whole-block barrier is required because all warps consume the shared batch.
        __syncthreads();

        // Process the shared batch as 8 packets of 32 candidates.
        // Each packet is:
        //   1) distance-evaluated in parallel by the warp,
        //   2) pruned against the current worst top-k element,
        //   3) sorted cooperatively,
        //   4) merged chunk-by-chunk into the private top-k state.
#pragma unroll
        for (int g = 0; g < WARPS_PER_BLOCK; ++g) {
            const int local_idx = g * WARP_SIZE_CONST + lane;

            float cand_dist = INVALID_DIST;
            int   cand_idx  = INVALID_INDEX;

            if (active && local_idx < points_in_batch) {
                const float2 p = sh_data[local_idx];
                cand_dist = squared_l2_2d(qx, qy, p);
                cand_idx  = batch_base + local_idx;
            }

            // Current pruning threshold = worst element in the sorted top-k list.
            const float worst_dist = __shfl_sync(FULL_MASK, top_dist[LAST_CHUNK], WARP_SIZE_CONST - 1);
            const int   worst_idx  = __shfl_sync(FULL_MASK, top_idx[LAST_CHUNK],  WARP_SIZE_CONST - 1);

            const bool relevant = better(cand_dist, cand_idx, worst_dist, worst_idx);

            // If no lane has a candidate better than the current threshold, skip the entire
            // packet update. This is critical: once the queue stabilizes, most packets are discarded
            // before the expensive sort/merge path.
            if (__any_sync(FULL_MASK, relevant)) {
                if (!relevant) {
                    cand_dist = INVALID_DIST;
                    cand_idx  = INVALID_INDEX;
                }

                // Sort the 32-candidate packet.
                warp_bitonic_sort_32(cand_dist, cand_idx);

                // Cascade the packet through the K/32 sorted chunks.
                // After each merge:
                //   * top_[t] receives the smaller half,
                //   * (cand_dist, cand_idx) carries the larger half to the next chunk.
#pragma unroll
                for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                    merge_sorted_32x32(top_dist[t], top_idx[t], cand_dist, cand_idx);
                }
            }
        }

        // Whole-block barrier before the next batch overwrites shared memory.
        __syncthreads();
    }

    // Store the final top-k list in row-major query-major layout:
    //   result[qid * K + j] is the j-th nearest neighbor of query[qid].
    if (active) {
        const std::size_t out_base = static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);

#pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const std::size_t out_idx = out_base + static_cast<std::size_t>(t * WARP_SIZE_CONST + lane);
            result[out_idx].first  = top_idx[t];
            result[out_idx].second = top_dist[t];
        }
    }
}

} // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // K is runtime-known but constrained to six power-of-two values.
    // Compile-time specialization keeps the warp-private state register-resident
    // and fully unrolls the fixed-size merge network.
#define KNN_LAUNCH_CASE(KVAL) \
    case KVAL: \
        knn_kernel<KVAL><<<grid, block>>>(query, query_count, data, data_count, result); \
        break

    switch (k) {
        KNN_LAUNCH_CASE(32);
        KNN_LAUNCH_CASE(64);
        KNN_LAUNCH_CASE(128);
        KNN_LAUNCH_CASE(256);
        KNN_LAUNCH_CASE(512);
        KNN_LAUNCH_CASE(1024);
        default:
            // Per the problem statement, k is always valid.
            break;
    }

#undef KNN_LAUNCH_CASE
}