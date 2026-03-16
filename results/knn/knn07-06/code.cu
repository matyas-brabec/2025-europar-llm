#include <cuda_runtime.h>
#include <utility>
#include <climits>

namespace {

// -----------------------------------------------------------------------------
// Kernel configuration tuned for modern datacenter GPUs (A100/H100 class).
//
// Design rationale:
// - One warp computes one query.
// - 256 threads/block = 8 warps/block = 8 queries per block.
//   This is a good balance between:
//     * enough queries per block to reuse the staged data batch,
//     * enough blocks in the grid when query_count is only in the low thousands,
//     * a worst-case shared-memory footprint that still permits 2 resident
//       blocks/SM for K=1024 on A100-class shared-memory limits.
// - 1792 staged points/block is the largest multiple of 256 that keeps the
//   worst-case block footprint below ~80 KiB even when each of the 8 warps owns
//   a 1024-entry candidate buffer. Each thread performs exactly 7 coalesced
//   float2 loads on full batches.
// -----------------------------------------------------------------------------
constexpr int kWarpSize       = 32;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr int kBatchPoints    = 1792;
constexpr int kBatchLoadIters = kBatchPoints / kBlockThreads;
constexpr unsigned kFullMask  = 0xFFFFFFFFu;

using ResultPair = std::pair<int, float>;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a whole number of warps.");
static_assert(kBatchPoints % kBlockThreads == 0, "Batch size chosen so that each thread performs an equal number of loads on full batches.");
static_assert(kBatchPoints % kWarpSize == 0, "Batch size must be a multiple of the warp size.");

// Lexicographic order on (distance, index).
// The problem statement does not constrain tie handling, but using the index as
// a secondary key gives a total order, which makes the bitonic networks
// deterministic and well-defined.
__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool pair_greater(float da, int ia, float db, int ib) {
    return (da > db) || ((da == db) && (ia > ib));
}

__device__ __forceinline__ void swap_pair(float &da, int &ia, float &db, int &ib) {
    const float td = da;
    const int   ti = ia;
    da = db; ia = ib;
    db = td; ib = ti;
}

__device__ __forceinline__ float squared_l2_2d(float qx, float qy, float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    // Squared Euclidean distance in 2D. One FFMA is enough here.
    return fmaf(dx, dx, dy * dy);
}

// Shared-memory bitonic sort of K (distance,index) pairs in ascending order.
// The buffer is warp-private in shared memory. Each lane owns a contiguous chunk
// of K/32 elements:
//   logical positions handled by lane L are [L*CHUNK, L*CHUNK + CHUNK-1].
// Only the lower-index endpoint of each compare/swap pair performs the swap, so
// there are no write races. __syncwarp() after every substage provides the
// required shared-memory visibility.
template <int K>
__device__ __forceinline__ void shared_bitonic_sort_pairs(int *cand_idx, float *cand_dist) {
    constexpr int CHUNK = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int base = lane * CHUNK;

    #pragma unroll
    for (int stage = 2; stage <= K; stage <<= 1) {
        #pragma unroll
        for (int j = stage >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int s = 0; s < CHUNK; ++s) {
                const int i = base + s;
                const int partner = i ^ j;

                if (partner > i) {
                    float di = cand_dist[i];
                    int   ii = cand_idx[i];
                    float dp = cand_dist[partner];
                    int   ip = cand_idx[partner];

                    const bool up = ((i & stage) == 0);
                    if (up ? pair_greater(di, ii, dp, ip)
                           : pair_less(di, ii, dp, ip)) {
                        cand_dist[i]       = dp;
                        cand_idx[i]        = ip;
                        cand_dist[partner] = di;
                        cand_idx[partner]  = ii;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// The sequence produced by step 2 of the requested algorithm is already
// bitonic. Therefore step 3 needs only the final "bitonic merge" half of the
// Bitonic Sort network, not the full construction of bitonic runs again.
// This is still exactly the relevant sorting phase of Bitonic Sort.
template <int K>
__device__ __forceinline__ void shared_bitonic_merge_up_pairs(int *cand_idx, float *cand_dist) {
    constexpr int CHUNK = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int base = lane * CHUNK;

    #pragma unroll
    for (int j = K >> 1; j > 0; j >>= 1) {
        #pragma unroll
        for (int s = 0; s < CHUNK; ++s) {
            const int i = base + s;
            const int partner = i ^ j;

            if (partner > i) {
                float di = cand_dist[i];
                int   ii = cand_idx[i];
                float dp = cand_dist[partner];
                int   ip = cand_idx[partner];

                if (pair_greater(di, ii, dp, ip)) {
                    cand_dist[i]       = dp;
                    cand_idx[i]        = ip;
                    cand_dist[partner] = di;
                    cand_idx[partner]  = ii;
                }
            }
        }
        __syncwarp(kFullMask);
    }
}

// Flush the warp-private candidate buffer into the warp-private intermediate
// result. This implements the requested merge procedure:
//
// 1. Pad unused candidate slots with (+inf, INT_MAX).
// 2. Sort the candidate buffer in ascending order using Bitonic Sort.
// 3. Build the bitonic merged sequence in-place:
//      merged[i] = min(buffer[i], result[K-1-i]).
// 4. Sort that bitonic sequence ascending using the bitonic-merge phase.
// 5. Copy the updated top-K back into the warp-private intermediate result and
//    refresh max_distance = distance of the K-th nearest neighbor.
template <int K>
__device__ __forceinline__
void flush_candidate_buffer(float (&result_dist)[K / kWarpSize],
                            int   (&result_idx )[K / kWarpSize],
                            float &max_distance,
                            int   *count_ptr,
                            int   *cand_idx,
                            float *cand_dist) {
    constexpr int CHUNK = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int base = lane * CHUNK;

    __syncwarp(kFullMask);

    const int count = *count_ptr;
    if (count == 0) {
        return;
    }

    // Pad the inactive tail of the buffer with sentinels so that the network
    // can always sort exactly K elements.
    #pragma unroll
    for (int s = 0; s < CHUNK; ++s) {
        const int pos = base + s;
        if (pos >= count) {
            cand_idx[pos]  = INT_MAX;
            cand_dist[pos] = CUDART_INF_F;
        }
    }
    __syncwarp(kFullMask);

    // Step 1: full sort of the candidate buffer.
    shared_bitonic_sort_pairs<K>(cand_idx, cand_dist);

    // Step 2: form the first K elements of the union as a bitonic sequence.
    // Result is already sorted ascending in registers. Its reverse element for
    // logical position i is at position K-1-i, which maps to lane (31-lane)
    // and slot (CHUNK-1-s) under the contiguous-chunk distribution.
    #pragma unroll
    for (int s = 0; s < CHUNK; ++s) {
        const int pos = base + s;

        const float rev_dist =
            __shfl_sync(kFullMask, result_dist[CHUNK - 1 - s], 31 - lane);
        const int rev_idx =
            __shfl_sync(kFullMask, result_idx[CHUNK - 1 - s], 31 - lane);

        const float buf_dist = cand_dist[pos];
        const int   buf_idx  = cand_idx[pos];

        if (pair_less(rev_dist, rev_idx, buf_dist, buf_idx)) {
            cand_dist[pos] = rev_dist;
            cand_idx[pos]  = rev_idx;
        }
    }
    __syncwarp(kFullMask);

    // Step 3: the sequence is already bitonic, so only the merge phase is
    // needed to sort it ascending.
    shared_bitonic_merge_up_pairs<K>(cand_idx, cand_dist);

    // Update the private intermediate result.
    #pragma unroll
    for (int s = 0; s < CHUNK; ++s) {
        const int pos = base + s;
        result_dist[s] = cand_dist[pos];
        result_idx[s]  = cand_idx[pos];
    }

    // The global K-th element lives in lane 31, local slot CHUNK-1.
    max_distance = __shfl_sync(kFullMask, result_dist[CHUNK - 1], 31);

    if (lane == 0) {
        *count_ptr = 0;
    }
    __syncwarp(kFullMask);
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 2)
void knn_kernel(const float2    *__restrict__ query,
                int query_count,
                const float2    *__restrict__ data,
                int data_count,
                ResultPair      *__restrict__ result) {
    static_assert(K >= 32 && K <= 1024, "K must be between 32 and 1024.");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of the warp size.");

    constexpr int CHUNK = K / kWarpSize;

    // Dynamic shared-memory layout:
    //   [ staged data batch | per-warp counts | per-warp candidate indices | per-warp candidate distances ]
    extern __shared__ __align__(16) unsigned char smem[];
    float2 *const sh_data      = reinterpret_cast<float2*>(smem);
    int    *const sh_counts    = reinterpret_cast<int*>(sh_data + kBatchPoints);
    int    *const sh_cand_idx  = sh_counts + kWarpsPerBlock;
    float  *const sh_cand_dist = reinterpret_cast<float*>(sh_cand_idx + kWarpsPerBlock * K);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & (kWarpSize - 1);

    const int query_idx   = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool query_live = (query_idx < query_count);

    // Warp-private intermediate top-K result. It is distributed across the
    // 32 lanes as CHUNK contiguous elements per lane.
    float result_dist[CHUNK];
    int   result_idx [CHUNK];

    #pragma unroll
    for (int s = 0; s < CHUNK; ++s) {
        result_dist[s] = CUDART_INF_F;
        result_idx [s] = INT_MAX;
    }

    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && query_live) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Per-warp candidate buffer in shared memory.
    int   *const count_ptr = sh_counts + warp_id;
    int   *const cand_idx  = sh_cand_idx  + warp_id * K;
    float *const cand_dist = sh_cand_dist + warp_id * K;

    if (lane == 0) {
        *count_ptr = 0;
    }
    __syncwarp(kFullMask);

    // max_distance = current K-th smallest distance.
    // It stays +inf until the intermediate result becomes fully populated.
    float max_distance = CUDART_INF_F;

    // Iterate over the database in staged batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += kBatchPoints) {
        int batch_count = data_count - batch_start;
        if (batch_count > kBatchPoints) {
            batch_count = kBatchPoints;
        }

        // Cooperative block load into shared memory.
        #pragma unroll
        for (int it = 0; it < kBatchLoadIters; ++it) {
            const int local = tid + it * kBlockThreads;
            if (local < batch_count) {
                sh_data[local] = data[batch_start + local];
            }
        }
        __syncthreads();

        // Each live warp scans the staged batch 32 points at a time so that:
        // - every lane processes at most one candidate per iteration,
        // - the warp can ballot/filter those 32 distances,
        // - one warp-aggregated atomicAdd reserves output slots in the shared
        //   candidate buffer for all accepted lanes at once.
        if (query_live) {
            for (int local_base = 0; local_base < batch_count; local_base += kWarpSize) {
                const int local = local_base + lane;
                const bool valid = (local < batch_count);

                float dist = 0.0f;
                const int data_idx = batch_start + local;

                if (valid) {
                    const float2 p = sh_data[local];
                    dist = squared_l2_2d(qx, qy, p.x, p.y);
                }

                // The problem statement requests filtering only by distance.
                bool hit = valid && (dist < max_distance);
                unsigned hit_mask = __ballot_sync(kFullMask, hit);
                int hit_count = __popc(hit_mask);

                if (hit_count != 0) {
                    // Once max_distance is finite, proactively flushing the
                    // existing buffer when the current 32-point group would make
                    // it full is safe and profitable: after the flush, the
                    // threshold can only tighten, so the current group is
                    // immediately re-filtered against the new max_distance.
                    if (max_distance < CUDART_INF_F) {
                        int count = 0;
                        if (lane == 0) {
                            count = *count_ptr;
                        }
                        count = __shfl_sync(kFullMask, count, 0);

                        if (count > 0 && count + hit_count >= K) {
                            flush_candidate_buffer<K>(
                                result_dist, result_idx, max_distance,
                                count_ptr, cand_idx, cand_dist);

                            // Re-filter this same 32-point group against the
                            // updated threshold.
                            hit = valid && (dist < max_distance);
                            hit_mask = __ballot_sync(kFullMask, hit);
                            hit_count = __popc(hit_mask);
                        }
                    }

                    if (hit_count != 0) {
                        int base = 0;

                        // Warp-aggregated form of the required atomicAdd:
                        // one atomicAdd reserves hit_count contiguous slots,
                        // and the accepted lanes use their prefix rank in the
                        // hit mask to map into that reserved segment.
                        if (lane == 0) {
                            base = atomicAdd(count_ptr, hit_count);
                        }
                        base = __shfl_sync(kFullMask, base, 0);

                        if (hit) {
                            const unsigned lower_mask = (1u << lane) - 1u;
                            const int rank = __popc(hit_mask & lower_mask);
                            const int pos = base + rank;
                            cand_idx[pos]  = data_idx;
                            cand_dist[pos] = dist;
                        }
                        __syncwarp(kFullMask);

                        // If the buffer has just reached capacity, flush it.
                        // In practice this is mainly the initialization path
                        // before max_distance becomes finite, and the K=32 path.
                        if (base + hit_count == K) {
                            flush_candidate_buffer<K>(
                                result_dist, result_idx, max_distance,
                                count_ptr, cand_idx, cand_dist);
                        }
                    }
                }
            }
        }

        // All warps must finish using the staged batch before it is overwritten.
        __syncthreads();
    }

    // Flush the last partial candidate buffer, if any.
    if (query_live) {
        if (*count_ptr != 0) {
            flush_candidate_buffer<K>(
                result_dist, result_idx, max_distance,
                count_ptr, cand_idx, cand_dist);
        }

        // Store the final sorted top-K neighbors.
        const int out_base = query_idx * K;
        const int lane_base = lane * CHUNK;

        #pragma unroll
        for (int s = 0; s < CHUNK; ++s) {
            ResultPair &dst = result[out_base + lane_base + s];
            dst.first  = result_idx[s];
            dst.second = result_dist[s];
        }
    }
}

template <int K>
constexpr int shared_bytes() {
    return static_cast<int>(
        kBatchPoints * sizeof(float2) +
        kWarpsPerBlock * sizeof(int) +
        kWarpsPerBlock * K * sizeof(int) +
        kWarpsPerBlock * K * sizeof(float));
}

template <int K>
inline void launch_knn(const float2 *query,
                       int query_count,
                       const float2 *data,
                       int data_count,
                       ResultPair *result) {
    const int grid = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int smem = shared_bytes<K>();

    // The kernel is intentionally shared-memory heavy; request the larger
    // per-block dynamic shared-memory limit and prefer shared-memory carveout.
    cudaFuncSetAttribute(knn_kernel<K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
    cudaFuncSetAttribute(knn_kernel<K>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    knn_kernel<K><<<grid, kBlockThreads, smem>>>(
        query, query_count, data, data_count, result);
}

} // anonymous namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    // Keep the API asynchronous like a normal CUDA kernel launch.
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   launch_knn<32>  (query, query_count, data, data_count, result); break;
        case 64:   launch_knn<64>  (query, query_count, data, data_count, result); break;
        case 128:  launch_knn<128> (query, query_count, data, data_count, result); break;
        case 256:  launch_knn<256> (query, query_count, data, data_count, result); break;
        case 512:  launch_knn<512> (query, query_count, data, data_count, result); break;
        case 1024: launch_knn<1024>(query, query_count, data, data_count, result); break;
        default:
            // Per the problem statement, k is always valid.
            break;
    }
}