#include <cuda_runtime.h>
#include <utility>
#include <cstddef>
#include <climits>

namespace knn_detail {

/*
  Warp-per-query 2D k-NN for power-of-two K in [32, 1024].

  Key design choices for modern data-center GPUs:
  - One warp computes one query.
  - One 256-thread block = 8 query warps. This gives good reuse of each staged data batch while
    keeping the worst-case shared-memory footprint comfortably below 96 KiB per block.
  - Each block stages 1024 data points (8 KiB) in shared memory. That batch size keeps the K=1024
    case at ~72 KiB/block total dynamic shared memory, so A100/H100 can still run multiple blocks/SM.
  - The intermediate top-K result is private to the warp and stored in an interleaved layout:
      lane L owns positions L, L+32, L+64, ...
    This is important because the merge step needs result[K-1-i]; with this layout the reversed
    position always lives in the opposite lane (31-L), so one warp shuffle per element is enough.
  - The candidate buffer lives in shared memory exactly as requested. Insertions use atomicAdd,
    but in a warp-aggregated form: one shared atomic reserves space for all qualifying lanes of the
    current 32-point step, then each qualifying lane computes its own offset.
*/

constexpr int KNN_BLOCK_THREADS    = 256;
constexpr int KNN_WARPS_PER_BLOCK  = KNN_BLOCK_THREADS / 32;
constexpr int KNN_BATCH_POINTS     = 1024;
constexpr unsigned FULL_MASK       = 0xffffffffu;

static_assert(KNN_BLOCK_THREADS % 32 == 0, "Block size must be a multiple of warp size.");
static_assert(KNN_BATCH_POINTS % 32 == 0, "Batch size must be a multiple of warp size.");

template <int K>
constexpr std::size_t knn_shared_bytes() {
    return static_cast<std::size_t>(KNN_BATCH_POINTS) * sizeof(float2) +
           static_cast<std::size_t>(KNN_WARPS_PER_BLOCK) * sizeof(int) +
           static_cast<std::size_t>(KNN_WARPS_PER_BLOCK) * static_cast<std::size_t>(K) * (sizeof(int) + sizeof(float));
}

static_assert(knn_shared_bytes<1024>() <= 96u * 1024u,
              "Worst-case shared-memory footprint should stay below 96 KiB/block.");

__device__ __forceinline__ float distance_sq_2d(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ bool pair_less(const float da, const int ia,
                                          const float db, const int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

/*
  Cooperative bitonic sort of K pairs (distance, index) stored in a warp-private slice of shared memory.
  Only one warp touches that slice, so __syncwarp() is sufficient between compare-exchange rounds.
*/
template <int K>
__device__ __forceinline__ void warp_bitonic_sort_shared(int* const buf_idx,
                                                         float* const buf_dist) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    constexpr int ITEMS_PER_LANE = K / 32;
    const int lane = threadIdx.x & 31;

    for (int stage = 2; stage <= K; stage <<= 1) {
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
            for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                const int i = lane + (t << 5);
                const int l = i ^ stride;

                if (l > i) {
                    const float di = buf_dist[i];
                    const float dl = buf_dist[l];
                    const int   ii = buf_idx[i];
                    const int   il = buf_idx[l];

                    const bool ascending = ((i & stage) == 0);
                    const bool do_swap = ascending ? pair_less(dl, il, di, ii)
                                                   : pair_less(di, ii, dl, il);

                    if (do_swap) {
                        buf_dist[i] = dl;
                        buf_dist[l] = di;
                        buf_idx[i]  = il;
                        buf_idx[l]  = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

/*
  Copy a sorted shared-memory buffer back into the warp-private interleaved top-K storage.
  The returned value is the current K-th neighbor distance, broadcast from lane 31.
*/
template <int K>
__device__ __forceinline__ float load_sorted_buffer_to_private(const int* const buf_idx,
                                                               const float* const buf_dist,
                                                               int (&best_idx)[K / 32],
                                                               float (&best_dist)[K / 32]) {
    constexpr int ITEMS_PER_LANE = K / 32;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + (t << 5);
        best_idx[t]   = buf_idx[pos];
        best_dist[t]  = buf_dist[pos];
    }

    const float lane_local_last = best_dist[ITEMS_PER_LANE - 1];
    return __shfl_sync(FULL_MASK, lane_local_last, 31);
}

/*
  Seed the intermediate result with the first K data points. This is a one-time cost per query;
  the main sweep over the very large dataset uses block-staged shared-memory batches.
*/
template <int K>
__device__ __forceinline__ float initialize_best_from_prefix(const float2* __restrict__ data,
                                                             const float qx,
                                                             const float qy,
                                                             int* const buf_idx,
                                                             float* const buf_dist,
                                                             int (&best_idx)[K / 32],
                                                             float (&best_dist)[K / 32]) {
    constexpr int ITEMS_PER_LANE = K / 32;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + (t << 5);
        const float2 p = data[pos];
        buf_idx[pos]  = pos;
        buf_dist[pos] = distance_sq_2d(qx, qy, p);
    }

    __syncwarp();
    warp_bitonic_sort_shared<K>(buf_idx, buf_dist);
    return load_sorted_buffer_to_private<K>(buf_idx, buf_dist, best_idx, best_dist);
}

/*
  Merge the shared-memory candidate buffer into the warp-private sorted intermediate result.

  Required procedure:
    1) Sort the buffer ascending with Bitonic Sort.
    2) Form the bitonic "small half" by taking min(buffer[i], result[K-1-i]).
    3) Sort that bitonic sequence ascending with Bitonic Sort.

  The merged result is written back to the private best_* arrays.
*/
template <int K>
__device__ __forceinline__ float flush_candidate_buffer(int* const buf_idx,
                                                        float* const buf_dist,
                                                        int* const buf_count,
                                                        const int candidate_count,
                                                        int (&best_idx)[K / 32],
                                                        float (&best_dist)[K / 32]) {
    constexpr int ITEMS_PER_LANE = K / 32;
    const int lane = threadIdx.x & 31;

    // Fill the unused tail with sentinels so the full-K sort/merge is always valid.
    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + (t << 5);
        if (pos >= candidate_count) {
            buf_idx[pos]  = INT_MAX;
            buf_dist[pos] = CUDART_INF_F;
        }
    }

    __syncwarp();
    warp_bitonic_sort_shared<K>(buf_idx, buf_dist);

    // Build the bitonic sequence of the smallest K elements directly back into the buffer.
    // With the interleaved private layout, result[K-1-(lane+32*t)] lives in lane (31-lane),
    // slot (ITEMS_PER_LANE-1-t).
    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + (t << 5);

        float bd = buf_dist[pos];
        int   bi = buf_idx[pos];

        const float rd = __shfl_sync(FULL_MASK, best_dist[ITEMS_PER_LANE - 1 - t], 31 - lane);
        const int   ri = __shfl_sync(FULL_MASK, best_idx [ITEMS_PER_LANE - 1 - t], 31 - lane);

        if (pair_less(rd, ri, bd, bi)) {
            bd = rd;
            bi = ri;
        }

        buf_dist[pos] = bd;
        buf_idx[pos]  = bi;
    }

    __syncwarp();
    warp_bitonic_sort_shared<K>(buf_idx, buf_dist);

    const float max_distance = load_sorted_buffer_to_private<K>(buf_idx, buf_dist, best_idx, best_dist);

    if (lane == 0) {
        *buf_count = 0;
    }
    __syncwarp();

    return max_distance;
}

template <int K>
__global__ void knn_kernel(const float2* __restrict__ query,
                           const int query_count,
                           const float2* __restrict__ data,
                           const int data_count,
                           std::pair<int, float>* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    constexpr int ITEMS_PER_LANE = K / 32;

    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int query_idx     = static_cast<int>(blockIdx.x) * KNN_WARPS_PER_BLOCK + warp_in_block;
    const bool active       = (query_idx < query_count);

    // Dynamic shared-memory layout:
    //   [ staged data batch | per-warp counts | per-warp candidate indices | per-warp candidate distances ]
    extern __shared__ __align__(16) unsigned char shared_raw[];
    float2* const sh_data     = reinterpret_cast<float2*>(shared_raw);
    int*    const sh_counts   = reinterpret_cast<int*>(sh_data + KNN_BATCH_POINTS);
    int*    const sh_buf_idx  = sh_counts + KNN_WARPS_PER_BLOCK;
    float*  const sh_buf_dist = reinterpret_cast<float*>(sh_buf_idx + KNN_WARPS_PER_BLOCK * K);

    int*   const buf_idx   = sh_buf_idx  + warp_in_block * K;
    float* const buf_dist  = sh_buf_dist + warp_in_block * K;
    int*   const buf_count = sh_counts   + warp_in_block;

    if (lane == 0) {
        *buf_count = 0;
    }
    __syncwarp();

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Private, warp-distributed intermediate result.
    int   best_idx[ITEMS_PER_LANE];
    float best_dist[ITEMS_PER_LANE];

    float max_distance = CUDART_INF_F;

    // Local shadow of the shared count. Only this warp updates its own counter, so the shadow stays
    // exact and avoids a shared-memory read before every 32-point step. atomicAdd is still used for
    // the actual reservation in the shared candidate buffer, as required.
    int candidate_count = 0;

    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    if (active) {
        max_distance = initialize_best_from_prefix<K>(data, qx, qy, buf_idx, buf_dist, best_idx, best_dist);
        if (lane == 0) {
            *buf_count = 0;
        }
        __syncwarp();
    }

    // Main sweep over the remaining points. The first K points already seeded the intermediate result.
    for (int batch_start = K; batch_start < data_count; batch_start += KNN_BATCH_POINTS) {
        int current_batch = data_count - batch_start;
        if (current_batch > KNN_BATCH_POINTS) {
            current_batch = KNN_BATCH_POINTS;
        }

        // Whole block stages the next batch into shared memory.
        for (int i = threadIdx.x; i < current_batch; i += KNN_BLOCK_THREADS) {
            sh_data[i] = data[batch_start + i];
        }
        __syncthreads();

        if (active) {
            // Process the staged batch in warp-sized steps of 32 points.
            // Pre-flushing if there is not enough room for the next step guarantees that the fixed-size
            // shared candidate buffer never overflows, without allocating any extra device memory.
            for (int p_base = 0; p_base < current_batch; p_base += 32) {
                int step_size = current_batch - p_base;
                if (step_size > 32) {
                    step_size = 32;
                }

                if (candidate_count > K - step_size) {
                    max_distance = flush_candidate_buffer<K>(buf_idx, buf_dist, buf_count, candidate_count,
                                                             best_idx, best_dist);
                    candidate_count = 0;
                }

                const int p = p_base + lane;
                const bool valid = (p < current_batch);

                float dist = 0.0f;
                if (valid) {
                    dist = distance_sq_2d(qx, qy, sh_data[p]);
                }

                const bool qualifies = valid && (dist < max_distance);
                const unsigned ballot = __ballot_sync(FULL_MASK, qualifies);
                const int n = __popc(ballot);

                if (n != 0) {
                    int base = 0;
                    if (lane == 0) {
                        base = atomicAdd(buf_count, n);
                    }
                    base = __shfl_sync(FULL_MASK, base, 0);

                    if (qualifies) {
                        const int offset = __popc(ballot & lane_mask_lt);
                        const int pos = base + offset;
                        buf_idx[pos]  = batch_start + p;
                        buf_dist[pos] = dist;
                    }

                    candidate_count += n;
                }

                __syncwarp();

                if (candidate_count == K) {
                    max_distance = flush_candidate_buffer<K>(buf_idx, buf_dist, buf_count, candidate_count,
                                                             best_idx, best_dist);
                    candidate_count = 0;
                }
            }
        }

        // All warps must be done reading sh_data before the next batch overwrites it.
        __syncthreads();
    }

    if (active) {
        if (candidate_count != 0) {
            max_distance = flush_candidate_buffer<K>(buf_idx, buf_dist, buf_count, candidate_count,
                                                     best_idx, best_dist);
            (void)max_distance;
        }

        // The private result is already sorted ascending, so writing it out preserves the required order.
        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int pos = lane + (t << 5);
            result[out_base + static_cast<std::size_t>(pos)].first  = best_idx[t];
            result[out_base + static_cast<std::size_t>(pos)].second = best_dist[t];
        }
    }
}

template <int K>
void launch_knn_template(const float2* query,
                         const int query_count,
                         const float2* data,
                         const int data_count,
                         std::pair<int, float>* result) {
    constexpr std::size_t smem_bytes = knn_shared_bytes<K>();

    const dim3 block(KNN_BLOCK_THREADS);
    const dim3 grid((query_count + KNN_WARPS_PER_BLOCK - 1) / KNN_WARPS_PER_BLOCK);

    // This kernel is intentionally shared-memory heavy; prefer the maximum shared carveout.
    cudaFuncSetAttribute(knn_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(knn_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    knn_kernel<K><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);
}

} // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:
            knn_detail::launch_knn_template<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_template<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_template<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_template<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_template<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_template<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees valid k, so this path is not expected.
            break;
    }
}