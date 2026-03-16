#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn2d_detail {

// Targeted tuning choices for A100/H100-class GPUs:
//
// * One warp computes one query, exactly as requested.
// * 8 warps/block = 256 threads/block:
//   - enough warps per block to amortize each shared-memory data tile across 8 queries,
//   - still enough blocks when query_count is only in the low thousands on 100+ SM GPUs.
// * ~80 KiB dynamic shared memory/block:
//   - large enough to hold the per-warp candidate buffers plus a multi-thousand-point data tile,
//   - small enough that two such blocks still fit on 164 KiB/SM parts like A100.
constexpr int kWarpSize         = 32;
constexpr int kWarpsPerBlock    = 8;
constexpr int kThreadsPerBlock  = kWarpSize * kWarpsPerBlock;
constexpr unsigned kFullMask    = 0xFFFFFFFFu;
constexpr std::size_t kSharedBudgetBytes = 80ull * 1024ull;

// The public API uses std::pair<int,float>.  Device-side std::pair support is
// implementation-dependent, so the kernel writes through a POD with the same
// compact 8-byte ABI that all mainstream CUDA-supported standard libraries use
// for std::pair<int,float>.
struct ResultPair {
    int   first;
    float second;
};
static_assert(sizeof(ResultPair) == sizeof(std::pair<int, float>),
              "Unexpected std::pair<int,float> ABI: expected a compact 8-byte {int,float} layout.");

// Squared Euclidean distance in 2D; no square root is taken because the problem
// explicitly requests squared distances.
__device__ __forceinline__ float sq_l2_2d(const float qx, const float qy,
                                          const float px, const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Bitonic sort of the warp-private intermediate top-k result.
//
// Storage layout:
//   global logical position p in [0, K)
//   is held by lane = p % 32, slot = p / 32.
//
// Because K is a power of two and a multiple of 32, the compare/swap network
// splits neatly into:
//   * stride < 32  -> partner lives in another lane, accessed with warp shuffles
//   * stride >= 32 -> partner lives in another local slot of the same lane
//
// The loops are intentionally fully unrolled.  This makes each slot index a
// compile-time constant after unrolling, which lets the compiler scalarize the
// per-lane arrays into registers instead of spilling them to local memory.
template <int K>
__device__ __forceinline__ void bitonic_sort_registers(float (&dist)[K / kWarpSize],
                                                       int (&idx)[K / kWarpSize]) {
    constexpr int ITEMS_PER_LANE = K / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride >= kWarpSize) {
                const int partner_delta = stride >> 5;

                #pragma unroll
                for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                    const int partner_t = t ^ partner_delta;
                    if (partner_t > t) {
                        const int  p         = lane + t * kWarpSize;
                        const bool ascending = ((p & size) == 0);

                        const float a_d = dist[t];
                        const int   a_i = idx[t];
                        const float b_d = dist[partner_t];
                        const int   b_i = idx[partner_t];

                        const bool swap = ascending ? (a_d > b_d) : (a_d < b_d);
                        if (swap) {
                            dist[t]         = b_d;
                            idx[t]          = b_i;
                            dist[partner_t] = a_d;
                            idx[partner_t]  = a_i;
                        }
                    }
                }
            } else {
                const bool lower = ((lane & stride) == 0);

                #pragma unroll
                for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                    const int  p         = lane + t * kWarpSize;
                    const bool ascending = ((p & size) == 0);

                    const float other_d = __shfl_xor_sync(kFullMask, dist[t], stride);
                    const int   other_i = __shfl_xor_sync(kFullMask, idx[t],  stride);

                    // In an ascending region the lower logical index keeps the min,
                    // the upper keeps the max.  In a descending region it is reversed.
                    const bool keep_min   = (lower == ascending);
                    const bool take_other = keep_min ? (other_d < dist[t]) : (other_d > dist[t]);

                    if (take_other) {
                        dist[t] = other_d;
                        idx[t]  = other_i;
                    }
                }
            }
        }
    }
}

// Bitonic sort of the shared-memory candidate buffer for one warp/query.
// Distances and indices are stored in separate arrays to avoid avoidable shared-
// memory bank conflicts from repeated 8-byte AoS accesses.
template <int K>
__device__ __forceinline__ void bitonic_sort_shared(float* dist, int* idx) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int i = lane; i < K; i += kWarpSize) {
                const int l = i ^ stride;
                if (l > i) {
                    const bool ascending = ((i & size) == 0);

                    const float a_d = dist[i];
                    const int   a_i = idx[i];
                    const float b_d = dist[l];
                    const int   b_i = idx[l];

                    const bool swap = ascending ? (a_d > b_d) : (a_d < b_d);
                    if (swap) {
                        dist[i] = b_d;
                        idx[i]  = b_i;
                        dist[l] = a_d;
                        idx[l]  = a_i;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Flushes one warp's shared candidate buffer into that warp's private
// intermediate top-k result.
//
// Required merge procedure:
//   0. intermediate result is already sorted ascending
//   1. sort the candidate buffer ascending with bitonic sort
//   2. form the bitonic sequence M[i] = min(buffer[i], result[K - i - 1])
//   3. bitonic-sort M ascending to obtain the updated intermediate result
//
// The candidate count is warp-private (one counter per warp), so only warp-local
// synchronization is necessary here.
template <int K>
__device__ __forceinline__ void flush_candidate_buffer(float* cand_dist,
                                                       int* cand_idx,
                                                       int* count_ptr,
                                                       float (&res_dist)[K / kWarpSize],
                                                       int (&res_idx)[K / kWarpSize],
                                                       float& max_distance) {
    constexpr int ITEMS_PER_LANE = K / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    volatile int* const count_v = reinterpret_cast<volatile int*>(count_ptr);

    int count = 0;
    if (lane == 0) {
        count = *count_v;
    }
    count = __shfl_sync(kFullMask, count, 0);

    if (count == 0) {
        return;
    }

    // Pad the not-yet-used tail with +INF so that a full K-element bitonic sort
    // still works for partially filled buffers.
    #pragma unroll
    for (int i = lane; i < K; i += kWarpSize) {
        if (i >= count) {
            cand_dist[i] = CUDART_INF_F;
            cand_idx[i]  = -1;
        }
    }
    __syncwarp(kFullMask);

    bitonic_sort_shared<K>(cand_dist, cand_idx);

    // Merge with the current result by pairing the ascending buffer with the
    // descending result.  With the cyclic lane/slot layout, logical position
    // K-1-(lane + 32*t) maps to source lane (31-lane), source slot (ITEMS-1-t).
    const int reverse_lane = (kWarpSize - 1) - lane;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int p = lane + t * kWarpSize;

        const float b_d = cand_dist[p];
        const int   b_i = cand_idx[p];

        const int   reverse_t = (ITEMS_PER_LANE - 1) - t;
        const float r_d = __shfl_sync(kFullMask, res_dist[reverse_t], reverse_lane);
        const int   r_i = __shfl_sync(kFullMask, res_idx[reverse_t],  reverse_lane);

        if (b_d < r_d) {
            res_dist[t] = b_d;
            res_idx[t]  = b_i;
        } else {
            res_dist[t] = r_d;
            res_idx[t]  = r_i;
        }
    }

    bitonic_sort_registers<K>(res_dist, res_idx);

    // The logical tail K-1 always lives in lane 31, local slot ITEMS_PER_LANE-1.
    max_distance = __shfl_sync(kFullMask, res_dist[ITEMS_PER_LANE - 1], kWarpSize - 1);

    if (lane == 0) {
        *count_v = 0;
    }
    __syncwarp(kFullMask);
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                ResultPair* __restrict__ result,
                int tile_elems) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");

    constexpr int ITEMS_PER_LANE = K / kWarpSize;

    // Dynamic shared memory layout, all in SoA form:
    //   [tile_x: tile_elems floats]
    //   [tile_y: tile_elems floats]
    //   [candidate_dist: kWarpsPerBlock * K floats]
    //   [candidate_idx : kWarpsPerBlock * K ints]
    //   [candidate_cnt : kWarpsPerBlock ints]
    extern __shared__ unsigned char smem_raw[];
    float* s_tile_x = reinterpret_cast<float*>(smem_raw);
    float* s_tile_y = s_tile_x + tile_elems;
    float* s_cand_dist = s_tile_y + tile_elems;
    int*   s_cand_idx  = reinterpret_cast<int*>(s_cand_dist + kWarpsPerBlock * K);
    int*   s_cand_cnt  = s_cand_idx + kWarpsPerBlock * K;

    const int lane     = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int query_id = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const bool active  = (query_id < query_count);

    float* const warp_cand_dist = s_cand_dist + warp * K;
    int*   const warp_cand_idx  = s_cand_idx  + warp * K;
    int*   const warp_cand_cnt  = s_cand_cnt  + warp;
    volatile int* const warp_cand_cnt_v = reinterpret_cast<volatile int*>(warp_cand_cnt);

    // Each warp owns exactly one candidate counter and one candidate buffer.
    if (lane == 0) {
        *warp_cand_cnt_v = 0;
    }

    // Load the query point once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active) {
        if (lane == 0) {
            const float2 q = query[query_id];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    // The intermediate top-k result is warp-private and distributed across the
    // 32 lanes.  Each lane owns K/32 pairs in registers.
    float res_dist[ITEMS_PER_LANE];
    int   res_idx[ITEMS_PER_LANE];
    float max_distance = CUDART_INF_F;

    // Initialization: use the first K data points to seed the private result.
    // Those K points are shared across all warps in the block, so we stage them
    // once into shared memory instead of having every warp fetch them separately.
    for (int i = static_cast<int>(threadIdx.x); i < K; i += kThreadsPerBlock) {
        const float2 p = data[i];
        s_tile_x[i] = p.x;
        s_tile_y[i] = p.y;
    }
    __syncthreads();

    if (active) {
        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int p = lane + t * kWarpSize;
            res_dist[t] = sq_l2_2d(qx, qy, s_tile_x[p], s_tile_y[p]);
            res_idx[t]  = p;
        }

        bitonic_sort_registers<K>(res_dist, res_idx);
        max_distance = __shfl_sync(kFullMask, res_dist[ITEMS_PER_LANE - 1], kWarpSize - 1);
    }
    __syncthreads();

    // Main batched scan over the remaining data points.
    for (int tile_begin = K; tile_begin < data_count; tile_begin += tile_elems) {
        int tile_count = data_count - tile_begin;
        if (tile_count > tile_elems) {
            tile_count = tile_elems;
        }

        // Whole block cooperatively loads the next tile to shared memory.
        for (int i = static_cast<int>(threadIdx.x); i < tile_count; i += kThreadsPerBlock) {
            const float2 p = data[tile_begin + i];
            s_tile_x[i] = p.x;
            s_tile_y[i] = p.y;
        }
        __syncthreads();

        if (active) {
            // Process the tile in warp-sized micro-batches so that at most 32 new
            // candidates appear at once.  Since K >= 32, a pre-flush on impending
            // overflow guarantees the current micro-batch always fits.
            for (int base = 0; base < tile_count; base += kWarpSize) {
                const int  local_pos = base + lane;
                const bool valid     = (local_pos < tile_count);

                float dist = 0.0f;
                bool candidate = false;

                if (valid) {
                    dist = sq_l2_2d(qx, qy, s_tile_x[local_pos], s_tile_y[local_pos]);
                    candidate = (dist < max_distance);
                }

                const unsigned mask = __ballot_sync(kFullMask, candidate);
                const int num_candidates = __popc(mask);

                if (num_candidates != 0) {
                    // The counter is warp-private, so a volatile read by lane 0 is
                    // sufficient here.  atomicAdd is still used for the actual slot
                    // reservation, as requested.
                    int current_count = 0;
                    if (lane == 0) {
                        current_count = *warp_cand_cnt_v;
                    }
                    current_count = __shfl_sync(kFullMask, current_count, 0);

                    // The candidate buffer is exactly size K, so flush before we
                    // would overflow it.
                    if (current_count + num_candidates > K) {
                        flush_candidate_buffer<K>(warp_cand_dist,
                                                  warp_cand_idx,
                                                  warp_cand_cnt,
                                                  res_dist,
                                                  res_idx,
                                                  max_distance);
                    }

                    // Reserve one contiguous segment for the whole warp's accepted
                    // lanes.  This is much faster than one atomicAdd per candidate
                    // while preserving the required atomic slot assignment semantics.
                    int start = 0;
                    if (lane == 0) {
                        start = atomicAdd(warp_cand_cnt, num_candidates);
                    }
                    start = __shfl_sync(kFullMask, start, 0);

                    if (candidate) {
                        const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const int rank = __popc(mask & lower_mask);
                        const int write_pos = start + rank;

                        warp_cand_dist[write_pos] = dist;
                        warp_cand_idx[write_pos]  = tile_begin + local_pos;
                    }

                    // Make the just-written candidates visible before a possible
                    // immediate flush when the buffer hits exactly K entries.
                    __syncwarp(kFullMask);

                    if (start + num_candidates == K) {
                        flush_candidate_buffer<K>(warp_cand_dist,
                                                  warp_cand_idx,
                                                  warp_cand_cnt,
                                                  res_dist,
                                                  res_idx,
                                                  max_distance);
                    }
                }
            }
        }

        // All warps must finish reading the current shared tile before the block
        // overwrites it with the next one.
        __syncthreads();
    }

    if (active) {
        // Merge the final partially filled candidate buffer, if any.
        flush_candidate_buffer<K>(warp_cand_dist,
                                  warp_cand_idx,
                                  warp_cand_cnt,
                                  res_dist,
                                  res_idx,
                                  max_distance);

        // Write the final sorted k-NN list back in row-major order:
        // result[query_id * K + j] = {index, squared_distance}
        const std::size_t out_base = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int logical_pos = lane + t * kWarpSize;
            result[out_base + static_cast<std::size_t>(logical_pos)] =
                ResultPair{res_idx[t], res_dist[t]};
        }
    }
}

template <int K>
inline std::size_t shared_bytes_for_tile(int tile_elems) {
    // 2 * tile_elems floats for x/y plus:
    //   K * kWarpsPerBlock floats for candidate distances,
    //   K * kWarpsPerBlock ints   for candidate indices,
    //   kWarpsPerBlock ints       for candidate counts.
    return static_cast<std::size_t>(2 * tile_elems + 2 * kWarpsPerBlock * K + kWarpsPerBlock) * sizeof(float);
}

template <int K>
inline void launch_knn_specialized(const float2* query,
                                   int query_count,
                                   const float2* data,
                                   int data_count,
                                   std::pair<int, float>* result) {
    int device = 0;
    (void)cudaGetDevice(&device);

    int max_optin_shared = 0;
    (void)cudaDeviceGetAttribute(&max_optin_shared,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                 device);

    const std::size_t max_shared = (max_optin_shared > 0)
        ? static_cast<std::size_t>(max_optin_shared)
        : kSharedBudgetBytes;

    const std::size_t fixed_bytes =
        static_cast<std::size_t>(2 * kWarpsPerBlock * K + kWarpsPerBlock) * sizeof(float);

    // Need at least K staged points for the initialization pass because the first
    // K data points are loaded cooperatively into shared memory and sorted into
    // the initial private top-k.
    const std::size_t min_budget = fixed_bytes + static_cast<std::size_t>(2 * K) * sizeof(float);

    std::size_t budget = (kSharedBudgetBytes < max_shared) ? kSharedBudgetBytes : max_shared;
    if (budget < min_budget) {
        budget = min_budget;
    }
    if (budget > max_shared) {
        budget = max_shared;
    }

    int tile_elems = static_cast<int>((budget - fixed_bytes) / (2 * sizeof(float)));
    tile_elems &= ~(kWarpSize - 1);
    if (tile_elems < K) {
        tile_elems = K;
    }

    std::size_t smem_bytes = shared_bytes_for_tile<K>(tile_elems);
    while (smem_bytes > max_shared && tile_elems > K) {
        tile_elems -= kWarpSize;
        smem_bytes = shared_bytes_for_tile<K>(tile_elems);
    }

    // On the target GPUs (A100/H100-class), the required footprint for all
    // supported K values fits.  We intentionally do not add a slow fallback path.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
    (void)cudaFuncSetCacheConfig(knn_kernel<K>, cudaFuncCachePreferShared);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    knn_kernel<K><<<blocks, kThreadsPerBlock, smem_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<ResultPair*>(result),
        tile_elems);
}

} // namespace knn2d_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Compile-time specialization on K is important here:
    // it keeps the per-lane K/32 intermediate result in registers and enables
    // full unrolling of the register-resident bitonic networks.
    switch (k) {
        case 32:   knn2d_detail::launch_knn_specialized<32>(query, query_count, data, data_count, result);   return;
        case 64:   knn2d_detail::launch_knn_specialized<64>(query, query_count, data, data_count, result);   return;
        case 128:  knn2d_detail::launch_knn_specialized<128>(query, query_count, data, data_count, result);  return;
        case 256:  knn2d_detail::launch_knn_specialized<256>(query, query_count, data, data_count, result);  return;
        case 512:  knn2d_detail::launch_knn_specialized<512>(query, query_count, data, data_count, result);  return;
        case 1024: knn2d_detail::launch_knn_specialized<1024>(query, query_count, data, data_count, result); return;
        default:
            // Per problem statement this path is unreachable because k is always a
            // supported power of two in [32, 1024].
            return;
    }
}