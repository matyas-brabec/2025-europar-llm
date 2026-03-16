#include <cuda_runtime.h>
#include <utility>
#include <type_traits>
#include <cstddef>
#include <climits>

namespace {

// Exact 2D k-NN for large query_count/data_count on modern NVIDIA data-center GPUs.
//
// Design summary:
// - One warp computes one query, exactly as requested.
// - One 256-thread block holds 8 independent queries, so every data point loaded into
//   shared memory is reused by 8 warps before being evicted.
// - Data are processed in 1024-point shared-memory batches. 1024 is large enough to
//   seed the maximum supported k (1024) from the very first batch without an extra pass.
// - Each warp stores its exact top-k in warp-private shared memory. Since k can be 1024,
//   a register-only representation would be too expensive; shared memory is the right fit.
// - The top-k buffer is double-buffered because the update step is a parallel truncated merge.
// - The top-k arrays are padded by one slot after every full group of 32 logical items
//   (except the last group). This removes the pathological bank conflicts that otherwise
//   appear when each lane writes a contiguous K/32 segment during the merge.
// - For every 32 cached data points, each lane computes one candidate distance. If none of
//   the 32 candidates beats the current kth distance, the update is skipped entirely.
//   Otherwise the 32 candidates are sorted in-register with warp shuffles and merged in
//   parallel with the current sorted top-k using merge-path partitioning.
// - Distances are squared Euclidean distances, as required.
// - Ties are allowed to resolve arbitrarily by the problem statement. Internally we use
//   (distance, index) ordering only to make sorts and merges deterministic. Candidates at
//   exactly the current cutoff distance are intentionally ignored because any tie resolution
//   is acceptable.

constexpr int kWarpSize = 32;
constexpr int kWarpShift = 5;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Hyper-parameters chosen for A100/H100-class GPUs.
// 8 warps/block gives high shared-memory reuse. For k=1024 it yields the same 8 active
// warps/SM as two 4-warp blocks would, but doubles reuse of the cached data batch.
// 1024 points/batch keeps the data cache modest (8 KiB) while guaranteeing the first batch
// always contains the initial seed for every supported k.
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kBatchPoints = 1024;

static_assert(kBatchPoints >= 1024, "Batch size must cover the maximum supported k");

// The public API uses std::pair<int,float>. Device code stores into an equivalent POD
// because the standard-library type itself is not a convenient device-side storage type.
struct ResultEntry {
    int first;
    float second;
};

static_assert(sizeof(ResultEntry) == sizeof(std::pair<int, float>),
              "std::pair<int,float> must occupy exactly 8 bytes");
static_assert(alignof(ResultEntry) == alignof(std::pair<int, float>),
              "Unexpected std::pair<int,float> alignment");

template<int K>
struct TopLayout {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "K must be a power of two in [32, 1024]");
    static_assert(K % kWarpSize == 0, "K must be a multiple of 32");
    static constexpr int kItemsPerLane = K / kWarpSize;
    // One padding slot after each full group of 32 logical elements except the last one.
    static constexpr int kPaddedSize = K + (K / kWarpSize) - 1;
};

__device__ __forceinline__ bool pair_less(const float a_d, const int a_i,
                                          const float b_d, const int b_i) {
    return (a_d < b_d) || ((a_d == b_d) && (a_i < b_i));
}

__device__ __forceinline__ bool pair_greater(const float a_d, const int a_i,
                                             const float b_d, const int b_i) {
    return pair_less(b_d, b_i, a_d, a_i);
}

__device__ __forceinline__ int top_phys(const int logical) {
    // Shared-memory padding map:
    // [0..31]   -> [0..31]
    // [32..63]  -> [33..64]
    // [64..95]  -> [66..97]
    // ...
    return logical + (logical >> kWarpShift);
}

__device__ __forceinline__ float sq_l2(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

template<int THREADS_PER_BLOCK, int BATCH_POINTS>
__device__ __forceinline__ void load_data_batch(float2* sh_data,
                                                const float2* __restrict__ data,
                                                const int base,
                                                const int count) {
    constexpr int kLoadsPerThread = (BATCH_POINTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int it = 0; it < kLoadsPerThread; ++it) {
        const int idx = tid + it * THREADS_PER_BLOCK;
        if (idx < count) {
            sh_data[idx] = data[base + idx];
        }
    }
}

__device__ __forceinline__ void warp_bitonic_sort32(float& d, int& i) {
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & (kWarpSize - 1);

    #pragma unroll
    for (unsigned size = 2; size <= kWarpSize; size <<= 1) {
        #pragma unroll
        for (unsigned stride = size >> 1; stride > 0; stride >>= 1) {
            const float other_d = __shfl_xor_sync(kFullMask, d, stride);
            const int   other_i = __shfl_xor_sync(kFullMask, i, stride);

            const bool lane_low   = ((lane & stride) == 0u);
            const bool ascending  = ((lane & size) == 0u);
            const bool self_gt    = pair_greater(d, i, other_d, other_i);
            const bool take_other = lane_low ? (self_gt == ascending)
                                             : (self_gt != ascending);

            if (take_other) {
                d = other_d;
                i = other_i;
            }
        }
    }
}

template<int K>
__device__ __forceinline__ void warp_bitonic_sort_shared(float* d, int* i) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int logical = lane; logical < K; logical += kWarpSize) {
                const int partner = logical ^ stride;
                if (partner > logical) {
                    const int phys_a = top_phys(logical);
                    const int phys_b = top_phys(partner);

                    const float a_d = d[phys_a];
                    const int   a_i = i[phys_a];
                    const float b_d = d[phys_b];
                    const int   b_i = i[phys_b];

                    const bool ascending = ((logical & size) == 0);
                    const bool swap = ascending ? pair_greater(a_d, a_i, b_d, b_i)
                                                : pair_less(a_d, a_i, b_d, b_i);

                    if (swap) {
                        d[phys_a] = b_d;
                        i[phys_a] = b_i;
                        d[phys_b] = a_d;
                        i[phys_b] = a_i;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

template<int K>
__device__ __forceinline__ int merge_path_find_b(const float* src_d, const int* src_i,
                                                 const float* cand_d, const int* cand_i,
                                                 const int m,
                                                 const int diag) {
    int low = diag - K;
    if (low < 0) low = 0;
    int high = (diag < m) ? diag : m;

    // Lower_bound on the merge diagonal over B's rank.
    while (low < high) {
        const int mid = (low + high) >> 1;
        const int a = diag - mid;

        const bool right_ok =
            (a == 0) || (mid == m) ||
            !pair_less(cand_d[mid], cand_i[mid],
                       src_d[top_phys(a - 1)], src_i[top_phys(a - 1)]);

        if (right_ok) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

template<int K>
__device__ __forceinline__ float warp_merge_topk(const float* src_d, const int* src_i,
                                                 float* dst_d, int* dst_i,
                                                 const float* cand_d, const int* cand_i,
                                                 const int m) {
    constexpr int ITEMS_PER_LANE = TopLayout<K>::kItemsPerLane;
    const int lane = threadIdx.x & (kWarpSize - 1);

    // Each lane writes one contiguous logical segment of the first K merged outputs.
    const int out_begin = lane * ITEMS_PER_LANE;
    const int out_end   = out_begin + ITEMS_PER_LANE;

    const int b_begin = merge_path_find_b<K>(src_d, src_i, cand_d, cand_i, m, out_begin);
    const int b_end   = merge_path_find_b<K>(src_d, src_i, cand_d, cand_i, m, out_end);

    int a = out_begin - b_begin;
    int b = b_begin;

    const int a_end = out_end - b_end;
    const int b_end_local = b_end;

    float last_d = CUDART_INF_F;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        bool take_b = false;
        if (b < b_end_local) {
            if (a >= a_end) {
                take_b = true;
            } else {
                const int phys_a = top_phys(a);
                take_b = pair_less(cand_d[b], cand_i[b], src_d[phys_a], src_i[phys_a]);
            }
        }

        float out_d;
        int out_i;

        if (take_b) {
            out_d = cand_d[b];
            out_i = cand_i[b];
            ++b;
        } else {
            const int phys_a = top_phys(a);
            out_d = src_d[phys_a];
            out_i = src_i[phys_a];
            ++a;
        }

        const int phys_out = top_phys(out_begin + t);
        dst_d[phys_out] = out_d;
        dst_i[phys_out] = out_i;
        last_d = out_d;
    }

    __syncwarp(kFullMask);
    // Lane 31 owns the logical tail [K - K/32, K), so its last output is the new cutoff.
    return last_d;
}

template<int K>
__device__ __forceinline__ void process_cached_batch(const float qx, const float qy,
                                                     const float2* sh_data,
                                                     const int start_pos,
                                                     const int batch_count,
                                                     const int global_base,
                                                     float*& src_d, int*& src_i,
                                                     float*& dst_d, int*& dst_i,
                                                     float* cand_d, int* cand_i,
                                                     float& worst) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int chunk = start_pos; chunk < batch_count; chunk += kWarpSize) {
        const int local_idx = chunk + lane;

        float d = CUDART_INF_F;
        int i = INT_MAX;

        if (local_idx < batch_count) {
            const float2 p = sh_data[local_idx];
            d = sq_l2(qx, qy, p);
            i = global_base + local_idx;
        }

        // Strict "< worst" is sufficient because the problem allows arbitrary tie resolution.
        const unsigned any_better = __ballot_sync(kFullMask, d < worst);
        if (any_better) {
            // Update with 32 candidates simultaneously:
            // sort the 32 lane-local candidates, then perform one parallel truncated merge.
            warp_bitonic_sort32(d, i);

            const int m = __popc(__ballot_sync(kFullMask, d < worst));

            cand_d[lane] = d;
            cand_i[lane] = i;
            __syncwarp(kFullMask);

            const float new_tail = warp_merge_topk<K>(src_d, src_i, dst_d, dst_i,
                                                      cand_d, cand_i, m);

            float* tmp_d = src_d;
            src_d = dst_d;
            dst_d = tmp_d;

            int* tmp_i = src_i;
            src_i = dst_i;
            dst_i = tmp_i;

            worst = __shfl_sync(kFullMask, new_tail, kWarpSize - 1);
        }
    }
}

template<int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                ResultEntry* __restrict__ result) {
    static_assert(BATCH_POINTS >= K, "The first batch must contain the initial top-k seed");
    static_assert(BATCH_POINTS % kWarpSize == 0, "Batch size must be a multiple of 32");

    extern __shared__ unsigned char smem_raw[];

    float2* sh_data = reinterpret_cast<float2*>(smem_raw);

    constexpr int PADDED_K = TopLayout<K>::kPaddedSize;

    float* sh_top0_d = reinterpret_cast<float*>(sh_data + BATCH_POINTS);
    int*   sh_top0_i = reinterpret_cast<int*>(sh_top0_d + WARPS_PER_BLOCK * PADDED_K);
    float* sh_top1_d = reinterpret_cast<float*>(sh_top0_i + WARPS_PER_BLOCK * PADDED_K);
    int*   sh_top1_i = reinterpret_cast<int*>(sh_top1_d + WARPS_PER_BLOCK * PADDED_K);
    float* sh_cand_d = reinterpret_cast<float*>(sh_top1_i + WARPS_PER_BLOCK * PADDED_K);
    int*   sh_cand_i = reinterpret_cast<int*>(sh_cand_d + WARPS_PER_BLOCK * kWarpSize);

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> kWarpShift;
    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active = (query_idx < query_count);

    // One query load per warp; broadcast to all 32 lanes.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    float* src_d = sh_top0_d + warp_id * PADDED_K;
    int*   src_i = sh_top0_i + warp_id * PADDED_K;
    float* dst_d = sh_top1_d + warp_id * PADDED_K;
    int*   dst_i = sh_top1_i + warp_id * PADDED_K;
    float* cand_d = sh_cand_d + warp_id * kWarpSize;
    int*   cand_i = sh_cand_i + warp_id * kWarpSize;

    // Cooperative load of the first shared-memory batch.
    const int first_batch_count = (data_count < BATCH_POINTS) ? data_count : BATCH_POINTS;
    load_data_batch<WARPS_PER_BLOCK * kWarpSize, BATCH_POINTS>(sh_data, data, 0, first_batch_count);
    __syncthreads();

    float worst = CUDART_INF_F;

    if (active) {
        // Seed the exact top-k from the first K cached points instead of starting from +inf.
        // This avoids the expensive early phase where nearly every candidate would qualify.
        constexpr int ITEMS_PER_LANE = TopLayout<K>::kItemsPerLane;

        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int logical = t * kWarpSize + lane;
            const int phys = top_phys(logical);
            const float2 p = sh_data[logical];
            src_d[phys] = sq_l2(qx, qy, p);
            src_i[phys] = logical;
        }

        __syncwarp(kFullMask);
        warp_bitonic_sort_shared<K>(src_d, src_i);

        if (lane == 0) {
            worst = src_d[top_phys(K - 1)];
        }
        worst = __shfl_sync(kFullMask, worst, 0);

        // The first batch may contain more than K points; process the remainder normally.
        process_cached_batch<K>(qx, qy, sh_data,
                                K, first_batch_count, 0,
                                src_d, src_i, dst_d, dst_i,
                                cand_d, cand_i,
                                worst);
    }

    // Ensure the whole block has finished consuming the first batch before it is overwritten.
    __syncthreads();

    // Remaining batches.
    for (int base = BATCH_POINTS; base < data_count; base += BATCH_POINTS) {
        const int batch_count = ((data_count - base) < BATCH_POINTS) ? (data_count - base) : BATCH_POINTS;
        load_data_batch<WARPS_PER_BLOCK * kWarpSize, BATCH_POINTS>(sh_data, data, base, batch_count);
        __syncthreads();

        if (active) {
            process_cached_batch<K>(qx, qy, sh_data,
                                    0, batch_count, base,
                                    src_d, src_i, dst_d, dst_i,
                                    cand_d, cand_i,
                                    worst);
        }

        __syncthreads();
    }

    if (active) {
        constexpr int ITEMS_PER_LANE = TopLayout<K>::kItemsPerLane;
        ResultEntry* out = result + static_cast<size_t>(query_idx) * K;

        // Write back in a striped pattern so the warp performs coalesced global stores.
        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int logical = t * kWarpSize + lane;
            const int phys = top_phys(logical);
            out[logical].first  = src_i[phys];
            out[logical].second = src_d[phys];
        }
    }
}

template<int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
constexpr size_t knn_shared_bytes() {
    return
        static_cast<size_t>(BATCH_POINTS) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * TopLayout<K>::kPaddedSize * sizeof(float) +
        static_cast<size_t>(WARPS_PER_BLOCK) * TopLayout<K>::kPaddedSize * sizeof(int) +
        static_cast<size_t>(WARPS_PER_BLOCK) * TopLayout<K>::kPaddedSize * sizeof(float) +
        static_cast<size_t>(WARPS_PER_BLOCK) * TopLayout<K>::kPaddedSize * sizeof(int) +
        static_cast<size_t>(WARPS_PER_BLOCK) * kWarpSize * sizeof(float) +
        static_cast<size_t>(WARPS_PER_BLOCK) * kWarpSize * sizeof(int);
}

template<int K>
inline void launch_knn_case(const float2* query,
                            const int query_count,
                            const float2* data,
                            const int data_count,
                            ResultEntry* result) {
    constexpr size_t kSharedBytes = knn_shared_bytes<K, kWarpsPerBlock, kBatchPoints>();
    auto kernel = knn_kernel<K, kWarpsPerBlock, kBatchPoints>;

    // This kernel is intentionally shared-memory-heavy. Opt into the required dynamic
    // shared memory and bias the cache policy toward shared memory.
    (void)cudaFuncSetAttribute(kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(kSharedBytes));
    (void)cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);

    const int grid = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    kernel<<<grid, kThreadsPerBlock, kSharedBytes>>>(query, query_count, data, data_count, result);
}

}  // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    ResultEntry* out = reinterpret_cast<ResultEntry*>(result);

    // Compile-time specialization on K lets the compiler fully know:
    // - K/32 items per lane,
    // - the shared-memory layout,
    // - the sort/merge loop trip counts.
    switch (k) {
        case 32:   launch_knn_case<32>(query, query_count, data, data_count, out);   return;
        case 64:   launch_knn_case<64>(query, query_count, data, data_count, out);   return;
        case 128:  launch_knn_case<128>(query, query_count, data, data_count, out);  return;
        case 256:  launch_knn_case<256>(query, query_count, data, data_count, out);  return;
        case 512:  launch_knn_case<512>(query, query_count, data, data_count, out);  return;
        case 1024: launch_knn_case<1024>(query, query_count, data, data_count, out); return;
        default:   return;  // Unreachable for valid inputs.
    }
}