#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// Tuned for A100/H100-class GPUs:
// - 8 warps/block gives good data reuse while still keeping enough blocks in flight
//   when query_count is only in the low thousands.
// - 1024 cached data points/batch keeps the shared-memory footprint low enough to
//   allow two resident blocks/SM even for K=1024.
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kDataBatchPoints = 1024;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kThreadsPerBlock <= 1024, "Block size exceeds CUDA limit.");
static_assert(kDataBatchPoints % kWarpSize == 0, "Batch size must be warp-aligned.");

// Device-side POD view of the host-side std::pair<int,float> API type.
// The kernel only needs the raw ABI layout, not the std::pair interface itself.
struct alignas(alignof(std::pair<int, float>)) PairIFPod {
    int   first;
    float second;
};

static_assert(sizeof(PairIFPod)  == sizeof(std::pair<int, float>), "Unexpected std::pair<int,float> size.");
static_assert(alignof(PairIFPod) == alignof(std::pair<int, float>), "Unexpected std::pair<int,float> alignment.");

__host__ __device__ constexpr size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

template <int K>
constexpr size_t shared_bytes_for_kernel() {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "K must be a power of two in [32, 1024].");
    static_assert(K % kWarpSize == 0, "K must be divisible by warp size.");
    static_assert(K <= kDataBatchPoints, "The first shared batch must hold the bootstrap top-K.");

    size_t bytes = 0;

    bytes = align_up(bytes, alignof(float2));
    bytes += kDataBatchPoints * sizeof(float2);

    bytes = align_up(bytes, alignof(int));
    bytes += kWarpsPerBlock * sizeof(int);            // candidate counts, one per warp/query

    bytes = align_up(bytes, alignof(int));
    bytes += kWarpsPerBlock * K * sizeof(int);        // candidate indices

    bytes = align_up(bytes, alignof(float));
    bytes += kWarpsPerBlock * K * sizeof(float);      // candidate distances

    return bytes;
}

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    // Any tie break is acceptable; index tie-break gives a deterministic total order.
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool keep_self(float self_d, int self_i, float other_d, int other_i, bool take_min) {
    // Returns whether the current element should keep its own value when compared
    // against "other", either for the min side or the max side of a compare-exchange.
    return take_min ? !pair_less(other_d, other_i, self_d, self_i)
                    : !pair_less(self_d, self_i, other_d, other_i);
}

__device__ __forceinline__ void compare_swap_local(float& a_d, int& a_i,
                                                   float& b_d, int& b_i,
                                                   bool ascending) {
    const bool swap_needed = ascending ? pair_less(b_d, b_i, a_d, a_i)
                                       : pair_less(a_d, a_i, b_d, b_i);
    if (swap_needed) {
        const float td = a_d; a_d = b_d; b_d = td;
        const int   ti = a_i; a_i = b_i; b_i = ti;
    }
}

__device__ __forceinline__ float squared_l2(float qx, float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

__device__ __forceinline__ int warp_prefix_rank(unsigned mask, int lane) {
    const unsigned lower = (lane == 0) ? 0u : ((1u << lane) - 1u);
    return __popc(mask & lower);
}

// Bitonic sort of K elements distributed across one warp's registers.
// Mapping: global position p = slot * 32 + lane, where each lane owns K/32 slots.
template <int K>
__device__ __forceinline__ void bitonic_sort_warp(float (&dist)[K / kWarpSize],
                                                  int   (&idx )[K / kWarpSize]) {
    constexpr int ITEMS = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll
    for (int k_stage = 2; k_stage <= K; k_stage <<= 1) {
        #pragma unroll
        for (int j_stage = k_stage >> 1; j_stage > 0; j_stage >>= 1) {
            if (j_stage < kWarpSize) {
                // Inter-lane compare-exchange: partner differs only in the lane bits.
                #pragma unroll
                for (int t = 0; t < ITEMS; ++t) {
                    const int p = (t << 5) | lane;

                    const float self_d  = dist[t];
                    const int   self_i  = idx[t];
                    const float other_d = __shfl_xor_sync(kFullWarpMask, self_d, j_stage);
                    const int   other_i = __shfl_xor_sync(kFullWarpMask, self_i, j_stage);

                    const bool take_min = (((p & k_stage) == 0) ^ ((p & j_stage) != 0));
                    const bool keep     = keep_self(self_d, self_i, other_d, other_i, take_min);

                    dist[t] = keep ? self_d : other_d;
                    idx[t]  = keep ? self_i : other_i;
                }
            } else {
                // Intra-lane compare-exchange: partner differs only in the slot bits.
                const int offset = j_stage >> 5;

                #pragma unroll
                for (int t = 0; t < ITEMS; ++t) {
                    if ((t & offset) == 0) {
                        const int u = t ^ offset;
                        const int p = (t << 5) | lane;
                        const bool ascending = ((p & k_stage) == 0);
                        compare_swap_local(dist[t], idx[t], dist[u], idx[u], ascending);
                    }
                }
            }
            __syncwarp(kFullWarpMask);
        }
    }
}

// Bitonic sort of a warp-private shared-memory candidate buffer.
// The buffer is stored as two SoA arrays in shared memory. Unused entries are
// filled with +inf / INT_MAX before sorting so that short buffers can be sorted
// with the exact same network.
template <int K>
__device__ __forceinline__ void bitonic_sort_shared_buffer(int* cand_idx,
                                                           float* cand_dist,
                                                           int valid_count) {
    constexpr int ITEMS = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);

    valid_count = (valid_count < K) ? valid_count : K;

    if (valid_count < K) {
        #pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const int p = (t << 5) | lane;
            if (p >= valid_count) {
                cand_dist[p] = CUDART_INF_F;
                cand_idx[p]  = INT_MAX;
            }
        }
    }
    __syncwarp(kFullWarpMask);

    #pragma unroll
    for (int k_stage = 2; k_stage <= K; k_stage <<= 1) {
        #pragma unroll
        for (int j_stage = k_stage >> 1; j_stage > 0; j_stage >>= 1) {
            if (j_stage < kWarpSize) {
                #pragma unroll
                for (int t = 0; t < ITEMS; ++t) {
                    const int p = (t << 5) | lane;

                    const float self_d  = cand_dist[p];
                    const int   self_i  = cand_idx[p];
                    const float other_d = __shfl_xor_sync(kFullWarpMask, self_d, j_stage);
                    const int   other_i = __shfl_xor_sync(kFullWarpMask, self_i, j_stage);

                    const bool take_min = (((p & k_stage) == 0) ^ ((p & j_stage) != 0));
                    const bool keep     = keep_self(self_d, self_i, other_d, other_i, take_min);

                    cand_dist[p] = keep ? self_d : other_d;
                    cand_idx[p]  = keep ? self_i : other_i;
                }
            } else {
                const int offset = j_stage >> 5;

                #pragma unroll
                for (int t = 0; t < ITEMS; ++t) {
                    if ((t & offset) == 0) {
                        const int u = t ^ offset;
                        const int p = (t << 5) | lane;
                        const int q = (u << 5) | lane;

                        float a_d = cand_dist[p];
                        int   a_i = cand_idx[p];
                        float b_d = cand_dist[q];
                        int   b_i = cand_idx[q];

                        const bool ascending = ((p & k_stage) == 0);
                        compare_swap_local(a_d, a_i, b_d, b_i, ascending);

                        cand_dist[p] = a_d; cand_idx[p] = a_i;
                        cand_dist[q] = b_d; cand_idx[q] = b_i;
                    }
                }
            }
            __syncwarp(kFullWarpMask);
        }
    }
}

// Merge shared candidate buffer into the register-resident intermediate top-K.
// The implementation follows the required sequence exactly:
//   1) sort candidate buffer ascending,
//   2) build a K-element bitonic merged sequence by pairwise min(buffer[i], top[K-1-i]),
//   3) sort the merged sequence ascending.
// To keep register pressure low for K=1024, the merged sequence is materialized back
// into the shared candidate buffer and sorted there; the updated top-K is then loaded
// back into registers.
template <int K>
__device__ __forceinline__ float merge_shared_buffer_into_top(int* cand_idx,
                                                              float* cand_dist,
                                                              int valid_count,
                                                              int   (&top_idx )[K / kWarpSize],
                                                              float (&top_dist)[K / kWarpSize]) {
    constexpr int ITEMS = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int rev_lane = (kWarpSize - 1) - lane;

    valid_count = (valid_count < K) ? valid_count : K;

    bitonic_sort_shared_buffer<K>(cand_idx, cand_dist, valid_count);

    // Step 2: write the K-element bitonic merged sequence back into the shared buffer.
    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = (t << 5) | lane;

        const float buf_d = cand_dist[p];
        const int   buf_i = cand_idx[p];

        const float top_rev_d = __shfl_sync(kFullWarpMask, top_dist[ITEMS - 1 - t], rev_lane);
        const int   top_rev_i = __shfl_sync(kFullWarpMask, top_idx [ITEMS - 1 - t], rev_lane);

        const bool take_buffer = keep_self(buf_d, buf_i, top_rev_d, top_rev_i, true);
        cand_dist[p] = take_buffer ? buf_d : top_rev_d;
        cand_idx[p]  = take_buffer ? buf_i : top_rev_i;
    }
    __syncwarp(kFullWarpMask);

    // Step 3: sort the merged bitonic sequence ascending.
    bitonic_sort_shared_buffer<K>(cand_idx, cand_dist, K);

    // Reload the updated top-K into the warp-private register copy.
    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = (t << 5) | lane;
        top_dist[t] = cand_dist[p];
        top_idx[t]  = cand_idx[p];
    }

    // Distance of the K-th nearest neighbor.
    return __shfl_sync(kFullWarpMask, top_dist[ITEMS - 1], kWarpSize - 1);
}

// Process one already-cached shared-memory batch of data points.
// The candidate buffer is appended to with a warp-aggregated shared-memory atomicAdd:
// a single atomic reserves a contiguous segment for all candidate lanes in the current
// micro-batch, and each participating lane computes its own position via a warp prefix.
template <int K>
__device__ __forceinline__ void process_loaded_batch(
    const float2* s_batch,
    int batch_base,
    int begin_offset,
    int batch_count,
    float qx,
    float qy,
    int* cand_idx,
    float* cand_dist,
    int* cand_count,
    int   (&top_idx )[K / kWarpSize],
    float (&top_dist)[K / kWarpSize],
    float& max_distance) {

    const int lane = threadIdx.x & (kWarpSize - 1);

    // Process one warp-sized micro-batch at a time so that if the candidate buffer becomes
    // full, at most 31 candidates overflow and need to be reconsidered after the flush.
    for (int step = begin_offset; step < batch_count; step += kWarpSize) {
        const int local = step + lane;
        const bool valid = (local < batch_count);

        float dist = 0.0f;
        int data_index = batch_base + local;
        bool is_candidate = false;

        if (valid) {
            const float2 p = s_batch[local];
            dist = squared_l2(qx, qy, p);
            is_candidate = (dist < max_distance);
        }

        const unsigned candidate_mask = __ballot_sync(kFullWarpMask, is_candidate);
        const int num_candidates = __popc(candidate_mask);

        int base = 0;
        if (lane == 0 && num_candidates != 0) {
            base = atomicAdd(cand_count, num_candidates);
        }
        base = __shfl_sync(kFullWarpMask, base, 0);

        bool overflow = false;
        if (is_candidate) {
            const int rank = warp_prefix_rank(candidate_mask, lane);
            const int pos  = base + rank;
            if (pos < K) {
                cand_idx[pos]  = data_index;
                cand_dist[pos] = dist;
            } else {
                overflow = true;
            }
        }

        __syncwarp(kFullWarpMask);

        if (num_candidates != 0 && (base + num_candidates) >= K) {
            max_distance = merge_shared_buffer_into_top<K>(cand_idx, cand_dist, K, top_idx, top_dist);

            if (lane == 0) {
                *cand_count = 0;
            }
            __syncwarp(kFullWarpMask);

            // Reconsider just the candidates from this triggering micro-batch that did not fit.
            const bool retry = overflow && (dist < max_distance);
            const unsigned retry_mask = __ballot_sync(kFullWarpMask, retry);
            const int retry_count = __popc(retry_mask);

            int retry_base = 0;
            if (lane == 0 && retry_count != 0) {
                retry_base = atomicAdd(cand_count, retry_count);
            }
            retry_base = __shfl_sync(kFullWarpMask, retry_base, 0);

            if (retry) {
                const int retry_rank = warp_prefix_rank(retry_mask, lane);
                const int retry_pos  = retry_base + retry_rank;
                cand_idx[retry_pos]  = data_index;
                cand_dist[retry_pos] = dist;
            }

            __syncwarp(kFullWarpMask);
        }
    }
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                PairIFPod* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "K must be a power of two in [32, 1024].");
    static_assert(K % kWarpSize == 0, "K must be divisible by warp size.");
    static_assert(K <= kDataBatchPoints, "Bootstrap top-K must fit into the first shared batch.");

    constexpr int ITEMS = K / kWarpSize;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const int query_idx = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const bool active_query = (query_idx < query_count);

    extern __shared__ unsigned char smem_raw[];

    // Dynamic shared-memory layout:
    //   [batch float2 cache][warp candidate counts][warp candidate indices][warp candidate distances]
    size_t offset = 0;

    offset = align_up(offset, alignof(float2));
    float2* s_batch = reinterpret_cast<float2*>(smem_raw + offset);
    offset += kDataBatchPoints * sizeof(float2);

    offset = align_up(offset, alignof(int));
    int* s_counts = reinterpret_cast<int*>(smem_raw + offset);
    offset += kWarpsPerBlock * sizeof(int);

    offset = align_up(offset, alignof(int));
    int* s_cand_idx_all = reinterpret_cast<int*>(smem_raw + offset);
    offset += kWarpsPerBlock * K * sizeof(int);

    offset = align_up(offset, alignof(float));
    float* s_cand_dist_all = reinterpret_cast<float*>(smem_raw + offset);

    int*   cand_count = s_counts + warp;
    int*   cand_idx   = s_cand_idx_all  + warp * K;
    float* cand_dist  = s_cand_dist_all + warp * K;

    float qx = 0.0f;
    float qy = 0.0f;
    float max_distance = CUDART_INF_F;

    // Warp-private intermediate top-K, distributed across registers.
    int   top_idx [ITEMS];
    float top_dist[ITEMS];

    // Load the first shared batch once.
    // This batch serves two purposes:
    //   1) initialize the register-resident top-K from the first K data points,
    //   2) continue the main scan with the remainder of the same cached batch if K < batch size.
    const int initial_batch_count = (data_count < kDataBatchPoints) ? data_count : kDataBatchPoints;
    for (int i = tid; i < initial_batch_count; i += kThreadsPerBlock) {
        s_batch[i] = data[i];
    }
    __syncthreads();

    if (active_query) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
            *cand_count = 0;
        }
        qx = __shfl_sync(kFullWarpMask, qx, 0);
        qy = __shfl_sync(kFullWarpMask, qy, 0);

        // Bootstrap the sorted intermediate result from the first K data points.
        #pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const int p = (t << 5) | lane;  // p in [0, K)
            const float2 pt = s_batch[p];
            top_idx[t]  = p;
            top_dist[t] = squared_l2(qx, qy, pt);
        }

        bitonic_sort_warp<K>(top_dist, top_idx);
        max_distance = __shfl_sync(kFullWarpMask, top_dist[ITEMS - 1], kWarpSize - 1);
        __syncwarp(kFullWarpMask);

        // Continue processing the remainder of the already-cached first batch.
        process_loaded_batch<K>(s_batch,
                                /*batch_base=*/0,
                                /*begin_offset=*/K,
                                /*batch_count=*/initial_batch_count,
                                qx, qy,
                                cand_idx, cand_dist, cand_count,
                                top_idx, top_dist,
                                max_distance);
    }

    __syncthreads();

    // Process all remaining batches.
    for (int batch_base = kDataBatchPoints; batch_base < data_count; batch_base += kDataBatchPoints) {
        const int batch_count = ((data_count - batch_base) < kDataBatchPoints)
                              ? (data_count - batch_base)
                              :  kDataBatchPoints;

        for (int i = tid; i < batch_count; i += kThreadsPerBlock) {
            s_batch[i] = data[batch_base + i];
        }
        __syncthreads();

        if (active_query) {
            process_loaded_batch<K>(s_batch,
                                    batch_base,
                                    /*begin_offset=*/0,
                                    batch_count,
                                    qx, qy,
                                    cand_idx, cand_dist, cand_count,
                                    top_idx, top_dist,
                                    max_distance);
        }

        __syncthreads();
    }

    if (active_query) {
        int final_count = 0;
        if (lane == 0) {
            final_count = *cand_count;
        }
        final_count = __shfl_sync(kFullWarpMask, final_count, 0);
        final_count = (final_count < K) ? final_count : K;

        if (final_count > 0) {
            max_distance = merge_shared_buffer_into_top<K>(cand_idx, cand_dist, final_count, top_idx, top_dist);
            (void)max_distance;
        }

        const size_t out_base = static_cast<size_t>(query_idx) * static_cast<size_t>(K);

        #pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const int p = (t << 5) | lane;
            result[out_base + p].first  = top_idx[t];
            result[out_base + p].second = top_dist[t];
        }
    }
}

template <int K>
void launch_knn(const float2* query,
                int query_count,
                const float2* data,
                int data_count,
                std::pair<int, float>* result) {
    constexpr size_t smem_bytes = shared_bytes_for_kernel<K>();

    // Required for kernels whose dynamic shared-memory use exceeds the default 48 KiB.
    static const bool configured = []() {
        (void)cudaFuncSetAttribute(knn_kernel<K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(smem_bytes));
        (void)cudaFuncSetAttribute(knn_kernel<K>,
                                   cudaFuncAttributePreferredSharedMemoryCarveout,
                                   100);
        return true;
    }();
    (void)configured;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K><<<grid, block, smem_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIFPod*>(result));
}

} // anonymous namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Intentionally asynchronous: caller retains normal CUDA stream/error handling control.
    switch (k) {
        case 32:   launch_knn<32>  (query, query_count, data, data_count, result); break;
        case 64:   launch_knn<64>  (query, query_count, data, data_count, result); break;
        case 128:  launch_knn<128> (query, query_count, data, data_count, result); break;
        case 256:  launch_knn<256> (query, query_count, data, data_count, result); break;
        case 512:  launch_knn<512> (query, query_count, data, data_count, result); break;
        case 1024: launch_knn<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are specified to be valid; keep a no-op fallback.
    }
}