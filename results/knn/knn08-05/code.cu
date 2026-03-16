#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// Warp-specialized 2D k-NN for squared Euclidean distance.
//
// Design choices tuned for modern data-center GPUs (A100/H100 class):
//   * One warp computes one query.
//   * 256 threads/block = 8 warps/block, so each staged data batch is reused by 8 queries.
//   * 2048 data points/batch. With K=1024 this yields:
//         2048 * sizeof(float2)                          = 16 KiB
//         8 warps * 1024 * (sizeof(float)+sizeof(int))   = 64 KiB
//       total dynamic shared memory per block            = 80 KiB
//     which still allows two resident blocks on 164 KiB SMs.
//
// The current top-k list is distributed across warp registers exactly as requested:
// each lane owns K/32 consecutive elements.
// A warp-private shared-memory candidate buffer collects only points closer than the
// current k-th distance. Whenever the buffer fills (or overflows on the next ballot),
// it is merged into the register-resident top-k using the specified bitonic procedure.

constexpr int kWarpSize         = 32;
constexpr int kBlockThreads     = 256;
constexpr int kWarpsPerBlock    = kBlockThreads / kWarpSize;
constexpr int kBatchPoints      = 2048;
constexpr int kBatchFloat4      = kBatchPoints / 2;                 // 1 float4 == 2 float2
constexpr int kFloat4LoadsPerThread = kBatchFloat4 / kBlockThreads; // 1024 / 256 = 4
constexpr unsigned kFullMask    = 0xFFFFFFFFu;
constexpr int kInvalidIndex     = INT_MAX;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of 32.");
static_assert(kBatchPoints % kWarpSize == 0, "Batch size must be a multiple of 32.");
static_assert(kBatchPoints % 2 == 0, "Batch size must be even for float4 vector loads.");
static_assert(kBatchFloat4 % kBlockThreads == 0, "Full-batch float4 loads must divide evenly.");

// Device code writes through a POD with the same binary layout as std::pair<int,float>.
// This avoids depending on device-side std::pair constructors/operators.
struct PairStorage {
    int   first;
    float second;
};

static_assert(sizeof(PairStorage) == 8, "PairStorage must be tightly packed.");
static_assert(sizeof(std::pair<int, float>) == sizeof(PairStorage),
              "std::pair<int,float> must have the same layout as {int,float}.");

// Total ordering used by the sorting network.
// Distances are primary keys, indices are deterministic tie-breakers.
__device__ __forceinline__
bool pair_less(const float a_dist, const int a_idx,
               const float b_dist, const int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__
void compare_swap_pair(float& a_dist, int& a_idx,
                       float& b_dist, int& b_idx,
                       const bool ascending) {
    const bool do_swap = ascending
        ? pair_less(b_dist, b_idx, a_dist, a_idx)
        : pair_less(a_dist, a_idx, b_dist, b_idx);

    if (do_swap) {
        const float tmp_dist = a_dist;
        const int   tmp_idx  = a_idx;
        a_dist = b_dist;
        a_idx  = b_idx;
        b_dist = tmp_dist;
        b_idx  = tmp_idx;
    }
}

__device__ __forceinline__
float squared_l2(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

template <int K>
__device__ __forceinline__
float current_max_distance(const float (&topk_dist)[K / kWarpSize]) {
    constexpr int kItemsPerThread = K / kWarpSize;
    return __shfl_sync(kFullMask, topk_dist[kItemsPerThread - 1], kWarpSize - 1);
}

// One fully compile-time-unrolled bitonic compare/exchange pass.
// Because K and K/32 are powers of two, once J reaches the lane-local chunk size,
// the partner moves to another lane but keeps the same register offset. This exactly
// matches the prompt and allows cross-lane exchanges via shuffle-xor on one register
// per offset.
template <int KStage, int J, int K>
__device__ __forceinline__
void bitonic_pass(float (&dist)[K / kWarpSize],
                  int   (&idx )[K / kWarpSize],
                  const int lane) {
    constexpr int kItemsPerThread = K / kWarpSize;

    if constexpr (J < kItemsPerThread) {
        // Partner stays inside the same lane; compare/swap can be done locally.
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int partner = r ^ J;
            if (partner > r) {
                const int logical_i = lane * kItemsPerThread + r;
                const bool ascending = ((logical_i & KStage) == 0);
                compare_swap_pair(dist[r], idx[r], dist[partner], idx[partner], ascending);
            }
        }
    } else {
        // Partner sits in another lane but at the same intra-lane register offset.
        constexpr int kLaneMask = J / kItemsPerThread;

        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const float self_dist  = dist[r];
            const int   self_idx   = idx[r];
            const float other_dist = __shfl_xor_sync(kFullMask, self_dist, kLaneMask);
            const int   other_idx  = __shfl_xor_sync(kFullMask, self_idx,  kLaneMask);

            const int  logical_i = lane * kItemsPerThread + r;
            const bool ascending = ((logical_i & KStage) == 0);
            const bool lower     = ((logical_i & J) == 0);

            // ascending: lower keeps min, upper keeps max
            // descending: lower keeps max, upper keeps min
            const bool take_min  = (ascending == lower);
            const bool take_self = take_min
                ? !pair_less(other_dist, other_idx, self_dist, self_idx) // self <= other
                : !pair_less(self_dist, self_idx, other_dist, other_idx); // self >= other

            if (!take_self) {
                dist[r] = other_dist;
                idx[r]  = other_idx;
            }
        }
    }

    if constexpr (J > 1) {
        bitonic_pass<KStage, (J >> 1), K>(dist, idx, lane);
    }
}

template <int KStage, int K>
__device__ __forceinline__
void bitonic_stages(float (&dist)[K / kWarpSize],
                    int   (&idx )[K / kWarpSize],
                    const int lane) {
    bitonic_pass<KStage, (KStage >> 1), K>(dist, idx, lane);
    if constexpr (KStage < K) {
        bitonic_stages<(KStage << 1), K>(dist, idx, lane);
    }
}

template <int K>
__device__ __forceinline__
void bitonic_sort(float (&dist)[K / kWarpSize],
                  int   (&idx )[K / kWarpSize],
                  const int lane) {
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K & (K - 1)) == 0,   "K must be a power of two.");
    static_assert((K % kWarpSize) == 0, "K must be divisible by 32.");
    bitonic_stages<2, K>(dist, idx, lane);
}

// Merge the warp-private candidate buffer into the register-resident intermediate result.
// The implementation follows the requested procedure:
//
//   0. Invariant: intermediate result is already sorted ascending.
//   1. Swap buffer and intermediate result so that the buffer moves to registers.
//   2. Sort that register-resident buffer with bitonic sort.
//   3. Form a bitonic top-k merge: new[i] = min(buffer[i], result[k-i-1]).
//   4. Sort the merged bitonic sequence ascending to obtain the updated top-k.
//
// Shared memory is warp-private here, so warp synchronization is sufficient.
template <int K>
__device__ __forceinline__
float merge_candidate_buffer(float (&topk_dist)[K / kWarpSize],
                             int   (&topk_idx )[K / kWarpSize],
                             float* warp_buffer_dist,
                             int*   warp_buffer_idx,
                             const int buffer_count,
                             const int lane) {
    constexpr int kItemsPerThread = K / kWarpSize;

    // Step 1: swap shared buffer <-> register-resident top-k, while padding the
    // candidate buffer with +inf for the final (possibly partially full) merge.
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int pos = lane * kItemsPerThread + r;

        const float buf_dist = (pos < buffer_count) ? warp_buffer_dist[pos] : CUDART_INF_F;
        const int   buf_idx  = (pos < buffer_count) ? warp_buffer_idx[pos]  : kInvalidIndex;

        warp_buffer_dist[pos] = topk_dist[r];
        warp_buffer_idx[pos]  = topk_idx[r];

        topk_dist[r] = buf_dist;
        topk_idx[r]  = buf_idx;
    }
    __syncwarp(kFullMask);

    // Step 2: sort the (former) candidate buffer now stored in registers.
    bitonic_sort<K>(topk_dist, topk_idx, lane);

    // Step 3: build the bitonic top-k merge against the reversed old result.
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int pos = lane * kItemsPerThread + r;
        const int rev = K - 1 - pos;

        const float other_dist = warp_buffer_dist[rev];
        const int   other_idx  = warp_buffer_idx[rev];

        if (pair_less(other_dist, other_idx, topk_dist[r], topk_idx[r])) {
            topk_dist[r] = other_dist;
            topk_idx[r]  = other_idx;
        }
    }

    // Step 4: final ascending sort of the bitonic sequence.
    bitonic_sort<K>(topk_dist, topk_idx, lane);

    return current_max_distance<K>(topk_dist);
}

// Process one data point per lane:
//   * filter by the current max_distance,
//   * compact passing lanes with a warp ballot,
//   * append them to the shared candidate buffer,
//   * flush/merge the buffer when needed.
//
// If the current ballot would overflow the buffer, we first merge the existing buffer,
// then re-ballot the same tile against the tightened max_distance so that newly rejected
// points are not appended unnecessarily.
template <int K>
__device__ __forceinline__
void consider_candidate(const bool  valid,
                        const float candidate_distance,
                        const int   candidate_index,
                        float (&topk_dist)[K / kWarpSize],
                        int   (&topk_idx )[K / kWarpSize],
                        float* warp_buffer_dist,
                        int*   warp_buffer_idx,
                        int&   buffer_count,
                        float& max_distance,
                        const unsigned lane_mask_lt,
                        const int lane) {
    bool is_candidate = valid && (candidate_distance < max_distance);
    unsigned cand_mask = __ballot_sync(kFullMask, is_candidate);
    int cand_count = __popc(cand_mask);

    if (cand_count == 0) {
        return;
    }

    if (buffer_count + cand_count > K) {
        __syncwarp(kFullMask);
        max_distance = merge_candidate_buffer<K>(
            topk_dist, topk_idx,
            warp_buffer_dist, warp_buffer_idx,
            buffer_count, lane);
        buffer_count = 0;

        // Re-apply the tighter threshold produced by the merge.
        is_candidate = valid && (candidate_distance < max_distance);
        cand_mask = __ballot_sync(kFullMask, is_candidate);
        cand_count = __popc(cand_mask);

        if (cand_count == 0) {
            return;
        }
    }

    if (is_candidate) {
        const int local_rank = __popc(cand_mask & lane_mask_lt);
        const int pos = buffer_count + local_rank;
        warp_buffer_dist[pos] = candidate_distance;
        warp_buffer_idx[pos]  = candidate_index;
    }

    // buffer_count is warp-uniform; every lane carries the same replicated scalar.
    buffer_count += cand_count;
    __syncwarp(kFullMask);

    if (buffer_count == K) {
        max_distance = merge_candidate_buffer<K>(
            topk_dist, topk_idx,
            warp_buffer_dist, warp_buffer_idx,
            buffer_count, lane);
        buffer_count = 0;
    }
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 2)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                PairStorage* __restrict__ result) {
    constexpr int kItemsPerThread = K / kWarpSize;

    // Shared-memory layout:
    //   [0, kBatchPoints)                              -> staged float2 data batch
    //   [kBatchPoints, kBatchPoints + kWarpsPerBlock*K) -> candidate distances
    //   [..]                                            -> candidate indices
    extern __shared__ float4 smem_vec[];
    float2* const s_data = reinterpret_cast<float2*>(smem_vec);
    float*  const s_buffer_dist = reinterpret_cast<float*>(s_data + kBatchPoints);
    int*    const s_buffer_idx  = reinterpret_cast<int*>(s_buffer_dist + kWarpsPerBlock * K);

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active = (query_idx < query_count);

    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    // Register-resident intermediate result: K/32 consecutive elements per lane.
    float topk_dist[kItemsPerThread];
    int   topk_idx [kItemsPerThread];
    #pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        topk_dist[i] = CUDART_INF_F;
        topk_idx[i]  = kInvalidIndex;
    }

    // Load the query once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    if (active) {
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    float* const warp_buffer_dist = s_buffer_dist + warp_id * K;
    int*   const warp_buffer_idx  = s_buffer_idx  + warp_id * K;

    int   buffer_count = 0;          // warp-uniform count of buffered candidates
    float max_distance = CUDART_INF_F;

    for (int batch_base = 0; batch_base < data_count; batch_base += kBatchPoints) {
        const int remaining  = data_count - batch_base;
        const int batch_size = (remaining >= kBatchPoints) ? kBatchPoints : remaining;

        if (batch_size == kBatchPoints) {
            // Full batches dominate; load them with aligned float4 vector loads.
            // data comes from cudaMalloc and batch_base is always even here, so
            // reinterpret_cast<float4*> is 16-byte aligned.
            const float4* const g_data4 =
                reinterpret_cast<const float4*>(data) + (batch_base >> 1);

            #pragma unroll
            for (int i = 0; i < kFloat4LoadsPerThread; ++i) {
                const int vec_idx = threadIdx.x + i * kBlockThreads;
                smem_vec[vec_idx] = g_data4[vec_idx];
            }
        } else {
            // Scalar tail path for the final partial batch.
            for (int i = threadIdx.x; i < batch_size; i += kBlockThreads) {
                s_data[i] = data[batch_base + i];
            }
        }

        __syncthreads();

        if (active) {
            if (batch_size == kBatchPoints) {
                // Full-batch path: all lanes are valid in every 32-point tile.
                for (int tile = 0; tile < kBatchPoints; tile += kWarpSize) {
                    const int data_idx = batch_base + tile + lane;
                    const float2 p = s_data[tile + lane];
                    const float dist = squared_l2(qx, qy, p);

                    consider_candidate<K>(
                        true, dist, data_idx,
                        topk_dist, topk_idx,
                        warp_buffer_dist, warp_buffer_idx,
                        buffer_count, max_distance,
                        lane_mask_lt, lane);
                }
            } else {
                // Tail path: the final tile may be partially valid.
                for (int tile = 0; tile < batch_size; tile += kWarpSize) {
                    const int local_idx = tile + lane;
                    const bool valid = (local_idx < batch_size);

                    float dist = CUDART_INF_F;
                    if (valid) {
                        const float2 p = s_data[local_idx];
                        dist = squared_l2(qx, qy, p);
                    }

                    const int data_idx = batch_base + local_idx;

                    consider_candidate<K>(
                        valid, dist, data_idx,
                        topk_dist, topk_idx,
                        warp_buffer_dist, warp_buffer_idx,
                        buffer_count, max_distance,
                        lane_mask_lt, lane);
                }
            }
        }

        // The staged batch is shared by the whole block, so the block must finish
        // consuming it before the next load overwrites the cache.
        __syncthreads();
    }

    if (active) {
        // Flush any residual candidates left in the buffer after the last batch.
        if (buffer_count > 0) {
            (void)merge_candidate_buffer<K>(
                topk_dist, topk_idx,
                warp_buffer_dist, warp_buffer_idx,
                buffer_count, lane);
        }

        // Write the final sorted top-k neighbors back in the required row-major layout.
        PairStorage* const out =
            result
            + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K)
            + static_cast<std::size_t>(lane) * static_cast<std::size_t>(kItemsPerThread);

        #pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            out[i].first  = topk_idx[i];
            out[i].second = topk_dist[i];
        }
    }
}

template <int K>
constexpr std::size_t shared_bytes_for_k() {
    return static_cast<std::size_t>(kBatchPoints) * sizeof(float2)
         + static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(K) * (sizeof(float) + sizeof(int));
}

template <int K>
void launch_knn(const float2* query,
                const int query_count,
                const float2* data,
                const int data_count,
                PairStorage* result) {
    const std::size_t shared_bytes = shared_bytes_for_k<K>();

    // Full dynamic shared-memory opt-in is required for the larger K values.
    const cudaError_t attr0 = cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    const cudaError_t attr1 = cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    (void)attr0;
    (void)attr1;

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K><<<grid, block, shared_bytes>>>(
        query, query_count,
        data, data_count,
        result);
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

    // K is specialized at compile time so that:
    //   * each lane's K/32 slice is a fixed-size register array,
    //   * the bitonic sorting network is fully unrolled,
    //   * no extra device memory allocation is needed.
    PairStorage* const out = reinterpret_cast<PairStorage*>(result);

    switch (k) {
        case 32:   launch_knn<32>(query, query_count, data, data_count, out);   return;
        case 64:   launch_knn<64>(query, query_count, data, data_count, out);   return;
        case 128:  launch_knn<128>(query, query_count, data, data_count, out);  return;
        case 256:  launch_knn<256>(query, query_count, data, data_count, out);  return;
        case 512:  launch_knn<512>(query, query_count, data, data_count, out);  return;
        case 1024: launch_knn<1024>(query, query_count, data, data_count, out); return;
        default:
            // The prompt guarantees valid K values, so this path is unreachable.
            return;
    }
}