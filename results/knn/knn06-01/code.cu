#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace knn_detail {

// This implementation targets the stated workload regime:
// - one warp computes one query,
// - the whole block cooperatively tiles the data array into shared memory,
// - each warp keeps a private top-k in a striped register/local layout,
// - each warp also owns a shared-memory candidate buffer of size k and a shared counter,
// - whenever the candidate buffer fills (or would overflow), it is merged with the
//   current top-k using all 32 threads in the warp.
//
// The kernel returns squared Euclidean distances exactly as requested, i.e. no sqrt.

constexpr int kWarpThreads = 32;
constexpr unsigned kFullMask = 0xffffffffu;
// A100's opt-in per-block dynamic shared-memory limit; H100 is larger.
// Launch configurations below are chosen to remain valid for both.
constexpr int kA100OptinSharedBytes = 163840;

// Device result memory is declared as std::pair<int, float>* in the public interface.
// We use a layout-compatible POD for device-side writes.
struct ResultPair {
    int first;
    float second;
};

static_assert(sizeof(ResultPair) == sizeof(std::pair<int, float>),
              "ResultPair must match std::pair<int, float> size");
static_assert(alignof(ResultPair) == alignof(std::pair<int, float>),
              "ResultPair must match std::pair<int, float> alignment");

__device__ __forceinline__ bool pair_less(float a_dist, int a_idx, float b_dist, int b_idx) {
    // Distances define the ordering. The index is only a deterministic tie-breaker;
    // the problem statement does not constrain tie handling.
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ int prefix_rank(unsigned mask, int lane) {
    // Rank of this lane among active bits in mask, counting only lower-numbered lanes.
    const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
    return __popc(mask & lower_mask);
}

__device__ __forceinline__ float squared_l2(float qx, float qy, const float2 &p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

template <int N>
__device__ __forceinline__ void bitonic_sort_shared(float *dist_buf, int *idx_buf, int lane) {
    static_assert((N & (N - 1)) == 0, "bitonic sort size must be a power of two");

    // Shared-memory bitonic sort over N key/value pairs owned by one warp.
    // We deliberately keep this generic across all legal K values. Because each merge
    // processes at most 2K elements and K <= 1024, N <= 2048.
#pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll 1
            for (int i = lane; i < N; i += kWarpThreads) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const float a_dist = dist_buf[i];
                    const float b_dist = dist_buf[partner];
                    const int a_idx = idx_buf[i];
                    const int b_idx = idx_buf[partner];

                    const bool ascending = ((i & size) == 0);
                    const bool do_swap =
                        ascending ? pair_less(b_dist, b_idx, a_dist, a_idx)
                                  : pair_less(a_dist, a_idx, b_dist, b_idx);

                    if (do_swap) {
                        dist_buf[i] = b_dist;
                        idx_buf[i] = b_idx;
                        dist_buf[partner] = a_dist;
                        idx_buf[partner] = a_idx;
                    }
                }
            }
            // The buffer is warp-private, so warp-level synchronization is sufficient.
            __syncwarp();
        }
    }
}

template <int K>
__device__ __forceinline__ void merge_candidate_buffer(
    float *warp_dist,
    int *warp_idx,
    int *warp_count,
    float (&best_dist)[K / kWarpThreads],
    int (&best_idx)[K / kWarpThreads],
    int lane,
    float &max_distance,
    int &candidate_count) {
    static_assert(K % kWarpThreads == 0, "K must be divisible by the warp size");
    constexpr int kChunk = K / kWarpThreads;

    // Shared layout for this warp:
    //   [0, K)   : candidate buffer
    //   [K, 2K)  : temporary copy of the current top-k
    //
    // The current top-k is already globally sorted. The candidate buffer is not.
    // We therefore build a 2K-element array in shared memory and run a full bitonic sort.
    // The smallest K entries are then stored back into the register-striped top-k.
#pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        const int pos = lane + t * kWarpThreads;
        warp_dist[K + pos] = best_dist[t];
        warp_idx[K + pos] = best_idx[t];
    }

    // Pad the unused candidate slots with +inf / INT_MAX so they sort to the end.
    if (candidate_count < K) {
        for (int pos = candidate_count + lane; pos < K; pos += kWarpThreads) {
            warp_dist[pos] = CUDART_INF_F;
            warp_idx[pos] = INT_MAX;
        }
    }

    __syncwarp();
    bitonic_sort_shared<2 * K>(warp_dist, warp_idx, lane);

#pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        const int pos = lane + t * kWarpThreads;
        best_dist[t] = warp_dist[pos];
        best_idx[t] = warp_idx[pos];
    }

    if (lane == 0) {
        *warp_count = 0;
    }
    __syncwarp();

    candidate_count = 0;

    // The k-th nearest neighbor is the last element in the globally sorted top-k.
    float kth = 0.0f;
    if (lane == kWarpThreads - 1) {
        kth = best_dist[kChunk - 1];
    }
    max_distance = __shfl_sync(kFullMask, kth, kWarpThreads - 1);
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
constexpr std::size_t shared_bytes_for_kernel() {
    static_assert(BLOCK_THREADS % kWarpThreads == 0, "BLOCK_THREADS must be warp-aligned");
    static_assert(BATCH_POINTS % kWarpThreads == 0, "BATCH_POINTS must be a multiple of 32");

    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpThreads;

    return static_cast<std::size_t>(BATCH_POINTS) * sizeof(float2) +
           static_cast<std::size_t>(kWarpsPerBlock) * sizeof(int) +
           static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(2 * K) * sizeof(float) +
           static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(2 * K) * sizeof(int);
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
__global__ void knn_kernel(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    ResultPair *__restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024]");
    static_assert(K % kWarpThreads == 0, "K must be divisible by 32");
    static_assert(BLOCK_THREADS % kWarpThreads == 0, "BLOCK_THREADS must be warp-aligned");
    static_assert(BATCH_POINTS % kWarpThreads == 0, "BATCH_POINTS must be a multiple of 32");

    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpThreads;
    constexpr int kChunk = K / kWarpThreads;

    // Use an int4-backed extern shared allocation to guarantee a sufficiently aligned base.
    extern __shared__ int4 smem_base[];
    unsigned char *smem = reinterpret_cast<unsigned char *>(smem_base);

    // Dynamic shared-memory layout:
    //   [BATCH_POINTS x float2]              block-wide data tile
    //   [kWarpsPerBlock x int]               per-warp candidate counters
    //   [kWarpsPerBlock x (2K) x float]      per-warp distances (candidate + temp)
    //   [kWarpsPerBlock x (2K) x int]        per-warp indices   (candidate + temp)
    float2 *s_data = reinterpret_cast<float2 *>(smem);
    int *s_counts = reinterpret_cast<int *>(s_data + BATCH_POINTS);
    float *s_buf_dist = reinterpret_cast<float *>(s_counts + kWarpsPerBlock);
    int *s_buf_idx = reinterpret_cast<int *>(s_buf_dist + kWarpsPerBlock * (2 * K));

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpThreads;
    const int lane = tid & (kWarpThreads - 1);
    const int query_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active = (query_idx < query_count);

    if (lane == 0) {
        s_counts[warp_id] = 0;
    }

    // One query point per warp. Load once and broadcast via warp shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-private top-k in striped layout:
    // lane L owns global positions L, L+32, L+64, ...
    float best_dist[kChunk];
    int best_idx[kChunk];
#pragma unroll
    for (int t = 0; t < kChunk; ++t) {
        best_dist[t] = CUDART_INF_F;
        best_idx[t] = INT_MAX;
    }

    float max_distance = CUDART_INF_F;
    int candidate_count = 0;

    float *warp_dist = s_buf_dist + warp_id * (2 * K);
    int *warp_idx = s_buf_idx + warp_id * (2 * K);

    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        const int current = ((data_count - batch_start) < BATCH_POINTS) ? (data_count - batch_start) : BATCH_POINTS;
        const float2 *batch_data = data + batch_start;

        // Whole-block cooperative load of the next data tile.
        for (int i = tid; i < current; i += BLOCK_THREADS) {
            s_data[i] = batch_data[i];
        }
        __syncthreads();

        if (active) {
            // Each iteration processes one warp-wide micro-batch of up to 32 points.
            // We first compute a ballot of lanes that qualify under the current threshold.
            // If inserting all of them would overflow the fixed-size candidate buffer, we merge first,
            // recompute the qualification under the updated threshold, and then reserve space.
#pragma unroll 1
            for (int base = 0; base < current; base += kWarpThreads) {
                const int local = base + lane;
                const bool valid = (local < current);

                float dist = CUDART_INF_F;
                if (valid) {
                    dist = squared_l2(qx, qy, s_data[local]);
                }

                bool qualify = valid && (dist < max_distance);
                unsigned mask = __ballot_sync(kFullMask, qualify);
                int hits = __popc(mask);

                if (hits) {
                    if (candidate_count + hits > K) {
                        merge_candidate_buffer<K>(
                            warp_dist, warp_idx, &s_counts[warp_id],
                            best_dist, best_idx, lane, max_distance, candidate_count);

                        qualify = valid && (dist < max_distance);
                        mask = __ballot_sync(kFullMask, qualify);
                        hits = __popc(mask);
                    }

                    if (hits) {
                        // One atomicAdd per warp-wide micro-batch, not per candidate.
                        // This still satisfies the requirement while reducing atomic traffic.
                        int base_pos = 0;
                        if (lane == 0) {
                            base_pos = atomicAdd(&s_counts[warp_id], hits);
                        }
                        base_pos = __shfl_sync(kFullMask, base_pos, 0);

                        if (qualify) {
                            const int pos = base_pos + prefix_rank(mask, lane);
                            warp_dist[pos] = dist;
                            warp_idx[pos] = batch_start + local;
                        }

                        candidate_count = base_pos + hits;

                        // If the buffer is now full, merge immediately.
                        __syncwarp();
                        if (candidate_count == K) {
                            merge_candidate_buffer<K>(
                                warp_dist, warp_idx, &s_counts[warp_id],
                                best_dist, best_idx, lane, max_distance, candidate_count);
                        }
                    }
                }
            }
        }

        // The next tile reuses the same shared-memory region for s_data.
        __syncthreads();
    }

    // Flush any remaining candidates after the final batch.
    if (active && candidate_count > 0) {
        merge_candidate_buffer<K>(
            warp_dist, warp_idx, &s_counts[warp_id],
            best_dist, best_idx, lane, max_distance, candidate_count);
    }

    if (active) {
        ResultPair *out = result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
#pragma unroll
        for (int t = 0; t < kChunk; ++t) {
            const int pos = lane + t * kWarpThreads;
            out[pos] = ResultPair{best_idx[t], best_dist[t]};
        }
    }
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
inline void launch_knn_config(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    ResultPair *result) {
    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpThreads;
    constexpr std::size_t kSharedBytes = shared_bytes_for_kernel<K, BLOCK_THREADS, BATCH_POINTS>();

    static_assert(kSharedBytes <= static_cast<std::size_t>(kA100OptinSharedBytes),
                  "Configuration exceeds A100 shared-memory limit");

    const int grid_x = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    // Preferred tuning for the stated workload:
    // 256 threads/block = 8 query-warps per block.
    // This balances two competing goals:
    // 1) enough blocks to keep >100-SM parts busy when query_count is only in the low thousands,
    // 2) enough queries per block to amortize each shared-memory data tile.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>
        <<<grid_x, BLOCK_THREADS, kSharedBytes>>>(query, query_count, data, data_count, result);
}

}  // namespace knn_detail

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k) {
    if (query_count <= 0) {
        return;
    }

    auto *out = reinterpret_cast<knn_detail::ResultPair *>(result);

    // Launch configurations are chosen for modern datacenter GPUs with large shared memory.
    // Batch sizes are occupancy-aware:
    // - K <= 256 : 2720 points keeps 3 blocks/SM at K=256 on 164KB-SMEM parts.
    // - K == 512 : 2016 points keeps 2 blocks/SM.
    // - K == 1024: 1 block/SM is unavoidable, so the batch is enlarged to 4064 points
    //              to reduce synchronization and global-to-shared staging overhead.
    switch (k) {
        case 32:
            knn_detail::launch_knn_config<32, 256, 2720>(query, query_count, data, data_count, out);
            break;
        case 64:
            knn_detail::launch_knn_config<64, 256, 2720>(query, query_count, data, data_count, out);
            break;
        case 128:
            knn_detail::launch_knn_config<128, 256, 2720>(query, query_count, data, data_count, out);
            break;
        case 256:
            knn_detail::launch_knn_config<256, 256, 2720>(query, query_count, data, data_count, out);
            break;
        case 512:
            knn_detail::launch_knn_config<512, 256, 2016>(query, query_count, data, data_count, out);
            break;
        case 1024:
            knn_detail::launch_knn_config<1024, 256, 4064>(query, query_count, data, data_count, out);
            break;
        default:
            // Inputs are stated to be valid; this default is only a defensive fallback.
            break;
    }
}