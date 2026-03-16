#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn_internal {

// NVIDIA warps are 32 lanes wide on all supported architectures.
constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
constexpr int INVALID_INDEX = -1;

// Tuned for modern data-center GPUs:
// - 128-thread blocks use a 1024-point batch so that even k=1024 keeps the
//   worst-case shared-memory footprint low enough for high residency.
// - 256/512-thread blocks use a 2048-point batch because their occupancy is
//   already limited by block size/shared memory on A100/H100, so a larger batch
//   reduces barrier overhead without hurting the intended occupancy target.
constexpr int THREADS_SMALL = 128;
constexpr int THREADS_MEDIUM = 256;
constexpr int THREADS_LARGE = 512;

constexpr int BATCH_SMALL = 1024;
constexpr int BATCH_LARGE = 2048;

template <int K, int THREADS_PER_BLOCK, int BATCH_POINTS>
struct KernelConfig {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "k must be a power of two in [32, 1024].");
    static_assert((THREADS_PER_BLOCK % WARP_SIZE) == 0,
                  "Block size must be a multiple of 32.");
    static_assert((BATCH_POINTS % WARP_SIZE) == 0,
                  "Batch size must be a multiple of 32.");
    static_assert((BATCH_POINTS % THREADS_PER_BLOCK) == 0,
                  "This implementation assumes an integer number of shared-memory loads per thread.");

    static constexpr int kWarpsPerBlock = THREADS_PER_BLOCK / WARP_SIZE;
    static constexpr int kItemsPerThread = K / WARP_SIZE;
    static constexpr int kLoadsPerThread = BATCH_POINTS / THREADS_PER_BLOCK;

    // Two float arrays are used for the shared-memory batch cache (SoA layout)
    // to avoid the 2-way bank conflicts that a float2 AoS layout would create.
    static constexpr std::size_t kBatchSharedBytes =
        static_cast<std::size_t>(BATCH_POINTS) * 2 * sizeof(float);

    // Each warp owns a private candidate buffer in shared memory:
    //   K distances + K indices.
    static constexpr std::size_t kDynamicSharedBytes =
        static_cast<std::size_t>(kWarpsPerBlock) * K * (sizeof(float) + sizeof(int));

    static constexpr std::size_t kTotalSharedBytes =
        kBatchSharedBytes + kDynamicSharedBytes;
};

// Warp-local bitonic sort over K elements distributed across the warp.
// Lane L stores K/32 consecutive elements.
// - If the comparator pair stays within a lane, we swap register values directly.
// - Otherwise, the two compared elements are at the same register index in two
//   different lanes, so we exchange them with shfl_xor.
template <int K>
__device__ __forceinline__ void bitonic_sort_warp(
    float (&dist)[K / WARP_SIZE],
    int (&idx)[K / WARP_SIZE]) {
    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

    const int lane = static_cast<int>(threadIdx.x) & (WARP_SIZE - 1);
    const int lane_base = lane * ITEMS_PER_THREAD;

    // Ties are intentionally left unspecified as permitted by the interface.
    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride < ITEMS_PER_THREAD) {
                // Intra-lane comparators: swap directly in registers.
                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r) {
                    const int partner = r ^ stride;
                    if (partner > r) {
                        const int global_i = lane_base + r;
                        const bool ascending = ((global_i & size) == 0);

                        const float a_dist = dist[r];
                        const float b_dist = dist[partner];

                        if ((ascending && (a_dist > b_dist)) ||
                            (!ascending && (a_dist < b_dist))) {
                            dist[r] = b_dist;
                            dist[partner] = a_dist;

                            const int a_idx = idx[r];
                            idx[r] = idx[partner];
                            idx[partner] = a_idx;
                        }
                    }
                }
            } else {
                // Inter-lane comparators: exchange the same register slot
                // between the two partner lanes.
                const int lane_xor = stride / ITEMS_PER_THREAD;

                #pragma unroll
                for (int r = 0; r < ITEMS_PER_THREAD; ++r) {
                    const float other_dist = __shfl_xor_sync(FULL_MASK, dist[r], lane_xor);
                    const int other_idx = __shfl_xor_sync(FULL_MASK, idx[r], lane_xor);

                    const int global_i = lane_base + r;
                    const bool ascending = ((global_i & size) == 0);
                    const bool lower_half = ((global_i & stride) == 0);
                    const bool keep_min = (ascending == lower_half);

                    if (keep_min) {
                        if (other_dist < dist[r]) {
                            dist[r] = other_dist;
                            idx[r] = other_idx;
                        }
                    } else {
                        if (dist[r] < other_dist) {
                            dist[r] = other_dist;
                            idx[r] = other_idx;
                        }
                    }
                }
            }
        }
    }
}

// Flush/merge path requested by the problem statement:
// 1. Swap the shared-memory candidate buffer with the register-resident result
//    so that the buffer is now in registers.
// 2. Bitonic-sort the buffer in ascending order.
// 3. Merge it with the previous intermediate result (now in shared memory) via
//    elementwise min(buffer[i], old_result[K-1-i]), producing a bitonic
//    sequence containing the smallest K elements.
// 4. Bitonic-sort again to restore the ascending top-k invariant.
template <int K>
__device__ __forceinline__ void merge_buffer_into_result(
    float* cand_dist_warp,
    int* cand_idx_warp,
    int& buffer_count,
    float (&result_dist)[K / WARP_SIZE],
    int (&result_idx)[K / WARP_SIZE],
    float& max_distance) {
    if (buffer_count == 0) {
        return;
    }

    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

    const int lane = static_cast<int>(threadIdx.x) & (WARP_SIZE - 1);
    const int lane_base = lane * ITEMS_PER_THREAD;

    // Ensure all prior candidate-buffer writes performed by the warp are visible
    // before the buffer is read back.
    __syncwarp(FULL_MASK);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int pos = lane_base + i;

        const float candidate_dist = (pos < buffer_count) ? cand_dist_warp[pos] : CUDART_INF_F;
        const int candidate_idx = (pos < buffer_count) ? cand_idx_warp[pos] : INVALID_INDEX;

        cand_dist_warp[pos] = result_dist[i];
        cand_idx_warp[pos] = result_idx[i];

        result_dist[i] = candidate_dist;
        result_idx[i] = candidate_idx;
    }

    // Shared memory now contains the old sorted intermediate result.
    __syncwarp(FULL_MASK);

    bitonic_sort_warp<K>(result_dist, result_idx);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int pos = lane_base + i;
        const int rev = K - 1 - pos;

        const float other_dist = cand_dist_warp[rev];
        if (other_dist < result_dist[i]) {
            result_dist[i] = other_dist;
            result_idx[i] = cand_idx_warp[rev];
        }
    }

    bitonic_sort_warp<K>(result_dist, result_idx);

    // The result is sorted ascending, so the K-th nearest neighbor is the last
    // globally stored element, owned by lane 31.
    const float lane_last =
        (lane == (WARP_SIZE - 1)) ? result_dist[ITEMS_PER_THREAD - 1] : 0.0f;
    max_distance = __shfl_sync(FULL_MASK, lane_last, WARP_SIZE - 1);

    buffer_count = 0;
}

template <int K, int THREADS_PER_BLOCK, int BATCH_POINTS>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result) {
    using Config = KernelConfig<K, THREADS_PER_BLOCK, BATCH_POINTS>;
    constexpr int WARPS_PER_BLOCK = Config::kWarpsPerBlock;
    constexpr int ITEMS_PER_THREAD = Config::kItemsPerThread;
    constexpr int LOADS_PER_THREAD = Config::kLoadsPerThread;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;

    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active = (query_idx < query_count);

    // Shared-memory batch cache in SoA form to avoid shared-memory bank conflicts
    // when an entire warp reads x/y for consecutive points.
    __shared__ float s_batch_x[BATCH_POINTS];
    __shared__ float s_batch_y[BATCH_POINTS];

    // Dynamic shared memory holds only the per-warp candidate buffers:
    // [warps * K distances][warps * K indices]
    extern __shared__ unsigned int s_dyn[];
    float* const cand_dist = reinterpret_cast<float*>(s_dyn);
    int* const cand_idx =
        reinterpret_cast<int*>(cand_dist + WARPS_PER_BLOCK * K);

    float* const cand_dist_warp = cand_dist + warp_id * K;
    int* const cand_idx_warp = cand_idx + warp_id * K;

    // Register-resident intermediate top-k, distributed across the warp.
    float result_dist[ITEMS_PER_THREAD];
    int result_idx[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        result_dist[i] = CUDART_INF_F;
        result_idx[i] = INVALID_INDEX;
    }

    int buffer_count = 0;
    float max_distance = CUDART_INF_F;

    // Load the query point once per warp and broadcast from lane 0.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);
    }

    // Scan the data set in shared-memory-backed batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        int valid_batch = data_count - batch_start;
        if (valid_batch > BATCH_POINTS) {
            valid_batch = BATCH_POINTS;
        }

        // Whole-block cooperative load into shared memory.
        #pragma unroll
        for (int it = 0; it < LOADS_PER_THREAD; ++it) {
            const int batch_pos = tid + it * THREADS_PER_BLOCK;
            if (batch_pos < valid_batch) {
                const float2 p = data[batch_start + batch_pos];
                s_batch_x[batch_pos] = p.x;
                s_batch_y[batch_pos] = p.y;
            }
        }

        __syncthreads();

        if (active) {
            // Process the shared-memory batch in warp-sized chunks so that every
            // lane computes one distance and ballot-compacts the accepted points.
            for (int chunk_base = 0; chunk_base < valid_batch; chunk_base += WARP_SIZE) {
                // The ballot can contribute at most 32 new candidates, so flush
                // early whenever fewer than 32 slots remain.
                if (buffer_count > (K - WARP_SIZE)) {
                    merge_buffer_into_result<K>(
                        cand_dist_warp, cand_idx_warp,
                        buffer_count, result_dist, result_idx, max_distance);
                }

                const int point_in_batch = chunk_base + lane;

                float dist = CUDART_INF_F;
                int data_index = INVALID_INDEX;
                bool keep = false;

                if (point_in_batch < valid_batch) {
                    data_index = batch_start + point_in_batch;
                    const float dx = qx - s_batch_x[point_in_batch];
                    const float dy = qy - s_batch_y[point_in_batch];
                    dist = fmaf(dx, dx, dy * dy);
                    keep = (dist < max_distance);
                }

                const unsigned keep_mask = __ballot_sync(FULL_MASK, keep);
                const int new_candidates = __popc(keep_mask);
                const int base = buffer_count;

                // Rank within the compacted ballot output.
                const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                const int rank = __popc(keep_mask & lower_mask);

                if (keep) {
                    const int pos = base + rank;
                    cand_dist_warp[pos] = dist;
                    cand_idx_warp[pos] = data_index;
                }

                buffer_count = base + new_candidates;

                if (buffer_count == K) {
                    merge_buffer_into_result<K>(
                        cand_dist_warp, cand_idx_warp,
                        buffer_count, result_dist, result_idx, max_distance);
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        if (buffer_count > 0) {
            merge_buffer_into_result<K>(
                cand_dist_warp, cand_idx_warp,
                buffer_count, result_dist, result_idx, max_distance);
        }

        // Write back the final sorted top-k.
        const int lane_base = lane * ITEMS_PER_THREAD;
        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K) +
            static_cast<std::size_t>(lane_base);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            result[out_base + i].first = result_idx[i];
            result[out_base + i].second = result_dist[i];
        }
    }
}

template <int K, int THREADS_PER_BLOCK, int BATCH_POINTS>
inline void launch_knn_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result) {
    using Config = KernelConfig<K, THREADS_PER_BLOCK, BATCH_POINTS>;
    constexpr int QUERIES_PER_BLOCK = Config::kWarpsPerBlock;

    const int blocks = (query_count + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;

    // Request the shared-memory carveout needed by the large-k specializations.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, THREADS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Config::kDynamicSharedBytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, THREADS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    (void)cudaFuncSetCacheConfig(
        knn_kernel<K, THREADS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncCachePreferShared);

    knn_kernel<K, THREADS_PER_BLOCK, BATCH_POINTS>
        <<<blocks, THREADS_PER_BLOCK, Config::kDynamicSharedBytes>>>(
            query, query_count, data, data_count, result);
}

template <int K>
inline void launch_knn_autotuned(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result) {
    int device = 0;
    int sm_count = 1;
    int max_optin_shared = 48 * 1024;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaDeviceGetAttribute(
        &max_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (sm_count <= 0) {
        sm_count = 1;
    }
    if (max_optin_shared <= 0) {
        max_optin_shared = 48 * 1024;
    }

    const std::size_t max_shared =
        static_cast<std::size_t>(max_optin_shared);

    constexpr std::size_t shared_small =
        KernelConfig<K, THREADS_SMALL, BATCH_SMALL>::kTotalSharedBytes;
    constexpr std::size_t shared_medium =
        KernelConfig<K, THREADS_MEDIUM, BATCH_LARGE>::kTotalSharedBytes;
    constexpr std::size_t shared_large =
        KernelConfig<K, THREADS_LARGE, BATCH_LARGE>::kTotalSharedBytes;

    const bool fit_small = (shared_small <= max_shared);
    const bool fit_medium = (shared_medium <= max_shared);
    const bool fit_large = (shared_large <= max_shared);

    // Heuristic:
    // - Larger blocks reuse every loaded data batch across more queries.
    // - Smaller blocks increase grid granularity when query_count is modest.
    // Use the largest tile that still gives roughly >= 75% SM coverage.
    const int threshold_large =
        (sm_count * (THREADS_LARGE / WARP_SIZE) * 3 + 3) / 4;
    const int threshold_medium =
        (sm_count * (THREADS_MEDIUM / WARP_SIZE) * 3 + 3) / 4;

    if (fit_large && query_count >= threshold_large) {
        launch_knn_specialized<K, THREADS_LARGE, BATCH_LARGE>(
            query, query_count, data, data_count, result);
    } else if (fit_medium && query_count >= threshold_medium) {
        launch_knn_specialized<K, THREADS_MEDIUM, BATCH_LARGE>(
            query, query_count, data, data_count, result);
    } else if (fit_small) {
        launch_knn_specialized<K, THREADS_SMALL, BATCH_SMALL>(
            query, query_count, data, data_count, result);
    } else if (fit_medium) {
        launch_knn_specialized<K, THREADS_MEDIUM, BATCH_LARGE>(
            query, query_count, data, data_count, result);
    } else {
        // This fallback should never be taken on the target A100/H100-class GPUs,
        // but 128 threads is the lowest shared-memory footprint among the tuned variants.
        launch_knn_specialized<K, THREADS_SMALL, BATCH_SMALL>(
            query, query_count, data, data_count, result);
    }
}

}  // namespace knn_internal

void run_knn(
    const float2* query,
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
            knn_internal::launch_knn_autotuned<32>(
                query, query_count, data, data_count, result);
            break;
        case 64:
            knn_internal::launch_knn_autotuned<64>(
                query, query_count, data, data_count, result);
            break;
        case 128:
            knn_internal::launch_knn_autotuned<128>(
                query, query_count, data, data_count, result);
            break;
        case 256:
            knn_internal::launch_knn_autotuned<256>(
                query, query_count, data, data_count, result);
            break;
        case 512:
            knn_internal::launch_knn_autotuned<512>(
                query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_internal::launch_knn_autotuned<1024>(
                query, query_count, data, data_count, result);
            break;
        default:
            // Input validity guarantees that this path is unreachable.
            break;
    }
}