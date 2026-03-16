#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn2d_detail {

/*
Exact brute-force 2D k-NN tuned for modern data-center GPUs.

Key design points:
- One warp computes one query.
- A block stages a large batch of data points in shared memory and all query warps in
  the block reuse that batch. This is the main bandwidth optimization because the data
  set is large and is scanned in full for every query.
- Each warp owns a private intermediate top-k structure stored in shared memory:
  a max-heap of (distance, index). The heap root is the current kth-best distance and
  therefore the pruning threshold.
- Only lane 0 mutates the heap. This is intentional: once the heap is full, the
  threshold quickly rejects almost all points in typical large-data k-NN workloads,
  so the serialized heap maintenance cost is amortized while all 32 lanes keep the
  distance pipeline full.
- k is specialized at compile time for the only valid values:
  {32, 64, 128, 256, 512, 1024}. This avoids dynamic-k control flow in the hot path.
- No extra device memory is allocated; all temporary storage is dynamic shared memory.

Shared-memory budget:
- We reserve 80 KiB per block. On A100/H100-class parts this still allows two resident
  blocks per SM with an opt-in shared-memory carveout, while giving each block a large
  cached data tile.
- A cached point consumes 8 bytes in shared memory (x and y as two floats).
- A heap slot also consumes 8 bytes (distance float + index int).
- For a chosen number of warps per block W, tile_points = 80KiB/8 - W*K.
  We choose the largest practical W for each K to maximize tile reuse across queries.
*/

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kSharedBudgetBytes = 80 * 1024;
constexpr int kSharedBudgetUnits = kSharedBudgetBytes / 8;  // 8 B per cached point or heap entry.

using result_pair = std::pair<int, float>;

template <int K, int WarpsPerBlock>
struct KnnConfig {
    static_assert(K == 32 || K == 64 || K == 128 || K == 256 || K == 512 || K == 1024,
                  "Unsupported K.");
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 32, "Invalid warp count.");
    static constexpr int WARPS_PER_BLOCK = WarpsPerBlock;
    static constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * kWarpSize;
    static constexpr int TILE_POINTS = kSharedBudgetUnits - WARPS_PER_BLOCK * K;
    static constexpr std::size_t SHARED_BYTES = kSharedBudgetBytes;

    static_assert(BLOCK_THREADS <= 1024, "Block too large.");
    static_assert(TILE_POINTS > 0, "No room left for the shared-memory tile.");
    static_assert((TILE_POINTS % kWarpSize) == 0, "Tile must be warp-aligned.");
};

template <int K>
struct KnnTraits : KnnConfig<K, (K <= 256 ? 32 : (K == 512 ? 16 : 8))> {};

// Plain squared L2 in 2D; the problem explicitly asks for squared Euclidean distance.
__device__ __forceinline__ float squared_l2_2d(const float qx,
                                               const float qy,
                                               const float px,
                                               const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

// Generic sift-down used by heap build and heap-sort. The heap is a max-heap on distance.
__device__ __forceinline__ void sift_down(float* heap_dist,
                                          int* heap_idx,
                                          int root,
                                          int heap_size) {
    const float root_dist = heap_dist[root];
    const int root_idx = heap_idx[root];

    int child = (root << 1) + 1;
    while (child < heap_size) {
        int best_child = child;
        float best_child_dist = heap_dist[child];

        const int right = child + 1;
        if (right < heap_size) {
            const float right_dist = heap_dist[right];
            if (right_dist > best_child_dist) {
                best_child = right;
                best_child_dist = right_dist;
            }
        }

        if (best_child_dist <= root_dist) {
            break;
        }

        heap_dist[root] = best_child_dist;
        heap_idx[root] = heap_idx[best_child];
        root = best_child;
        child = (root << 1) + 1;
    }

    heap_dist[root] = root_dist;
    heap_idx[root] = root_idx;
}

template <int K>
__device__ __forceinline__ void build_max_heap(float* heap_dist, int* heap_idx) {
#pragma unroll 1
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        sift_down(heap_dist, heap_idx, i, K);
    }
}

template <int K>
__device__ __forceinline__ void replace_root_if_better(float* heap_dist,
                                                       int* heap_idx,
                                                       const float cand_dist,
                                                       const int cand_idx) {
    if (cand_dist >= heap_dist[0]) {
        return;
    }

    int root = 0;
    int child = 1;

    while (child < K) {
        int best_child = child;
        float best_child_dist = heap_dist[child];

        const int right = child + 1;
        if (right < K) {
            const float right_dist = heap_dist[right];
            if (right_dist > best_child_dist) {
                best_child = right;
                best_child_dist = right_dist;
            }
        }

        if (best_child_dist <= cand_dist) {
            break;
        }

        heap_dist[root] = best_child_dist;
        heap_idx[root] = heap_idx[best_child];
        root = best_child;
        child = (root << 1) + 1;
    }

    heap_dist[root] = cand_dist;
    heap_idx[root] = cand_idx;
}

// Append until the heap is full, then turn the buffer into a max-heap exactly once.
// After that, use the heap root as the pruning threshold.
template <int K>
__device__ __forceinline__ void heap_push_or_replace(float* heap_dist,
                                                     int* heap_idx,
                                                     int& heap_size,
                                                     float& threshold0,
                                                     const float cand_dist,
                                                     const int cand_idx) {
    if (heap_size < K) {
        heap_dist[heap_size] = cand_dist;
        heap_idx[heap_size] = cand_idx;
        ++heap_size;

        if (heap_size == K) {
            build_max_heap<K>(heap_dist, heap_idx);
            threshold0 = heap_dist[0];
        }
    } else if (cand_dist < threshold0) {
        replace_root_if_better<K>(heap_dist, heap_idx, cand_dist, cand_idx);
        threshold0 = heap_dist[0];
    }
}

// In-place heap-sort of the max-heap. Result is ascending by distance, which matches
// result[i * k + j] being the j-th nearest neighbor.
template <int K>
__device__ __forceinline__ void heap_sort_ascending(float* heap_dist, int* heap_idx) {
#pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        const float max_dist = heap_dist[0];
        const int max_idx = heap_idx[0];

        heap_dist[0] = heap_dist[end];
        heap_idx[0] = heap_idx[end];
        heap_dist[end] = max_dist;
        heap_idx[end] = max_idx;

        sift_down(heap_dist, heap_idx, 0, end);
    }
}

// Write std::pair members directly to avoid relying on a device-side pair assignment operator.
__device__ __forceinline__ void store_result(result_pair* out,
                                             const std::size_t pos,
                                             const int idx,
                                             const float dist) {
    out[pos].first = idx;
    out[pos].second = dist;
}

template <int K>
__global__ __launch_bounds__(KnnTraits<K>::BLOCK_THREADS, 2)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                result_pair* __restrict__ result) {
    using Traits = KnnTraits<K>;

    // Shared layout:
    //   s_data_x[TILE_POINTS]
    //   s_data_y[TILE_POINTS]
    //   s_heap_dist[WARPS_PER_BLOCK * K]
    //   s_heap_idx [WARPS_PER_BLOCK * K]
    //
    // The shared data tile is stored as SoA instead of float2 AoS so that a warp walking
    // 32 consecutive points performs conflict-free 32-bit shared loads.
    extern __shared__ unsigned char shared_raw[];
    float* const s_data_x = reinterpret_cast<float*>(shared_raw);
    float* const s_data_y = s_data_x + Traits::TILE_POINTS;
    float* const s_heap_dist = s_data_y + Traits::TILE_POINTS;
    int* const s_heap_idx = reinterpret_cast<int*>(s_heap_dist + Traits::WARPS_PER_BLOCK * K);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_idx = static_cast<int>(blockIdx.x) * Traits::WARPS_PER_BLOCK + warp_id;
    const bool warp_active = (query_idx < query_count);

    // Each warp gets a disjoint K-slot heap segment in shared memory.
    float* const warp_heap_dist = s_heap_dist + warp_id * K;
    int* const warp_heap_idx = s_heap_idx + warp_id * K;

    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active) {
        float2 q = make_float2(0.0f, 0.0f);
        if (lane == 0) {
            q = query[query_idx];
        }
        qx = __shfl_sync(kFullMask, q.x, 0);
        qy = __shfl_sync(kFullMask, q.y, 0);
    }

    // Only lane 0 owns the logical heap state; the rest of the warp sees it via shuffles.
    int heap_size = 0;
    float threshold0 = CUDART_INF_F;

    constexpr int kLoadIters =
        (Traits::TILE_POINTS + Traits::BLOCK_THREADS - 1) / Traits::BLOCK_THREADS;

    for (int data_base = 0; data_base < data_count; data_base += Traits::TILE_POINTS) {
        int tile_count = data_count - data_base;
        if (tile_count > Traits::TILE_POINTS) {
            tile_count = Traits::TILE_POINTS;
        }

        // Cooperative block-wide load of the next data tile.
#pragma unroll
        for (int it = 0; it < kLoadIters; ++it) {
            const int t = threadIdx.x + it * Traits::BLOCK_THREADS;
            if (t < tile_count) {
                const float2 p = data[data_base + t];
                s_data_x[t] = p.x;
                s_data_y[t] = p.y;
            }
        }
        __syncthreads();

        if (warp_active) {
            // Process the shared tile in warp-sized micro-batches.
            //
            // Correctness note:
            //   The pruning threshold is sampled once per 32-point micro-batch.
            //   This is exact because once the heap is full, the kth-best distance
            //   can only decrease. A stale threshold may admit extra candidates,
            //   but it cannot hide a true top-k candidate.
#pragma unroll 1
            for (int local_block = 0; local_block < tile_count; local_block += kWarpSize) {
                const int local = local_block + lane;
                const bool valid = (local < tile_count);

                float dist = 0.0f;
                int idx = 0;
                if (valid) {
                    dist = squared_l2_2d(qx, qy, s_data_x[local], s_data_y[local]);
                    idx = data_base + local;
                }

                const int heap_size_lane0 = __shfl_sync(kFullMask, heap_size, 0);
                const bool heap_full = (heap_size_lane0 == K);
                const float threshold = __shfl_sync(kFullMask, threshold0, 0);

                const bool accept = valid && (!heap_full || dist < threshold);
                unsigned accept_mask = __ballot_sync(kFullMask, accept);

                // All lanes participate in the shuffles; only lane 0 updates the heap.
                while (accept_mask != 0u) {
                    const int src_lane = __ffs(accept_mask) - 1;
                    const float cand_dist = __shfl_sync(kFullMask, dist, src_lane);
                    const int cand_idx = __shfl_sync(kFullMask, idx, src_lane);

                    if (lane == 0) {
                        heap_push_or_replace<K>(
                            warp_heap_dist, warp_heap_idx, heap_size, threshold0, cand_dist, cand_idx);
                    }

                    accept_mask &= accept_mask - 1;
                }

                __syncwarp(kFullMask);
            }
        }

        // Ensure every warp is done reading the tile before the next block-wide load reuses it.
        __syncthreads();
    }

    if (warp_active) {
        if (lane == 0) {
            // data_count >= K by contract, so the heap is guaranteed to be full here.
            heap_sort_ascending<K>(warp_heap_dist, warp_heap_idx);
        }

        __syncwarp(kFullMask);

        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

#pragma unroll
        for (int j = lane; j < K; j += kWarpSize) {
            store_result(result, out_base + static_cast<std::size_t>(j),
                         warp_heap_idx[j], warp_heap_dist[j]);
        }
    }
}

template <int K>
inline void launch_knn_variant(const float2* query,
                               const int query_count,
                               const float2* data,
                               const int data_count,
                               result_pair* result) {
    using Traits = KnnTraits<K>;

    // These kernels are intentionally shared-memory heavy; request the carveout explicitly.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Traits::SHARED_BYTES));
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(Traits::BLOCK_THREADS);
    const dim3 grid((query_count + Traits::WARPS_PER_BLOCK - 1) / Traits::WARPS_PER_BLOCK);

    // Launch-only wrapper by design: inputs and outputs are device pointers, so keeping the
    // call asynchronous on the default stream is usually the most useful behavior.
    knn_kernel<K><<<grid, block, Traits::SHARED_BYTES>>>(query, query_count, data, data_count, result);
}

}  // namespace knn2d_detail

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
            knn2d_detail::launch_knn_variant<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn2d_detail::launch_knn_variant<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn2d_detail::launch_knn_variant<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn2d_detail::launch_knn_variant<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn2d_detail::launch_knn_variant<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn2d_detail::launch_knn_variant<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // Input contract says k is always valid; keep the default case as a no-op.
            break;
    }
}