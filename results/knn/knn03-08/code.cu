#include <cuda_runtime.h>
#include <utility>

namespace knn_detail {

// Brute-force 2D k-NN specialized for modern data-center GPUs.
//
// Key design choices:
//  - One warp (32 threads) owns one query.
//  - The full thread block cooperatively stages a tile of the data set into shared memory.
//  - Each warp keeps a private top-k state as a max-heap in shared memory.
//    Only lane 0 mutates that heap; all 32 lanes still participate in the distance scan.
//  - Candidate filtering is warp-wide: for each warp-step (up to 32 candidates, one/lane),
//    only distances strictly smaller than the current worst top-k distance are compacted
//    into a tiny 32-entry scratch buffer. This is fast because once the heap is full,
//    accepted replacements become very rare.
//  - No additional device memory is allocated; all temporary storage lives in dynamic
//    shared memory.
//  - Distances are squared L2 distances, exactly as requested.
//
// Two CTA configurations are provided:
//  - 256 threads / 8 warps / 8 queries per block
//  - 512 threads / 16 warps / 16 queries per block
// A simple heuristic chooses the larger CTA when query_count is large enough that the
// reduced grid size is still acceptable; in return, data tiles are reused across more
// queries, reducing total global-memory traffic.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

__device__ __forceinline__ float squared_l2(const float qx, const float qy,
                                            const float px, const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

// Standard max-heap sift-down. The heap stores the current top-k nearest neighbors,
// so the heap root is the current *worst* (largest distance) among the kept neighbors.
__device__ __forceinline__ void max_heap_sift_down(float* dist, int* idx, int root, int count) {
    float root_dist = dist[root];
    int root_idx = idx[root];
    int i = root;

    while (true) {
        int child = (i << 1) + 1;
        if (child >= count) {
            break;
        }

        const int right = child + 1;
        if (right < count && dist[right] > dist[child]) {
            child = right;
        }

        if (dist[child] <= root_dist) {
            break;
        }

        dist[i] = dist[child];
        idx[i] = idx[child];
        i = child;
    }

    dist[i] = root_dist;
    idx[i] = root_idx;
}

template <int K>
__device__ __forceinline__ void build_max_heap(float* dist, int* idx) {
#pragma unroll 1
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        max_heap_sift_down(dist, idx, i, K);
    }
}

// Lane-0-only helper: consume the candidates compacted for one warp-step.
// During the initial fill phase, candidates are appended until K items are present;
// then the heap is built once. After that, only candidates better than the current
// worst distance replace the heap root.
template <int K>
__device__ __forceinline__ void consume_accepted(float* heap_dist, int* heap_idx,
                                                 int& heap_count, float& worst,
                                                 const float* scratch_dist,
                                                 const int* scratch_idx,
                                                 int accepted) {
#pragma unroll 1
    for (int t = 0; t < accepted; ++t) {
        const float d = scratch_dist[t];
        const int id = scratch_idx[t];

        if (heap_count < K) {
            heap_dist[heap_count] = d;
            heap_idx[heap_count] = id;
            ++heap_count;

            if (heap_count == K) {
                build_max_heap<K>(heap_dist, heap_idx);
                worst = heap_dist[0];
            }
        } else if (d < worst) {
            heap_dist[0] = d;
            heap_idx[0] = id;
            max_heap_sift_down(heap_dist, heap_idx, 0, K);
            worst = heap_dist[0];
        }
    }
}

// Lane-0-only finalization: repeatedly pop the max-heap root and write it from the
// back of the output slice. This yields ascending order by distance in result[0..K-1].
template <int K>
__device__ __forceinline__ void drain_heap_to_result_ascending(float* heap_dist, int* heap_idx,
                                                               std::pair<int, float>* out) {
#pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        out[end].first = heap_idx[0];
        out[end].second = heap_dist[0];

        heap_dist[0] = heap_dist[end];
        heap_idx[0] = heap_idx[end];
        max_heap_sift_down(heap_dist, heap_idx, 0, end);
    }

    out[0].first = heap_idx[0];
    out[0].second = heap_dist[0];
}

// One warp-step helper:
//  - each lane has at most one candidate
//  - accepted lanes are compacted into a 32-entry warp-private scratch buffer
//  - lane 0 consumes the compacted list and updates the heap
//
// The scratch buffer is intentionally only 32 entries deep (one per lane), which keeps
// the shared-memory footprint tiny even for K = 1024.
template <int K>
__device__ __forceinline__ void process_step(float dist, int global_idx, bool accept,
                                             unsigned lane_mask_lt, int lane,
                                             float* scratch_dist, int* scratch_idx,
                                             float* heap_dist, int* heap_idx,
                                             int& heap_count, float& worst) {
    const unsigned mask = __ballot_sync(kFullMask, accept);

    if (mask != 0u) {
        const int prefix = __popc(mask & lane_mask_lt);

        if (accept) {
            scratch_dist[prefix] = dist;
            scratch_idx[prefix] = global_idx;
        }

        const int accepted = __popc(mask);

        // Make scratch writes visible before lane 0 consumes them.
        __syncwarp(kFullMask);

        if (lane == 0) {
            consume_accepted<K>(heap_dist, heap_idx, heap_count, worst,
                                scratch_dist, scratch_idx, accepted);
        }

        // Ensure the updated threshold (worst distance) is ready before the next step.
        __syncwarp(kFullMask);
    }
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
constexpr size_t shared_bytes_for() {
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;
    return static_cast<size_t>(2) * BATCH_POINTS * sizeof(float) +                  // batch x/y
           static_cast<size_t>(WARPS_PER_BLOCK) * K * (sizeof(float) + sizeof(int)) + // heap
           static_cast<size_t>(WARPS_PER_BLOCK) * kWarpSize *
               (sizeof(float) + sizeof(int));                                       // 32-entry scratch/warp
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result) {
    static_assert((BLOCK_THREADS % kWarpSize) == 0, "BLOCK_THREADS must be a multiple of 32.");
    static_assert((BATCH_POINTS % kWarpSize) == 0, "BATCH_POINTS must be a multiple of 32.");
    static_assert((BATCH_POINTS % BLOCK_THREADS) == 0,
                  "BATCH_POINTS must be an integer multiple of BLOCK_THREADS.");
    static_assert((BATCH_POINTS & (BATCH_POINTS - 1)) == 0,
                  "BATCH_POINTS is used with a power-of-two full/tail split.");

    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;
    constexpr int LOAD_ITERS = BATCH_POINTS / BLOCK_THREADS;
    constexpr int BATCH_STEPS = BATCH_POINTS / kWarpSize;

    extern __shared__ unsigned char smem_raw[];

    // Shared-memory layout:
    //   [batch_x | batch_y | per-warp heap dist | per-warp heap idx | per-warp scratch dist | per-warp scratch idx]
    //
    // The staged batch is stored in SoA form (x[] / y[]) rather than as float2[] to avoid
    // the mild bank-conflict pattern of consecutive 64-bit shared loads.
    float* sm_batch_x = reinterpret_cast<float*>(smem_raw);
    float* sm_batch_y = sm_batch_x + BATCH_POINTS;

    float* sm_heap_dist_all = sm_batch_y + BATCH_POINTS;
    int* sm_heap_idx_all =
        reinterpret_cast<int*>(sm_heap_dist_all + WARPS_PER_BLOCK * K);

    float* sm_scratch_dist_all =
        reinterpret_cast<float*>(sm_heap_idx_all + WARPS_PER_BLOCK * K);
    int* sm_scratch_idx_all =
        reinterpret_cast<int*>(sm_scratch_dist_all + WARPS_PER_BLOCK * kWarpSize);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & (kWarpSize - 1);
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active = (query_idx < query_count);

    float* heap_dist = sm_heap_dist_all + warp * K;
    int* heap_idx = sm_heap_idx_all + warp * K;
    float* scratch_dist = sm_scratch_dist_all + warp * kWarpSize;
    int* scratch_idx = sm_scratch_idx_all + warp * kWarpSize;

    float qx = 0.0f;
    float qy = 0.0f;

    if (active) {
        float2 q = make_float2(0.0f, 0.0f);
        if (lane == 0) {
            q = query[query_idx];
        }
        qx = __shfl_sync(kFullMask, q.x, 0);
        qy = __shfl_sync(kFullMask, q.y, 0);
    }

    // These variables are only semantically meaningful on lane 0, but they are kept as
    // ordinary scalars so lane 0 can broadcast the threshold with __shfl_sync.
    int heap_count = 0;
    float worst = CUDART_INF_F;

    // Separate the hot path (full batches) from the single tail batch to remove all
    // per-candidate bounds checks for almost the entire scan.
    const int full_end = data_count & ~(BATCH_POINTS - 1);

    for (int base = 0; base < full_end; base += BATCH_POINTS) {
#pragma unroll
        for (int it = 0; it < LOAD_ITERS; ++it) {
            const int i = tid + it * BLOCK_THREADS;
            const float2 p = data[base + i];
            sm_batch_x[i] = p.x;
            sm_batch_y[i] = p.y;
        }

        __syncthreads();

        if (active) {
            for (int step = 0; step < BATCH_STEPS; ++step) {
                const int batch_i = step * kWarpSize + lane;
                const float threshold = __shfl_sync(kFullMask, worst, 0);

                const float px = sm_batch_x[batch_i];
                const float py = sm_batch_y[batch_i];
                const float dist = squared_l2(qx, qy, px, py);

                process_step<K>(dist, base + batch_i, dist < threshold,
                                lane_mask_lt, lane,
                                scratch_dist, scratch_idx,
                                heap_dist, heap_idx,
                                heap_count, worst);
            }
        }

        // Required because the next outer-loop iteration overwrites the staged batch.
        __syncthreads();
    }

    if (full_end < data_count) {
        const int tail_n = data_count - full_end;

        for (int i = tid; i < tail_n; i += BLOCK_THREADS) {
            const float2 p = data[full_end + i];
            sm_batch_x[i] = p.x;
            sm_batch_y[i] = p.y;
        }

        __syncthreads();

        if (active) {
            for (int step_base = 0; step_base < tail_n; step_base += kWarpSize) {
                const int batch_i = step_base + lane;
                const bool valid = (batch_i < tail_n);
                const float threshold = __shfl_sync(kFullMask, worst, 0);

                float dist = CUDART_INF_F;
                if (valid) {
                    dist = squared_l2(qx, qy, sm_batch_x[batch_i], sm_batch_y[batch_i]);
                }

                process_step<K>(dist, full_end + batch_i, valid && (dist < threshold),
                                lane_mask_lt, lane,
                                scratch_dist, scratch_idx,
                                heap_dist, heap_idx,
                                heap_count, worst);
            }
        }

        // No post-processing __syncthreads() is required here because the kernel ends
        // its data scan after the tail tile; each warp now only touches its private heap.
    }

    if (active && lane == 0) {
        drain_heap_to_result_ascending<K>(heap_dist, heap_idx, result + query_idx * K);
    }
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
inline void launch_knn_config(const float2* query,
                              int query_count,
                              const float2* data,
                              int data_count,
                              std::pair<int, float>* result) {
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (blocks <= 0) {
        return;
    }

    constexpr size_t smem_bytes = shared_bytes_for<K, BLOCK_THREADS, BATCH_POINTS>();

    // Opt in to the larger dynamic shared-memory budget required by the K=1024 path and
    // bias the kernel toward shared memory. The target hardware in the prompt (A100/H100)
    // supports these settings.
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    // The launch is intentionally asynchronous with respect to the host.
    knn_kernel<K, BLOCK_THREADS, BATCH_POINTS>
        <<<blocks, BLOCK_THREADS, smem_bytes>>>(query, query_count, data, data_count, result);
}

template <int K>
inline void launch_knn_adaptive(const float2* query,
                                int query_count,
                                const float2* data,
                                int data_count,
                                std::pair<int, float>* result,
                                int sm_count) {
    // 1024-point tiles are large enough to amortize CTA-wide barriers and small enough to
    // fit alongside the per-warp K-sized heaps at K = 1024.
    constexpr int BATCH_POINTS = 1024;

    // Heuristic:
    //  - 16-query CTAs (512 threads) double batch reuse vs 8-query CTAs.
    //  - For smaller query batches, 8-query CTAs preserve a denser grid and avoid
    //    underfilling the GPU.
    //
    // The threshold is intentionally simple and bandwidth-oriented.
    if (query_count >= (sm_count << 3)) {
        launch_knn_config<K, 512, BATCH_POINTS>(query, query_count, data, data_count, result);
    } else {
        launch_knn_config<K, 256, BATCH_POINTS>(query, query_count, data, data_count, result);
    }
}

}  // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    int device = 0;
    int sm_count = 1;
    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

#define KNN_LAUNCH_CASE(KVAL) \
    case KVAL: \
        knn_detail::launch_knn_adaptive<KVAL>(query, query_count, data, data_count, result, sm_count); \
        break

    switch (k) {
        KNN_LAUNCH_CASE(32);
        KNN_LAUNCH_CASE(64);
        KNN_LAUNCH_CASE(128);
        KNN_LAUNCH_CASE(256);
        KNN_LAUNCH_CASE(512);
        KNN_LAUNCH_CASE(1024);
        default:
            // Per the API contract, k is always valid.
            break;
    }

#undef KNN_LAUNCH_CASE
}