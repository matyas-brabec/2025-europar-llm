#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn_detail {

// Warp-specialized k-NN for 2D points.
// Key design points, matching the requested execution model:
//   * one warp computes one query;
//   * the whole block stages a tile of input data points in shared memory;
//   * each warp owns a private shared-memory slice:
//       [0, K)   : current top-K intermediate result (sorted after every flush)
//       [K, 2*K) : unsorted candidate buffer
//   * candidates farther than the current K-th neighbor are rejected early;
//   * when the candidate buffer fills or would overflow, the warp merges
//     "intermediate result + candidates" cooperatively.
//
// The merge path is implemented as a full in-place bitonic sort of 2*K items
// inside the warp's private shared-memory slice. For this problem size
// (K <= 1024) and expected acceptance rate (it drops quickly once the threshold
// tightens), this is a robust and fast shared-memory-heavy strategy on A100/H100
// without requiring any extra device allocation.

using ResultPair = std::pair<int, float>;

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Cooperative in-warp bitonic sort over shared-memory arrays.
// Distances are the primary key; indices are only used as a deterministic
// tie-breaker (the API allows arbitrary tie resolution anyway).
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(float* dist, int* idx, int lane) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort length must be a power of two.");

    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += kWarpSize) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const float di = dist[i];
                    const float dp = dist[partner];
                    const int ii = idx[i];
                    const int ip = idx[partner];

                    const bool ascending = ((i & size) == 0);
                    const bool do_swap =
                        ascending ? ((di > dp) || ((di == dp) && (ii > ip)))
                                  : ((di < dp) || ((di == dp) && (ii < ip)));

                    if (do_swap) {
                        dist[i] = dp;
                        dist[partner] = di;
                        idx[i] = ip;
                        idx[partner] = ii;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Flush the candidate buffer into the intermediate top-K.
// The first K entries hold the current intermediate result; the second K
// entries are the candidate buffer. Missing candidate slots are padded with +inf
// so the final 2*K in-place sort leaves the K smallest items in [0, K).
template <int K>
__device__ __forceinline__ void merge_candidate_buffer(
    float* dist, int* idx, int lane, int& cand_count, float& threshold) {
    if (cand_count == 0) {
        return;
    }

#pragma unroll
    for (int pos = K + lane; pos < 2 * K; pos += kWarpSize) {
        if ((pos - K) >= cand_count) {
            dist[pos] = CUDART_INF_F;
            idx[pos] = -1;
        }
    }
    __syncwarp(kFullMask);

    bitonic_sort_shared<2 * K>(dist, idx, lane);

    float kth = 0.0f;
    if (lane == 0) {
        kth = dist[K - 1];
    }
    threshold = __shfl_sync(kFullMask, kth, 0);
    cand_count = 0;
}

template <int K, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                ResultPair* __restrict__ result) {
    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024,
                  "K must be a power of two in [32, 1024].");
    static_assert(BLOCK_THREADS % kWarpSize == 0,
                  "BLOCK_THREADS must be a multiple of warp size.");

    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpSize;
    constexpr int kTileSteps = BLOCK_THREADS / kWarpSize;
    constexpr std::size_t kPerWarpFloatBytes =
        static_cast<std::size_t>(2 * K) * sizeof(float);
    constexpr std::size_t kPerWarpIntBytes =
        static_cast<std::size_t>(2 * K) * sizeof(int);
    constexpr std::size_t kPerWarpBytes = kPerWarpFloatBytes + kPerWarpIntBytes;

    extern __shared__ unsigned char shared_raw[];

    // Shared-memory layout:
    //   tile_x[BLOCK_THREADS]
    //   tile_y[BLOCK_THREADS]
    //   warp 0: dist[2*K], idx[2*K]
    //   warp 1: dist[2*K], idx[2*K]
    //   ...
    // Splitting the staged data tile into SoA (x/y) avoids the bank behavior of
    // repeated float2 accesses while keeping the same total byte footprint.
    float* tile_x = reinterpret_cast<float*>(shared_raw);
    float* tile_y = tile_x + BLOCK_THREADS;
    unsigned char* warp_storage = reinterpret_cast<unsigned char*>(tile_y + BLOCK_THREADS);

    const int warp_local = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);

    unsigned char* my_storage =
        warp_storage + static_cast<std::size_t>(warp_local) * kPerWarpBytes;
    float* dist = reinterpret_cast<float*>(my_storage);                      // [0, 2*K)
    int* idx = reinterpret_cast<int*>(my_storage + kPerWarpFloatBytes);     // [0, 2*K)

    const int query_idx = blockIdx.x * kWarpsPerBlock + warp_local;
    const bool valid = (query_idx < query_count);

    // One query point per warp; lane 0 loads, then broadcasts.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && valid) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Initialize the intermediate result to +inf / invalid index.
    if (valid) {
#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            dist[i] = CUDART_INF_F;
            idx[i] = -1;
        }
        __syncwarp(kFullMask);
    }

    int cand_count = 0;
    float threshold = CUDART_INF_F;

    // Stream over the data set in block-sized tiles staged in shared memory.
    // All threads participate in the tile load, even if their warp has no valid
    // query in the tail block; that keeps the cooperative load fully coalesced.
    for (int tile_base = 0; tile_base < data_count; tile_base += BLOCK_THREADS) {
        const int global_idx = tile_base + threadIdx.x;
        if (global_idx < data_count) {
            const float2 p = data[global_idx];
            tile_x[threadIdx.x] = p.x;
            tile_y[threadIdx.x] = p.y;
        }
        __syncthreads();

        const int remaining = data_count - tile_base;
        const int tile_count = (remaining < BLOCK_THREADS) ? remaining : BLOCK_THREADS;

        // Each warp walks the cached tile with one element per lane per step.
        // The loop trip count is fixed by BLOCK_THREADS, which keeps the warp
        // fully converged for ballot/shuffle usage; inactive lanes simply use
        // has_elem == false on the tail step of the last tile.
        for (int step = 0; step < kTileSteps; ++step) {
            const int j = step * kWarpSize + lane;
            const bool has_elem = valid && (j < tile_count);

            float d = 0.0f;
            if (has_elem) {
                const float dx = qx - tile_x[j];
                const float dy = qy - tile_y[j];
                d = fmaf(dx, dx, dy * dy);  // squared Euclidean distance
            }

            bool pass = has_elem && (d < threshold);
            unsigned pass_mask = __ballot_sync(kFullMask, pass);
            int pass_count = __popc(pass_mask);

            // If this warp-sized insertion would overflow the K-entry candidate
            // buffer, first flush the current buffer, then re-test the current
            // warp chunk against the tighter post-merge threshold.
            if (cand_count + pass_count > K) {
                merge_candidate_buffer<K>(dist, idx, lane, cand_count, threshold);

                pass = has_elem && (d < threshold);
                pass_mask = __ballot_sync(kFullMask, pass);
                pass_count = __popc(pass_mask);
            }

            const unsigned lane_mask_lt = (1u << lane) - 1u;
            const int local_offset = __popc(pass_mask & lane_mask_lt);

            if (pass) {
                const int out = K + cand_count + local_offset;
                dist[out] = d;
                idx[out] = tile_base + j;
            }
            __syncwarp(kFullMask);

            cand_count += pass_count;

            // Flush immediately when the buffer becomes full. This both satisfies
            // the requested behavior and tightens the rejection threshold early.
            if (cand_count == K) {
                merge_candidate_buffer<K>(dist, idx, lane, cand_count, threshold);
            }
        }

        // Ensure every warp is done consuming the staged tile before it is reused
        // by the next block-wide cooperative load.
        __syncthreads();
    }

    if (valid && cand_count > 0) {
        merge_candidate_buffer<K>(dist, idx, lane, cand_count, threshold);
    }

    // The first K entries are sorted ascending by distance after the final flush.
    if (valid) {
        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            result[out_base + i].first = idx[i];
            result[out_base + i].second = dist[i];
        }
    }
}

template <int K, int BLOCK_THREADS>
constexpr std::size_t shared_bytes_for_kernel() {
    return static_cast<std::size_t>(2 * BLOCK_THREADS) * sizeof(float) +
           static_cast<std::size_t>(BLOCK_THREADS / kWarpSize) *
               (static_cast<std::size_t>(2 * K) * sizeof(float) +
                static_cast<std::size_t>(2 * K) * sizeof(int));
}

inline int current_sm_count() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    return (sms > 0) ? sms : 0;
}

// Host-side launcher for a single (K, BLOCK_THREADS) specialization.
// The kernel is heavily shared-memory bound by design, so we explicitly opt in
// to the required dynamic shared-memory size and request a full shared carveout.
template <int K, int BLOCK_THREADS>
inline void launch_knn_kernel(const float2* query,
                              int query_count,
                              const float2* data,
                              int data_count,
                              ResultPair* result) {
    constexpr std::size_t shmem = shared_bytes_for_kernel<K, BLOCK_THREADS>();

    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpSize;
    const int grid = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    knn_kernel<K, BLOCK_THREADS><<<grid, BLOCK_THREADS, shmem>>>(
        query, query_count, data, data_count, result);
}

// Block-size heuristic:
//   * 256 threads  => 8 queries/block
//   * 512 threads  => 16 queries/block
//
// For the prompt's "low-thousands" query counts on large GPUs, 1024-thread
// blocks would too often reduce the number of resident blocks below the number
// of SMs. The 256/512 choice is a better compromise:
//
//   * 256 threads keeps enough independent blocks in flight when query_count is
//     only modestly above 1k;
//   * 512 threads doubles data-tile reuse and halves block-wide tile barriers
//     once there are enough queries to populate all SMs with 16-query blocks.
template <int K>
inline void dispatch_for_k(const float2* query,
                           int query_count,
                           const float2* data,
                           int data_count,
                           ResultPair* result,
                           int sm_count) {
    const bool use_512 = (sm_count > 0) && (query_count >= 16 * sm_count);
    if (use_512) {
        launch_knn_kernel<K, 512>(query, query_count, data, data_count, result);
    } else {
        launch_knn_kernel<K, 256>(query, query_count, data, data_count, result);
    }
}

// K=1024 needs 256-thread blocks to stay within the A100/H100 per-block shared
// memory budget while still keeping 8 queries/block.
template <>
inline void dispatch_for_k<1024>(const float2* query,
                                 int query_count,
                                 const float2* data,
                                 int data_count,
                                 ResultPair* result,
                                 int /*sm_count*/) {
    launch_knn_kernel<1024, 256>(query, query_count, data, data_count, result);
}

}  // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    // Standard CUDA convention: asynchronous launch on the current default stream.
    if (query_count <= 0) {
        return;
    }

    const int sm_count = knn_detail::current_sm_count();

    switch (k) {
        case 32:
            knn_detail::dispatch_for_k<32>(query, query_count, data, data_count, result, sm_count);
            break;
        case 64:
            knn_detail::dispatch_for_k<64>(query, query_count, data, data_count, result, sm_count);
            break;
        case 128:
            knn_detail::dispatch_for_k<128>(query, query_count, data, data_count, result, sm_count);
            break;
        case 256:
            knn_detail::dispatch_for_k<256>(query, query_count, data, data_count, result, sm_count);
            break;
        case 512:
            knn_detail::dispatch_for_k<512>(query, query_count, data, data_count, result, sm_count);
            break;
        case 1024:
            knn_detail::dispatch_for_k<1024>(query, query_count, data, data_count, result, sm_count);
            break;
        default:
            // By contract, k is always a valid power of two in [32, 1024].
            break;
    }
}