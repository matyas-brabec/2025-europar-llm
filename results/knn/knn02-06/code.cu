#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace detail {

// Exact 2D k-NN for squared Euclidean distance.
//
// Design:
//   * One warp owns one query for the whole scan.
//   * A block stages the data set into shared-memory tiles so each global load is reused by many queries.
//   * The current top-k for a query is kept as an unsorted warp-private set distributed across the 32 lanes:
//       - each lane owns K/32 (distance, index) pairs in registers;
//       - the warp also tracks the current worst element in that set.
//   * Exactness does not require the set to stay sorted while scanning:
//       - if a new candidate is not better than the current worst, it cannot enter the top-k;
//       - otherwise it replaces one current worst element;
//       - this preserves the exact k smallest distances seen so far.
//   * Sorting is deferred until the very end, where each warp performs one bitonic sort over its final K pairs
//     and writes the results in ascending distance order.
//
// No additional device allocations are used. The same dynamic shared-memory buffer is deliberately overlaid:
//   * during the scan it is interpreted as a float2 tile of data points;
//   * after the scan it is reinterpreted as per-warp scratch for the final bitonic sort.
// The two lifetimes do not overlap, so the required shared memory is only max(tile_bytes, sort_scratch_bytes).

constexpr int kWarpSize = 32;
constexpr unsigned int kFullMask = 0xFFFFFFFFu;

struct alignas(8) Candidate {
    float dist;
    int idx;
};

struct ArgMaxPair {
    float dist;
    int lane;
};

__device__ __forceinline__ float sq_l2_2d(const float qx, const float qy, const float2& p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

template <int M>
__device__ __forceinline__ void recompute_local_worst(
    const float (&best_dist)[M],
    float& local_worst_dist,
    int& local_worst_slot) {
    local_worst_dist = best_dist[0];
    local_worst_slot = 0;
#pragma unroll
    for (int i = 1; i < M; ++i) {
        if (best_dist[i] > local_worst_dist) {
            local_worst_dist = best_dist[i];
            local_worst_slot = i;
        }
    }
}

// Warp-wide arg-max over one value per lane.
// The physical lane id is kept separate from the "best lane" carried by the reduction.
__device__ __forceinline__ ArgMaxPair warp_argmax(const float value_in, const int lane) {
    float best_value = value_in;
    int best_lane = lane;

#pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(kFullMask, best_value, offset);
        const int other_lane = __shfl_down_sync(kFullMask, best_lane, offset);

        if (lane < kWarpSize - offset) {
            // Ties are arbitrary for correctness; choosing the smaller lane makes the owner deterministic.
            if ((other_value > best_value) || (other_value == best_value && other_lane < best_lane)) {
                best_value = other_value;
                best_lane = other_lane;
            }
        }
    }

    ArgMaxPair out;
    out.dist = __shfl_sync(kFullMask, best_value, 0);
    out.lane = __shfl_sync(kFullMask, best_lane, 0);
    return out;
}

// Warp-synchronous exact insertion into the distributed unsorted top-k.
// __shfl_sync / warp_argmax synchronize the participating warp lanes, so no extra __syncwarp() is needed here.
template <int M>
__device__ __forceinline__ void try_insert_candidate(
    const float cand_dist,
    const int cand_idx,
    float (&best_dist)[M],
    int (&best_idx)[M],
    float& local_worst_dist,
    int& local_worst_slot,
    float& global_worst_dist,
    int& global_worst_lane,
    const int lane) {
    if (cand_dist < global_worst_dist) {
        if (lane == global_worst_lane) {
            best_dist[local_worst_slot] = cand_dist;
            best_idx[local_worst_slot] = cand_idx;
            recompute_local_worst(best_dist, local_worst_dist, local_worst_slot);
        }

        const ArgMaxPair argmax = warp_argmax(local_worst_dist, lane);
        global_worst_dist = argmax.dist;
        global_worst_lane = argmax.lane;
    }
}

__device__ __forceinline__ bool candidate_less(const Candidate& a, const Candidate& b) {
    // Distances are the primary key. The index tie-break only makes the final order deterministic;
    // the problem statement allows any tie resolution.
    return (a.dist < b.dist) || (a.dist == b.dist && a.idx < b.idx);
}

template <int K>
__device__ __forceinline__ void sort_and_store_results(
    const float (&best_dist)[K / kWarpSize],
    const int (&best_idx)[K / kWarpSize],
    Candidate* warp_scratch,
    std::pair<int, float>* result_base,
    const int lane) {
    constexpr int M = K / kWarpSize;

    // Materialize the warp-private register state into shared memory once.
#pragma unroll
    for (int slot = 0; slot < M; ++slot) {
        const int pos = slot * kWarpSize + lane;
        warp_scratch[pos].dist = best_dist[slot];
        warp_scratch[pos].idx = best_idx[slot];
    }
    __syncwarp();

    // Final exact ordering: bitonic sort over K elements, one warp per query.
    // This is done once per query, so it is much cheaper than maintaining a fully sorted list during the scan.
#pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < K; i += kWarpSize) {
                const int ixj = i ^ stride;
                if (ixj > i) {
                    Candidate a = warp_scratch[i];
                    Candidate b = warp_scratch[ixj];
                    const bool ascending = ((i & size) == 0);
                    const bool do_swap = ascending ? candidate_less(b, a) : candidate_less(a, b);
                    if (do_swap) {
                        warp_scratch[i] = b;
                        warp_scratch[ixj] = a;
                    }
                }
            }
            __syncwarp();
        }
    }

    // Write sorted results back in row-major order.
    for (int pos = lane; pos < K; pos += kWarpSize) {
        result_base[pos].first = warp_scratch[pos].idx;
        result_base[pos].second = warp_scratch[pos].dist;
    }
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    std::pair<int, float>* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert(K % kWarpSize == 0, "K must be divisible by the warp size.");
    static_assert(TILE_POINTS >= K, "The first tile must be large enough to seed the initial exact top-k.");
    static_assert(WARPS_PER_BLOCK > 0 && WARPS_PER_BLOCK * kWarpSize <= 1024, "Invalid block size.");

    constexpr int M = K / kWarpSize;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_in_block = tid >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool active = query_idx < query_count;

    extern __shared__ __align__(8) unsigned char smem[];
    float2* const data_tile = reinterpret_cast<float2*>(smem);

    // Shared-memory overlay:
    //   * phase 1: smem[0 .. tile_bytes) is a float2 tile
    //   * phase 2: smem[0 .. scratch_bytes) is Candidate scratch for all warps in the block
    Candidate* const sort_scratch = reinterpret_cast<Candidate*>(smem);
    Candidate* const my_scratch = sort_scratch + warp_in_block * K;

    const float qx = active ? query[query_idx].x : 0.0f;
    const float qy = active ? query[query_idx].y : 0.0f;

    // Warp-private distributed top-k state, fully resident in registers.
    float best_dist[M];
    int best_idx[M];

    float local_worst_dist = 0.0f;
    int local_worst_slot = 0;
    float global_worst_dist = CUDART_INF_F;
    int global_worst_lane = 0;

    // Main scan: process the entire data set in shared-memory tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_count = data_count - tile_start;
        if (tile_count > TILE_POINTS) {
            tile_count = TILE_POINTS;
        }

        // Cooperative block-wide load from global memory to shared memory.
        for (int i = tid; i < tile_count; i += blockDim.x) {
            data_tile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active) {
            if (tile_start == 0) {
                // Seed the exact top-k with the first K points.
                // This avoids a costly "empty-to-full" insertion phase and is exact because K <= TILE_POINTS.
#pragma unroll
                for (int slot = 0; slot < M; ++slot) {
                    const int local_idx = slot * kWarpSize + lane;
                    const float d = sq_l2_2d(qx, qy, data_tile[local_idx]);
                    best_dist[slot] = d;
                    best_idx[slot] = local_idx;  // tile_start == 0 here, so local index is the global index.
                }

                recompute_local_worst(best_dist, local_worst_dist, local_worst_slot);
                const ArgMaxPair argmax = warp_argmax(local_worst_dist, lane);
                global_worst_dist = argmax.dist;
                global_worst_lane = argmax.lane;
            }

            // The first K points of the first tile are already consumed by initialization.
            const int begin = (tile_start == 0) ? K : 0;

            // Process one 32-point group at a time so the warp stays synchronized.
            // Each lane computes one distance; any candidate better than the current threshold is inserted exactly.
            for (int group_base = begin; group_base < tile_count; group_base += kWarpSize) {
                const int local_idx = group_base + lane;

                float d = CUDART_INF_F;
                if (local_idx < tile_count) {
                    d = sq_l2_2d(qx, qy, data_tile[local_idx]);
                }

                // Lanes whose candidate is currently below the global worst enter the mask.
                // The mask is based on the threshold at the beginning of the 32-point group.
                // After every successful insertion the threshold can only decrease, so each candidate is rechecked
                // before insertion to preserve exactness.
                unsigned int mask = __ballot_sync(kFullMask, d < global_worst_dist);

                while (mask != 0u) {
                    const int src_lane = __ffs(mask) - 1;
                    const float cand_dist = __shfl_sync(kFullMask, d, src_lane);
                    const int cand_idx = tile_start + group_base + src_lane;

                    try_insert_candidate(
                        cand_dist,
                        cand_idx,
                        best_dist,
                        best_idx,
                        local_worst_dist,
                        local_worst_slot,
                        global_worst_dist,
                        global_worst_lane,
                        lane);

                    mask &= (mask - 1u);
                }
            }
        }

        // The tile storage is about to be overwritten on the next iteration, so all warps in the block
        // must finish reading it before the next cooperative load.
        __syncthreads();
    }

    // Final phase: reuse the same shared memory as per-warp scratch and sort once.
    if (active) {
        sort_and_store_results<K>(
            best_dist,
            best_idx,
            my_scratch,
            result + static_cast<std::size_t>(query_idx) * K,
            lane);
    }
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
inline void launch_knn_specialized(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    std::pair<int, float>* result) {
    constexpr std::size_t tile_bytes = static_cast<std::size_t>(TILE_POINTS) * sizeof(float2);
    constexpr std::size_t scratch_bytes = static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(Candidate);
    constexpr std::size_t shared_bytes = (tile_bytes > scratch_bytes) ? tile_bytes : scratch_bytes;

    // Opt in to large dynamic shared-memory allocations on modern data-center GPUs.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    // This kernel intentionally spends shared memory to reduce global traffic, so a shared-heavy carveout is preferred.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    (void)cudaFuncSetCacheConfig(
        knn_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>,
        cudaFuncCachePreferShared);

    const dim3 block(WARPS_PER_BLOCK * kWarpSize);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>
        <<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result);
}

}  // namespace detail

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

    // Launch configurations are tuned for A100/H100-class GPUs:
    //   * small/medium k use 16 warps/block to maximize shared-tile reuse across queries without hurting occupancy;
    //   * k=1024 uses 8 warps/block to stay out of the 1-block/SM regime while still reusing each tile across 8 queries;
    //   * tile size grows with k when it is "free" in shared-memory footprint because the final sort scratch dominates.
    //
    // The launch remains asynchronous, matching conventional CUDA runtime behavior.
    switch (k) {
        case 32:
            detail::launch_knn_specialized<32, 16, 4096>(query, query_count, data, data_count, result);
            break;
        case 64:
            detail::launch_knn_specialized<64, 16, 4096>(query, query_count, data, data_count, result);
            break;
        case 128:
            detail::launch_knn_specialized<128, 16, 4096>(query, query_count, data, data_count, result);
            break;
        case 256:
            detail::launch_knn_specialized<256, 16, 4096>(query, query_count, data, data_count, result);
            break;
        case 512:
            detail::launch_knn_specialized<512, 16, 8192>(query, query_count, data, data_count, result);
            break;
        case 1024:
            detail::launch_knn_specialized<1024, 8, 8192>(query, query_count, data, data_count, result);
            break;
        default:
            // Unreachable for valid inputs (k is guaranteed to be a power of two in [32, 1024]).
            break;
    }
}