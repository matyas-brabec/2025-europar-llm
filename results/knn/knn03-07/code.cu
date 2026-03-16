#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace knn_cuda_detail {

// Exact brute-force k-NN for 2D points.
// Design summary:
//   * One warp owns one query.
//   * The thread block cooperatively stages a tile of data points in shared memory.
//   * Each warp keeps a private unsorted top-K buffer in its own shared-memory stripe.
//   * During the scan we only maintain the current worst element of that K-set.
//     A new candidate is interesting iff its distance is smaller than that threshold.
//   * Accepted candidates replace the current worst element; only the owner lane of
//     that element rescans its local stripe, then the warp recomputes the global worst
//     with a warp argmax.
//   * The final K results are sorted once at the end with an in-warp bitonic sort.
//
// This avoids heap maintenance in the hot path, keeps the tile cache in shared memory,
// and does not allocate any extra device memory.

constexpr int kWarpSize = 32;

// 80 KiB/block is a deliberate choice:
//   * It fits comfortably on A100/H100.
//   * On A100, 2 * 80 KiB = 160 KiB, so two such blocks can still reside per SM.
// The remaining shared memory after the per-warp top-K buffers is used for the data tile.
constexpr int kTargetSharedBytes = 80 * 1024;

// The block width is chosen as large as possible while still leaving room for at least
// 2048 cached data points. This maximizes data reuse across queries in the same block.
template <int K, int BLOCK_THREADS>
struct KnnConfig {
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(BLOCK_THREADS % kWarpSize == 0, "BLOCK_THREADS must be a multiple of 32.");
    static_assert(BLOCK_THREADS <= 1024, "BLOCK_THREADS exceeds the CUDA block-size limit.");

    static constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpSize;
    static constexpr int kTopKBytes = kWarpsPerBlock * K * (sizeof(float) + sizeof(int));
    static constexpr int kTilePoints =
        ((kTargetSharedBytes - kTopKBytes) / static_cast<int>(sizeof(float2)) / kWarpSize) * kWarpSize;
    static constexpr int kSharedBytes =
        static_cast<int>(sizeof(float2)) * kTilePoints + kTopKBytes;

    static_assert(kTilePoints >= 2048, "Tile is smaller than the intended minimum.");
    static_assert(kTilePoints >= K, "Tile must hold the initial K points.");
    static_assert(kSharedBytes <= kTargetSharedBytes, "Shared-memory budget exceeded.");
};

using ResultPair = std::pair<int, float>;

__device__ __forceinline__ void store_result(ResultPair *dst, int index, float dist) {
    // Write members directly to avoid depending on device-callable std::pair constructors.
    dst->first = index;
    dst->second = dist;
}

__device__ __forceinline__ bool pair_greater(float da, int ia, float db, int ib) {
    return (da > db) || ((da == db) && (ia > ib));
}

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ void warp_argmax(
    float lane_max_d,
    int lane_max_slot,
    float &global_max_d,
    int &global_max_lane,
    int &global_max_slot) {
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    float best_d = lane_max_d;
    int best_lane = lane;
    int best_slot = lane_max_slot;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_d = __shfl_down_sync(kFullMask, best_d, offset);
        const int other_lane = __shfl_down_sync(kFullMask, best_lane, offset);
        const int other_slot = __shfl_down_sync(kFullMask, best_slot, offset);

        if (other_d > best_d) {
            best_d = other_d;
            best_lane = other_lane;
            best_slot = other_slot;
        }
    }

    global_max_d = __shfl_sync(kFullMask, best_d, 0);
    global_max_lane = __shfl_sync(kFullMask, best_lane, 0);
    global_max_slot = __shfl_sync(kFullMask, best_slot, 0);
}

template <int K>
__device__ __forceinline__ void rescan_lane_max(
    const float *warp_best_dist,
    float &lane_max_d,
    int &lane_max_slot) {
    constexpr int kItemsPerLane = K / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    lane_max_d = warp_best_dist[lane];
    lane_max_slot = 0;

    #pragma unroll
    for (int s = 1; s < kItemsPerLane; ++s) {
        const float d = warp_best_dist[(s << 5) + lane];
        if (d > lane_max_d) {
            lane_max_d = d;
            lane_max_slot = s;
        }
    }
}

template <int K>
__device__ __forceinline__ void initialize_topk(
    float qx,
    float qy,
    const float *s_data_x,
    const float *s_data_y,
    float *warp_best_dist,
    int *warp_best_idx) {
    constexpr int kItemsPerLane = K / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    #pragma unroll
    for (int s = 0; s < kItemsPerLane; ++s) {
        const int pos = (s << 5) + lane;
        const float dx = qx - s_data_x[pos];
        const float dy = qy - s_data_y[pos];
        const float d = fmaf(dx, dx, dy * dy);
        warp_best_dist[pos] = d;
        warp_best_idx[pos] = pos;
    }
}

template <int K>
__device__ __forceinline__ void process_tile(
    float qx,
    float qy,
    const float *s_data_x,
    const float *s_data_y,
    int tile_base,
    int start,
    int tile_count,
    float *warp_best_dist,
    int *warp_best_idx,
    float &lane_max_d,
    int &lane_max_slot,
    float &global_max_d,
    int &global_max_lane,
    int &global_max_slot) {
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    // After the initial warm-up with the first K points, the number of accepted
    // candidates is typically small. The hot path is therefore:
    //   compare against the current threshold -> reject quickly.
    for (int t = start + lane; t < tile_count; t += kWarpSize) {
        const float dx = qx - s_data_x[t];
        const float dy = qy - s_data_y[t];
        const float d = fmaf(dx, dx, dy * dy);
        const int idx = tile_base + t;

        unsigned pending = __ballot_sync(kFullMask, d < global_max_d);

        while (pending != 0u) {
            const int src_lane = __ffs(static_cast<int>(pending)) - 1;
            const float cand_d = __shfl_sync(kFullMask, d, src_lane);
            const int cand_idx = __shfl_sync(kFullMask, idx, src_lane);

            // Re-check against the updated threshold: earlier accepted candidates
            // may have lowered it since the ballot was taken.
            if (cand_d < global_max_d) {
                if (lane == global_max_lane) {
                    const int pos = (global_max_slot << 5) + lane;
                    warp_best_dist[pos] = cand_d;
                    warp_best_idx[pos] = cand_idx;

                    // Only the lane that owned the replaced worst element needs to
                    // rescan its local stripe; other lanes' local maxima are unchanged.
                    rescan_lane_max<K>(warp_best_dist, lane_max_d, lane_max_slot);
                }

                warp_argmax(lane_max_d, lane_max_slot, global_max_d, global_max_lane, global_max_slot);
            }

            pending &= (pending - 1u);
        }
    }
}

template <int K>
__device__ __forceinline__ void sort_and_store(
    float *warp_best_dist,
    int *warp_best_idx,
    ResultPair *result,
    int query_idx) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    // Shared-memory bitonic sort over this warp's private K entries.
    // The buffer is intentionally unsorted during the scan; sorting only once
    // at the end is far cheaper than keeping the set globally ordered.
    __syncwarp();

    for (unsigned size = 2; size <= static_cast<unsigned>(K); size <<= 1) {
        for (unsigned stride = size >> 1; stride > 0; stride >>= 1) {
            for (int pos = lane; pos < K; pos += kWarpSize) {
                const int partner = static_cast<int>(static_cast<unsigned>(pos) ^ stride);

                if (partner > pos) {
                    const bool ascending = ((static_cast<unsigned>(pos) & size) == 0u);

                    const float a_d = warp_best_dist[pos];
                    const int a_i = warp_best_idx[pos];
                    const float b_d = warp_best_dist[partner];
                    const int b_i = warp_best_idx[partner];

                    const bool do_swap = ascending
                        ? pair_greater(a_d, a_i, b_d, b_i)
                        : pair_less(a_d, a_i, b_d, b_i);

                    if (do_swap) {
                        warp_best_dist[pos] = b_d;
                        warp_best_dist[partner] = a_d;
                        warp_best_idx[pos] = b_i;
                        warp_best_idx[partner] = a_i;
                    }
                }
            }

            __syncwarp();
        }
    }

    const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

    for (int pos = lane; pos < K; pos += kWarpSize) {
        store_result(result + out_base + static_cast<std::size_t>(pos),
                     warp_best_idx[pos],
                     warp_best_dist[pos]);
    }
}

template <int K, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void knn_kernel(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    ResultPair *__restrict__ result) {
    using Config = KnnConfig<K, BLOCK_THREADS>;
    constexpr int kWarpsPerBlock = Config::kWarpsPerBlock;
    constexpr int kTilePoints = Config::kTilePoints;
    constexpr unsigned kFullMask = 0xFFFFFFFFu;

    // Shared layout:
    //   s_data_x[TilePoints]
    //   s_data_y[TilePoints]
    //   s_best_dist[WarpsPerBlock * K]
    //   s_best_idx [WarpsPerBlock * K]
    //
    // The staged data tile is stored as SoA instead of AoS(float2) to avoid the
    // 64-bit shared-memory bank conflicts that would arise from warp-wide float2 loads.
    extern __shared__ __align__(16) unsigned char shared_mem[];
    float *s_data_x = reinterpret_cast<float *>(shared_mem);
    float *s_data_y = s_data_x + kTilePoints;
    float *s_best_dist = s_data_y + kTilePoints;
    int *s_best_idx = reinterpret_cast<int *>(s_best_dist + kWarpsPerBlock * K);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;
    const int query_idx = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool warp_active = (query_idx < query_count);

    float *warp_best_dist = s_best_dist + warp_id * K;
    int *warp_best_idx = s_best_idx + warp_id * K;

    float qx = 0.0f;
    float qy = 0.0f;

    if (warp_active) {
        float2 q;
        if (lane == 0) {
            q = query[query_idx];
        }
        qx = __shfl_sync(kFullMask, q.x, 0);
        qy = __shfl_sync(kFullMask, q.y, 0);
    }

    // First tile: load once, initialize each active warp's private top-K from the first K data points,
    // then continue with the remaining points in that tile.
    int first_tile_count = (data_count < kTilePoints) ? data_count : kTilePoints;
    for (int i = tid; i < first_tile_count; i += BLOCK_THREADS) {
        const float2 p = data[i];
        s_data_x[i] = p.x;
        s_data_y[i] = p.y;
    }
    __syncthreads();

    float lane_max_d = 0.0f;
    int lane_max_slot = 0;
    float global_max_d = 0.0f;
    int global_max_lane = 0;
    int global_max_slot = 0;

    if (warp_active) {
        initialize_topk<K>(qx, qy, s_data_x, s_data_y, warp_best_dist, warp_best_idx);
        rescan_lane_max<K>(warp_best_dist, lane_max_d, lane_max_slot);
        warp_argmax(lane_max_d, lane_max_slot, global_max_d, global_max_lane, global_max_slot);

        process_tile<K>(
            qx, qy,
            s_data_x, s_data_y,
            0, K, first_tile_count,
            warp_best_dist, warp_best_idx,
            lane_max_d, lane_max_slot,
            global_max_d, global_max_lane, global_max_slot);
    }
    __syncthreads();

    // Remaining tiles.
    for (int tile_base = kTilePoints; tile_base < data_count; tile_base += kTilePoints) {
        const int remaining = data_count - tile_base;
        const int tile_count = (remaining < kTilePoints) ? remaining : kTilePoints;

        for (int i = tid; i < tile_count; i += BLOCK_THREADS) {
            const float2 p = data[tile_base + i];
            s_data_x[i] = p.x;
            s_data_y[i] = p.y;
        }
        __syncthreads();

        if (warp_active) {
            process_tile<K>(
                qx, qy,
                s_data_x, s_data_y,
                tile_base, 0, tile_count,
                warp_best_dist, warp_best_idx,
                lane_max_d, lane_max_slot,
                global_max_d, global_max_lane, global_max_slot);
        }
        __syncthreads();
    }

    if (warp_active) {
        sort_and_store<K>(warp_best_dist, warp_best_idx, result, query_idx);
    }
}

template <int K, int BLOCK_THREADS>
inline void launch_knn_impl(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    ResultPair *result) {
    using Config = KnnConfig<K, BLOCK_THREADS>;

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + Config::kWarpsPerBlock - 1) / Config::kWarpsPerBlock);

    // Opt in to the full dynamic shared-memory budget used by this kernel.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        Config::kSharedBytes);

    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K, BLOCK_THREADS><<<grid, block, Config::kSharedBytes>>>(
        query, query_count, data, data_count, result);
}

} // namespace knn_cuda_detail

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

    // Largest practical block for each K while still leaving room for a useful shared tile:
    //   K in {  32,   64,  128,  256 } -> 32 warps/block (1024 threads)
    //   K = 512                           -> 16 warps/block (512 threads)
    //   K = 1024                          ->  8 warps/block (256 threads)
    //
    // This maximizes reuse of the staged data tile across many queries in the same block.
    switch (k) {
        case 32:
            knn_cuda_detail::launch_knn_impl<32, 1024>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_cuda_detail::launch_knn_impl<64, 1024>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_cuda_detail::launch_knn_impl<128, 1024>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_cuda_detail::launch_knn_impl<256, 1024>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_cuda_detail::launch_knn_impl<512, 512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_cuda_detail::launch_knn_impl<1024, 256>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees valid K, so this path is only a safeguard.
            break;
    }
}