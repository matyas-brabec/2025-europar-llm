#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace knn_detail {

// Design summary:
//
// * One warp owns one query.
// * The current top-k for that query is kept private to the warp in a striped layout:
//       lane l owns ranks l, l+32, l+64, ...
//   This makes the reverse-order access used by the bitonic top-k merge a simple lane-reversal
//   shuffle (`source_lane = 31 - lane`).
// * Each warp also owns a k-entry candidate buffer in shared memory plus a shared counter.
// * The whole block streams the `data` array through a large shared-memory tile. All query warps
//   reuse that tile, which is the dominant optimization for million-point data sets.
// * Whenever the candidate buffer is merged, the steps follow the requested scheme exactly:
//     1) sort buffer ascending with bitonic sort,
//     2) bitonic min-merge with the reversed intermediate result,
//     3) sort again to obtain the updated ascending intermediate result.
//
// Tuning choice:
//
// For k <= 512 we use 32 warps/block (1024 threads) to maximize reuse of each shared tile.
// For k == 1024 each warp needs an 8 KiB candidate buffer, so 16 warps/block is the best fit
// that still leaves room for a useful data tile under the 160 KiB opt-in shared-memory budget
// supported by A100 and H100-class GPUs.

constexpr int      kWarpSize          = 32;
constexpr unsigned kFullMask          = 0xFFFFFFFFu;
constexpr size_t   kSharedBudgetBytes = 160 * 1024;

// Compile-time launch/shared-memory configuration for a given (K, warps/block).
template <int K, int WARPS_PER_BLOCK>
struct KernelConfig {
    static_assert(K >= kWarpSize && K <= 1024 && (K & (K - 1)) == 0, "K must be a power of two in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert((WARPS_PER_BLOCK * kWarpSize) <= 1024, "Block size exceeds CUDA limit.");

    static constexpr int items_per_lane = K / kWarpSize;
    static constexpr int block_threads  = WARPS_PER_BLOCK * kWarpSize;

    // Candidate buffer uses a structure-of-arrays layout in shared memory to keep the bitonic
    // sort/merge on 4-byte banks.
    static constexpr size_t candidate_dist_bytes = static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(float);
    static constexpr size_t candidate_idx_bytes  = static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(int);
    static constexpr size_t count_bytes          = static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);

    // Leave enough shared memory for the tile after reserving all per-warp candidate storage.
    static_assert(candidate_dist_bytes + candidate_idx_bytes + count_bytes + kWarpSize * sizeof(float2) <= kSharedBudgetBytes,
                  "Configuration leaves no room for even one warp-wide data chunk.");

    static constexpr size_t available_tile_bytes = kSharedBudgetBytes - candidate_dist_bytes - candidate_idx_bytes - count_bytes;
    static constexpr int    raw_tile_points      = static_cast<int>(available_tile_bytes / sizeof(float2));

    // Round the tile size down to a multiple of a warp:
    //   * simplifies per-warp chunking,
    //   * preserves 16-byte alignment for float4 vectorized tile loads because K and tile_points
    //     are both multiples of 32, so every tile_base is also a multiple of 32.
    static constexpr int tile_points = (raw_tile_points / kWarpSize) * kWarpSize;

    static_assert(tile_points >= kWarpSize, "Tile must contain at least one warp-wide chunk.");
    static_assert((tile_points % kWarpSize) == 0, "Tile size must be warp-aligned.");

    static constexpr size_t shared_bytes =
        static_cast<size_t>(tile_points) * sizeof(float2) +
        candidate_dist_bytes +
        candidate_idx_bytes +
        count_bytes;

    static_assert(shared_bytes <= kSharedBudgetBytes, "Shared-memory budget exceeded.");
};

template <int K> struct PreferredWarps;
template <> struct PreferredWarps<32>   { static constexpr int value = 32; };
template <> struct PreferredWarps<64>   { static constexpr int value = 32; };
template <> struct PreferredWarps<128>  { static constexpr int value = 32; };
template <> struct PreferredWarps<256>  { static constexpr int value = 32; };
template <> struct PreferredWarps<512>  { static constexpr int value = 32; };
template <> struct PreferredWarps<1024> { static constexpr int value = 16; };

// Total order on (distance, index). The index tie-break is not required for correctness of the
// problem statement, but it gives the bitonic network a deterministic strict weak order.
__device__ __forceinline__ bool pair_less(float ad, int ai, float bd, int bi) {
    return (ad < bd) || ((ad == bd) && (ai < bi));
}

__device__ __forceinline__ bool pair_greater(float ad, int ai, float bd, int bi) {
    return pair_less(bd, bi, ad, ai);
}

__device__ __forceinline__ float squared_l2(const float2 p, float qx, float qy) {
    const float dx = p.x - qx;
    const float dy = p.y - qy;
    return fmaf(dx, dx, dy * dy);
}

// Warp-cooperative in-place bitonic sort of K key/value pairs stored in shared memory.
// Each lane owns the striped indices lane, lane+32, lane+64, ...
template <int K>
__device__ __forceinline__ void bitonic_sort_shared(float* dist, int* idx, int lane) {
    constexpr int ITEMS = K / kWarpSize;

    // Avoid fully unrolling the outer bitonic stages; that would inflate code size heavily for
    // K = 1024. The per-lane striped loop is still fully unrolled.
    #pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int t = 0; t < ITEMS; ++t) {
                const int i = lane + t * kWarpSize;
                const int j = i ^ stride;

                if (j > i) {
                    const float di = dist[i];
                    const int   ii = idx[i];
                    const float dj = dist[j];
                    const int   ij = idx[j];

                    const bool ascending = ((i & size) == 0);
                    if ((ascending  && pair_greater(di, ii, dj, ij)) ||
                        (!ascending && pair_less   (di, ii, dj, ij))) {
                        dist[i] = dj; idx[i] = ij;
                        dist[j] = di; idx[j] = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Seed the private intermediate result with the first K data points and sort it ascending.
// This tightens max_distance immediately, which sharply improves later filtering.
template <int K>
__device__ __forceinline__ float seed_topk(
    const float2* __restrict__ data,
    float qx,
    float qy,
    float* scratch_dist,
    int* scratch_idx,
    float (&best_dist)[K / kWarpSize],
    int   (&best_idx )[K / kWarpSize],
    int lane)
{
    constexpr int ITEMS = K / kWarpSize;

    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = lane + t * kWarpSize;
        scratch_dist[p] = squared_l2(data[p], qx, qy);
        scratch_idx [p] = p;
    }
    __syncwarp();

    bitonic_sort_shared<K>(scratch_dist, scratch_idx, lane);

    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = lane + t * kWarpSize;
        best_dist[t] = scratch_dist[p];
        best_idx [t] = scratch_idx [p];
    }

    // In striped layout, rank K-1 is held by lane 31 at slot ITEMS-1.
    float kth = best_dist[ITEMS - 1];
    return __shfl_sync(kFullMask, kth, kWarpSize - 1);
}

// Merge the shared candidate buffer into the warp-private intermediate result.
// Steps:
//   1) pad the unused tail with (+inf, INT_MAX) and sort the whole K-entry buffer ascending,
//   2) write cand[i] = min(cand[i], best[K-1-i]) to form a bitonic sequence,
//   3) sort that bitonic sequence ascending,
//   4) copy back to the private striped top-k and refresh max_distance.
template <int K>
__device__ __forceinline__ void flush_candidates(
    float* cand_dist,
    int* cand_idx,
    int* count_ptr,
    float (&best_dist)[K / kWarpSize],
    int   (&best_idx )[K / kWarpSize],
    float& max_distance,
    int lane)
{
    constexpr int ITEMS = K / kWarpSize;

    int count = 0;
    if (lane == 0) {
        count = *count_ptr;
    }
    count = __shfl_sync(kFullMask, count, 0);

    if (count == 0) {
        return;
    }

    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = lane + t * kWarpSize;
        if (p >= count) {
            cand_dist[p] = CUDART_INF_F;
            cand_idx [p] = INT_MAX;
        }
    }
    __syncwarp();

    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

    const int reverse_lane = kWarpSize - 1 - lane;

    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = lane + t * kWarpSize;

        const float bd = cand_dist[p];
        const int   bi = cand_idx [p];

        // Reverse rank K-1-p maps to:
        //   source lane = 31 - lane
        //   source slot = ITEMS - 1 - t
        const float rd_local = best_dist[ITEMS - 1 - t];
        const int   ri_local = best_idx [ITEMS - 1 - t];
        const float rd       = __shfl_sync(kFullMask, rd_local, reverse_lane);
        const int   ri       = __shfl_sync(kFullMask, ri_local, reverse_lane);

        if (pair_less(bd, bi, rd, ri)) {
            cand_dist[p] = bd;
            cand_idx [p] = bi;
        } else {
            cand_dist[p] = rd;
            cand_idx [p] = ri;
        }
    }
    __syncwarp();

    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

    #pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int p = lane + t * kWarpSize;
        best_dist[t] = cand_dist[p];
        best_idx [t] = cand_idx [p];
    }

    float kth = best_dist[ITEMS - 1];
    max_distance = __shfl_sync(kFullMask, kth, kWarpSize - 1);

    if (lane == 0) {
        *count_ptr = 0;
    }
    __syncwarp();
}

template <int K, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result)
{
    using Config = KernelConfig<K, WARPS_PER_BLOCK>;
    constexpr int ITEMS       = Config::items_per_lane;
    constexpr int TILE_POINTS = Config::tile_points;

    // Dynamic shared-memory layout:
    //   [ float2 tile[TILE_POINTS] ]
    //   [ float  cand_dist[WARPS_PER_BLOCK * K] ]
    //   [ int    cand_idx [WARPS_PER_BLOCK * K] ]
    //   [ int    cand_count[WARPS_PER_BLOCK] ]
    extern __shared__ __align__(16) unsigned char shared_raw[];

    float2* const tile          = reinterpret_cast<float2*>(shared_raw);
    float*  const cand_dist_all = reinterpret_cast<float*>(tile + TILE_POINTS);
    int*    const cand_idx_all  = reinterpret_cast<int*>(cand_dist_all + WARPS_PER_BLOCK * K);
    int*    const cand_count    = cand_idx_all + WARPS_PER_BLOCK * K;

    const int lane       = threadIdx.x & (kWarpSize - 1);
    const int warp_local = threadIdx.x >> 5;
    const int query_idx  = blockIdx.x * WARPS_PER_BLOCK + warp_local;
    const bool active    = (query_idx < query_count);

    float* const cand_dist = cand_dist_all + warp_local * K;
    int*   const cand_idx  = cand_idx_all  + warp_local * K;
    int*   const count_ptr = cand_count    + warp_local;

    if (lane == 0) {
        *count_ptr = 0;
    }

    // Broadcast the query coordinates from lane 0.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-private intermediate top-k in striped layout.
    float best_dist[ITEMS];
    int   best_idx [ITEMS];
    float max_distance = CUDART_INF_F;

    if (active) {
        max_distance = seed_topk<K>(data, qx, qy, cand_dist, cand_idx, best_dist, best_idx, lane);
    }

    // Process the rest of the data in shared-memory tiles.
    for (int tile_base = K; tile_base < data_count; tile_base += TILE_POINTS) {
        const int remaining = data_count - tile_base;
        const int batch_n   = (remaining < TILE_POINTS) ? remaining : TILE_POINTS;

        // Vectorized block-wide load: because K and TILE_POINTS are multiples of 32, tile_base is
        // always a multiple of 32, so (data + tile_base) remains 16-byte aligned and can be read
        // as float4 safely.
        const int pair_count = batch_n >> 1;
        const float4* const data_vec = reinterpret_cast<const float4*>(data + tile_base);
        float4* const tile_vec       = reinterpret_cast<float4*>(tile);

        for (int i = threadIdx.x; i < pair_count; i += Config::block_threads) {
            tile_vec[i] = data_vec[i];
        }
        if ((batch_n & 1) && threadIdx.x == 0) {
            tile[batch_n - 1] = data[tile_base + batch_n - 1];
        }

        __syncthreads();

        if (active) {
            // Each warp scans the cached tile in chunks of 32 points so that at most one candidate
            // per lane is generated before the next buffer-capacity decision.
            for (int offset = 0; offset < batch_n; offset += kWarpSize) {
                const int local_i = offset + lane;

                float dist = 0.0f;
                int   data_index = 0;
                bool  is_candidate = false;

                if (local_i < batch_n) {
                    const float2 p = tile[local_i];
                    dist = squared_l2(p, qx, qy);
                    data_index = tile_base + local_i;
                    is_candidate = (dist < max_distance);
                }

                const unsigned candidate_mask = __ballot_sync(kFullMask, is_candidate);
                const int num_candidates = __popc(candidate_mask);

                if (num_candidates != 0) {
                    int count = 0;
                    if (lane == 0) {
                        count = *count_ptr;
                    }
                    count = __shfl_sync(kFullMask, count, 0);

                    // If inserting this whole wave would overflow the k-entry buffer, merge the
                    // existing buffer first. The candidate decisions in the current wave were made
                    // against the old max_distance. That is still safe: max_distance only shrinks,
                    // so keeping these "possibly now too loose" candidates cannot lose correctness;
                    // the subsequent top-k merge will discard them if needed.
                    if (count + num_candidates > K) {
                        flush_candidates<K>(cand_dist, cand_idx, count_ptr,
                                            best_dist, best_idx, max_distance, lane);
                        count = 0;
                    }

                    if (is_candidate) {
                        const int pos = atomicAdd(count_ptr, 1);
                        cand_dist[pos] = dist;
                        cand_idx [pos] = data_index;
                    }
                    __syncwarp();

                    if (count + num_candidates == K) {
                        flush_candidates<K>(cand_dist, cand_idx, count_ptr,
                                            best_dist, best_idx, max_distance, lane);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        // Final merge for the residual, partially filled candidate buffer.
        flush_candidates<K>(cand_dist, cand_idx, count_ptr,
                            best_dist, best_idx, max_distance, lane);

        const size_t out_base = static_cast<size_t>(query_idx) * K;

        #pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const size_t p = static_cast<size_t>(lane + t * kWarpSize);
            result[out_base + p].first  = best_idx[t];
            result[out_base + p].second = best_dist[t];
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
inline void launch_knn_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    using Config = KernelConfig<K, WARPS_PER_BLOCK>;

    // Prefer shared memory over L1 because this kernel is explicitly tile-reuse driven.
    (void)cudaFuncSetCacheConfig(knn_kernel<K, WARPS_PER_BLOCK>, cudaFuncCachePreferShared);
    (void)cudaFuncSetAttribute(knn_kernel<K, WARPS_PER_BLOCK>,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    (void)cudaFuncSetAttribute(knn_kernel<K, WARPS_PER_BLOCK>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(Config::shared_bytes));

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    knn_kernel<K, WARPS_PER_BLOCK>
        <<<blocks, Config::block_threads, Config::shared_bytes>>>(
            query, query_count, data, data_count, result);
}

template <int K>
inline void launch_knn_preferred(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    launch_knn_specialized<K, PreferredWarps<K>::value>(query, query_count, data, data_count, result);
}

} // namespace knn_detail

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0) {
        return;
    }

    // k only takes six legal values; dispatching to fully specialized kernels keeps the per-lane
    // private top-k size compile-time constant and lets the compiler optimize heavily.
    switch (k) {
        case 32:
            knn_detail::launch_knn_preferred<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_preferred<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_preferred<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_preferred<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_preferred<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_preferred<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem guarantees that k is valid, so this path is unreachable for legal input.
            break;
    }
}