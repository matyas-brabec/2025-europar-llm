#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

// Exact brute-force 2D k-NN for modern NVIDIA GPUs.
//
// Design:
// - One warp owns one query for the entire scan.
// - The whole CTA streams the database through a shared-memory tile so every loaded point
//   is reused by all warps in the block.
// - Each warp keeps a private top-k list in shared memory. A second private buffer is used
//   as a merge destination so updates are fully warp-parallel and never suffer in-place
//   read-after-write hazards.
// - Updates are batch-oriented:
//     * common sparse case: if exactly one new candidate beats the current threshold,
//       insert it with a warp-parallel shift.
//     * general case: sort the 32 per-lane candidates cooperatively, then merge the
//       sorted 32-way batch with the current top-k cooperatively.
// - k is dispatched at runtime to a compile-time specialization so K/32 is constant,
//   which lets the compiler fully unroll the per-lane work.
//
// Distances are squared Euclidean distances as requested.
//
// Note on the result type:
// Writing std::pair<int,float> directly in device code is awkward on many standard library
// implementations because pair's constructors/operators are often host-only. The output
// memory comes from cudaMalloc (raw storage), so we write through a trivial ABI-compatible
// proxy {int,float}. This relies on the standard CUDA toolchain's de-facto pair layout.
struct ResultPairDevice {
    int   first;
    float second;
};

static_assert(std::is_trivially_copyable<std::pair<int, float>>::value,
              "std::pair<int,float> must be trivially copyable for device-side raw writes.");
static_assert(sizeof(ResultPairDevice) == sizeof(std::pair<int, float>),
              "Unexpected std::pair<int,float> layout; device result proxy size mismatch.");

constexpr int      WARP_THREADS  = 32;
constexpr unsigned FULL_MASK     = 0xffffffffu;
constexpr int      INVALID_INDEX = INT_MAX;
constexpr float    INF_DISTANCE  = CUDART_INF_F;

// Deterministic total order used only internally to sort/merge candidates.
// The problem allows arbitrary tie resolution; the index tie-break is just a convenient
// way to make the order total and deterministic.
__device__ __forceinline__ bool pair_less(float a_dist, int a_idx,
                                          float b_dist, int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

// 32-way bitonic sort over one (distance,index) pair per lane.
__device__ __forceinline__ void warp_sort32(float &dist, int &idx, int lane) {
#pragma unroll
    for (int size = 2; size <= WARP_THREADS; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float other_dist = __shfl_xor_sync(FULL_MASK, dist, stride);
            const int   other_idx  = __shfl_xor_sync(FULL_MASK, idx,  stride);

            const bool lower_lane = ((lane & stride) == 0);
            const bool ascending  = ((lane & size) == 0);
            const bool want_min   = ascending ? lower_lane : !lower_lane;
            const bool self_less  = pair_less(dist, idx, other_dist, other_idx);
            const bool take_other = want_min ? !self_less : self_less;

            dist = take_other ? other_dist : dist;
            idx  = take_other ? other_idx  : idx;
        }
    }
}

// Fast path for the common post-convergence case where exactly one new candidate beats the
// current k-th distance. Lane 0 finds the insertion point; the whole warp materializes the
// shifted output in parallel.
template <int K>
__device__ __forceinline__ void warp_single_insert(const float *cur_dist,
                                                   const int   *cur_idx,
                                                   float cand_dist,
                                                   int   cand_idx,
                                                   float *out_dist,
                                                   int   *out_idx,
                                                   int lane) {
    constexpr int CHUNK = K / WARP_THREADS;

    int pos = 0;
    if (lane == 0) {
        int lo = 0;
        int hi = K;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (pair_less(cur_dist[mid], cur_idx[mid], cand_dist, cand_idx)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        pos = lo;
    }
    pos = __shfl_sync(FULL_MASK, pos, 0);

    const int base = lane * CHUNK;
#pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const int out = base + t;
        if (out < pos) {
            out_dist[out] = cur_dist[out];
            out_idx[out]  = cur_idx[out];
        } else if (out == pos) {
            out_dist[out] = cand_dist;
            out_idx[out]  = cand_idx;
        } else {
            out_dist[out] = cur_dist[out - 1];
            out_idx[out]  = cur_idx[out - 1];
        }
    }
}

// General warp-parallel merge:
// - A[0..K) is the current sorted top-k.
// - B[0..32) is the sorted per-lane candidate batch (with +inf sentinels for inactive lanes).
// - We keep only the first K outputs of merge(A,B).
//
// Lane l owns the contiguous output strip [l*CHUNK, (l+1)*CHUNK). Its strip start is found
// by a tiny merge-path search over the candidate axis. Because B has only 32 elements, this
// search is cheap and fully unrolled.
template <int K>
__device__ __forceinline__ void warp_merge32(const float *cur_dist,
                                             const int   *cur_idx,
                                             float cand_dist,
                                             int   cand_idx,
                                             float *out_dist,
                                             int   *out_idx,
                                             int lane) {
    constexpr int CHUNK = K / WARP_THREADS;
    constexpr int MERGE_SEARCH_ITERS = 6;  // ceil(log2(32 + 1))

    const int diag = lane * CHUNK;

    int low  = 0;
    int high = (diag < WARP_THREADS) ? diag : WARP_THREADS;

#pragma unroll
    for (int iter = 0; iter < MERGE_SEARCH_ITERS; ++iter) {
        if (low < high) {
            const int mid = (low + high + 1) >> 1;
            const int a   = diag - mid;

            const float b_prev_dist = __shfl_sync(FULL_MASK, cand_dist, mid - 1);
            const int   b_prev_idx  = __shfl_sync(FULL_MASK, cand_idx,  mid - 1);

            const float a_curr_dist = (a < K) ? cur_dist[a] : INF_DISTANCE;
            const int   a_curr_idx  = (a < K) ? cur_idx[a]  : INVALID_INDEX;

            if (pair_less(b_prev_dist, b_prev_idx, a_curr_dist, a_curr_idx)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
    }

    int a = diag - low;
    int b = low;

    const int out_base = diag;
#pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const float a_dist = (a < K) ? cur_dist[a] : INF_DISTANCE;
        const int   a_idx  = (a < K) ? cur_idx[a]  : INVALID_INDEX;

        const float b_dist = (b < WARP_THREADS) ? __shfl_sync(FULL_MASK, cand_dist, b) : INF_DISTANCE;
        const int   b_idx  = (b < WARP_THREADS) ? __shfl_sync(FULL_MASK, cand_idx,  b) : INVALID_INDEX;

        const bool take_b = pair_less(b_dist, b_idx, a_dist, a_idx);

        out_dist[out_base + t] = take_b ? b_dist : a_dist;
        out_idx [out_base + t] = take_b ? b_idx  : a_idx;

        b += take_b ? 1 : 0;
        a += take_b ? 0 : 1;
    }
}

template <int K>
__global__ void knn2d_kernel(const float2 * __restrict__ query,
                             int query_count,
                             const float2 * __restrict__ data,
                             int data_count,
                             ResultPairDevice * __restrict__ result) {
    static_assert(K >= WARP_THREADS && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");

    constexpr int CHUNK = K / WARP_THREADS;

    // Shared-memory layout:
    //   [ block-wide data tile | per-warp cur_dist | per-warp cur_idx |
    //     per-warp nxt_dist    | per-warp nxt_idx ]
    extern __shared__ unsigned char smem_raw[];
    float2 *tile = reinterpret_cast<float2 *>(smem_raw);

    const int warps_per_block = blockDim.x / WARP_THREADS;

    float *dist_buf0 = reinterpret_cast<float *>(tile + blockDim.x);
    int   *idx_buf0  = reinterpret_cast<int *>(dist_buf0 + warps_per_block * K);
    float *dist_buf1 = reinterpret_cast<float *>(idx_buf0  + warps_per_block * K);
    int   *idx_buf1  = reinterpret_cast<int *>(dist_buf1 + warps_per_block * K);

    const int tid     = threadIdx.x;
    const int lane    = tid & (WARP_THREADS - 1);
    const int warp_id = tid >> 5;

    const int query_idx = blockIdx.x * warps_per_block + warp_id;
    const bool active   = (query_idx < query_count);

    float *cur_dist = dist_buf0 + warp_id * K;
    int   *cur_idx  = idx_buf0  + warp_id * K;
    float *nxt_dist = dist_buf1 + warp_id * K;
    int   *nxt_idx  = idx_buf1  + warp_id * K;

    // Initialize the current top-k buffer to (+inf, invalid).
#pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const int pos = t * WARP_THREADS + lane;
        cur_dist[pos] = INF_DISTANCE;
        cur_idx[pos]  = INVALID_INDEX;
    }

    // Broadcast the warp's query point once.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    if (active) {
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);
    }

    // Keep the current threshold in a register and update it only when the top-k changes.
    float threshold = INF_DISTANCE;

    // Stream the full dataset through a shared tile.
    for (int tile_base = 0; tile_base < data_count; tile_base += blockDim.x) {
        const int g = tile_base + tid;
        if (g < data_count) {
            tile[tid] = data[g];
        }

        __syncthreads();

        int tile_count = data_count - tile_base;
        if (tile_count > blockDim.x) tile_count = blockDim.x;

        if (active) {
            // Each warp consumes the tile in 32-point micro-batches so every lane naturally
            // owns one candidate in the batch.
            for (int local_base = 0; local_base < tile_count; local_base += WARP_THREADS) {
                const int local = local_base + lane;

                float cand_dist = INF_DISTANCE;
                int   cand_idx  = INVALID_INDEX;

                if (local < tile_count) {
                    const float2 p  = tile[local];
                    const float  dx = qx - p.x;
                    const float  dy = qy - p.y;
                    cand_dist = fmaf(dx, dx, dy * dy);
                    cand_idx  = tile_base + local;
                }

                const bool qualifies = (cand_dist < threshold);
                const unsigned mask  = __ballot_sync(FULL_MASK, qualifies);

                if (mask != 0u) {
                    const int count = __popc(mask);

                    if (count == 1) {
                        // Sparse-update fast path.
                        const int src_lane = __ffs(mask) - 1;
                        const float best_dist = __shfl_sync(FULL_MASK, cand_dist, src_lane);
                        const int   best_idx  = __shfl_sync(FULL_MASK, cand_idx,  src_lane);

                        warp_single_insert<K>(cur_dist, cur_idx, best_dist, best_idx,
                                              nxt_dist, nxt_idx, lane);
                    } else {
                        // General batch-update path:
                        // 1) keep only candidates below the current threshold,
                        // 2) sort the 32-lane candidate batch cooperatively,
                        // 3) merge it cooperatively with the current top-k.
                        if (!qualifies) {
                            cand_dist = INF_DISTANCE;
                            cand_idx  = INVALID_INDEX;
                        }

                        warp_sort32(cand_dist, cand_idx, lane);
                        warp_merge32<K>(cur_dist, cur_idx, cand_dist, cand_idx,
                                        nxt_dist, nxt_idx, lane);
                    }

                    // Shared-memory ordering for the per-warp private buffers.
                    __syncwarp(FULL_MASK);

                    float *tmp_dist = cur_dist; cur_dist = nxt_dist; nxt_dist = tmp_dist;
                    int   *tmp_idx  = cur_idx;  cur_idx  = nxt_idx;  nxt_idx  = tmp_idx;

                    if (lane == 0) {
                        threshold = cur_dist[K - 1];
                    }
                    threshold = __shfl_sync(FULL_MASK, threshold, 0);
                }
            }
        }

        // The tile is about to be overwritten by the next global-memory batch.
        __syncthreads();
    }

    if (active) {
        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        // Strided final write so each iteration issues a coalesced 32-wide store.
#pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            const int pos = t * WARP_THREADS + lane;
            result[out_base + pos].first  = cur_idx[pos];
            result[out_base + pos].second = cur_dist[pos];
        }
    }
}

// Shared-memory footprint per resident warp:
// - one warp's worth of staged data points in the block-wide tile: 32 * sizeof(float2)
// - two private K-entry buffers for distances and indices
template <int K>
static inline std::size_t shared_bytes_per_warp() {
    return static_cast<std::size_t>(WARP_THREADS) * sizeof(float2) +
           static_cast<std::size_t>(2) * K * (sizeof(float) + sizeof(int));
}

// Host-side launch helper.
// The block-shape heuristic is intentionally reuse-biased:
// loading the data tile once and reusing it across many resident queries is the dominant
// optimization for this exact all-pairs scan. We only shrink the CTA width when the query
// count is small enough that very wide CTAs would starve the grid.
template <int K>
static inline void launch_knn_typed(const float2 *query,
                                    int query_count,
                                    const float2 *data,
                                    int data_count,
                                    ResultPairDevice *result) {
    if (query_count <= 0) return;

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const int max_threads_warps = prop.maxThreadsPerBlock / WARP_THREADS;
    const std::size_t shared_limit =
        static_cast<std::size_t>(prop.sharedMemPerBlockOptin > 0 ?
                                 prop.sharedMemPerBlockOptin :
                                 prop.sharedMemPerBlock);

    const std::size_t per_warp_smem = shared_bytes_per_warp<K>();

    int max_warps_by_shared = static_cast<int>(shared_limit / per_warp_smem);
    if (max_warps_by_shared < 1) max_warps_by_shared = 1;

    int max_warps = (max_threads_warps < max_warps_by_shared) ? max_threads_warps : max_warps_by_shared;
    if (max_warps > query_count) max_warps = query_count;
    if (max_warps < 1) max_warps = 1;

    // Aim for roughly half as many CTAs as SMs before shrinking CTAs any further.
    // These CTAs are long-running and already carry many warps each, so this keeps enough
    // grid-level parallelism while preserving strong tile reuse.
    int target_blocks = (prop.multiProcessorCount > 1) ? ((prop.multiProcessorCount + 1) / 2) : 1;
    int reuse_cap_warps = query_count / target_blocks;
    if (reuse_cap_warps < 1) reuse_cap_warps = 1;

    int warps_per_block = (max_warps < reuse_cap_warps) ? max_warps : reuse_cap_warps;

    // Round down to an even number of warps when possible; this tends to produce
    // scheduler-friendly CTA shapes without materially changing the reuse trade-off.
    if (warps_per_block > 1) {
        warps_per_block &= ~1;
    }
    if (warps_per_block < 1) warps_per_block = 1;

    const int block_threads = warps_per_block * WARP_THREADS;
    const std::size_t shared_bytes = per_warp_smem * static_cast<std::size_t>(warps_per_block);
    const int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Opt in to the required dynamic shared memory. The launch remains asynchronous;
    // callers can synchronize or query errors outside this function as needed.
    cudaFuncSetAttribute(knn2d_kernel<K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
    cudaFuncSetAttribute(knn2d_kernel<K>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    knn2d_kernel<K><<<grid_blocks, block_threads, shared_bytes>>>(
        query, query_count, data, data_count, result);
}

}  // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) return;

    ResultPairDevice *result_proxy = reinterpret_cast<ResultPairDevice *>(result);

    // Runtime dispatch over the small fixed set of supported K values.
    switch (k) {
        case 32:   launch_knn_typed<32>(query, query_count, data, data_count, result_proxy);   break;
        case 64:   launch_knn_typed<64>(query, query_count, data, data_count, result_proxy);   break;
        case 128:  launch_knn_typed<128>(query, query_count, data, data_count, result_proxy);  break;
        case 256:  launch_knn_typed<256>(query, query_count, data, data_count, result_proxy);  break;
        case 512:  launch_knn_typed<512>(query, query_count, data, data_count, result_proxy);  break;
        case 1024: launch_knn_typed<1024>(query, query_count, data, data_count, result_proxy); break;
        default:
            // By contract k is always valid, so this path is unreachable for correct callers.
            break;
    }
}