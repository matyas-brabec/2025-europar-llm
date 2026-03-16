#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cmath>
#include <utility>

namespace {

// Warp size is fixed by the problem statement.
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Tuning choice:
// - 256 threads/block = 8 warps/block = 8 queries/block.
//   This gives good data reuse (the block-wide shared-memory tile is reused by 8 queries)
//   while still producing enough blocks when query_count is in the low-thousands.
// - 2048 points/tile = 16 KiB shared tile.
//   Together with the worst-case candidate buffer for K=1024 (64 KiB dynamic shared),
//   total shared memory is exactly 80 KiB/block. Two such blocks fit exactly into the
//   160 KiB shared-memory budget of A100-class SMs.
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kDataTilePoints = 2048;

// Internal sentinel used only for padded / unused slots inside the sort/merge network.
// Smaller indices are considered "better" for equal distances, so INT_MAX makes padded
// +inf entries the worst possible items and keeps them at the tail after sorting.
constexpr int kInvalidIndex = INT_MAX;

template <int K>
constexpr std::size_t candidate_buffer_bytes() {
    return static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(K) *
           (sizeof(float) + sizeof(int));
}

constexpr std::size_t kTileSharedBytes =
    static_cast<std::size_t>(kDataTilePoints) * sizeof(float2);

// Sanity-check the chosen hyper-parameters: worst-case shared memory is 80 KiB/block.
static_assert(kTileSharedBytes + candidate_buffer_bytes<1024>() == 81920u,
              "Shared-memory footprint must remain 80 KiB/block at K=1024.");

// Total ordering used by the bitonic network.
// Distances are primary keys; indices are only a deterministic tie-breaker.
// The problem statement allows any tie resolution, but a total order makes the
// compare/exchange network well-defined.
__device__ __forceinline__ bool pair_less(const float a_dist, const int a_idx,
                                          const float b_dist, const int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ void swap_pair(float &a_dist, int &a_idx,
                                          float &b_dist, int &b_idx) {
    const float tmp_dist = a_dist;
    const int tmp_idx = a_idx;
    a_dist = b_dist;
    a_idx = b_idx;
    b_dist = tmp_dist;
    b_idx = tmp_idx;
}

// Squared Euclidean distance in 2D. No square root is taken because the required output
// is the squared L2 norm.
__device__ __forceinline__ float squared_l2(const float qx, const float qy,
                                            const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

// Bitonic sort over K elements distributed across one warp.
// Layout invariant:
//   - Each thread owns ELEMS_PER_THREAD = K / 32 consecutive positions.
//   - Therefore, when the compare distance "stride" is smaller than ELEMS_PER_THREAD,
//     both partners live in the same thread and can be swapped locally.
//   - Otherwise, both partners live in different lanes but always at the same local
//     register index, so warp shuffles exchange them directly.
template <int K, int ELEMS_PER_THREAD>
__device__ __forceinline__ void bitonic_sort_regs(float (&dist)[ELEMS_PER_THREAD],
                                                  int (&idx)[ELEMS_PER_THREAD],
                                                  const int lane) {
    const int lane_base = lane * ELEMS_PER_THREAD;

    #pragma unroll
    for (int stage = 2; stage <= K; stage <<= 1) {
        #pragma unroll
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
            if (stride < ELEMS_PER_THREAD) {
                // Same-thread compare/exchange.
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                    const int partner_e = e ^ stride;
                    if (partner_e > e) {
                        const int g = lane_base + e;
                        const bool ascending = ((g & stage) == 0);
                        const bool do_swap =
                            ascending
                                ? pair_less(dist[partner_e], idx[partner_e], dist[e], idx[e])
                                : pair_less(dist[e], idx[e], dist[partner_e], idx[partner_e]);

                        if (do_swap) {
                            swap_pair(dist[e], idx[e], dist[partner_e], idx[partner_e]);
                        }
                    }
                }
            } else {
                // Cross-thread compare/exchange via shuffles.
                const int lane_delta = stride / ELEMS_PER_THREAD;
                const int partner_lane = lane ^ lane_delta;

                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                    const float other_dist = __shfl_sync(kFullMask, dist[e], partner_lane);
                    const int other_idx = __shfl_sync(kFullMask, idx[e], partner_lane);

                    const int g = lane_base + e;
                    const bool ascending = ((g & stage) == 0);
                    const bool keep_min = ascending ? (lane < partner_lane)
                                                    : (lane > partner_lane);

                    const bool take_other =
                        keep_min ? pair_less(other_dist, other_idx, dist[e], idx[e])
                                 : pair_less(dist[e], idx[e], other_dist, other_idx);

                    if (take_other) {
                        dist[e] = other_dist;
                        idx[e] = other_idx;
                    }
                }
            }

            // Bitonic networks are stage-based. An explicit warp barrier preserves the stage
            // boundary and also satisfies the prompt requirement to synchronize warp
            // communication appropriately.
            __syncwarp(kFullMask);
        }
    }
}

// Flushes the shared candidate buffer into the register-resident intermediate result.
// This implements exactly the merge procedure requested in the prompt:
//
// 0. Invariant: best_* is sorted ascending.
// 1. Swap the buffer and intermediate result so the buffer moves into registers.
// 2. Sort the register-resident buffer with bitonic sort.
// 3. Form the bitonic sequence merged[i] = min(buffer[i], best[K-1-i]).
// 4. Sort that bitonic sequence again to obtain the updated intermediate result.
//
// The candidate buffer is warp-private inside shared memory, so __syncwarp is sufficient.
template <int K, int ELEMS_PER_THREAD>
__device__ __forceinline__ void flush_candidate_buffer(
    float (&best_dist)[ELEMS_PER_THREAD],
    int (&best_idx)[ELEMS_PER_THREAD],
    float *cand_dist,
    int *cand_idx,
    int &buf_count,
    float &max_distance,
    bool &filter_open,
    const int lane) {
    if (buf_count == 0) {
        return;
    }

    const int lane_base = lane * ELEMS_PER_THREAD;
    const int active_count = buf_count;

    __syncwarp(kFullMask);

    // Step 1: swap the shared candidate buffer with the sorted intermediate result.
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int g = lane_base + e;
        const bool valid_candidate = (g < active_count);

        const float candidate_dist = valid_candidate ? cand_dist[g] : CUDART_INF_F;
        const int candidate_idx = valid_candidate ? cand_idx[g] : kInvalidIndex;

        cand_dist[g] = best_dist[e];
        cand_idx[g] = best_idx[e];

        best_dist[e] = candidate_dist;
        best_idx[e] = candidate_idx;
    }

    __syncwarp(kFullMask);

    // Step 2: sort the former candidate buffer now held in registers.
    bitonic_sort_regs<K, ELEMS_PER_THREAD>(best_dist, best_idx, lane);

    // Step 3: bitonic merge construction with the reversed current best from shared memory.
    const int reverse_base = K - 1 - lane_base;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int reverse_g = reverse_base - e;
        const float other_dist = cand_dist[reverse_g];
        const int other_idx = cand_idx[reverse_g];

        if (pair_less(other_dist, other_idx, best_dist[e], best_idx[e])) {
            best_dist[e] = other_dist;
            best_idx[e] = other_idx;
        }
    }

    // Step 4: sort the bitonic sequence to restore the ascending invariant.
    bitonic_sort_regs<K, ELEMS_PER_THREAD>(best_dist, best_idx, lane);

    // Update the pruning threshold: the k-th nearest element lives in the last register
    // of lane 31 because each lane stores a consecutive chunk.
    const float kth_local =
        (lane == (kWarpSize - 1)) ? best_dist[ELEMS_PER_THREAD - 1] : 0.0f;
    max_distance = __shfl_sync(kFullMask, kth_local, kWarpSize - 1);

    // While max_distance remains +inf, the filter is intentionally disabled so the first
    // K elements are admitted unconditionally. This also keeps the algorithm correct if
    // some computed distances overflow to +inf before a finite threshold is established.
    filter_open = (max_distance == CUDART_INF_F);
    buf_count = 0;

    __syncwarp(kFullMask);
}

template <int K, int THREADS_PER_BLOCK, int DATA_TILE_POINTS>
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void knn_kernel(const float2 *__restrict__ query,
                const int query_count,
                const float2 *__restrict__ data,
                const int data_count,
                std::pair<int, float> *__restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= kWarpSize && K <= 1024, "K must be in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");
    static_assert((THREADS_PER_BLOCK % kWarpSize) == 0,
                  "Block size must be a multiple of warp size.");
    static_assert((DATA_TILE_POINTS % kWarpSize) == 0,
                  "Tile size must be a multiple of warp size.");

    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / kWarpSize;
    constexpr int ELEMS_PER_THREAD = K / kWarpSize;

    // Static shared tile reused by the entire block across 8 queries.
    __shared__ float2 tile[DATA_TILE_POINTS];

    // Dynamic shared memory holds the per-warp candidate buffers:
    //   [WARPS_PER_BLOCK * K floats][WARPS_PER_BLOCK * K ints]
    extern __shared__ unsigned char smem_raw[];
    float *const cand_dist_all = reinterpret_cast<float *>(smem_raw);
    int *const cand_idx_all =
        reinterpret_cast<int *>(cand_dist_all + WARPS_PER_BLOCK * K);

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int lane_base = lane * ELEMS_PER_THREAD;
    const unsigned lower_lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);

    float *const cand_dist = cand_dist_all + warp_id * K;
    int *const cand_idx = cand_idx_all + warp_id * K;

    // Blocks advance over query groups in a grid-stride fashion. With the default launch
    // configuration this loop usually executes only once, but keeping it grid-strided
    // makes the kernel robust to any future launch policy.
    const int query_group_stride = static_cast<int>(gridDim.x) * WARPS_PER_BLOCK;

    for (int query_base = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK;
         query_base < query_count;
         query_base += query_group_stride) {
        const int query_idx = query_base + warp_id;
        const bool active_query = (query_idx < query_count);

        // One query per warp. Lane 0 loads the query point, then broadcasts it.
        float qx = 0.0f;
        float qy = 0.0f;
        if (active_query && lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);

        // Register-resident intermediate result. Each thread owns K/32 consecutive slots.
        float best_dist[ELEMS_PER_THREAD];
        int best_idx[ELEMS_PER_THREAD];

        // buf_count is intentionally replicated in every lane; ballot counts are identical in
        // all lanes, so the value stays consistent without shared-memory traffic.
        int buf_count = 0;
        float max_distance = CUDART_INF_F;
        bool filter_open = true;

        if (active_query) {
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                best_dist[e] = CUDART_INF_F;
                best_idx[e] = kInvalidIndex;
            }
        }

        // Stream the data set through shared memory in block-wide tiles.
        for (int data_base = 0; data_base < data_count; data_base += DATA_TILE_POINTS) {
            const int remaining = data_count - data_base;
            const int tile_count = (remaining < DATA_TILE_POINTS) ? remaining : DATA_TILE_POINTS;
            const float2 *const data_base_ptr = data + data_base;

            // Whole-block cooperative load of the next tile.
            for (int load_idx = tid; load_idx < tile_count; load_idx += THREADS_PER_BLOCK) {
                tile[load_idx] = data_base_ptr[load_idx];
            }
            __syncthreads();

            if (active_query) {
                // Process one warp-sized group from shared memory at a time so a single ballot
                // tells the warp how many candidates passed the distance threshold.
                for (int base = 0; base < tile_count; base += kWarpSize) {
                    const int local_idx = base + lane;
                    const bool valid = (local_idx < tile_count);

                    float distance = 0.0f;
                    int data_idx = kInvalidIndex;

                    if (valid) {
                        const float2 p = tile[local_idx];
                        distance = squared_l2(qx, qy, p);
                        data_idx = data_base + local_idx;
                    }

                    bool pass = valid && (filter_open || (distance < max_distance));
                    unsigned mask = __ballot_sync(kFullMask, pass);
                    int accepted = __popc(mask);

                    // If the current warp ballot would overflow the candidate buffer, flush the
                    // existing buffer first, then re-filter the current 32 distances against the
                    // updated threshold.
                    if (buf_count + accepted > K) {
                        flush_candidate_buffer<K, ELEMS_PER_THREAD>(
                            best_dist, best_idx,
                            cand_dist, cand_idx,
                            buf_count, max_distance, filter_open,
                            lane);

                        pass = valid && (filter_open || (distance < max_distance));
                        mask = __ballot_sync(kFullMask, pass);
                        accepted = __popc(mask);
                    }

                    if (accepted != 0) {
                        const int offset = buf_count;

                        if (pass) {
                            const int rank = __popc(mask & lower_lane_mask);
                            cand_dist[offset + rank] = distance;
                            cand_idx[offset + rank] = data_idx;
                        }

                        __syncwarp(kFullMask);

                        buf_count = offset + accepted;

                        if (buf_count == K) {
                            flush_candidate_buffer<K, ELEMS_PER_THREAD>(
                                best_dist, best_idx,
                                cand_dist, cand_idx,
                                buf_count, max_distance, filter_open,
                                lane);
                        }
                    }
                }
            }

            // The tile is block-shared and will be overwritten by the next iteration.
            __syncthreads();
        }

        // Final partial buffer flush, if needed.
        if (active_query) {
            if (buf_count != 0) {
                flush_candidate_buffer<K, ELEMS_PER_THREAD>(
                    best_dist, best_idx,
                    cand_dist, cand_idx,
                    buf_count, max_distance, filter_open,
                    lane);
            }

            // Store the final sorted K nearest neighbors.
            std::pair<int, float> *out =
                result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K) + lane_base;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                out[e].first = best_idx[e];
                out[e].second = best_dist[e];
            }
        }
    }
}

// Launch helper for a compile-time K. K must be specialized because:
// - Each thread stores K/32 neighbors in registers.
// - The bitonic network size depends on K.
// - The shared candidate buffer size depends on K.
template <int K>
void launch_knn_case(const float2 *query,
                     const int query_count,
                     const float2 *data,
                     const int data_count,
                     std::pair<int, float> *result) {
    constexpr int THREADS = kThreadsPerBlock;
    constexpr int TILE = kDataTilePoints;
    constexpr std::size_t DYNAMIC_SHARED_BYTES = candidate_buffer_bytes<K>();

    // The shared data tile is static (16 KiB). Only the per-warp candidate buffer is dynamic.
    // Prefer maximum shared-memory carveout because the kernel relies on shared-memory tiling
    // much more than on L1 capacity.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, THREADS, TILE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(DYNAMIC_SHARED_BYTES));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, THREADS, TILE>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > 0) {
        knn_kernel<K, THREADS, TILE><<<blocks, THREADS, DYNAMIC_SHARED_BYTES>>>(
            query, query_count, data, data_count, result);
    }
}

} // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   launch_knn_case<32>(query, query_count, data, data_count, result); break;
        case 64:   launch_knn_case<64>(query, query_count, data, data_count, result); break;
        case 128:  launch_knn_case<128>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn_case<256>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn_case<512>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn_case<1024>(query, query_count, data, data_count, result); break;
        default: break; // Inputs are guaranteed valid by the problem statement.
    }
}